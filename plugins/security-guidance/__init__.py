"""security-guidance plugin — pattern-matched security policy on file writes.

The ``pre_tool_call`` hook scans content passed to ``write_file``, ``patch``,
and skill mutation tools. High-confidence dangerous patterns are blocked by
default; lower-confidence matches remain warnings appended by
``transform_tool_result``. This preserves useful guidance without allowing
known unsafe deserialization and command-injection patterns to execute first.

An operator can explicitly select audited warn-only behavior with
``security_guidance.warn_only: true`` or ``SECURITY_GUIDANCE_WARN_ONLY=1``.
``SECURITY_GUIDANCE_BLOCK=1`` is the stricter compatibility mode that blocks
all matches, including lower-confidence patterns. Pattern data comes from
Anthropic's ``claude-plugins-official`` under Apache-2.0; see LICENSE/NOTICE.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from . import patterns as _patterns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tool names whose args carry "code being written to disk" we want to scan.
# Maps tool name -> (path_arg_name, content_arg_names).  For tools with multiple
# possible content fields (patch's old/new_string vs raw patch text), we scan
# every populated string field.
_TARGET_TOOLS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "write_file": ("path", ("content",)),
    "patch": ("path", ("new_string", "patch")),
    # skill_manage write_file / patch sub-actions land here. file_path holds
    # the relative path inside the skill dir; we scan it the same way.
    "skill_manage": ("file_path", ("file_content", "new_string")),
}

# Cap on how much content we scan. Above this we skip — pattern matching a
# 10 MB blob has poor signal-to-noise and would slow down the agent loop.
_MAX_SCAN_BYTES = 256 * 1024
_WARN_ONLY_AUDITED = False

_HIGH_CONFIDENCE_RULES = frozenset(
    {
        "child_process_exec",
        "new_function_injection",
        "eval_injection",
        "pickle_deserialization",
        "os_system_injection",
        "python_subprocess_shell",
        "unsafe_yaml_load",
        "marshal_loads",
        "shelve_open",
        "pickle_variants_load",
        "torch_unsafe_load",
        "yaml_unsafe_load_variants",
        "pickle_wrapper_load",
    }
)


def _block_mode_enabled() -> bool:
    return os.environ.get("SECURITY_GUIDANCE_BLOCK", "").lower() in {"1", "true", "yes", "on"}


def _config_warn_only_enabled() -> bool:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        return False
    if not isinstance(cfg, dict):
        return False
    sections = [
        cfg.get("security_guidance"),
        cfg.get("security-guidance"),
        (cfg.get("plugins") or {}).get("security_guidance")
        if isinstance(cfg.get("plugins"), dict)
        else None,
        (cfg.get("plugins") or {}).get("security-guidance")
        if isinstance(cfg.get("plugins"), dict)
        else None,
    ]
    return any(
        isinstance(section, dict)
        and str(section.get("warn_only", "")).lower() in {"1", "true", "yes", "on"}
        for section in sections
    )


def _warn_only_enabled() -> bool:
    env_value = os.environ.get("SECURITY_GUIDANCE_WARN_ONLY", "").lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    return _config_warn_only_enabled()


def _audit_warn_only_override() -> None:
    global _WARN_ONLY_AUDITED
    if _WARN_ONLY_AUDITED:
        return
    _WARN_ONLY_AUDITED = True
    source = (
        "SECURITY_GUIDANCE_WARN_ONLY"
        if os.environ.get("SECURITY_GUIDANCE_WARN_ONLY", "").lower()
        in {"1", "true", "yes", "on"}
        else "security_guidance.warn_only"
    )
    logger.warning(
        "security-guidance warn-only override is active via %s; "
        "high-confidence dangerous writes will warn instead of blocking.",
        source,
    )


def _plugin_disabled() -> bool:
    return os.environ.get("SECURITY_GUIDANCE_DISABLE", "").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


# Pre-compile the regex patterns once.  Substring patterns stay as plain
# strings — ``str.__contains__`` is faster than a regex of literal chars.
_COMPILED: List[Dict[str, Any]] = []
for _rule in _patterns.SECURITY_PATTERNS:
    _entry: Dict[str, Any] = {
        "ruleName": _rule["ruleName"],
        "reminder": _rule["reminder"],
        "path_filter": _rule.get("path_filter"),
        "path_check": _rule.get("path_check"),
        "substrings": tuple(_rule.get("substrings", ())),
        "regex": None,
    }
    _re_src = _rule.get("regex")
    if _re_src:
        try:
            _entry["regex"] = re.compile(_re_src)
        except re.error as _err:
            logger.warning(
                "security-guidance: skipping rule %s — invalid regex %r: %s",
                _rule["ruleName"], _re_src, _err,
            )
            continue
    _COMPILED.append(_entry)


def _scan_content(path: str, content: str) -> List[Tuple[str, str]]:
    """Return [(ruleName, reminder), ...] for every pattern that matches.

    ``path`` is used by per-rule path filters (path_filter / path_check).
    Each rule fires at most once per call — multiple matches of the same
    rule collapse into a single warning entry.
    """
    if not content or len(content.encode("utf-8", errors="ignore")) > _MAX_SCAN_BYTES:
        return []
    hits: List[Tuple[str, str]] = []
    for entry in _COMPILED:
        # path_check: rule fires PURELY on path match (no content regex). Used
        # for blanket "you're editing a sensitive file, here are reminders"
        # warnings — github_actions_workflow is the canonical example.
        path_check = entry.get("path_check")
        if path_check is not None:
            try:
                if path_check(path or ""):
                    hits.append((entry["ruleName"], entry["reminder"]))
            except Exception:
                pass
            # Path-check rules don't also pattern-match content; move on.
            continue
        # path_filter: rule is skipped when the path filter returns False
        # (e.g. Python-only rules skip .js files; eval_injection skips .md)
        path_filter = entry.get("path_filter")
        if path_filter is not None:
            try:
                if not path_filter(path or ""):
                    continue
            except Exception:
                continue
        matched = False
        for sub in entry["substrings"]:
            if sub in content:
                matched = True
                break
        if not matched and entry["regex"] is not None:
            if entry["regex"].search(content):
                matched = True
        if matched:
            hits.append((entry["ruleName"], entry["reminder"]))
    return hits


def _extract_path_and_content(tool_name: str, args: Any) -> List[Tuple[str, str]]:
    """Return [(path, content), ...] for a tool call.  Empty if nothing to scan."""
    spec = _TARGET_TOOLS.get(tool_name)
    if spec is None or not isinstance(args, dict):
        return []
    path_key, content_keys = spec
    path = args.get(path_key) or ""
    if not isinstance(path, str):
        path = ""
    out: List[Tuple[str, str]] = []
    for ck in content_keys:
        val = args.get(ck)
        if isinstance(val, str) and val:
            out.append((path, val))
    return out


def _format_warning_block(findings: List[Tuple[str, str]]) -> str:
    """Render findings into a Markdown block appended to the tool result."""
    names = ", ".join(name for name, _ in findings)
    lines = [
        "",
        "---",
        f"⚠️ Security guidance — {len(findings)} pattern{'s' if len(findings) != 1 else ''} matched ({names})",
        "",
    ]
    for _, reminder in findings:
        lines.append(reminder)
        lines.append("")
    lines.append(
        "Pattern matches can be false positives. If the construct is safe in this "
        "context, briefly document why in a code comment and continue. Otherwise, "
        "fix the code before moving on."
    )
    return "\n".join(lines)


def _high_confidence_findings(findings: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [(name, reminder) for name, reminder in findings if name in _HIGH_CONFIDENCE_RULES]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _scan_args(tool_name: str, args: Any) -> List[Tuple[str, str]]:
    """Common scan path used by both pre_tool_call (block mode) and
    transform_tool_result (warn mode)."""
    if _plugin_disabled():
        return []
    findings: List[Tuple[str, str]] = []
    for path, content in _extract_path_and_content(tool_name, args):
        findings.extend(_scan_content(path, content))
    return findings


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[Dict[str, str]]:
    """Block high-confidence findings unless warn-only is explicitly enabled."""
    strict_block = _block_mode_enabled()
    if not strict_block and _warn_only_enabled():
        _audit_warn_only_override()
        return None
    findings = _scan_args(tool_name, args)
    if not findings:
        return None
    blocked_findings = findings if strict_block else _high_confidence_findings(findings)
    if not blocked_findings:
        return None
    return {
        "action": "block",
        "message": (
            "security-guidance refused this write: "
            + _format_warning_block(blocked_findings)
            + "\n\nTo override for an audited warn-only run, set "
            "security_guidance.warn_only: true in config.yaml or "
            "SECURITY_GUIDANCE_WARN_ONLY=1 in the environment and retry. "
            "SECURITY_GUIDANCE_BLOCK=1 enables strict block-all mode."
        ),
    }


def _on_transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    """Warn-mode hook: append a security-warning block to the tool result.

    Returning a string replaces the result that the model sees in the next
    turn. Returning None leaves the result unchanged.
    """
    # Block mode handles findings via pre_tool_call; nothing for this hook
    # to do in that case (the tool didn't run, so there's no result to wrap).
    if _block_mode_enabled():
        return None
    if _warn_only_enabled():
        _audit_warn_only_override()
    findings = _scan_args(tool_name, args)
    if not findings:
        return None
    if not isinstance(result, str):
        return None
    # Don't decorate error results — the model already has bigger problems.
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and "error" in parsed and len(parsed) <= 2:
            return None
    except (ValueError, TypeError):
        pass
    return result + "\n\n" + _format_warning_block(findings)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
