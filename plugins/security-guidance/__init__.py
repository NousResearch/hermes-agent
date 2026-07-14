"""security-guidance plugin — fast pattern-matched security warnings on file writes.

Wires one behaviour:

* ``transform_tool_result`` hook — scans the *content being written* by
  ``write_file`` / ``patch`` / ``skill_manage`` (write/patch modes) for known
  dangerous code patterns (eval(, pickle.load, yaml.load, os.system,
  subprocess(shell=True), dangerouslySetInnerHTML, verify=False, ECB,
  XXE-prone XML parsers, GitHub Actions ``${{ github.event.* }}`` injection,
  torch.load without ``weights_only=True``, ...). When any pattern matches,
  the plugin appends a ``⚠️ Security warning`` block to the JSON tool-result
  string. The file is still written; the model sees the warning in the next
  turn's tool message and can self-correct.

Why not block? Patterns have a non-trivial false-positive rate (``eval(`` in
a tokenizer, ``yaml.load`` already wrapped in ``yaml.SafeLoader``, ECB inside
a test fixture). Blocking would force every false positive into an approval
prompt or an interrupted workflow. Warning is the right severity for layer
1 — the agent reads the warning and either fixes the code or briefly
documents why the construct is safe.

For block-mode (refuse the write entirely), set
``SECURITY_GUIDANCE_BLOCK=1``. This trades convenience for strictness and
is intended for shared dev environments where unsafe-by-default patterns
are policy violations.

Pattern data lives in ``patterns.py``, forked verbatim from Anthropic's
``claude-plugins-official`` under Apache-2.0. See ``LICENSE`` and ``NOTICE``
in this directory.
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


def _block_mode_enabled() -> bool:
    return os.environ.get("SECURITY_GUIDANCE_BLOCK", "").lower() in {"1", "true", "yes", "on"}


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
                logger.debug("Suppressed exception", exc_info=True)
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
    """In block mode, refuse the write if any pattern matches.

    Default mode is non-blocking — we return None here and let
    ``transform_tool_result`` append a warning to the result instead.
    """
    if not _block_mode_enabled():
        return None
    findings = _scan_args(tool_name, args)
    if not findings:
        return None
    return {
        "action": "block",
        "message": (
            "security-guidance refused this write: "
            + _format_warning_block(findings)
            + "\n\nTo override, unset SECURITY_GUIDANCE_BLOCK and retry."
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
        logger.debug("Suppressed exception", exc_info=True)
    return result + "\n\n" + _format_warning_block(findings)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
