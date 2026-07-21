"""security-guidance plugin — pattern-matched security warnings on code writes.

Layer 1: Fast static analysis that scans the *content being written* by
``write_file``, ``patch``, ``skill_manage``, ``execute_code``, and
``terminal`` for 25 known-dangerous code patterns (eval(, pickle.load,
yaml.load, os.system, subprocess(shell=True), dangerouslySetInnerHTML,
verify=False, ECB, XXE-prone XML parsers, GitHub Actions injection,
torch.load without weights_only=True, ...).

Two warning delivery paths:

* ``transform_tool_result`` hook — appends a ``⚠️ Security warning`` block to the
  JSON tool-result string. The file is still written; the model sees the
  warning in the next turn and can self-correct.
* ``pre_llm_call`` hook — injects accumulated findings into the LLM prompt
  context as a markdown advisory. Useful when the model needs the full
  severity-ranked picture in one place.

Block mode (``SECURITY_GUIDANCE_BLOCK=1``) refuses the write entirely.
Disable switch (``SECURITY_GUIDANCE_DISABLE=1``) is the kill switch.

On-demand audit via the ``security_scan`` tool::

    security_scan target="/src" scope="directory"

Pattern data lives in ``patterns.py``, forked verbatim from Anthropic's
``claude-plugins-official`` under Apache-2.0. See ``LICENSE`` and ``NOTICE``
in this directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import patterns as _patterns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tool names whose args carry "code being written to disk" or "code/commands
# about to be executed" that we want to scan.
_TARGET_TOOLS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "write_file": ("path", ("content",)),
    "patch": ("path", ("new_string", "patch")),
    # skill_manage write_file / patch sub-actions.
    "skill_manage": ("file_path", ("file_content", "new_string")),
    # execute_code writes a temp file and runs it — the code itself is
    # the content to scan.
    "execute_code": ("", ("code",)),
    # terminal commands can contain shell injection patterns.
    "terminal": ("", ("command",)),
}

_MAX_SCAN_BYTES = 256 * 1024

_SEVERITY_ORDER = {"critical": 1, "high": 2, "medium": 3, "low": 4, "info": 5}

# Map upstream rule names (from patterns.py) to severity tiers.
_RULE_SEVERITY: Dict[str, str] = {
    "github_actions_workflow":          "high",
    "child_process_exec":               "critical",
    "new_function_injection":           "critical",
    "eval_injection":                   "critical",
    "react_dangerously_set_html":       "high",
    "document_write_xss":               "high",
    "innerHTML_xss":                    "high",
    "pickle_deserialization":           "critical",
    "os_system_injection":              "critical",
    "python_subprocess_shell":          "critical",
    "go_exec_shell_injection":          "critical",
    "unsafe_yaml_load":                 "critical",
    "node_createcipher_no_iv":          "high",
    "aes_ecb_mode":                     "high",
    "tls_verification_disabled":        "high",
    "marshal_loads":                    "critical",
    "shelve_open":                      "critical",
    "xml_unsafe_parse":                 "high",
    "pickle_variants_load":             "critical",
    "outerHTML_xss":                    "high",
    "insertAdjacentHTML_xss":           "high",
    "script_src_without_sri":           "medium",
    "torch_unsafe_load":                "critical",
    "yaml_unsafe_load_variants":        "critical",
    "pickle_wrapper_load":              "critical",
}

_SEVERITY_EMOJI = {
    "critical": "🚨",
    "high":     "⚠️",
    "medium":   "🔶",
    "low":      "🔹",
    "info":     "ℹ️",
}


def _block_mode_enabled() -> bool:
    return os.environ.get("SECURITY_GUIDANCE_BLOCK", "").lower() in {"1", "true", "yes", "on"}


def _plugin_disabled() -> bool:
    return os.environ.get("SECURITY_GUIDANCE_DISABLE", "").lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# In-memory per-turn advisory buffer (used by post_tool_call → pre_llm_call)
# ---------------------------------------------------------------------------

_TURN_ADVISORIES: List[Dict[str, Any]] = []


def _add_advisory(rule_name: str, reminder: str) -> None:
    _TURN_ADVISORIES.append({
        "ruleName": rule_name,
        "severity": _RULE_SEVERITY.get(rule_name, "medium"),
        "reminder": reminder,
    })


def _flush_advisories() -> str:
    """Build a concise markdown advisory for injection into the LLM prompt."""
    if not _TURN_ADVISORIES:
        return ""

    lines: List[str] = ["", "🔒 Security Guidance — findings from this turn:", ""]
    for item in sorted(
        _TURN_ADVISORIES,
        key=lambda x: _SEVERITY_ORDER.get(x["severity"], 99),
    ):
        sev = item["severity"]
        emoji = _SEVERITY_EMOJI.get(sev, "ℹ️")
        lines.append(f"{emoji} **{sev.upper()}** — ``{item['ruleName']}``")
        lines.append(f"   → {item['reminder'].split(chr(10))[0]}")
        lines.append("")

    lines.append(
        "Pattern matches can be false positives. If the construct is safe in this "
        "context, briefly document why in a code comment and continue. Otherwise, "
        "fix the code before moving on."
    )
    _TURN_ADVISORIES.clear()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

_COMPILED: List[Dict[str, Any]] = []
for _rule in _patterns.SECURITY_PATTERNS:
    _entry: Dict[str, Any] = {
        "ruleName": _rule["ruleName"],
        "severity": _RULE_SEVERITY.get(_rule["ruleName"], "medium"),
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


def _scan_content(path: str, content: str) -> List[Tuple[str, str, str]]:
    """Return [(ruleName, severity, reminder), ...] for every pattern that matches.

    ``path`` is used by per-rule path filters (path_filter / path_check).
    Each rule fires at most once per call.
    """
    if not content or len(content.encode("utf-8", errors="ignore")) > _MAX_SCAN_BYTES:
        return []
    hits: List[Tuple[str, str, str]] = []
    for entry in _COMPILED:
        path_check = entry.get("path_check")
        if path_check is not None:
            try:
                if path_check(path or ""):
                    hits.append((
                        entry["ruleName"],
                        entry["severity"],
                        entry["reminder"],
                    ))
            except Exception:
                pass
            continue
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
            hits.append((
                entry["ruleName"],
                entry["severity"],
                entry["reminder"],
            ))
    return hits


def _extract_path_and_content(tool_name: str, args: Any) -> List[Tuple[str, str]]:
    """Return [(path, content), ...] for a tool call. Empty if nothing to scan."""
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


def _format_warning_block(findings: List[Tuple[str, str, str]]) -> str:
    """Render findings into a Markdown block appended to the tool result."""
    # Sort by severity (critical first)
    findings = sorted(
        findings,
        key=lambda x: _SEVERITY_ORDER.get(x[1], 99),
    )
    names = ", ".join(name for name, _, _ in findings)
    lines = [
        "",
        "---",
        f"⚠️ Security guidance — {len(findings)} pattern{'s' if len(findings) != 1 else ''} matched ({names})",
        "",
    ]
    for rule_name, severity, reminder in findings:
        emoji = _SEVERITY_EMOJI.get(severity, "ℹ️")
        lines.append(f"{emoji} **{severity.upper()}** — ``{rule_name}``")
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

def _scan_args(tool_name: str, args: Any) -> List[Tuple[str, str, str]]:
    """Common scan path used by hooks."""
    if _plugin_disabled():
        return []
    findings: List[Tuple[str, str, str]] = []
    for path, content in _extract_path_and_content(tool_name, args):
        findings.extend(_scan_content(path, content))
    return findings


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[Dict[str, str]]:
    """In block mode, refuse the write if any pattern matches."""
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
    """Warn-mode hook: append a security-warning block to the tool result."""
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
        pass
    return result + "\n\n" + _format_warning_block(findings)


def _on_post_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> None:
    """Buffer findings so they can be injected via pre_llm_call."""
    if _block_mode_enabled() or _plugin_disabled():
        return
    for rule_name, severity, reminder in _scan_args(tool_name, args):
        _add_advisory(rule_name, reminder)


def _on_pre_llm_call(
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    system_message: Optional[str] = None,
    **_: Any,
) -> Optional[Dict[str, Any]]:
    """Inject accumulated advisories into the prompt context before the LLM call."""
    advisory = _flush_advisories()
    if not advisory:
        return None
    return {"context": advisory}


# ---------------------------------------------------------------------------
# Tool: security_scan (on-demand audit)
# ---------------------------------------------------------------------------

def _walk_code_files(root: Path) -> List[Path]:
    """Yield readable code files under root, skipping venvs and common junk."""
    SKIP_DIRS = {
        ".git", ".hermes", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".pytest_cache", ".mypy_cache", ".tox",
    }
    CODE_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".rb", ".php", ".c", ".cpp", ".h", ".cs", ".swift", ".kt",
        ".scala", ".sh", ".bash", ".zsh", ".ps1", ".sql", ".html",
        ".htm", ".xml", ".yaml", ".yml", ".json", ".md",
    }
    files: List[Path] = []
    for entry in root.rglob("*"):
        if any(part in SKIP_DIRS for part in entry.parts):
            continue
        if entry.is_file() and entry.suffix.lower() in CODE_EXTS:
            files.append(entry)
    return files


def security_scan(args: Dict[str, Any], **kwargs: Any) -> str:
    """Scan files, directories, or raw text for common security vulnerability patterns.

    Args:
        target (str): File path, directory path, or raw text to scan.
        scope (str, optional): ``file`` | ``directory`` | ``text``. Auto-inferred if omitted.
    """
    target = args.get("target", "")
    scope = args.get("scope", "")
    if not target:
        return "❌ No target provided. Pass ``target`` (path or text)."

    if not scope:
        p = Path(target)
        if p.exists():
            scope = "directory" if p.is_dir() else "file"
        else:
            scope = "text"

    findings: List[Tuple[str, str, str]] = []

    if scope == "file":
        path = Path(target)
        if not path.exists():
            return f"❌ File not found: ``{target}``"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"❌ Cannot read ``{target}``: {exc}"
        findings = _scan_content(str(path), text)

    elif scope == "directory":
        root = Path(target)
        if not root.exists():
            return f"❌ Directory not found: ``{target}``"
        for fp in _walk_code_files(root):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            findings.extend(_scan_content(str(fp), text))

    elif scope == "text":
        findings = _scan_content("input_text", target)
    else:
        return f"❌ Unknown scope ``{scope}``. Use ``file`` | ``directory`` | ``text``."

    if not findings:
        return "✅ No known vulnerability patterns detected in target."

    lines = [f"# Security Scan Report — {target}", ""]
    summary: Dict[str, int] = {}
    for rule_name, severity, _ in findings:
        summary[severity] = summary.get(severity, 0) + 1

    lines.append("## Summary")
    for sev in ("critical", "high", "medium", "low", "info"):
        if summary.get(sev):
            emoji = _SEVERITY_EMOJI.get(sev, "ℹ️")
            lines.append(f"- {emoji} **{sev.upper()}**: {summary[sev]}")
    lines.append("")

    lines.append("## Findings")
    for rule_name, severity, reminder in sorted(
        findings, key=lambda x: _SEVERITY_ORDER.get(x[1], 99)
    ):
        emoji = _SEVERITY_EMOJI.get(severity, "ℹ️")
        lines.append(f"### {emoji} ``{rule_name}`` ({severity.upper()})")
        lines.append(reminder)
        lines.append("")

    return "\n".join(lines)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_tool(
        name="security_scan",
        description=(
            "Scan files, directories, or raw text for known-dangerous code patterns. "
            "Supports SQL injection, XSS, command injection, hardcoded secrets, unsafe eval, "
            "path traversal, SSRF, insecure deserialization, and more. "
            "Args: target (str) — path or text; scope (str, optional) — file | directory | text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "File path, directory path, or raw text to scan.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["file", "directory", "text"],
                    "description": "Optional: 'file', 'directory', or 'text'. Auto-inferred if omitted.",
                },
            },
            "required": ["target"],
        },
        handler=security_scan,
    )
