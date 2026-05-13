"""Format LSP diagnostics for inclusion in tool output.

The model sees a compact, severity-filtered, line-bounded summary of
diagnostics introduced by the latest edit.  Format matches what
OpenCode's ``lsp/diagnostic.ts`` and Claude Code's
``formatDiagnosticsSummary`` produce — ``<diagnostics>`` blocks with
1-indexed line/column, capped at ``MAX_PER_FILE`` errors.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Severity-1 only by default — warnings/info/hints would flood the
# agent.  Lift this in config under ``lsp.severities`` if needed.
SEVERITY_NAMES = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "HINT"}
DEFAULT_SEVERITIES = frozenset({1})  # ERROR only

MAX_PER_FILE = 20
MAX_TOTAL_CHARS = 4000


def format_diagnostic(d: Dict[str, Any]) -> str:
    """One-line representation of a single diagnostic."""
    sev = SEVERITY_NAMES.get(d.get("severity") or 1, "ERROR")
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    line = int(start.get("line", 0)) + 1
    col = int(start.get("character", 0)) + 1
    msg = str(d.get("message") or "").rstrip()
    code = d.get("code")
    code_part = f" [{code}]" if code not in (None, "") else ""
    source = d.get("source")
    source_part = f" ({source})" if source else ""
    return f"{sev} [{line}:{col}] {msg}{code_part}{source_part}"


def report_for_file(
    file_path: str,
    diagnostics: List[Dict[str, Any]],
    *,
    severities: frozenset = DEFAULT_SEVERITIES,
    max_per_file: int = MAX_PER_FILE,
) -> str:
    """Build a ``<diagnostics file=...>`` block for one file.

    Returns an empty string when no diagnostics pass the severity
    filter, so callers can do ``if block:`` to skip empty cases.
    """
    if not diagnostics:
        return ""
    filtered = [d for d in diagnostics if (d.get("severity") or 1) in severities]
    if not filtered:
        return ""
    limited = filtered[:max_per_file]
    extra = len(filtered) - len(limited)
    lines = [format_diagnostic(d) for d in limited]
    body = "\n".join(lines)
    if extra > 0:
        body += f"\n... and {extra} more"
    return f"<diagnostics file=\"{file_path}\">\n{body}\n</diagnostics>"


def truncate(s: str, *, limit: int = MAX_TOTAL_CHARS) -> str:
    """Hard-cap a formatted summary string."""
    if len(s) <= limit:
        return s
    marker = "\n…[truncated]"
    return s[: limit - len(marker)] + marker


__all__ = [
    "SEVERITY_NAMES",
    "DEFAULT_SEVERITIES",
    "MAX_PER_FILE",
    "format_diagnostic",
    "report_for_file",
    "truncate",
]
