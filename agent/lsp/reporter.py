"""Format LSP diagnostics for inclusion in tool output.

The model sees a compact, severity-filtered, line-bounded ``<diagnostics>``
block (1-indexed line/column, capped at ``MAX_PER_FILE``) for diagnostics
introduced by the latest edit.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List

# ERROR only by default — warnings/info/hints would flood the agent.
SEVERITY_NAMES = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "HINT"}
DEFAULT_SEVERITIES = frozenset({1})

MAX_PER_FILE = 20
MAX_TOTAL_CHARS = 4000

# Per-field caps bound any single attacker-controlled identifier that can
# ride into the model's tool output via an LSP diagnostic message.
MAX_MESSAGE_CHARS = 300
MAX_CODE_CHARS = 80
MAX_SOURCE_CHARS = 80


def _sanitize_field(value: Any, *, limit: int) -> str:
    """Make a language-server field safe to embed in a tool-result block.

    ``message``/``code``/``source`` come from a server that just parsed user-controlled code, so a
    hostile repo can smuggle instruction-shaped text through identifier names.  We collapse CR/LF,
    drop control chars, cap the length, and HTML-escape ``< > &`` so the text can't close
    ``<diagnostics>`` early.  ``None``/empty → ``""`` so callers can omit the part.
    """
    if value is None:
        return ""
    raw = str(value).replace("\r", " ").replace("\n", " ")
    raw = "".join(ch for ch in raw if ch == " " or ch.isprintable())
    return html.escape(raw.strip()[:limit], quote=False)


def format_diagnostic(d: Dict[str, Any]) -> str:
    """One-line representation of a single diagnostic (fields sanitized)."""
    sev = SEVERITY_NAMES.get(d.get("severity") or 1, "ERROR")
    start = (d.get("range") or {}).get("start") or {}
    line = int(start.get("line", 0)) + 1
    col = int(start.get("character", 0)) + 1
    msg = _sanitize_field(d.get("message"), limit=MAX_MESSAGE_CHARS)
    code = _sanitize_field(d.get("code"), limit=MAX_CODE_CHARS)
    source = _sanitize_field(d.get("source"), limit=MAX_SOURCE_CHARS)
    return f"{sev} [{line}:{col}] {msg}{f' [{code}]' if code else ''}{f' ({source})' if source else ''}"


def report_for_file(
    file_path: str,
    diagnostics: List[Dict[str, Any]],
    *,
    severities: frozenset = DEFAULT_SEVERITIES,
    max_per_file: int = MAX_PER_FILE,
) -> str:
    """Build a ``<diagnostics file=...>`` block; ``""`` when nothing passes the severity filter."""
    filtered = [d for d in diagnostics or [] if (d.get("severity") or 1) in severities]
    if not filtered:
        return ""
    body = "\n".join(format_diagnostic(d) for d in filtered[:max_per_file])
    if len(filtered) > max_per_file:
        body += f"\n... and {len(filtered) - max_per_file} more"
    # quote=True also escapes ``"`` so a crafted file name can't break out of
    # the ``file="..."`` attribute and synthesize new tags.
    safe_path = html.escape(file_path, quote=True)
    return f"<diagnostics file=\"{safe_path}\">\n{body}\n</diagnostics>"


def truncate(s: str, *, limit: int = MAX_TOTAL_CHARS) -> str:
    """Hard-cap a formatted summary string."""
    if len(s) <= limit:
        return s
    marker = "\n…[truncated]"
    return s[: limit - len(marker)] + marker


__all__ = ["SEVERITY_NAMES", "DEFAULT_SEVERITIES", "MAX_PER_FILE", "format_diagnostic", "report_for_file", "truncate"]
