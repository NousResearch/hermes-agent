"""Format LSP diagnostics for inclusion in tool output.

The model sees a compact, severity-filtered, line-bounded summary of
diagnostics introduced by the latest edit.  Format matches what
OpenCode's ``lsp/diagnostic.ts`` and Claude Code's
``formatDiagnosticsSummary`` produce — ``<diagnostics>`` blocks with
1-indexed line/column, capped at ``MAX_PER_FILE`` errors.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List

# Severity-1 only by default — warnings/info/hints would flood the
# agent.  Lift this in config under ``lsp.severities`` if needed.
SEVERITY_NAMES = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "HINT"}
DEFAULT_SEVERITIES = frozenset({1})  # ERROR only

MAX_PER_FILE = 20
MAX_TOTAL_CHARS = 4000

# Per-field caps for diagnostic content sourced from the language server.
# These bound the length of any single attacker-controlled identifier that
# can ride into the model's tool output via an LSP diagnostic message.
MAX_MESSAGE_CHARS = 300
MAX_CODE_CHARS = 80
MAX_SOURCE_CHARS = 80


def _sanitize_field(value: Any, *, limit: int) -> str:
    """Make a language-server field safe to embed in a tool-result block.

    Diagnostic ``message``, ``code``, and ``source`` originate from a
    language server that has just parsed user-controlled source code, so
    they're untrusted from the agent's point of view. A hostile repo can
    place instruction-shaped text inside identifier names, type aliases,
    or import paths so the resulting diagnostic echoes that text back
    into the ``<diagnostics>`` block the model reads.

    This helper:

    * Collapses CR/LF so a raw newline can't synthesize a new line in the
      formatted block.
    * Drops non-printable ASCII control characters that have no business
      in a single-line summary.
    * Caps length per-field so a long identifier can't push past the
      block boundary.
    * HTML-escapes ``< > &`` so the result can't close ``<diagnostics>``
      early or open a new tag.

    Returns ``""`` for ``None`` / empty so the surrounding format string
    naturally omits the part (mirrors the prior ``if code not in {None,
    ""}`` check at call sites).
    """
    if value is None:
        return ""
    raw = str(value)
    # Collapse newlines so identifier text with raw \n can't fake new lines.
    raw = raw.replace("\r", " ").replace("\n", " ")
    # Drop ASCII control chars; keep regular spaces.
    raw = "".join(ch for ch in raw if ch == " " or ch.isprintable())
    raw = raw.strip()[:limit]
    return html.escape(raw, quote=False)


def format_diagnostic(d: Dict[str, Any]) -> str:
    """One-line representation of a single diagnostic.

    ``message``, ``code``, and ``source`` are sanitized before
    interpolation — see ``_sanitize_field``.
    """
    sev = SEVERITY_NAMES.get(d.get("severity") or 1, "ERROR")
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    line = int(start.get("line", 0)) + 1
    col = int(start.get("character", 0)) + 1
    msg = _sanitize_field(d.get("message"), limit=MAX_MESSAGE_CHARS)
    code = _sanitize_field(d.get("code"), limit=MAX_CODE_CHARS)
    code_part = f" [{code}]" if code else ""
    source = _sanitize_field(d.get("source"), limit=MAX_SOURCE_CHARS)
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
    # quote=True escapes both ``"`` and ``&`` so a crafted file name like
    # ``foo"><script`` can't break out of the ``file="..."`` attribute and
    # synthesize new tags inside the tool output.
    safe_path = html.escape(file_path, quote=True)
    return f"<diagnostics file=\"{safe_path}\">\n{body}\n</diagnostics>"


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
