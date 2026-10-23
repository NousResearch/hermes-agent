"""Bounded fallback tool summaries and recognition across compression passes."""

from __future__ import annotations

from agent.compression_marker import _COMPRESSION_MARKER_RE, _elision_marker, elide
from agent.context_compressor_tail import is_demoted_stub


def _has_terminal_compression_marker(content: str) -> bool:
    """Whether ``content`` ends in one complete, count-bearing compression marker."""
    return content.endswith("⟫") and any(
        match.end() == len(content) - 1
        for match in _COMPRESSION_MARKER_RE.finditer(content)
    )


def is_summary_stub(content: str) -> bool:
    """True for a bounded one-line result summary or a lean-tail demotion stub."""
    from agent.context_compressor import _AUTO_FOCUS_MAX_CHARS

    if not content.startswith("["):
        return False
    return (
        (" chars)" in content and len(content) < 400)
        or (
            content.startswith("[mcp__")
            and len(content) <= _AUTO_FOCUS_MAX_CHARS
            and " chars result): " in content
            and _has_terminal_compression_marker(content)
        )
        or is_demoted_stub(content)
    )


def summarize_unknown_tool_result(tool_name: str, args: dict, content: str) -> str:
    """Keep bounded MCP evidence; other unknown tools retain the count-only fallback."""
    from agent.context_compressor import _AUTO_FOCUS_TURN_MAX_CHARS, _result_failure_suffix

    content_len = len(content)
    first_arg = "".join(f" {k}={str(v)[:40]}" for k, v in list(args.items())[:2])
    if tool_name.startswith("mcp__") and content:
        excerpt = " ".join(content.split()) or "[whitespace-only result]"
        excerpt = elide(excerpt, _AUTO_FOCUS_TURN_MAX_CHARS)
        if not _has_terminal_compression_marker(excerpt):
            excerpt += _elision_marker(omitted=0, total=len(excerpt))
        return (
            f"[{tool_name}]{first_arg} ({content_len:,} chars result):"
            f"{_result_failure_suffix(content)} "
            f"{excerpt}"
        )
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)" + _result_failure_suffix(content)
