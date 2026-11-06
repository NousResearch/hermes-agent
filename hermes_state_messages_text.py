"""Text/count helpers for stored message columns (topical sibling of ``hermes_state_messages``).

Moved verbatim from the messages facade so the facade only shrinks (code-health
file-size ratchet); behavior is unchanged.
"""

from __future__ import annotations

from typing import Any

from agent.message_sanitization import _sanitize_surrogates


def _tool_calls_count(tool_calls: Any) -> int:
    return 0 if tool_calls is None else (len(tool_calls) if isinstance(tool_calls, list) else 1)


def _tool_calls_len(raw: Any, scalar: int = 0) -> int:
    """Count of a stored ``tool_calls`` column: list length, *scalar* for a truthy non-list, else 0."""
    from hermes_state_messages import _parse_tool_calls
    parsed = _parse_tool_calls(raw)
    return len(parsed) if isinstance(parsed, list) else (scalar if parsed else 0)


def _scrub_surrogates(value: Any) -> Any:
    """Lone surrogates make sqlite3 raise UnicodeEncodeError and abort the whole write."""
    return _sanitize_surrogates(value) if isinstance(value, str) else value
