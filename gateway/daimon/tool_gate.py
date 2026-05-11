# gateway/daimon/tool_gate.py
"""Session-scoped tool call gating for Daimon user sessions."""
from __future__ import annotations

import threading
from typing import Optional

from gateway.daimon.tool_limiter import ToolLimiter

# Global registry of active session limiters.
# The pre_tool_call hook looks up the session's limiter here.
_session_limiters: dict[str, ToolLimiter] = {}
_lock = threading.Lock()


def register_limiter(session_id: str, limiter: ToolLimiter) -> None:
    """Register a tool limiter for a session."""
    with _lock:
        _session_limiters[session_id] = limiter


def unregister_limiter(session_id: str) -> None:
    """Remove limiter when session ends."""
    with _lock:
        _session_limiters.pop(session_id, None)


def get_limiter(session_id: str) -> Optional[ToolLimiter]:
    """Get the limiter for a session, if any."""
    with _lock:
        return _session_limiters.get(session_id)


def check_tool_call(session_id: str, tool_name: str) -> Optional[str]:
    """Check if a tool call is allowed for a session.

    Args:
        session_id: The session identifier (typically the Discord thread_id,
                    which is used as the session key throughout Daimon).
        tool_name: The tool being called.

    Returns None if allowed (or no limiter registered).
    Returns a denial message string if blocked.

    Check + record is atomic to prevent parallel tool calls from exceeding limits.
    """
    with _lock:
        limiter = _session_limiters.get(session_id)
        if limiter is None:
            return None  # No limiter = no restrictions (admin or non-daimon)

        if not limiter.check(tool_name):
            return limiter.denial_message(tool_name)

        limiter.record(tool_name)
        return None


def active_session_count() -> int:
    """Number of sessions with active limiters."""
    with _lock:
        return len(_session_limiters)
