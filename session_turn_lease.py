"""Shared wording and recognition for session turn-lease wait refreshes."""

import re


SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE = (
    "⏳ Still waiting for the other Hermes process on "
    "this session ({elapsed_seconds}s)..."
)

_SESSION_TURN_LEASE_REFRESH_RE = None


def session_turn_lease_refresh_re():
    """Compile the matcher from the same template used by the emit site."""
    global _SESSION_TURN_LEASE_REFRESH_RE
    if _SESSION_TURN_LEASE_REFRESH_RE is None:
        parts = re.split(
            r"\{[^{}]*\}",
            SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE,
        )
        _SESSION_TURN_LEASE_REFRESH_RE = re.compile(
            r"[\d,]+".join(re.escape(part) for part in parts),
            re.IGNORECASE,
        )
    return _SESSION_TURN_LEASE_REFRESH_RE


def format_session_turn_lease_wait_refresh(elapsed_seconds: int) -> str:
    """Format one periodic session turn-lease wait refresh."""
    return SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE.format(
        elapsed_seconds=elapsed_seconds
    )


def is_session_turn_lease_wait_refresh(message: object) -> bool:
    """Return whether ``message`` is the periodic lease-wait refresh."""
    return bool(session_turn_lease_refresh_re().search(str(message or "")))
