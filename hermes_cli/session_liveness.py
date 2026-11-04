"""Is a stored session live? One verdict for ``hermes status --json`` and every session list.

The active-session lease registry (``hermes_cli.active_sessions``) is the authority: CLI and
TUI/Desktop sessions hold a lease for as long as their process owns the session, and a dead
owner's lease reads dead the moment its pid is gone. ``ended_at IS NULL`` plus recent activity
is only a fallback for sources whose owner holds no process-lifetime lease (messaging gateway
turns lease per turn, cron runs not at all) and for an unreadable registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import AbstractSet, Any, Optional

logger = logging.getLogger(__name__)

ACTIVITY_WINDOW_S = 300

# Sources whose owner leases the session for its whole open lifetime, so "no live lease"
# means "no live owner": a crashed CLI must not read live for another ACTIVITY_WINDOW_S.
# ``local`` is the gateway's label for a CLI-attached session once its first turn ran.
PROCESS_LEASED_SOURCES = frozenset({"cli", "local", "tui", "desktop"})


def leased_session_ids(registry_home: str | Path | None = None) -> Optional[frozenset[str]]:
    """Session ids with a live lease in ``registry_home``'s registry; None when unreadable."""
    from hermes_cli.active_sessions import ActiveSessionRegistryError, read_live_session_leases

    try:
        leases = read_live_session_leases(registry_home)
    except (ActiveSessionRegistryError, OSError) as exc:
        logger.debug("active-session registry unreadable for %s: %s", registry_home, exc)
        return None
    return frozenset(str(e.get("session_id") or "") for e in leases)


def session_is_active(row: dict[str, Any], now: float, leased: Optional[AbstractSet[str]]) -> bool:
    """The live verdict for one session row (``leased``: ``leased_session_ids`` or None)."""
    if row.get("ended_at") is not None:
        return False
    if leased is not None:
        if row.get("id") in leased:
            return True
        if row.get("source") in PROCESS_LEASED_SOURCES:
            return False
    last = row.get("last_active") or row.get("started_at") or 0
    return now - last < ACTIVITY_WINDOW_S
