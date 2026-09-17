"""Detect sessions stranded under a session-key namespace no live profile claims.

Session keys are namespaced ``agent:<ns>:`` (``gateway/session.py::_session_key_namespace``):
``agent:main`` for the default profile, ``agent:<name>`` for a named one. That namespace is
derived from configuration, so a configuration change can silently move every future key —
``gateway.multiplex_profiles`` flipping to default-on did exactly that, stranding 246 sessions
(9,764 messages) on one box with no warning.

``hermes_state_gateway.rekey_profile_state(old, new)`` already migrates a namespace correctly,
but it only runs on an explicit profile rename. This module closes the un-announced case: at
boot, report any namespace holding sessions that no live profile would produce, so the operator
can migrate instead of silently starting from empty chats.

Detection only — it never rewrites the database. Choosing a migration target is the operator's
call, and a wrong guess would merge two profiles' histories.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Iterable

logger = logging.getLogger(__name__)


def find_orphaned_namespaces(db_path: Path, live_profiles: Iterable[str]) -> Dict[str, int]:
    """Map ``profile name -> session count`` for namespaces no live profile claims.

    Best-effort by construction: a missing, unreadable or schema-less database returns ``{}``.
    A guard that can crash the gateway is worse than the bug it detects.
    """
    from gateway.session import profile_from_session_key_namespace

    claimed = {str(name) for name in live_profiles}
    counts: Dict[str, int] = {}
    try:
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        rows = conn.execute(
            "SELECT session_key FROM sessions WHERE session_key LIKE 'agent:%'"
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    finally:
        conn.close()

    for (session_key,) in rows:
        if not session_key:
            continue
        parts = str(session_key).split(":")
        if len(parts) < 2:
            continue
        profile = profile_from_session_key_namespace(parts[1])
        if profile not in claimed:
            counts[profile] = counts.get(profile, 0) + 1
    return counts


def warn_orphaned_namespaces(db_path: Path, live_profiles: Iterable[str]) -> Dict[str, int]:
    """Log one actionable warning per orphaned namespace; return what was found."""
    try:
        orphans = find_orphaned_namespaces(db_path, live_profiles)
    except Exception:
        logger.debug("Orphan-namespace check failed", exc_info=True)
        return {}
    for profile, count in sorted(orphans.items()):
        logger.warning(
            "%d session(s) are stored under profile namespace %r, which no live profile claims. "
            "They will NOT resolve — those chats start empty instead of continuing. "
            "If this profile was renamed or a namespace default changed, migrate with "
            "`hermes profiles migrate-identity %s <live-profile>`; otherwise this is stale history.",
            count, profile, profile,
        )
    return orphans
