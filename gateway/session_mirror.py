"""Opt-in persistence for externally initiated gateway turns mirrored to Kanban.

All reads and writes execute while the caller holds the source profile's runtime scope.
The Kanban record contains routing metadata only; it never copies conversation content.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionMirrorRef:
    board: str
    mirror_id: int


def _profile_name(source: Any) -> str:
    from gateway.session_identity import identity_of

    identity = identity_of(source)
    if identity is not None:
        return str(identity.runtime_profile)
    name = getattr(source, "profile", None)
    return name.strip() if isinstance(name, str) and name.strip() else "default"


def _retention_policy(config: Any) -> Optional[int]:
    """Read ``kanban.session_mirror.retention_days`` without the enable/allowlist gates.

    Retention outlives the feature. A user who switches mirroring off — or drops a
    profile/platform from the allowlists — to stop collecting has already collected
    rows naming their chats and sessions; tying the sweep to the same switch that
    stops collection strands every one of them forever. The value is read on its own
    so ``enabled: false`` stops collection without freezing expiry.
    """
    if not isinstance(config, dict):
        return None
    section = config.get("kanban")
    settings = section.get("session_mirror") if isinstance(section, dict) else None
    if not isinstance(settings, dict):
        return None
    retention_days = settings.get("retention_days", 30)
    if isinstance(retention_days, bool) or not isinstance(retention_days, int) or not 0 <= retention_days <= 3650:
        logger.warning("Invalid kanban.session_mirror.retention_days; session mirroring is disabled")
        return None
    return retention_days


# One indexed sweep per profile, mirrored on the notifier GC's cadence
# (gateway/kanban_watchers.py) so the cost is a background write, never per turn.
_SWEEP_INTERVAL_SECONDS = 3600.0
_sweep_next_at: dict[str, float] = {}


def _maybe_sweep_mirrors(config: Any, profile: str) -> None:
    """Run the retention sweep for one profile, at most hourly, whatever the policy says.

    Called on every external turn before the enable gate. ``retention_days: 0``
    means expiry is off, and an invalid value is rejected by ``_retention_policy``,
    so both return before the board is touched.
    """
    retention_days = _retention_policy(config)
    if not retention_days:
        return
    now = time.monotonic()
    if now < _sweep_next_at.get(profile, 0.0):
        return
    _sweep_next_at[profile] = now + _SWEEP_INTERVAL_SECONDS

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db_session_mirror import prune_expired_mirrors

    try:
        with kbc.connect_closing(board=kb.get_current_board()) as conn:
            prune_expired_mirrors(conn, retention_days=retention_days, profile=profile)
    except Exception:
        logger.warning("Could not prune expired Kanban session mirrors", exc_info=True)


def _mirror_policy(config: Any, profile: str, platform: str) -> Optional[int]:
    if not isinstance(config, dict):
        return None
    section = config.get("kanban")
    settings = section.get("session_mirror") if isinstance(section, dict) else None
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        return None
    if settings.get("mode") != "read_only":
        return None
    profiles = settings.get("profiles")
    platforms = settings.get("platforms")
    if not isinstance(profiles, list) or not isinstance(platforms, list):
        return None
    if not all(isinstance(value, str) and value.strip() for value in profiles + platforms):
        return None
    if profile.casefold() not in {value.strip().casefold() for value in profiles}:
        return None
    if platform.casefold() not in {value.strip().casefold() for value in platforms}:
        return None
    return _retention_policy(config)


def begin_session_mirror(
    event: Any, source: Any, session_id: str, config: Any, *, scheduled_heartbeat: bool = False,
) -> Optional[SessionMirrorRef]:
    """Create/mark a mirror only after a real external event enters an agent run."""
    if scheduled_heartbeat or event is None or getattr(event, "internal", False):
        return None
    message_id = getattr(event, "message_id", None)
    if not isinstance(message_id, (str, int)) or not str(message_id).strip():
        return None
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    platform = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None))
    platform = platform.strip() if isinstance(platform, str) else ""
    profile = _profile_name(source)
    if not platform or not profile:
        return None
    # Before the gate on purpose: switching mirroring off (or narrowing the
    # allowlists) must stop COLLECTION, not freeze expiry of what was already
    # collected. `_maybe_sweep_mirrors` is the only call site that reaches the
    # sweep when no turn is mirrored.
    _maybe_sweep_mirrors(config, profile)
    retention_days = _mirror_policy(config, profile, platform)
    if retention_days is None:
        return None
    chat_id = getattr(source, "chat_id", None)
    if not isinstance(chat_id, str) or not chat_id.strip():
        return None

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db_session_mirror import (
        create_or_get_mirror, mark_mirror_running, prune_expired_mirrors,
    )

    board = kb.get_current_board()
    timestamp = int(time.time())
    try:
        with kbc.connect_closing(board=board) as conn:
            prune_expired_mirrors(conn, retention_days=retention_days, now=timestamp, profile=profile)
            mirror_id, _created = create_or_get_mirror(
                conn,
                profile=profile,
                platform=platform,
                chat_id=chat_id,
                thread_id=getattr(source, "thread_id", None),
                session_id=session_id,
                message_id=str(message_id),
                now=timestamp,
            )
            mark_mirror_running(conn, mirror_id, now=timestamp)
        return SessionMirrorRef(board=str(board), mirror_id=mirror_id)
    except Exception:
        logger.warning("Could not create the Kanban session mirror; continuing the gateway turn", exc_info=True)
        return None


def finish_session_mirror(ref: Optional[SessionMirrorRef], result: Any = None, *, cancelled: bool = False) -> None:
    """Persist the agent-run outcome, independently of whether outbound delivery succeeded."""
    if ref is None:
        return
    if cancelled or (isinstance(result, dict) and result.get("interrupted")):
        status = "cancelled"
    elif (
        not isinstance(result, dict)
        or result.get("failed")
        or result.get("partial")
        or result.get("error")
        or result.get("completed") is False
    ):
        status = "failed"
    else:
        status = "completed"

    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db_session_mirror import finish_mirror

    try:
        with kbc.connect_closing(board=ref.board) as conn:
            finish_mirror(conn, ref.mirror_id, status)
    except Exception:
        logger.warning("Could not finalize the Kanban session mirror; the gateway turn remains unaffected", exc_info=True)
