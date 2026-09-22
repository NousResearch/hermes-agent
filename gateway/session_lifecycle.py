"""SessionStore explicit suspension, crash-recovery markers, pruning and shared clock/id helpers."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from hermes_state_ids import new_session_id

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.session")

TYPED_EVENT_RECOVERY_METADATA_KEY = "_gateway_system_event_recovery_v1"
_TYPED_EVENT_RECOVERY_VERSION = 1


def _now() -> datetime:
    """Return the current local time."""
    return datetime.now()


def _new_session_id(now: datetime) -> str:
    return new_session_id(now, hex_len=8)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_iso(value) -> Optional[datetime]:
    """``datetime.fromisoformat`` that returns None for empty/malformed input."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# Auto-continue freshness window (1 hour) after the ``resume_pending`` mark; ``gateway/run.py``
# bridges config.yaml ``agent.gateway_auto_continue_freshness`` into the env var at startup.
_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60


def auto_continue_freshness_window() -> float:
    """Resume-scheduler freshness window; stale automation never discards the transcript."""
    raw = os.environ.get("HERMES_AUTO_CONTINUE_FRESHNESS")
    try:
        return float(raw) if raw else float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    except (TypeError, ValueError):
        return float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)


class SessionLifecycleMixin:
    """SessionStore explicit boundaries and crash-recovery markers."""

    def _is_session_ended_in_db(self, session_id: str) -> bool:
        """True iff state.db has this session with a non-null end_reason (same staleness test as
        ``_prune_stale_sessions_locked``; no DB/row or DB error -> False). Lets routing self-heal a
        session ended while the gateway stays alive. Store resolved from the owning profile.

        Used by ``get_or_create_session`` to self-heal at routing time: ``_prune_stale_sessions_locked``
        only runs at startup, so a session ended in the DB while the gateway stays alive (any path that
        finalizes the row without clearing sessions.json) would otherwise be reused as a live routing key
        and silently swallow every subsequent message until the next restart (#54878 — the live-gateway
        variant of #52804/FM9). DB errors are non-fatal — never block routing on a failed lookup.
        The store is resolved from the row's owning profile rather than the ambient scope: an unscoped
        background writer keeps its own copy of the same session, and comparing against that copy reports a
        live session as ended (#66887).
        """
        db = self._db_for_session_id(session_id)
        if not db or not session_id:
            return False
        try:
            row = db.get_session(session_id)
        except Exception:
            return False
        return bool(row is not None and row.get("end_reason") is not None)

    def _route_reset_reason(self, entry: SessionEntry) -> Optional[str]:
        """Only explicit suspension replaces a routed conversation; time never does."""
        return "suspended" if entry.suspended else None

    def _update_entry(self, session_key: str, mutate) -> bool:
        """Apply ``mutate(entry)`` under ``_lock`` and full-save; False when the entry is missing
        or *mutate* returned False (nothing to persist)."""
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or mutate(entry) is False:
                return False
            self._save()
            return True

    def _update_all_entries_locked(self, mutate) -> int:
        """Apply ``mutate(entry) -> bool`` to every entry under ``_lock``; save once if any
        returned True. Returns the count that did."""
        with self._lock:
            self._ensure_loaded_locked()
            changed = sum(1 for entry in self._entries.values() if mutate(entry))
            if changed:
                self._save()
        return changed

    def suspend_session(self, session_key: str) -> bool:
        """Mark a session suspended so it auto-resets on next access (/stop). True if it existed.

        Used by ``/stop`` to prevent stuck sessions from being resumed after a gateway restart (#7536).
        """
        return self._update_entry(session_key, lambda e: setattr(e, "suspended", True))

    def _set_turn_marker_locked(self, session_key: str, entry: SessionEntry, token, started_at) -> None:
        """Persist the active-turn pair BEFORE publishing it in memory, so a failed write can
        neither leak an unowned token nor drop a live one. Lock held."""
        candidate = entry.to_dict()
        candidate["active_turn_token"] = token
        candidate["active_turn_started_at"] = _iso(started_at)
        if started_at is not None:
            # Keeps the legacy 120s startup heuristic working for an older binary during a rolling
            # downgrade/upgrade window.
            candidate["updated_at"] = started_at.isoformat()
        self._save_entry(session_key, entry_data=candidate, lock_held=True)
        entry.active_turn_token = token
        entry.active_turn_started_at = started_at
        if started_at is not None:
            entry.updated_at = started_at

    def mark_turn_active(
        self,
        session_key: str,
        *,
        expected_session_id: Optional[str] = None,
        clear_typed_event_recovery: bool = False,
    ) -> Optional[str]:
        """Persist exact ownership of the agent turn running for *session_key*.

        The opaque token is returned to the caller and must be supplied to
        :meth:`clear_turn_active`.  Re-marking replaces the previous token so
        a stale asynchronous unwind cannot clear a newer turn.
        """
        token = uuid.uuid4().hex
        with self._lock:
            self._ensure_loaded_locked()
            entry = self._entries.get(session_key)
            if entry is None:
                return None
            if expected_session_id is not None and entry.session_id != expected_session_id:
                return None
            if clear_typed_event_recovery and expected_session_id is None:
                return None
            now = _now()
            candidate = entry.to_dict()
            candidate["active_turn_token"] = token
            candidate["active_turn_started_at"] = now.isoformat()
            # Keep the legacy 120-second startup heuristic effective during a
            # rolling downgrade/upgrade window where an older binary cannot
            # understand the exact marker fields.
            candidate["updated_at"] = now.isoformat()

            recovery_aliases = (
                [
                    alias
                    for alias in self._entries.values()
                    if alias.session_id == expected_session_id
                    and TYPED_EVENT_RECOVERY_METADATA_KEY in alias.metadata
                ]
                if clear_typed_event_recovery
                else []
            )
            if recovery_aliases:
                data, generation = self._snapshot_routing_locked()
                active_data = data[session_key]
                active_data["active_turn_token"] = token
                active_data["active_turn_started_at"] = now.isoformat()
                active_data["updated_at"] = now.isoformat()
                for alias in recovery_aliases:
                    alias_data = data[alias.session_key]
                    metadata = dict(alias_data.get("metadata") or {})
                    metadata.pop(TYPED_EVENT_RECOVERY_METADATA_KEY, None)
                    alias_data["metadata"] = metadata
                self._persist_routing_data(data, generation)
                for alias in recovery_aliases:
                    alias.metadata.pop(TYPED_EVENT_RECOVERY_METADATA_KEY, None)
            else:
                # Persist before publishing the marker in memory.  If the durable
                # write raises, a later unrelated save cannot leak an unowned token.
                self._save_entry(
                    session_key,
                    entry_data=candidate,
                    lock_held=True,
                )
            entry.active_turn_token = token
            entry.active_turn_started_at = now
            entry.updated_at = now
        return token


    def clear_turn_active(self, session_key: str, token: str) -> bool:
        """Compare-and-swap clear an active-turn marker; ``False`` when the entry disappeared or a
        newer turn owns it."""
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None or entry.active_turn_token != token:
                return False
            self._set_turn_marker_locked(session_key, entry, None, None)
        return True

    def recover_interrupted_turns(self, max_age_seconds: int = 60 * 60) -> int:
        """Promote crash-left turn markers into ``resume_pending`` (unclean startup only).
        Old/invalid markers are cleared without resuming; suspended sessions are never re-armed.
        Returns the number of newly promoted sessions."""
        now = _now()
        max_age = timedelta(seconds=max(0, max_age_seconds))
        promoted = 0

        def _promote(entry: SessionEntry) -> bool:
            nonlocal promoted
            if entry.session_id in self._typed_event_recovery_session_ids_locked():
                return False
            if not entry.active_turn_token:
                return False
            started_at = entry.active_turn_started_at
            try:
                marker_is_stale = started_at is None or (
                    max_age_seconds > 0 and now - started_at > max_age
                )
            except TypeError:
                # Mixed aware/naive timestamps: clear rather than risk an unsafe old resume.
                marker_is_stale = True
            if not marker_is_stale and not entry.suspended:
                if entry.resume_pending:
                    # A drain-timeout marker is more specific; keep it.
                    if entry.last_resume_marked_at is None:
                        entry.last_resume_marked_at = now
                else:
                    entry.resume_pending = True
                    entry.resume_reason = "restart_interrupted"
                    entry.last_resume_marked_at = now  # freshness starts at discovery
                    promoted += 1
            entry.active_turn_token = None
            entry.active_turn_started_at = None
            return True

        self._update_all_entries_locked(_promote)
        return promoted

    def discard_active_turn_markers(self) -> int:
        """Clear orphan turn markers after a verified clean shutdown."""
        def _discard(entry: SessionEntry) -> bool:
            if not entry.active_turn_token and entry.active_turn_started_at is None:
                return False
            entry.active_turn_token = None
            entry.active_turn_started_at = None
            return True
        return self._update_all_entries_locked(_discard)

    def mark_resume_pending(self, session_key: str, reason: str = "restart_timeout") -> bool:
        """Mark a session resumable after a restart interruption (keeps the session_id/transcript,
        unlike ``suspend_session``). True if marked."""
        def _apply(entry: SessionEntry):
            if entry.session_id in self._typed_event_recovery_session_ids_locked():
                return False
            if entry.suspended:  # never override an explicit ``suspended`` (hard forced-wipe)
                return False
            entry.resume_pending = True
            entry.resume_reason = reason
            entry.last_resume_marked_at = _now()
        return self._update_entry(session_key, _apply)

    def clear_resume_pending(self, session_key: str) -> bool:
        """Clear the resume-pending flag after a successful resumed turn; True if cleared."""
        def _apply(entry: SessionEntry):
            if not entry.resume_pending:
                return False
            entry.resume_pending = False
            entry.resume_reason = None
            entry.last_resume_marked_at = None
        return self._update_entry(session_key, _apply)

    def prune_old_entries(self, max_age_days: int) -> int:
        """Drop routing entries idle (by ``updated_at``) for more than max_age_days; suspended
        entries and entries with active background processes are kept. Only the key -> session_id
        mapping is dropped (the transcript stays). ``max_age_days <= 0`` disables. Returns count."""
        if max_age_days is None or max_age_days <= 0:
            return 0
        cutoff = _now() - timedelta(days=max_age_days)
        with self._lock:
            self._ensure_loaded_locked()
            removed_keys = [
                key for key, entry in list(self._entries.items())
                if not entry.suspended
                # The callback is keyed by session_key, NOT session_id.
                and not self._has_active_processes_safe(entry.session_key, context="prune")
                and entry.updated_at < cutoff
            ]
            for key in removed_keys:
                self._entries.pop(key, None)
            if removed_keys:
                self._save()
        if removed_keys:
            logger.info("SessionStore pruned %d entries older than %d days",
                        len(removed_keys), max_age_days)
        return len(removed_keys)

    def suspend_recently_active(self, max_age_seconds: int = 120) -> int:
        """Mark sessions active within *max_age_seconds* as ``resume_pending`` after a crash/fast
        restart (already-pending and suspended entries are skipped). Returns the number marked.

        Called on gateway startup after a crash or fast restart to preserve in-flight sessions instead of
        destroying their conversation history (#7536). Only marks sessions updated within *max_age_seconds*
        to avoid touching long-idle sessions. Sets ``resume_pending=True`` so the next incoming message on
        the same session_key auto-resumes from the existing transcript.
        """
        cutoff = _now() - timedelta(seconds=max_age_seconds)

        def _mark(entry: SessionEntry) -> bool:
            if entry.session_id in self._typed_event_recovery_session_ids_locked():
                return False
            if entry.resume_pending or entry.suspended or entry.updated_at < cutoff:
                return False
            entry.resume_pending = True
            entry.resume_reason = "restart_interrupted"
            entry.last_resume_marked_at = _now()
            return True
        return self._update_all_entries_locked(_mark)


    def _typed_event_recovery_state_locked(self, session_id: str) -> str:
        """Return ``none``, ``owned`` or ``conflict`` for a physical session.

        Presence is fail-closed: malformed data, or a marker naming another
        physical session, suppresses generic recovery until a real user turn
        replaces it under the session lease.
        """
        found = False
        for candidate in self._entries.values():
            if candidate.session_id != session_id:
                continue
            if TYPED_EVENT_RECOVERY_METADATA_KEY not in candidate.metadata:
                continue
            found = True
            marker = candidate.metadata.get(TYPED_EVENT_RECOVERY_METADATA_KEY)
            if not isinstance(marker, dict):
                return "conflict"
            if marker.get("version") != _TYPED_EVENT_RECOVERY_VERSION:
                return "conflict"
            if marker.get("session_id") != session_id:
                return "conflict"
            owner_id = marker.get("owner_id")
            if not isinstance(owner_id, str) or not owner_id:
                return "conflict"
        return "owned" if found else "none"


    def _typed_event_recovery_session_ids_locked(self) -> set[str]:
        """Return every physical session with a present recovery marker."""
        return {
            entry.session_id
            for entry in self._entries.values()
            if TYPED_EVENT_RECOVERY_METADATA_KEY in entry.metadata
        }


    def typed_event_recovery_owner_state(
        self,
        session_key: str,
        expected_session_id: str,
    ) -> str:
        """Read the fail-closed typed recovery state for an exact route."""
        with self._lock:
            self._ensure_loaded_locked()
            entry = self._entries.get(session_key)
            if entry is None or entry.session_id != expected_session_id:
                return "conflict"
            return self._typed_event_recovery_state_locked(expected_session_id)


    def mark_typed_event_recovery_owner(
        self,
        session_key: str,
        expected_session_id: str,
        owner_id: str,
    ) -> bool:
        """Persist trusted recovery ownership before a typed turn reads history.

        A later authorized typed event may replace a valid owner after it has
        acquired the same physical-session lease.  Malformed ownership,
        suspension, or an older generic resume obligation fails closed.
        """
        if not isinstance(owner_id, str) or not owner_id:
            return False
        with self._lock:
            self._ensure_loaded_locked()
            entry = self._entries.get(session_key)
            if entry is None or entry.session_id != expected_session_id:
                return False
            aliases = [
                candidate
                for candidate in self._entries.values()
                if candidate.session_id == expected_session_id
            ]
            if any(candidate.suspended or candidate.resume_pending for candidate in aliases):
                return False
            if self._typed_event_recovery_state_locked(expected_session_id) == "conflict":
                return False

            marker = {
                "version": _TYPED_EVENT_RECOVERY_VERSION,
                "session_id": expected_session_id,
                "owner_id": owner_id,
            }
            data, generation = self._snapshot_routing_locked()
            for candidate in aliases:
                candidate_data = data[candidate.session_key]
                metadata = dict(candidate_data.get("metadata") or {})
                metadata[TYPED_EVENT_RECOVERY_METADATA_KEY] = marker
                candidate_data["metadata"] = metadata

            # Publish in-memory state only after the complete alias snapshot is
            # durable.  A failed write cannot leak a marker through a later save.
            self._persist_routing_data(data, generation)
            for candidate in aliases:
                candidate.metadata[TYPED_EVENT_RECOVERY_METADATA_KEY] = dict(marker)
            return True
