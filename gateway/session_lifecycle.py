"""SessionStore reset/expiry policy and crash-recovery markers (idle/daily reset, expiry
finalization, active-turn tokens, resume_pending, suspension, pruning) plus the shared clock/id
helpers. Mixin split out of ``gateway/session.py``; bound onto ``SessionStore`` via the MRO."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gateway.session import SessionEntry, SessionSource

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.session")


def _now() -> datetime:
    """Return the current local time."""
    return datetime.now()


def _new_session_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


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
    """Auto-continue freshness window in seconds (one source of truth for the resume scheduler and
    the routing-time zombie gate); env var, default when unset/malformed; non-positive disables."""
    raw = os.environ.get("HERMES_AUTO_CONTINUE_FRESHNESS")
    try:
        return float(raw) if raw else float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    except (TypeError, ValueError):
        return float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)


class SessionLifecycleMixin:
    """SessionStore reset/expiry policy and crash-recovery markers."""

    def set_expiry_finalized(
        self, entry: SessionEntry, *, clear_model_override: bool = True
    ) -> None:
        """Mark a session entry expiry-finalized in memory, sessions.json, AND state.db (single
        write-path for the expiry watcher). ``clear_model_override=False`` = flag only."""
        with self._lock:
            entry.expiry_finalized = True
            if clear_model_override:
                # Finalization is a conversation boundary: a later message must not rehydrate it.
                entry.model_override = None
            self._save()
        # Background caller never entered ``_profile_runtime_scope``: resolve the store by key.
        # The expiry watcher calls this from a background task that never entered
        # ``_profile_runtime_scope``, so resolve the store from the key rather than from the ambient scope
        # (#66887).
        _db = self._db_for_key(entry.session_key)
        if not _db:
            return
        setter = getattr(_db, "set_expiry_finalized", None)
        if callable(setter):
            try:
                setter(entry.session_id, True)
            except Exception as exc:
                logger.debug("Session DB expiry_finalized write failed for %s: %s", entry.session_id, exc)
        try:
            # Without a durable ``session_reset`` end_reason, later agent cleanup ends the row as
            # ``agent_close``, which stale-route recovery treats as resumable.
            _db.promote_to_session_reset(entry.session_id)
        except Exception as exc:
            logger.debug("Session DB promote_to_session_reset failed for %s: %s", entry.session_id, exc)

    @staticmethod
    def _policy_reset_reason(policy, updated_at: datetime) -> Optional[str]:
        """Return "idle"/"daily" when *updated_at* is overdue under *policy*, else None."""
        if policy.mode == "none":
            return None
        now = _now()
        if policy.mode in {"idle", "both"} and now > updated_at + timedelta(minutes=policy.idle_minutes):
            return "idle"
        if policy.mode in {"daily", "both"}:
            today_reset = now.replace(hour=policy.at_hour, minute=0, second=0, microsecond=0)
            if now.hour < policy.at_hour:
                today_reset -= timedelta(days=1)
            if updated_at < today_reset:
                return "daily"
        return None

    def _is_session_expired(self, entry: SessionEntry) -> bool:
        """Whether the reset policy has expired *entry* (expiry watcher); sessions with active
        background processes never expire."""
        if self._has_active_processes_safe(entry.session_key, context="expiry"):
            logger.debug("Session %s not expired — active background processes", entry.session_key)
            return False
        policy = self.config.get_reset_policy(platform=entry.platform, session_type=entry.chat_type)
        return self._policy_reset_reason(policy, entry.updated_at) is not None

    def is_session_finalizable(self, entry: SessionEntry) -> bool:
        """True if the expiry watcher will *ever* finalize this session; ``mode == "none"`` never
        expires, so the agent-cache sweep must reap its agent itself. Policy errors -> False."""
        try:
            policy = self.config.get_reset_policy(platform=entry.platform, session_type=entry.chat_type)
            return policy.mode != "none"
        except Exception:
            return False

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

    def _should_reset(self, entry: SessionEntry, source: SessionSource) -> Optional[str]:
        """Reset reason ("idle"/"daily") if policy says reset, else None; sessions with active
        background processes are never reset."""
        session_key = self._generate_session_key(source)
        if self._has_active_processes_safe(session_key, context="reset"):
            logger.debug("Session reset skipped for %s — active background processes", session_key)
            return None
        policy = self.config.get_reset_policy(platform=source.platform, session_type=source.chat_type)
        return self._policy_reset_reason(policy, entry.updated_at)

    def _route_reset_reason(
        self, entry: SessionEntry, source: SessionSource, now: datetime
    ) -> Optional[str]:
        """Reset decision for an existing route (no lock; DB/config I/O). ``suspended`` always
        resets; otherwise the reset policy decides, and a still-pending resume marker is also
        freshness-gated — but ``session_reset.mode: none`` (user opted out of ALL automatic
        resets) makes an expired marker fall through to a normal resume, never a silent fresh
        session."""
        if entry.suspended:
            return "suspended"
        reason = self._should_reset(entry, source)
        if reason or not entry.resume_pending:
            return reason
        policy = self.config.get_reset_policy(platform=source.platform, session_type=source.chat_type)
        if policy.mode == "none":
            return None
        window = auto_continue_freshness_window()
        ref_time = entry.last_resume_marked_at or entry.updated_at
        if window > 0 and (now - ref_time).total_seconds() > window:
            return "resume_pending_expired"
        return None

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

    def mark_turn_active(self, session_key: str) -> Optional[str]:
        """Persist exact ownership of the running agent turn; returns the opaque token for
        :meth:`clear_turn_active`. Re-marking replaces the previous token so a stale asynchronous
        unwind cannot clear a newer turn."""
        token = uuid.uuid4().hex
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return None
            self._set_turn_marker_locked(session_key, entry, token, _now())
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
            if entry.resume_pending or entry.suspended or entry.updated_at < cutoff:
                return False
            entry.resume_pending = True
            entry.resume_reason = "restart_interrupted"
            entry.last_resume_marked_at = _now()
            return True
        return self._update_all_entries_locked(_mark)
