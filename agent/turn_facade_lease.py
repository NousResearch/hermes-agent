"""Durable cross-process session turn lease for ``TurnFacadeMixin.run_conversation``.

One process at a time may load -> run -> flush a session shared through state.db (Desktop, CLI
resume, gateway, background delivery). ``admit_durable_turn_lease`` acquires the row lease (or
returns the early result the façade must hand back); ``DurableTurnLease`` owns the periodic
refresher, the turn-liveness watchdog wiring, and the lease-loss / stall interrupt plumbing. Both
timers run via the shared scheduler (``agent/periodic_scheduler.py``; timer thread orders,
bodies run on per-handle workers), not per-turn threads.
"""
import logging
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")

# ``tool_reason`` for a lost session turn lease: attributes the stop to the lease, not the user (#112647).
_REASON_LEASE_LOST = "session turn lease lost"

LEASE_TTL_SECONDS = 300.0
LEASE_WAIT_SECONDS = 1800.0


class DurableTurnLease:
    """An admitted session turn lease plus the periodic timers that keep it alive and watch the turn.

    ``stop`` is shared by the refresher and the liveness watchdog; ``turn_active`` gates every
    interrupt so a late refresher miss can never hard-interrupt the NEXT turn. Both are read and
    written only under ``_lock``.
    """

    def __init__(self, agent, db, session_id: str, holder: str) -> None:
        self.agent = agent
        self.db = db
        self.session_id = session_id  # id at admission; release always targets this row
        self.holder = holder
        self.stop = threading.Event()
        self.refresh_interval = float(getattr(agent, "_session_turn_lease_refresh_interval", 60.0))
        self._lock = threading.Lock()
        self.turn_active = False
        self.interrupt_message: Optional[str] = None
        self.watchdog = None  # TurnLivenessWatchdog when configured
        self.timer_handles: list = []  # periodic_scheduler handles, cancelled in join_threads

    def _current_session_id(self) -> str:
        return getattr(self.agent, "session_id", None) or self.session_id

    def build_threads(self) -> None:
        """Create (not schedule) the liveness watchdog when configured: lease renewal is NOT
        evidence of progress; a silently stalled turn would renew forever."""
        try:
            from hermes_cli.config import load_config_readonly

            liveness_config = load_config_readonly() or {}
        except Exception:
            liveness_config = {}
        from agent import turn_liveness

        timeout_s, poll_s = turn_liveness.resolve_turn_liveness_settings(liveness_config)
        if timeout_s is not None:
            self.watchdog = turn_liveness.TurnLivenessWatchdog(
                self.agent, session_id=self._current_session_id(), timeout_s=timeout_s,
                poll_s=poll_s, stop_event=self.stop,
                activity_lock=self.agent._liveness_activity_lock(),
                is_turn_active=self.is_turn_active, commit_abort=self.commit_liveness_abort,
                deactivate_turn=self.stop_refresher,
            )

    def start(self) -> None:
        with self._lock:
            self.turn_active = True
        # Stamp the activity clock at turn entry: `_last_activity_ts` persists across turns, so
        # without this the watchdog would measure idle from the PREVIOUS turn and abort a fresh one.
        self.agent._touch_activity("starting new turn")
        from agent.periodic_scheduler import schedule

        self.timer_handles.append(schedule(self.refresh_tick, self.refresh_interval))
        if self.watchdog is not None:
            self.timer_handles.append(self.watchdog.schedule())

    def stop_refresher(self) -> None:
        """Stop renewal and deactivate the turn. Also the watchdog's deactivate callback: a wedge the
        hard interrupt cannot unwind must not keep the lease alive forever; TTL expiry lets
        stale-turn cleanup reclaim the row."""
        with self._lock:
            self.turn_active = False
            self.stop.set()

    deactivate_after_liveness_abort = stop_refresher

    def join_threads(self, timeout: float = 1.0) -> None:
        """Cancel both timers; ``wait=timeout`` mirrors the old ``thread.join(timeout)`` so an
        in-flight tick finishes before ``clear_interrupt`` runs."""
        for handle in self.timer_handles:
            handle.cancel(wait=timeout)

    def release(self) -> None:
        """Release the row and drop the agent's holder attrs (only if they still name this lease)."""
        agent = self.agent
        try:
            self.db.release_session_turn_lease(self.session_id, self.holder)
        except Exception:
            logger.error("Failed to release session turn lease: %s", self.session_id, exc_info=True)
        if getattr(agent, "_active_session_turn_lease_holder", None) == self.holder:
            agent._active_session_turn_lease_holder = None
            agent._active_session_turn_lease_ttl_seconds = None

    def is_turn_active(self) -> bool:
        with self._lock:
            return self.turn_active

    def _interrupt_turn(self, message: str) -> None:
        """Lease-loss interrupts fire UNCONDITIONALLY (no generation claim): a lost lease means
        this process no longer owns the session. Only the watchdog's stalls can be spuriously stale."""
        with self._lock:
            if self.stop.is_set() or not self.turn_active:
                return
            self.interrupt_message = message
            try:
                self.agent.interrupt(message, hard_cancel=True, tool_reason=_REASON_LEASE_LOST)
            except Exception:
                self.agent._interrupt_requested = True
                self.agent._interrupt_message = message
                self.agent._tool_interrupt_reason = _REASON_LEASE_LOST

    def commit_liveness_abort(self, snapshot, message: str) -> bool:
        """Commit point for the watchdog's stall observation.

        Revalidates the observed ``(generation, timestamp)`` under the SAME lock ``_touch_activity``
        uses, so a turn that resumed while the stall was logged is never hard-cancelled; the
        revalidated generation is consumed by ``interrupt(require_generation=...)`` with the first
        publication in ONE critical section. If ``interrupt`` raises, the abort declines FAIL-CLOSED.
        Returns False when stale or already winding down."""
        agent = self.agent
        with agent._liveness_activity_lock():
            current_generation = getattr(agent, "_turn_liveness_activity_generation", 0)
            if (current_generation, getattr(agent, "_last_activity_ts", None)) != (
                snapshot.generation, snapshot.activity_ts
            ):
                return False
        with self._lock:
            if self.stop.is_set() or not self.turn_active:
                return False
        try:
            published = agent.interrupt(
                message, hard_cancel=True, tool_reason="turn liveness watchdog",
                require_generation=current_generation,
            )
        except Exception:
            logger.debug("Turn liveness abort interrupt raised; declining the abort", exc_info=True)
            published = False
        if published is False:
            # Claim went stale between revalidation and the hammer: real progress landed.
            return False
        with self._lock:
            self.interrupt_message = message
        return True

    def clear_interrupt(self) -> None:
        """Clear only the interrupt admitted by this lease's refresher/watchdog. Run AFTER join."""
        message = self.interrupt_message
        if not message:
            return
        agent = self.agent
        from tools.interrupt import set_interrupt as _set_interrupt

        with getattr(agent, "_pending_redirect_lock", None) or nullcontext():
            if getattr(agent, "_interrupt_message", None) != message:
                return
            agent._interrupt_requested = False
            agent._interrupt_message = None
            getattr(agent, "_hard_interrupt_requested", threading.Event()).clear()
            agent._interrupt_thread_signal_pending = False
            if agent._execution_thread_id is not None:
                _set_interrupt(False, agent._execution_thread_id)

    def refresh_tick(self):
        """One periodic renewal (every ``refresh_interval`` via the shared scheduler); a miss or
        error interrupts the turn. Returning False stops the timer.

        The holder-qualified UPDATE fences a late refresher from a successor lease. The façade's
        finally sets ``stop`` before releasing, so a holder-fenced miss observed after stop is not
        a loss."""
        if self.stop.is_set():
            return False
        try:
            if self.db.refresh_session_turn_lease(
                self._current_session_id(), self.holder, ttl_seconds=LEASE_TTL_SECONDS
            ):
                return None
            if self.stop.is_set():
                return False
            logger.error(
                "Lost session turn lease while turn is active: %s", self._current_session_id()
            )
            self._interrupt_turn("Session turn lease lost; stopping to protect the transcript.")
        except Exception:
            if self.stop.is_set():
                return False
            logger.warning(
                "Failed to refresh session turn lease: %s", self._current_session_id(), exc_info=True,
            )
            self._interrupt_turn(
                "Session turn lease could not be refreshed; stopping to protect the transcript."
            )
        return False


@dataclass
class TurnLeaseAdmission:
    """Outcome of ``admit_durable_turn_lease``: exactly one of ``lease`` / ``early_result`` may be set."""

    lease: Optional[DurableTurnLease] = None
    early_result: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None


def _durable_session_exists(db, session_id: str) -> bool:
    try:
        return db.get_session(session_id) is not None
    except Exception:
        # A locked / non-WAL read is not proof the row is absent; treating probe failure as "fresh"
        # ran fail-open at the exact contention point. Acquire, or fail closed.
        logger.warning(
            # Acquire (or fail closed if acquire itself cannot) rather than start load/run/flush
            # unsynchronized. get_session returns None — it does not raise — when the row is missing. See
            # #84234.
            "Could not check durable session before turn lease; "
            "will acquire rather than run without serialization",
            exc_info=True,
        )
        return True


def _committed_history_id(message) -> Optional[int]:
    if not isinstance(message, dict) or message.get("_db_persisted") is not True:
        return None
    row_id = message.get("_row_id")
    return row_id if type(row_id) is int and row_id > 0 else None


def _desktop_replay_projection(db, session_id):
    from agent.replay_cleanup import sanitize_replay_history
    from agent.turn_context import drop_stale_api_content

    native, _display = db.get_resume_conversations(session_id)
    native_by_id = {_committed_history_id(message): message for message in native}
    if None in native_by_id:
        raise ValueError("session_history_conflict: unanchored native projection")
    projected = sanitize_replay_history(native)
    rewritten = set()
    for message in projected:
        row_id = _committed_history_id(message)
        if row_id is not None and message != native_by_id[row_id]:
            # A stale sidecar must never override the native UNKNOWN-effect rewrite.
            drop_stale_api_content(message)
            rewritten.add(row_id)
    return native_by_id, projected, rewritten


def _native_recovery_rows(parent, rows):
    from agent.replay_cleanup import sanitize_replay_history

    # Native dangling-tail notices are synthetic, so have no DB identity. Validate
    # them against their ALREADY anchored call, using the native producer. Its fresh
    # timestamp is not provenance; all other generated fields must match exactly.
    generated = sanitize_replay_history([parent])[1:]
    clean = lambda message: {key: value for key, value in message.items() if key != "timestamp"}
    return len(rows) <= len(generated) and all(
        clean(row) == clean(expected) for row, expected in zip(rows, generated)
    )


def _hydrate_immediate_history(db, session_id: str, history):
    """Reconcile an anchored Desktop cache against its native sanitized model projection.

    Row identity authorizes reuse of richer cached objects; replay safety rewrites take
    precedence. Client-owned/marker-only histories are excluded by the admission caller.
    """
    native_by_id, projected, rewritten = _desktop_replay_projection(db, session_id)
    safe_by_id = {_committed_history_id(message): message for message in projected
                  if _committed_history_id(message) is not None}
    removed = set(native_by_id) - set(safe_by_id)
    cached_ids = [_committed_history_id(message) for message in history
                  if _committed_history_id(message) is not None]
    if len(set(cached_ids)) != len(cached_ids):
        raise ValueError("session_history_conflict: duplicate cached anchors")
    if any(row_id not in native_by_id for row_id in cached_ids):
        raise ValueError("session_history_conflict: removed or foreign cached anchors")
    if cached_ids != [row_id for row_id in native_by_id if row_id in set(cached_ids)]:
        raise ValueError("session_history_conflict: reordered cached anchors")
    # Only the canonical sanitizer may remove an anchored cached block here. DB rewind,
    # wrong-session rows and alternation-repair ambiguity still fail closed above.
    working = [message for message in history if _committed_history_id(message) not in removed]
    anchors = [(index, row_id) for index, message in enumerate(working)
               if (row_id := _committed_history_id(message)) is not None]
    if not anchors:
        raise ValueError("session_history_conflict: sanitization removed every cached anchor")
    cached = {row_id: working[index] for index, row_id in anchors}
    projected_ids = list(safe_by_id)
    if next(iter(cached)) != projected_ids[0]:
        raise ValueError("session_history_conflict: cached prefix is incomplete")

    unrepaired = db.get_messages_as_conversation(
        session_id, repair_alternation=False, include_row_ids=True
    )
    if [_committed_history_id(message) for message in unrepaired] != list(native_by_id):
        for row_id, message in cached.items():
            if row_id not in rewritten and message != safe_by_id[row_id]:
                raise ValueError("session_history_conflict: native replay repair changed anchors")
    for row_id in rewritten:
        if row_id in cached:
            safe = safe_by_id[row_id]
            current = cached[row_id]
            if (any(current.get(key) != safe.get(key)
                    for key in ("content", "effect_disposition", "tool_call_id"))
                    or "api_content" in current):
                cached[row_id] = safe


    extras = {}
    parent_id = None
    for message in projected:
        row_id = _committed_history_id(message)
        if row_id is not None:
            parent_id = row_id
        else:
            extras.setdefault(parent_id, []).append(message)
    for row_id, rows in extras.items():
        if row_id is None or not _native_recovery_rows(safe_by_id[row_id], rows):
            raise ValueError("session_history_conflict: unsupported native synthetic projection")

    prefix = working[:anchors[0][0]]
    suffix = []
    for offset, (index, row_id) in enumerate(anchors):
        stop = anchors[offset + 1][0] if offset + 1 < len(anchors) else len(working)
        following = working[index + 1:stop]
        if not following:
            continue
        notices = []
        for message in following:
            if message.get("role") == "tool" and _native_recovery_rows(safe_by_id[row_id], notices + [message]):
                notices.append(message)
            else:
                if message.get("role") == "tool" and safe_by_id[row_id].get("tool_calls"):
                    raise ValueError("session_history_conflict: ambiguous recovery result")
                break
        if notices:
            native_notices = extras.get(row_id, [])
            if native_notices and len(native_notices) != len(notices):
                raise ValueError("session_history_conflict: changed native recovery notices")
            extras[row_id] = notices
        remainder = following[len(notices):]
        if remainder:
            if offset + 1 < len(anchors):
                raise ValueError("session_history_conflict: interleaved caller-only history")
            suffix = remainder

    result = list(prefix)
    for row_id, message in safe_by_id.items():
        result.append(cached.get(row_id, message))
        result.extend(extras.get(row_id, []))
    result.extend(suffix)
    return history if len(result) == len(history) and all(a is b for a, b in zip(result, history)) else result


def admit_durable_turn_lease(
    agent, *, session_id: str, relay_turn_id: str, task_context: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]], gateway_system_event: Optional[Any] = None,
) -> TurnLeaseAdmission:
    """Acquire the session turn lease when the session is durable; build (not start) its threads.

    Mutates ``task_context["session_id"]`` and ``agent.session_id`` when the wait forced a resume-id
    reload. Returns an ``early_result`` (interrupted / timed out) instead of a lease when admission
    fails; the caller returns it verbatim."""
    db = getattr(agent, "_session_db", None)
    admission = TurnLeaseAdmission(conversation_history=conversation_history)
    if db is None or not session_id:
        return admission
    # A fresh session id has no durable transcript to race over, and callers may supply an
    # in-memory seed before the row exists — reloading would erase it. Check the concrete type:
    # MagicMock-style shims accept any attribute without the protocol.
    if (
        getattr(agent, "_persist_disabled", False)
        or not _durable_session_exists(db, session_id)
        or not callable(getattr(type(db), "acquire_session_turn_lease", None))
    ):
        return admission
    # Row proven to exist — suppress the redundant create attempt.
    agent._session_db_created = True
    holder = (
        f"pid={os.getpid()}:turn={relay_turn_id}:platform={task_context['platform'] or 'unknown'}"
    )
    waited = False

    def _on_wait(elapsed: float) -> None:
        nonlocal waited
        waited = True
        agent._emit_status(
            "⏳ Another Hermes process is using this session; "
            "waiting for it to finish before starting your turn..."
            if elapsed < 1.0 else
            f"⏳ Still waiting for the other Hermes process on this session ({int(elapsed)}s)..."
        )

    if not db.acquire_session_turn_lease(
        session_id, holder, ttl_seconds=LEASE_TTL_SECONDS, wait_seconds=LEASE_WAIT_SECONDS,
        on_wait=_on_wait, should_abort=lambda: getattr(agent, "_interrupt_requested", False),
    ):
        admission.early_result = _lease_not_acquired_result(agent, session_id, conversation_history)
        return admission

    # Assign only after admission so the finally cannot release a holder that never owned the
    # row; persist paths read the agent attr so a late flush is fenced in the same transaction.
    lease = DurableTurnLease(agent, db, session_id, holder)
    agent._active_session_turn_lease_holder = holder
    agent._active_session_turn_lease_ttl_seconds = LEASE_TTL_SECONDS
    try:
        if gateway_system_event is not None:
            from gateway.internal_events import GatewaySystemEvent
            if (not isinstance(gateway_system_event, GatewaySystemEvent)
                    or gateway_system_event.expected_session_id != session_id):
                raise ValueError("typed gateway event has an invalid physical session")
            # Process-local receipt caches and Desktop history can be stale. Check
            # committed admission rows only after owning the physical DB lease,
            # before staging the next event or entering the model loop.
            durable = db.get_messages_as_conversation(
                session_id, repair_alternation=False, include_row_ids=True
            )
            for message in durable:
                metadata = message.get("display_metadata") or {}
                if (message.get("role") == "developer" and isinstance(metadata, dict)
                        and metadata.get("plugin_id") == gateway_system_event.plugin_id
                        and metadata.get("event_id") == gateway_system_event.event_id):
                    raise ValueError("typed gateway event was already admitted to this session")
        if waited:
            agent._emit_status("Session is free; loading the latest transcript...")
            # The holder may have compressed/rotated the session while we waited: reload only
            # AFTER admission. Preserve the existing waited resume/compression policy.
            latest_session_id = db.resolve_resume_session_id(session_id)
            if latest_session_id:
                agent.session_id = latest_session_id
                task_context["session_id"] = latest_session_id
            reloaded = db.get_messages_as_conversation(
                agent.session_id, repair_alternation=True, include_row_ids=True
            )
            # A follow-up that aborted an earlier wait carries that turn's never-persisted input
            # only in memory (see carry_unadmitted_user_message); the reload would drop it.
            from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT
            reloaded.extend(
                m for m in (conversation_history or [])
                if isinstance(m, dict) and m.get(_PERSIST_AFTER_ADMISSION_INTERRUPT)
                and "_row_id" not in m
            )
            admission.conversation_history = reloaded
        elif (task_context["platform"] == "desktop"
              and conversation_history
              and any(_committed_history_id(message) is not None for message in conversation_history)
              and all(isinstance(message, dict) for message in conversation_history)
              and not any(message.get("_db_persisted") and _committed_history_id(message) is None
                          for message in conversation_history)
              and all(callable(getattr(type(db), name, None)) for name in (
                  "get_messages_as_conversation", "get_resume_conversations",
                  "resolve_resume_session_id", "_session_turn_lease_key",
              ))):
            # Only the native Desktop DB-owned cache opts in. Gateway/API/client-owned
            # histories keep their existing immediate-admission semantics.
            admission.conversation_history = _hydrate_immediate_history(
                db, session_id, conversation_history
            )
            # A compression tip can move before an uncontended acquisition, too. Ordinary
            # continuation/branch rows do not necessarily share this lease; refuse those here.
            latest_session_id = db.resolve_resume_session_id(session_id)
            if latest_session_id and latest_session_id != session_id:
                if db._session_turn_lease_key(latest_session_id) != db._session_turn_lease_key(session_id):
                    raise ValueError("session_history_conflict: resume target outside acquired lease")
                history = admission.conversation_history or []
                persisted = [i for i, message in enumerate(history) if message.get("_db_persisted")]
                # Native recovery notices belong to the old anchored call's context, even
                # though they have no persisted row. They must leave with it on adoption.
                context_indices = set(persisted)
                for index in persisted:
                    notices = []
                    for offset in range(index + 1, len(history)):
                        if offset in context_indices:
                            break
                        message = history[offset]
                        if message.get("role") != "tool" or not _native_recovery_rows(
                            history[index], notices + [message]
                        ):
                            break
                        notices.append(message)
                        context_indices.add(offset)
                if context_indices and any(index not in context_indices
                                           for index in range(min(context_indices), max(context_indices) + 1)):
                    raise ValueError("session_history_conflict: interleaved compression history")
                if history and not persisted:
                    raise ValueError("session_history_conflict: unanchored compression history")
                _, latest, _ = _desktop_replay_projection(db, latest_session_id)
                admission.conversation_history = (
                    history[:persisted[0]] + latest + history[max(context_indices) + 1:]
                    if persisted else latest
                )
                agent.session_id = latest_session_id
                task_context["session_id"] = latest_session_id
        lease.build_threads()
    except BaseException:
        # The façade never saw this lease; release here so an admitted row is not leaked.
        lease.release()
        raise
    admission.lease = lease
    return admission


def carry_unadmitted_user_message(
    early_result: Dict[str, Any], user_message: Any, persist_user_message: Any, *,
    timestamp: Optional[float], display_kind: Optional[str], display_metadata: Optional[Dict[str, Any]],
    platform_id: Optional[str],
) -> None:
    """A follow-up that interrupted the lease wait must not consume the accepted input: append it to
    the early result's history so the follow-up turn sees it and persists it (the flush honours
    ``_PERSIST_AFTER_ADMISSION_INTERRUPT`` because this turn never owned the lease). A hard stop
    (``/stop``) cancels the input instead."""
    hard_interrupted = early_result.pop("_hard_interrupted", False)
    if hard_interrupted or not early_result.get("interrupted") or user_message in (None, ""):
        return
    from agent.message_metadata import append_message
    from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT

    durable_content = user_message
    if persist_user_message is not None and (
        not isinstance(user_message, list) or isinstance(persist_user_message, list)
    ):
        durable_content = persist_user_message
    deferred_user: Dict[str, Any] = {
        "role": "user", "content": durable_content, _PERSIST_AFTER_ADMISSION_INTERRUPT: True,
    }
    if isinstance(user_message, str) and user_message != durable_content:
        deferred_user["api_content"] = user_message
    if display_kind:
        deferred_user["display_kind"] = display_kind
    if display_metadata:
        deferred_user["display_metadata"] = display_metadata
    if platform_id is not None:
        deferred_user["platform_message_id"] = platform_id
    append_message(early_result["messages"], deferred_user, timestamp=timestamp)


def _lease_not_acquired_result(agent, session_id: str, conversation_history) -> Dict[str, Any]:
    base = {"messages": list(conversation_history or []), "api_calls": 0, "completed": False}
    if getattr(agent, "_interrupt_requested", False):
        logger.info("session turn lease wait aborted by interrupt: %s", session_id)
        hard_event = getattr(agent, "_hard_interrupt_requested", None)
        hard_interrupted = bool(
            callable(getattr(hard_event, "is_set", None)) and hard_event.is_set()
        )
        result = {
            "final_response": (
                "Stopped waiting for another Hermes process on this session. "
                "Your message was not processed."
            ),
            **base,
            "interrupted": True,
        }
        if hard_interrupted:
            result["_hard_interrupted"] = True
        if getattr(agent, "_interrupt_message", None):
            result["interrupt_message"] = agent._interrupt_message
        # The finalizer never runs on this early return; clear so a cached agent doesn't
        # fail-close the next turn.
        try:
            agent.clear_interrupt()
        except Exception:
            agent._interrupt_requested = False
            agent._interrupt_message = None
        return result
    # Fail closed like gateway TurnLeaseTimeoutError: surface a resend notice, not a bare TimeoutError.
    timeout_msg = (
        "⏳ Another Hermes process kept this session busy too long. Your message was not "
        "processed - wait for the other process to finish, then send it again."
    )
    logger.error("session turn lease wait timed out for %s", session_id)
    try:
        agent._emit_warning(timeout_msg)
    except Exception:
        logger.debug("Failed to emit session turn lease timeout warning", exc_info=True)
    # Stamped so Desktop/TUI show "session busy, send again" instead of code="unknown".
    return {
        "final_response": timeout_msg,
        **base,
        "failed": True,
        "error": f"session_turn_lease_timeout:{session_id}",
        "failure_reason": "session_busy",
        "failure_retryable": True,
    }
