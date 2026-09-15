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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")

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
                self.agent.interrupt(message, hard_cancel=True)
            except Exception:
                self.agent._interrupt_requested = True
                self.agent._interrupt_message = message

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
                message, hard_cancel=True, require_generation=current_generation
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
    """Outcome of ``admit_durable_turn_lease``: exactly one of ``lease`` / ``early_result`` may be set.

    ``drained_delivery_ids`` names the spooled inbound deliveries this turn took ownership of, in
    drain order, so the caller can report what the turn absorbed.
    """

    lease: Optional[DurableTurnLease] = None
    early_result: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    drained_delivery_ids: List[str] = field(default_factory=list)


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


def admit_durable_turn_lease(
    agent, *, session_id: str, relay_turn_id: str, task_context: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]],
    inbound_delivery: Optional[Dict[str, Any]] = None,
) -> TurnLeaseAdmission:
    """Acquire the session turn lease when the session is durable; build (not start) its threads.

    Mutates ``task_context["session_id"]`` and ``agent.session_id`` when the wait forced a resume-id
    reload. Returns an ``early_result`` (interrupted / timed out) instead of a lease when admission
    fails; the caller returns it verbatim. ``inbound_delivery`` — a mapping with ``message`` and
    optional ``author`` / ``delivery_id`` — marks a turn started for a message that arrived from
    outside this process: losing the lease wait spools that message onto the CONVERSATION instead of
    dropping it, and an admitted turn drains whatever is already spooled for the conversation as its
    first user message (see ``agent/turn_facade_lease`` + ``tools/bot_live_delivery``)."""
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
        admission.early_result = _lease_admission_result(
            agent, db, session_id, conversation_history, inbound_delivery,
        )
        return admission

    # Assign only after admission so the finally cannot release a holder that never owned the
    # row; persist paths read the agent attr so a late flush is fenced in the same transaction.
    lease = DurableTurnLease(agent, db, session_id, holder)
    agent._active_session_turn_lease_holder = holder
    agent._active_session_turn_lease_ttl_seconds = LEASE_TTL_SECONDS
    try:
        if waited:
            agent._emit_status("Session is free; loading the latest transcript...")
            # The holder may have compressed/rotated the session while we waited: reload only
            # AFTER admission; an immediate acquisition skips this (needless prompt-cache miss).
            latest_session_id = db.resolve_resume_session_id(session_id)
            if latest_session_id:
                agent.session_id = latest_session_id
                task_context["session_id"] = latest_session_id
            admission.conversation_history = db.get_messages_as_conversation(
                agent.session_id, repair_alternation=True, include_row_ids=True
            )
        lease.build_threads()
        # Mail that lost an earlier wait is owned by this turn: drained AFTER admission/reload so it
        # lands on the latest transcript, and never able to fail the turn (see _drain_lease_queue).
        admission.conversation_history, admission.drained_delivery_ids = _drain_lease_queue(
            session_id, admission.conversation_history,
        )
    except BaseException:
        # The façade never saw this lease; release here so an admitted row is not leaked.
        lease.release()
        raise
    admission.lease = lease
    return admission


def _lease_queue_home():
    """Profile home whose delivery store holds this conversation's spooled inbound mail.

    Resolved from the process ``HERMES_HOME`` (every profile-scoped process sets it) so the queue
    lands in the same store ``deliver_to_live_owner`` uses for that profile.
    """
    from hermes_constants import get_process_hermes_home

    return get_process_hermes_home()


def _lease_queue_dir(home) -> str:
    from tools.bot_live_delivery import DELIVERY_DIR_NAME

    return os.path.join(str(home), "runtime", DELIVERY_DIR_NAME)


def _lease_waiter_observation(db, session_id: str) -> Dict[str, Any]:
    """Holder / queue depth for the admission receipt.

    Each accessor is probed with its own default, so a store that predates the waiter queue reports
    nothing rather than breaking an admission.
    """
    def _probe(name: str, default):
        probe = getattr(db, name, None)
        if not callable(probe):
            return default
        try:
            return probe(session_id)
        except Exception:
            logger.debug("Turn lease observation failed: %s", name, exc_info=True)
            return default

    return {
        "holder": _probe("current_session_turn_lease_holder", None),
        "waiter_age_s": _probe("oldest_session_turn_waiter_age", None),
        "queue_depth": _probe("session_turn_waiter_count", 0),
    }


def _lease_admission_result(
    agent, db, session_id: str, conversation_history, inbound_delivery,
) -> Dict[str, Any]:
    """Early result for a turn that could not take the conversation's lease.

    A human turn keeps the fail-closed timeout: nothing is spooled on its behalf, so the caller must
    resend. An inbound delivery is spooled onto the CONVERSATION and the result reports the queue
    admission instead, because that message has an owner waiting on it and no retry of its own.
    """
    message = ""
    if inbound_delivery and not getattr(agent, "_interrupt_requested", False):
        message = str(inbound_delivery.get("message") or "").strip()
    if not message:
        return _lease_not_acquired_result(agent, session_id, conversation_history)
    observation = _lease_waiter_observation(db, session_id)
    try:
        from tools.bot_live_delivery import enqueue_lease_delivery

        record = enqueue_lease_delivery(
            _lease_queue_home(), conversation_id=session_id, message=message,
            holder=str(observation.get("holder") or ""),
            author=inbound_delivery.get("author"),
            delivery_id=inbound_delivery.get("delivery_id"),
        )
    except Exception:
        # Fail closed: without a durable spool the message would vanish, so report the timeout.
        logger.warning(
            "Could not queue the inbound delivery for %s; reporting the lease timeout instead",
            session_id, exc_info=True,
        )
        return _lease_not_acquired_result(agent, session_id, conversation_history)
    notice = (
        "Another Hermes process is using this session. Your message was queued and will be the "
        "first thing the next turn on this session reads."
    )
    try:
        agent._emit_status("⏳ " + notice)
    except Exception:
        logger.debug("Failed to emit queued-delivery notice", exc_info=True)
    return {
        "final_response": notice,
        "messages": list(conversation_history or []),
        "api_calls": 0,
        "completed": False,
        "status": "queued",
        "queued": True,
        "delivery_id": record.get("delivery_id"),
        **observation,
    }


def _drain_lease_queue(session_id: str, conversation_history):
    """Move this conversation's spooled deliveries into the transcript of the turn that owns it.

    Returns ``(history, delivery_ids)``. Records are claimed one at a time and settled as
    ``drained_into_turn`` once the body is in the transcript, so each message is run at most once. A
    delivery that arrives while a user message is already pending JOINS that message: strict role
    alternation is never broken to inject mail. Any unexpected failure leaves the transcript
    untouched — a turn admitted normally must not die because of what is in the spool.
    """
    history = list(conversation_history or [])
    if not session_id:
        return history, []
    try:
        home = _lease_queue_home()
        if not os.path.isdir(_lease_queue_dir(home)):
            return history, []
        from tools.bot_live_delivery import claim_lease_delivery, complete_lease_delivery

        bodies: List[str] = []
        drained: List[str] = []
        while True:
            record = claim_lease_delivery(home, conversation_id=session_id)
            if record is None:
                break
            delivery_id = str(record.get("delivery_id") or "")
            body = str(record.get("message") or "").strip()
            if body:
                bodies.append(body)
            if delivery_id:
                drained.append(delivery_id)
            try:
                complete_lease_delivery(
                    home, delivery_id, status="settled", reason="drained_into_turn",
                )
            except Exception:
                # Stays claimed: inspectable, and never re-run by a later turn.
                logger.warning(
                    "Queued delivery %s drained but not settled", delivery_id, exc_info=True,
                )
        if not bodies:
            return history, drained
        text = "\n\n".join(bodies)
        if history and history[-1].get("role") == "user":
            merged = dict(history[-1])
            existing = merged.get("content")
            merged["content"] = (
                f"{existing}\n\n{text}" if isinstance(existing, str) and existing else text
            )
            history[-1] = merged
        else:
            history.append({"role": "user", "content": text})
        return history, drained
    except Exception:
        logger.warning("Could not drain queued deliveries for %s", session_id, exc_info=True)
        return list(conversation_history or []), []


def _lease_not_acquired_result(agent, session_id: str, conversation_history) -> Dict[str, Any]:
    base = {"messages": list(conversation_history or []), "api_calls": 0, "completed": False}
    if getattr(agent, "_interrupt_requested", False):
        logger.info("session turn lease wait aborted by interrupt: %s", session_id)
        result = {
            "final_response": (
                "Stopped waiting for another Hermes process on this session. "
                "Your message was not processed."
            ),
            **base,
            "interrupted": True,
        }
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
    return {
        "final_response": timeout_msg,
        **base,
        "failed": True,
        "error": f"session_turn_lease_timeout:{session_id}",
    }
