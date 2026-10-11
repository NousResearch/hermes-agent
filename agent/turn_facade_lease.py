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
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hermes_state_pidns import holder_namespace_token
from agent.session_identity import check_expected_turn_identity, conditional_turn_pending, consume_expected_turn_identity, identity_transaction

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")

# ``tool_reason`` for a lost session turn lease: attributes the stop to the lease, not the user (#112647).
_REASON_LEASE_LOST = "session turn lease lost"

LEASE_TTL_SECONDS = 300.0
LEASE_WAIT_SECONDS = 1800.0
# SessionDB's default write patience: a renewal waits this long for the write lock, never past the authority.
_REFRESH_WRITE_PATIENCE_S = 20.0
# Headroom before expiry for one BEGIN IMMEDIATE blocked in the writer's 1s busy handler, plus jitter.
_REFRESH_EXPIRY_MARGIN_S = 2.0


class DurableTurnLease:
    """An admitted session turn lease plus the periodic timers that keep it alive and watch the turn.

    ``stop`` is shared by the refresher and the liveness watchdog; ``turn_active`` gates every
    interrupt so a late refresher miss can never hard-interrupt the NEXT turn. Both are read and
    written only under ``_lock``.
    """

    def __init__(self, agent, db, session_id: str, holder: str, *, expires_at: Optional[float] = None) -> None:
        self.agent = agent
        self.db = db
        self.session_id = session_id  # id at admission; release always targets this row
        self.holder = holder
        # Wall clock, same as the row's expires_at, never later than it: the row's committed expiry at
        # admission, then a timestamp taken BEFORE each successful renewal (the store stamps expiry
        # before its lock wait). Only a successful renewal moves it.
        self._authority_deadline = time.time() + LEASE_TTL_SECONDS if expires_at is None else expires_at
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
        from hermes_cli.observability.shared_metrics_process import arm_turn
        arm_turn(self.agent)
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

    def _stop_on_exhausted_authority(self) -> bool:
        """State.db stayed locked until no renewal can land before the row expires: stop now."""
        logger.warning("Session turn lease refresh stayed locked through its lifetime: %s", self._current_session_id())
        self._interrupt_turn("Session turn lease could not be refreshed; stopping to protect the transcript.")
        return False

    def refresh_tick(self):
        """One periodic renewal (every ``refresh_interval`` via the shared scheduler). A real
        miss (rowcount 0) or a non-lock error interrupts the turn. A SQLite lock is a missed
        tick only while the next attempt still lands before ``_authority_deadline``. Returning
        False stops the timer.

        Acquisition reclaims a row once ``expires_at <= now``, even if the old process is
        alive. Holder-qualified release does not block that path, so a run of lock errors
        must not keep this turn active up to and past the deadline. A successful renewal
        is the only thing that moves the deadline.

        The holder-qualified UPDATE fences a late refresher from a successor lease. The façade's
        finally sets ``stop`` before releasing, so a holder-fenced miss observed after stop is not
        a loss."""
        if self.stop.is_set():
            return False
        started = time.time()
        # The whole renewal must finish before the authority runs out: a renewal still waiting on the
        # write lock past expiry keeps this turn running while a successor reclaims the row. The
        # margin covers one BEGIN IMMEDIATE that blocks in SQLite's busy handler past the patience.
        patience = self._authority_deadline - _REFRESH_EXPIRY_MARGIN_S - started
        if patience <= 0:
            return self._stop_on_exhausted_authority()
        try:
            if self.db.refresh_session_turn_lease(
                self._current_session_id(), self.holder, ttl_seconds=LEASE_TTL_SECONDS,
                patience_s=min(_REFRESH_WRITE_PATIENCE_S, patience),
            ):
                self._authority_deadline = started + LEASE_TTL_SECONDS
                return None
            if self.stop.is_set():
                return False
            logger.error(
                "Lost session turn lease while turn is active: %s", self._current_session_id()
            )
            self._interrupt_turn("Session turn lease lost; stopping to protect the transcript.")
        except Exception as exc:
            if self.stop.is_set():
                return False
            from hermes_state_errors import is_sqlite_lock_error

            if is_sqlite_lock_error(exc):
                # The scheduler will not try again until one interval from now. If that attempt
                # could not finish before the row expiry, stop before a successor can claim it.
                if time.time() + self.refresh_interval + _REFRESH_EXPIRY_MARGIN_S >= self._authority_deadline:
                    return self._stop_on_exhausted_authority()
                logger.warning(
                    "Session turn lease refresh hit a SQLite lock; will retry: %s",
                    self._current_session_id(),
                )
                return None
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
    early_result: Optional[dict[str, Any]] = None
    conversation_history: Optional[list[dict[str, Any]]] = None


def _committed_lease_expiry(db, session_id: str, holder: str, floor: float) -> float:
    """The admitted row's committed ``expires_at``; when unreadable, ``floor + TTL`` (taken before
    acquisition, so never later than the real expiry). Overstating it lets a successor reclaim the
    row while this turn still believes it holds the lease."""
    reader = getattr(db, "session_turn_lease_expires_at", None)
    if callable(reader):
        try:
            committed = reader(session_id, holder)
            if isinstance(committed, (int, float)):
                return float(committed)
        except Exception:
            logger.debug("Could not read the admitted turn lease expiry; using the pre-acquire floor", exc_info=True)
    return floor + LEASE_TTL_SECONDS


def _durable_session_exists(db, session_id: str) -> Optional[bool]:
    """True / False when the row read answered; None when it failed and the state is unknown."""
    try:
        return db.get_session(session_id) is not None
    except Exception:
        # A locked / non-WAL read proves neither presence nor absence: get_session returns None —
        # it does not raise — when the row is missing. See #84234.
        logger.warning(
            "Could not check durable session after turn lease admission; row state is unknown",
            exc_info=True,
        )
        return None


@identity_transaction
def _reload_admitted_turn(agent, db, session_id, task_context, admission, conversation_history, reload_needed, announced):
    # Read the row only now: the previous holder may have created or deleted it while this
    # turn waited, so an answer from before admission can be stale either way.
    check_expected_turn_identity(agent, db)
    if conditional_turn_pending():
        # Even an uncontended acquisition cannot adopt a pre-existing descendant.
        # Compare under the same engine transaction, after durable admission.
        check_expected_turn_identity(agent, db, db.resolve_resume_session_id(session_id, strict=True))
    durable = _durable_session_exists(db, session_id)
    if durable:
        # Row proven to exist — suppress the redundant create attempt. A missing row leaves
        # the flag alone: the flush heals a row deleted under a live agent (#123583). So does
        # an unknown one: the create is an upsert that never overwrites an existing row.
        agent._session_db_created = True
    # Reload only a transcript that may exist: callers may seed a fresh id in memory before its
    # row is written, and reloading an absent row would erase that seed. An unknown row still
    # reads the transcript after a wait and adopts it only if it returns rows; that read
    # raising ends the turn.
    if reload_needed and durable is not False:
        if announced:
            agent._emit_status("Session is free; loading the latest transcript...")
        # The holder may have compressed/rotated the session while we waited: reload only
        # AFTER admission; an immediate acquisition skips this (needless prompt-cache miss).
        latest_session_id = db.resolve_resume_session_id(session_id)
        check_expected_turn_identity(agent, db, latest_session_id)
        if latest_session_id:
            agent.session_id = latest_session_id
            task_context["session_id"] = latest_session_id
        reloaded = db.get_messages_as_conversation(
            agent.session_id, repair_alternation=True, include_row_ids=True
        )
        # Decide on the stored rows alone: an unknown row that reloads nothing keeps the
        # caller's history, which already holds any carried input below.
        if durable or reloaded:
            # A follow-up that aborted an earlier wait carries that turn's never-persisted
            # input only in memory (see carry_unadmitted_user_message); the reload drops it.
            from agent.session_persistence import _PERSIST_AFTER_ADMISSION_INTERRUPT
            reloaded.extend(
                m for m in (conversation_history or [])
                if isinstance(m, dict) and m.get(_PERSIST_AFTER_ADMISSION_INTERRUPT)
                and "_row_id" not in m
            )
            admission.conversation_history = reloaded


def admit_durable_turn_lease(
    agent, *, session_id: str, relay_turn_id: str, task_context: dict[str, Any],
    conversation_history: Optional[list[dict[str, Any]]],
) -> TurnLeaseAdmission:
    """Acquire the session turn lease (the row need not exist yet); build (not start) its threads.

    Mutates ``task_context["session_id"]`` and ``agent.session_id`` when the wait forced a resume-id
    reload. Returns an ``early_result`` (interrupted / timed out) instead of a lease when admission
    fails; the caller returns it verbatim."""
    db = getattr(agent, "_session_db", None)
    admission = TurnLeaseAdmission(conversation_history=conversation_history)
    if db is None or not session_id:
        return admission
    # Check the concrete type: MagicMock-style shims accept any attribute without the protocol.
    if (
        getattr(agent, "_persist_disabled", False)
        or not callable(getattr(type(db), "acquire_session_turn_lease", None))
    ):
        return admission
    # A session id without a row still takes the lease: client-addressed ids (API server
    # X-Hermes-Session-Id, /v1/runs session_id, fingerprint-derived chat ids) are not
    # process-unique, and the first turn creates the row mid-turn, so a second writer would
    # otherwise find the row, take an unheld lease and interleave its turn into this one.
    # pidns stamp: see hermes_state_pidns.
    holder = (
        f"pid={os.getpid()}{holder_namespace_token()}:turn={relay_turn_id}"
        f":platform={task_context['platform'] or 'unknown'}"
    )
    reload_needed = announced = False

    def _on_contended() -> None:
        # A busy state.db, not a known holder: say nothing, but still reload after admission,
        # since the busy writer may have been the previous holder's final flush (one that fits
        # inside the write patience never gets here, before or after this signal existed).
        nonlocal reload_needed
        reload_needed = True

    def _on_wait(elapsed: float) -> None:
        nonlocal reload_needed, announced
        reload_needed = announced = True
        agent._emit_status(
            "⏳ Another Hermes process is using this session; "
            "waiting for it to finish before starting your turn..."
            if elapsed < 1.0 else
            f"⏳ Still waiting for the other Hermes process on this session ({int(elapsed)}s)..."
        )

    authority_floor = time.time()  # every acquisition attempt stamps its expiry after this instant
    if not db.acquire_session_turn_lease(
        session_id, holder, ttl_seconds=LEASE_TTL_SECONDS, wait_seconds=LEASE_WAIT_SECONDS,
        on_wait=_on_wait, on_contended=_on_contended,
        should_abort=lambda: getattr(agent, "_interrupt_requested", False),
    ):
        admission.early_result = _lease_not_acquired_result(agent, session_id, conversation_history)
        return admission

    # Assign only after admission so the finally cannot release a holder that never owned the
    # row; persist paths read the agent attr so a late flush is fenced in the same transaction.
    lease = DurableTurnLease(
        agent, db, session_id, holder, expires_at=_committed_lease_expiry(db, session_id, holder, authority_floor),
    )
    agent._active_session_turn_lease_holder = holder
    agent._active_session_turn_lease_ttl_seconds = LEASE_TTL_SECONDS
    try:
        _reload_admitted_turn(agent, db, session_id, task_context, admission, conversation_history, reload_needed, announced)
        consume_expected_turn_identity()
        lease.build_threads()
    except BaseException:
        # The façade never saw this lease; release here so an admitted row is not leaked.
        lease.release()
        raise
    admission.lease = lease
    return admission


def carry_unadmitted_user_message(
    early_result: dict[str, Any], user_message: Any, persist_user_message: Any, *,
    timestamp: Optional[float], display_kind: Optional[str], display_metadata: Optional[dict[str, Any]],
    platform_id: Optional[str],
) -> None:
    """A follow-up that interrupted the lease wait must not consume the accepted input: append it to
    the early result's history so the follow-up turn sees it and persists it (the flush honours
    ``_PERSIST_AFTER_ADMISSION_INTERRUPT`` because this turn never owned the lease). A hard stop
    (``/stop``) cancels the input instead."""
    if conditional_turn_pending():
        early_result.pop("_hard_interrupted", None)
        return
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
    deferred_user: dict[str, Any] = {
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


def _lease_not_acquired_result(agent, session_id: str, conversation_history) -> dict[str, Any]:
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
