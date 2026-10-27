"""Per-session task lifecycle: "may this task still start work?"

The session lifecycle answers "is the chat still open?".  The task lifecycle answers
a different question: "may a late background result still wake this session?".

Those are different states.  ``/stop`` ends a TASK, not a session: the user keeps
chatting, but no background event may re-activate the finished work.  The existing
fail-closed path (``gateway/run_notifications_ownership.py``) only knows *session*
end reasons, so a stopped task looked exactly like an open one and every late
process completion re-entered the model (ADR-0081).

Scope
-----
Deliberately small and self-contained: a state store, the gate predicate, the
refusal register, and the two helpers the gateway calls.  It imports nothing from
the gateway at module level, so it can be unit-tested on its own.

States
------
``active``    the task may still start work; a late result may wake the session.
``drained``   the task finished; late results are recorded, not delivered.
``cancelled`` the task was aborted; late results are recorded, not delivered, and
              the task's own processes are terminated.

``drained`` and ``cancelled`` are own states, NOT session end reasons: the session
stays open so the user can keep writing.  A new USER turn re-arms the task to
``active`` -- a late background result must never be able to do that itself.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("gateway.run")

TASK_ACTIVE = "active"
TASK_DRAINED = "drained"
TASK_CANCELLED = "cancelled"

#: States in which a task may no longer start work.  ``drained`` and ``cancelled``
#: both block the injection; only ``cancelled`` additionally terminates processes.
CLOSED_TASK_STATES = frozenset({TASK_DRAINED, TASK_CANCELLED})

#: Marker a model emits at the end of its final report to say "this assignment is
#: over".  The model REPORTS, the runtime ENFORCES: the gate below is the lock,
#: the marker is only the trigger (ADR-0081).
TASK_END_MARKER = "[TASK_DONE]"


@dataclass
class TaskLifecycle:
    """The lifecycle state of ONE task of a session."""

    session_key: str = ""
    state: str = TASK_ACTIVE
    started_at: float = field(default_factory=time.time)
    #: The turn's unique task id (``agent._current_task_id``).  A background job
    #: inherits it as ``owner_task_id``, so it identifies the task a late result
    #: belongs to -- far more robust than a timestamp (ADR-0081 R2).
    task_id: str = ""
    closed_at: float = 0.0
    reason: str = ""
    #: ``True`` when the close was INFERRED from "the turn ended and nothing was
    #: in flight" rather than declared (marker or ``/stop``).  An inferred close
    #: is a guess, so a result whose job already ran under this task is still
    #: delivered -- otherwise a job that registers a moment after the turn ends
    #: would be swallowed by a guess.
    inferred: bool = False
    #: Refused late results, kept structured so they stay retrievable without a
    #: chat notification (ADR-0081 Anforderung 5).
    refused: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        return self.state in CLOSED_TASK_STATES

    @property
    def is_cancelled(self) -> bool:
        return self.state == TASK_CANCELLED

    def close(self, state: str, *, reason: str = "", inferred: bool = False) -> None:
        """Close this task.  Idempotent, first close wins: a later ``drained``
        must not downgrade a ``cancelled``."""
        if state not in CLOSED_TASK_STATES:
            raise ValueError(f"not a closing state: {state!r}")
        if self.is_closed:
            return
        self.state = state
        self.closed_at = time.time()
        self.reason = reason or self.reason
        # A declared close (marker, /stop) always wins over a later inferred one,
        # but an inferred close is never downgraded into a declared one either.
        self.inferred = inferred


class TaskLifecycleRegistry:
    """Per-session task states, keyed by ``session_key``.

    A session keeps the history of its tasks, because a late result must be
    attributed to the task it belongs to: a result that started BEFORE the task
    was closed belongs to the closed task and is refused, even after a new user
    instruction re-armed the session with a fresh task.
    """

    #: A session keeps only its recent task history.  Late results that resolve
    #: to a trimmed task fall back to the time-based attribution and then fail
    #: open -- bounded memory beats an unbounded leak in a long-lived gateway.
    MAX_TASKS_PER_SESSION = 20
    #: Refused results stay retrievable, but a pathological loop must not grow a
    #: single task's record without bound.
    MAX_REFUSED_PER_TASK = 50
    #: Sessions kept in memory.  Only CLOSED sessions are evicted; an active one
    #: is never dropped, because that would let a late result re-arm finished work.
    MAX_SESSIONS = 500

    def __init__(self) -> None:
        self._by_session: dict[str, list[TaskLifecycle]] = {}

    # -- lifecycle -------------------------------------------------------
    def begin(self, session_key: str, *, task_id: str = "") -> Optional[TaskLifecycle]:
        """Start (or re-arm) the task of a session.  Called on a USER turn.

        Re-arming is what keeps a late background result from re-activating a
        finished task: only a new user instruction opens a task again.  The
        previous task stays in the history -- its late results are still refused.
        """
        if not session_key:
            return None
        lifecycle = TaskLifecycle(
            session_key=session_key, state=TASK_ACTIVE, task_id=task_id or "",
        )
        tasks = self._by_session.setdefault(session_key, [])
        tasks.append(lifecycle)
        self._trim_session(tasks)
        self._evict_closed_sessions()
        return lifecycle

    @classmethod
    def _trim_session(cls, tasks: list[TaskLifecycle]) -> None:
        """Keep the most recent tasks of a session, newest last.

        The current (last) task is always kept: dropping it would lose the task
        a late result has to be attributed to.
        """
        excess = len(tasks) - cls.MAX_TASKS_PER_SESSION
        if excess > 0:
            del tasks[:excess]

    def _evict_closed_sessions(self, limit: Optional[int] = None) -> None:
        """Drop the oldest sessions whose last task is closed.

        An inferred close is enough here: the session is not waiting for work
        any more, and a later user turn simply starts a fresh history.
        """
        limit = self.MAX_SESSIONS if limit is None else limit
        while len(self._by_session) > limit:
            oldest: Optional[str] = None
            oldest_stamp = float("inf")
            for session_key, tasks in self._by_session.items():
                if not tasks or not tasks[-1].is_closed:
                    continue
                stamp = tasks[-1].closed_at or tasks[-1].started_at
                if stamp < oldest_stamp:
                    oldest, oldest_stamp = session_key, stamp
            if oldest is None:
                # Every session still has an active task: keep them all rather
                # than evicting the state the gate depends on.
                return
            self._by_session.pop(oldest, None)

    def close(
        self, session_key: str, state: str, *, reason: str = "", inferred: bool = False,
    ) -> Optional[TaskLifecycle]:
        """Close the current task of a session.  Returns the closed lifecycle, or
        ``None`` when there is no task to close."""
        if not session_key:
            return None
        tasks = self._by_session.get(session_key)
        if not tasks:
            lifecycle = TaskLifecycle(session_key=session_key, state=TASK_ACTIVE)
            self._by_session.setdefault(session_key, []).append(lifecycle)
            tasks = self._by_session[session_key]
        lifecycle = tasks[-1]
        lifecycle.close(state, reason=reason, inferred=inferred)
        logger.info(
            "Session %s: task closed as %s (%s)",
            session_key, lifecycle.state, lifecycle.reason or "no reason",
        )
        return lifecycle

    def get(self, session_key: str) -> Optional[TaskLifecycle]:
        """The CURRENT task of a session, or ``None`` when there is none."""
        if not session_key:
            return None
        tasks = self._by_session.get(session_key)
        return tasks[-1] if tasks else None

    def tasks(self, session_key: str) -> list[TaskLifecycle]:
        if not session_key:
            return []
        return list(self._by_session.get(session_key, ()))

    def state_of(self, session_key: str) -> str:
        """The state of the session's current task.  ``active`` when unknown."""
        lifecycle = self.get(session_key)
        return lifecycle.state if lifecycle is not None else TASK_ACTIVE

    # -- the gate --------------------------------------------------------
    def may_start_work(
        self,
        session_key: str,
        started_at: Optional[float] = None,
        task_id: Optional[str] = None,
    ) -> bool:
        """Whether a background result may still start a turn for this session.

        Attribution is by the result's ``owner_task_id`` -- the unique id of the
        turn that spawned the job.  That is stable across restarts, retries and
        overlapping tasks, unlike a timestamp.

        A result belongs to the task it was started in.  If that task is closed,
        the result is refused -- even if a later user instruction already re-armed
        the session with a new task.  Fail-open when the result cannot be
        attributed: blocking a result we cannot place would silently swallow
        legitimate work (ADR-0081 R1/R2).
        """
        tasks = self._by_session.get(session_key)
        if not tasks:
            return True
        owner = self._owner_of(tasks, started_at=started_at, task_id=task_id)
        if owner is None:
            # Not attributable: fail open rather than swallow a legitimate result.
            return True
        if not owner.is_closed:
            return True
        # An INFERRED close ("the turn ended and nothing was in flight") is a
        # guess, not a statement.  If the job demonstrably ran under this task,
        # the guess is what was wrong -- deliver the result instead of dropping
        # work that the runtime merely failed to see at turn end.
        if owner.inferred and (started_at is None or started_at <= owner.closed_at):
            logger.info(
                "Delivering a late result for session %s: the task was closed by inference, "
                "and the job started under it.", session_key,
            )
            return True
        return False

    @staticmethod
    def _owner_of(
        tasks: list[TaskLifecycle],
        *,
        started_at: Optional[float] = None,
        task_id: Optional[str] = None,
    ) -> Optional[TaskLifecycle]:
        """The task a result belongs to.

        Attribution uses every stable signal the event carries, in order of
        reliability:

        1. ``owner_task_id`` -- the id of the task that spawned the job.  It
           identifies the task unambiguously whenever the runtime hands out a
           per-turn id.
        2. ``started_at`` -- the turn-granular fallback.  A result belongs to the
           task that was already running when it started.

        A result that predates every known task belongs to none of them: the
        caller then fails open rather than swallow legitimate work.
        """
        if task_id:
            for task in tasks:
                if task.task_id and task.task_id == task_id:
                    return task
            # The owner id is not one of ours (a different runtime scope, or a
            # task that never registered).  Fall through to the time-based
            # attribution instead of trusting an id we cannot place.
        if started_at is None:
            return tasks[-1]
        owner: Optional[TaskLifecycle] = None
        for task in tasks:
            if started_at >= task.started_at:
                owner = task
            else:
                break
        return owner

    def record_refusal(
        self,
        session_key: str,
        *,
        reason: str = "",
        detail: Optional[dict[str, Any]] = None,
        task_id: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> None:
        """Record a refused late result so it stays retrievable without a chat
        notification (ADR-0081 Anforderung 5).

        The entry is filed on the task that OWNS the result, not on whichever
        task is current: after a re-arm the two differ, and a record pointing at
        the wrong task is worse than no record.
        """
        tasks = self._by_session.setdefault(session_key, [])
        if not tasks:
            tasks.append(TaskLifecycle(session_key=session_key, state=TASK_ACTIVE))
        owner = self._owner_of(tasks, started_at=started_at, task_id=task_id) or tasks[-1]
        entry: dict[str, Any] = {"reason": reason or "task closed", "at": time.time()}
        if detail:
            entry.update(detail)
        owner.refused.append(entry)
        excess = len(owner.refused) - self.MAX_REFUSED_PER_TASK
        if excess > 0:
            del owner.refused[:excess]
        logger.info(
            "Refused late background result for session %s: %s", session_key, entry["reason"],
        )

    def refused_results(self, session_key: str) -> list[dict[str, Any]]:
        """Every refused result of this session, across all of its tasks."""
        refused: list[dict[str, Any]] = []
        for task in self.tasks(session_key):
            refused.extend(task.refused)
        return refused

    def clear_session(self, session_key: str) -> None:
        """Forget a session's task state (conversation boundary)."""
        if session_key:
            self._by_session.pop(session_key, None)

    def snapshot(self) -> dict[str, Any]:
        """Structured view for diagnostics."""
        return {
            session_key: [
                {
                    "state": task.state,
                    "reason": task.reason,
                    "inferred": task.inferred,
                    "refused": list(task.refused),
                }
                for task in tasks
            ]
            for session_key, tasks in self._by_session.items()
        }


#: Process-local registry.  The gateway is one process and session keys are
#: profile-namespaced, so a module-level store matches how ``process_registry``
#: is scoped.
task_lifecycles = TaskLifecycleRegistry()


def strip_task_end_marker(text: str) -> tuple[str, bool]:
    """Remove the task-end marker from a model response.

    Returns ``(clean_text, marker_found)``.  The marker is a report, not
    content: it must not reach the user.
    """
    if not text or TASK_END_MARKER not in text:
        return text, False
    return text.replace(TASK_END_MARKER, "").strip(), True


def agent_result_reports_task_end(agent_result: Any) -> bool:
    """Whether the model reported "this assignment is over" (ADR-0081).

    The marker is a REPORT, not content: the delivery path removes it with
    :func:`strip_task_end_marker` and only the runtime acts on it.
    """
    if not isinstance(agent_result, dict):
        return False
    text = agent_result.get("final_response") or ""
    return isinstance(text, str) and TASK_END_MARKER in text


def session_has_live_work(session_key: str) -> bool:
    """Whether the session still has work in flight that it is waiting for.

    Live background processes and live delegations both count: an agent that
    started a long runner and ended its turn is waiting for that runner, so the
    assignment is not over yet (ADR-0081: "Turn-Ende ohne weitere Arbeitsabsicht").
    """
    if not session_key:
        return False
    try:
        from tools.process_registry import process_registry
        for session in process_registry.list_sessions(session_key=session_key):
            if not getattr(session, "exited", False):
                return True
    except Exception:  # pragma: no cover - registry lookups must never break a turn
        logger.debug("Could not read live processes for session %s", session_key, exc_info=True)
    try:
        from tools.async_delegation import has_live_for_session
        if has_live_for_session(session_key=session_key):
            return True
    except Exception:  # pragma: no cover
        logger.debug("Could not read live delegations for session %s", session_key, exc_info=True)
    return False


def close_finished_task(
    session_key: str, *, marker_found: bool, has_live_work: bool, reason: str = "",
) -> Optional[TaskLifecycle]:
    """Close the task of a session when the assignment is over.

    The assignment is over when the model reported it (marker) or when the turn
    left no work in flight.  A turn that still has work in flight keeps the task
    ``active``, so its late result is still delivered.
    """
    if not session_key:
        return None
    if marker_found:
        return task_lifecycles.close(
            session_key, TASK_DRAINED, reason=reason or "model reported task end",
        )
    if not has_live_work:
        # Inferred, not declared: the turn ended and we SAW nothing in flight.
        # A job that registers a moment later is still legitimate work.
        return task_lifecycles.close(
            session_key, TASK_DRAINED, reason=reason or "turn ended without pending work",
            inferred=True,
        )
    return None


# ---------------------------------------------------------------------------
# Gateway wiring
# ---------------------------------------------------------------------------
# Kein gateway-Import: diese Datei bleibt frei von Zyklen und einzeln testbar.

class GatewayTaskLifecycleMixin:
    """Task-lifecycle hooks for :class:`~gateway.run.GatewayRunner`."""

    # -- 1. the gate -----------------------------------------------------
    def _task_may_start_work(self, evt: dict[str, Any]) -> bool:
        """Whether a background result may still start a turn for its session.

        Fail-open for an unknown session: blocking a result we cannot attribute
        would silently swallow legitimate work (ADR-0081 R1/R2).
        """
        
        session_key = str(evt.get("session_key") or "").strip()
        if not session_key:
            return True
        started_at = evt.get("started_at")
        try:
            started_at = float(started_at) if started_at is not None else None
        except (TypeError, ValueError):
            started_at = None
        # The owner task id identifies the turn that spawned the job -- a stable,
        # unique id, unlike a timestamp (ADR-0081 R2).
        owner_task_id = str(evt.get("owner_task_id") or evt.get("task_id") or "").strip()
        if task_lifecycles.may_start_work(
            session_key, started_at=started_at, task_id=owner_task_id or None,
        ):
            return True
        task_lifecycles.record_refusal(
            session_key,
            reason="task closed before the result arrived",
            # Attribute the record to the task that owns the result, not to
            # whatever task happens to be current after a re-arm.
            task_id=owner_task_id or None,
            started_at=started_at,
            detail={
                "process_id": evt.get("session_id"),
                "owner_task_id": evt.get("owner_task_id"),
                "exit_code": evt.get("exit_code"),
                "completion_reason": evt.get("completion_reason"),
            },
        )
        logger.warning(
            "Refused late background result for session %s (process %s): the task was closed.",
            session_key, evt.get("session_id", "?"),
        )
        return False

    # -- 2. re-arm on a user turn ---------------------------------------
    def _task_begin_turn(
        self, session_key: str, *, is_internal: bool, task_id: str = "",
    ) -> None:
        """Open (or re-open) the task of a session.

        Only a USER turn re-arms.  An internal event (a background result waking
        the session) must never re-open a task -- that is exactly the
        reactivation this fixes.
        """
        
        if not session_key or is_internal:
            return
        task_lifecycles.begin(session_key, task_id=task_id or "")

    # -- 3. close on turn end -------------------------------------------
    def _task_close_on_turn_end(
        self, session_key: str, *, marker_found: bool = False, reason: str = "",
    ) -> None:
        """Close the task when the assignment is over.

        The assignment is over when the model reported it (``[TASK_DONE]``) or
        when the turn ended with no work in flight.  A turn that still has a
        background job or a live delegation keeps the task open, so its result is
        still delivered.
        """
        
        if not session_key:
            return
        try:
            live = session_has_live_work(session_key)
        except Exception:  # pragma: no cover - never break a turn on a registry read
            logger.debug("Could not check live work for %s", session_key, exc_info=True)
            return
        close_finished_task(
            session_key, marker_found=marker_found, has_live_work=live,
            reason=reason,
        )

    # -- 4. explicit abort ----------------------------------------------
    def _task_cancel(self, session_key: str, *, reason: str = "") -> None:
        """Cancel the task of a session and terminate its own processes.

        This is the "job ownership with abort effect" of ADR-0081: the processes
        that belonged to the cancelled task are stopped through the existing
        ``ProcessRegistry``, so nothing of that task keeps running.
        """
        
        if not session_key:
            return
        task_lifecycles.close(session_key, "cancelled", reason=reason or "user stop")
        self._task_terminate_processes(session_key)

    @staticmethod
    def _task_terminate_processes(session_key: str) -> int:
        """Stop the background processes of a cancelled task.

        Only processes that belonged to THIS session are touched, and only those
        that are still running.  Failures are logged, never swallowed
        (ADR-0081 Anforderung 5).
        """
        if not session_key:
            return 0
        try:
            from tools.process_registry import process_registry
            stopped = 0
            for session in process_registry.list_sessions(session_key=session_key):
                if getattr(session, "exited", False):
                    continue
                result = process_registry.kill_process(
                    session.id, source="task_cancelled", consume_output=True,
                )
                if result.get("status") in {"killed", "already_exited"}:
                    stopped += 1
            if stopped:
                logger.info(
                    "Cancelled task of session %s: stopped %d background process(es).",
                    session_key, stopped,
                )
            return stopped
        except Exception:
            logger.warning(
                "Failed to terminate background processes of cancelled task %s",
                session_key, exc_info=True,
            )
            return 0
