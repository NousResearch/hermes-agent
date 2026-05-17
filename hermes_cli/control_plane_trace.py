"""CP-S0 control-plane event trace instrumentation — observe-only.

Slice 0 of control-plane observability for the ``#hermes-main`` lane
(Hermes / Agent OS / DocOps / AgentFlow / Kanban / AI-memory / ACK).

This module is **read-only / dry-run only**. It builds JSON-serializable
event records that describe control-plane incidents so they can be
inspected, logged, and asserted on in tests. It performs **no durable
DB writes** and changes **no live delivery behaviour** — every event it
emits carries ``dry_run = True``.

Two incident shapes are reproducible here:

* **Kanban ACK loss** — a task whose verdict is ``GO`` but whose ACK to
  the original requester never lands. The verdict fact and the ACK fact
  are recorded as *separate* events so a ``GO`` verdict stays observable
  independently of delivery success. Known delivery failure modes:
  ``no_live_gateway_runner`` / ``no_subscription`` / ``no_target``.

* **Review handoff state-machine anomalies** — implementation workers
  must close a parent as ``done_GO_for_review`` when a linked review child
  already exists. ``implementation_running → blocked_review_required``
  with a waiting child, ``review_todo_stalled_due_parent_blocked``, and
  ``final_ack_missing`` are explicit anomaly/eval events.

* **Stale lane / compaction contamination** — a stale source candidate
  (an old compaction summary, a stale todo, a contaminated lane) that
  conflicts with the current lane/context authority. Recorded as a
  read-only ``lane_contract_conflict_detected`` event; the trace never
  claims to have *resolved* the conflict.

Integrating this with the live kanban ``task_events`` table is
deliberately out of scope for Slice 0 — callers that want durability
should serialize :meth:`EventTrace.to_json` and store it explicitly as
dry-run/test evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import json
import time

# ---------------------------------------------------------------------------
# Stable vocabulary
# ---------------------------------------------------------------------------

EVENT_TASK_VERDICT_RECORDED = "task_verdict_recorded"
EVENT_ACK_REQUESTED = "ack_requested"
EVENT_ACK_FAILED = "ack_failed"
EVENT_RESUME_PACKET_HYDRATED = "resume_packet_hydrated"
EVENT_LANE_CONTRACT_CONFLICT_DETECTED = "lane_contract_conflict_detected"
EVENT_TASK_STATUS_TRANSITION = "task_status_transition"
EVENT_REVIEW_TODO_STALLED_DUE_PARENT_BLOCKED = "review_todo_stalled_due_parent_blocked"
EVENT_FINAL_ACK_MISSING = "final_ack_missing"

#: Canonical event kinds, in the order Slice 0 was scoped for.
EVENT_KINDS = (
    EVENT_TASK_VERDICT_RECORDED,
    EVENT_ACK_REQUESTED,
    EVENT_ACK_FAILED,
    EVENT_RESUME_PACKET_HYDRATED,
    EVENT_LANE_CONTRACT_CONFLICT_DETECTED,
    EVENT_TASK_STATUS_TRANSITION,
    EVENT_REVIEW_TODO_STALLED_DUE_PARENT_BLOCKED,
    EVENT_FINAL_ACK_MISSING,
)

#: Task-status transition names used by review handoff evals.
KNOWN_TASK_STATUS_TRANSITIONS = (
    "implementation_running→done_GO_for_review",
    "implementation_running→blocked_review_required",
)

#: Transition names that are anomalies when a review child already exists.
ANOMALOUS_TASK_STATUS_TRANSITIONS = (
    "implementation_running→blocked_review_required",
)

#: Review-child stall kinds caused by parent handoff state misuse.
KNOWN_REVIEW_STALL_REASONS = ("parent_blocked_review_required",)

#: ACK delivery failure modes observed on the gateway notifier path.
KNOWN_ACK_ERRORS = (
    "no_live_gateway_runner",  # gateway process not running to deliver
    "no_subscription",  # task has no kanban_notify_subs row
    "no_target",  # subscription exists but platform/chat target is empty
)

#: ACK statuses that represent a *non-delivered* ACK (the ACK-loss shape).
NON_DELIVERED_ACK_STATUSES = ("FAILED", "PENDING")

#: Stale-source categories for lane-contract conflicts.
KNOWN_STALE_KINDS = ("compaction", "todo", "lane-contamination")

#: The lane this Slice 0 instrumentation defaults to.
DEFAULT_LANE = "#hermes-main"

#: Wall-clock callable returning integer epoch seconds.
Clock = Callable[[], int]


def _default_clock() -> int:
    return int(time.time())


# ---------------------------------------------------------------------------
# Event record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlPlaneEvent:
    """One immutable, JSON-serializable control-plane trace event.

    Fields are intentionally stable so inspection tooling and tests can
    match on them:

    * ``seq`` — 1-based ordinal within its trace.
    * ``kind`` — one of :data:`EVENT_KINDS`.
    * ``ts`` — integer epoch seconds (deterministic under an injected clock).
    * ``dry_run`` — always ``True`` for Slice 0 (observe-only).
    * ``lane`` — owning control-plane lane.
    * ``payload`` — kind-specific JSON-serializable detail.
    """

    seq: int
    kind: str
    ts: int
    lane: str
    payload: dict[str, Any]
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with a stable key order."""
        return {
            "seq": self.seq,
            "kind": self.kind,
            "ts": self.ts,
            "dry_run": self.dry_run,
            "lane": self.lane,
            "payload": dict(self.payload),
        }


# ---------------------------------------------------------------------------
# Trace collector
# ---------------------------------------------------------------------------


@dataclass
class EventTrace:
    """An ordered, in-memory collection of :class:`ControlPlaneEvent`.

    Pure and read-only with respect to the rest of the system: recording
    an event only appends to this object's own list. Inject ``clock`` for
    hermetic, byte-stable traces in tests.
    """

    lane: str = DEFAULT_LANE
    clock: Clock = _default_clock
    events: list[ControlPlaneEvent] = field(default_factory=list)

    # -- generic recording -------------------------------------------------

    def record(self, kind: str, payload: dict[str, Any]) -> ControlPlaneEvent:
        """Append an event of ``kind`` with ``payload`` and return it."""
        if kind not in EVENT_KINDS:
            raise ValueError(
                f"unknown control-plane event kind {kind!r}; "
                f"expected one of {EVENT_KINDS}"
            )
        ev = ControlPlaneEvent(
            seq=len(self.events) + 1,
            kind=kind,
            ts=int(self.clock()),
            lane=self.lane,
            payload=dict(payload),
        )
        self.events.append(ev)
        return ev

    # -- kind-specific builders -------------------------------------------

    def task_verdict_recorded(
        self, task_id: str, *, verdict: str, run_id: Optional[int] = None, **extra: Any
    ) -> ControlPlaneEvent:
        """Record that a task reached a verdict (e.g. ``GO`` / ``NO_GO``).

        Note this event carries *no* ``ack_status``: the verdict is a
        fact independent of whether the ACK was later delivered.
        """
        payload: dict[str, Any] = {"task_id": task_id, "verdict": verdict}
        if run_id is not None:
            payload["run_id"] = run_id
        payload.update(extra)
        return self.record(EVENT_TASK_VERDICT_RECORDED, payload)

    def ack_requested(
        self, task_id: str, *, target: Optional[str] = None, **extra: Any
    ) -> ControlPlaneEvent:
        """Record that an ACK delivery was attempted for ``task_id``."""
        payload: dict[str, Any] = {"task_id": task_id, "target": target}
        payload.update(extra)
        return self.record(EVENT_ACK_REQUESTED, payload)

    def ack_failed(
        self,
        task_id: str,
        *,
        ack_error: str,
        ack_status: str = "FAILED",
        task_verdict: Optional[str] = None,
        **extra: Any,
    ) -> ControlPlaneEvent:
        """Record a non-delivered ACK (the ACK-loss shape).

        ``ack_status`` must be a non-delivered status (``FAILED`` /
        ``PENDING``) — a delivered ACK is not an ACK-loss event.
        """
        if ack_status not in NON_DELIVERED_ACK_STATUSES:
            raise ValueError(
                f"ack_failed requires a non-delivered ack_status "
                f"{NON_DELIVERED_ACK_STATUSES}; got {ack_status!r}"
            )
        payload: dict[str, Any] = {
            "task_id": task_id,
            "ack_status": ack_status,
            "ack_error": ack_error,
        }
        if task_verdict is not None:
            payload["task_verdict"] = task_verdict
        payload.update(extra)
        return self.record(EVENT_ACK_FAILED, payload)

    def resume_packet_hydrated(
        self, task_id: str, *, source: str, fields: Optional[list[str]] = None, **extra: Any
    ) -> ControlPlaneEvent:
        """Record that a resume packet was hydrated for ``task_id``."""
        payload: dict[str, Any] = {
            "task_id": task_id,
            "source": source,
            "fields": list(fields or []),
        }
        payload.update(extra)
        return self.record(EVENT_RESUME_PACKET_HYDRATED, payload)

    def lane_contract_conflict_detected(
        self,
        *,
        stale_source: str,
        stale_kind: str,
        current_lane: str,
        current_authority: str,
        **extra: Any,
    ) -> ControlPlaneEvent:
        """Record a stale-source vs current-authority conflict (read-only).

        ``resolution`` is hard-coded to ``observed_only``: Slice 0 never
        mutates lane state, it only reports the contamination shape.
        """
        if stale_kind not in KNOWN_STALE_KINDS:
            raise ValueError(
                f"unknown stale_kind {stale_kind!r}; expected one of "
                f"{KNOWN_STALE_KINDS}"
            )
        payload: dict[str, Any] = {
            "stale_source": stale_source,
            "stale_kind": stale_kind,
            "current_lane": current_lane,
            "current_authority": current_authority,
            "resolution": "observed_only",
        }
        payload.update(extra)
        return self.record(EVENT_LANE_CONTRACT_CONFLICT_DETECTED, payload)

    def task_status_transition(
        self,
        task_id: str,
        *,
        transition: str,
        child_review_task_id: Optional[str] = None,
        expected_transition: str = "implementation_running→done_GO_for_review",
        **extra: Any,
    ) -> ControlPlaneEvent:
        """Record a Kanban task state transition for handoff evals.

        ``implementation_running→blocked_review_required`` is marked as an
        anomaly when a linked/pre-created review child exists, because it
        strands the child behind a parent that dependency promotion will not
        consider satisfied.
        """
        if transition not in KNOWN_TASK_STATUS_TRANSITIONS:
            raise ValueError(
                f"unknown task status transition {transition!r}; expected one of "
                f"{KNOWN_TASK_STATUS_TRANSITIONS}"
            )
        payload: dict[str, Any] = {
            "task_id": task_id,
            "transition": transition,
            "expected_transition": expected_transition,
            "anomaly": transition in ANOMALOUS_TASK_STATUS_TRANSITIONS,
        }
        if child_review_task_id is not None:
            payload["child_review_task_id"] = child_review_task_id
        payload.update(extra)
        return self.record(EVENT_TASK_STATUS_TRANSITION, payload)

    def review_todo_stalled_due_parent_blocked(
        self,
        review_task_id: str,
        *,
        parent_task_id: str,
        reason: str = "parent_blocked_review_required",
        **extra: Any,
    ) -> ControlPlaneEvent:
        """Record a review child stuck in ``todo`` behind a blocked parent."""
        if reason not in KNOWN_REVIEW_STALL_REASONS:
            raise ValueError(
                f"unknown review stall reason {reason!r}; expected one of "
                f"{KNOWN_REVIEW_STALL_REASONS}"
            )
        payload: dict[str, Any] = {
            "review_task_id": review_task_id,
            "parent_task_id": parent_task_id,
            "reason": reason,
            "anomaly": True,
        }
        payload.update(extra)
        return self.record(EVENT_REVIEW_TODO_STALLED_DUE_PARENT_BLOCKED, payload)

    def final_ack_missing(
        self,
        task_id: str,
        *,
        task_verdict: str,
        ack_status: str = "MISSING",
        return_to: Optional[str] = None,
        **extra: Any,
    ) -> ControlPlaneEvent:
        """Record a terminal verdict that lacks an origin-channel final ACK."""
        payload: dict[str, Any] = {
            "task_id": task_id,
            "task_verdict": task_verdict,
            "ack_status": ack_status,
            "return_to": return_to,
            "anomaly": True,
        }
        payload.update(extra)
        return self.record(EVENT_FINAL_ACK_MISSING, payload)

    # -- serialization -----------------------------------------------------

    def as_dicts(self) -> list[dict[str, Any]]:
        """Return all events as JSON-serializable dicts."""
        return [e.to_dict() for e in self.events]

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the whole trace to a deterministic JSON string."""
        return json.dumps(self.as_dicts(), ensure_ascii=False, indent=indent)

    def write_jsonl(self, path: str | Path) -> int:
        """Persist this dry-run trace as newline-delimited JSON.

        This is the only durable/logging helper in Slice 0, and it is
        explicitly caller-driven: no live gateway/Kanban path invokes it
        implicitly. Parent directories are created for temp/artifact paths,
        then one JSON object per event is appended with stable key order.
        Returns the number of written events for simple smoke assertions.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            for row in self.as_dicts():
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                fh.write("\n")
        return len(self.events)


# ---------------------------------------------------------------------------
# Incident reproducers
# ---------------------------------------------------------------------------


def build_ack_loss_trace(
    task_id: str,
    *,
    ack_error: str,
    ack_status: str = "FAILED",
    verdict: str = "GO",
    lane: str = DEFAULT_LANE,
    chat_id: Optional[str] = None,
    clock: Optional[Clock] = None,
) -> EventTrace:
    """Reproduce the Kanban ACK-loss incident shape as an event trace.

    Emits, in order: ``task_verdict_recorded`` (the verdict, e.g.
    ``GO``) → ``ack_requested`` → ``ack_failed`` (carrying ``ack_status``
    and ``ack_error``). The verdict and ACK facts stay separate so a
    ``GO`` task with a ``FAILED``/``PENDING`` ACK is unambiguous.
    """
    trace = EventTrace(lane=lane, clock=clock or _default_clock)
    target = f"discord:{chat_id}" if chat_id else None
    trace.task_verdict_recorded(task_id, verdict=verdict)
    trace.ack_requested(task_id, target=target)
    trace.ack_failed(
        task_id,
        ack_error=ack_error,
        ack_status=ack_status,
        task_verdict=verdict,
    )
    return trace


def build_lane_conflict_trace(
    *,
    stale_source: str,
    stale_kind: str,
    current_lane: str,
    current_authority: str,
    clock: Optional[Clock] = None,
) -> EventTrace:
    """Reproduce the stale lane/compaction contamination incident shape.

    Emits a single read-only ``lane_contract_conflict_detected`` event
    pinning the stale source candidate against the current lane/context
    authority.
    """
    trace = EventTrace(lane=current_lane, clock=clock or _default_clock)
    trace.lane_contract_conflict_detected(
        stale_source=stale_source,
        stale_kind=stale_kind,
        current_lane=current_lane,
        current_authority=current_authority,
    )
    return trace


def build_review_handoff_anomaly_trace(
    *,
    implementation_task_id: str,
    review_task_id: str,
    lane: str = DEFAULT_LANE,
    return_to: Optional[str] = None,
    clock: Optional[Clock] = None,
) -> EventTrace:
    """Reproduce the blocked-parent/todo-review/final-ACK anomaly chain.

    This is the regression shape for implementation cards that should have
    ended as ``done_GO_for_review`` but instead blocked with
    ``review-required`` while a review child was already linked.
    """
    trace = EventTrace(lane=lane, clock=clock or _default_clock)
    trace.task_status_transition(
        implementation_task_id,
        transition="implementation_running→blocked_review_required",
        child_review_task_id=review_task_id,
    )
    trace.review_todo_stalled_due_parent_blocked(
        review_task_id,
        parent_task_id=implementation_task_id,
    )
    trace.final_ack_missing(
        implementation_task_id,
        task_verdict="GO-for-review-not-promoted",
        return_to=return_to,
        review_task_id=review_task_id,
    )
    return trace
