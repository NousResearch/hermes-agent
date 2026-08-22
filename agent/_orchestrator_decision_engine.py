"""Hermes Orchestrator Decision Engine (pure deterministic planner).

This module is a PURE decision engine: it takes a state, a set of worker
capabilities, and a set of restrictions, and returns the next action
without ever executing workers, opening sockets, calling LLMs, or mutating
the Kanban board.

Public API:
    DecisionEngine.plan(state, capabilities, restrictions) -> Decision

States (closed enum):
    READY    — task is ready to be worked on.
    RUNNING  — task is currently being worked on by a worker.
    WAITING  — task is waiting on an external event (e.g., async I/O).
    BLOCKED  — task is blocked by a hard dependency.
    FAILED   — task failed; may be retryable or terminal.
    DONE     — task is complete.

Actions (closed enum):
    RUN_WORKER — schedule a worker (engine does NOT execute).
    WAIT       — wait one tick (no decision change yet).
    ASK_HUMAN  — escalate to human via clarify.
    RETRY      — retry the previously failed step.
    FINISH     — task done; no further action.
    STOP       — terminate with no retry.

Decision output:
    next_action          — str (Action enum)
    selected_worker      — str | None
    rationale            — str (human-readable explanation)
    confidence           — float (0.0..1.0)
    stop_reason          — str | None
    discarded_workers    — list of (worker_id, reason)

Restrictions (closed set):
    "no_http"     — workers requiring HTTP are discarded.
    "no_llm"      — workers requiring LLM are discarded.
    "no_mutation" — workers that mutate state are discarded.
    "no_spawn"    — workers that spawn subprocesses are discarded.

Worker capabilities (per-worker):
    worker_id       — str
    handles_states  — list of valid source states
    requires_http   — bool
    requires_llm    — bool
    mutates_state   — bool
    spawns_subproc  — bool
    is_retryable    — bool (whether the worker can be retried after FAILED)
    recovery_kind   — "retryable" | "terminal" | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


_VALID_STATES = {"READY", "RUNNING", "WAITING", "BLOCKED", "FAILED", "DONE"}
_VALID_ACTIONS = {"RUN_WORKER", "WAIT", "ASK_HUMAN", "RETRY", "FINISH", "STOP"}
_VALID_RESTRICTIONS = {"no_http", "no_llm", "no_mutation", "no_spawn"}


@dataclass
class WorkerCapability:
    """Capability profile for a single worker."""
    worker_id: str
    handles_states: list
    requires_http: bool = False
    requires_llm: bool = False
    mutates_state: bool = False
    spawns_subproc: bool = False
    is_retryable: bool = False
    recovery_kind: str | None = None  # "retryable" | "terminal" | None

    def __post_init__(self):
        for s in self.handles_states:
            if s not in _VALID_STATES:
                raise ValueError(
                    f"worker {self.worker_id}: invalid handles_states entry {s!r}"
                )
        if self.recovery_kind not in (None, "retryable", "terminal"):
            raise ValueError(
                f"worker {self.worker_id}: invalid recovery_kind {self.recovery_kind!r}"
            )


@dataclass
class Decision:
    """Output of plan() — the next action to take."""
    next_action: str
    selected_worker: str | None
    rationale: str
    confidence: float
    stop_reason: str | None
    discarded_workers: list = field(default_factory=list)

    def __post_init__(self):
        if self.next_action not in _VALID_ACTIONS:
            raise ValueError(f"invalid action: {self.next_action!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence {self.confidence} out of [0.0, 1.0]"
            )


@dataclass
class OrchestratorState:
    """Canonical orchestrator state at one decision point."""
    state: str
    last_worker_id: str | None = None
    last_worker_status: str | None = None  # "success" | "failure" | None
    failure_count: int = 0
    human_input_required: bool = False
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.state not in _VALID_STATES:
            raise ValueError(f"invalid state: {self.state!r}")


@dataclass
class DecisionEngine:
    """Pure deterministic decision engine."""

    def plan(
        self,
        state: OrchestratorState,
        capabilities: Iterable[WorkerCapability],
        restrictions: Iterable[str] | None = None,
    ) -> Decision:
        """Return the next Decision given state, capabilities, restrictions."""
        restrictions = set(restrictions or [])
        invalid = restrictions - _VALID_RESTRICTIONS
        if invalid:
            raise ValueError(f"invalid restrictions: {invalid}")

        caps = list(capabilities)
        discarded: list[tuple[str, str]] = []

        # Filter capabilities by restrictions.
        eligible = []
        for c in caps:
            reason = self._is_disqualified(c, restrictions)
            if reason is not None:
                discarded.append((c.worker_id, reason))
            else:
                eligible.append(c)

        s = state.state

        # DONE → FINISH (no worker).
        if s == "DONE":
            return Decision(
                next_action="FINISH",
                selected_worker=None,
                rationale="task DONE; no further action",
                confidence=1.0,
                stop_reason=None,
                discarded_workers=discarded,
            )

        # FAILED → RETRY if retryable, else STOP.
        if s == "FAILED":
            if state.last_worker_id:
                last_cap = next(
                    (c for c in caps if c.worker_id == state.last_worker_id), None,
                )
                if last_cap and last_cap.is_retryable and last_cap.recovery_kind == "retryable":
                    # Cap retries at 3 by failure_count.
                    if state.failure_count >= 3:
                        return Decision(
                            next_action="STOP",
                            selected_worker=None,
                            rationale=(
                                f"FAILED on worker={state.last_worker_id}; "
                                f"failure_count={state.failure_count} >= 3 cap"
                            ),
                            confidence=0.95,
                            stop_reason="retry_cap_exhausted",
                            discarded_workers=discarded,
                        )
                    return Decision(
                        next_action="RETRY",
                        selected_worker=state.last_worker_id,
                        rationale=(
                            f"FAILED on worker={state.last_worker_id}; "
                            f"recovery_kind=retryable; failure_count={state.failure_count}"
                        ),
                        confidence=0.9,
                        stop_reason=None,
                        discarded_workers=discarded,
                    )
            return Decision(
                next_action="STOP",
                selected_worker=None,
                rationale=(
                    f"FAILED on worker={state.last_worker_id}; "
                    f"recovery_kind=terminal or unknown"
                ),
                confidence=0.95,
                stop_reason="terminal_failure",
                discarded_workers=discarded,
            )

        # BLOCKED → ASK_HUMAN.
        if s == "BLOCKED":
            return Decision(
                next_action="ASK_HUMAN",
                selected_worker=None,
                rationale="task BLOCKED; escalate to human",
                confidence=0.95,
                stop_reason=None,
                discarded_workers=discarded,
            )

        # RUNNING, WAITING → WAIT.
        if s in ("RUNNING", "WAITING"):
            return Decision(
                next_action="WAIT",
                selected_worker=None,
                rationale=f"state {s}; wait one tick for progression",
                confidence=0.9,
                stop_reason=None,
                discarded_workers=discarded,
            )

        # READY → pick first eligible worker that handles READY.
        if s == "READY":
            if state.human_input_required:
                return Decision(
                    next_action="ASK_HUMAN",
                    selected_worker=None,
                    rationale="READY but human_input_required=true",
                    confidence=0.95,
                    stop_reason=None,
                    discarded_workers=discarded,
                )
            ready_workers = [c for c in eligible if "READY" in c.handles_states]
            if ready_workers:
                # Deterministic: pick first by worker_id (sorted).
                ready_workers_sorted = sorted(ready_workers, key=lambda c: c.worker_id)
                pick = ready_workers_sorted[0]
                return Decision(
                    next_action="RUN_WORKER",
                    selected_worker=pick.worker_id,
                    rationale=(
                        f"READY + {len(ready_workers)} eligible worker(s); "
                        f"selected {pick.worker_id} (deterministic first)"
                    ),
                    confidence=0.85,
                    stop_reason=None,
                    discarded_workers=discarded,
                )
            # No eligible worker for READY.
            return Decision(
                next_action="ASK_HUMAN",
                selected_worker=None,
                rationale=(
                    f"READY + 0 eligible worker(s); "
                    f"discarded={[(w, r) for w, r in discarded]}"
                ),
                confidence=0.9,
                stop_reason="no_eligible_worker",
                discarded_workers=discarded,
            )

        # Should not reach here (state validated above).
        return Decision(
            next_action="STOP",
            selected_worker=None,
            rationale=f"unhandled state {s!r}",
            confidence=0.5,
            stop_reason="unhandled_state",
            discarded_workers=discarded,
        )

    @staticmethod
    def _is_disqualified(
        cap: WorkerCapability, restrictions: set,
    ) -> str | None:
        """Return disqualification reason or None."""
        if "no_http" in restrictions and cap.requires_http:
            return "requires_http but no_http restriction active"
        if "no_llm" in restrictions and cap.requires_llm:
            return "requires_llm but no_llm restriction active"
        if "no_mutation" in restrictions and cap.mutates_state:
            return "mutates_state but no_mutation restriction active"
        if "no_spawn" in restrictions and cap.spawns_subproc:
            return "spawns_subproc but no_spawn restriction active"
        return None