"""Harness run driver: the strict execution loop around the Hermes agent.

One harness iteration maps to one agent turn (``AIAgent.chat``) — the turn
loop's natural unit. That preserves prompt caching (no mid-conversation
rebuilds) and role alternation (harness guidance rides as the next user
turn, never a synthetic injection). The driver decides CONTINUE /
COMPLETED / BLOCKED / FAILED / STOPPED / BUDGET_EXHAUSTED from state,
evidence, verification, policy, and budget — never the model.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from . import recovery as _recovery
from .budget import BudgetGovernor
from .knowledge import KnowledgeCandidate, extract as extract_knowledge
from .state import (
    TERMINAL_OUTCOMES,
    Checkpoint,
    ExecutionBudget,
    ExecutionState,
    FeatureLock,
    FeatureState,
    FeatureStatus,
    Outcome,
    ScopeReason,
    StepStatus,
    Task,
    TaskStatus,
    ToolObservation,
    VerificationResult,
)
from .store import HarnessStore
from .verify import completion_allowed, verify


@dataclass
class StepResult:
    """Structured outcome of one agent turn, as seen by the harness."""

    summary: str = ""
    status_hint: str = StepStatus.CONTINUE
    observations: List[ToolObservation] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    proposed_knowledge: List[KnowledgeCandidate] = field(default_factory=list)
    error: Optional[str] = None
    transient_error: bool = False


class AgentStep(Protocol):
    def __call__(
        self, task: Task, feature: FeatureState, context: Dict[str, Any]
    ) -> StepResult: ...


_STAGNATION_LIMIT = 5


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class HarnessRunner:
    """Owns task runners: gates, budgets, recovery, persistence, completion."""

    def __init__(
        self, store: HarnessStore, make_step: Callable[[Task], AgentStep]
    ) -> None:
        self._store = store
        self._make_step = make_step
        self._locks: Dict[str, FeatureLock] = {}
        self._governors: Dict[str, BudgetGovernor] = {}
        self._seen: Dict[str, Dict[str, int]] = {}
        self._stagnant: Dict[str, int] = {}
        self._stop_flags: Dict[str, threading.Event] = {}
        self._outcomes: Dict[str, str] = {}

    # -- task lifecycle ---------------------------------------------------

    def create(
        self, goal: str, success_criteria: List[str], *, task_type: str = "FEATURE"
    ) -> Task:
        if not goal:
            raise ValueError("goal is required")
        task = Task(
            id=_new_id("t"),
            goal=goal,
            success_criteria=success_criteria or ["done"],
            type=task_type,
        )
        feature = FeatureState(
            id=f"{task.id}-f1",
            task_id=task.id,
            name="main",
            status=FeatureStatus.IMPLEMENTING,
        )
        task.feature_id = feature.id
        lock = FeatureLock()
        lock.select(
            feature, reason=ScopeReason.USER_SCOPE_CHANGE, evidence="task created"
        )
        self._locks[task.id] = lock
        self._governors[task.id] = BudgetGovernor(task.budget)
        self._store.save_task(task)
        self._store.save_feature(feature)
        self._store.save_execution(
            ExecutionState(task_id=task.id, feature_id=feature.id)
        )
        self._store.append_event("TASK", f"TASK_CREATED:{task.id}")
        return task

    def _load(self, task_id: str) -> Task:
        task = self._store.load_task(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id!r}")
        if task_id not in self._locks:
            lock = FeatureLock()
            for feature in self._store.features_for_task(task_id):
                if task.feature_id in (None, feature.id):
                    lock.select(feature)
                    task.feature_id = feature.id
            self._locks[task_id] = lock
        if task_id not in self._governors:
            self._governors[task_id] = BudgetGovernor(task.budget)
        if task_id not in self._outcomes:
            terminal = self._store.task_terminal_outcome(task_id)
            if terminal:
                self._outcomes[task_id] = terminal
        return task

    def status(self, task_id: str) -> Dict[str, Any]:
        task = self._load(task_id)
        state = self._store.load_execution(task_id)
        observations = self._store.observations_for_task(task_id)
        return {
            "id": task_id,
            "goal": task.goal,
            "status": task.status,
            "outcome": self._outcomes.get(task_id, ""),
            "iteration": state.iteration if state else 0,
            "observations": len(observations),
            "feature_id": task.feature_id,
        }

    # -- one gated iteration ------------------------------------------------

    def compile_context(self, task: Task, feature: FeatureState) -> Dict[str, Any]:
        """Smallest sufficient context doc for the next turn (bounded)."""
        knowledge = self._store.list_knowledge()[:5]
        return {
            "goal": task.goal,
            "success_criteria": task.success_criteria,
            "feature": feature.to_dict(),
            "hypothesis": feature.hypothesis,
            "known_issues": feature.known_issues[:5],
            "knowledge": [k.content for k in knowledge],
            "budget_remaining": self._governors[task.id].budget.to_dict()
            if task.id in self._governors
            else {},
        }

    def step(self, task_id: str) -> str:
        task = self._load(task_id)
        if self._outcomes.get(task_id) in TERMINAL_OUTCOMES:
            raise RuntimeError(
                f"task {task_id} is terminal ({self._outcomes[task_id]}); start a new task"
            )
        governor = self._governors[task_id]
        lock = self._locks[task_id]
        if lock.active is None:
            return Outcome.BLOCKED
        feature = lock.active

        state = self._store.load_execution(task_id) or ExecutionState(
            task_id=task_id, feature_id=feature.id
        )
        budget = (
            task.budget
            if isinstance(task.budget, ExecutionBudget)
            else ExecutionBudget()
        )
        if governor.exhausted():
            return self._finish(
                task, Outcome.BUDGET_EXHAUSTED, state, "budget exhausted"
            )

        context_id = f"ctx-{task_id}-{state.iteration + 1}"
        context = self.compile_context(task, feature)
        self._store.save_context(context_id, context)
        self._store.append_event("RUN", f"CONTEXT_COMPILED:{context_id}")

        if not governor.consume_iteration():
            return self._finish(
                task, Outcome.BUDGET_EXHAUSTED, state, "iteration budget exhausted"
            )

        step_fn = self._make_step(task)
        try:
            result = step_fn(task, feature, context)
        except Exception as exc:  # noqa: BLE001 — harness must survive agent errors
            result = StepResult(error=str(exc)[:500], transient_error=False)

        if result.error is not None:
            return self._recover(
                task,
                feature,
                state,
                result,
                context_id,
                governor,
                action_fp="",
                failure=result.error,
                failure_class=_recovery.classify_failure(
                    result.error, transient=result.transient_error
                ),
            )

        if result.assumptions:
            newest = result.assumptions[0]
            if feature.hypothesis and feature.hypothesis != newest:
                self._store.append_event("RUN", "HYPOTHESIS_CHANGED")
            feature.hypothesis = newest

        for obs in result.observations:
            self._store.append_event("RUN", f"OBSERVATION_RECORDED:{obs.id}")
            self._store.save_observation(task.id, obs.to_dict())
            governor.consume_tool_call()
            if not obs.success:
                return self._recover(
                    task,
                    feature,
                    state,
                    result,
                    context_id,
                    governor,
                    action_fp=f"{obs.tool}\x00{obs.summary}",
                    failure=obs.summary,
                    failure_class=_recovery.classify_failure(obs.summary),
                )

        progressed = _recovery.progress_made(
            new_evidence=bool(result.evidence),
            implementation_progress=any(o.success for o in result.observations),
        )
        if not progressed:
            count = self._stagnant.get(task_id, 0) + 1
            self._stagnant[task_id] = count
            if count >= _STAGNATION_LIMIT:
                return self._finish(task, Outcome.STOPPED, state, "stagnation limit")
            self._store.append_event("RUN", f"STAGNATION:{count}")
            return Outcome.CONTINUE
        self._stagnant[task_id] = 0

        checks = self._checks_from_observations(result)
        verification = verify(checks, task.success_criteria)
        if not verification.passed:
            return self._recover(
                task,
                feature,
                state,
                result,
                context_id,
                governor,
                action_fp="verification",
                failure=";".join(verification.failures) or "verification failed",
                failure_class=_recovery.FailureClass.DETERMINISTIC,
            )
        self._store.append_event("RUN", "VERIFICATION_PASSED")

        if result.proposed_knowledge:
            items = extract_knowledge(
                result.proposed_knowledge,
                self._store.list_knowledge(),
                has_evidence=bool(result.evidence)
                or any(o.success for o in result.observations),
            )
            for item in items:
                self._store.save_knowledge(item)
                self._store.append_event("RUN", f"KNOWLEDGE_STORED:{item.id}")

        state.iteration += 1
        state.last_evidence = list(result.evidence)
        self._store.save_execution(state)
        self._store.save_feature(feature)
        self._store.save_task(task)

        if result.status_hint == StepStatus.DONE and completion_allowed(
            True, True, verification.passed, not state.open_questions
        ):
            return self._finish(task, Outcome.COMPLETED, state, result.summary)
        if result.status_hint == StepStatus.BLOCKED:
            return self._finish(task, Outcome.BLOCKED, state, result.summary)
        if result.status_hint == StepStatus.FAILED:
            return self._finish(task, Outcome.FAILED, state, result.summary)
        return Outcome.CONTINUE

    # -- helpers --------------------------------------------------------------

    def _checks_from_observations(self, result: StepResult) -> list:
        from .verify import CheckStrength, VerificationCheck

        return [
            VerificationCheck(
                name=o.tool or "tool",
                passed=o.success,
                detail=o.summary,
                strength=CheckStrength.RUNTIME,
            )
            for o in result.observations
        ]

    def _recover(
        self,
        task: Task,
        feature: FeatureState,
        state: ExecutionState,
        result: StepResult,
        context_id: str,
        governor: BudgetGovernor,
        *,
        action_fp: str,
        failure: str,
        failure_class: str,
    ) -> str:
        seen = self._seen.setdefault(task.id, {})
        key = f"{failure}\x00{feature.hypothesis or ''}\x00{action_fp}"
        seen[key] = seen.get(key, 0) + 1
        strategy = _recovery.decide(
            failure, feature.hypothesis or "", action_fp, failure_class, seen
        )
        self._store.append_event("RUN", f"RECOVERY:{strategy}:{failure[:200]}")
        if strategy == _recovery.Strategy.RETRY:
            if governor.consume_retry():
                return Outcome.CONTINUE
            return self._finish(
                task, Outcome.BUDGET_EXHAUSTED, state, "retry budget exhausted"
            )
        if strategy == _recovery.Strategy.REPLAN:
            if governor.consume_replan():
                self._checkpoint(task, feature, state, context_id, "before-replan")
                return Outcome.CONTINUE
            return self._finish(
                task, Outcome.BUDGET_EXHAUSTED, state, "replan budget exhausted"
            )
        if strategy in (_recovery.Strategy.STOP, _recovery.Strategy.ESCALATE):
            return self._finish(task, Outcome.STOPPED, state, f"recovery:{strategy}")
        return Outcome.CONTINUE

    def _checkpoint(
        self,
        task: Task,
        feature: FeatureState,
        state: ExecutionState,
        context_id: str,
        reason: str,
    ) -> Checkpoint:
        existing = self._store.checkpoints_for_task(task.id)
        checkpoint = Checkpoint(
            id=f"cp-{task.id}-{len(existing) + 1}",
            task_id=task.id,
            feature_id=feature.id,
            state=state,
            context_ref=context_id,
            reason=reason,
        )
        self._store.save_checkpoint(checkpoint)
        self._store.append_event("RUN", f"CHECKPOINT:{checkpoint.id}:{reason}")
        return checkpoint

    def _finish(
        self, task: Task, outcome: str, state: ExecutionState, detail: str
    ) -> str:
        phase = {
            "COMPLETED": TaskStatus.COMPLETED,
            "BLOCKED": TaskStatus.BLOCKED,
            "FAILED": TaskStatus.FAILED,
            "STOPPED": TaskStatus.STOPPED,
            "BUDGET_EXHAUSTED": TaskStatus.BUDGET_EXHAUSTED,
        }.get(outcome, task.status)
        task.status = phase
        state.phase = phase
        self._store.save_task(task)
        self._store.save_execution(state)
        if outcome in TERMINAL_OUTCOMES:
            self._checkpoint(
                task,
                self._locks[task.id].active
                or FeatureState(id="?", task_id=task.id, name="?"),
                state,
                "",
                f"task-{outcome.lower()}",
            )
            self._store.append_event("TASK", f"TASK_{outcome}:{task.id}")
            if detail:
                self._store.append_event(
                    "TASK", f"TASK_DETAIL:{task.id}:{detail[:200]}"
                )
        self._outcomes[task.id] = outcome
        return outcome

    # -- multi-iteration runs -----------------------------------------------------

    def run(self, task_id: str, max_rounds: int = 50) -> str:
        flag = self._stop_flags.setdefault(task_id, threading.Event())
        flag.clear()
        outcome = Outcome.CONTINUE
        for _ in range(max_rounds):
            if flag.is_set():
                task = self._load(task_id)
                state = self._store.load_execution(task_id) or ExecutionState(
                    task_id=task_id, feature_id=task.feature_id or ""
                )
                outcome = self._finish(task, Outcome.STOPPED, state, "cancelled")
                break
            outcome = self.step(task_id)
            if outcome in TERMINAL_OUTCOMES:
                break
        return outcome

    def pause(self, task_id: str) -> None:
        self._stop_flags.setdefault(task_id, threading.Event()).set()
        task = self._load(task_id)
        state = self._store.load_execution(task_id) or ExecutionState(
            task_id=task_id, feature_id=task.feature_id or ""
        )
        feature = self._locks[task_id].active or FeatureState(
            id=task.feature_id or "?", task_id=task_id, name="?"
        )
        self._checkpoint(task, feature, state, "", "paused")
        self._store.append_event("TASK", f"TASK_PAUSED:{task_id}")

    def cancel(self, task_id: str) -> None:
        self.pause(task_id)
        self._store.append_event("TASK", f"TASK_CANCELLED:{task_id}")

    def resume(self, task_id: str, max_rounds: int = 50) -> str:
        """Continue from persisted state + retrieval, never replayed history."""
        task = self._load(task_id)
        if self._outcomes.get(task_id) in TERMINAL_OUTCOMES:
            raise RuntimeError(f"task {task_id} is terminal; start a new task")
        self._store.append_event("TASK", f"TASK_RESUMED:{task_id}")
        return self.run(task_id, max_rounds)
