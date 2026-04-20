from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass
class TaskEnvelope:
    session_id: str
    task_id: str
    workflow: str
    backend: str = "legacy"
    thread_id: str = ""
    run_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    name: str
    status: str = "pending"
    started_at: float | None = None
    completed_at: float | None = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class WorkflowTrace:
    workflow: str
    backend: str
    status: str = "pending"
    steps: list[WorkflowStep] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    failed_step: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class HermesTaskOrchestrator:
    def __init__(self, backend: str = "legacy"):
        self.backend = (backend or "legacy").strip().lower()

    def run(
        self,
        envelope: TaskEnvelope,
        *,
        steps: Iterable[WorkflowStep],
        executor: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.backend == "langgraph":
            self._ensure_langgraph_available()
        trace = WorkflowTrace(
            workflow=envelope.workflow,
            backend=self.backend,
            status="running",
            steps=list(steps),
        )
        try:
            result = self._run_legacy(trace, executor)
        except Exception as exc:
            trace.status = "failed"
            trace.failed_step = trace.steps[-1].name if trace.steps else None
            raise
        trace.status = "completed"
        return {
            "status": trace.status,
            "result": result,
            "trace": trace.to_dict(),
            "completed_steps": [step.name for step in trace.steps if step.status == "completed"],
            "artifacts": trace.artifacts,
            "failed_step": trace.failed_step,
        }

    def _run_legacy(self, trace: WorkflowTrace, executor: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        if trace.steps:
            step = trace.steps[0]
            step.status = "running"
            step.started_at = time.time()
        result = executor()
        if trace.steps:
            step.status = "completed"
            step.completed_at = time.time()
            step.artifacts = dict(result.get("artifacts") or {}) if isinstance(result, dict) else {}
        if isinstance(result, dict) and result.get("artifacts"):
            trace.artifacts.update(result["artifacts"])
        return result

    def _ensure_langgraph_available(self) -> None:
        try:
            __import__("langgraph")
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph orchestration backend is optional and requires the 'langgraph' package to be installed."
            ) from exc
