"""Multi-agent orchestration façade (DAG + delegate_task + reflex hooks)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Sequence

from orchestration.agent_bus import AgentCommunicationBus
from orchestration.delegate_runner import run_delegate_node
from orchestration.learning import record_failure
from orchestration.registry import OrchestratorRegistry
from orchestration.task_graph import TaskGraph
from orchestration.types import GraphTaskRun, GraphTaskSpec, TaskStatus


class MultiAgentOrchestrator:
    """Coordinates profile-scoped Hermes workers on a dependency graph."""

    def __init__(self, orchestration_id: str | None = None) -> None:
        self.orchestration_id = orchestration_id or uuid.uuid4().hex[:12]
        self.bus = AgentCommunicationBus()
        self.registry = OrchestratorRegistry()

    async def run(
        self,
        parent_agent: Any,
        specs: Sequence[GraphTaskSpec],
    ) -> Mapping[str, GraphTaskRun]:
        graph = TaskGraph(specs)
        runs: Dict[str, GraphTaskRun] = graph.runs()

        async def execute(run: GraphTaskRun) -> None:
            run.status = TaskStatus.RUNNING
            try:
                payload = await run_delegate_node(
                    parent_agent,
                    run.spec,
                    orch_id=self.orchestration_id,
                )
                run.extra["delegate"] = payload
                if not payload.get("ok"):
                    msg = payload.get("error") or payload.get("status") or "delegate failed"
                    raise RuntimeError(str(msg))
                run.summary = str(payload.get("summary") or "")
                run.status = TaskStatus.DONE
            except Exception as exc:
                run.status = TaskStatus.FAILED
                run.error = str(exc)
                record_failure(
                    orchestration_id=self.orchestration_id,
                    task_id=run.spec.task_id,
                    error=str(exc),
                    trajectory_snippet=run.summary,
                )

        return await graph.run(execute, runs=runs)
