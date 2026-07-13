"""Execution planning and artifact ingestion for Hermes OS work graphs."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Sequence

from .runtime_policies import RuntimePolicy, evaluate_runtime_policy
from .work_graph import ExecutionResult, ValidationResult, build_execution_queue, ingest_execution_result


@dataclass(frozen=True)
class ExecutionBatch:
    batch_id: str
    items: List[Dict[str, object]]
    dry_run: bool = True


@dataclass(frozen=True)
class DryRunExecutionReport:
    project_id: str
    batches: List[ExecutionBatch]
    policies: List[Dict[str, object]] = field(default_factory=list)
    expected_artifacts: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def plan_execution_batches(graph, dry_run: bool = True) -> List[ExecutionBatch]:
    queue = build_execution_queue(graph, dry_run=dry_run)
    return [
        ExecutionBatch(
            batch_id="batch-%03d" % (index + 1),
            items=[item],
            dry_run=dry_run,
        )
        for index, item in enumerate(queue)
    ]


def build_dry_run_execution_report(graph, policy: RuntimePolicy | None = None):
    batches = plan_execution_batches(graph, dry_run=True)
    policy_records = []
    artifacts = []
    for batch in batches:
        for item in batch.items:
            decision = evaluate_runtime_policy(
                action="write" if item.get("approval_required") else "read",
                estimated_cost_usd=0.0,
                retry_count=0,
                approved=False,
                policy=policy,
            )
            policy_records.append(decision.audit)
            artifacts.append("hermes-os://artifact/%s/%s" % (graph.project_id, item["node_id"]))
    return DryRunExecutionReport(
        project_id=graph.project_id,
        batches=batches,
        policies=policy_records,
        expected_artifacts=artifacts,
    )


def ingest_artifacts(graph, node_id: str, artifacts: Sequence[str], repository=None):
    result = ExecutionResult(
        node_id=node_id,
        status="completed",
        runtime_provider="official-hermes-agent",
        artifacts=list(artifacts),
    )
    updated = ingest_execution_result(graph, result)
    validations = []
    for validation in updated.validation_results:
        if validation.node_id == node_id:
            validations.append(ValidationResult(
                node_id=node_id,
                status="passed" if artifacts else "failed",
                rules=validation.rules,
                evidence=list(artifacts),
                failures=[] if artifacts else ["missing-artifact"],
            ))
        else:
            validations.append(validation)
    updated = type(updated)(
        project_id=updated.project_id,
        nodes=updated.nodes,
        dependencies=updated.dependencies,
        assignments=updated.assignments,
        execution_results=updated.execution_results,
        validation_results=validations,
        findings=updated.findings,
    )
    if repository:
        from .persistence import persist_agent_artifact, persist_work_graph

        for index, artifact in enumerate(artifacts):
            persist_agent_artifact(
                repository,
                "%s:%s:%s" % (graph.project_id, node_id, index),
                {"project_id": graph.project_id, "node_id": node_id, "content_ref": artifact},
            )
        persist_work_graph(repository, graph.project_id, updated)
    return updated
