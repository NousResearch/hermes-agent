"""Checkpointed long-running workflow prototype."""

from dataclasses import dataclass, field
from typing import Dict, List

from .delegation import DelegationEngine


@dataclass
class WorkflowCheckpoint:
    step: str
    status: str
    output_ref: str


@dataclass
class WorkflowRun:
    workflow_id: str
    checkpoints: List[WorkflowCheckpoint] = field(default_factory=list)

    def latest_step(self):
        return self.checkpoints[-1].step if self.checkpoints else None


class CheckpointedWorkflow:
    def __init__(self, workflow_id, steps, delegation_engine=None):
        self.run = WorkflowRun(workflow_id=workflow_id)
        self.steps = steps
        self.delegation_engine = delegation_engine or DelegationEngine()

    def execute(self, base_payload):
        completed_steps = {checkpoint.step for checkpoint in self.run.checkpoints}
        for step in self.steps:
            if step in completed_steps:
                continue
            payload = dict(base_payload)
            payload["task_id"] = "%s:%s" % (base_payload["task_id"], step)
            payload["task_type"] = step
            result = self.delegation_engine.delegate(_delegation_request_from_payload(payload))
            self.run.checkpoints.append(WorkflowCheckpoint(
                step=step,
                status=result.response.status,
                output_ref=result.persisted_outputs[0] if result.persisted_outputs else "",
            ))
            if result.response.status not in {"completed", "dry_run"}:
                break
        return self.run


def _delegation_request_from_payload(payload):
    from .contracts import DelegationRequest
    return DelegationRequest(
        task_id=payload["task_id"],
        project_id=payload["project_id"],
        task_type=payload["task_type"],
        prompt=payload["prompt"],
        working_directory=payload["working_directory"],
        opt_in_runtime=payload.get("opt_in_runtime", False),
        dry_run=payload.get("dry_run", True),
    )
