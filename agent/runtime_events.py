from __future__ import annotations

from agent.runtime_types import RunEventRecord

RUN_CREATED = "RunCreated"
RUN_STATE_CHANGED = "RunStateChanged"
STEP_STARTED = "StepStarted"
STEP_COMPLETED = "StepCompleted"
STEP_FAILED = "StepFailed"
TOOL_CALL_STARTED = "ToolCallStarted"
TOOL_CALL_COMPLETED = "ToolCallCompleted"
TOOL_CALL_FAILED = "ToolCallFailed"
DELEGATION_STARTED = "DelegationStarted"
DELEGATION_COMPLETED = "DelegationCompleted"
DELEGATION_FAILED = "DelegationFailed"
INTERRUPTION_CREATED = "InterruptionCreated"
INTERRUPTION_RESUMED = "InterruptionResumed"
FINAL_RESPONSE_DELIVERED = "FinalResponseDelivered"
ARTIFACT_CREATED = "ArtifactCreated"


def make_event(*, run_id: str, event_type: str, step_id: str | None = None, **payload):
    return RunEventRecord.create(
        run_id=run_id,
        step_id=step_id,
        event_type=event_type,
        payload=payload,
    )
