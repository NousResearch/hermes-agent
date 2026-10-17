"""Shared raw-value contract for dashboard create and execution-control PATCH."""
from typing import Optional

from pydantic import BaseModel, model_validator

from hermes_cli.kanban_db_controls import CONTROL_FIELDS, validate_controls


class ExecutionControlBody(BaseModel):
    goal_mode: bool = False
    goal_max_turns: Optional[int] = None
    max_retries: Optional[int] = None
    max_runtime_seconds: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def raw_execution_controls(cls, values):
        if isinstance(values, dict):
            validate_controls({key: values[key] for key in CONTROL_FIELDS if key in values})
        return values


def control_patch(payload) -> dict:
    fields = payload.model_fields_set
    controls = fields.intersection(CONTROL_FIELDS)
    if controls and fields.difference(CONTROL_FIELDS):
        raise ValueError("execution controls cannot be combined with other edits")
    return {key: getattr(payload, key) for key in CONTROL_FIELDS if key in controls}
