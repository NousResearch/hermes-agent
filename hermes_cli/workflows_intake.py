"""Minimal intake readiness checks for workflow trigger input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hermes_cli.workflows_expr import eval_condition
from hermes_cli.workflows_spec import TriggerSpec


@dataclass(frozen=True)
class IntakeEvaluation:
    ready: bool
    status: str
    messages: list[str]
    criteria: dict[str, Any]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False


def evaluate_intake(trigger: TriggerSpec, input_data: dict[str, Any]) -> IntakeEvaluation:
    messages: list[str] = []
    criteria: dict[str, Any] = {"fields": {}, "messages": messages}

    for name, field in trigger.input_schema.items():
        value = input_data.get(name, field.default)
        field_ok = True
        if field.required and _is_missing(value):
            messages.append(f"{name} is required")
            field_ok = False
        if isinstance(value, str):
            if field.min_length is not None and len(value) < field.min_length:
                messages.append(f"{name} must be at least {field.min_length} characters")
                field_ok = False
            if field.max_length is not None and len(value) > field.max_length:
                messages.append(f"{name} must be at most {field.max_length} characters")
                field_ok = False
            if field.max_bytes is not None and len(value.encode("utf-8")) > field.max_bytes:
                messages.append(f"{name} must be at most {field.max_bytes} bytes")
                field_ok = False
        if field.kind in {"number", "integer"} and not isinstance(value, bool):
            try:
                number = float(value)
            except (TypeError, ValueError):
                if not _is_missing(value):
                    messages.append(f"{name} must be a number")
                    field_ok = False
            else:
                if field.kind == "integer" and not number.is_integer():
                    messages.append(f"{name} must be an integer")
                    field_ok = False
                if field.min is not None and number < field.min:
                    messages.append(f"{name} must be at least {field.min:g}")
                    field_ok = False
                if field.max is not None and number > field.max:
                    messages.append(f"{name} must be at most {field.max:g}")
                    field_ok = False
        criteria["fields"][name] = field_ok

    ready_when = trigger.intake.ready_when
    if ready_when:
        try:
            ready = eval_condition(ready_when, {"input": input_data})
        except ValueError as exc:
            ready = False
            messages.append(f"ready_when invalid: {exc}")
        criteria["ready_when"] = ready
        if not ready and not messages:
            messages.append("ready_when is false")

    ready = not messages
    return IntakeEvaluation(
        ready=ready,
        status="queued" if ready else "needs_input",
        messages=messages,
        criteria=criteria,
    )
