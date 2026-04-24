"""Validation helpers for the Codex Bridge skill CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


ALLOWED_SANDBOXES = {"read-only", "workspace-write"}
ALLOWED_APPROVAL_POLICIES = {"untrusted", "on-request"}
ALLOWED_DECISIONS = {"accept", "acceptForSession", "decline", "cancel"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
SMOKE_SENTINEL = "CODEX_ASYNC_OK"


class ValidationError(ValueError):
    """Raised when a CLI input or bridge output fails validation."""


def parse_json_object(value: str | None, *, field_name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{field_name} must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValidationError(f"{field_name} must be a JSON object.")
    return parsed


def validate_sandbox(sandbox: str) -> str:
    if sandbox == "danger-full-access":
        raise ValidationError("danger-full-access is not allowed for Codex Bridge.")
    if sandbox not in ALLOWED_SANDBOXES:
        allowed = ", ".join(sorted(ALLOWED_SANDBOXES))
        raise ValidationError(f"sandbox must be one of: {allowed}.")
    return sandbox


def validate_approval_policy(approval_policy: str) -> str:
    if approval_policy not in ALLOWED_APPROVAL_POLICIES:
        allowed = ", ".join(sorted(ALLOWED_APPROVAL_POLICIES))
        raise ValidationError(f"approval_policy must be one of: {allowed}.")
    return approval_policy


def validate_start_input(prompt: str, cwd: str, sandbox: str, approval_policy: str) -> None:
    if not prompt or not prompt.strip():
        raise ValidationError("start prompt must be non-empty.")
    cwd_path = Path(cwd).expanduser()
    if not cwd_path.exists() or not cwd_path.is_dir():
        raise ValidationError(f"cwd must be an existing directory: {cwd}")
    validate_sandbox(sandbox)
    validate_approval_policy(approval_policy)


def validate_task_id(action: str, task_id: str | None) -> None:
    if not task_id or not str(task_id).strip():
        raise ValidationError(f"{action} requires task_id.")


def validate_steer_input(task_id: str | None, instruction: str | None) -> None:
    validate_task_id("steer", task_id)
    if not instruction or not instruction.strip():
        raise ValidationError("steer requires instruction.")


def validate_interrupt_input(task_id: str | None) -> None:
    validate_task_id("interrupt", task_id)


def validate_status_input(task_id: str | None) -> None:
    validate_task_id("status", task_id)


def validate_respond_input(
    task_id: str | None,
    request_id: str | None,
    decision: str,
    answers: Mapping[str, Any] | None,
) -> None:
    validate_task_id("respond", task_id)
    if not request_id or not str(request_id).strip():
        raise ValidationError("respond requires request_id.")
    if decision not in ALLOWED_DECISIONS:
        allowed = ", ".join(sorted(ALLOWED_DECISIONS))
        raise ValidationError(f"decision must be one of: {allowed}.")
    if answers is not None and not isinstance(answers, Mapping):
        raise ValidationError("answers must be a JSON object.")


def validate_start_output(data: Mapping[str, Any]) -> None:
    if data.get("success") is not True:
        raise ValidationError("start output must have success=true.")
    protocol = data.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValidationError("start output must include protocol.")
    if protocol.get("mailbox") is not False:
        raise ValidationError("start output must have protocol.mailbox=false.")
    transport = str(protocol.get("transport") or "")
    if "app-server" not in transport:
        raise ValidationError("start output protocol.transport must include app-server.")
    task = data.get("task")
    if not isinstance(task, Mapping):
        raise ValidationError("start output must include task.")
    required = {
        "hermes_task_id": "task id",
        "codex_thread_id": "thread id",
        "codex_turn_id": "turn id",
    }
    for key, label in required.items():
        if not task.get(key):
            raise ValidationError(f"start output missing {label}.")


def validate_bridge_output(action: str, data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping):
        raise ValidationError("bridge output must be a JSON object.")
    if data.get("success") is not True and data.get("error"):
        raise ValidationError(str(data["error"]))
    if action == "start":
        validate_start_output(data)
        return
    if "success" in data and data.get("success") is not True:
        raise ValidationError(str(data.get("error") or f"{action} failed."))


def contains_text(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(contains_text(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(contains_text(v, needle) for v in value)
    return False


def validate_smoke_test_result(status_data: Mapping[str, Any]) -> None:
    task = status_data.get("task")
    if not isinstance(task, Mapping):
        raise ValidationError("smoke-test status output must include task.")
    status = task.get("status")
    if status != "completed":
        raise ValidationError(f"smoke-test final status must be completed, got {status!r}.")
    searchable = {
        "recent_events": task.get("recent_events", []),
        "final_summary": task.get("final_summary"),
    }
    if not contains_text(searchable, SMOKE_SENTINEL):
        raise ValidationError(f"smoke-test output did not include {SMOKE_SENTINEL}.")
