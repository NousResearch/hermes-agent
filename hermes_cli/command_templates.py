from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.task_contracts import (
    ORCHESTRATION_HINTS_SCHEMA,
    ORCHESTRATION_HINTS_VERSION,
    TaskContract,
    build_named_workflow_artifact,
    validate_orchestration_hints,
    validate_task_contract,
)


WORK_STARTING_COMMANDS: frozenset[str] = frozenset(
    {
        "handoff",
        "init-deep",
        "start-work",
        "ralph-loop",
        "ulw-loop",
    }
)


@dataclass(frozen=True)
class CommandInvocation:
    command_name: str
    raw_args: str
    task_contract: dict[str, Any]
    named_workflow: dict[str, Any] | None
    orchestration_hints: dict[str, Any]
    prompt_text: str


@dataclass(frozen=True)
class CommandTemplate:
    name: str
    summary: str
    default_request: str
    contract_builder: Callable[..., dict[str, Any]]
    hints_builder: Callable[..., dict[str, Any]]


_GENERIC_SKILLS = ["repo-navigation", "implementation", "verification"]
_GENERIC_TOOLS = ["read_file", "search_files", "patch", "terminal"]


def _normalize_request(raw_args: str, default_request: str) -> str:
    text = str(raw_args or "").strip()
    return text or default_request


def _parse_explicit_task_contract(raw_args: str) -> dict[str, Any] | None:
    text = str(raw_args or "").strip()
    if not text or text[0] not in "[{":
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("expected a JSON object")
    return payload


def _request_text_for_contract(
    raw_args: str,
    default_request: str,
    *,
    explicit_contract: dict[str, Any] | None = None,
) -> str:
    if explicit_contract is not None:
        return str(explicit_contract.get("task") or "").strip() or default_request
    return _normalize_request(raw_args, default_request)


def _validated_explicit_contract(explicit_contract: dict[str, Any]) -> dict[str, Any]:
    return validate_task_contract(explicit_contract).model_dump()


def _attach_explicit_contract_metadata(
    hints: dict[str, Any],
    *,
    command_name: str,
    session_id: str | None,
    cwd: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "command": command_name,
        "session_id": session_id,
        "cwd": cwd,
        "input_mode": "explicit_json_contract",
        "preserve_exact_task_contract": True,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    hints["invocation_metadata"] = metadata
    return validate_orchestration_hints(hints).model_dump()


def _base_contract(
    *,
    command_name: str,
    request_text: str,
    session_id: str | None,
    cwd: str | None,
    expected_outcome: str,
    must_do: list[Any],
    must_not_do: list[Any],
    required_skills: list[str] | None = None,
    required_tools: list[str] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = {
        "command": command_name,
        "request": request_text,
        "session_id": session_id,
        "cwd": cwd,
    }
    if extra_context:
        context.update(extra_context)
    return validate_task_contract(
        {
            "task": request_text,
            "expected_outcome": expected_outcome,
            "required_skills": list(required_skills or _GENERIC_SKILLS),
            "required_tools": list(required_tools or _GENERIC_TOOLS),
            "must_do": must_do,
            "must_not_do": must_not_do,
            "context": context,
        }
    ).model_dump()


def _base_hints(*, command_name: str, request_text: str, loop_style: str = "single_pass") -> dict[str, Any]:
    return validate_orchestration_hints(
        {
            "schema": ORCHESTRATION_HINTS_SCHEMA,
            "schema_version": ORCHESTRATION_HINTS_VERSION,
            "command": command_name,
            "loop_style": loop_style,
            "request": request_text,
            "bounded_context": {
                "enabled": True,
                "max_hermes_hierarchy_files": 3,
                "task_contract_precedence": "preserve_existing_fields",
            },
        }
    ).model_dump()


def _handoff_contract(*, raw_args: str, session_id: str | None, cwd: str | None) -> dict[str, Any]:
    raw_args = str(raw_args or "").strip()
    explicit_contract = _parse_explicit_task_contract(raw_args)
    request_text = _request_text_for_contract(
        raw_args,
        "Continue the handed-off work from the current session state.",
        explicit_contract=explicit_contract,
    )
    if explicit_contract is not None:
        return _validated_explicit_contract(explicit_contract)
    return _base_contract(
        command_name="handoff",
        request_text=request_text,
        session_id=session_id,
        cwd=cwd,
        expected_outcome="A concise takeover that continues the active task without dropping the prior contract.",
        must_do=[
            "inspect the current conversation and workspace before acting",
            {"handoff": ["preserve active scope", "report what was resumed"]},
        ],
        must_not_do=[
            "do not discard or rewrite an explicit structured task contract",
            "do not widen scope beyond the handed-off task",
        ],
        extra_context={"handoff_mode": "resume_or_consume_contract"},
    )


def _handoff_hints(*, raw_args: str, session_id: str | None, cwd: str | None) -> dict[str, Any]:
    explicit_contract = _parse_explicit_task_contract(raw_args)
    request_text = _request_text_for_contract(
        raw_args,
        "Continue the handed-off work from the current session state.",
        explicit_contract=explicit_contract,
    )
    hints = _base_hints(command_name="handoff", request_text=request_text)
    hints["handoff"] = {"consume_existing_contract": True, "announce_resumption": True}
    if explicit_contract is not None:
        return _attach_explicit_contract_metadata(
            hints,
            command_name="handoff",
            session_id=session_id,
            cwd=cwd,
            extra_metadata={"handoff_mode": "resume_or_consume_contract"},
        )
    return hints


def _start_work_contract(*, raw_args: str, session_id: str | None, cwd: str | None) -> dict[str, Any]:
    explicit_contract = _parse_explicit_task_contract(raw_args)
    request_text = _request_text_for_contract(
        raw_args,
        "Start the requested work from the current session context.",
        explicit_contract=explicit_contract,
    )
    if explicit_contract is not None:
        return _validated_explicit_contract(explicit_contract)
    return _base_contract(
        command_name="start-work",
        request_text=request_text,
        session_id=session_id,
        cwd=cwd,
        expected_outcome="A concrete work start with scoped execution and fresh verification evidence.",
        must_do=[
            "identify the exact target before editing",
            {"verification": ["run focused checks", "report exact outputs"]},
        ],
        must_not_do=[
            "do not skip prerequisite inspection",
            "do not mutate canonical Wave 1 fields through prose-only instructions",
        ],
    )


def _start_work_hints(*, raw_args: str, session_id: str | None, cwd: str | None) -> dict[str, Any]:
    explicit_contract = _parse_explicit_task_contract(raw_args)
    request_text = _request_text_for_contract(
        raw_args,
        "Start the requested work from the current session context.",
        explicit_contract=explicit_contract,
    )
    hints = _base_hints(command_name="start-work", request_text=request_text)
    hints["work_start"] = {"emit_contract_ack": True}
    if explicit_contract is not None:
        return _attach_explicit_contract_metadata(
            hints,
            command_name="start-work",
            session_id=session_id,
            cwd=cwd,
        )
    return hints


def _loop_contract(*, command_name: str, raw_args: str, session_id: str | None, cwd: str | None) -> dict[str, Any]:
    explicit_contract = _parse_explicit_task_contract(raw_args)
    request_text = _request_text_for_contract(
        raw_args,
        f"Run the {command_name} workflow for the current objective.",
        explicit_contract=explicit_contract,
    )
    if explicit_contract is not None:
        return _validated_explicit_contract(explicit_contract)
    return _base_contract(
        command_name=command_name,
        request_text=request_text,
        session_id=session_id,
        cwd=cwd,
        expected_outcome=f"A bounded {command_name} execution loop with explicit progress and completion criteria.",
        must_do=[
            "keep the loop bounded and deterministic",
            {"progress": ["state what remains", "stop when completion criteria are met"]},
        ],
        must_not_do=[
            "do not recurse indefinitely",
            "do not overwrite preserved task-contract context fields",
        ],
        extra_context={
            "loop_family": command_name,
            "command_runtime": {
                "command_name": command_name,
                "runtime_mode": "ralph" if command_name == "ralph-loop" else "ultrawork",
                "continuation_semantics": (
                    {
                        "retry_on_failed_or_interrupted": True,
                        "stop_requires_explicit_exit": True,
                    }
                    if command_name == "ralph-loop"
                    else {
                        "completion_gate": "open_todos_block_done",
                        "require_open_work_closure": True,
                    }
                ),
            },
        },
    )


def _loop_hints(*, command_name: str, raw_args: str, session_id: str | None, cwd: str | None) -> dict[str, Any]:
    explicit_contract = _parse_explicit_task_contract(raw_args)
    request_text = _request_text_for_contract(
        raw_args,
        f"Run the {command_name} workflow for the current objective.",
        explicit_contract=explicit_contract,
    )
    hints = _base_hints(command_name=command_name, request_text=request_text, loop_style=command_name.replace("-", "_"))
    hints["loop"] = {
        "family": command_name,
        "bounded": True,
        "continuation_semantics": (
            {
                "retry_on_failed_or_interrupted": True,
                "stop_requires_explicit_exit": True,
            }
            if command_name == "ralph-loop"
            else {
                "completion_gate": "open_todos_block_done",
                "require_open_work_closure": True,
            }
        ),
    }
    if explicit_contract is not None:
        return _attach_explicit_contract_metadata(
            hints,
            command_name=command_name,
            session_id=session_id,
            cwd=cwd,
            extra_metadata={"loop_family": command_name},
        )
    return hints


def render_command_prompt(invocation: CommandInvocation) -> str:
    lines = [
        f"[OMO command {invocation.command_name}]",
        "Use the following structured external input as authoritative command context.",
        "TASK_CONTRACT_JSON:",
        json.dumps(invocation.task_contract, indent=2, ensure_ascii=False),
        "ORCHESTRATION_HINTS_JSON:",
        json.dumps(invocation.orchestration_hints, indent=2, ensure_ascii=False),
    ]
    if invocation.named_workflow:
        lines.extend(
            [
                "NAMED_WORKFLOW_JSON:",
                json.dumps(invocation.named_workflow, indent=2, ensure_ascii=False),
            ]
        )
    lines.append(
        f"USER_REQUEST: {invocation.orchestration_hints.get('request') or invocation.task_contract.get('task', '')}"
    )
    return "\n".join(lines)


def _build_invocation(template: CommandTemplate, *, raw_args: str, session_id: str | None, cwd: str | None) -> CommandInvocation:
    task_contract = template.contract_builder(raw_args=raw_args, session_id=session_id, cwd=cwd)
    orchestration_hints = template.hints_builder(raw_args=raw_args, session_id=session_id, cwd=cwd)
    named_workflow = None
    invocation_metadata = orchestration_hints.get("invocation_metadata") if isinstance(orchestration_hints, dict) else None
    explicit_contract_supplied = (
        isinstance(invocation_metadata, dict)
        and invocation_metadata.get("input_mode") == "explicit_json_contract"
    )
    if template.name == "start-work" and not explicit_contract_supplied:
        named_workflow = build_named_workflow_artifact(
            objective=str(task_contract.get("task") or orchestration_hints.get("request") or "").strip(),
            specialist=None,
            archetype="generalist",
            route_category="deep",
            runtime_mode="default",
            delegation_profile="implementation",
            task_contract=task_contract,
        )
    invocation = CommandInvocation(
        command_name=template.name,
        raw_args=str(raw_args or "").strip(),
        task_contract=task_contract,
        named_workflow=named_workflow,
        orchestration_hints=orchestration_hints,
        prompt_text="",
    )
    return CommandInvocation(
        command_name=invocation.command_name,
        raw_args=invocation.raw_args,
        task_contract=invocation.task_contract,
        named_workflow=invocation.named_workflow,
        orchestration_hints=invocation.orchestration_hints,
        prompt_text=render_command_prompt(invocation),
    )


def _template_registry() -> dict[str, CommandTemplate]:
    from agent.init_deep import build_init_deep_task_contract, build_init_deep_hints

    return {
        "handoff": CommandTemplate(
            name="handoff",
            summary="Resume handed-off work using a structured task contract.",
            default_request="Continue the handed-off work from the current session state.",
            contract_builder=_handoff_contract,
            hints_builder=_handoff_hints,
        ),
        "init-deep": CommandTemplate(
            name="init-deep",
            summary="Start a deep initialization pass with repo and task triage.",
            default_request="Perform deep initialization for the current task.",
            contract_builder=build_init_deep_task_contract,
            hints_builder=build_init_deep_hints,
        ),
        "start-work": CommandTemplate(
            name="start-work",
            summary="Start execution using a structured work contract.",
            default_request="Start the requested work from the current session context.",
            contract_builder=_start_work_contract,
            hints_builder=_start_work_hints,
        ),
        "ralph-loop": CommandTemplate(
            name="ralph-loop",
            summary="Run the Ralph loop with bounded progress state.",
            default_request="Run the ralph-loop workflow for the current objective.",
            contract_builder=lambda **kwargs: _loop_contract(command_name="ralph-loop", **kwargs),
            hints_builder=lambda **kwargs: _loop_hints(command_name="ralph-loop", **kwargs),
        ),
        "ulw-loop": CommandTemplate(
            name="ulw-loop",
            summary="Run the ULW loop with bounded progress state.",
            default_request="Run the ulw-loop workflow for the current objective.",
            contract_builder=lambda **kwargs: _loop_contract(command_name="ulw-loop", **kwargs),
            hints_builder=lambda **kwargs: _loop_hints(command_name="ulw-loop", **kwargs),
        ),
    }


COMMAND_TEMPLATES = _template_registry()


def resolve_work_command_template(command_name: str | None) -> CommandTemplate | None:
    name = str(command_name or "").strip().lower().lstrip("/")
    return COMMAND_TEMPLATES.get(name)


def build_command_invocation(
    command_name: str,
    *,
    raw_args: str = "",
    session_id: str | None = None,
    cwd: str | None = None,
) -> CommandInvocation:
    template = resolve_work_command_template(command_name)
    if template is None:
        raise KeyError(f"Unknown OMO work command: {command_name}")
    normalized_cwd = str(Path(cwd).resolve()) if cwd else None
    return _build_invocation(template, raw_args=raw_args, session_id=session_id, cwd=normalized_cwd)


__all__ = [
    "COMMAND_TEMPLATES",
    "CommandInvocation",
    "CommandTemplate",
    "WORK_STARTING_COMMANDS",
    "build_command_invocation",
    "render_command_prompt",
    "resolve_work_command_template",
]
