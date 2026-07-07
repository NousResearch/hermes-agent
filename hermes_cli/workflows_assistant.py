"""Validate assistant-authored workflow drafts."""

from __future__ import annotations

import importlib
import json
import re
from typing import Any, Callable

AIAgent = None

from pydantic import BaseModel, Field, ValidationError

from hermes_cli.workflows_capabilities import (
    IMPLEMENTED_NODE_TYPES,
    IMPLEMENTED_TRIGGER_TYPES,
)
from hermes_cli.workflows_spec import WorkflowSpec, validate_graph

_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)
Runner = Callable[[str], str]


class AssistantValidationError(ValueError):
    """Raised when assistant workflow output cannot be accepted."""


class WorkflowDraftResult(BaseModel):
    spec: WorkflowSpec
    summary: str = ""
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)
    valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


def _agent_class() -> Any:
    global AIAgent
    if AIAgent is not None:
        return AIAgent
    try:
        agent_cls = getattr(importlib.import_module("run_agent"), "AIAgent")
    except Exception as exc:
        raise AssistantValidationError("Hermes AIAgent runtime is unavailable") from exc
    AIAgent = agent_cls
    return agent_cls


def _extract_json_object(text: str) -> dict[str, Any]:
    match = _JSON_FENCE_RE.search(text)
    raw = match.group(1) if match else text
    data = json.loads(raw.strip())
    if not isinstance(data, dict):
        raise AssistantValidationError("assistant payload must be a JSON object")
    return data


def _ensure_supported_primitives(spec: WorkflowSpec) -> None:
    errors: list[str] = []
    for trigger in spec.triggers:
        if trigger.type not in IMPLEMENTED_TRIGGER_TYPES:
            errors.append(f"unsupported trigger type {trigger.type}")
    for node_id, node in spec.nodes.items():
        if node.type not in IMPLEMENTED_NODE_TYPES:
            errors.append(f"unsupported node type {node.type} on node {node_id}")
    if errors:
        raise AssistantValidationError("; ".join(errors))


def parse_assistant_payload(payload: dict[str, Any] | str) -> WorkflowDraftResult:
    try:
        data = _extract_json_object(payload) if isinstance(payload, str) else payload
        if not isinstance(data, dict):
            raise AssistantValidationError("assistant payload must be a JSON object")
        if "spec" not in data:
            raise AssistantValidationError("assistant payload requires spec")
        spec = WorkflowSpec.model_validate(data["spec"])
        validate_graph(spec)
        _ensure_supported_primitives(spec)
        return WorkflowDraftResult(
            spec=spec,
            summary=data.get("summary", ""),
            assumptions=data.get("assumptions", []),
            questions=data.get("questions", []),
            warnings=data.get("warnings", []),
            unsupported_requests=data.get("unsupported_requests", []),
        )
    except AssistantValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise AssistantValidationError(f"invalid JSON: {exc.msg}") from exc
    except (ValidationError, ValueError, TypeError) as exc:
        raise AssistantValidationError(str(exc)) from exc


def _json_schema_instruction() -> str:
    return """Return JSON only with this shape:
{
  "summary": "short description of the workflow",
  "assumptions": ["assumption, if any"],
  "questions": ["blocking question, if any"],
  "warnings": ["important warning, if any"],
  "unsupported_requests": ["requested behavior that cannot use implemented primitives"],
  "spec": {
    "id": "lowercase_snake_case_workflow_id",
    "name": "Human-readable workflow name",
    "version": 1,
    "triggers": [{"type": "manual", "id": "manual"}],
    "nodes": {
      "node_id": {"type": "pass", "output": {}}
    },
    "edges": []
  }
}
Do not include Markdown fences or prose outside the JSON object."""


def _assistant_rules() -> str:
    return """Rules:
- Use manual trigger unless schedule requested.
- Allowed triggers: manual, schedule.
- Allowed nodes: pass, switch, agent_task, wait, parallel, join, fail.
- Do not emit webhook, kanban_event, send_message, or subworkflow.
- Every agent_task must include profile, title, and text prompt.
- Every agent_task prompt must ask for JSON-only output and include a result contract.
- Prefer simple graphs, no unrequested flexibility.
- Use lowercase snake_case node ids."""


def build_draft_prompt(goal: str) -> str:
    return "\n\n".join(
        [
            "Draft a Hermes workflow from this goal.",
            _json_schema_instruction(),
            _assistant_rules(),
            "Goal:",
            goal.strip(),
        ]
    )


def build_refine_prompt(spec: WorkflowSpec, instruction: str) -> str:
    spec_json = json.dumps(spec.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True)
    return "\n\n".join(
        [
            "Refine this Hermes workflow spec using the instruction.",
            _json_schema_instruction(),
            _assistant_rules(),
            "Current workflow spec JSON:",
            spec_json,
            "Instruction:",
            instruction.strip(),
        ]
    )


def _resolve_assistant_runtime() -> dict[str, Any]:
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = load_config()
    model_cfg = config.get("model") if isinstance(config, dict) else None
    model = ""
    if isinstance(model_cfg, dict):
        model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    elif isinstance(model_cfg, str):
        model = model_cfg.strip()

    runtime = dict(resolve_runtime_provider(requested=None, target_model=model or None) or {})
    runtime_model = str(runtime.get("model") or "").strip()
    runtime["model"] = runtime_model or model

    return runtime


def default_model_runner(prompt: str) -> str:
    agent_cls = _agent_class()

    runtime = _resolve_assistant_runtime()
    kwargs = {
        "model": runtime.get("model", ""),
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "acp_command": runtime.get("command"),
        "acp_args": runtime.get("args"),
        "credential_pool": runtime.get("credential_pool"),
        "enabled_toolsets": [],
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "platform": "workflow_assistant",
        "max_iterations": 3,
    }

    agent = agent_cls(**kwargs)
    agent.suppress_status_output = True
    result = agent.run_conversation(prompt)
    if isinstance(result, dict) and "final_response" in result:
        return str(result.get("final_response") or "")
    return str(result or "")


def _call_with_repair(prompt: str, runner: Runner, repair_attempts: int) -> WorkflowDraftResult:
    attempts_left = max(0, repair_attempts)
    attempt_prompt = prompt
    while True:
        try:
            return parse_assistant_payload(runner(attempt_prompt))
        except AssistantValidationError as exc:
            if attempts_left <= 0:
                raise
            attempts_left -= 1
            attempt_prompt = "\n\n".join(
                [
                    "The previous workflow draft failed validation.",
                    f"Validation error: {exc}",
                    "Original prompt:",
                    prompt,
                    "Return JSON only with a corrected workflow draft.",
                ]
            )


def draft_workflow(goal: str, *, runner: Runner, repair_attempts: int = 1) -> WorkflowDraftResult:
    if not str(goal or "").strip():
        raise AssistantValidationError("workflow goal is required")
    return _call_with_repair(build_draft_prompt(goal), runner, repair_attempts)


def draft_workflow_with_default_runner(goal: str) -> WorkflowDraftResult:
    return draft_workflow(goal, runner=default_model_runner)


def refine_workflow(
    spec: WorkflowSpec,
    instruction: str,
    *,
    runner: Runner,
    repair_attempts: int = 1,
) -> WorkflowDraftResult:
    if not str(instruction or "").strip():
        raise AssistantValidationError("refine instruction is required")
    return _call_with_repair(build_refine_prompt(spec, instruction), runner, repair_attempts)


def refine_workflow_with_default_runner(spec: WorkflowSpec, instruction: str) -> WorkflowDraftResult:
    return refine_workflow(spec, instruction, runner=default_model_runner)
