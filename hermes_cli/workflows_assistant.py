"""Validate assistant-authored workflow drafts."""

from __future__ import annotations

import importlib
import json
import re
from typing import Any, Callable

AIAgent = None

from pydantic import BaseModel, Field, ValidationError

from hermes_cli.workflows_capabilities import require_implemented_primitives
from hermes_cli.workflows_spec import (
    WorkflowSpec,
    reject_unknown_spec_fields,
    validate_graph,
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
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


def _decode_json_dict(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    fenced = [match.group(1) for match in _JSON_FENCE_RE.finditer(text)]
    if fenced:
        return [data for raw in fenced if (data := _decode_json_dict(raw)) is not None]

    data = _decode_json_dict(text)
    if data is not None:
        return [data]

    found: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        idx = text.find("{", idx)
        if idx == -1:
            break
        try:
            data, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            # The first JSON-looking object is malformed. Do not keep scanning into
            # its children and accidentally accept a nested draft from a broken
            # wrapper object.
            if not found:
                break
            idx += 1
            continue
        if isinstance(data, dict):
            found.append(data)
        idx += max(end, 1)
    return found


def _ensure_supported_primitives(spec: WorkflowSpec) -> None:
    try:
        require_implemented_primitives(spec)
    except ValueError as exc:
        raise AssistantValidationError(str(exc)) from exc


def _ensure_agent_task_contracts(spec: WorkflowSpec) -> None:
    errors = [
        f"agent_task node {node_id} requires a non-empty result_contract"
        for node_id, node in spec.nodes.items()
        if node.type == "agent_task" and not node.result_contract
    ]
    if errors:
        raise AssistantValidationError("; ".join(errors))


def _parse_assistant_dict(data: dict[str, Any]) -> WorkflowDraftResult:
    try:
        if "spec" not in data:
            raise AssistantValidationError("assistant payload requires spec")
        reject_unknown_spec_fields(data["spec"])
        spec = WorkflowSpec.model_validate(data["spec"])
        validate_graph(spec)
        _ensure_supported_primitives(spec)
        _ensure_agent_task_contracts(spec)
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
    except (ValidationError, ValueError, TypeError) as exc:
        raise AssistantValidationError(str(exc)) from exc


def parse_assistant_payload(payload: dict[str, Any] | str) -> WorkflowDraftResult:
    try:
        if isinstance(payload, str):
            candidates = _json_object_candidates(payload)
            if not candidates:
                raise AssistantValidationError("invalid JSON: assistant response did not contain a JSON object")
            last_error: AssistantValidationError | None = None
            for data in reversed(candidates):
                try:
                    return _parse_assistant_dict(data)
                except AssistantValidationError as exc:
                    last_error = exc
            raise last_error or AssistantValidationError("assistant payload requires spec")
        if not isinstance(payload, dict):
            raise AssistantValidationError("assistant payload must be a JSON object")
        return _parse_assistant_dict(payload)
    except AssistantValidationError:
        raise
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
      "agent_node": {"type": "agent_task", "profile": "worker", "title": "Do work", "prompt": "Return JSON only with keys: summary (string), status (string).", "result_contract": {"summary": "string", "status": "string"}},
      "done": {"type": "pass", "output": {}}
    },
    "edges": [
      {"from": "agent_node", "to": "done"}
    ]
  }
}
IMPORTANT: Each edge MUST be a JSON object with "from" and "to" keys, NOT a string. For switch/parallel branches use dot notation in the "from" field: {"from": "switch_id.case_name", "to": "target_node"}.
For manual workflows that need user data at start, declare fields in triggers[].input and reference them in agent prompts with ${ input.field_name }. Example for repository workflows: {"type": "manual", "id": "manual", "input": {"repo_path": {"type": "string", "description": "Path to the repository to inspect"}}}; then prompts should reference ${ input.repo_path }.
Optional agent_task routing fields, only when the user explicitly asks for provider/model routing: "provider": "provider-slug", "model": "model-name".
Do not include Markdown fences or prose outside the JSON object."""


def _workflow_patterns() -> str:
    return """Workflow logic patterns (use these when the user's goal requires them):

1. LINEAR CHAIN (default): trigger -> agent_task -> pass -> done
   Use when the goal is a simple sequence with no decision points.
   Example: "fetch data, summarize it, send it"

2. CONDITIONAL BRANCH (switch): Use a switch node when the goal has "if", "when", "or", "depending on", or any decision point.
   The switch node has cases (each with a name) and a default. Each case routes to a different downstream node via edges like {"from": "switch_id.case_name", "to": "target"}.
   The switch node's "default" field names a fallback node when no case matches.
   Example goal: "review code, then deploy if approved, or report why not"
   Pattern:
     nodes: { "review": {type: agent_task, ...}, "decide": {type: switch, cases: [{name: "approved"}, {name: "rejected"}], default: "report"}, "deploy": {type: agent_task, ...}, "report": {type: agent_task, ...}, "done": {type: pass} }
     edges: [{"from": "review", "to": "decide"}, {"from": "decide.approved", "to": "deploy"}, {"from": "decide.rejected", "to": "report"}, {"from": "deploy", "to": "done"}, {"from": "report", "to": "done"}]
   Key: the switch node evaluates its cases and routes to exactly one branch. The "default" field is the fallback target node id. Every case target and default target must be an existing node id.

   README parity example: manual trigger input should include repo_path; review reads ${ input.repo_path }; decide has cases [{name: "needs_update"}, {name: "no_update"}] and default "no_changes"; nodes must include review, decide, update_readme, no_changes, done; edges must include review -> decide, decide.needs_update -> update_readme, decide.no_update -> no_changes, update_readme -> done, no_changes -> done.

3. PARALLEL FAN-OUT + JOIN: Use parallel + join when the goal has "and also", "in parallel", "simultaneously", or independent workstreams that converge.
   The parallel node requires branch-suffixed edges: {"from": "parallel_id.branch_name", "to": "target"}.
   The join node waits for all incoming branches before continuing.
   Example goal: "run tests and lint in parallel, then report combined results"
   Pattern:
     nodes: { "fanout": {type: parallel}, "tests": {type: agent_task, ...}, "lint": {type: agent_task, ...}, "collect": {type: join}, "report": {type: pass} }
     edges: [{"from": "fanout.tests", "to": "tests"}, {"from": "fanout.lint", "to": "lint"}, {"from": "tests", "to": "collect"}, {"from": "lint", "to": "collect"}, {"from": "collect", "to": "report"}]

4. ERROR HANDLING (catch): Any node can have a "catch" field naming a fallback node. If the node fails, execution routes to the catch target instead of aborting.
   Example: { "review": {type: agent_task, catch: "handle_error", ...}, "handle_error": {type: fail, output: {message: "review failed"}} }

5. RETRY: agent_task nodes support "retry": {"max_attempts": 3, "delay_seconds": 5, "backoff_seconds": 10, "multiplier": 2} for transient failures.

CRITICAL: When the user's goal contains any of these words, you MUST use the corresponding pattern:
- "if", "when", "unless", "depending on", "whether", "or" -> switch node with cases
- "in parallel", "simultaneously", "also", "both", "and independently" -> parallel + join
- "retry", "try again" -> retry spec on agent_task
- "on error", "if it fails", "fallback" -> catch field
Do NOT produce a linear chain when the goal describes branching or conditional outcomes."""


def _assistant_rules() -> str:
    return """Rules:
- Use manual trigger unless schedule requested.
- Allowed triggers: manual, schedule.
- Allowed nodes: pass, switch, agent_task, wait, parallel, join, fail.
- Do not emit webhook, kanban_event, send_message, or subworkflow.
- Every agent_task must include profile, title, text prompt, and result_contract with required downstream keys.
- Every agent_task prompt must ask for JSON-only output matching its result_contract.
- Manual workflows that require user-specific data (repo path, file path, URL, issue id, branch name, prompt text, destination, etc.) must declare those fields in the manual trigger's input object and reference them in prompts as ${ input.field_name }. Do not hide required run-start data inside assumptions.
- Use provider/model only when requested: only include provider and model fields when the user explicitly asks for provider/model routing; otherwise omit them so the profile defaults apply.
- If provider/model routing is requested, set provider and model on each affected agent_task cell independently.
- Use the workflow logic patterns above. Match the pattern to the user's goal structure, not always a linear chain.
- When the goal has a decision point ("if X then Y else Z"), use a switch node with named cases and edges using "switch_id.case_name -> target". The switch default and every case edge target must name an existing node.
- When the goal has independent parallel work, use parallel + join nodes with "parallel_id.branch_name -> target" edges.
- Prefer complete end-to-end flows: every branch must reach a terminal node (pass or fail). Do not leave dangling branches.
- Use lowercase snake_case node ids."""


def build_draft_prompt(goal: str) -> str:
    return "\n\n".join(
        [
            "Draft a Hermes workflow from this goal.",
            _json_schema_instruction(),
            _workflow_patterns(),
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
            _workflow_patterns(),
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
        "suppress_status_output": True,
    }

    agent = agent_cls(**kwargs)
    result = agent.run_conversation(prompt)
    if isinstance(result, dict) and "final_response" in result:
        return str(result.get("final_response") or "")
    return str(result or "")


def _validation_error_summary(error: AssistantValidationError) -> str:
    text = str(error)
    if "invalid JSON" in text:
        return "invalid JSON"
    if "unknown field" in text:
        return "unknown workflow field"
    if "unsupported" in text:
        return "unsupported workflow primitive"
    if "requires a non-empty result_contract" in text:
        return "missing agent_task result_contract"
    if "workflow must define at least one node" in text:
        return "workflow has no nodes"
    if "Field required" in text or "field required" in text:
        return "missing required schema field"
    return "schema or graph validation failed"


def _build_repair_prompt(
    original_prompt: str,
    previous_output: str,
    error: AssistantValidationError,
) -> str:
    return "\n\n".join(
        [
            original_prompt,
            "Previous assistant output failed validation.",
            f"Validation error summary: {_validation_error_summary(error)}",
            "Return a corrected JSON object only. Do not include Markdown fences or commentary.",
            "Previous assistant output:",
            previous_output[:6000],
        ]
    )


def _call_with_repair(prompt: str, runner: Runner, repair_attempts: int) -> WorkflowDraftResult:
    attempts = max(0, int(repair_attempts or 0))
    current_prompt = prompt
    last_error: AssistantValidationError | None = None

    for attempt in range(attempts + 1):
        output = runner(current_prompt)
        try:
            return parse_assistant_payload(output)
        except AssistantValidationError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            current_prompt = _build_repair_prompt(prompt, output, exc)

    assert last_error is not None
    raise AssistantValidationError(
        "assistant draft failed validation "
        f"({_validation_error_summary(last_error)}); revise the request or workflow and retry"
    ) from last_error


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
