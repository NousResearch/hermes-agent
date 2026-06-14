"""Validated request/response contracts between Hermes OS and Hermes Agent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import VALIDATION_ERROR, AdapterError, adapter_error


AGENT_KINDS = {
    "research",
    "coding",
    "testing",
    "review",
    "documentation",
    "deployment",
    "template",
    "experiment",
}

RUNTIME_PROVIDERS = {
    "official-hermes-agent",
    "deepseek-direct",
    "dry-run",
}

RESPONSE_STATUSES = {
    "completed",
    "failed",
    "timeout",
    "unavailable",
    "dry_run",
}


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    require_approval: bool = True


@dataclass(frozen=True)
class AgentRequest:
    task_id: str
    project_id: str
    agent_kind: str
    prompt: str
    working_directory: str
    context: Dict[str, Any] = field(default_factory=dict)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    runtime_provider: str = "official-hermes-agent"
    timeout_seconds: int = 120
    dry_run: bool = False


@dataclass(frozen=True)
class RuntimeCost:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class AgentResponse:
    task_id: str
    status: str
    output: str = ""
    artifacts: List[str] = field(default_factory=list)
    errors: List[AdapterError] = field(default_factory=list)
    duration_ms: int = 0
    cost: RuntimeCost = field(default_factory=RuntimeCost)
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None


@dataclass(frozen=True)
class DelegationRequest:
    task_id: str
    project_id: str
    task_type: str
    prompt: str
    working_directory: str
    opt_in_runtime: bool = False
    dry_run: bool = True


@dataclass(frozen=True)
class RuntimeStatus:
    available: bool
    provider: str
    version: str = "unknown"
    latency_ms: int = 0
    recent_errors: List[str] = field(default_factory=list)


def validate_agent_request(payload):
    if not isinstance(payload, dict):
        return None, adapter_error(VALIDATION_ERROR, "Agent request must be an object")

    required = ["task_id", "project_id", "agent_kind", "prompt", "working_directory"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        return None, adapter_error(VALIDATION_ERROR, "Missing required fields: " + ", ".join(missing))

    agent_kind = str(payload["agent_kind"])
    if agent_kind not in AGENT_KINDS:
        return None, adapter_error(VALIDATION_ERROR, "Unknown agent kind: " + agent_kind)

    runtime_provider = str(payload.get("runtime_provider", "official-hermes-agent"))
    if runtime_provider not in RUNTIME_PROVIDERS:
        return None, adapter_error(VALIDATION_ERROR, "Unknown runtime provider: " + runtime_provider)

    tool_policy_payload = payload.get("tool_policy") or {}
    if not isinstance(tool_policy_payload, dict):
        return None, adapter_error(VALIDATION_ERROR, "tool_policy must be an object")

    request = AgentRequest(
        task_id=str(payload["task_id"]),
        project_id=str(payload["project_id"]),
        agent_kind=agent_kind,
        prompt=str(payload["prompt"]),
        working_directory=str(payload["working_directory"]),
        context=payload.get("context") if isinstance(payload.get("context"), dict) else {},
        tool_policy=ToolPolicy(
            allowed_tools=[str(item) for item in tool_policy_payload.get("allowed_tools", [])],
            denied_tools=[str(item) for item in tool_policy_payload.get("denied_tools", [])],
            require_approval=bool(tool_policy_payload.get("require_approval", True)),
        ),
        runtime_provider=runtime_provider,
        timeout_seconds=int(payload.get("timeout_seconds", 120)),
        dry_run=bool(payload.get("dry_run", False)),
    )
    return request, None


def validate_agent_response(payload):
    if not isinstance(payload, dict):
        return None, adapter_error(VALIDATION_ERROR, "Agent response must be an object")
    if not payload.get("task_id"):
        return None, adapter_error(VALIDATION_ERROR, "Agent response missing task_id")
    status = str(payload.get("status", ""))
    if status not in RESPONSE_STATUSES:
        return None, adapter_error(VALIDATION_ERROR, "Unknown response status: " + status)
    return AgentResponse(
        task_id=str(payload["task_id"]),
        status=status,
        output=str(payload.get("output", "")),
        artifacts=[str(item) for item in payload.get("artifacts", [])],
        duration_ms=int(payload.get("duration_ms", 0)),
        stdout=str(payload.get("stdout", "")),
        stderr=str(payload.get("stderr", "")),
        exit_code=payload.get("exit_code"),
    ), None
