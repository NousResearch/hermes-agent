"""Static remote-exposure policy for the Mission Control MCP bridge.

This module is intentionally inert. It defines future remote policy metadata
and validation helpers only; it does not start servers, read secrets, perform
network calls, or execute Mission Control tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class PolicyValidationError(ValueError):
    """Raised when the static Mission Control MCP policy is unsafe."""


FORBIDDEN_REMOTE_TOOL_NAMES: tuple[str, ...] = (
    "send_email",
    "publish_video",
    "activate_payment",
    "delete_files",
    "run_unbounded_codex",
    "run_codex",
    "start_codex",
    "start_worker",
    "start_hermes_run",
    "autonomous_computer_use",
    "browser_control",
    "mouse_control",
    "keyboard_control",
    "start_bulk_outreach",
    "arbitrary_shell",
    "reveal_secret",
    "update_credentials",
)

FORBIDDEN_REMOTE_TOOL_CLASSES: tuple[str, ...] = (
    "shell/terminal",
    "browser/computer-use",
    "email/outreach send",
    "publishing",
    "payment/customer",
    "destructive mutation",
    "credential reveal/update",
    "broad Hermes registry",
    "worker/Codex/Hermes-run dispatch",
)

_ALLOWED_REMOTE_POSTURES = {"disabled", "deferred", "eligible_first", "local_only"}
_ALLOWED_ACCESS_CLASSES = {"read", "write"}
_PACKET_WRITE_TOOLS = {
    "save_next_codex_prompt",
    "import_worker_result",
    "save_block_flag_packet",
}


@dataclass(frozen=True)
class RemoteToolPolicy:
    tool_name: str
    remote_enabled: bool
    remote_posture: str
    required_scope: str
    access: str
    risk_class: str
    audit_required: bool
    redaction_required: bool
    confirmation_required: bool
    rate_limit: Mapping[str, int]
    output_sensitivity: str
    failure_behavior: str
    local_only: bool
    executes_or_dispatches: bool
    exposes_secret_material: bool
    tool_classes: tuple[str, ...] = ("mission_control",)


REMOTE_TOOL_POLICIES: dict[str, RemoteToolPolicy] = {
    "get_project_status": RemoteToolPolicy(
        tool_name="get_project_status",
        remote_enabled=False,
        remote_posture="eligible_first",
        required_scope="mission_control.read.project_status",
        access="read",
        risk_class="medium",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 30, "service_per_minute": 120},
        output_sensitivity="medium",
        failure_behavior="return controlled warning, no partial secrets",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only"),
    ),
    "get_open_tasks": RemoteToolPolicy(
        tool_name="get_open_tasks",
        remote_enabled=False,
        remote_posture="eligible_first",
        required_scope="mission_control.read.tasks",
        access="read",
        risk_class="medium",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 30, "service_per_minute": 120},
        output_sensitivity="medium",
        failure_behavior="return controlled warning",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only"),
    ),
    "get_latest_worker_results": RemoteToolPolicy(
        tool_name="get_latest_worker_results",
        remote_enabled=False,
        remote_posture="eligible_first",
        required_scope="mission_control.read.worker_results",
        access="read",
        risk_class="high",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 15, "service_per_minute": 60},
        output_sensitivity="high",
        failure_behavior="return untrusted-data warning",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only", "untrusted_display_data"),
    ),
    "get_repo_status": RemoteToolPolicy(
        tool_name="get_repo_status",
        remote_enabled=False,
        remote_posture="eligible_first",
        required_scope="mission_control.read.repo_status",
        access="read",
        risk_class="medium",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 30, "service_per_minute": 120},
        output_sensitivity="medium",
        failure_behavior="return not_probed or controlled warning",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only"),
    ),
    "get_approval_gates": RemoteToolPolicy(
        tool_name="get_approval_gates",
        remote_enabled=False,
        remote_posture="eligible_first",
        required_scope="mission_control.read.approvals",
        access="read",
        risk_class="medium",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 30, "service_per_minute": 120},
        output_sensitivity="medium",
        failure_behavior="return controlled warning",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only"),
    ),
    "get_recent_audit_log": RemoteToolPolicy(
        tool_name="get_recent_audit_log",
        remote_enabled=False,
        remote_posture="deferred",
        required_scope="mission_control.read.audit",
        access="read",
        risk_class="high",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 10, "service_per_minute": 30},
        output_sensitivity="high",
        failure_behavior="return bounded redacted page or warning",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only", "audit_read"),
    ),
    "list_mission_packets": RemoteToolPolicy(
        tool_name="list_mission_packets",
        remote_enabled=False,
        remote_posture="deferred",
        required_scope="mission_control.read.packets",
        access="read",
        risk_class="medium-high",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 20, "service_per_minute": 60},
        output_sensitivity="medium-high",
        failure_behavior="return bounded redacted page",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only", "packet_read"),
    ),
    "get_mission_packet": RemoteToolPolicy(
        tool_name="get_mission_packet",
        remote_enabled=False,
        remote_posture="deferred",
        required_scope="mission_control.read.packets",
        access="read",
        risk_class="high",
        audit_required=True,
        redaction_required=True,
        confirmation_required=False,
        rate_limit={"actor_per_minute": 20, "service_per_minute": 60},
        output_sensitivity="high",
        failure_behavior="return redacted not-found or warning",
        local_only=False,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "read_only", "packet_read"),
    ),
    "save_next_codex_prompt": RemoteToolPolicy(
        tool_name="save_next_codex_prompt",
        remote_enabled=False,
        remote_posture="local_only",
        required_scope="mission_control.write.prompt_packet",
        access="write",
        risk_class="high",
        audit_required=True,
        redaction_required=True,
        confirmation_required=True,
        rate_limit={"actor_per_minute": 5, "service_per_minute": 20},
        output_sensitivity="medium-high",
        failure_behavior="reject and audit rejected request",
        local_only=True,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "packet_write", "local_packet_only"),
    ),
    "import_worker_result": RemoteToolPolicy(
        tool_name="import_worker_result",
        remote_enabled=False,
        remote_posture="local_only",
        required_scope="mission_control.write.worker_result_packet",
        access="write",
        risk_class="high",
        audit_required=True,
        redaction_required=True,
        confirmation_required=True,
        rate_limit={"actor_per_minute": 5, "service_per_minute": 20},
        output_sensitivity="high",
        failure_behavior="reject and audit rejected request",
        local_only=True,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "packet_write", "local_packet_only", "untrusted_display_data"),
    ),
    "save_block_flag_packet": RemoteToolPolicy(
        tool_name="save_block_flag_packet",
        remote_enabled=False,
        remote_posture="local_only",
        required_scope="mission_control.write.block_flag_packet",
        access="write",
        risk_class="medium-high",
        audit_required=True,
        redaction_required=True,
        confirmation_required=True,
        rate_limit={"actor_per_minute": 5, "service_per_minute": 20},
        output_sensitivity="medium",
        failure_behavior="reject and audit rejected request",
        local_only=True,
        executes_or_dispatches=False,
        exposes_secret_material=False,
        tool_classes=("mission_control", "packet_write", "local_packet_only"),
    ),
}


def list_remote_policy_tools() -> list[str]:
    return sorted(REMOTE_TOOL_POLICIES)


def get_remote_tool_policy(tool_name: str) -> RemoteToolPolicy:
    try:
        return REMOTE_TOOL_POLICIES[tool_name]
    except KeyError as exc:
        raise PolicyValidationError(f"Unknown remote Mission Control MCP tool: {tool_name}") from exc


def assert_no_forbidden_remote_tools(policies: Mapping[str, RemoteToolPolicy] | None = None) -> None:
    entries = REMOTE_TOOL_POLICIES if policies is None else policies
    forbidden_names = set(FORBIDDEN_REMOTE_TOOL_NAMES)
    forbidden_classes = set(FORBIDDEN_REMOTE_TOOL_CLASSES)
    for key, entry in entries.items():
        if key in forbidden_names or entry.tool_name in forbidden_names:
            raise PolicyValidationError(f"forbidden remote tool in Mission Control MCP policy: {entry.tool_name}")
        blocked_classes = set(entry.tool_classes) & forbidden_classes
        if blocked_classes:
            blocked = ", ".join(sorted(blocked_classes))
            raise PolicyValidationError(f"forbidden remote class in Mission Control MCP policy: {entry.tool_name}: {blocked}")


def validate_remote_policy(policies: Mapping[str, RemoteToolPolicy] | None = None) -> None:
    entries = REMOTE_TOOL_POLICIES if policies is None else policies
    assert_no_forbidden_remote_tools(entries)
    for key, entry in entries.items():
        if key != entry.tool_name:
            raise PolicyValidationError(f"policy key/tool mismatch: {key} != {entry.tool_name}")
        if entry.remote_posture not in _ALLOWED_REMOTE_POSTURES:
            raise PolicyValidationError(f"invalid remote_posture for {entry.tool_name}: {entry.remote_posture}")
        if entry.access not in _ALLOWED_ACCESS_CLASSES:
            raise PolicyValidationError(f"invalid access class for {entry.tool_name}: {entry.access}")
        if not entry.required_scope.startswith("mission_control."):
            raise PolicyValidationError(f"required_scope must start with mission_control.: {entry.tool_name}")
        if not entry.audit_required:
            raise PolicyValidationError(f"audit_required must stay true: {entry.tool_name}")
        if not entry.redaction_required:
            raise PolicyValidationError(f"redaction_required must stay true: {entry.tool_name}")
        if entry.executes_or_dispatches:
            raise PolicyValidationError(f"executes_or_dispatches must stay false: {entry.tool_name}")
        if entry.exposes_secret_material:
            raise PolicyValidationError(f"exposes_secret_material must stay false: {entry.tool_name}")
        if not entry.rate_limit.get("actor_per_minute") or not entry.rate_limit.get("service_per_minute"):
            raise PolicyValidationError(f"rate_limit metadata is required: {entry.tool_name}")
        if entry.access == "write" or entry.tool_name in _PACKET_WRITE_TOOLS:
            if entry.remote_enabled:
                raise PolicyValidationError(f"write tool must stay remote-disabled: {entry.tool_name}")
            if entry.remote_posture != "local_only" or not entry.local_only:
                raise PolicyValidationError(f"write tool must stay local-only: {entry.tool_name}")
            if not entry.confirmation_required:
                raise PolicyValidationError(f"write tool requires confirmation metadata: {entry.tool_name}")
        elif entry.remote_enabled:
            raise PolicyValidationError(f"remote_enabled must stay false: {entry.tool_name}")
