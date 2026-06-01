from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest


PHASE4_TOOLS = {
    "get_project_status",
    "get_open_tasks",
    "get_latest_worker_results",
    "get_repo_status",
    "get_approval_gates",
    "get_recent_audit_log",
    "list_mission_packets",
    "get_mission_packet",
    "save_next_codex_prompt",
    "import_worker_result",
    "save_block_flag_packet",
}

READ_ONLY_TOOLS = {
    "get_project_status",
    "get_open_tasks",
    "get_latest_worker_results",
    "get_repo_status",
    "get_approval_gates",
    "get_recent_audit_log",
    "list_mission_packets",
    "get_mission_packet",
}

PACKET_WRITE_TOOLS = {
    "save_next_codex_prompt",
    "import_worker_result",
    "save_block_flag_packet",
}

FORBIDDEN_TOOLS = {
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
}

FORBIDDEN_CLASSES = {
    "shell/terminal",
    "browser/computer-use",
    "email/outreach send",
    "publishing",
    "payment/customer",
    "destructive mutation",
    "credential reveal/update",
    "broad Hermes registry",
    "worker/Codex/Hermes-run dispatch",
}


def test_remote_policy_defaults_disable_every_tool():
    from hermes_cli import mission_control_mcp_policy as policy

    policy.validate_remote_policy()

    assert set(policy.list_remote_policy_tools()) == PHASE4_TOOLS
    for entry in policy.REMOTE_TOOL_POLICIES.values():
        assert entry.remote_enabled is False
        assert entry.required_scope.startswith("mission_control.")
        assert entry.audit_required is True
        assert entry.redaction_required is True
        assert entry.executes_or_dispatches is False
        assert entry.exposes_secret_material is False
        assert entry.rate_limit["actor_per_minute"] > 0
        assert entry.rate_limit["service_per_minute"] >= entry.rate_limit["actor_per_minute"]


def test_read_only_tools_are_deferred_or_eligible_but_not_enabled():
    from hermes_cli import mission_control_mcp_policy as policy

    for name in READ_ONLY_TOOLS:
        entry = policy.get_remote_tool_policy(name)
        assert entry.access == "read"
        assert entry.remote_posture in {"deferred", "eligible_first"}
        assert entry.remote_enabled is False
        assert entry.local_only is False
        assert entry.confirmation_required is False


def test_packet_write_tools_are_remote_disabled_local_only():
    from hermes_cli import mission_control_mcp_policy as policy

    for name in PACKET_WRITE_TOOLS:
        entry = policy.get_remote_tool_policy(name)
        assert entry.access == "write"
        assert entry.remote_posture == "local_only"
        assert entry.remote_enabled is False
        assert entry.local_only is True
        assert entry.confirmation_required is True


def test_forbidden_names_and_classes_are_absent_from_policy():
    from hermes_cli import mission_control_mcp_policy as policy

    assert set(policy.FORBIDDEN_REMOTE_TOOL_NAMES).issuperset(FORBIDDEN_TOOLS)
    assert set(policy.FORBIDDEN_REMOTE_TOOL_CLASSES).issuperset(FORBIDDEN_CLASSES)
    assert not (set(policy.list_remote_policy_tools()) & FORBIDDEN_TOOLS)

    for entry in policy.REMOTE_TOOL_POLICIES.values():
        assert not (set(entry.tool_classes) & FORBIDDEN_CLASSES)


def test_policy_tool_set_aligns_with_phase4_local_mcp_allowlist():
    from hermes_cli import mission_control_mcp as mcp
    from hermes_cli import mission_control_mcp_policy as policy

    assert set(policy.list_remote_policy_tools()) == set(mcp.list_tool_names())
    assert not (set(policy.list_remote_policy_tools()) & set(mcp.BLOCKED_TOOL_NAMES))


def test_validate_remote_policy_fails_if_forbidden_tool_is_injected():
    from hermes_cli import mission_control_mcp_policy as policy

    entries = dict(policy.REMOTE_TOOL_POLICIES)
    source = policy.get_remote_tool_policy("get_project_status")
    entries["run_codex"] = dataclasses.replace(source, tool_name="run_codex")

    with pytest.raises(policy.PolicyValidationError, match="forbidden remote tool"):
        policy.validate_remote_policy(entries)


def test_validate_remote_policy_fails_if_forbidden_class_is_injected():
    from hermes_cli import mission_control_mcp_policy as policy

    entries = dict(policy.REMOTE_TOOL_POLICIES)
    source = policy.get_remote_tool_policy("get_project_status")
    entries[source.tool_name] = dataclasses.replace(source, tool_classes=("shell/terminal",))

    with pytest.raises(policy.PolicyValidationError, match="forbidden remote class"):
        policy.validate_remote_policy(entries)


def test_validate_remote_policy_fails_if_write_tool_is_remote_enabled():
    from hermes_cli import mission_control_mcp_policy as policy

    entries = dict(policy.REMOTE_TOOL_POLICIES)
    source = policy.get_remote_tool_policy("save_next_codex_prompt")
    entries[source.tool_name] = dataclasses.replace(source, remote_enabled=True)

    with pytest.raises(policy.PolicyValidationError, match="write tool must stay remote-disabled"):
        policy.validate_remote_policy(entries)


def test_validate_remote_policy_fails_if_audit_or_redaction_is_disabled():
    from hermes_cli import mission_control_mcp_policy as policy

    source = policy.get_remote_tool_policy("get_project_status")

    entries = dict(policy.REMOTE_TOOL_POLICIES)
    entries[source.tool_name] = dataclasses.replace(source, audit_required=False)
    with pytest.raises(policy.PolicyValidationError, match="audit_required"):
        policy.validate_remote_policy(entries)

    entries = dict(policy.REMOTE_TOOL_POLICIES)
    entries[source.tool_name] = dataclasses.replace(source, redaction_required=False)
    with pytest.raises(policy.PolicyValidationError, match="redaction_required"):
        policy.validate_remote_policy(entries)


def test_policy_module_does_not_introduce_remote_transport_or_oauth_server_symbols():
    source = Path("hermes_cli/mission_control_mcp_policy.py").read_text(encoding="utf-8")

    forbidden_symbols = {
        "FastAPI",
        "APIRouter",
        "FastMCP",
        "uvicorn",
        "@app.",
        "add_api_route",
        "websocket",
        "streamable-http",
        "sse_transport",
        "authorization_url",
        "token_endpoint",
    }
    for symbol in forbidden_symbols:
        assert symbol not in source
