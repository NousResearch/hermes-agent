"""Tests for concrete AgentCyber routing guard and S0-S5 gates."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.cyber_policy import (
    evaluate_execution_gate,
    gate_tool_call_for_agent,
    load_asset_registry,
)
from agent.cyber_breakglass import BreakGlassStore
from agent.cyber_routing import (
    CyberRoute,
    ProviderPreference,
    classify_cyber_route,
    agentcyber_routing_action,
    apply_agentcyber_route_guard,
    is_local_or_open_weight_runtime,
    restore_agentcyber_route_runtime,
)


def test_builtin_bc_asset_registry_authorizes_lab_and_public_assets():
    registry = load_asset_registry({"agent_cyber": {"include_builtin_bc_assets": True}})

    assert registry.matches_any(["192.168.1.120"], gate="S2")
    assert registry.matches_any(["bde.it.com"], gate="S2")
    assert registry.matches_any(["api.breakingcircuits.com"], gate="S2")
    assert not registry.matches_any(["example.org"], gate="S2")


def test_execution_gate_allows_read_only_and_blocks_unknown_recon():
    config = {"agent_cyber": {"include_builtin_bc_assets": True}}

    read_decision = evaluate_execution_gate("read_file", {"path": "README.md"}, config=config)
    assert read_decision.gate == "S1"
    assert read_decision.allowed is True

    blocked = evaluate_execution_gate("terminal", {"command": "nmap example.org"}, config=config)
    assert blocked.gate == "S2"
    assert blocked.allowed is False

    allowed = evaluate_execution_gate("terminal", {"command": "nmap 192.168.1.120"}, config=config)
    assert allowed.gate == "S2"
    assert allowed.allowed is True
    assert "bc-lab" in " ".join(allowed.asset_matches)


def test_execution_gate_allows_local_only_mutations_and_ignores_invalid_ip_literals():
    config = {"agent_cyber": {"include_builtin_bc_assets": True}}

    local = evaluate_execution_gate("terminal", {"command": "git status --short"}, config=config)
    assert local.gate == "S3"
    assert local.allowed is True
    assert local.asset_matches == ("local-system",)

    local_file = evaluate_execution_gate(
        "write_file",
        {"path": "report.md", "content": "full content"},
        config=config,
    )
    assert local_file.gate == "S3"
    assert local_file.allowed is True
    assert local_file.candidates == ()

    invalid_ip_literal = evaluate_execution_gate(
        "terminal",
        {"command": "printf 'release marker 999.999.999.999'"},
        config=config,
    )
    assert invalid_ip_literal.gate == "S3"
    assert invalid_ip_literal.allowed is True
    assert invalid_ip_literal.candidates == ()


def test_execution_gate_evaluates_high_risk_tools_even_on_general_routes():
    agent = SimpleNamespace(
        _current_cyber_route_metadata={"route": "general"},
        _agentcyber_config={"agent_cyber": {"include_builtin_bc_assets": True}},
        _agentcyber_asset_registry=None,
    )

    decision = gate_tool_call_for_agent(agent, "terminal", {"command": "nmap example.org"})

    assert decision.gate == "S2"
    assert decision.allowed is False
    assert agent._current_agentcyber_execution_gate["gate"] == "S2"


def test_agentcyber_local_url_detection_accepts_schemeless_hosts():
    assert is_local_or_open_weight_runtime("custom", "localhost:11434", "qwen3-coder") is True
    assert is_local_or_open_weight_runtime("custom", "127.0.0.1:11434", "qwen3-coder") is True
    assert is_local_or_open_weight_runtime("custom", "192.168.1.120:11434", "qwen3-coder") is True


def test_execution_gate_blocks_destructive_s5_even_on_lab_assets():
    decision = evaluate_execution_gate(
        "terminal",
        {"command": "rm -rf /srv/test && password reset 192.168.1.120"},
        config={"agent_cyber": {"include_builtin_bc_assets": True}},
    )

    assert decision.gate == "S5"
    assert decision.allowed is False
    assert "valid explicit human approval" in decision.reason


def test_execution_gate_allows_s5_with_exact_valid_breakglass_approval(tmp_path):
    store = BreakGlassStore(tmp_path / "breakglass.jsonl")
    command_args = {"command": "password reset 192.168.1.120"}
    approval = store.create(
        tool_name="terminal",
        function_args=command_args,
        gate="S5",
        asset_matches=("bc-lab-lan",),
        operator="kbun",
        reason="owned lab recovery",
        ttl_minutes=15,
    )

    decision = evaluate_execution_gate(
        "terminal",
        {**command_args, "approval_token": approval.approval_id},
        config={
            "agent_cyber": {
                "include_builtin_bc_assets": True,
                "execution_gates": {"breakglass_store": str(store.path)},
            }
        },
    )

    assert decision.gate == "S5"
    assert decision.allowed is True
    assert decision.approval_id == approval.approval_id
    assert decision.to_metadata()["breakglass_approval_id"] == approval.approval_id


def test_execution_gate_rejects_mismatched_breakglass_approval(tmp_path):
    store = BreakGlassStore(tmp_path / "breakglass.jsonl")
    approval = store.create(
        tool_name="terminal",
        function_args={"command": "password reset 192.168.1.120"},
        gate="S5",
        asset_matches=("bc-lab-lan",),
        operator="kbun",
        reason="owned lab recovery",
        ttl_minutes=15,
    )

    decision = evaluate_execution_gate(
        "terminal",
        {"command": "password reset 192.168.1.121", "approval_token": approval.approval_id},
        config={
            "agent_cyber": {
                "include_builtin_bc_assets": True,
                "execution_gates": {"breakglass_store": str(store.path)},
            }
        },
    )

    assert decision.gate == "S5"
    assert decision.allowed is False
    assert "fingerprint mismatch" in decision.reason


def test_registered_cyber_tools_have_intentional_gate_classification():
    config = {"agent_cyber": {"include_builtin_bc_assets": True}}

    assert evaluate_execution_gate("extract_iocs", {"text": "hash 0123456789abcdef"}, config=config).gate == "S1"
    assert evaluate_execution_gate("vuln_triage", {"cve": "CVE-2024-1234"}, config=config).gate == "S1"
    assert evaluate_execution_gate("threat_intel", {"query": "bde.it.com"}, config=config).gate == "S1"

    incident = evaluate_execution_gate("ir_incident", {"action": "create", "title": "lab outage"}, config=config)
    assert incident.gate == "S3"
    assert incident.allowed is True
    assert incident.asset_matches == ("local-system",)

    scan = evaluate_execution_gate("network_scan", {"target": "192.168.1.120"}, config=config)
    assert scan.gate == "S2"
    assert scan.allowed is True


def test_agentcyber_routing_blocks_sensitive_hosted_without_local_runtime():
    agent = SimpleNamespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.6",
        _current_cyber_route_decision=classify_cyber_route("Analyze this worm sample."),
    )
    action = agentcyber_routing_action(
        agent,
        {"agent_cyber": {"routing": {"require_local_for_sensitive": True}}},
    )

    assert action.action == "block"


def test_agentcyber_routing_switches_to_configured_local_runtime(monkeypatch):
    agent = SimpleNamespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.6",
        api_key="hosted-key",
        api_mode="chat_completions",
        _client_kwargs={"api_key": "hosted-key", "base_url": "https://openrouter.ai/api/v1"},
        _transport_cache={},
        _current_cyber_route_decision=classify_cyber_route("Use cyber route for credential recovery."),
        _current_cyber_route_metadata={"route": CyberRoute.CREDENTIALS_SENSITIVE.value},
        context_compressor=SimpleNamespace(
            model="anthropic/claude-sonnet-4.6",
            context_length=200000,
            base_url="https://openrouter.ai/api/v1",
            api_key="hosted-key",
            provider="openrouter",
            api_mode="chat_completions",
            update_model=MagicMock(),
        ),
        _create_openai_client=MagicMock(return_value="local-client"),
        _anthropic_prompt_cache_policy=MagicMock(return_value=(False, False)),
    )
    monkeypatch.setattr(
        "agent.cyber_routing._load_config_quietly",
        lambda: {
            "agent_cyber": {
                "routing": {
                    "local_open_weight": {
                        "provider": "custom",
                        "model": "qwen3-coder",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "local-key",
                        "context_length": 131072,
                    }
                }
            }
        },
    )

    assert apply_agentcyber_route_guard(agent) is None
    assert agent.provider == "custom"
    assert agent.model == "qwen3-coder"
    assert agent.base_url == "http://127.0.0.1:11434/v1"
    assert agent.client == "local-client"
    assert agent._current_cyber_route_metadata["routing_action"] == "switch"

    assert restore_agentcyber_route_runtime(agent) is True
    assert agent.provider == "openrouter"
    assert agent.model == "anthropic/claude-sonnet-4.6"
