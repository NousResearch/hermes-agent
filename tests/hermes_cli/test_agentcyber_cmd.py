"""Tests for AgentCyber CLI status/setup helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli.agentcyber import apply_agentcyber_setup, build_status_report, agentcyber_command


def test_agentcyber_status_reports_tool_visibility_and_safe_runtime():
    config = {
        "agent_cyber": {
            "routing": {
                "enabled": True,
                "require_local_for_sensitive": True,
                "local_open_weight": {
                    "provider": "ollama",
                    "model": "qwen3-coder:30b",
                    "base_url": "http://192.168.1.120:11434/v1",
                    "api_key": "secret-value",
                },
            }
        },
        "platform_toolsets": {"cli": ["hermes-cli", "cyber"]},
    }

    report = build_status_report(
        config,
        platform="cli",
        health_fn=lambda _runtime: {"ok": True, "reason": "stubbed"},
    )

    assert report["agent_cyber"]["routing_enabled"] is True
    runtime = report["agent_cyber"]["local_open_weight"]
    assert runtime["provider"] == "ollama"
    assert runtime["model"] == "qwen3-coder:30b"
    assert runtime["base_url_present"] is True
    assert runtime["api_key_present"] is True
    assert "secret-value" not in str(report)
    assert report["toolsets"]["cyber_visible"] is True
    assert report["toolsets"]["live_usb_visible"] is True
    assert report["toolsets"]["cyber_enabled"] is True
    assert "network_scan" in report["toolsets"]["registered_tools"]


def test_agentcyber_setup_enables_cyber_and_local_runtime_without_live_usb_by_default():
    config = {"platform_toolsets": {"cli": ["hermes-cli"]}}

    updated = apply_agentcyber_setup(
        config,
        platform="cli",
        provider="ollama",
        model="qwen3-coder:30b",
        base_url="http://192.168.1.120:11434/v1",
    )

    routing = updated["agent_cyber"]["routing"]
    assert routing["enabled"] is True
    assert routing["require_local_for_sensitive"] is True
    assert routing["local_open_weight"]["provider"] == "ollama"
    assert routing["local_open_weight"]["model"] == "qwen3-coder:30b"
    assert "cyber" in updated["platform_toolsets"]["cli"]
    assert "live_usb" not in updated["platform_toolsets"]["cli"]


def test_agentcyber_setup_can_enable_live_usb_explicitly():
    updated = apply_agentcyber_setup({}, enable_live_usb=True)

    assert "cyber" in updated["platform_toolsets"]["cli"]
    assert "live_usb" in updated["platform_toolsets"]["cli"]


def test_agentcyber_breakglass_create_list_revoke_json(tmp_path, capsys):
    store = tmp_path / "breakglass.jsonl"
    create_args = SimpleNamespace(
        agentcyber_action="breakglass",
        breakglass_action="create",
        tool="terminal",
        args_json=json.dumps({"command": "password reset 192.168.1.120"}),
        operator="kbun",
        reason="owned lab recovery",
        ttl_minutes=15,
        store=str(store),
        apply=True,
        json=True,
    )

    assert agentcyber_command(create_args) == 0
    created = json.loads(capsys.readouterr().out)
    approval_id = created["approval_id"]
    assert approval_id.startswith("bg_")
    assert created["dry_run"] is False
    assert "password reset" in created["redacted_args"]["command"]

    assert agentcyber_command(SimpleNamespace(agentcyber_action="breakglass", breakglass_action="list", store=str(store), json=True)) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["approval_id"] == approval_id
    assert listed[0]["revoked"] is False

    assert agentcyber_command(SimpleNamespace(agentcyber_action="breakglass", breakglass_action="revoke", approval_id=approval_id, store=str(store), json=True)) == 0
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["approval_id"] == approval_id
    assert revoked["revoked"] is True


def test_agentcyber_breakglass_rejects_non_s5(tmp_path):
    args = SimpleNamespace(
        agentcyber_action="breakglass",
        breakglass_action="create",
        tool="read_file",
        args_json=json.dumps({"path": "README.md"}),
        operator="kbun",
        reason="not destructive",
        ttl_minutes=15,
        store=str(tmp_path / "breakglass.jsonl"),
        apply=True,
        json=True,
    )

    with pytest.raises(SystemExit, match="only for S5"):
        agentcyber_command(args)
