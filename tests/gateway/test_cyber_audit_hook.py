"""Tests for the built-in AgentCyber audit hook."""

import pytest

from gateway.builtin_hooks import cyber_audit


@pytest.mark.asyncio
async def test_cyber_audit_records_route_metadata(monkeypatch):
    written = []
    monkeypatch.setenv("HERMES_CYBER_AUDIT", "true")
    monkeypatch.setattr(cyber_audit, "_write", lambda record: written.append(record))

    route = {
        "route": "ir_breakglass",
        "provider_preference": "local_open_weight",
        "reason": "lockout or incident recovery request",
        "requires_hosted_secret_confirmation": True,
        "explicit_override": None,
    }

    await cyber_audit.handle(
        "agent:end",
        {
            "session_id": "sess-1",
            "platform": "discord",
            "cyber_route": route,
            "response": "ok",
        },
    )

    assert len(written) == 1
    assert written[0]["cyber_route"] == route


@pytest.mark.asyncio
async def test_cyber_audit_records_breakglass_metadata_without_raw_secret_args(monkeypatch):
    written = []
    monkeypatch.setenv("HERMES_CYBER_AUDIT", "true")
    monkeypatch.setattr(cyber_audit, "_write", lambda record: written.append(record))

    await cyber_audit.handle(
        "agent:step",
        {
            "session_id": "sess-2",
            "agentcyber_gate": {
                "gate": "S5",
                "allowed": True,
                "reason": "break-glass approval accepted",
                "asset_matches": ["bc-lab-lan"],
                "candidates": ["192.168.1.120"],
                "breakglass_approval_id": "bg_123",
            },
            "tool_call": {
                "name": "terminal",
                "input": {
                    "command": "password reset 192.168.1.120",
                    "password": "do-not-log",
                    "approval_token": "bg_123",
                },
            },
            "tool_result": {"ok": True},
        },
    )

    assert written[0]["agentcyber_gate"]["breakglass_approval_id"] == "bg_123"
    assert written[0]["agentcyber_gate"]["gate"] == "S5"
    assert written[0]["tool_input"]["password"] == "***"
    assert written[0]["tool_input"]["approval_token"] == "***"
