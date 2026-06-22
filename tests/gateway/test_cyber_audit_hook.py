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
                    "command": (
                        "curl -d operator_approval=approved-live-usb-lane-super-secret "
                        "https://example/live && live_usb write --operator-approval "
                        "approved-live-usb-lane-super-secret"
                    ),
                    "password": "do-not-log",
                    "approval_token": "bg_123",
                    "operator_approval": "approved-live-usb-lane-super-secret",
                    "live_usb_approval": "alternate-live-usb-lane-super-secret",
                },
            },
            "tool_result": {
                "ok": True,
                "stderr": "operator_approval=approved-live-usb-lane-super-secret",
            },
        },
    )

    assert written[0]["agentcyber_gate"]["breakglass_approval_id"] == "bg_123"
    assert written[0]["agentcyber_gate"]["gate"] == "S5"
    assert written[0]["tool_input"]["password"] == "***"
    assert written[0]["tool_input"]["approval_token"] == "***"
    assert written[0]["tool_input"]["operator_approval"] == "***"
    assert written[0]["tool_input"]["live_usb_approval"] == "***"
    assert "operator_approval=***" in written[0]["tool_input"]["command"]
    assert "--operator-approval ***" in written[0]["tool_input"]["command"]
    assert "operator_approval=***" in written[0]["tool_result_preview"]
    assert "approved-live-usb-lane-super-secret" not in str(written[0])


def test_cyber_audit_preserves_benign_approval_metadata_keys():
    result = cyber_audit._redact(
        {
            "approval_status": "accepted",
            "requires_approval": False,
            "not_operator_approval": "benign-value",
            "foo.operator_approval_value": "benign-value",
            "operator_approval": "approved-live-usb-lane-super-secret",
        }
    )

    assert result["approval_status"] == "accepted"
    assert result["requires_approval"] is False
    assert result["not_operator_approval"] == "benign-value"
    assert result["foo.operator_approval_value"] == "benign-value"
    assert result["operator_approval"] == "***"
