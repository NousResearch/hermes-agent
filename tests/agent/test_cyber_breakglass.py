"""Tests for AgentCyber break-glass approval store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.cyber_breakglass import (
    BreakGlassStore,
    fingerprint_tool_call,
    redacted_args,
    validate_approval,
)


def _now() -> datetime:
    return datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)


def test_fingerprint_ignores_approval_token_but_not_action_changes():
    args = {"command": "password reset 192.168.1.120", "approval_token": "bg_123"}

    assert fingerprint_tool_call("terminal", args) == fingerprint_tool_call(
        "terminal",
        {"command": "password reset 192.168.1.120"},
    )
    assert fingerprint_tool_call("terminal", args) != fingerprint_tool_call(
        "terminal",
        {"command": "password reset 192.168.1.121"},
    )


def test_redacted_args_remove_tokens_and_secret_values():
    args = {
        "command": "use token sk-abcdefghijklmnopqrstuvwxyz123456",
        "password": "super-secret",
        "approval_token": "bg_should_not_persist",
    }

    redacted = redacted_args(args)

    assert "approval_token" not in redacted
    assert redacted["password"] == "[REDACTED]"
    assert "sk-" not in redacted["command"]


def test_store_create_get_revoke_and_validate(tmp_path):
    store = BreakGlassStore(tmp_path / "breakglass.jsonl")
    args = {"command": "password reset 192.168.1.120"}

    approval = store.create(
        tool_name="terminal",
        function_args=args,
        gate="S5",
        asset_matches=("bc-lab-lan",),
        operator="kbun",
        reason="owned lab recovery",
        ttl_minutes=15,
        now=_now(),
    )

    loaded = store.get(approval.approval_id)
    assert loaded is not None
    assert loaded.operator == "kbun"
    assert loaded.redacted_args == args

    check = validate_approval(
        approval_id=approval.approval_id,
        tool_name="terminal",
        function_args={**args, "approval_token": approval.approval_id},
        gate="S5",
        asset_matches=("bc-lab-lan", "bc-lab-key-hosts"),
        store=store,
        now=_now() + timedelta(minutes=1),
    )
    assert check.allowed is True
    assert check.approval == approval

    revoked = store.revoke(approval.approval_id, now=_now() + timedelta(minutes=2))
    assert revoked.revoked is True
    assert validate_approval(
        approval_id=approval.approval_id,
        tool_name="terminal",
        function_args=args,
        gate="S5",
        asset_matches=("bc-lab-lan",),
        store=store,
        now=_now() + timedelta(minutes=3),
    ).allowed is False


def test_validate_rejects_expired_mismatched_and_unknown(tmp_path):
    store = BreakGlassStore(tmp_path / "breakglass.jsonl")
    args = {"command": "password reset 192.168.1.120"}
    approval = store.create(
        tool_name="terminal",
        function_args=args,
        gate="S5",
        asset_matches=("bc-lab-lan",),
        operator="kbun",
        reason="owned lab recovery",
        ttl_minutes=1,
        now=_now(),
    )

    assert validate_approval(
        approval_id="missing",
        tool_name="terminal",
        function_args=args,
        gate="S5",
        asset_matches=("bc-lab-lan",),
        store=store,
        now=_now(),
    ).reason == "unknown break-glass approval"
    assert validate_approval(
        approval_id=approval.approval_id,
        tool_name="terminal",
        function_args=args,
        gate="S5",
        asset_matches=("bc-lab-lan",),
        store=store,
        now=_now() + timedelta(minutes=2),
    ).reason == "break-glass approval is expired"
    assert validate_approval(
        approval_id=approval.approval_id,
        tool_name="terminal",
        function_args={"command": "password reset 192.168.1.121"},
        gate="S5",
        asset_matches=("bc-lab-lan",),
        store=store,
        now=_now(),
    ).reason == "break-glass approval action fingerprint mismatch"
    assert validate_approval(
        approval_id=approval.approval_id,
        tool_name="terminal",
        function_args=args,
        gate="S5",
        asset_matches=("other-asset",),
        store=store,
        now=_now(),
    ).reason == "break-glass approval asset scope mismatch"


def test_store_raises_on_unknown_revoke(tmp_path):
    with pytest.raises(KeyError):
        BreakGlassStore(tmp_path / "breakglass.jsonl").revoke("nope")
