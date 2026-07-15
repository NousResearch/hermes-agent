from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from gateway.owner_approval_escalation import (
    OwnerApprovalEscalationError,
    escalate_production_team_approval,
    owner_escalation_enabled,
)


OWNER_ID = "1279454038731264061"
GUILD_ID = "1282725267068157972"
OWNER_CHANNEL_ID = "1504852355588423801"
SOURCE_CHANNEL_ID = "1504852355588423802"
SOURCE_THREAD_ID = "1504852355588423803"
SOURCE_MESSAGE_ID = "1504852355588423804"
TEAM_USER_ID = "1504852355588423805"
APPROVAL_ID = "a" * 32
COMMAND = "deploy --token super-secret-value"
COMMAND_SHA256 = hashlib.sha256(COMMAND.encode()).hexdigest()
PLAN_EVENT_ID = "11111111-1111-5111-8111-111111111111"
HANDOFF_EVENT_ID = "22222222-2222-5222-8222-222222222222"


def _config():
    return {
        "approvals": {
            "gateway_authorized_user_ids": [OWNER_ID],
            "gateway_owner_escalation": {
                "enabled": True,
                "owner_user_id": OWNER_ID,
                "owner_guild_id": GUILD_ID,
                "owner_channel_id": OWNER_CHANNEL_ID,
                "owner_target_type": "guild_channel",
            },
        }
    }


def _source(**changes):
    values = {
        "platform": SimpleNamespace(value="discord"),
        "delivered_via_upstream_relay": True,
        "chat_type": "thread",
        "scope_id": GUILD_ID,
        "guild_id": GUILD_ID,
        "chat_id": SOURCE_CHANNEL_ID,
        "thread_id": SOURCE_THREAD_ID,
        "parent_chat_id": SOURCE_CHANNEL_ID,
        "message_id": SOURCE_MESSAGE_ID,
        "user_id": TEAM_USER_ID,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _workspace(**changes):
    value = {
        "status": "exact",
        "case_id": "case:owner-escalation",
        "plan_id": "plan:deploy",
        "plan_revision": 7,
        "plan_event_id": PLAN_EVENT_ID,
    }
    value.update(changes)
    return value


def _approval():
    return {
        "approval_id": APPROVAL_ID,
        "command": COMMAND,
        "description": "task prose that must not cross the boundary",
        "command_sha256": COMMAND_SHA256,
    }


def _success_route_receipt():
    return json.dumps(
        {
            "success": True,
            "terminal_event_type": "route_back.sent",
            "receipt": {
                "message_id": "1504852355588423999",
                "readback_verified": True,
            },
        }
    )


def test_success_persists_before_guild_acl_receipted_send_without_prose_leak():
    calls = []

    def append(**kwargs):
        calls.append(("append", kwargs))
        return json.dumps(
            {
                "success": True,
                "case_id": "case:owner-escalation",
                "event_id": HANDOFF_EVENT_ID,
            }
        )

    def prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return True

    def route(**kwargs):
        calls.append(("route", kwargs))
        return _success_route_receipt()

    def activate(**kwargs):
        calls.append(("activate", kwargs))
        return True

    receipt = escalate_production_team_approval(
        config=_config(),
        source=_source(),
        session_key="agent:main:discord:thread:team",
        approval_data=_approval(),
        workspace_resolver=lambda **_: _workspace(),
        active_plan_revision=lambda **_: 7,
        event_appender=append,
        route_back_executor=route,
        binding_preparer=prepare,
        binding_activator=activate,
        binding_clearer=lambda **_: pytest.fail("successful send must not clear"),
    )

    assert [name for name, _ in calls] == [
        "prepare",
        "append",
        "route",
        "activate",
    ]
    assert receipt.approval_id == APPROVAL_ID
    assert receipt.handoff_event_id == HANDOFF_EVENT_ID
    append_call = calls[1][1]
    route_call = calls[2][1]
    assert append_call["event_type"] == "handoff.waiting"
    assert route_call["target_ref"] == {
        "id": OWNER_ID,
        "channel_id": OWNER_CHANNEL_ID,
        "guild_id": GUILD_ID,
        "target_type": "guild_channel",
    }
    assert f"/approve {APPROVAL_ID}" in route_call["message"]
    assert f"/deny {APPROVAL_ID} <reason>" in route_call["message"]
    assert SOURCE_THREAD_ID in route_call["message"]
    assert COMMAND_SHA256 in route_call["message"]
    assert "public" not in route_call["message"].casefold()
    serialized_boundary = json.dumps(
        [calls[0][1], append_call, route_call],
        sort_keys=True,
    )
    assert COMMAND not in serialized_boundary
    assert "super-secret-value" not in serialized_boundary
    assert "task prose that must not cross the boundary" not in serialized_boundary


def test_missing_exact_plan_fails_before_handoff_binding_or_send():
    writes = []
    with pytest.raises(
        OwnerApprovalEscalationError,
        match="canonical_active_plan_required",
    ):
        escalate_production_team_approval(
            config=_config(),
            source=_source(),
            session_key="session",
            approval_data=_approval(),
            workspace_resolver=lambda **_: {"status": "none"},
            active_plan_revision=lambda **_: 7,
            event_appender=lambda **kw: writes.append(("append", kw)),
            route_back_executor=lambda **kw: writes.append(("route", kw)),
            binding_preparer=lambda **kw: writes.append(("bind", kw)),
        )
    assert writes == []


def test_changed_active_revision_fails_before_any_write():
    writes = []
    with pytest.raises(
        OwnerApprovalEscalationError,
        match="canonical_active_plan_changed",
    ):
        escalate_production_team_approval(
            config=_config(),
            source=_source(),
            session_key="session",
            approval_data=_approval(),
            workspace_resolver=lambda **_: _workspace(),
            active_plan_revision=lambda **_: 8,
            event_appender=lambda **kw: writes.append(kw),
        )
    assert writes == []


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            _source(delivered_via_upstream_relay=False),
            "owner_escalation_source_not_authenticated",
        ),
        (_source(chat_type="dm"), "owner_escalation_source_not_guild_lane"),
        (
            _source(scope_id="1282725267068157999", guild_id="1282725267068157999"),
            "owner_escalation_cross_guild_forbidden",
        ),
    ],
)
def test_untrusted_non_guild_or_cross_guild_source_fails_before_write(source, code):
    writes = []
    with pytest.raises(OwnerApprovalEscalationError, match=code):
        escalate_production_team_approval(
            config=_config(),
            source=source,
            session_key="session",
            approval_data=_approval(),
            workspace_resolver=lambda **_: writes.append("workspace"),
        )
    assert writes == []


def test_route_back_block_clears_prepared_binding_and_never_reports_sent():
    calls = []

    def append(**kwargs):
        calls.append("append")
        return {
            "success": True,
            "case_id": "case:owner-escalation",
            "event_id": HANDOFF_EVENT_ID,
        }

    with pytest.raises(OwnerApprovalEscalationError, match="owner_route_back_not_sent"):
        escalate_production_team_approval(
            config=_config(),
            source=_source(),
            session_key="session",
            approval_data=_approval(),
            workspace_resolver=lambda **_: _workspace(),
            active_plan_revision=lambda **_: 7,
            event_appender=append,
            route_back_executor=lambda **_: {
                "success": False,
                "terminal_event_type": "route_back.blocked",
                "route_back_record": {"success": True, "outcome": "blocked"},
            },
            binding_preparer=lambda **_: calls.append("prepare") or True,
            binding_activator=lambda **_: calls.append("activate") or True,
            binding_clearer=lambda **_: calls.append("clear"),
        )
    assert calls == ["prepare", "append", "clear"]


def test_default_binding_wiring_resolves_only_exact_owner_in_source_lane(monkeypatch):
    from tools import approval as approval_module

    monkeypatch.setattr(
        approval_module,
        "_canonical_active_plan_matches",
        lambda **_: True,
    )
    session_key = "agent:main:discord:thread:team-member"
    entry = approval_module._ApprovalEntry(
        {
            "command": COMMAND,
            "description": "private task prose",
            "command_sha256": COMMAND_SHA256,
        }
    )
    approval_module._gateway_queues[session_key] = [entry]
    try:
        escalate_production_team_approval(
            config=_config(),
            source=_source(),
            session_key=session_key,
            approval_data=entry.data,
            workspace_resolver=lambda **_: _workspace(),
            active_plan_revision=lambda **_: 7,
            event_appender=lambda **_: {
                "success": True,
                "case_id": "case:owner-escalation",
                "event_id": HANDOFF_EVENT_ID,
            },
            route_back_executor=lambda **_: _success_route_receipt(),
        )

        public_snapshot = approval_module.get_pending_gateway_approvals(session_key)
        assert "_owner_escalation_binding" not in public_snapshot[0]
        approval_module.clear_gateway_owner_escalation_binding(
            session_key,
            entry.approval_id,
        )
        assert approval_module.resolve_gateway_owner_escalation_by_id(
            entry.approval_id,
            "once",
            owner_user_id=OWNER_ID,
            owner_guild_id=GUILD_ID,
            response_lane_id=SOURCE_THREAD_ID,
        ) == 1
        assert entry.result == "once"
        assert entry.event.is_set()
    finally:
        approval_module._gateway_queues.pop(session_key, None)


def test_config_gate_is_literal_and_exact_config_is_enforced():
    assert owner_escalation_enabled(_config()) is True
    disabled = _config()
    disabled["approvals"]["gateway_owner_escalation"]["enabled"] = "true"
    assert owner_escalation_enabled(disabled) is False

    drifted = _config()
    drifted["approvals"]["gateway_owner_escalation"]["extra"] = True
    with pytest.raises(
        OwnerApprovalEscalationError,
        match="owner_escalation_config_not_exact",
    ):
        escalate_production_team_approval(
            config=drifted,
            source=_source(),
            session_key="session",
            approval_data=_approval(),
        )
