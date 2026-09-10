"""Real control/approval effects must not publish metadata before audience consent."""

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway import hosted_room_messaging as rooms
from gateway import hosted_room_messaging_approvals as approvals
from gateway import group_home_consent as consent
from gateway.config import Platform
from gateway.group_home_identity import acknowledgement
from tests.gateway.test_group_home_consent import home, command


@pytest.fixture
def emergency_home(home):
    with sqlite3.connect(home.service.db_path) as conn:
        members = json.loads(
            conn.execute(
                "SELECT members_json FROM hosted_rooms WHERE room_id='release-room'"
            ).fetchone()[0]
        )
        members[0]["display_name"] = "PrivateBotMarker"
        conn.execute(
            "UPDATE hosted_rooms SET name=?, members_json=? WHERE room_id='release-room'",
            ("PrivateRoomMarker", json.dumps(members)),
        )
    room = next(
        item
        for item in rooms.list_messaging_rooms(home.service)
        if item["room_id"] == "release-room"
    )
    home.number = rooms.room_reference(room)
    home.calls = []
    home.service.service = SimpleNamespace(
        approve_room_task=lambda *args, **kwargs: home.calls.append((args, kwargs))
    )
    action = {
        "kind": "approval",
        "member_id": "default",
        "task_id": "task-1",
        "request_id": "request-1",
        "execution_generation": 1,
        "authority_gateway_id": room["authority_gateway_id"],
        "authority_epoch": room["authority_epoch"],
        "approval": {
            "description": "PrivateDescriptionMarker",
            "command": "PrivateCommandMarker",
            "choices": ["once", "deny"],
        },
    }
    home.service.room_status["pending_actions"] = [action]
    pending = approvals.pending_approvals_for_room(home.service, room)[0]
    home.code = approvals.approval_reference(pending)
    return home


def no_private(value):
    assert "Private" not in str(value)
    assert "release-room" not in str(value) and "request-1" not in str(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "deny"])
async def test_emergency_success_is_generic_but_effect_occurs(emergency_home, action):
    state = emergency_home
    value = await command(
        state,
        f"/group {state.number} {action}"
        + (f" {state.code}" if action == "deny" else ""),
    )
    assert value == consent.text(
        "stop_requested" if action == "stop" else "deny_requested"
    )
    no_private(value)
    assert state.service.stopped if action == "stop" else state.calls
    assert not state.runner.config.get_home_channel(
        Platform.TELEGRAM
    ).group_audience_ack
    if action == "deny":
        assert state.calls[0][1]["choice"] == "deny"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "deny"])
async def test_emergency_private_error_is_not_returned(
    emergency_home, monkeypatch, action
):
    state = emergency_home

    def failed(*args, **kwargs):
        raise approvals.MessagingApprovalTerminalError("PrivateErrorMarker")

    if action == "stop":
        monkeypatch.setattr(state.service, "stop_room", failed)
    else:
        state.service.service.approve_room_task = failed
    value = await command(
        state,
        f"/group {state.number} {action}"
        + (f" {state.code}" if action == "deny" else ""),
    )
    assert value == consent.text("control_unavailable")
    no_private(value)


@pytest.mark.asyncio
async def test_deny_completed_private_receipt_and_queued_decision_are_safe(
    emergency_home,
):
    state = emergency_home
    state.service.service = None
    event = replace(state.event, text=f"/group {state.number} deny {state.code}")
    result = await state.runner._handle_rooms_command(event)
    assert result == consent.text("deny_requested")
    key = "approval:" + rooms.messaging_event_id(event)
    receipt = approvals.approval_command(state.service.db_path, command_id=key)
    assert receipt["choice"] == "deny" and receipt["state"] == "pending"
    approvals.complete_approval_command(
        state.service.db_path, command_id=key, result="PrivateReceiptMarker"
    )
    replay = await state.runner._handle_rooms_command(event)
    assert replay == consent.text("control_handled")
    no_private(replay)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "deny"])
async def test_consented_control_keeps_existing_detail(emergency_home, action):
    state = emergency_home
    home = state.runner.config.get_home_channel(Platform.TELEGRAM)
    home.group_audience_ack = acknowledgement(home)
    value = await command(
        state,
        f"/group {state.number} {action}"
        + (f" {state.code}" if action == "deny" else ""),
    )
    assert ("PrivateRoomMarker" if action == "stop" else "PrivateBotMarker") in value
