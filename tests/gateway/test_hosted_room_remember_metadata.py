"""A missing or invalid operation identity must retain the ordinary approval path."""

import pytest

from gateway import hosted_room_messaging_approvals as approvals


def action(key=None, choices=None):
    return {
        "kind": "approval", "authority_gateway_id": "gateway", "authority_epoch": 1,
        "task_id": "task", "execution_generation": 1, "request_id": "request",
        "approval": {
            "command": "rm build", "description": "Delete build output",
            "remember_context": "Local, folder /workspace",
            "choices": choices or ["once", "deny", "remember"],
            **({"remember_key": key} if key is not None else {}),
        },
    }


def test_opaque_operation_identity_survives_the_pending_journal(tmp_path):
    db = tmp_path / "state.db"
    stored = approvals.persist_pending_approval(db, room_id="room", member_id="writer", action=action("a" * 64))
    loaded = approvals.list_pending_approvals(db, room_id="room")
    assert loaded[0]["approval"] == stored["approval"]
    assert loaded[0]["approval"]["remember_key"] == "a" * 64
    assert "remember" in loaded[0]["approval"]["choices"]


@pytest.mark.parametrize("key", [None, "", "short", "A" * 64, "g" * 64, 42])
def test_invalid_identity_cannot_offer_remember(key):
    pending = approvals.normalize_pending_approval("room", "writer", action(key))
    assert pending["approval"]["choices"] == ["once", "deny"]
    assert "remember_key" not in pending["approval"]


def test_a_key_does_not_create_permission_when_target_did_not_offer_it():
    pending = approvals.normalize_pending_approval("room", "writer", action("a" * 64, ["once", "deny"]))
    assert pending["approval"]["choices"] == ["once", "deny"]
    assert "remember_key" not in pending["approval"]


@pytest.mark.parametrize("context", [None, "", "\n", {}, "x" * 385])
def test_key_without_reviewable_scope_keeps_one_time_approval(context):
    request = action("a" * 64)
    request["approval"]["remember_context"] = context
    pending = approvals.normalize_pending_approval("room", "writer", request)
    assert pending["approval"]["choices"] == ["once", "deny"]
    assert "remember_context" not in pending["approval"]


def test_replacement_request_cannot_inherit_previous_operation_identity(tmp_path):
    db = tmp_path / "state.db"
    approvals.persist_pending_approval(db, room_id="room", member_id="writer", action=action("a" * 64))
    replacement = action()
    replacement["request_id"] = "replacement"
    approvals.persist_pending_approval(db, room_id="room", member_id="writer", action=replacement)
    loaded = approvals.list_pending_approvals(db, room_id="room")[0]
    assert loaded["request_id"] == "replacement"
    assert "remember_key" not in loaded["approval"]
    assert loaded["approval"]["choices"] == ["once", "deny"]
