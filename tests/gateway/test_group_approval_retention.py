"""Room deletion and ordinary receipt retention also retire permission payloads/links."""

import time

import pytest

from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_room_approval_rules as rules
from gateway import hosted_rooms
from tests.gateway.test_hosted_room_approval_rules import pending, finish, transaction


def grant(db):
    request, attempt = pending(db)
    approvals.begin_approval_command(db, command_id="grant", pending=request, choice="remember")
    approvals.apply_pending_decision(db, pending=request, choice="once", command_id="grant", apply=lambda: {"resolved": 1})
    finish(db, attempt)


def test_normal_automatic_usage_keeps_links_within_receipt_retention(tmp_path):
    db = tmp_path / "state.db"
    grant(db)
    for index in range(2, 7):
        with transaction(db) as conn:
            conn.execute("UPDATE hosted_room_messaging_approval_commands SET updated_at=? WHERE state='completed'",
                         (time.time() - approvals.COMMAND_RETENTION_SECONDS - 1,))
        request, attempt = pending(db, suffix=str(index))
        assert approvals.queue_remembered_approval(db, room_id="group-a", member_id="writer", action=request)
        commands = approvals.list_pending_approval_commands(db, room_id="group-a")
        assert len(commands) == 1
        command_id = commands[0]["command_id"]
        approvals.apply_pending_decision(db, pending=request, choice="once", command_id=command_id, apply=lambda: {"resolved": 1})
        finish(db, attempt)
        with transaction(db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM hosted_room_approval_rule_commands").fetchone()[0] == 1
            assert len(rules.list_rules(conn, "group-a")) == 1


def test_purged_room_leaves_no_remembered_payload_or_quota(tmp_path):
    db = tmp_path / "state.db"
    grant(db)
    room = hosted_rooms.room_state(db, room_id="group-a")
    decisions_before = approvals.approval_command(db, command_id="grant")
    hosted_rooms.disband_room(db, room_id="group-a", expected_epoch=room["authority_epoch"], expected_gateway_id="home")
    hosted_rooms.prune_disbanded_rooms(db, now=time.time() + hosted_rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1)
    with transaction(db) as conn:
        for table in ("hosted_room_approval_rules", "hosted_room_approval_rule_commands"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert approvals.approval_command(db, command_id="grant") == decisions_before


def test_revoked_auto_identity_cannot_be_reintroduced_after_link_pruning(tmp_path):
    db = tmp_path / "state.db"
    grant(db)
    request, _ = pending(db, suffix="2")
    assert approvals.queue_remembered_approval(db, room_id="group-a", member_id="writer", action=request)
    command_id = approvals.list_pending_approval_commands(db, room_id="group-a")[0]["command_id"]
    with transaction(db) as conn:
        rule = rules.list_rules(conn, "group-a")[0]
        rules.revoke_rule(conn, "group-a", rule["rule_id"])
        rules.prune_rules(conn, now=time.time())
        assert conn.execute("SELECT 1 FROM hosted_room_approval_rule_commands WHERE command_id=?", (command_id,)).fetchone() is None
    with pytest.raises(approvals.MessagingApprovalTerminalError, match="receipt"):
        approvals.apply_pending_decision(db, pending=request, choice="once", command_id=command_id,
                                         apply=lambda: pytest.fail("revoked automatic decision reached target"))
    with pytest.raises(approvals.MessagingApprovalError, match="current remembered permission"):
        approvals.begin_approval_command(db, command_id=command_id, pending=request, choice="once")
    assert approvals.begin_approval_command(db, command_id="human-deny", pending=request, choice="deny")["choice"] == "deny"
