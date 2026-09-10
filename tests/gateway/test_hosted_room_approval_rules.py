"""Remembered permissions are narrow, revocable decisions with a confirmed first grant."""

from contextlib import contextmanager
import sqlite3
import time

import pytest

from gateway import hosted_room_approval_rules as rules
from gateway import hosted_room_driver as driver
from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_rooms


@contextmanager
def transaction(db):
    conn = approvals._connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    finally:
        conn.close()


def pending(db, *, room_id="group-a", key="a" * 64, suffix="1", context="Local, folder /workspace"):
    room = hosted_rooms.create_room(db, room_id=room_id, name=room_id, authority_gateway_id="home", members=[
        {"member_id": "writer", "profile": "writer"}, {"member_id": "reviewer", "profile": "reviewer"},
    ])
    lease = driver.acquire_lease(db, room_id=room_id, gateway_id="home", authority_epoch=room["authority_epoch"],
                                 process_generation="worker", ttl_seconds=60, clock=time.time)
    identity = driver.TaskIdentity(room_id, "task-" + suffix, "thread-" + suffix, "turn-" + suffix)
    task = driver.admit_task(db, identity, payload={"target_profile": "writer", "target_member_id": "writer",
                                                 "prompt": "Synthetic test", "source_event_seq": int(suffix)}, clock=time.time)
    attempt = driver.start_task(db, identity, lease, expected_cancel_generation=task["cancel_generation"], clock=time.time)
    action = {"kind": "approval", "authority_gateway_id": "home", "authority_epoch": room["authority_epoch"],
              "member_id": "writer", "profile": "writer", "task_id": identity.task_id,
              "execution_generation": attempt.execution_generation, "request_id": "request-" + suffix,
              "observer_generation": "worker", "observer_lease_generation": lease.lease_generation,
              "approval": {"command": "rm build", "description": "Delete build output",
                           "choices": ["once", "deny", "remember"], "remember_key": key,
                           "remember_context": context}}
    return approvals.persist_pending_approval(db, room_id=room_id, member_id="writer", action=action), attempt


def stage(db, request, command_id="remember-1"):
    approvals.begin_approval_command(db, command_id=command_id, pending=request, choice="once")
    with transaction(db) as conn:
        return rules.stage_rule(conn, request, command_id)


def complete(db, command_id="remember-1", result="Approved once."):
    with transaction(db) as conn:
        conn.execute("UPDATE hosted_room_messaging_approval_commands SET state='completed', result_text=?, "
                     "application_started_at=? WHERE command_id=?", (result, time.time(), command_id))
        rules.complete_rule_decision(conn, command_id, result)


def finish(db, attempt):
    driver.settle_task(db, attempt, status="settled", settlement_id="settle-" + attempt.identity.task_id,
                      result={"text": "Done"}, clock=time.time)


def test_permission_is_inert_until_exact_first_approval_succeeds(tmp_path):
    db = tmp_path / "state.db"
    request, _ = pending(db)
    rule = stage(db, request)
    with transaction(db) as conn:
        assert rules.list_rules(conn, "group-a") == []
        assert rules.matching_rule(conn, request) is None
        rules.complete_rule_decision(conn, "remember-1", "Approved once.")
        assert rules.list_rules(conn, "group-a") == []
    complete(db)
    with transaction(db) as conn:
        assert rules.matching_rule(conn, request)["rule_id"] == rule["rule_id"]


def test_rule_matches_a_later_request_but_not_another_group_or_operation(tmp_path):
    db = tmp_path / "state.db"
    first, attempt = pending(db)
    rule = stage(db, first)
    complete(db)
    finish(db, attempt)
    later, _ = pending(db, suffix="2")
    other_group, _ = pending(db, room_id="group-b")
    with transaction(db) as conn:
        assert rules.matching_rule(conn, later)["rule_id"] == rule["rule_id"]
        assert rules.matching_rule(conn, other_group) is None
        changed = {**later, "approval": {**later["approval"], "remember_key": "b" * 64}}
        with pytest.raises(rules.ApprovalRuleError, match="changed"):
            rules.matching_rule(conn, changed)


def test_denial_and_revocation_cannot_activate_a_staged_permission(tmp_path):
    db = tmp_path / "state.db"
    request, _ = pending(db)
    rule = stage(db, request)
    complete(db, result="Denied.")
    with transaction(db) as conn:
        assert rules.list_rules(conn, "group-a") == []
        assert rules.revoke_rule(conn, "group-a", rule["rule_id"])
    complete(db)
    with transaction(db) as conn:
        assert rules.list_rules(conn, "group-a") == []
        with pytest.raises(rules.ApprovalRuleError, match="removed"):
            rules.stage_rule(conn, request, "remember-1")


def test_revocation_clears_unstarted_automatic_decision_without_losing_the_question(tmp_path):
    db = tmp_path / "state.db"
    request, attempt = pending(db)
    rule = stage(db, request)
    complete(db)
    finish(db, attempt)
    later, _ = pending(db, suffix="2")
    approvals.begin_approval_command(db, command_id="approval-rule:2", pending=later, choice="once", rule_ref=rule)
    with transaction(db) as conn:
        rules.bind_rule_use(conn, later, "approval-rule:2", rule)
        rules.require_live_rule_use(conn, later, "approval-rule:2")
        assert rules.revoke_rule(conn, "group-a", rule["rule_id"])
        assert conn.execute("SELECT 1 FROM hosted_room_messaging_approval_commands WHERE command_id='approval-rule:2'").fetchone() is None
        assert conn.execute("SELECT request_id FROM hosted_room_pending_approvals WHERE room_id='group-a'").fetchone()[0] == later["request_id"]
    denied = approvals.begin_approval_command(db, command_id="human-deny", pending=later, choice="deny")
    assert denied["choice"] == "deny"


def test_revocation_does_not_erase_an_already_started_decision(tmp_path):
    db = tmp_path / "state.db"
    request, attempt = pending(db)
    rule = stage(db, request)
    complete(db)
    finish(db, attempt)
    later, _ = pending(db, suffix="2")
    approvals.begin_approval_command(db, command_id="approval-rule:2", pending=later, choice="once", rule_ref=rule)
    with transaction(db) as conn:
        rules.bind_rule_use(conn, later, "approval-rule:2", rule)
        conn.execute("UPDATE hosted_room_messaging_approval_commands SET application_started_at=? WHERE command_id='approval-rule:2'", (time.time(),))
        assert rules.revoke_rule(conn, "group-a", rule["rule_id"])
        assert conn.execute("SELECT 1 FROM hosted_room_messaging_approval_commands WHERE command_id='approval-rule:2'").fetchone()


def test_stale_observation_and_stopped_task_cannot_create_a_permission(tmp_path):
    db = tmp_path / "state.db"
    request, attempt = pending(db)
    with transaction(db) as conn:
        with pytest.raises(rules.ApprovalRuleError, match="observer"):
            rules.stage_rule(conn, {**request, "observer_generation": "legacy"}, "legacy")
    finish(db, attempt)
    with transaction(db) as conn:
        with pytest.raises(rules.ApprovalRuleError, match="no longer running"):
            rules.stage_rule(conn, request, "late")


def test_member_and_authority_changes_disable_matching(tmp_path):
    db = tmp_path / "state.db"
    request, _ = pending(db)
    rule = stage(db, request)
    complete(db)
    with transaction(db) as conn:
        conn.execute("UPDATE hosted_rooms SET authority_epoch=authority_epoch+1 WHERE room_id='group-a'")
        assert rules.list_rules(conn, "group-a") == []
        assert rules.revoke_rule(conn, "group-b", rule["rule_id"]) is False


def test_permission_creation_requires_an_existing_write_transaction(tmp_path):
    db = tmp_path / "state.db"
    request, _ = pending(db)
    conn = approvals._connect(db)
    try:
        with pytest.raises(rules.ApprovalRuleError, match="transaction"):
            rules.stage_rule(conn, request, "unfenced")
    finally:
        conn.close()


def test_a_stale_automatic_snapshot_cannot_recreate_approval_after_revocation(tmp_path):
    db = tmp_path / "state.db"
    request, attempt = pending(db)
    rule = stage(db, request)
    complete(db)
    finish(db, attempt)
    later, _ = pending(db, suffix="2")
    approvals.begin_approval_command(db, command_id="approval-rule:2", pending=later, choice="once", rule_ref=rule)
    with transaction(db) as conn:
        assert rules.revoke_rule(conn, "group-a", rule["rule_id"])
    called = []
    with pytest.raises(approvals.MessagingApprovalTerminalError, match="receipt"):
        approvals.apply_pending_decision(db, pending=later, choice="once", command_id="approval-rule:2",
                                         apply=lambda: called.append(True) or {"resolved": 1})
    assert called == []
    assert approvals.approval_command(db, command_id="approval-rule:2") is None
    assert approvals.begin_approval_command(db, command_id="manual-deny", pending=later, choice="deny")["choice"] == "deny"


def test_existing_one_time_intent_cannot_be_upgraded_to_remember(tmp_path):
    db = tmp_path / "state.db"
    request, _ = pending(db)
    approvals.begin_approval_command(db, command_id="once-1", pending=request, choice="once")
    for command_id in ("once-1", "another-command"):
        with pytest.raises(approvals.MessagingApprovalError, match="cannot become"):
            approvals.begin_approval_command(db, command_id=command_id, pending=request, choice="remember")
    with transaction(db) as conn:
        assert rules.list_rules(conn, "group-a") == []


def test_remembered_choice_still_sends_once_and_requires_confirmed_application(tmp_path):
    db = tmp_path / "state.db"
    request, _ = pending(db)
    plan = approvals.begin_approval_command(db, command_id="remember-1", pending=request, choice="remember")
    assert plan["choice"] == "once"
    with transaction(db) as conn:
        assert rules.list_rules(conn, "group-a") == []
    result = approvals.apply_pending_decision(db, pending=request, choice="once", command_id=plan["command_id"],
                                             apply=lambda: {"resolved": 1})
    assert result["resolved"] == 1
    with transaction(db) as conn:
        assert len(rules.list_rules(conn, "group-a")) == 1
