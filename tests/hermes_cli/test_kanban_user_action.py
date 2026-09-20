"""First-class user-action state, delivery ledger, and restart-safe supervision."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_user_action as kua


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    conn = kbc.connect(home / "kanban.db")
    try:
        yield conn, home
    finally:
        conn.close()


def _running(conn):
    tid = kb.create_task(conn, title="operator gate", assignee="worker")
    assert kb.claim_task(conn, tid)
    return tid


def _action(reason="credential missing"):
    return {
        "incomplete_status": "Waiting for operator prerequisite",
        "reason": reason,
        "execution_location": "gateway host",
        "action": "Install the credential in the profile secret store",
        "expected_success": "the credential availability probe succeeds",
        "automatic_continuation": "No continue response is needed; the persisted env_present probe resumes this task automatically.",
    }


def test_structured_needs_input_is_canonical_and_persists_exact_payload(board):
    conn, _ = board
    tid = _running(conn)
    assert kb.block_task(
        conn, tid, kind="needs_input", reason="credential missing",
        user_action=_action(), readiness_probe={"kind": "env_present", "name": "CANARY_KEY"},
    )
    assert kb.get_task(conn, tid).status == "needs_user_action"
    state = kua.get_user_action(conn, tid)
    assert state.payload == _action()
    assert state.readiness_probe == {"kind": "env_present", "name": "CANARY_KEY"}


def test_legacy_needs_input_gets_actionable_fallback_atomically(board):
    conn, _ = board
    tid = _running(conn)
    assert kb.block_task(conn, tid, kind="needs_input", reason="choose a value")
    assert kb.get_task(conn, tid).status == "needs_user_action"
    state = kua.get_user_action(conn, tid)
    assert state.payload["reason"] == "choose a value"
    assert tid in state.payload["action"]
    assert "reply" in state.payload["execution_location"].lower()
    assert "No continue" in state.payload["automatic_continuation"]


def test_legacy_blocked_needs_input_migrates_with_actionable_payload(board):
    conn, home = board
    tid = kb.create_task(conn, title="legacy input", assignee="worker")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='blocked', block_kind='needs_input', "
            "last_failure_error='legacy choice required' WHERE id=?",
            (tid,),
        )
    conn.close()
    kbc._INITIALIZED_PATHS.discard(str(home / "kanban.db"))
    reopened = kbc.connect(home / "kanban.db")
    try:
        assert kb.get_task(reopened, tid).status == "needs_user_action"
        state = kua.get_user_action(reopened, tid)
        assert state is not None
        assert state.payload["reason"] == "legacy choice required"
        assert state.readiness_probe == {"kind": "task_unblocked", "task_id": tid}
    finally:
        reopened.close()


def test_probe_supervisor_survives_reopen_and_auto_resumes(board, monkeypatch):
    conn, home = board
    tid = _running(conn)
    monkeypatch.delenv("CANARY_KEY", raising=False)
    kb.block_task(
        conn, tid, kind="capability", reason="credential missing", user_action=_action(),
        readiness_probe={"kind": "env_present", "name": "CANARY_KEY"},
    )
    conn.close()
    reopened = kbc.connect(home / "kanban.db")
    monkeypatch.setenv("CANARY_KEY", "present")
    try:
        assert kua.supervise_user_actions(reopened) == [tid]
        assert kb.get_task(reopened, tid).status == "ready"
        assert kua.get_user_action(reopened, tid).resolved_at is not None
        assert sum(e.kind == "user_action_ready" for e in kb.list_events(reopened, tid)) == 1
        assert kua.supervise_user_actions(reopened) == []
    finally:
        reopened.close()


def test_delivery_ledger_dedup_retry_and_material_change(board):
    conn, _ = board
    tid = _running(conn)
    kb.block_task(conn, tid, kind="needs_input", reason="first", user_action=_action("first"))
    state = kua.get_user_action(conn, tid)
    first = kua.claim_delivery(conn, tid, "telegram", "opaque-destination", state.fingerprint)
    assert first and first["attempts"] == 1
    kua.record_delivery_result(conn, first["id"], acknowledged=False, provider="telegram", error="token=secret chat=123")
    retry = kua.claim_delivery(conn, tid, "telegram", "opaque-destination", state.fingerprint)
    assert retry and retry["id"] == first["id"] and retry["attempts"] == 2
    kua.record_delivery_result(conn, retry["id"], acknowledged=True, provider="telegram", message_id="42")
    assert kua.claim_delivery(conn, tid, "telegram", "opaque-destination", state.fingerprint) is None

    assert kb.unblock_task(conn, tid)
    assert kb.claim_task(conn, tid)
    kb.block_task(conn, tid, kind="needs_input", reason="changed", user_action=_action("changed"))
    changed = kua.get_user_action(conn, tid)
    assert changed.fingerprint != state.fingerprint
    assert kua.claim_delivery(conn, tid, "telegram", "opaque-destination", changed.fingerprint)


def test_delivery_ledger_concurrent_claim_has_exactly_one_winner(board):
    conn, home = board
    tid = _running(conn)
    kb.block_task(conn, tid, kind="needs_input", reason="first", user_action=_action("first"))
    state = kua.get_user_action(conn, tid)
    assert state is not None
    fingerprint = state.fingerprint
    db_path = home / "kanban.db"
    barrier = threading.Barrier(2)
    claims = []
    delivered = []

    def claim_and_deliver():
        worker_conn = kbc.connect(db_path)
        try:
            barrier.wait()
            claim = kua.claim_delivery(
                worker_conn, tid, "telegram", "opaque-destination", fingerprint,
            )
            claims.append(claim)
            if claim is not None:
                delivered.append(claim["id"])
        finally:
            worker_conn.close()

    workers = [threading.Thread(target=claim_and_deliver) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert not worker.is_alive()

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert delivered == [winners[0]["id"]]
    row = conn.execute(
        "SELECT id, attempts, result FROM kanban_user_action_deliveries "
        "WHERE task_id=? AND destination_key=? AND material_fingerprint=?",
        (tid, winners[0]["destination_key"], fingerprint),
    ).fetchone()
    assert tuple(row) == (winners[0]["id"], 1, "pending")


def test_spawn_without_pid_never_remains_running(board, monkeypatch):
    conn, _ = board
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    tid = kb.create_task(conn, title="no worker", assignee="worker", max_retries=1)
    kbd.dispatch_once(conn, spawn_fn=lambda *_args: None, max_spawn=1)
    task = kb.get_task(conn, tid)
    assert task.status == "needs_user_action"
    assert task.worker_pid is None
    assert kua.get_user_action(conn, tid).payload["reason"]
    event = [e for e in kb.list_events(conn, tid) if e.kind == "needs_user_action"][-1]
    assert event.payload["material_fingerprint"] == kua.get_user_action(conn, tid).fingerprint


def test_preclaim_capability_failure_emits_actionable_event(board, monkeypatch):
    conn, _ = board
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    monkeypatch.setattr(
        "hermes_cli.kanban_skill_validation.unavailable_profile_skills",
        lambda _profile, _skills: ["required-skill"],
    )
    tid = kb.create_task(conn, title="capability gate", assignee="worker", skills=["required-skill"])
    kbd.dispatch_once(conn, spawn_fn=lambda *_args: 123, max_spawn=1)
    assert kb.get_task(conn, tid).status == "needs_user_action"
    event = [e for e in kb.list_events(conn, tid) if e.kind == "needs_user_action"][-1]
    assert event.payload["user_action"]["action"]


def test_user_action_task_can_be_completed_without_forced_status_rewrite(board):
    conn, _ = board
    tid = _running(conn)
    assert kb.block_task(conn, tid, kind="needs_input", reason="operator resolved externally")
    assert kb.complete_task(conn, tid, summary="resolved")
    assert kb.get_task(conn, tid).status == "done"


def test_capability_classifier_covers_required_operator_classes():
    cases = {
        "NoNewPrivileges prevents sudo while running as root": "privilege",
        "missing credential API_TOKEN": "credential",
        "approval required for command": "approval",
        "press the physical reset button": "physical_action",
        "provider/tool unavailable": "provider_or_tool",
    }
    for text, expected in cases.items():
        assert kua.classify_capability_failure(text)["class"] == expected
