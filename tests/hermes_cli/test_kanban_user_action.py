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
    monkeypatch.setattr(kua, "_LAUNCH_ENVIRONMENTS", {})
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


def test_failed_block_fence_rolls_back_persisted_user_action(board):
    conn, home = board
    tid = _running(conn)
    running = kb.get_task(conn, tid)
    assert running is not None and running.current_run_id is not None

    assert not kb.block_task(
        conn, tid, kind="needs_input", reason="stale mutation",
        expected_run_id=running.current_run_id + 1, user_action=_action("stale mutation"),
        readiness_probe={"kind": "env_present", "name": "CANARY_KEY"},
    )

    with kbc.connect(home / "kanban.db") as observer:
        task = kb.get_task(observer, tid)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == running.current_run_id
        assert observer.execute(
            "SELECT COUNT(*) FROM kanban_user_actions WHERE task_id=?", (tid,),
        ).fetchone()[0] == 0


def test_valid_block_fence_atomically_persists_action_and_run_state(board):
    conn, home = board
    tid = _running(conn)
    running = kb.get_task(conn, tid)
    assert running is not None and running.current_run_id is not None

    assert kb.block_task(
        conn, tid, kind="needs_input", reason="credential missing",
        expected_run_id=running.current_run_id, user_action=_action(),
        readiness_probe={"kind": "env_present", "name": "CANARY_KEY"},
    )

    with kbc.connect(home / "kanban.db") as observer:
        task = kb.get_task(observer, tid)
        state = kua.get_user_action(observer, tid)
        run = observer.execute(
            "SELECT status, outcome, summary FROM task_runs WHERE id=?",
            (running.current_run_id,),
        ).fetchone()
        assert task is not None and task.status == "needs_user_action"
        assert task.current_run_id is None
        assert state is not None and state.payload == _action()
        assert state.readiness_probe == {"kind": "env_present", "name": "CANARY_KEY"}
        assert observer.execute(
            "SELECT COUNT(*) FROM kanban_user_actions WHERE task_id=?", (tid,),
        ).fetchone()[0] == 1
        assert run is not None
        assert tuple(run) == ("blocked", "blocked", "credential missing")
        events = kb.list_events(observer, tid)
        blocked = [event for event in events if event.kind == "needs_user_action"]
        assert len(blocked) == 1
        assert blocked[0].run_id == running.current_run_id
        assert blocked[0].payload["material_fingerprint"] == state.fingerprint


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
    kua.register_launch_environment("worker", os.environ)
    try:
        assert kua.supervise_user_actions(reopened) == [tid]
        assert kb.get_task(reopened, tid).status == "ready"
        assert kua.get_user_action(reopened, tid).resolved_at is not None
        assert sum(e.kind == "user_action_ready" for e in kb.list_events(reopened, tid)) == 1
        assert kua.supervise_user_actions(reopened) == []
    finally:
        reopened.close()


def test_env_present_launch_environments_are_isolated_by_exact_target_profile(board, monkeypatch):
    conn, _ = board
    key = "KANBAN_PROFILE_SCOPE_CANARY"
    monkeypatch.setenv(key, "ambient-must-not-be-used")

    task_a = kb.create_task(conn, title="profile A gate", assignee="profile-a")
    task_b = kb.create_task(conn, title="profile B gate", assignee="profile-b")
    task_missing = kb.create_task(conn, title="missing profile gate", assignee="profile-missing")
    for task_id in (task_a, task_b, task_missing):
        assert kb.claim_task(conn, task_id)
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="credential missing",
            user_action=_action(), readiness_probe={"kind": "env_present", "name": key},
        )

    kua.register_launch_environment("profile-a", {key: "value-a"})
    assert kua.supervise_user_actions(conn) == [task_a]
    assert kb.get_task(conn, task_b).status == "needs_user_action"
    assert kb.get_task(conn, task_missing).status == "needs_user_action"

    kua.register_launch_environment("profile-b", {key: "value-b"})
    assert kua.launch_environment("profile-a") == {key: "value-a"}
    assert kua.launch_environment("profile-b") == {key: "value-b"}
    assert kua.supervise_user_actions(conn) == [task_b]

    # A -> B -> A returns A's original snapshot. Re-registration cannot let a
    # later ambient/profile-B state overwrite the launch environment A captured.
    kua.register_launch_environment("profile-a", {key: "corrupted-by-b"})
    assert kua.launch_environment("profile-a") == {key: "value-a"}
    assert kb.get_task(conn, task_missing).status == "needs_user_action"


def test_env_present_same_profile_restart_recaptures_and_resumes(board, monkeypatch):
    conn, home = board
    tid = kb.create_task(conn, title="restart gate", assignee="restart-profile")
    assert kb.claim_task(conn, tid)
    assert kb.block_task(
        conn, tid, kind="needs_input", reason="credential missing",
        user_action=_action(), readiness_probe={"kind": "env_present", "name": "RESTART_CANARY"},
    )
    conn.close()

    # A new process starts with an empty in-memory registry and captures the
    # same profile's new launch environment before supervising the reopened DB.
    monkeypatch.setattr(kua, "_LAUNCH_ENVIRONMENTS", {})
    kua.register_launch_environment("restart-profile", {"RESTART_CANARY": "present"})
    reopened = kbc.connect(home / "kanban.db")
    try:
        assert kua.supervise_user_actions(reopened) == [tid]
        assert kb.get_task(reopened, tid).status == "ready"
    finally:
        reopened.close()


def test_dispatch_capture_uses_each_target_profiles_own_environment(board, monkeypatch, tmp_path):
    _conn, _ = board
    key = "KANBAN_PROFILE_SCOPE_CANARY"
    profile_a = tmp_path / "profiles" / "profile-a"
    profile_b = tmp_path / "profiles" / "profile-b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    (profile_a / ".env").write_text(f"{key}=value-a\n", encoding="utf-8")
    (profile_b / ".env").write_text(f"{key}=value-b\n", encoding="utf-8")
    monkeypatch.setenv(key, "ambient-launch-value")
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [
            ("default", tmp_path / ".hermes"),
            ("profile-a", profile_a),
            ("profile-b", profile_b),
        ],
    )

    kua.register_available_launch_environments()

    default_env = kua.launch_environment("default")
    a_env = kua.launch_environment("profile-a")
    b_env = kua.launch_environment("profile-b")
    assert default_env is not None and default_env[key] == "ambient-launch-value"
    assert a_env is not None and a_env[key] == "value-a"
    assert b_env is not None and b_env[key] == "value-b"
    assert kua.launch_environment("profile-missing") is None


def test_profile_discovery_failure_does_not_break_unrelated_readiness_probe(board, monkeypatch, tmp_path):
    conn, _ = board
    marker = tmp_path / "ready"
    marker.touch()
    tid = _running(conn)
    assert kb.block_task(
        conn, tid, kind="needs_input", reason="wait for marker", user_action=_action(),
        readiness_probe={"kind": "path_exists", "path": str(marker)},
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: (_ for _ in ()).throw(OSError("profile inventory unavailable")),
    )

    result = kbd.dispatch_once(conn, dry_run=True, max_spawn=0)

    task = kb.get_task(conn, tid)
    assert result.promoted == 0
    assert task is not None and task.status == "ready"


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
