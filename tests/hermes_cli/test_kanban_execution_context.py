from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


def context(task_type: str, *, sha: str = "a" * 40, actor: str | None = None) -> dict:
    return {
        "lane": task_type,
        "task_type": task_type,
        "requested_action": task_type,
        "pr": {
            "repository": "example/repo",
            "id": 7,
            "state": "open",
            "head_sha": sha,
        },
        "expected_sha": sha,
        "actor": actor or task_type,
        "role": task_type,
    }


def test_validation_context_round_trips_and_allows_existing_pr(conn):
    for task_type in ("integration", "validation", "qa", "review", "release"):
        tid = kb.create_task(
            conn,
            title=task_type,
            assignee=task_type,
            execution_context=context(task_type),
        )
        task = kb.get_task(conn, tid)
        assert task.execution_context["pr"]["head_sha"] == "a" * 40
        assert kbd.check_respawn_guard(conn, tid) is None, task_type


def test_missing_or_malformed_context_fails_closed_for_validation(conn):
    missing = kb.create_task(conn, title="missing", assignee="qa")
    malformed = kb.create_task(conn, title="malformed", assignee="qa")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET execution_context = ? WHERE id = ?", ('{"lane":"qa"}', malformed))
    assert kbd.check_respawn_guard(conn, missing) == "unsafe_context"
    assert kbd.check_respawn_guard(conn, malformed) == "unsafe_context"


def test_sha_mismatch_and_unknown_context_are_blocked(conn):
    mismatch = kb.create_task(conn, title="mismatch", assignee="review")
    bad_context = context("review", sha="a" * 40) | {"expected_sha": "b" * 40}
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET execution_context = ? WHERE id = ?", (json.dumps(bad_context), mismatch))
    unknown = kb.create_task(conn, title="unknown", assignee="review")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET execution_context = ? WHERE id = ?", (
            json.dumps(context("review") | {"task_type": "unknown", "role": "unknown", "requested_action": "unknown"}),
            unknown,
        ))
    assert kbd.check_respawn_guard(conn, mismatch) == "sha_mismatch"
    assert kbd.check_respawn_guard(conn, unknown) == "unsafe_context"


def test_valid_implementation_context_needs_recent_pr_url_to_block(conn):
    valid = kb.create_task(
        conn, title="valid implementation", assignee="implementation",
        execution_context=context("implementation"),
    )
    assert kbd.check_respawn_guard(conn, valid) is None
    kb.add_comment(conn, valid, "worker", "https://github.com/example/repo/pull/7")
    assert kbd.check_respawn_guard(conn, valid) == "active_pr"


def test_force_bypass_is_rejected(conn):
    implementation = kb.create_task(conn, title="implementation", assignee="implementation")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET execution_context = ? WHERE id = ?", (
            json.dumps(context("implementation") | {"force": True}), implementation,
        ))
    kb.add_comment(conn, implementation, "worker", "https://github.com/example/repo/pull/7")
    assert kbd.check_respawn_guard(conn, implementation) == "unsafe_context"


def test_pid_persistence_loses_fence_without_partial_state(conn):
    tid = kb.create_task(conn, title="fenced", assignee="integration")
    claimed = kb.claim_task(conn, tid, claimer="host:worker")
    assert claimed is not None
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_lock = ? WHERE id = ?", ("host:replacement", tid))
    assert not kbd._set_worker_pid(
        conn, tid, 4242,
        expected_run_id=claimed.current_run_id,
        expected_claim_lock=claimed.claim_lock,
    )
    row = conn.execute("SELECT worker_pid FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_pid"] is None
    assert conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'spawned'", (tid,)
    ).fetchone() is None


def test_context_is_persisted_in_created_event_without_raw_secrets(conn):
    tid = kb.create_task(
        conn,
        title="persist",
        assignee="qa",
        execution_context=context("qa"),
    )
    row = conn.execute("SELECT execution_context FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert json.loads(row["execution_context"])["pr"]["id"] == 7
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created'", (tid,)
    ).fetchone()
    assert "execution_context" not in (event["payload"] or "")


def test_validated_pr_claims_one_worker_and_fenced_readback(conn, monkeypatch):
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._profile_exists_fn", lambda: lambda _: True)
    tid = kb.create_task(
        conn, title="integrate", assignee="integration",
        execution_context=context("integration"),
    )
    calls = []
    first = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: calls.append(task.id))
    second = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: calls.append(task.id))
    assert [item[0] for item in first.spawned] == [tid]
    assert second.spawned == []
    assert calls == [tid]
    assert kb.get_task(conn, tid).status == "running"
    assert kbd.evaluate_respawn_guard(conn, tid) is kbd.GuardDecision.ALLOW_ONE_WORKER
