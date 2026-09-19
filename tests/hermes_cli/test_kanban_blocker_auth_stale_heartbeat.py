"""Dispatcher contracts for durable auth holds and bounded missing-heartbeat recovery."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


@pytest.fixture
def spawnable_profiles(monkeypatch):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)


def test_first_blocker_auth_becomes_one_typed_block_with_one_event(
    conn, spawnable_profiles,
):
    task_id = kb.create_task(conn, title="credential needed", assignee="coder")
    conn.execute(
        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
        ("authentication failed", task_id),
    )
    conn.commit()

    first = kbd.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: 0)
    second = kbd.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: 0)

    task = conn.execute(
        "SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    kinds = [
        row["kind"] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        )
    ]
    assert first.respawn_guarded == []
    assert second.respawn_guarded == []
    assert dict(task) == {"status": "blocked", "block_kind": "capability"}
    assert kinds.count("blocked") == 1
    assert "respawn_guarded" not in kinds


def test_explicit_unblock_clears_auth_guard_and_resumes_exactly_once(
    conn, spawnable_profiles,
):
    task_id = kb.create_task(conn, title="credential repaired", assignee="coder")
    conn.execute(
        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
        ("invalid api key", task_id),
    )
    conn.commit()
    kbd.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: 0)

    assert kb.unblock_task(conn, task_id) is True
    spawned: list[str] = []

    def spawn(task, *_args, **_kwargs):
        spawned.append(task.id)
        return 0

    kbd.dispatch_once(conn, spawn_fn=spawn)
    kbd.dispatch_once(conn, spawn_fn=spawn)

    assert spawned == [task_id]
    assert conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()["last_failure_error"] is None


def test_null_heartbeat_recovery_escalates_after_configured_bound(conn, monkeypatch):
    task_id = kb.create_task(conn, title="missing heartbeat", assignee="coder")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    run_id = claimed.current_run_id
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET started_at = ?, worker_pid = ?, last_heartbeat_at = NULL, "
        "claim_expires = ? WHERE id = ?",
        (now - kbd._STALE_HEARTBEAT_GAP_SECONDS - 1, 424242, now - 1, task_id),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = ? WHERE id = ?",
        (now - kbd._STALE_HEARTBEAT_GAP_SECONDS - 1, run_id),
    )
    conn.commit()
    monkeypatch.setattr(
        kb,
        "_terminate_reclaimed_worker",
        lambda *_args, **_kwargs: {
            "host_local": True,
            "termination_attempted": True,
            "terminated": False,
        },
    )

    for _ in range(2):
        assert kb.release_stale_claims(
            conn, stale_timeout_seconds=1, reclaim_defer_max_attempts=2,
        ) == 0
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?", (now - 1, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET claim_expires = ? WHERE id = ?", (now - 1, run_id),
        )
        conn.commit()
    before_escalation = conn.execute(
        "SELECT claim_expires FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()["claim_expires"]

    assert kb.release_stale_claims(
        conn, stale_timeout_seconds=1, reclaim_defer_max_attempts=2,
    ) == 0
    after_escalation = conn.execute(
        "SELECT claim_expires FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()["claim_expires"]
    events = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
    ).fetchall()

    assert after_escalation == before_escalation
    assert [event["kind"] for event in events].count("reclaim_deferred") == 2
    assert [event["kind"] for event in events].count("reclaim_escalated") == 1


def test_gateway_passes_configured_reclaim_defer_bound_to_dispatcher():
    from gateway import kanban_watchers_dispatcher as dispatcher

    settings = dispatcher._resolve_dispatcher_settings(
        {"reclaim_defer_max_attempts": 2}, kb,
    )

    assert settings.reclaim_defer_max_attempts == 2
