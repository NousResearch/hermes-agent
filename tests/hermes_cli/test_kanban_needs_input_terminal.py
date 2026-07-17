"""Regression matrix for terminal ``needs_input`` Kanban blocks.

These tests exercise the production SQLite control paths.  No live board,
worker process, provider, or LLM is used.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _running_task(conn, *, title: str = "needs input") -> str:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    assert kb.claim_task(conn, task_id, claimer="worker") is not None
    return task_id


def _counts(conn, task_id: str) -> dict[str, int]:
    return {
        "runs": int(
            conn.execute(
                "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)
            ).fetchone()[0]
        ),
        "spawned": int(
            conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='spawned'",
                (task_id,),
            ).fetchone()[0]
        ),
        "promoted": int(
            conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='promoted'",
                (task_id,),
            ).fetchone()[0]
        ),
        "specified": int(
            conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='specified'",
                (task_id,),
            ).fetchone()[0]
        ),
    }


def test_repeated_needs_input_block_never_routes_to_triage(kanban_home: Path) -> None:
    """Historical root path: unblock then same-kind re-block must stay terminal."""
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        assert kb.block_task(conn, task_id, reason="choose A or B", kind="needs_input")
        assert kb.unblock_task(
            conn,
            task_id,
            actor="operator:test",
            reason="answer supplied",
        )
        assert kb.claim_task(conn, task_id, claimer="worker") is not None
        assert kb.block_task(conn, task_id, reason="need another decision", kind="needs_input")

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert task_id not in decomp.list_triage_ids()


def test_needs_input_remains_quiet_across_ten_dispatch_cycles(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        assert kb.block_task(conn, task_id, reason="human decision", kind="needs_input")
        before = _counts(conn, task_id)

        for _ in range(10):
            result = kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: None)
            assert result.spawned == []
            assert kb.get_task(conn, task_id).status == "blocked"

        assert _counts(conn, task_id) == before


def test_explicit_unblock_clears_needs_input_and_persists_audit_event(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        assert kb.block_task(conn, task_id, reason="need approval", kind="needs_input")
        assert kb.unblock_task(
            conn,
            task_id,
            actor="controller:test",
            reason="approval recorded",
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.block_kind is None
        assert task.block_recurrences == 0

        event = [e for e in kb.list_events(conn, task_id) if e.kind == "unblocked"][-1]
        assert event.task_id == task_id
        assert event.created_at > 0
        assert event.payload == {
            "previous_status": "blocked",
            "previous_block_kind": "needs_input",
            "status": "ready",
            "actor": "controller:test",
            "reason": "approval recorded",
        }

        spawned = kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: None)
        assert len(spawned.spawned) == 1
        assert kb.get_task(conn, task_id).status == "running"
        assert len(kb.list_runs(conn, task_id)) == 2


def test_stale_recovery_cannot_requeue_active_needs_input_marker(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind='needs_input', claim_expires=? WHERE id=?",
                (int(time.time()) - 1, task_id),
            )
        assert kb.release_stale_claims(conn) == 1
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: None).spawned == []


def test_unmet_dependency_and_manual_block_remain_non_dispatchable(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        manual = _running_task(conn, title="manual block")
        assert kb.block_task(conn, manual, reason="operator hold")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, manual).status == "blocked"

        result = kb.dispatch_once(conn, spawn_fn=lambda *_args, **_kwargs: None)
        assert child not in {task_id for task_id, _profile, _path in result.spawned}
        assert manual not in {task_id for task_id, _profile, _path in result.spawned}


def test_non_needs_input_triage_still_follows_normal_specification_policy(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="oversized technical work", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind='transient' WHERE id=?", (task_id,)
            )

    assert task_id in decomp.list_triage_ids()

    with kb.connect_closing() as conn:
        assert kb.specify_triage_task(
            conn,
            task_id,
            body="bounded technical work contract",
            assignee="worker",
            author="auto-decomposer:test",
        )
        kb.recompute_ready(conn)
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"


def test_needs_input_terminal_state_persists_after_database_reopen(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        assert kb.block_task(conn, task_id, reason="human answer", kind="needs_input")

    with kb.connect_closing() as reopened:
        task = kb.get_task(reopened, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert kb.dispatch_once(
            reopened, spawn_fn=lambda *_args, **_kwargs: None
        ).spawned == []


def test_stale_status_columns_cannot_bypass_needs_input_guard(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        ready_id = kb.create_task(conn, title="stale ready marker", assignee="worker")
        triage_id = kb.create_task(conn, title="stale triage marker", triage=True)
        blocked_id = kb.create_task(conn, title="stale blocked marker", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', block_kind='needs_input' WHERE id=?",
                (ready_id,),
            )
            conn.execute(
                "UPDATE tasks SET block_kind='needs_input' WHERE id=?",
                (triage_id,),
            )
            conn.execute(
                "UPDATE tasks SET status='blocked', block_kind='needs_input' WHERE id=?",
                (blocked_id,),
            )

        assert kb.claim_task(conn, ready_id, claimer="worker") is None
        assert kb.get_task(conn, ready_id).status == "ready"

        kb.recompute_ready(conn)
        assert kb.get_task(conn, blocked_id).status == "blocked"
        ok, error = kb.promote_task(
            conn, blocked_id, actor="operator:test", reason="wrong verb"
        )
        assert ok is False
        assert error == "task needs explicit unblock before promotion"

        assert triage_id not in decomp.list_triage_ids()
        assert not kb.specify_triage_task(
            conn, triage_id, body="must not specify", assignee="worker"
        )
        assert kb.decompose_triage_task(
            conn,
            triage_id,
            root_assignee="worker",
            children=[{"title": "must not exist", "parents": []}],
        ) is None
        outcome = decomp.decompose_task(triage_id)
        assert outcome.ok is False
        assert outcome.reason == "task needs explicit unblock before decomposition"
        assert kb.get_task(conn, triage_id).status == "triage"


def test_timeout_stale_and_crash_recovery_land_needs_input_in_blocked(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    old = int(time.time()) - 7200
    local_lock = kb._claimer_id()

    with kb.connect_closing() as conn:
        timeout_id = _running_task(conn, title="timeout recovery")
        stale_id = _running_task(conn, title="heartbeat recovery")
        crash_id = _running_task(conn, title="crash recovery")

        with kb.write_txn(conn):
            for task_id in (timeout_id, stale_id, crash_id):
                task = kb.get_task(conn, task_id)
                conn.execute(
                    "UPDATE tasks SET block_kind='needs_input', started_at=?, "
                    "last_heartbeat_at=NULL, claim_lock=? WHERE id=?",
                    (old, local_lock, task_id),
                )
                conn.execute(
                    "UPDATE task_runs SET started_at=?, last_heartbeat_at=NULL, claim_lock=? "
                    "WHERE id=?",
                    (old, local_lock, task.current_run_id),
                )
            conn.execute(
                "UPDATE tasks SET max_runtime_seconds=1, worker_pid=91001 WHERE id=?",
                (timeout_id,),
            )
            conn.execute(
                "UPDATE tasks SET worker_pid=NULL WHERE id=?",
                (stale_id,),
            )
            conn.execute(
                "UPDATE tasks SET worker_pid=91003, last_heartbeat_at=? WHERE id=?",
                (int(time.time()), crash_id),
            )

        assert kb.enforce_max_runtime(
            conn, signal_fn=lambda _pid, _sig: None
        ) == [timeout_id]
        assert kb.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=lambda _pid, _sig: None
        ) == [stale_id]
        assert kb.detect_crashed_workers(conn) == [crash_id]

        for task_id in (timeout_id, stale_id, crash_id):
            task = kb.get_task(conn, task_id)
            assert task.status == "blocked"
            assert task.block_kind == "needs_input"
            assert kb.claim_task(conn, task_id, claimer="worker") is None
