"""Red E2E contracts for durable Kanban publication and worker supervision.

These regressions intentionally cross the public CLI, SQLite store, dispatcher,
and dashboard projection boundaries.  They describe the incident contract before
the production fix; do not replace them with source-text or mock-only assertions.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


_REPO = Path(__file__).resolve().parents[2]
_UNKNOWN_SKILL = "contract-skill-that-does-not-exist"


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    conn = kbc.connect(home / "kanban.db")
    try:
        yield conn, home
    finally:
        conn.close()


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("tasks", "task_links", "task_events", "task_runs")
    }


def _cli(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update({"HERMES_HOME": str(home), "PYTHONPATH": str(_REPO)})
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _cli_json(home: Path, *args: str):
    result = _cli(home, *args, "--json")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _dashboard_task_dict(task: kb.Task) -> dict:
    plugin_file = _REPO / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_contract_projection", plugin_file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._task_dict(task)


def _task(conn: sqlite3.Connection, task_id: str) -> kb.Task:
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


def _expire(conn: sqlite3.Connection, task_id: str, *, heartbeat_age: int = 7200) -> None:
    old = int(time.time()) - heartbeat_age
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at=?, claim_expires=?, last_heartbeat_at=? WHERE id=?",
            (old, old, old, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET started_at=?, claim_expires=?, last_heartbeat_at=? "
            "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
            (old, old, old, task_id),
        )


def test_cli_rejects_unknown_profile_skill_before_any_publication(board):
    conn, home = board
    profile_home = home / "profiles" / "worker"
    profile_home.mkdir(parents=True)
    before = _counts(conn)

    result = _cli(
        home,
        "create",
        "invalid explicit skill",
        "--assignee",
        "worker",
        "--skill",
        _UNKNOWN_SKILL,
        "--json",
    )

    after = _counts(conn)
    message = result.stdout + result.stderr
    assert result.returncode != 0
    assert "worker" in message and _UNKNOWN_SKILL in message
    assert after == before, "validation failure must leave no task/link/event/run residue"


def test_dispatcher_classifies_legacy_unknown_skill_without_claim_or_spawn(board, monkeypatch):
    conn, _ = board
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    task_id = kb.create_task(
        conn,
        title="legacy externally inserted invalid skill",
        assignee="worker",
        skills=[_UNKNOWN_SKILL],
    )
    spawned: list[str] = []

    result = kbd.dispatch_once(
        conn,
        spawn_fn=lambda task, _workspace: spawned.append(task.id) or os.getpid(),
        max_spawn=1,
    )

    task = kb.get_task(conn, task_id)
    runs = kb.list_runs(conn, task_id)
    assert spawned == [] and result.spawned == []
    assert task is not None and task.status != "running"
    assert task.last_failure_error and "worker" in task.last_failure_error
    assert _UNKNOWN_SKILL in task.last_failure_error
    assert runs and runs[-1].ended_at is not None and runs[-1].outcome == "spawn_failed"


def test_connected_dag_rejects_cycles_and_orchestrator_descendant_wait_atomically(board):
    conn, _ = board
    root = kb.create_task(conn, title="orchestrator", assignee="worker")
    child = kb.create_task(conn, title="implementation", assignee="worker", parents=[root], creator_task_id=root)
    before_links = int(conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0])

    with pytest.raises(ValueError, match="itself"):
        kb.link_tasks(conn, root, root)
    with pytest.raises(ValueError, match="cycle"):
        kb.link_tasks(conn, child, root)
    assert int(conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0]) == before_links

    claimed = kb.claim_task(conn, root)
    assert claimed is not None
    descendant_blocked = kb.block_task(conn, root, kind="dependency", reason=f"waiting for child {child}")
    assert _task(conn, root).status == "needs_user_action"
    assert kb.unblock_task(conn, root)
    assert kb.claim_task(conn, root) is not None

    assert kb.complete_task(conn, root, result="orchestration published")
    kb.recompute_ready(conn)
    assert _task(conn, child).status == "ready"
    created = next(e for e in kb.list_events(conn, child) if e.kind == "created")
    assert created.payload["creator_task_id"] == root
    assert descendant_blocked is False, "an orchestrator must complete, never block on its own descendant"


@pytest.mark.parametrize(
    ("failure_kind", "worker_pid", "heartbeat_age", "worker_started_at"),
    [
        ("dead-pid", 999_999_991, 0, None),
        ("stale-heartbeat", os.getpid(), 7200, None),
        ("fingerprint-mismatch", os.getpid(), 0, "foreign-boot:1|1"),
    ],
)
def test_cli_list_and_show_share_honest_running_projection(
    board, failure_kind, worker_pid, heartbeat_age, worker_started_at
):
    conn, home = board
    task_id = kb.create_task(conn, title=failure_kind, assignee="worker", creator_task_id="t_origin")
    kb.claim_task(conn, task_id)
    kbd._set_worker_pid(conn, task_id, worker_pid)
    if heartbeat_age:
        _expire(conn, task_id, heartbeat_age=heartbeat_age)
    if worker_started_at:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET worker_started_at=? WHERE id=?", (worker_started_at, task_id))

    listed = next(item for item in _cli_json(home, "list") if item["id"] == task_id)
    shown = _cli_json(home, "show", task_id)["task"]

    for projected in (listed, shown):
        assert projected["status"] == "running"
        assert projected["operational_status"] == "recovering"
        assert projected["creator_task_id"] == "t_origin"
        assert projected["root_task_id"] == "t_origin"


def test_dead_worker_is_reclaimed_with_bounded_retry_and_never_projected_running(board, monkeypatch):
    conn, _ = board
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    task_id = kb.create_task(conn, title="crashing worker", assignee="worker", max_retries=2)
    kb.claim_task(conn, task_id)
    kbd._set_worker_pid(conn, task_id, 999_999_991)
    _expire(conn, task_id)

    projected_before = _dashboard_task_dict(_task(conn, task_id))
    crashed = kbd.detect_crashed_workers(conn)
    first = _task(conn, task_id)

    assert crashed == [task_id]
    assert first.status == "ready" and first.consecutive_failures == 1

    kb.claim_task(conn, task_id)
    kbd._set_worker_pid(conn, task_id, 999_999_992)
    _expire(conn, task_id)
    assert kbd.detect_crashed_workers(conn) == [task_id]
    final = _task(conn, task_id)
    assert final.status == "ready" and final.consecutive_failures == 2
    recovery = [e for e in kb.list_events(conn, task_id) if e.kind == "recovery_scheduled"][-1]
    assert recovery.payload["attempt"] == 2
    assert recovery.payload["deadline_at"] > recovery.created_at
    assert projected_before["status"] == "running"
    assert projected_before["operational_status"] == "recovering"


def test_stale_heartbeat_is_reclaimed_and_never_projected_running(board, monkeypatch):
    conn, _ = board
    task_id = kb.create_task(conn, title="wedged worker", assignee="worker")
    kb.claim_task(conn, task_id)
    kbd._set_worker_pid(conn, task_id, os.getpid())
    _expire(conn, task_id, heartbeat_age=7200)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **k: {"terminated": True})

    projected_before = _dashboard_task_dict(_task(conn, task_id))
    stale = kbd.detect_stale_running(conn, stale_timeout_seconds=3600, signal_fn=lambda *_: None)

    assert stale == [task_id]
    assert _task(conn, task_id).status == "ready"
    assert projected_before["status"] == "running"
    assert projected_before["operational_status"] == "recovering"


def test_recycled_pid_is_reclaimed_without_signal_and_never_projected_running(board):
    conn, _ = board
    task_id = kb.create_task(conn, title="recycled pid", assignee="worker")
    kb.claim_task(conn, task_id)
    kbd._set_worker_pid(conn, task_id, os.getpid())
    _expire(conn, task_id)
    foreign = "foreign-boot:1|1"
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at=? WHERE id=?", (foreign, task_id))
        conn.execute(
            "UPDATE task_runs SET worker_started_at=? WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
            (foreign, task_id),
        )
    signals: list[tuple[int, int]] = []

    projected_before = _dashboard_task_dict(_task(conn, task_id))
    reclaimed = kb.release_stale_claims(conn, signal_fn=lambda pid, sig: signals.append((pid, sig)))

    assert reclaimed == 1 and signals == []
    assert _task(conn, task_id).status == "ready"
    assert projected_before["status"] == "running"
    assert projected_before["operational_status"] == "recovering"


def test_continuation_remains_root_traceable_and_dependency_projected(board):
    conn, _ = board
    root = kb.create_task(conn, title="goal root", assignee="orchestrator", session_id="session-root")
    phase = kb.create_task(conn, title="phase", assignee="worker", parents=[root], creator_task_id=root)
    continuation = kb.create_task(
        conn,
        title="phase continuation",
        assignee="worker",
        parents=[phase],
        creator_task_id=phase,
    )

    projected = _dashboard_task_dict(_task(conn, continuation))
    assert projected["status"] == "todo"
    assert projected["operational_status"] == "dependency-wait"
    assert projected["root_task_id"] == root
    assert projected["continuation_of"] == phase
    assert _task(conn, continuation).session_id == "session-root"
