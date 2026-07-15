from __future__ import annotations

import json
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def isolated_board(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    return db_path


def _run(conn, task_id, *, ended_at=None, outcome=None, heartbeat=None):
    cur = conn.execute(
        "INSERT INTO task_runs(task_id,profile,status,started_at,ended_at,outcome,last_heartbeat_at,max_runtime_seconds) VALUES (?,'implementer',?,?,?,?,?,3600)",
        (task_id, "completed" if ended_at is not None else "running", int(time.time()) - 30, ended_at, outcome, heartbeat),
    )
    return int(cur.lastrowid)


def _flatten(roots):
    seen, stack = [], list(roots)
    while stack:
        node = stack.pop(); seen.append(node); stack.extend(node["children"])
    return seen


def test_empty_snapshot_is_stable_and_read_only(isolated_board):
    before = isolated_board.read_bytes()
    assert kb.get_activity_snapshot(board="default")["roots"] == []
    assert before == isolated_board.read_bytes()


def test_pinned_database_cannot_be_relabelled_as_another_board(isolated_board):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Default private task", assignee="worker")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,)); _run(conn, task_id, heartbeat=int(time.time())); conn.commit()
    with pytest.raises(PermissionError):
        kb.get_activity_snapshot(board="alpha")
    assert kb.get_activity_snapshot(board="default")["roots"][0]["title"] == "Default private task"


def test_projection_is_allowlisted_and_redacts_block_reason(isolated_board):
    now = int(time.time())
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="Parent", assignee="architect")
        child = kb.create_task(conn, title="Child", assignee="implementer", parents=[parent])
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,)); conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (child,))
        parent_run = _run(conn, parent, ended_at=now, outcome="completed", heartbeat=now - 20); child_run = _run(conn, child, heartbeat=now - 5)
        conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES (?,'blocked',?,?)", (child, json.dumps({"reason": "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"}), now)); conn.commit()
    root = kb.get_activity_snapshot(board="default")["roots"][0]; node = root["children"][0]
    assert root["run"]["run_id"] == parent_run and node["run"]["run_id"] == child_run
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in node["block_reason"]
    assert "body" not in node and "worker_pid" not in node["run"] and "error" not in node["run"]


def test_old_completed_and_backlog_only_work_are_hidden(isolated_board):
    now = int(time.time())
    with kb.connect_closing() as conn:
        recent = kb.create_task(conn, title="Recent", assignee="worker"); old = kb.create_task(conn, title="Old", assignee="worker")
        conn.execute("UPDATE tasks SET status='done' WHERE id IN (?,?)", (recent, old)); _run(conn, recent, ended_at=now - 5, outcome="completed"); _run(conn, old, ended_at=now - kb.ACTIVITY_RECENT_COMPLETION_SECONDS - 10, outcome="completed"); conn.commit()
    ids = [node["task_id"] for node in kb.get_activity_snapshot(board="default")["roots"]]
    assert recent in ids and old not in ids
    with kb.connect_closing() as conn:
        conn.execute("UPDATE task_runs SET ended_at=? WHERE task_id=?", (now - 1000, recent))
        for status in ("triage", "todo", "scheduled", "ready", "review"):
            task_id = kb.create_task(conn, title=status, assignee="worker"); conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()
    assert kb.get_activity_snapshot(board="default")["roots"] == []


def test_connected_statuses_and_multi_parent_edges_are_retained(isolated_board):
    now = int(time.time())
    with kb.connect_closing() as conn:
        a = kb.create_task(conn, title="Parent A", assignee="worker"); b = kb.create_task(conn, title="Parent B", assignee="worker")
        shared = kb.create_task(conn, title="Shared", assignee="reviewer", parents=[a, b]); scheduled = kb.create_task(conn, title="Scheduled", assignee="worker", parents=[shared])
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (a,)); conn.execute("UPDATE tasks SET status='review' WHERE id=?", (shared,)); conn.execute("UPDATE tasks SET status='scheduled' WHERE id=?", (scheduled,)); _run(conn, a, heartbeat=now); conn.commit()
    nodes = _flatten(kb.get_activity_snapshot(board="default")["roots"]); node = next(n for n in nodes if n["task_id"] == shared)
    assert set(node["parents"]) == {a, b} and node["status"] == "review" and node["children"][0]["status"] == "scheduled"


def test_recursive_graph_expansion_is_capped_and_keeps_external_parent(isolated_board):
    with kb.connect_closing() as conn:
        ids = [kb.create_task(conn, title=f"Node {i}", assignee="worker") for i in range(kb.ACTIVITY_MAX_TASKS + 30)]
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (ids[0],)); _run(conn, ids[0], heartbeat=int(time.time()))
        conn.executemany("INSERT INTO task_links(parent_id,child_id) VALUES (?,?)", zip(ids, ids[1:]))
        omitted = kb.create_task(conn, title="Omitted parent", assignee="worker"); conn.execute("INSERT INTO task_links(parent_id,child_id) VALUES (?,?)", (omitted, ids[0])); conn.commit()
    snap = kb.get_activity_snapshot(board="default"); nodes = _flatten(snap["roots"])
    assert len(nodes) <= kb.ACTIVITY_MAX_TASKS and snap["truncated"] is True
    assert omitted in next(n for n in nodes if n["task_id"] == ids[0])["parents"]


def test_parent_edge_cap_is_per_child(isolated_board):
    now = int(time.time())
    with kb.connect_closing() as conn:
        a = kb.create_task(conn, title="Child A", assignee="worker"); z = kb.create_task(conn, title="Child Z", assignee="worker")
        conn.execute("UPDATE tasks SET status='running' WHERE id IN (?,?)", (a, z)); _run(conn, a, heartbeat=now); _run(conn, z, heartbeat=now)
        parents = [kb.create_task(conn, title=f"Parent {i}", assignee="worker") for i in range(kb.ACTIVITY_MAX_PARENTS_PER_TASK + 4)]; later = kb.create_task(conn, title="Later", assignee="worker")
        conn.executemany("INSERT INTO task_links(parent_id,child_id) VALUES (?,?)", [(p, a) for p in parents] + [(later, z)]); conn.commit()
    snap = kb.get_activity_snapshot(board="default"); nodes = _flatten(snap["roots"])
    assert len(next(n for n in nodes if n["task_id"] == a)["parents"]) == kb.ACTIVITY_MAX_PARENTS_PER_TASK
    assert next(n for n in nodes if n["task_id"] == z)["parents"] == [later] and snap["truncated"] is True


def test_graph_scan_cap_sets_truncation(isolated_board):
    count = kb.ACTIVITY_MAX_GRAPH_SCAN + 5; ids = [f"bulk-{i:04d}" for i in range(count)]; now = int(time.time())
    with kb.connect_closing() as conn:
        conn.executemany("INSERT INTO tasks(id,title,assignee,status,created_at) VALUES (?,?,'worker',?,?)", [(x, x, "running" if i == 0 else "todo", now + i) for i, x in enumerate(ids)])
        _run(conn, ids[0], heartbeat=now); conn.executemany("INSERT INTO task_links(parent_id,child_id) VALUES (?,?)", zip(ids, ids[1:])); conn.commit()
    snap = kb.get_activity_snapshot(board="default")
    assert snap["truncated"] is True and len(_flatten(snap["roots"])) <= kb.ACTIVITY_MAX_TASKS


def test_cycles_fail_boundedly_without_duplicates(isolated_board):
    with kb.connect_closing() as conn:
        a = kb.create_task(conn, title="A", assignee="worker"); b = kb.create_task(conn, title="B", assignee="worker")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (a,)); _run(conn, a, heartbeat=int(time.time())); conn.execute("INSERT INTO task_links VALUES (?,?)", (a, b)); conn.execute("INSERT INTO task_links VALUES (?,?)", (b, a)); conn.commit()
    seen = [n["task_id"] for n in _flatten(kb.get_activity_snapshot(board="default")["roots"])]
    assert sorted(seen) == sorted([a, b]) and len(seen) == len(set(seen))


def test_missing_board_does_not_get_created(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"; monkeypatch.setenv("HERMES_HOME", str(home)); monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    target = home / "kanban" / "boards" / "missing" / "kanban.db"
    with pytest.raises(Exception): kb.get_activity_snapshot(board="missing")
    assert not target.exists()
