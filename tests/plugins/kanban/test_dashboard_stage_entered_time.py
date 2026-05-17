"""Tests for the per-stage timestamp (``entered_status_at``) on the
Kanban dashboard API.

The feature adds a second timestamp alongside ``created_at`` so the UI can
render "created 3h ago / in this column 12m ago" on every card. The value
is derived from ``task_events`` rather than a new column, so these tests
exercise the SQL helper directly *and* through the REST surface:

  * GET /board                — batch path (one aggregate query for all tasks)
  * GET /tasks/{id}           — single-task path
  * PATCH /tasks/{id}         — verify the patched response also includes it
  * Fallback for fresh todo / triage tasks (no entry event yet)

Layout mirrors ``tests/plugins/test_kanban_dashboard_plugin.py``: dynamic
import of plugin_api.py, isolated HERMES_HOME, bare-FastAPI test client.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures — same isolation pattern as the sibling archive-API test file.
# ---------------------------------------------------------------------------


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[3]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_stage_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin_mod():
    return _load_plugin_router()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home, plugin_mod):
    app = FastAPI()
    app.include_router(plugin_mod.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def _create_task(client, title="t", assignee=None, parents=None):
    payload = {"title": title}
    if assignee is not None:
        payload["assignee"] = assignee
    if parents is not None:
        payload["parents"] = parents
    r = client.post("/api/plugins/kanban/tasks", json=payload)
    assert r.status_code == 200, r.text
    return r.json()["task"]


def _get_task(client, task_id):
    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200, r.text
    return r.json()["task"]


def _board_tasks(client):
    """Flatten /board response into a {task_id: task_dict} map."""
    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200, r.text
    out: dict[str, dict] = {}
    for col in r.json()["columns"]:
        for t in col["tasks"]:
            out[t["id"]] = t
    return out


# ---------------------------------------------------------------------------
# 1. Promoted: ready status uses the "promoted" event timestamp.
# ---------------------------------------------------------------------------


def test_entered_status_at_ready_uses_promoted_event(client):
    """A no-parent task is created -> immediately promoted to 'ready'.

    ``entered_status_at`` should reflect the ``promoted`` event timestamp
    rather than ``created_at`` (even though they are usually equal here,
    they are distinct columns in ``task_events`` and the code path differs).
    """
    t = _create_task(client, "ready task")
    assert t["status"] == "ready"

    # /tasks/:id surfaces entered_status_at on the full detail view.
    detail = _get_task(client, t["id"])
    assert detail["entered_status_at"] is not None
    assert isinstance(detail["entered_status_at"], int)
    # Derived age block should also include entered_status_age_seconds.
    age = detail.get("age") or {}
    assert "entered_status_age_seconds" in age

    # /board surfaces the same value (batch path).
    board_tasks = _board_tasks(client)
    assert t["id"] in board_tasks
    assert board_tasks[t["id"]]["entered_status_at"] == detail["entered_status_at"]


# ---------------------------------------------------------------------------
# 2. Claimed: running status uses the "claimed" event timestamp,
#    NOT the older "promoted" timestamp.
# ---------------------------------------------------------------------------


def test_entered_status_at_running_distinct_from_promoted(client, kanban_home):
    """Claiming a task should advance ``entered_status_at`` past the
    ``promoted`` timestamp."""
    t = _create_task(client, "claim me", assignee="worker-a")
    assert t["status"] == "ready"

    promoted_ts = _get_task(client, t["id"])["entered_status_at"]
    assert promoted_ts is not None

    # Sleep at least 1s so the new event has a strictly later created_at
    # (kanban_db stores integer seconds).
    time.sleep(1.1)

    # Claim atomically by talking to kanban_db directly (the dashboard
    # PATCH endpoint rejects status=running on purpose; see
    # test_patch_status_running_rejected — claims must go through
    # claim_task so a task_runs row is created with the lock).
    conn = kb.connect()
    try:
        claimed = kb.claim_task(conn, t["id"], claimer="worker-a")
    finally:
        conn.close()
    assert claimed is not None, "claim_task should succeed on a ready task"
    assert claimed.status == "running"

    detail = _get_task(client, t["id"])
    assert detail["status"] == "running"
    running_ts = detail["entered_status_at"]
    assert running_ts is not None
    assert running_ts >= promoted_ts, (
        f"running entered_status_at ({running_ts}) must be >= promoted ({promoted_ts})"
    )
    # And strictly greater because we slept 1.1s between the two events.
    assert running_ts > promoted_ts

    # Batch path on /board agrees with the single-task path.
    board_tasks = _board_tasks(client)
    assert board_tasks[t["id"]]["entered_status_at"] == running_ts


# ---------------------------------------------------------------------------
# 3. Completed: done status uses the "completed" event timestamp.
# ---------------------------------------------------------------------------


def test_entered_status_at_done_uses_completed_event(client, kanban_home):
    t = _create_task(client, "ship it")
    assert t["status"] == "ready"

    time.sleep(1.1)

    # Done = PATCH /tasks/:id with status=done; this routes through
    # complete_task internally and writes a 'completed' event.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "done", "result": "shipped"},
    )
    assert r.status_code == 200, r.text
    patched = r.json()["task"]
    assert patched["status"] == "done"
    # PATCH response itself should carry entered_status_at for the new status.
    assert patched["entered_status_at"] is not None

    detail = _get_task(client, t["id"])
    assert detail["entered_status_at"] == patched["entered_status_at"]
    # Strictly later than the original ready timestamp.
    assert detail["entered_status_at"] > t["created_at"]


# ---------------------------------------------------------------------------
# 4. Blocked: status=blocked uses the "blocked" event timestamp.
# ---------------------------------------------------------------------------


def test_entered_status_at_blocked_uses_blocked_event(client):
    t = _create_task(client, "needs input")
    time.sleep(1.1)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "blocked", "block_reason": "waiting on x"},
    )
    assert r.status_code == 200, r.text
    blocked = r.json()["task"]
    assert blocked["status"] == "blocked"
    assert blocked["entered_status_at"] is not None
    assert blocked["entered_status_at"] > t["created_at"]


# ---------------------------------------------------------------------------
# 5. Fresh todo: no entry event yet -> falls back to created_at.
# ---------------------------------------------------------------------------


def test_entered_status_at_todo_falls_back_to_created_at(client):
    """A child whose parent isn't done sits in ``todo`` with no
    ``restored`` / ``claim_rejected`` event in its history. The helper
    should return ``created_at`` for that case (NOT None and NOT the
    promoted/claimed timestamp from some sibling task)."""
    parent = _create_task(client, "parent")
    child = _create_task(client, "child", parents=[parent["id"]])
    assert child["status"] == "todo"

    detail = _get_task(client, child["id"])
    assert detail["entered_status_at"] == child["created_at"]


# ---------------------------------------------------------------------------
# 6. Batch path: list-endpoint correctly distinguishes per-task timestamps
#    when several tasks live in different statuses simultaneously.
# ---------------------------------------------------------------------------


def test_batch_entered_status_at_does_not_leak_between_tasks(client, kanban_home):
    """Three tasks: one ready, one done, one blocked. Each must report its
    OWN status's event timestamp via the /board (batch) path — no
    cross-pollination between task_ids."""
    a = _create_task(client, "ready-task")           # ends up ready
    b = _create_task(client, "done-task")            # will be marked done
    c = _create_task(client, "blocked-task")         # will be marked blocked

    time.sleep(1.1)
    rb = client.patch(
        f"/api/plugins/kanban/tasks/{b['id']}",
        json={"status": "done", "result": "ok"},
    )
    assert rb.status_code == 200, rb.text
    done_entered = rb.json()["task"]["entered_status_at"]

    time.sleep(1.1)
    rc = client.patch(
        f"/api/plugins/kanban/tasks/{c['id']}",
        json={"status": "blocked", "block_reason": "x"},
    )
    assert rc.status_code == 200, rc.text
    blocked_entered = rc.json()["task"]["entered_status_at"]

    board_tasks = _board_tasks(client)
    ta, tb, tc = board_tasks[a["id"]], board_tasks[b["id"]], board_tasks[c["id"]]
    assert ta["status"] == "ready"
    assert tb["status"] == "done"
    assert tc["status"] == "blocked"
    assert ta["entered_status_at"] is not None
    assert tb["entered_status_at"] == done_entered
    assert tc["entered_status_at"] == blocked_entered
    # The done & blocked timestamps must not equal the ready task's
    # promoted timestamp (would indicate a row-mixing bug in the
    # batch query).
    assert tb["entered_status_at"] != ta["entered_status_at"]
    assert tc["entered_status_at"] != ta["entered_status_at"]
    assert tc["entered_status_at"] > tb["entered_status_at"]


# ---------------------------------------------------------------------------
# 7. Direct helper unit test: _batch_entered_status_at and the
#    single-task helper agree on the same inputs.
# ---------------------------------------------------------------------------


def test_batch_and_single_helpers_agree(client, plugin_mod, kanban_home):
    """If batch and per-task helpers disagree, the dashboard would show
    different ages depending on which endpoint loaded the card. Guard
    that they stay in sync."""
    t1 = _create_task(client, "t1")
    t2 = _create_task(client, "t2")
    time.sleep(1.1)
    client.patch(
        f"/api/plugins/kanban/tasks/{t2['id']}",
        json={"status": "blocked", "block_reason": "y"},
    )

    conn = kb.connect()
    try:
        tasks = kb.list_tasks(conn)
        batch_map = plugin_mod._batch_entered_status_at(conn, tasks)
        for task in tasks:
            single = plugin_mod._entered_status_at_for_task(
                conn, task.id, task.status, task.created_at,
            )
            assert batch_map[task.id] == single, (
                f"task {task.id} status={task.status}: "
                f"batch={batch_map[task.id]} single={single}"
            )
    finally:
        conn.close()
