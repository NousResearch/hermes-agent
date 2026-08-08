"""Tests for the kanban mid-run steer channel.

Covers the three layers the feature spans:

  * ``kanban_db.queue_steer`` / ``pop_pending_steer`` — the mailbox, its
    preconditions, and one-shot delivery.
  * ``tools.kanban_tools.fetch_pending_steer_from_env`` — the worker-side
    bridge, which must be inert outside a dispatcher-spawned worker.
  * ``POST /tasks/{task_id}/steer`` and ``POST /runs/{run_id}/steer`` — the
    REST surface.

The end-to-end path (queued steer reaching the model's tool result) is
covered by ``tests/run_agent/test_steer.py``, which already exercises
``_apply_pending_steer_to_tool_results``; this file stops at the seam where
the board hands text to ``AIAgent.steer()``.
"""

from __future__ import annotations

import importlib.util
import secrets
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    mod_name = "hermes_dashboard_plugin_kanban_steer_test"
    if mod_name in sys.modules:
        return sys.modules[mod_name].router

    spec = importlib.util.spec_from_file_location(mod_name, plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kb.connect()
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _running_task(conn, *, title="steer me", assignee="worker"):
    """Create a task in ``running`` with an open run, as the dispatcher does.

    Unlike the helper in ``test_kanban_worker_runs.py`` this also stamps
    ``tasks.current_run_id`` — ``queue_steer`` pins to it, because a steer
    that cannot name a run cannot be kept off a respawned attempt.
    """
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    lock = secrets.token_hex(8)
    future = int(time.time()) + 3600
    cur = conn.execute(
        "INSERT INTO task_runs "
        "(task_id, status, claim_lock, claim_expires, worker_pid, started_at) "
        "VALUES (?, 'running', ?, ?, ?, ?)",
        (task_id, lock, future, 4242, int(time.time())),
    )
    run_id = cur.lastrowid
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "worker_pid=?, current_run_id=? WHERE id=?",
        (lock, future, 4242, run_id, task_id),
    )
    conn.commit()
    return task_id, run_id


# ---------------------------------------------------------------------------
# kanban_db.queue_steer
# ---------------------------------------------------------------------------

def test_queue_steer_returns_live_run_id(conn):
    task_id, run_id = _running_task(conn)
    assert kb.queue_steer(conn, task_id, "check the migrations too") == run_id


def test_queue_steer_rejects_non_running_task(conn):
    task_id = kb.create_task(conn, title="not started", assignee="worker")
    with pytest.raises(ValueError, match="not running"):
        kb.queue_steer(conn, task_id, "too early")


def test_queue_steer_rejects_unknown_task(conn):
    with pytest.raises(ValueError, match="unknown task"):
        kb.queue_steer(conn, "t_nope", "hello")


def test_queue_steer_rejects_empty_text(conn):
    task_id, _ = _running_task(conn)
    with pytest.raises(ValueError, match="required"):
        kb.queue_steer(conn, task_id, "   ")


def test_queue_steer_rejects_oversized_text(conn):
    task_id, _ = _running_task(conn)
    with pytest.raises(ValueError, match="limit"):
        kb.queue_steer(conn, task_id, "x" * (kb._STEER_MAX_CHARS + 1))


def test_queue_steer_rejects_stale_run_id(conn):
    """A steer written against a reclaimed attempt must not land on its successor."""
    task_id, run_id = _running_task(conn)
    with pytest.raises(ValueError, match="no longer the live run"):
        kb.queue_steer(conn, task_id, "stale", expected_run_id=run_id - 1)


def test_queue_steer_caps_undelivered_backlog(conn):
    task_id, _ = _running_task(conn)
    for i in range(kb._STEER_MAX_PENDING):
        kb.queue_steer(conn, task_id, f"note {i}")
    with pytest.raises(ValueError, match="already queued undelivered"):
        kb.queue_steer(conn, task_id, "one too many")


# ---------------------------------------------------------------------------
# kanban_db.pop_pending_steer
# ---------------------------------------------------------------------------

def test_pop_returns_none_when_empty(conn):
    task_id, run_id = _running_task(conn)
    assert kb.pop_pending_steer(conn, task_id, run_id=run_id) is None


def test_pop_joins_multiple_steers_in_order(conn):
    task_id, run_id = _running_task(conn)
    kb.queue_steer(conn, task_id, "first")
    kb.queue_steer(conn, task_id, "second")
    assert kb.pop_pending_steer(conn, task_id, run_id=run_id) == "first\nsecond"


def test_pop_is_one_shot(conn):
    """Delivery must not repeat — the model has already been shown the text."""
    task_id, run_id = _running_task(conn)
    kb.queue_steer(conn, task_id, "only once")
    assert kb.pop_pending_steer(conn, task_id, run_id=run_id) == "only once"
    assert kb.pop_pending_steer(conn, task_id, run_id=run_id) is None


def test_pop_ignores_other_runs(conn):
    task_id, run_id = _running_task(conn)
    kb.queue_steer(conn, task_id, "for the first attempt")
    assert kb.pop_pending_steer(conn, task_id, run_id=run_id + 1) is None


def test_steer_lifecycle_is_visible_in_events(conn):
    task_id, run_id = _running_task(conn)
    kb.queue_steer(conn, task_id, "look at auth.log")
    kb.pop_pending_steer(conn, task_id, run_id=run_id)
    kinds = [e.kind for e in kb.list_events(conn, task_id)]
    assert "steer_queued" in kinds
    assert "steer_delivered" in kinds


# ---------------------------------------------------------------------------
# tools.kanban_tools.fetch_pending_steer_from_env
# ---------------------------------------------------------------------------

def _reset_steer_poll_rate_limit():
    import tools.kanban_tools as kt
    kt._steer_poll_last_attempt = 0.0


def test_bridge_is_inert_outside_worker(conn, monkeypatch):
    """A normal chat session must never touch the board's steer mailbox."""
    from tools.kanban_tools import fetch_pending_steer_from_env

    task_id, _ = _running_task(conn)
    kb.queue_steer(conn, task_id, "not for you")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    _reset_steer_poll_rate_limit()
    assert fetch_pending_steer_from_env() is None


def test_bridge_requires_a_run_id(conn, monkeypatch):
    """Without a run id we cannot tell this attempt's steers from a stale one's."""
    from tools.kanban_tools import fetch_pending_steer_from_env

    task_id, _ = _running_task(conn)
    kb.queue_steer(conn, task_id, "ambiguous")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    _reset_steer_poll_rate_limit()
    assert fetch_pending_steer_from_env() is None


def test_bridge_delivers_for_the_live_run(conn, monkeypatch):
    from tools.kanban_tools import fetch_pending_steer_from_env

    task_id, run_id = _running_task(conn)
    kb.queue_steer(conn, task_id, "prefer the smaller diff")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    _reset_steer_poll_rate_limit()
    assert fetch_pending_steer_from_env() == "prefer the smaller diff"


def test_bridge_is_rate_limited(conn, monkeypatch):
    """Back-to-back tool batches must not each pay a DB round trip."""
    from tools.kanban_tools import fetch_pending_steer_from_env

    task_id, run_id = _running_task(conn)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    _reset_steer_poll_rate_limit()
    assert fetch_pending_steer_from_env() is None  # consumes the budget
    kb.queue_steer(conn, task_id, "arrives on the next poll")
    assert fetch_pending_steer_from_env() is None


# ---------------------------------------------------------------------------
# agent bridge — the seam into AIAgent.steer()
# ---------------------------------------------------------------------------

class _StubAgent:
    def __init__(self):
        self.steered = []

    def steer(self, text):
        self.steered.append(text)
        return True


def test_agent_bridge_routes_board_steer_through_agent_steer(conn, monkeypatch):
    from agent.agent_runtime_helpers import _pull_board_steer

    task_id, run_id = _running_task(conn)
    kb.queue_steer(conn, task_id, "stop after the next step")
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    _reset_steer_poll_rate_limit()

    agent = _StubAgent()
    _pull_board_steer(agent)
    assert agent.steered == ["stop after the next step"]


def test_agent_bridge_swallows_bridge_failures(monkeypatch):
    """The agent loop must survive a broken board — never raise into it."""
    import tools.kanban_tools as kt
    from agent.agent_runtime_helpers import _pull_board_steer

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_gone")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.setattr(
        kt, "fetch_pending_steer_from_env", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    _pull_board_steer(_StubAgent())  # must not raise


# ---------------------------------------------------------------------------
# REST — POST /tasks/{task_id}/steer
# ---------------------------------------------------------------------------

def test_post_task_steer_queues(client, conn):
    task_id, run_id = _running_task(conn)
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/steer",
        json={"text": "use the existing helper"},
    )
    assert r.status_code == 200
    assert r.json() == {
        "ok": True,
        "task_id": task_id,
        "run_id": run_id,
        "status": "queued",
    }


def test_post_task_steer_404_unknown_task(client):
    r = client.post(
        "/api/plugins/kanban/tasks/t_missing/steer", json={"text": "hello"}
    )
    assert r.status_code == 404


def test_post_task_steer_409_not_running(client, conn):
    """The fallback for a non-running task is a comment, not a steer."""
    task_id = kb.create_task(conn, title="queued", assignee="worker")
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/steer", json={"text": "hello"}
    )
    assert r.status_code == 409
    assert "not running" in r.json()["detail"]


def test_post_task_steer_400_empty_text(client, conn):
    task_id, _ = _running_task(conn)
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/steer", json={"text": "  "}
    )
    assert r.status_code == 400


def test_post_task_steer_409_stale_run_guard(client, conn):
    task_id, run_id = _running_task(conn)
    r = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/steer",
        json={"text": "stale", "run_id": run_id - 1},
    )
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# REST — POST /runs/{run_id}/steer
# ---------------------------------------------------------------------------

def test_post_run_steer_queues(client, conn):
    task_id, run_id = _running_task(conn)
    r = client.post(
        f"/api/plugins/kanban/runs/{run_id}/steer",
        json={"text": "narrow the scope"},
    )
    assert r.status_code == 200
    assert r.json()["task_id"] == task_id


def test_post_run_steer_404_unknown_run(client):
    r = client.post("/api/plugins/kanban/runs/999999/steer", json={"text": "x"})
    assert r.status_code == 404


def test_post_run_steer_409_ended_run(client, conn):
    task_id, run_id = _running_task(conn)
    conn.execute(
        "UPDATE task_runs SET ended_at = ? WHERE id = ?", (int(time.time()), run_id)
    )
    conn.commit()
    r = client.post(
        f"/api/plugins/kanban/runs/{run_id}/steer", json={"text": "too late"}
    )
    assert r.status_code == 409
    assert "already ended" in r.json()["detail"]
