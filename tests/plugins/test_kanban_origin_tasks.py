"""GET /origin-tasks: the desktop's batched, profile-scoped origin lookup.

Same router-on-bare-FastAPI harness as the other kanban plugin tests. Boards are driven by the REAL
producers (``claim_task``, ``heartbeat_worker``, ``_set_worker_pid`` — including its real process
fingerprint), then read back through the route: activity must come from the current run's evidence,
never from a bare claim, a fingerprint mistaken for a time, or an earlier run's heartbeat.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_notify as kbn
from hermes_cli.kanban_origin import MAX_SEED_SESSIONS, origin_key
from hermes_state import SessionDB

URL = "/api/plugins/kanban/origin-tasks"


def _router():
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_kanban_origin_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(var, raising=False)
    kb.init_db()
    state = SessionDB(db_path=home / "state.db")
    state.create_session("origin", source="cli")
    state.close()
    return home


@pytest.fixture
def api(home):
    app = FastAPI()
    app.include_router(_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _linked(conn, title="t"):
    """A ready task the origin session ordered and is subscribed to (so the producer indexed it)."""
    task = kb.create_task(conn, title=title, assignee="default", session_id="origin")
    kbn.add_notify_sub(conn, task_id=task, platform="tui", chat_id="origin", notifier_profile="default")
    return task


def _activity(api):
    body = api.get(URL, params={"session_ids": "origin"}).json()
    return {r["task_id"]: (r["task"]["activity"], r["task"]["activity_evidence"]) for r in body["refs"]}


def test_activity_comes_from_the_current_runs_evidence(api):
    with kbc.connect_closing() as conn:
        spawned, beating, reserved, reclaimed, skewed = (_linked(conn, n) for n in "abcde")
        for task in (spawned, beating, reserved, reclaimed, skewed):
            assert kb.claim_task(conn, task) is not None  # opens a real run

        kbd._set_worker_pid(conn, spawned, os.getpid())  # real spawn: pid + process fingerprint + event
        fingerprint = conn.execute(
            "SELECT worker_started_at FROM task_runs WHERE task_id = ?", (spawned,)).fetchone()[0]
        kbd.heartbeat_worker(conn, beating)

        # An earlier run's heartbeat survives on the task row; its replacement run has no evidence at all.
        kbd.heartbeat_worker(conn, reclaimed)
        conn.execute("UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?", (int(time.time()) - 100, reclaimed))
        conn.execute("UPDATE task_runs SET ended_at = ?, outcome = 'crashed' WHERE task_id = ?",
                     (int(time.time()), reclaimed))
        conn.execute("UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL WHERE id = ?",
                     (reclaimed,))
        assert kb.claim_task(conn, reclaimed) is not None
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 10, reclaimed))

        # Beats in the future / before the epoch are not evidence.
        conn.execute("UPDATE task_runs SET last_heartbeat_at = ? WHERE task_id = ?", (int(time.time()) + 9999, skewed))
        conn.execute("UPDATE tasks SET last_heartbeat_at = -5 WHERE id = ?", (skewed,))
        conn.commit()

    # worker_started_at is a fingerprint ("<epoch>|<ticks>" / "unverified"), not a timestamp.
    assert isinstance(fingerprint, str)

    assert _activity(api) == {
        spawned: ("background", "spawn"),
        beating: ("background", "heartbeat"),
        reserved: ("reserved", "claim"),  # a live claim alone is a reservation
        reclaimed: ("unknown", "none"),  # old heartbeat + expired claim never implies a running worker
        skewed: ("reserved", "claim"),
    }

    # Evidence older than the claim window is stale, not running.
    old = int(time.time()) - 5 * kb._resolve_claim_ttl_seconds()
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE task_runs SET last_heartbeat_at = ? WHERE task_id = ?", (old, beating))
        conn.execute("UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?", (old, beating))
        conn.execute("UPDATE task_events SET created_at = ? WHERE task_id = ? AND kind = 'spawned'", (old, spawned))
        conn.commit()

    stale = _activity(api)
    assert stale[spawned] == ("stale", "spawn_old")
    assert stale[beating] == ("stale", "heartbeat_old")


def test_queue_wait_and_input_states_are_distinct(api):
    with kbc.connect_closing() as conn:
        waiting_on = kb.create_task(conn, title="parent", assignee="default")
        child = _linked(conn, "child")
        kb.link_tasks(conn, parent_id=waiting_on, child_id=child)
        blocked = _linked(conn, "blocked")
        asking = _linked(conn, "asking")
        done = _linked(conn, "done")
        for task, cols in ((blocked, "status = 'blocked', block_kind = 'transient'"),
                           (asking, "status = 'blocked', block_kind = 'needs_input'"),
                           (done, "status = 'done'")):
            conn.execute(f"UPDATE tasks SET {cols} WHERE id = ?", (task,))
        conn.commit()

    assert _activity(api) == {
        child: ("waiting", "parents"),
        blocked: ("blocked", "block_kind"),
        asking: ("needs-input", "block_kind"),
        done: ("done", "status"),
    }


def test_read_is_side_effect_free_and_tolerates_odd_metadata(api, home):
    with kbc.connect_closing() as conn:
        task = _linked(conn)
    store = SessionDB(db_path=home / "state.db")
    store.set_meta(origin_key("origin", "default", task), "7")  # a scalar where JSON object was written
    store.close()
    before = hashlib.sha256((home / "state.db").read_bytes()).hexdigest()

    body = api.get(URL, params={"session_ids": "origin"}).json()

    assert [(r["task_id"], r["evidence"], r["indexed_at"]) for r in body["refs"]] == [(task, "ok", 0)]
    assert hashlib.sha256((home / "state.db").read_bytes()).hexdigest() == before


def test_unknown_sessions_and_oversized_requests_are_reported_not_dropped(api):
    ids = ["origin", "not-in-this-profile"] + [f"x{n}" for n in range(MAX_SEED_SESSIONS)]

    body = api.get(URL, params={"session_ids": ",".join(ids)}).json()

    assert body["truncated"]["sessions"] is True
    assert "not-in-this-profile" in body["unknown_sessions"]
    assert body["refs"] == []


def test_an_unreadable_store_is_an_error_never_an_empty_list(api, home):
    (home / "state.db").write_bytes(b"this is not a sqlite database" * 64)
    for sidecar in ("-wal", "-shm"):
        (home / f"state.db{sidecar}").unlink(missing_ok=True)

    response = api.get(URL, params={"session_ids": "origin"})

    assert response.status_code >= 500
