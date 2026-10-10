"""HTTP surface for governed WAIT_AND_RESUME episodes:
POST /tasks/{id}/schedule and POST /tasks/{id}/unblock (typed refusals).

The router is attached to a bare FastAPI app (as in
``test_kanban_dashboard_plugin.py``); the session-token gate lives in
``hermes_cli.web_server`` and is exercised through the real app at the end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_episode as kbe

API = "/api/plugins/kanban"


def _load_plugin_router():
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_kanban_episode_test", plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
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
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix=API)
    return TestClient(app)


def _task(status_running: bool = False) -> tuple[str, int | None]:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="worker")
        if not status_running:
            return tid, None
        return tid, int(kb.claim_task(conn, tid, claimer="testhost:1").current_run_id)


def _status(tid: str) -> str:
    with kbc.connect() as conn:
        return kb.get_task(conn, tid).status


def _code(r) -> str:
    return r.json()["detail"]["code"]


def test_pre_launch_schedule_then_unblock(client):
    tid, _ = _task()
    r = client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "PRE_LAUNCH", "reason": "capacity"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["status"] == "scheduled" and body["task_id"] == tid
    token = body["episode_token"]
    assert token.startswith(f"hkep1.{tid}.")
    assert _status(tid) == "scheduled"

    r = client.post(f"{API}/tasks/{tid}/unblock", json={"expected_episode_token": token})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "task_id": tid, "status": "ready"}

    r = client.post(f"{API}/tasks/{tid}/unblock", json={"expected_episode_token": token})
    assert r.status_code == 409 and _code(r) == "EPISODE_STALE"
    assert r.json()["detail"]["current_status"] == "ready"


def test_mid_run_schedule_exact_run(client):
    tid, run_id = _task(status_running=True)
    r = client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "MID_RUN", "expected_run_id": run_id + 99})
    assert r.status_code == 409 and _code(r) == "RUN_MISMATCH"
    assert _status(tid) == "running"
    r = client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "MID_RUN", "expected_run_id": run_id})
    assert r.status_code == 200 and r.json()["episode_token"]
    assert _status(tid) == "scheduled"


@pytest.mark.parametrize("body", [{"mode": "MID_RUN"}, {"mode": "PRE_LAUNCH", "expected_run_id": 1}])
def test_inconsistent_mode_fields_are_422(client, body):
    tid, _ = _task()
    r = client.post(f"{API}/tasks/{tid}/schedule", json=body)
    assert r.status_code == 422 and _code(r) == "INVALID_REQUEST"
    assert _status(tid) == "ready"


def test_unknown_mode_and_missing_token_are_request_validation_422(client):
    tid, _ = _task()
    assert client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "SOMETIME"}).status_code == 422
    assert client.post(f"{API}/tasks/{tid}/unblock", json={}).status_code == 422


def test_source_is_wait_and_not_idle(client):
    waiting, _ = _task()
    first = client.post(f"{API}/tasks/{waiting}/schedule", json={"mode": "PRE_LAUNCH"})
    assert first.status_code == 200
    r = client.post(f"{API}/tasks/{waiting}/schedule", json={"mode": "PRE_LAUNCH"})
    assert r.status_code == 409 and _code(r) == "SOURCE_IS_WAIT"
    assert r.json()["detail"]["current_status"] == "scheduled"

    running, _ = _task(status_running=True)
    r = client.post(f"{API}/tasks/{running}/schedule", json={"mode": "PRE_LAUNCH"})
    assert r.status_code == 409 and _code(r) == "NOT_IDLE"


def test_status_not_allowed(client):
    tid, _ = _task()
    r = client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "MID_RUN", "expected_run_id": 1})
    assert r.status_code == 409 and _code(r) == "STATUS_NOT_ALLOWED"


def test_token_malformed_and_foreign_are_422(client):
    a, _ = _task()
    b, _ = _task()
    ta = client.post(f"{API}/tasks/{a}/schedule", json={"mode": "PRE_LAUNCH"}).json()["episode_token"]
    client.post(f"{API}/tasks/{b}/schedule", json={"mode": "PRE_LAUNCH"})
    r = client.post(f"{API}/tasks/{b}/unblock", json={"expected_episode_token": "nope"})
    assert r.status_code == 422 and _code(r) == "TOKEN_MALFORMED"
    r = client.post(f"{API}/tasks/{b}/unblock", json={"expected_episode_token": ta})
    assert r.status_code == 422 and _code(r) == "TOKEN_FOREIGN"
    assert _status(b) == "scheduled"


def test_unknown_task_and_board_are_404(client):
    r = client.post(f"{API}/tasks/t_nope0000/schedule", json={"mode": "PRE_LAUNCH"})
    assert r.status_code == 404 and _code(r) == "TASK_NOT_FOUND"
    r = client.post(f"{API}/tasks/t_nope0000/unblock",
                    json={"expected_episode_token": kbe._mint_episode_token("t_nope0000")})
    assert r.status_code == 404 and _code(r) == "TASK_NOT_FOUND"
    tid, _ = _task()
    r = client.post(f"{API}/tasks/{tid}/schedule?board=no-such-board", json={"mode": "PRE_LAUNCH"})
    assert r.status_code == 404


def test_legacy_patch_park_mints_no_token(client):
    tid, _ = _task()
    r = client.patch(f"{API}/tasks/{tid}", json={"status": "scheduled", "block_reason": "later"})
    assert r.status_code == 200 and r.json()["task"]["status"] == "scheduled"
    events = client.get(f"{API}/tasks/{tid}").json()["events"]
    sched = [e for e in events if e["kind"] == "scheduled"]
    assert len(sched) == 1 and sched[0]["payload"] == {"reason": "later"}


def test_human_patch_unblock_stales_the_governed_token(client):
    tid, _ = _task()
    token = client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "PRE_LAUNCH"}).json()["episode_token"]
    # Human drag back to ready goes through the legacy, token-less unblock_task.
    r = client.patch(f"{API}/tasks/{tid}", json={"status": "ready"})
    assert r.status_code == 200 and r.json()["task"]["status"] == "ready"
    # ...and a human re-park creates a different episode the old token cannot touch.
    assert client.patch(f"{API}/tasks/{tid}", json={"status": "scheduled"}).status_code == 200
    r = client.post(f"{API}/tasks/{tid}/unblock", json={"expected_episode_token": token})
    assert r.status_code == 409 and _code(r) == "EPISODE_STALE"
    assert _status(tid) == "scheduled"


def test_comment_during_wait_does_not_stale(client):
    tid, _ = _task()
    token = client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "PRE_LAUNCH"}).json()["episode_token"]
    assert client.post(f"{API}/tasks/{tid}/comments", json={"body": "noted", "author": "op"}).status_code == 200
    assert client.patch(f"{API}/tasks/{tid}", json={"title": "renamed", "priority": 3}).status_code == 200
    r = client.post(f"{API}/tasks/{tid}/unblock", json={"expected_episode_token": token})
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_event_feed_exposes_token_only_in_governed_park_payload(client):
    tid, _ = _task()
    token = client.post(f"{API}/tasks/{tid}/schedule", json={"mode": "PRE_LAUNCH"}).json()["episode_token"]
    events = client.get(f"{API}/tasks/{tid}").json()["events"]
    carrying = [e for e in events if "episode_token" in json.dumps(e.get("payload") or {})]
    assert [e["kind"] for e in carrying] == ["scheduled"]
    assert carrying[0]["payload"]["episode_token"] == token


def test_governed_episode_actions_require_session_auth(monkeypatch):
    """Through the real dashboard app: the governed verbs sit behind the same session
    gate as every plugin route — an episode token is a precondition, never a credential."""
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    anonymous = TestClient(app)
    r = anonymous.post(f"{API}/tasks/t_fake/schedule", json={"mode": "PRE_LAUNCH"})
    assert r.status_code == 401
    r = anonymous.post(f"{API}/tasks/t_fake/unblock",
                       json={"expected_episode_token": "hkep1.t_fake." + "0" * 32})
    assert r.status_code == 401
