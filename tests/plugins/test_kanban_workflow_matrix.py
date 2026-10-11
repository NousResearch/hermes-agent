"""Manual move matrix: ``DEFAULT_WORKFLOW.manual`` vs. the live ``PATCH /tasks/{id}``.

Every ``src -> dst`` pair is requested through the dashboard API on a parentless task.
The workflow allow-list must agree with what the server accepts, in both directions.
Also covers ``GET /workflow``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_workflow as kw

W = kw.DEFAULT_WORKFLOW

# Accepted moves whose card lands somewhere other than the requested column.
# review -> todo goes through reopen_review_task, which re-gates to ``ready``.
_KNOWN_REDIRECTS = {("review", "todo"): "ready"}


def _load_plugin_router():
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_kanban_workflow_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/k")
    return TestClient(app)


def _task_in(client, src: str) -> str:
    tid = client.post("/k/tasks", json={"title": src, "assignee": "a", "triage": src == "triage"}).json()["task"]["id"]
    with kbc.connect_closing() as conn:
        if src == "todo":
            assert client.patch(f"/k/tasks/{tid}", json={"status": "todo"}).status_code == 200
        elif src == "running":
            assert kb.claim_task(conn, tid) is not None
        elif src == "blocked":
            assert kb.block_task(conn, tid, reason="x")
        elif src == "scheduled":
            assert kb.schedule_task(conn, tid, reason="x")
        elif src == "review":
            assert kb.request_review(conn, tid, summary="s", force=True, reviewer="reviewer")
        elif src == "done":
            assert kb.complete_task(conn, tid, summary="s", force=True)
        elif src == kw.ARCHIVED:
            assert kb.archive_task(conn, tid)
        assert kb.get_task(conn, tid).status == src
    return tid


@pytest.mark.parametrize("src", (*W.keys(), kw.ARCHIVED))
def test_manual_moves_match_server(client, src):
    mismatches = []
    for dst in (*W.keys(), kw.ARCHIVED):
        if dst == src:
            continue
        tid = _task_in(client, src)
        # ``done`` needs completion evidence; that is a verb gate, not a transition rule.
        body = {"status": dst, **({"summary": "s"} if dst == "done" else {})}
        r = client.patch(f"/k/tasks/{tid}", json=body)
        with kbc.connect_closing() as conn:
            landed = kb.get_task(conn, tid).status
        # A refusal must be a domain rejection (400 bad verb / 409 refused transition);
        # anything else (5xx, 422) is a bug, never an acceptable "not allowed".
        if r.status_code not in (200, 400, 409):
            mismatches.append(f"{src}->{dst}: unexpected HTTP {r.status_code}: {r.text[:200]}")
            continue
        accepted = r.status_code == 200
        if accepted != W.can_move(src, dst):
            mismatches.append(f"{src}->{dst}: server {r.status_code}, workflow can_move={W.can_move(src, dst)}")
        elif accepted and landed != _KNOWN_REDIRECTS.get((src, dst), dst):
            mismatches.append(f"{src}->{dst}: landed in {landed}")
    assert not mismatches, "\n".join(mismatches)


def test_get_workflow(client):
    r = client.get("/k/workflow")
    assert r.status_code == 200
    assert r.json() == W.to_dict()
