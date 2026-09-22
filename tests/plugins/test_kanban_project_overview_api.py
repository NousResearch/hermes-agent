"""Kanban dashboard plugin: project overview + plan-derived ETA + feedback.

Mirrors test_kanban_board_project_api.py's bare-FastAPI harness and exercises the
Projects surface: GET /projects/{ref}/overview (stage funnel, % complete, plan
milestones/ETA), PATCH /projects/{ref} (lead/plan), and POST
/projects/{ref}/feedback (routes a card to the project's lead).
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
from hermes_cli import projects_db as pdb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_overview_test", plugin_file)
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
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


@pytest.fixture
def project(kanban_home, tmp_path):
    repo = tmp_path / "widget-repo"
    repo.mkdir()
    (repo / "PLAN.md").write_text(
        "# Widget plan\n"
        "eta: 2026-12-01\n"
        "\n"
        "- [x] Scaffold — 2026-09-15\n"
        "- [ ] Ship v1 (2026-11-20)\n"
        "- [ ] GA\n",
        encoding="utf-8",
    )
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(conn, name="Widget", primary_path=str(repo))
    return {"id": pid, "primary_path": str(repo)}


def _create_task(client, board: str, title: str) -> str:
    r = client.post(f"/api/plugins/kanban/tasks?board={board}", json={"title": title})
    assert r.status_code == 200, r.text
    return r.json()["task"]["id"]


def test_overview_derives_stages_plan_and_eta(client, project):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})
    done_id = _create_task(client, "widget", "first")
    _create_task(client, "widget", "second")
    r = client.patch(f"/api/plugins/kanban/tasks/{done_id}?board=widget", json={"status": "done", "summary": "shipped"})
    assert r.status_code == 200, r.text

    r = client.patch(
        f"/api/plugins/kanban/projects/{project['id']}",
        json={"board_slug": "widget", "lead": "widget-chief"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["project"]["lead"] == "widget-chief"

    d = client.get(f"/api/plugins/kanban/projects/{project['id']}/overview").json()
    assert d["location"] == project["primary_path"]
    assert d["lead"] == "widget-chief"
    assert d["board"] == {"slug": "widget", "name": "Widget"}
    assert d["percent_complete"] == 50
    assert d["eta"] == "2026-12-01"  # explicit eta: line wins over milestone dates
    assert d["totals"]["total"] == 2 and d["totals"]["done"] == 1

    stages = {s["name"]: s["count"] for s in d["stages"]}
    assert stages["done"] == 1 and stages["ready"] == 1

    plan = d["plan"]
    assert plan["exists"] is True
    assert plan["total_count"] == 3 and plan["done_count"] == 1 and plan["percent"] == 33
    titles = [m["title"] for m in plan["milestones"]]
    assert "Scaffold" in titles and "Ship v1" in titles
    assert plan["milestones"][0]["date"] == "2026-09-15"


def test_overview_unknown_project_404(client):
    assert client.get("/api/plugins/kanban/projects/p_nope/overview").status_code == 404


def test_overview_without_board_is_unlinked(client, project):
    d = client.get(f"/api/plugins/kanban/projects/{project['id']}/overview").json()
    assert d["status"] == "unlinked"
    assert d["board"] is None
    assert d["percent_complete"] is None


def test_patch_project_clears_lead(client, project):
    client.patch(f"/api/plugins/kanban/projects/{project['id']}", json={"lead": "someone"})
    r = client.patch(f"/api/plugins/kanban/projects/{project['id']}", json={"lead": ""})
    assert r.status_code == 200
    assert r.json()["project"]["lead"] is None


def test_feedback_creates_card_assigned_to_lead(client, project):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})
    client.patch(f"/api/plugins/kanban/projects/{project['id']}", json={"board_slug": "widget", "lead": "widget-chief"})

    r = client.post(
        f"/api/plugins/kanban/projects/{project['id']}/feedback",
        json={"body": "please add dark mode", "author": "owner"},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["lead"] == "widget-chief" and d["board"] == "widget"

    conn = kbc.connect(board="widget")
    try:
        task = kb.get_task(conn, d["task_id"])
        assert task is not None
        assert task.assignee == "widget-chief"
        assert task.status == "ready"
        assert "dark mode" in (task.body or "")
    finally:
        conn.close()


def test_feedback_requires_body(client, project):
    r = client.post(f"/api/plugins/kanban/projects/{project['id']}/feedback", json={"body": "   "})
    assert r.status_code == 400


def test_overview_panels_and_batch_summary(client, project):
    client.post("/api/plugins/kanban/boards", json={"slug": "widget", "name": "Widget"})
    client.patch(
        f"/api/plugins/kanban/projects/{project['id']}", json={"board_slug": "widget", "lead": "widget-chief"}
    )
    tid = client.post(
        f"/api/plugins/kanban/tasks?board=widget",
        json={"title": "needs input", "assignee": "widget-worker"},
    ).json()["task"]["id"]
    assert client.patch(f"/api/plugins/kanban/tasks/{tid}?board=widget", json={"status": "blocked"}).status_code == 200

    d = client.get(f"/api/plugins/kanban/projects/{project['id']}/overview").json()
    for key in ("attention", "team", "activity", "velocity"):
        assert key in d, key
    assert any(a["id"] == tid for a in d["attention"])
    assert any(t["name"] == "widget-worker" for t in d["team"])
    assert d["velocity"]["done_30d"] >= 0

    rows = client.get("/api/plugins/kanban/projects/overview").json()["projects"]
    row = next(p for p in rows if p["id"] == project["id"])
    assert row["board_slug"] == "widget" and row["lead"] == "widget-chief"
    assert row["totals"]["blocked"] == 1
    assert row["status"] == "blocked"
    assert row["plan_total"] == 3 and row["plan_done"] == 1
