"""Cutover regression tests (t_bbeeabb0, 2026-09-21): the legacy JSON task
mirror is dead and the dock pane's /kanban_* proxies serve real data only.

Guards:
  1. usage-meter plugin_api.py exposes NO /fleet_tasks* routes (404 on hit).
  2. The transitional mock layer is gone: HERMES_USAGE_METER_FORCE_MOCK=1 can
     no longer produce {kanban_mock: true} / X-Kanban-Mock responses.
  3. /kanban_board still serves the live/direct layer with the contract §2.4
     eta block (backlog_clear_at + per_task covering every open task).

The usage-meter plugin is an EXTERNAL desktop plugin (not repo-resident), so
these tests skip when that install is absent (e.g. CI elsewhere).
Run: python -m pytest tests/plugins/test_kanban_dock_cutover.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
USAGE_METER_API = Path(os.environ.get(
    "HERMES_USAGE_METER_PLUGIN_API",
    str(Path.home() / "AppData/Local/hermes/plugins/usage-meter/dashboard/plugin_api.py"),
))

pytestmark = pytest.mark.skipif(
    not USAGE_METER_API.exists(), reason=f"external usage-meter plugin absent: {USAGE_METER_API}")


@pytest.fixture()
def pane_app(tmp_path, monkeypatch):
    """Fresh sandbox board + plugin module loaded the way web_server does."""
    for pin in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(pin, raising=False)
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    slug = "dockcut"
    monkeypatch.setenv("HERMES_USAGE_METER_KANBAN_BOARD", slug)
    # the mock layer is dead: the force flag must be inert — prove it under load
    monkeypatch.setenv("HERMES_USAGE_METER_FORCE_MOCK", "1")

    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing(board=slug) as conn:
        conn.execute("SELECT 1 FROM tasks").fetchone()

    spec = importlib.util.spec_from_file_location("cutover_usage_meter_api", USAGE_METER_API)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/usage-meter")
    return mod, TestClient(app)


def test_mirror_routes_are_gone(pane_app):
    mod, client = pane_app
    paths = {getattr(r, "path", "") for r in mod.router.routes}
    assert "/fleet_tasks" not in paths
    assert "/fleet_tasks/reorder" not in paths
    assert client.get("/api/plugins/usage-meter/fleet_tasks").status_code == 404
    assert client.post("/api/plugins/usage-meter/fleet_tasks/reorder",
                       json={"orderedIds": []}).status_code == 404


def test_mock_layer_is_dead_even_with_force_flag(pane_app):
    mod, client = pane_app
    r = client.get("/api/plugins/usage-meter/kanban_mode")
    assert r.status_code == 200
    assert r.json()["mode"] in ("live", "direct"), r.json()
    r = client.get("/api/plugins/usage-meter/kanban_board")
    assert r.status_code == 200
    assert not r.json().get("kanban_mock")
    assert r.headers.get("X-Kanban-Mock") is None


def test_kanban_board_serves_eta_block_over_open_tasks(pane_app):
    mod, client = pane_app
    # seed two tasks through the proxy itself (real write path)
    r1 = client.post("/api/plugins/usage-meter/kanban_tasks",
                     json={"title": "cutover-test A", "p_band": "P1", "est_hours": 0.5, "triage": False})
    assert r1.status_code == 200, r1.text[:200]
    r2 = client.post("/api/plugins/usage-meter/kanban_tasks",
                     json={"title": "cutover-test B", "p_band": "P0", "est_hours": 2.0, "triage": False})
    assert r2.status_code == 200, r2.text[:200]

    r = client.get("/api/plugins/usage-meter/kanban_board")
    assert r.status_code == 200
    body = r.json()
    eta = body.get("eta") or {}
    assert isinstance(eta.get("backlog_clear_at"), (int, float))
    rows = [t for c in body.get("columns", []) for t in c.get("tasks", [])
            if t.get("status") not in ("done", "archived")]
    assert rows, "seeded tasks must appear"
    for t in rows:
        assert t["id"] in eta["per_task"], f"eta missing {t['id']}"
    # P0 (2.0h) and P1 (0.5h) on 8 lanes finish near-simultaneously; clear = max
    now = float(body.get("now") or eta.get("now"))
    assert eta["backlog_clear_at"] > now
