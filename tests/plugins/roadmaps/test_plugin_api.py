"""Tests for the Roadmaps dashboard plugin backend (plugins/roadmaps/dashboard/plugin_api.py).

The plugin mounts as /api/plugins/roadmaps/ inside the dashboard's FastAPI app,
but here we attach its router to a bare FastAPI instance so we can test the
REST surface without spinning up the whole dashboard (same pattern as
tests/plugins/test_kanban_dashboard_plugin.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import projects_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load plugins/roadmaps/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[3]
    plugin_file = repo_root / "plugins" / "roadmaps" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_roadmaps_test", plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def roadmaps_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so the plugin resolves a throwaway projects.db."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The plugin resolves `default` -> the process HERMES_HOME; seed a project
    # row so the roadmap FK (project_id REFERENCES projects(id)) is satisfiable.
    db = tmp_path / "projects.db"
    conn = projects_db.connect(db)
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES (?, ?, ?, 1)", ("p1", "p1", "p1"))
    conn.commit()
    conn.close()
    return tmp_path


@pytest.fixture
def client(roadmaps_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/roadmaps")
    return TestClient(app)


PROFILE = "default"
PROJECT = "p1"


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------


def test_list_empty(client):
    r = client.get(f"/api/plugins/roadmaps/roadmaps?profile={PROFILE}&project_id={PROJECT}")
    assert r.status_code == 200
    body = r.json()
    assert body["roadmaps"] == []
    assert body["scope"] == {"profile_id": PROFILE, "project_id": PROJECT}


def test_planning_rules_default_version(client):
    r = client.get("/api/plugins/roadmaps/planning-rules")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body and "rules" in body


def test_planning_rules_unknown_version(client):
    r = client.get("/api/plugins/roadmaps/planning-rules?version=does-not-exist")
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == 5063


# ---------------------------------------------------------------------------
# Create + conflict
# ---------------------------------------------------------------------------


def test_create_then_conflict(client):
    r = client.post(
        f"/api/plugins/roadmaps/roadmaps?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "title": "dogfooding v1"},
    )
    assert r.status_code == 200
    body = r.json()
    rid = body["roadmap_id"]
    assert body["state"] == "draft"

    # Same explicit roadmap_id again -> conflict (5067 -> 409).
    r2 = client.post(
        f"/api/plugins/roadmaps/roadmaps?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "title": "dup", "roadmap_id": rid},
    )
    assert r2.status_code == 409
    assert r2.json()["detail"]["code"] == 5067


# ---------------------------------------------------------------------------
# Archive (needs a roadmap; re-seed via create)
# ---------------------------------------------------------------------------


def test_archive(client):
    r = client.post(
        f"/api/plugins/roadmaps/roadmaps?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "title": "archive me"},
    )
    rid = r.json()["roadmap_id"]

    # Archived is a terminal transition; expected_version=0 (no active version yet).
    r2 = client.post(
        f"/api/plugins/roadmaps/roadmaps/{rid}/archive?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "expected_version": 0},
    )
    assert r2.status_code == 200
    assert r2.json()["roadmap"]["lifecycle_state"] == "archived"

    # Archiving again -> invalid transition (5066 -> 422).
    r3 = client.post(
        f"/api/plugins/roadmaps/roadmaps/{rid}/archive?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "expected_version": 0},
    )
    assert r3.status_code == 422
    assert r3.json()["detail"]["code"] == 5066


# ---------------------------------------------------------------------------
# Plan lifecycle: create -> validate -> activate
# ---------------------------------------------------------------------------


def _create_and_get(client, title="plan me"):
    r = client.post(
        f"/api/plugins/roadmaps/roadmaps?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "title": title},
    )
    assert r.status_code == 200
    return r.json()["roadmap_id"]


def test_plan_validate_activate(client):
    rid = _create_and_get(client)

    # Version 2 becomes the first real plan (version 1 is the empty marker).
    r = client.post(
        f"/api/plugins/roadmaps/roadmaps/{rid}/plans?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "nodes": [], "relations": [], "todos": []},
    )
    assert r.status_code == 200
    version = r.json()["version"]

    r = client.post(
        f"/api/plugins/roadmaps/roadmaps/{rid}/plans/{version}/validate?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "expected_version": 0},
    )
    assert r.status_code == 200

    r = client.post(
        f"/api/plugins/roadmaps/roadmaps/{rid}/plans/{version}/activate?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "expected_version": 0},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Security: profile scope isolation
# ---------------------------------------------------------------------------


def test_unknown_profile_fails_closed(client):
    # A syntactically valid but nonexistent profile must not resolve/seed a DB.
    r = client.get("/api/plugins/roadmaps/roadmaps?profile=nosuchprofile&project_id=p1")
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == 5063


def test_node_mutation_missing_scope(client):
    rid = _create_and_get(client)
    r = client.post(
        f"/api/plugins/roadmaps/roadmaps/{rid}/nodes/nope/claim?profile={PROFILE}&project_id={PROJECT}",
        json={"actor": "pierre", "expected_version": 0},
    )
    # Node "nope" does not exist -> 5065 -> 404.
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == 5065
