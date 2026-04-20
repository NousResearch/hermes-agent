"""End-to-end FastAPI tests using TestClient.

Each test gets its own isolated $HERMES_HOME via the conftest fixtures, so the
server's Store points at a fresh tmp dir.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hermes_office.eventbus import EventBus
from hermes_office.server import build_app
from hermes_office.skill_resolver import SkillResolver
from hermes_office.store import Store


@pytest.fixture()
def app(office_root: Path):
    store = Store(office_root)
    store.boot_from_disk()
    bus = EventBus()
    resolver = SkillResolver(weights_path=store.weights_path)
    return build_app(store=store, bus=bus, resolver=resolver, runtime_default="simulated")


@pytest.fixture()
def client(app):
    with TestClient(app) as c:
        yield c


# ── health & meta ──────────────────────────────────────────────────────────


def test_health_returns_zero_counts_on_clean_root(client: TestClient, office_root: Path):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["departments"] == 0
    assert body["employees"] == 0
    assert body["office_root"] == str(office_root)


def test_capacity_endpoint_returns_report(client: TestClient):
    r = client.get("/api/capacity?model=gemma4-e2b-hermes")
    assert r.status_code == 200
    body = r.json()
    assert "recommended_concurrency" in body
    assert "expected_p95_latency_ms" in body


def test_toolsets_and_skills_lists_are_arrays(client: TestClient):
    r = client.get("/api/toolsets")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    r = client.get("/api/skills")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_presets_returns_eight_roles(client: TestClient):
    r = client.get("/api/presets")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 8
    ids = {p["id"] for p in body}
    assert {"researcher", "coder", "writer", "designer", "analyst", "translator", "tutor", "helper"} <= ids


# ── skills/resolve ──────────────────────────────────────────────────────────


def test_resolve_endpoint(client: TestClient):
    r = client.post("/api/skills/resolve", json={"text": "draw a logo"})
    assert r.status_code == 200
    body = r.json()
    assert "image_gen" in body["recommended_toolsets"]


def test_resolve_rejects_empty(client: TestClient):
    r = client.post("/api/skills/resolve", json={"text": ""})
    assert r.status_code in (422, 400)


# ── departments ────────────────────────────────────────────────────────────


def test_department_crud_round_trip(client: TestClient):
    r = client.post("/api/departments", json={"name": "Marketing", "mission": "Sell stuff", "color": "#22c55e"})
    assert r.status_code == 201, r.text
    dept = r.json()
    dept_id = dept["id"]

    listed = client.get("/api/departments").json()
    assert any(d["id"] == dept_id for d in listed)

    patched = client.patch(f"/api/departments/{dept_id}", json={"name": "Marketing 2"}).json()
    assert patched["name"] == "Marketing 2"

    deleted = client.delete(f"/api/departments/{dept_id}").json()
    assert deleted["deleted_dept"] == dept_id


def test_department_invalid_color_rejected(client: TestClient):
    r = client.post("/api/departments", json={"name": "x", "color": "orange"})
    assert r.status_code == 422


def test_department_not_found(client: TestClient):
    r = client.patch("/api/departments/dept_nopenope", json={"name": "x"})
    assert r.status_code == 404


# ── employees ──────────────────────────────────────────────────────────────


def test_employee_lifecycle(client: TestClient):
    dept = client.post("/api/departments", json={"name": "Lab"}).json()
    payload = {
        "department_id": dept["id"],
        "name": "Eve",
        "role": "Researcher",
        "model": "gemma4-e2b-hermes",
        "runtime": "simulated",
    }
    r = client.post("/api/employees", json=payload)
    assert r.status_code == 201, r.text
    emp = r.json()
    assert emp["name"] == "Eve"

    got = client.get(f"/api/employees/{emp['id']}").json()
    assert got["cli_command"].startswith("hermes chat")
    assert "gemma4-e2b-hermes" in got["cli_command"]

    patched = client.patch(f"/api/employees/{emp['id']}", json={"name": "Evelyn"}).json()
    assert patched["name"] == "Evelyn"
    assert patched["revision"] == emp["revision"] + 1

    cli = client.get(f"/api/employees/{emp['id']}/cli-command").json()
    assert cli["command"].startswith("hermes chat")

    deleted = client.delete(f"/api/employees/{emp['id']}").json()
    assert deleted["deleted"] == emp["id"]


def test_employee_create_rejects_unknown_dept(client: TestClient):
    r = client.post("/api/employees", json={
        "department_id": "dept_unknown00",
        "name": "x",
        "model": "m",
    })
    assert r.status_code == 422


# ── tasks ──────────────────────────────────────────────────────────────────


def test_task_dispatch_runs_simulated_runtime(client: TestClient):
    dept = client.post("/api/departments", json={"name": "Ops"}).json()
    emp = client.post("/api/employees", json={
        "department_id": dept["id"],
        "name": "Echo",
        "model": "gemma4-e2b-hermes",
        "runtime": "simulated",
    }).json()
    r = client.post("/api/tasks", json={"employee_id": emp["id"], "text": "say hi"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "queued"
    assert body["employee_id"] == emp["id"]


def test_task_requires_employee_or_dept(client: TestClient):
    r = client.post("/api/tasks", json={"text": "do thing"})
    assert r.status_code == 422


def test_task_routes_to_dept_with_no_employees_errors(client: TestClient):
    dept = client.post("/api/departments", json={"name": "Empty"}).json()
    r = client.post("/api/tasks", json={"department_id": dept["id"], "text": "do it"})
    assert r.status_code == 422


def test_list_tasks_collapses_status_transitions_per_id(client: TestClient, app):
    """Regression: ``GET /api/tasks`` must return exactly one row per task id
    (the latest status), not one per append-only log entry. Otherwise the
    UI shows the same task three times (queued + running + done)."""
    from datetime import datetime, timezone

    from hermes_office.models import Task

    dept = client.post("/api/departments", json={"name": "Q"}).json()
    emp = client.post("/api/employees", json={
        "department_id": dept["id"], "name": "Bot", "model": "m",
    }).json()
    store = app.state.store
    base = Task(
        department_id=dept["id"],
        employee_id=emp["id"],
        text="hello",
        status="queued",
    )
    store.append_task(base)
    store.append_task(base.model_copy(update={
        "status": "running",
        "started_at": datetime.now(tz=timezone.utc),
    }))
    store.append_task(base.model_copy(update={
        "status": "done",
        "started_at": datetime.now(tz=timezone.utc),
        "finished_at": datetime.now(tz=timezone.utc),
        "result_summary": "ok",
    }))

    rows = client.get("/api/tasks?limit=50").json()
    matches = [r for r in rows if r["id"] == base.id]
    assert len(matches) == 1, f"expected 1 row, got {len(matches)}: {matches}"
    assert matches[0]["status"] == "done"
    assert matches[0]["result_summary"] == "ok"


# ── export / import ────────────────────────────────────────────────────────


def test_export_import_round_trip(client: TestClient):
    dept = client.post("/api/departments", json={"name": "X"}).json()
    client.post("/api/employees", json={
        "department_id": dept["id"], "name": "Y", "model": "m",
    })
    payload = client.get("/api/export").json()
    assert payload["version"] == 1

    # wipe and reimport
    client.delete(f"/api/departments/{dept['id']}")
    assert client.get("/api/departments").json() == []
    r = client.post("/api/import", json=payload)
    assert r.status_code == 200
    listed = client.get("/api/departments").json()
    assert any(d["id"] == dept["id"] for d in listed)


# ── ws ─────────────────────────────────────────────────────────────────────


def test_websocket_emits_hello(client: TestClient):
    with client.websocket_connect("/ws/office") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["kind"] == "hello"
