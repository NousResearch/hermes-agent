"""Kanban dashboard plugin: GET/HEAD /read-model/signal (issue #86808).

The generic Kanban read-model surface exists so external dashboard
plugins can read operational graph/status signals without over-fetching
sensitive fields (task body/result/workspace path, run error/summary/
metadata) or coupling to the raw SQLite schema. This covers the REST
half of the contract: strict query validation, the allowlisted
redaction, and the no-store cache header.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_signal_test", plugin_file)
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
def sensitive_task(kanban_home):
    """A task carrying every field the signal surface must never expose."""
    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(
            conn,
            title="fix the thing",
            body="SECRET-PROMPT-BODY",
            workspace_path="/home/alice/.ssh/id_rsa",
            board="default",
        )
        conn.execute(
            "UPDATE tasks SET result = ?, claim_lock = ?, last_failure_error = ? WHERE id = ?",
            ("SECRET-RESULT", "SECRET-CLAIM-TOKEN", "SECRET-RAW-ERROR", task_id),
        )
        conn.commit()
    finally:
        conn.close()
    return task_id


def test_signal_read_model_requires_board(client):
    r = client.get("/api/plugins/kanban/read-model/signal")
    assert r.status_code == 400


def test_signal_read_model_rejects_unknown_param(client):
    r = client.get("/api/plugins/kanban/read-model/signal?board=default&color=blue")
    assert r.status_code == 400


def test_signal_read_model_rejects_duplicate_param(client):
    r = client.get("/api/plugins/kanban/read-model/signal?board=default&board=default")
    assert r.status_code == 400


def test_signal_read_model_redacts_sensitive_fields(client, sensitive_task):
    r = client.get("/api/plugins/kanban/read-model/signal?board=default")
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "private, no-store"

    body = r.json()
    task = next(t for t in body["tasks"] if t["id"] == sensitive_task)
    assert task["title"] == "fix the thing"
    assert task["status"]

    blob = str(body)
    for secret in ("SECRET-PROMPT-BODY", "SECRET-RESULT", "SECRET-CLAIM-TOKEN", "SECRET-RAW-ERROR", "/home/alice/.ssh/id_rsa"):
        assert secret not in blob

    for forbidden_key in ("body", "result", "workspace_path", "claim_lock", "last_failure_error"):
        assert forbidden_key not in task


def test_signal_read_model_head(client, sensitive_task):
    r = client.head("/api/plugins/kanban/read-model/signal?board=default")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "private, no-store"
    assert r.content == b""


def test_capabilities_lists_signal_capabilities(client):
    r = client.get("/api/plugins/kanban/capabilities")
    assert r.status_code == 200
    caps = r.json()["capabilities"]
    assert "kanban.read_model.signal.v1" in caps
    assert "kanban.events.signal.v1" in caps
