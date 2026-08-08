"""Per-task model/provider override — DB layer, worker spawn, dashboard API.

Covers the model-dropdown feature: kanban_db.set_model_override(),
create_task(model_override=..., provider_override=...), the dispatcher
passing ``-m <model> --provider <name>`` to the worker, and the dashboard
PATCH/bulk/model-options surfaces.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    yield c
    c.close()


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_model_override_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


# ---------------------------------------------------------------------------
# DB layer — set_model_override
# ---------------------------------------------------------------------------


def test_set_and_clear_model_override(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")
    assert kb.set_model_override(conn, tid, "gpt-5.6-sol", provider="openai")
    t = kb.get_task(conn, tid)
    assert t.model_override == "gpt-5.6-sol"
    assert t.provider_override == "openai"

    # Clearing the model clears the provider too.
    assert kb.set_model_override(conn, tid, None)
    t = kb.get_task(conn, tid)
    assert t.model_override is None
    assert t.provider_override is None


def test_provider_without_model_rejected(conn):
    tid = kb.create_task(conn, title="t", assignee="worker")
    with pytest.raises(ValueError):
        kb.set_model_override(conn, tid, None, provider="openrouter")
    with pytest.raises(ValueError):
        kb.create_task(
            conn, title="t2", assignee="worker", provider_override="openrouter",
        )


def test_create_task_with_model_and_provider(conn):
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="qwen-max", provider_override="openrouter",
    )
    t = kb.get_task(conn, tid)
    assert t.model_override == "qwen-max"
    assert t.provider_override == "openrouter"
    # Creation event carries the override for auditability.
    ev = next(e for e in kb.list_events(conn, tid) if e.kind == "created")
    assert ev.payload["model_override"] == "qwen-max"
    assert ev.payload["provider_override"] == "openrouter"


def test_migration_adds_provider_override_column(conn):
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "model_override" in cols
    assert "provider_override" in cols


# ---------------------------------------------------------------------------
# Worker spawn — argv carries -m and --provider
# ---------------------------------------------------------------------------


def _spawn_and_capture(monkeypatch, tmp_path, task):
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(task, str(workspace))
    return captured["cmd"]


def test_spawn_passes_model_and_provider(monkeypatch, tmp_path, conn):
    tid = kb.create_task(
        conn, title="t", assignee="elias",
        model_override="glm-5", provider_override="openrouter",
    )
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    i = cmd.index("-m")
    assert cmd[i + 1] == "glm-5"
    j = cmd.index("--provider")
    assert j == i + 2
    assert cmd[j + 1] == "openrouter"


# ---------------------------------------------------------------------------
# Dashboard API — PATCH / bulk / create / model-options
# ---------------------------------------------------------------------------


def _create(client, **kwargs):
    body = {"title": "task", "assignee": "worker"}
    body.update(kwargs)
    r = client.post("/api/plugins/kanban/tasks", json=body)
    assert r.status_code == 200, r.text
    return r.json()["task"]


def test_patch_sets_model_override(client):
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"model_override": "gpt-5.6-sol", "provider_override": "openai"},
    )
    assert r.status_code == 200, r.text
    updated = r.json()["task"]
    assert updated["model_override"] == "gpt-5.6-sol"
    assert updated["provider_override"] == "openai"


def test_bulk_model_override(client):
    t1 = _create(client)
    t2 = _create(client)
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={
            "ids": [t1["id"], t2["id"]],
            "model_override": "fallback-model",
            "provider_override": "nous",
        },
    )
    assert r.status_code == 200, r.text
    assert all(entry["ok"] for entry in r.json()["results"])
    for tid in (t1["id"], t2["id"]):
        got = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert got["model_override"] == "fallback-model"
        assert got["provider_override"] == "nous"


def test_model_options_endpoint_shape(client, monkeypatch):
    """The endpoint returns {providers: [{slug,label,models}]} and degrades
    to an empty catalog when the inventory substrate raises."""
    r = client.get("/api/plugins/kanban/model-options")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    assert isinstance(data["providers"], list)
    for row in data["providers"]:
        assert "slug" in row and "label" in row and "models" in row
        assert isinstance(row["models"], list)
        assert len(row["models"]) >= 1  # empty-model rows are filtered out


# ---------------------------------------------------------------------------
# Per-task reasoning effort — the depth half of the board's model picker
# ---------------------------------------------------------------------------


def test_reasoning_effort_normalizes_and_rejects(conn):
    tid = kb.create_task(conn, title="t", assignee="worker", reasoning_effort="  HIGH ")
    assert kb.get_task(conn, tid).reasoning_effort == "high"

    # "none" is a VALUE (thinking off), not a clear.
    assert kb.set_reasoning_effort(conn, tid, "none")
    assert kb.get_task(conn, tid).reasoning_effort == "none"

    # Empty clears back to "inherit the profile".
    assert kb.set_reasoning_effort(conn, tid, "")
    assert kb.get_task(conn, tid).reasoning_effort is None

    with pytest.raises(ValueError):
        kb.set_reasoning_effort(conn, tid, "extremely-hard")


def test_reasoning_effort_survives_clearing_the_model(conn):
    """Depth and model are independent knobs: dropping a model override must
    not silently reset the thinking depth the operator chose."""
    tid = kb.create_task(
        conn, title="t", assignee="worker",
        model_override="glm-5", provider_override="openrouter",
        reasoning_effort="ultra",
    )
    assert kb.set_model_override(conn, tid, None)
    t = kb.get_task(conn, tid)
    assert t.model_override is None
    assert t.provider_override is None
    assert t.reasoning_effort == "ultra"


def test_reasoning_effort_without_a_model_override(conn):
    """A task may run the profile's OWN model at a different depth."""
    tid = kb.create_task(conn, title="t", assignee="worker", reasoning_effort="low")
    t = kb.get_task(conn, tid)
    assert t.model_override is None
    assert t.reasoning_effort == "low"


def test_spawn_passes_reasoning_without_a_model(monkeypatch, tmp_path, conn):
    tid = kb.create_task(conn, title="t", assignee="elias", reasoning_effort="high")
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    assert "-m" not in cmd
    i = cmd.index("--reasoning")
    assert cmd[i + 1] == "high"


def test_spawn_omits_reasoning_when_unset(monkeypatch, tmp_path, conn):
    tid = kb.create_task(conn, title="t", assignee="elias")
    task = kb.get_task(conn, tid)
    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)
    assert "--reasoning" not in cmd


def test_worker_cli_accepts_the_reasoning_flag():
    """The dispatcher's --reasoning must be a real flag on the worker's CLI —
    a spawn arg no parser accepts fails every dispatch."""
    from hermes_cli._parser import build_top_level_parser

    parser = build_top_level_parser()[0]
    args = parser.parse_args(["--cli", "chat", "-q", "hi", "--reasoning", "high"])
    assert args.reasoning == "high"


def test_patch_sets_and_clears_reasoning_effort(client):
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"reasoning_effort": "xhigh"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["task"]["reasoning_effort"] == "xhigh"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"clear_reasoning_effort": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["task"]["reasoning_effort"] is None


def test_patch_rejects_an_unknown_level(client):
    task = _create(client)
    r = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}",
        json={"reasoning_effort": "bogus"},
    )
    assert r.status_code == 400


def test_create_accepts_reasoning_effort(client):
    task = _create(client, reasoning_effort="minimal")
    assert task["reasoning_effort"] == "minimal"


def test_bulk_reasoning_effort(client):
    t1 = _create(client)
    t2 = _create(client)
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [t1["id"], t2["id"]], "reasoning_effort": "max"},
    )
    assert r.status_code == 200, r.text
    assert all(entry["ok"] for entry in r.json()["results"])
    for tid in (t1["id"], t2["id"]):
        got = client.get(f"/api/plugins/kanban/tasks/{tid}").json()["task"]
        assert got["reasoning_effort"] == "max"
