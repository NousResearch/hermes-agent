"""Per-task model/provider override — DB layer, worker spawn, dashboard API.

Covers the model-dropdown feature: kanban_db.set_model_override(),
create_task(model_override=..., provider_override=...), the dispatcher
passing task identity to the owner for trusted policy resolution, and the dashboard
PATCH/bulk/model-options surfaces.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


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
    c = kbc.connect()
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
# Worker spawn — task identity resolves trusted owner policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, expected_model, expected_provider, expected_reasoning",
    [
        ({"model_override": "glm-5", "provider_override": "openrouter",
          "reasoning_effort": "high"}, "glm-5", "openrouter", "high"),
        ({"reasoning_effort": "high"}, "profile-model", "custom", "high"),
        ({"reasoning_effort": "none"}, "profile-model", "custom", None),
        ({}, "profile-model", "custom", "low"),
    ],
)
def test_spawn_resolves_claimed_task_policy(
    monkeypatch, tmp_path, conn, kanban_home, overrides,
    expected_model, expected_provider, expected_reasoning,
):
    """The worker passes identity, while the owner reads overrides from SQLite."""
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli import kanban_worker_client
    from hermes_state_runtime import RuntimeStoreError

    profile = kanban_home / "profiles" / "elias"
    profile.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tid = kb.create_task(conn, title="t", assignee="elias",
                         workspace_kind="dir", workspace_path=str(workspace), **overrides)
    task = kb.claim_task(conn, tid)
    assert task is not None
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(cmd=list(cmd), env=kwargs["env"])
        kwargs["stdout"].close()
        return SimpleNamespace(pid=4245)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kbd._default_spawn(task, str(workspace))
    assert captured["cmd"] == [sys.executable, "-m", "hermes_cli.kanban_worker_client"]
    assert captured["env"]["HERMES_HOME"] == str(profile)
    actor = Principal("owner", str(profile), frozenset({"session:create"}), "transport")
    connection = SimpleNamespace(
        authority=SimpleNamespace(profile_id=str(profile)), actor=actor, native_owner=True)
    config = {"model": {"default": "profile-model", "provider": "custom"},
              "agent": {"reasoning_effort": "medium",
                        "reasoning_overrides": {"profile-model": "low", "glm-5": "low"}},
              "platform_toolsets": {"cli": ["terminal"]}}
    original_config = json.dumps(config, sort_keys=True)

    async def resolve_at_owner(params, board_db):
        params = dict(params, db=board_db)
        policy, _ = build_kanban_policy(connection, params, config)
        assert policy.model == expected_model
        assert policy.provider == expected_provider
        expected = ({"enabled": False} if expected_reasoning is None else
                    {"enabled": True, "effort": expected_reasoning})
        assert policy.reasoning_config == expected
        assert policy.cwd == str(workspace) and "kanban" in policy.toolsets
        context = json.loads(policy.kanban_json)
        assert (context["task_id"], context["run_id"], context["claim_lock"]) == (
            task.id, task.current_run_id, task.claim_lock)
        # A worker cannot replace trusted settings or reuse another claim.
        with pytest.raises(RuntimeStoreError, match="invalid_params"):
            build_kanban_policy(connection, params | {"model": "forged"}, config)
        with pytest.raises(RuntimeStoreError, match="invalid_kanban_claim"):
            build_kanban_policy(connection, params | {"claim_lock": "forged"}, config)
        assert json.dumps(config, sort_keys=True) == original_config
        captured["resolved"] = True
        return 0

    monkeypatch.setattr(kanban_worker_client, "run", resolve_at_owner)
    with monkeypatch.context() as child:
        for key, value in captured["env"].items():
            child.setenv(key, value)
        assert kanban_worker_client.main() == 0
    assert captured["resolved"]


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
