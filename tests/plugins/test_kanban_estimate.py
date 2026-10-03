"""Kanban dashboard plugin: task effort estimate.

The estimate endpoints call the auto-routed auxiliary model and parse a
compact JSON reply (tokens + complexity + rationale). Parser tests replace
``call_llm``; profile-isolation tests retain real auxiliary resolution and
replace only the HTTP transport, with synthetic homes and credentials.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from contextlib import nullcontext
import json

import httpx

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_est_test", plugin_file)
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


def _fake_resp(content: str, model: str = "aux-mini"):
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], model=model)


def test_estimate_parses_model_json(client, monkeypatch):
    task_id = client.post("/api/plugins/kanban/tasks", json={"title": "big refactor"}).json()["task"]["id"]

    import agent.auxiliary_client as aux

    def fake_call_llm(**kwargs):
        assert kwargs.get("task") == "kanban_estimator"
        return _fake_resp('{"est_tokens": 42000, "complexity": "M", "rationale": "multi-file edit"}')

    monkeypatch.setattr(aux, "call_llm", fake_call_llm)

    body = client.post(f"/api/plugins/kanban/tasks/{task_id}/estimate").json()
    assert body["ok"] is True
    assert body["est_tokens"] == 42000
    assert body["complexity"] == "M"
    assert body["rationale"] == "multi-file edit"
    assert body["model"] == "aux-mini"


def test_estimate_tolerates_unparseable_reply(client, monkeypatch):
    task_id = client.post("/api/plugins/kanban/tasks", json={"title": "vague"}).json()["task"]["id"]

    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "call_llm", lambda **kw: _fake_resp("I cannot estimate this, sorry."))

    assert client.post(f"/api/plugins/kanban/tasks/{task_id}/estimate").json()["ok"] is False


def test_estimate_unknown_task_404(client):
    assert client.post("/api/plugins/kanban/tasks/t_missing/estimate").status_code == 404


def test_estimate_from_text_no_task(client, monkeypatch):
    """The create dialog estimates from typed title/body before a task exists."""
    import agent.auxiliary_client as aux
    monkeypatch.setattr(
        aux, "call_llm",
        lambda **kw: _fake_resp('{"est_tokens": 8000, "complexity": "S", "rationale": "localized"}'),
    )
    body = client.post(
        "/api/plugins/kanban/estimate", json={"title": "tweak a label", "body": "in settings"}
    ).json()
    assert body["ok"] is True
    assert body["est_tokens"] == 8000
    assert body["complexity"] == "S"


def test_estimate_from_text_requires_title(client):
    assert client.post("/api/plugins/kanban/estimate", json={"title": "  ", "body": "x"}).json()["ok"] is False


@pytest.mark.parametrize("existing_task", [False, True], ids=["text", "existing-task"])
def test_estimate_real_auxiliary_profile_isolation(client, kanban_home, monkeypatch, existing_task):
    """Real config/secret/client resolution, with only HTTP transport replaced."""
    import os
    from agent import secret_scope as secrets
    from agent.portal_tags import get_affinity_scope
    from hermes_cli.web_server_profiles import _config_profile_scope
    from hermes_constants import get_hermes_home
    from tui_gateway import launch_profile_policy as launch

    a = kanban_home
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    for home, label in ((a, "a"), (b, "b")):
        (home / ".env").write_text(f"OPENROUTER_API_KEY=synthetic-{label}\n", encoding="utf-8")
        (home / "config.yaml").write_text(
            "model:\n  provider: custom\n  default: main-model\n"
            "  base_url: https://estimate.invalid/v1\n  api_key: ${OPENROUTER_API_KEY}\n"
            "auxiliary:\n  kanban_estimator:\n    provider: custom\n"
            f"    model: estimate-{label}\n", encoding="utf-8")
    monkeypatch.setattr(launch, "_snapshot", None)
    monkeypatch.setattr(secrets, "_MULTIPLEX_ACTIVE", False)
    # A launch-only credential survives activation; later ambient changes must not.
    monkeypatch.setenv("LAUNCH_ONLY_KEY", "synthetic-launch")
    launch.activate_multi_profile_hosting()
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-ambient-poison")
    monkeypatch.setenv("LAUNCH_ONLY_KEY", "synthetic-late-poison")
    seen = []
    fail = False

    def offline_send(self, request, **kwargs):
        nonlocal fail
        assert request.url.host == "estimate.invalid"
        payload = json.loads(request.content)
        scope = secrets.current_secret_scope()
        assert scope is not None
        seen.append((get_hermes_home(), request.headers["authorization"], payload["model"]))
        assert secrets.get_secret("LAUNCH_ONLY_KEY") == (
            "synthetic-launch" if get_hermes_home() == a else None)
        assert get_affinity_scope()
        if fail:
            return httpx.Response(422, request=request, json={"error": {"message": "offline failure"}})
        return httpx.Response(200, request=request, json={
            "id": "offline", "object": "chat.completion", "created": 0,
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content":
                    '{"est_tokens": 8000, "complexity": "S", "rationale": "offline"}'}}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", offline_send)
    original_scope = secrets.current_secret_scope()
    original_affinity = get_affinity_scope()
    for label in ("a", "b", "a"):
        home = a if label == "a" else b
        # Default dashboard requests are unscoped; routed callers already own B.
        with (nullcontext() if label == "a" else _config_profile_scope("b")):
            owner_scope = secrets.current_secret_scope()
            kb.init_db()
            if existing_task:
                task = client.post("/api/plugins/kanban/tasks", json={"title": "offline task"}).json()["task"]
                url, kwargs = f"/api/plugins/kanban/tasks/{task['id']}/estimate", {}
            else:
                url, kwargs = "/api/plugins/kanban/estimate", {"json": {"title": "offline task"}}
            result = client.post(url, **kwargs).json()
            assert result.get("ok") is True, result
            assert result["model"] == f"estimate-{label}"
            assert seen[-1] == (home, f"Bearer synthetic-{label}", f"estimate-{label}")
            # Exercise cleanup in the same context (TestClient alone masks leaks in its worker).
            plugin = sys.modules["hermes_kanban_plugin_est_test"]
            assert plugin._run_estimate("offline task", None, task_id=None)["ok"]
            assert secrets.current_secret_scope() is owner_scope
            assert get_affinity_scope() == original_affinity
            fail = True
            error = plugin._run_estimate("offline task", None, task_id=None)
            fail = False
            assert error == {"ok": False, "reason": "LLM error: UnprocessableEntityError"}
            assert secrets.current_secret_scope() is owner_scope
            assert get_hermes_home() == home
            assert get_affinity_scope() == original_affinity
        assert secrets.current_secret_scope() is original_scope
        assert get_hermes_home() == a
    assert os.environ["OPENROUTER_API_KEY"] == "synthetic-ambient-poison"
    assert secrets.is_multiplex_active()
    with pytest.raises(secrets.UnscopedSecretError):
        secrets.get_secret("OPENROUTER_API_KEY")
    # A home-only caller must not combine B's config with A's launch credentials.
    from hermes_cli.web_server_profiles import _hermes_home_scope
    with _hermes_home_scope(b):
        assert plugin._run_estimate("offline task", None, task_id=None) == {
            "ok": False, "reason": "LLM error: UnscopedSecretError"}
    assert secrets.current_secret_scope() is original_scope


def test_estimate_preserves_empty_scope_and_existing_affinity(client, monkeypatch):
    import agent.auxiliary_client as aux
    from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope
    from agent.portal_tags import get_affinity_scope, reset_affinity_scope, set_affinity_scope

    empty_scope = {}
    secret_token = set_secret_scope(empty_scope)
    affinity_token = set_affinity_scope("owning-conversation")
    def fake_call(**kwargs):
        assert current_secret_scope() is empty_scope
        assert get_affinity_scope() == "owning-conversation"
        raise ValueError("offline failure")
    monkeypatch.setattr(aux, "call_llm", fake_call)
    try:
        plugin = sys.modules["hermes_kanban_plugin_est_test"]
        assert plugin._run_estimate("offline task", None, task_id=None) == {
            "ok": False, "reason": "LLM error: ValueError"}
        assert current_secret_scope() is empty_scope
        assert get_affinity_scope() == "owning-conversation"
    finally:
        reset_affinity_scope(affinity_token)
        reset_secret_scope(secret_token)
