"""API model selections resolve before the outbound agent boundary (#101424)."""

import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_constants import get_hermes_home


@pytest.fixture
def runtime(monkeypatch):
    """Keep HTTP, config, provider and session resolution real; replace only the LLM boundary."""
    import gateway.run

    home = get_hermes_home()
    (home / "config.yaml").write_text(json.dumps({
        "model": {"provider": "anthropic", "default": "claude-configured"},
    }))
    monkeypatch.setattr(gateway.run, "_hermes_home", home)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-model-selection")
    captured = []

    def create_agent(**kwargs):
        captured.append(kwargs)
        agent = MagicMock()
        agent.model = kwargs["model"]
        agent.provider = kwargs["provider"]
        agent.session_id = kwargs["session_id"]
        agent.memory_manager = kwargs["memory_manager"]
        agent._last_compaction_in_place = False
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.run_conversation.return_value = {"final_response": "done", "messages": []}
        return agent

    monkeypatch.setattr("run_agent.AIAgent", create_agent)
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["runs", "chat/completions", "responses", "session"])
@pytest.mark.parametrize(("model", "provider", "expected_model"), [
    ("@anthropic:claude-selected", None, "claude-selected"),
    ("anthropic::claude-selected", "anthropic", "claude-selected"),
    ("default", None, "claude-configured"),
    ("@anthropic:default", None, "claude-configured"),
    ("@anthropic:hermes-agent", None, "claude-configured"),
    ("vendor/model:variant", "anthropic", "vendor/model:variant"),
])
async def test_request_model_is_resolved_before_agent_construction(
    runtime, endpoint, model, provider, expected_model,
):
    """A provider-qualified selection never reaches the agent as its UI identifier."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    app["api_server_adapter"] = adapter
    selection = {"model": model}
    if provider:
        selection["provider"] = provider
    try:
        async with TestClient(TestServer(app)) as client:
            if endpoint == "session":
                response = await client.post("/api/sessions", json=selection)
                assert response.status == 201
                session = (await response.json())["session"]
                assert session["model"] == (None if expected_model == "claude-configured" else expected_model)
                response = await client.post(
                    f"/api/sessions/{session['id']}/chat", json={"message": "hello", **selection})
            elif endpoint == "chat/completions":
                response = await client.post(f"/v1/{endpoint}", json={
                    "messages": [{"role": "user", "content": "hello"}], **selection})
            else:
                response = await client.post(f"/v1/{endpoint}", json={"input": "hello", **selection})
            assert response.status == (202 if endpoint == "runs" else 200)
            if endpoint == "runs":
                run_id = (await response.json())["run_id"]
                # Consume the actual terminal stream instead of sleeping/polling for construction.
                events = await client.get(f"/v1/runs/{run_id}/events")
                assert "run.completed" in await events.text()
            assert len(runtime) == 1
            assert runtime[0]["model"] == expected_model
            assert runtime[0]["provider"] == "anthropic"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_route", [False, True])
async def test_stored_default_alias_resolves_before_agent_construction(runtime, configured_route):
    """Legacy rows use the gateway default unless an explicit default route exists."""
    route = {"model": "claude-routed", "provider": "anthropic"}
    extra = {"model_routes": {"default": route}} if configured_route else {}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra))
    db = adapter._ensure_session_db()
    assert db is not None
    # Seed what the old create handler persisted, bypassing the new request normalization.
    db.create_session("legacy-default", "api_server", model="default", model_config={
        "browser_model_lock": {"model": "default", "provider": "",
                               "confirmed": False, "route_source": "raw_request"},
    })
    old_model_config = db.get_session("legacy-default")["model_config"]
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    app["api_server_adapter"] = adapter
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/sessions/legacy-default/chat", json={"message": "hello"})
            assert response.status == 200
            assert (await response.json())["message"]["content"] == "done"
            assert len(runtime) == 1
            assert runtime[0]["model"] == (route["model"] if configured_route else "claude-configured")
            assert runtime[0]["provider"] == "anthropic"
            assert db.get_session("legacy-default")["model_config"] == old_model_config
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [
    {"model": "default"},
    {"model": "@anthropic:default"},
])
async def test_default_selection_preserves_configured_routes(runtime, selection):
    """A configured default alias still outranks the gateway default, including session locks."""
    route = {"model": "claude-routed", "provider": "anthropic"}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"model_routes": {"default": route}}))
    # Seed the pre-existing persisted shape, without passing it through the new request parser.
    db = adapter._ensure_session_db()
    assert db is not None
    db.create_session("locked", "api_server", model="claude-locked", model_config={
        "browser_model_lock": {"model": "claude-locked", "provider": "anthropic",
                               "confirmed": True, "route_source": "raw_request"},
    })
    old_model_config = db.get_session("locked")["model_config"]
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    app["api_server_adapter"] = adapter
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/sessions/locked/chat", json={"message": "hello"})
            assert response.status == 200
            assert (await response.json())["runtime"]["model_lock"] == "confirmed"
            assert runtime[-1]["model"] == "claude-locked"
            assert db.get_session("locked")["model_config"] == old_model_config
            response = await client.post("/api/sessions/locked/chat", json={"message": "hello", **selection})
            assert response.status == 200
            assert runtime[-1]["model"] == "claude-locked"
            response = await client.post("/v1/runs", json={"input": "hello", **selection})
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            events = await client.get(f"/v1/runs/{run_id}/events")
            assert "run.completed" in await events.text()
            assert runtime[-1]["model"] == route["model"]
            response = await client.post("/api/sessions", json={
                "id": "routed", "require_model_lock": True, **selection})
            assert response.status == 201
            assert (await response.json())["session"]["model"] == "default"
            response = await client.post("/api/sessions/routed/chat", json={"message": "hello"})
            assert response.status == 200
            assert runtime[-1]["model"] == route["model"]
    finally:
        await adapter.disconnect()
