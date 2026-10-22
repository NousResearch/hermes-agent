"""Per-run fallback selection through the real HTTP handler and runtime assembly."""

import functools
import json
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.fixture
def fallback_runtime(monkeypatch):
    from hermes_constants import get_hermes_home
    from gateway import run

    home = get_hermes_home()
    config = {
        "model": {"default": "global-primary", "provider": "global-provider"},
        "fallback_providers": [
            {"model": "global-backup-one", "provider": "backup-provider-one"},
            {"model": "global-backup-two", "provider": "backup-provider-two"},
        ],
    }
    config_path = home / "config.yaml"
    config_text = json.dumps(config)
    config_path.write_text(config_text)
    monkeypatch.setattr(run, "_hermes_home", home)

    # Replace only external credentials/agent execution, not the handler, selection,
    # runtime-kwargs assembly, config loaders, fallback loader, or run lifecycle.
    def resolve_runtime_provider(requested=None, **kwargs):
        return {
            "provider": requested or config["model"]["provider"],
            "api_key": "test-key",
            "base_url": "https://provider.invalid/v1",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", resolve_runtime_provider
    )
    captured = []

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.session_id = kwargs["session_id"]

        def run_conversation(self, **kwargs):
            return {"final_response": "done"}

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    yield adapter, app, captured, config
    assert config_path.read_text() == config_text
    assert run.GatewayRunner._load_fallback_model() == config["fallback_providers"]
    if adapter._session_db is not None:
        adapter._session_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection, fallback, locked",
    [
        ({}, None, False),
        ({}, {"model": "request-backup", "provider": "request-provider"}, False),
        ({}, {"model": "request-backup"}, False),
        ({"model": "request-primary", "provider": "request-provider"}, {"model": "request-backup"}, False),
        ({"model": "route-alias"}, {"model": "request-backup"}, False),
        ({"session_id": "session-override"}, {"model": "request-backup"}, False),
        ({}, None, True),
        ({}, {"model": "request-backup", "provider": "request-provider"}, True),
    ],
)
async def test_run_fallback_is_local_to_effective_primary(
    fallback_runtime, monkeypatch, selection, fallback, locked
):
    adapter, app, captured, config = fallback_runtime
    adapter._model_routes["route-alias"] = {"model": "route-primary", "provider": "route-provider"}
    if selection.get("session_id"):
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda key: (
            {"model": "session-primary", "provider": "session-provider"}
            if key == "session-override" else None
        ))
    if locked:
        monkeypatch.setattr(adapter, "_create_agent", functools.partial(
            adapter._create_agent, confirmed_runtime_lock=True
        ))
    body = {"input": "hello", **selection}
    if fallback is not None:
        body["fallback_model"] = fallback
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/runs", json=body)
        assert response.status == 202
        run_id = (await response.json())["run_id"]
        events = await client.get(f"/v1/runs/{run_id}/events")
        assert "run.completed" in await events.text()
        actual = captured[-1]
        if locked:
            assert actual["fallback_model"] is None
        elif fallback is None:
            assert actual["fallback_model"] == config["fallback_providers"]
        else:
            assert actual["fallback_model"] == [{
                "model": fallback["model"],
                "provider": fallback.get("provider", actual["provider"]),
            }]
        expected_primary = {
            "route-alias": ("route-primary", "route-provider"),
            "request-primary": ("request-primary", "request-provider"),
        }.get(selection.get("model"), ("global-primary", "global-provider"))
        if selection.get("session_id"):
            expected_primary = ("session-primary", "session-provider")
        assert (actual["model"], actual["provider"]) == expected_primary

        # A later run with no override must still inherit the entire global chain.
        if not locked:
            response = await client.post("/v1/runs", json={"input": "next"})
            assert response.status == 202
            run_id = (await response.json())["run_id"]
            events = await client.get(f"/v1/runs/{run_id}/events")
            assert "run.completed" in await events.text()
            assert captured[-1]["fallback_model"] == config["fallback_providers"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [
    None, "model", [], 1, True, {}, {"provider": "provider"},
    {"model": None}, {"model": ""}, {"model": " \t\n"}, {"model": 1},
    {"model": []}, {"model": {}}, {"model": True},
    {"model": "backup", "provider": None}, {"model": "backup", "provider": " "},
    {"model": "backup", "provider": 1}, {"model": "backup", "provider": {}},
    {"model": "backup", "api_key": "secret"},
    {"model": "backup", "base_url": "https://untrusted.invalid"},
    {"model": "backup", "unexpected": True},
])
async def test_invalid_fallback_rejected_before_run_admission_or_idempotency(
    fallback_runtime, monkeypatch, fallback
):
    adapter, app, captured, _ = fallback_runtime
    lookup = MagicMock(side_effect=AssertionError("must not look up idempotency"))
    reserve = MagicMock(side_effect=AssertionError("must not reserve idempotency"))
    admission = MagicMock(side_effect=AssertionError("must not admit run"))
    monkeypatch.setattr(adapter._run_idempotency_store, "lookup", lookup)
    monkeypatch.setattr(adapter._run_idempotency_store, "reserve", reserve)
    monkeypatch.setattr(adapter, "_concurrency_limited_response", admission)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/runs", json={"input": "hello", "fallback_model": fallback},
            headers={"Idempotency-Key": "same-key"},
        )
        assert response.status == 400
        assert (await response.json())["error"]["code"] == "invalid_fallback_model"
    lookup.assert_not_called()
    reserve.assert_not_called()
    admission.assert_not_called()
    assert captured == []
    assert not adapter._run_owners
    assert not adapter._run_streams
    assert not adapter._active_run_tasks
    assert adapter._pending_agent_requests == 0
