"""Bot Chat destination parity at route and retained-destination boundaries."""
import asyncio
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from gateway.platforms.webhook_delivery import snapshot_destination
from hermes_state_runtime import RuntimeStoreError


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "receiver").mkdir(parents=True)
    (home / "profiles" / "sender").mkdir()
    for name in ('receiver', 'sender'):
        (home / 'profiles' / name / 'config.yaml').write_text('{}')
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "fixture-secret"}))


@pytest.mark.parametrize("target", ["bot-chat", "bot-chat:receiver"])
@pytest.mark.parametrize("mode", [
    {},
    {"deliver_only": True},
    {"coalesce": {"key": "issue.number", "window_seconds": 1}},
    {"cron_job": "existing-job"},
    {"mirror_to_session": True},
])
def test_static_and_dynamic_accept_supported_modes(adapter, target, mode):
    route = {"deliver": target, "profile": "sender", **mode}
    adapter._validate_route("events", route)
    assert adapter._dynamic_route_allowed("events", route)


@pytest.mark.parametrize("target", ["bot-chat", "bot-chat:receiver"])
@pytest.mark.parametrize("extra", [{"chat_id": "elsewhere"}, {"session_id": "elsewhere"}])
def test_arbitrary_delivery_extras_remain_rejected(adapter, target, extra):
    route = {"deliver": target, "deliver_extra": extra}
    with pytest.raises(ValueError):
        adapter._validate_route("events", route)
    assert not adapter._dynamic_route_allowed("events", route)


@pytest.mark.parametrize("target", [
    "bot-chat:", "bot-chat:../receiver", "bot-chat:receiver/session",
    "bot-chat:receiver,other", "bot-chat:receiver:thread",
])
def test_invalid_explicit_addresses_are_rejected(adapter, target):
    route = {"deliver": target}
    with pytest.raises(ValueError):
        adapter._validate_route("events", route)
    assert not adapter._dynamic_route_allowed("events", route)


@pytest.mark.parametrize("target", ["bot-chat", "bot-chat:receiver"])
def test_snapshot_retains_execution_and_destination_profiles(adapter, target):
    delivery = {"deliver": target, "deliver_extra": {}, "route": "events",
                "profile": "sender", "mirror": True}
    retained = snapshot_destination(adapter, delivery)
    assert retained["deliver"] == target
    assert retained["profile"] == "sender"
    assert retained["route"] == "events"
    assert retained["deliver_extra"] == {}


@pytest.mark.parametrize("target", ["bot-chat", "bot-chat:receiver"])
def test_snapshot_refuses_extra_addressing(adapter, target):
    with pytest.raises(RuntimeStoreError):
        snapshot_destination(adapter, {"deliver": target, "deliver_extra": {"chat_id": "elsewhere"},
                                       "route": "events"})


@pytest.mark.parametrize("conflict", [
    {"deliver_only": True},
    {"cron_job": "existing-job"},
])
def test_coalesce_remains_agent_only(adapter, conflict):
    route = {"deliver": "bot-chat", "coalesce": {"key": "issue.number", "window_seconds": 1},
             **conflict}
    with pytest.raises(ValueError, match="deliver_only|cron_job"):
        adapter._validate_route("events", route)
    assert not adapter._dynamic_route_allowed("events", route)


def test_direct_and_cron_remain_mutually_exclusive(adapter):
    route = {"deliver": "bot-chat", "deliver_only": True, "cron_job": "existing-job"}
    with pytest.raises(ValueError, match="deliver_only|cron_job"):
        adapter._validate_route("events", route)


@pytest.mark.asyncio
async def test_cron_trigger_uses_job_delivery_not_webhook_destination(monkeypatch):
    from gateway.platforms.webhook import _INSECURE_NO_AUTH

    route = {"secret": _INSECURE_NO_AUTH, "deliver": "bot-chat", "cron_job": "existing-job",
             "prompt": "Event {number}"}
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {"events": route}}))
    adapter._validate_route("events", route)
    calls = []

    def run_job(job_ref, extra_prompt=None):
        calls.append((job_ref, extra_prompt))
        return {"claimed": True, "success": True, "error": None}

    monkeypatch.setattr("tools.cronjob_tools.execute_job_for_event", run_job)

    async def unexpected_delivery(*args, **kwargs):
        pytest.fail("cron trigger must not deliver via the webhook target")

    monkeypatch.setattr(adapter, "_deliver_to", unexpected_delivery)
    monkeypatch.setattr(adapter, "handle_message", unexpected_delivery)
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/webhooks/events", json={"number": 7},
                                     headers={"X-GitHub-Delivery": "event-1"})
        assert response.status == 202
        tasks = list(adapter._background_tasks)
        if tasks:
            await asyncio.gather(*tasks)
    assert calls and calls[0][0] == "existing-job"
    assert "Event 7" in calls[0][1]
