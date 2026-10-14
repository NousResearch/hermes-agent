"""``wake_origin_session`` webhook route mode."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH

KEY = "agent:main:telegram:group:-1001:42"


def _adapter(routes=None):
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "s", "routes": routes or {}}))


def _runner(session_id="session-1", tip=None):
    runner = MagicMock()
    telegram = MagicMock()
    runner._authorization_adapter.return_value = telegram
    runner._session_key_for_source.return_value = KEY
    runner.session_store.peek_session_id.return_value = session_id
    runner.session_store._db_for_key.return_value.get_compression_tip.side_effect = lambda sid: tip or sid
    return runner, telegram


def _payload(**overrides):
    origin = {"platform": "telegram", "chatId": "-1001", "chatType": "group", "threadId": "42", "userId": "u1",
              "sessionId": "session-1", "sessionKey": KEY, **overrides}
    return {"watch": {"origin": {k: v for k, v in origin.items() if v is not None}}}


async def _wake(adapter, payload, profile=None):
    with patch("gateway.wake.deliver_wake", new=AsyncMock()) as wake:
        await adapter._wake_origin_session(payload, "new message", profile)
    return wake


@pytest.mark.asyncio
async def test_wakes_the_origin_session_through_its_own_adapter():
    adapter, (runner, telegram) = _adapter(), _runner()
    adapter.gateway_runner = runner
    wake = await _wake(adapter, _payload())
    runner._authorization_adapter.assert_called_once_with(Platform.TELEGRAM, None)
    wake.assert_awaited_once()
    assert wake.await_args.args[0] is telegram
    kwargs = wake.await_args.kwargs
    assert (kwargs["text"], kwargs["session_id"]) == ("new message", "session-1")
    assert (kwargs["source"].chat_id, kwargs["source"].thread_id) == ("-1001", "42")


@pytest.mark.asyncio
async def test_mismatched_session_key_is_rejected():
    adapter = _adapter()
    adapter.gateway_runner = _runner()[0]
    with pytest.raises(ValueError, match="session key"):
        await _wake(adapter, _payload(sessionKey="wrong"))


@pytest.mark.asyncio
async def test_session_replaced_by_new_or_reset_is_rejected():
    adapter = _adapter()
    adapter.gateway_runner = _runner(session_id="new-session")[0]  # the old chain does not reach new-session
    with pytest.raises(ValueError, match="replaced"):
        await _wake(adapter, _payload(sessionId="old-session"))


@pytest.mark.asyncio
async def test_session_rotated_by_compression_wakes_the_live_tip():
    adapter, (runner, _) = _adapter(), _runner(session_id="tip-session", tip="tip-session")
    adapter.gateway_runner = runner
    wake = await _wake(adapter, _payload(sessionId="old-session"))
    assert wake.await_args.kwargs["session_id"] == "tip-session"


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["webhook", "local", "api_server"])
async def test_unwakeable_platforms_are_rejected(platform):
    adapter = _adapter()
    adapter.gateway_runner = _runner()[0]
    with pytest.raises(ValueError, match="cannot be woken"):
        await _wake(adapter, _payload(platform=platform))


@pytest.mark.asyncio
async def test_unknown_platform_is_rejected():
    adapter = _adapter()
    adapter.gateway_runner = _runner()[0]
    with pytest.raises(ValueError):
        await _wake(adapter, _payload(platform="nonesuch"))


@pytest.mark.asyncio
@pytest.mark.parametrize("route_profile,origin_profile", [("alice", "bob"), ("alice", None), (None, "bob")])
async def test_profile_mismatch_is_rejected(route_profile, origin_profile):
    adapter = _adapter()
    adapter.gateway_runner = _runner()[0]
    with pytest.raises(ValueError, match="profile"):
        await _wake(adapter, _payload(profile=origin_profile), route_profile)


@pytest.mark.asyncio
async def test_matching_profile_selects_that_profiles_adapter():
    adapter, (runner, _) = _adapter(), _runner()
    adapter.gateway_runner = runner
    await _wake(adapter, _payload(profile="alice"), "alice")
    runner._authorization_adapter.assert_called_once_with(Platform.TELEGRAM, "alice")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"watch": "x"}, {"watch": {"origin": "x"}}, {"watch": {"origin": {}}},
                                     {"watch": {"origin": {"platform": "telegram"}}}, []])
async def test_missing_or_malformed_origin_is_rejected(payload):
    adapter = _adapter()
    adapter.gateway_runner = _runner()[0]
    with pytest.raises(ValueError):
        await _wake(adapter, payload)


@pytest.mark.parametrize("extra", [{"deliver_only": True, "deliver": "telegram"}, {"cron_job": "j"},
                                   {"coalesce": {"key": "id"}}])
def test_route_cannot_combine_wake_with_another_dispatch_mode(extra):
    adapter = _adapter()
    with pytest.raises(ValueError, match="mutually exclusive"):
        adapter._validate_route("r", {"wake_origin_session": True, **extra})
    assert not adapter._dynamic_route_allowed("r", {"wake_origin_session": True, "secret": "s", **extra})


def test_route_requires_a_real_secret():
    adapter = _adapter()
    adapter._host = "127.0.0.1"  # INSECURE_NO_AUTH is otherwise refused on its own for non-loopback hosts
    with pytest.raises(ValueError, match="real HMAC secret"):
        adapter._validate_route("r", {"wake_origin_session": True, "secret": _INSECURE_NO_AUTH})
    assert not adapter._dynamic_route_allowed("r", {"wake_origin_session": True, "secret": _INSECURE_NO_AUTH})
    adapter._validate_route("r", {"wake_origin_session": True, "secret": "s"})


@pytest.mark.asyncio
async def test_http_wake_is_202_and_duplicate_delivery_wakes_once():
    # A real secret means a signed request; reach the handler past auth by stubbing the body reader.
    adapter = _adapter({"w": {"wake_origin_session": True, "secret": "s", "prompt": "hi"}})
    adapter.gateway_runner = _runner()[0]
    body = json.dumps(_payload()).encode()
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async def authed(request, name, route):
        return body, None
    with patch.object(adapter, "_read_authenticated_body", new=authed), \
            patch("gateway.wake.deliver_wake", new=AsyncMock()) as wake:
        async with TestClient(TestServer(app)) as client:
            first = await client.post("/webhooks/w", data=body, headers={"X-Request-ID": "w1:m1"})
            second = await client.post("/webhooks/w", data=body, headers={"X-Request-ID": "w1:m1"})
            assert first.status == 202 and (await first.json())["mode"] == "origin_session"
            assert (await second.json())["status"] == "duplicate"
    wake.assert_awaited_once()
    assert not adapter._background_tasks  # no one-shot webhook session was spawned


@pytest.mark.asyncio
async def test_http_failure_is_502_without_the_reason():
    adapter = _adapter({"w": {"wake_origin_session": True, "secret": "s", "prompt": "hi"}})
    adapter.gateway_runner = _runner()[0]
    body = json.dumps(_payload(sessionKey="wrong")).encode()
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async def authed(request, name, route):
        return body, None
    with patch.object(adapter, "_read_authenticated_body", new=authed):
        async with TestClient(TestServer(app)) as client:
            resp = await client.post("/webhooks/w", data=body)
            assert resp.status == 502
            assert (await resp.json())["error"] == "Origin-session wake failed"
