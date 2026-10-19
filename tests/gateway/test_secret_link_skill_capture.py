"""Trigger 2: a skill turn on Telegram with a missing key gets a link (key prefilled); the
waiting tool gets only the key name and a success flag. Other messaging platforms keep the hint."""

import asyncio
import re
import threading
from unittest.mock import AsyncMock

import pytest

from gateway import secret_link
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.gateway.secret_link_helpers import (
    BASE, SENTINEL, clock_fixture, env_text, get_form, hermes_root_fixture, restarts_fixture, secret_client,
    store_fixture, submit)
from tests.waiting import async_wait_until

pytestmark = pytest.mark.asyncio
_LINK_RE = re.compile(re.escape(BASE) + r"/secret-link/([A-Za-z0-9_-]+)")
_SESSION_KEY = "agent:main:telegram:dm:4242"
_MISSING = [{"name": "DEMO_API_KEY", "prompt": "Enter value for DEMO_API_KEY"}]


def _runner(platform=Platform.TELEGRAM):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True, token="t")})
    adapter = type("_Adapter", (), {"platform": platform})()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))
    runner.adapters = {platform: adapter}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner, adapter


def _source(platform=Platform.TELEGRAM):
    return SessionSource(platform=platform, chat_id="4242", chat_type="dm", user_id="7")


def _capture_in_turn(platform="telegram"):
    """Run the real skill secret-capture path as a gateway agent thread would."""
    from gateway.session_context import set_session_vars
    from tools import skills_tool

    set_session_vars(platform=platform, chat_id="4242", session_key=_SESSION_KEY)
    return skills_tool._capture_required_environment_variables("demo-skill", _MISSING)


async def _link_token(adapter):
    await async_wait_until(lambda: adapter.send.await_count >= 1)
    return _LINK_RE.search(adapter.send.await_args_list[0].args[1]).group(1)


async def test_missing_key_on_telegram_gets_a_prefilled_link_and_a_bare_success_result(
        hermes_root, store, restarts):
    runner, adapter = _runner()
    requests = secret_link.open_turn_secret_requests(
        runner, _source(), _SESSION_KEY, asyncio.get_running_loop(), lambda: False)
    try:
        capture = asyncio.create_task(asyncio.to_thread(_capture_in_turn))
        token = await _link_token(adapter)
        async with secret_client() as client:
            _r, html, nonce = await get_form(client, token)
            assert 'value="DEMO_API_KEY"' in html
            resp, _ = await submit(client, token, nonce, key="DEMO_API_KEY", profiles=("default",))
        assert resp.status == 200
        result = await capture
    finally:
        requests.close()
    assert result == {"missing_names": [], "setup_skipped": False, "gateway_setup_hint": None}
    assert f"DEMO_API_KEY={SENTINEL}" in env_text(hermes_root)
    assert SENTINEL not in repr(result)


async def test_a_missing_key_save_does_not_restart_the_owner_profile_the_turn_waits_in(
        hermes_root, store, restarts):
    """Skill secrets are read per turn: restarting the owner gateway would kill the very turn that
    waits for the link. The turn finishes first; other written profiles still restart."""
    restarts.running = {"default", "alpha"}
    runner, adapter = _runner()  # owner profile: default
    requests = secret_link.open_turn_secret_requests(
        runner, _source(), _SESSION_KEY, asyncio.get_running_loop(), lambda: False)
    try:
        pending = asyncio.create_task(asyncio.to_thread(requests.request, "DEMO_API_KEY"))
        token = await _link_token(adapter)
        async with secret_client() as client:
            _r, _h, nonce = await get_form(client, token)
            resp, _ = await submit(client, token, nonce, key="DEMO_API_KEY", profiles=("default", "alpha"))
        assert resp.status == 200
        assert await pending == {"key": "DEMO_API_KEY", "success": True}  # the turn is released ...
        await asyncio.gather(*list(secret_link._FOLLOWUPS))
    finally:
        requests.close()
    assert restarts == ["alpha"]  # ... and only the other profile restarted
    assert f"DEMO_API_KEY={SENTINEL}" in env_text(hermes_root)


async def test_the_requester_returns_only_the_key_and_the_success_flag(hermes_root, store, restarts):
    runner, adapter = _runner()
    requests = secret_link.open_turn_secret_requests(
        runner, _source(), _SESSION_KEY, asyncio.get_running_loop(), lambda: False)
    try:
        pending = asyncio.create_task(asyncio.to_thread(requests.request, "DEMO_API_KEY"))
        token = await _link_token(adapter)
        async with secret_client() as client:
            _r, _h, nonce = await get_form(client, token)
            await submit(client, token, nonce, key="DEMO_API_KEY", profiles=("alpha",))
        assert await pending == {"key": "DEMO_API_KEY", "success": True}
    finally:
        requests.close()


async def test_an_expired_link_reports_failure_to_the_waiting_turn(hermes_root, store, clock):
    runner, adapter = _runner()
    requests = secret_link.open_turn_secret_requests(
        runner, _source(), _SESSION_KEY, asyncio.get_running_loop(), lambda: False)
    try:
        capture = asyncio.create_task(asyncio.to_thread(_capture_in_turn))
        await _link_token(adapter)
        clock.now += 601
        store.sweep()
        result = await capture
    finally:
        requests.close()
    assert result == {"missing_names": ["DEMO_API_KEY"], "setup_skipped": True, "gateway_setup_hint": None}


async def test_turn_end_unblocks_a_waiting_request(hermes_root, store):
    runner, adapter = _runner()
    requests = secret_link.open_turn_secret_requests(
        runner, _source(), _SESSION_KEY, asyncio.get_running_loop(), lambda: False)
    pending = asyncio.create_task(asyncio.to_thread(requests.request, "DEMO_API_KEY"))
    await _link_token(adapter)
    requests.close()
    assert await pending == {"key": "DEMO_API_KEY", "success": False}
    assert secret_link.current_turn_secret_requester() is None


async def test_no_public_url_on_telegram_sends_the_notice_not_the_unsupported_hint(hermes_root, store):
    (hermes_root / "config.yaml").write_text("{}\n")
    runner, adapter = _runner()
    requests = secret_link.open_turn_secret_requests(
        runner, _source(), _SESSION_KEY, asyncio.get_running_loop(), lambda: False)
    try:
        result = await asyncio.to_thread(_capture_in_turn)
    finally:
        requests.close()
    assert result["gateway_setup_hint"] is None
    assert result["missing_names"] == ["DEMO_API_KEY"]
    assert adapter.send.await_args_list[0].args[1] == secret_link.NOT_CONFIGURED_MESSAGE


async def test_other_messaging_platforms_keep_the_unsupported_hint(hermes_root, store):
    runner, _adapter = _runner(Platform.DISCORD)
    requests = secret_link.open_turn_secret_requests(
        runner, _source(Platform.DISCORD), _SESSION_KEY, asyncio.get_running_loop(), lambda: False)
    try:
        result = await asyncio.to_thread(_capture_in_turn, "discord")
    finally:
        requests.close()
    assert result["gateway_setup_hint"] == GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
