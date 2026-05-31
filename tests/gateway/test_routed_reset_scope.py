"""A /reset in a routed context clears the worker's session, never the host's."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _runner():
    r = object.__new__(gateway_run.GatewayRunner)
    r.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig()})
    r._worker_pool = MagicMock()
    r._worker_pool.acquire = AsyncMock(return_value=MagicMock(base_url="http://127.0.0.1:5001", key="k"))
    r._profile_rate_limiter = None
    adapter = MagicMock()
    adapter.send = AsyncMock()
    r.adapters = {Platform.TELEGRAM: adapter}
    r.session_store = MagicMock()  # host store — must NOT be touched
    r._reply_anchor_for_event = MagicMock(return_value=None)
    return r, adapter


def _reset_event(text="/reset"):
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="100", thread_id="t1", chat_type="group", user_id="u1")
    from gateway.platforms.base import MessageEvent, MessageType

    ev = MessageEvent(text=text, message_type=MessageType.TEXT, source=src)
    ev.routed_profile = "coder"
    return ev, src


@pytest.mark.asyncio
async def test_reset_forwards_to_worker_with_profile_key():
    r, adapter = _runner()
    client = MagicMock()
    client.reset_session = AsyncMock()
    r._make_worker_client = MagicMock(return_value=client)

    ev, src = _reset_event()
    assert await r._maybe_dispatch_routed(ev, src) is True

    client.reset_session.assert_awaited_once()
    assert client.reset_session.await_args.args[0].startswith("agent:coder:")
    r.session_store.reset_session.assert_not_called()  # host untouched
    adapter.send.assert_awaited()


@pytest.mark.asyncio
async def test_plain_message_does_not_reset():
    r, _ = _runner()
    r._dispatch_to_worker = AsyncMock()
    r._reset_routed_worker = AsyncMock()
    ev, src = _reset_event(text="just chatting")
    await r._maybe_dispatch_routed(ev, src)
    r._reset_routed_worker.assert_not_awaited()
    r._dispatch_to_worker.assert_awaited_once()
