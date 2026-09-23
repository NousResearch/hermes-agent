"""Every adapter-side session key goes through one seam (#88715, invariant 4).

A key an adapter derives for batching / queueing / busy detection must be the key the runner
derives for the same event, for a primary bot and for a secondary-owned bot alike. Yuanbao's
``DispatchMiddleware`` used the free ``build_session_key()`` (no profile) while ``handle_message``
keyed under ``agent:<owner>:``, so a secondary bot's per-group queue and RecallGuard entries lived
in a lane the runner never popped.
"""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.yuanbao import DispatchMiddleware, InboundContext, YuanbaoAdapter
from gateway.profile_routing import parse_profile_routes


def _yuanbao(owner):
    adapter = YuanbaoAdapter(PlatformConfig(extra={
        "app_id": "k", "app_secret": "s", "ws_url": "wss://x", "api_domain": "https://x",
        "group_sessions_per_user": True, "thread_sessions_per_user": False}))
    adapter.set_owner_profile(owner)
    return adapter


def _runner(adapter, owner):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.profile_routes = parse_profile_routes([])
    runner._primary_profile_name = "default"
    runner.adapters = {} if owner else {Platform.YUANBAO: adapter}
    runner._profile_adapters = {owner: {Platform.YUANBAO: adapter}} if owner else {}
    adapter.gateway_runner = runner
    return runner


@pytest.mark.parametrize("owner", [None, "acme"])
def test_adapter_batch_key_equals_runner_session_key(owner):
    adapter = _yuanbao(owner)
    runner = _runner(adapter, owner)
    source = adapter.build_source(chat_id="grp-1", chat_type="group", user_id="u1")
    ctx = InboundContext(adapter=adapter, chat_type="group", chat_id="grp-1", raw_text="hi", msg_id="m1", source=source)
    seen = []

    async def go():
        async def _next():
            pass

        async def _capture(event):
            seen.append(adapter._event_session_key(event))

        adapter.handle_message = _capture
        await DispatchMiddleware().handle(ctx, _next)
        queued = list(adapter._group_queues)
        await asyncio.sleep(0.05)  # let the group consumer dispatch once
        return queued

    queued = asyncio.run(go())
    runner_key = runner._session_key_for_source(source)
    expected_ns = f"agent:{owner}" if owner else "agent:main"
    assert runner_key.startswith(expected_ns + ":"), runner_key
    assert queued == [runner_key], (queued, runner_key)
    assert seen == [runner_key]
    assert adapter._processing_msg_ids == {runner_key: "m1"}
