"""Routing seam: routed messages dispatch to a worker, unrouted stay in-process,
and a dispatch failure is fail-closed (visible error, never silent host fallback)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _runner(adapter=None):
    r = object.__new__(gateway_run.GatewayRunner)
    r.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig()})
    r._worker_pool = None
    r.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    return r


def _event(profile=None, text="hi"):
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")
    ev = MagicMock()
    ev.text = text
    ev.source = src
    ev.routed_profile = profile
    ev.channel_prompt = None
    ev.media_urls = []
    return ev, src


@pytest.mark.asyncio
async def test_unrouted_returns_false():
    r = _runner()
    r._dispatch_to_worker = AsyncMock()
    ev, src = _event(profile=None)
    assert await r._maybe_dispatch_routed(ev, src) is False
    r._dispatch_to_worker.assert_not_awaited()


@pytest.mark.asyncio
async def test_routed_dispatches_and_handles():
    r = _runner()
    r._dispatch_to_worker = AsyncMock()
    ev, src = _event(profile="coder")
    assert await r._maybe_dispatch_routed(ev, src) is True
    r._dispatch_to_worker.assert_awaited_once()
    assert r._dispatch_to_worker.await_args.args[2] == "coder"


@pytest.mark.asyncio
async def test_dispatch_failure_is_fail_closed():
    adapter = MagicMock()
    adapter.send = AsyncMock()
    r = _runner(adapter)
    r._reply_anchor_for_event = MagicMock(return_value=None)
    r._dispatch_to_worker = AsyncMock(side_effect=RuntimeError("worker down"))
    ev, src = _event(profile="coder")
    # Handled (returns True) so the caller never falls through to the host handler.
    assert await r._maybe_dispatch_routed(ev, src) is True
    adapter.send.assert_awaited()  # a visible error went to the user
    assert "coder" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_dispatch_uses_profile_prefixed_session_key():
    adapter = MagicMock()
    adapter.send = AsyncMock()
    r = _runner(adapter)
    r._reply_anchor_for_event = MagicMock(return_value=None)

    fake_worker = MagicMock(base_url="http://127.0.0.1:5001", key="k")
    r._worker_pool = MagicMock()
    r._worker_pool.acquire = AsyncMock(return_value=fake_worker)

    client = MagicMock()
    client.dispatch = AsyncMock(return_value={"output": "done"})
    r._make_worker_client = MagicMock(return_value=client)

    ev, src = _event(profile="research")
    await r._dispatch_to_worker(ev, src, "research")

    r._worker_pool.acquire.assert_awaited_once_with("research")
    assert client.dispatch.await_args.kwargs["session_id"].startswith("agent:research:")
    adapter.send.assert_awaited()  # final output delivered


@pytest.mark.asyncio
async def test_dispatch_requests_continuity_and_supplies_approval_handler():
    adapter = MagicMock()
    adapter.send = AsyncMock()
    r = _runner(adapter)
    r._reply_anchor_for_event = MagicMock(return_value=None)

    fake_worker = MagicMock(base_url="http://127.0.0.1:5001", key="k")
    r._worker_pool = MagicMock()
    r._worker_pool.acquire = AsyncMock(return_value=fake_worker)
    client = MagicMock()
    client.dispatch = AsyncMock(return_value={"output": "done"})
    r._make_worker_client = MagicMock(return_value=client)

    ev, src = _event(profile="research")
    await r._dispatch_to_worker(ev, src, "research")

    kw = client.dispatch.await_args.kwargs
    assert kw["continue_session"] is True  # routed turns keep memory across turns
    assert kw["approval_handler"] is not None  # approvals are handled (fail-closed), not dropped


@pytest.mark.asyncio
async def test_routed_approval_is_denied_with_visible_notice():
    adapter = MagicMock()
    adapter.send = AsyncMock()
    r = _runner(adapter)
    r._reply_anchor_for_event = MagicMock(return_value=None)

    fake_worker = MagicMock(base_url="http://127.0.0.1:5001", key="k")
    r._worker_pool = MagicMock()
    r._worker_pool.acquire = AsyncMock(return_value=fake_worker)

    captured = {}

    async def dispatch(**kwargs):
        captured["choice"] = await kwargs["approval_handler"]({"tool": "shell"})
        return {"output": "done"}

    client = MagicMock()
    client.dispatch = AsyncMock(side_effect=dispatch)
    r._make_worker_client = MagicMock(return_value=client)

    ev, src = _event(profile="research")
    await r._dispatch_to_worker(ev, src, "research")

    assert captured["choice"] == "deny"
    assert any("approval" in c.args[1].lower() for c in adapter.send.await_args_list)


@pytest.mark.asyncio
async def test_maintain_worker_pool_reaps_and_sweeps():
    r = _runner()
    pool = MagicMock()
    pool.reap_exited = AsyncMock()
    pool.sweep_idle = AsyncMock()
    r._worker_pool = pool
    await r._maintain_worker_pool()
    pool.reap_exited.assert_awaited_once()
    pool.sweep_idle.assert_awaited_once()


@pytest.mark.asyncio
async def test_maintain_worker_pool_noop_without_pool():
    r = _runner()
    r._worker_pool = None
    await r._maintain_worker_pool()  # must not raise
