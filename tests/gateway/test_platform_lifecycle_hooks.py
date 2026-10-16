"""Tests for the platform:* lifecycle hook events (platform:connected, platform:fatal,
platform:needs_attention), which let a hook react to a platform going down or coming back."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from tests.gateway.test_platform_reconnect import StubAdapter, _make_runner


def _runner_with_hooks():
    runner = _make_runner()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._background_tasks = set()
    runner._fatal_handler_tasks = set()
    runner._reconnect_watcher_task = None
    return runner


async def _emitted(runner):
    """Let the background emit tasks run, then return (event, context) pairs."""
    await asyncio.gather(*runner._background_tasks)
    return [c.args for c in runner.hooks.emit.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_reconnect", [False, True])
async def test_successful_connect_emits_platform_connected(is_reconnect):
    runner = _runner_with_hooks()

    await runner._connect_adapter_with_timeout(StubAdapter(succeed=True), Platform.TELEGRAM, is_reconnect=is_reconnect)

    assert await _emitted(runner) == [("platform:connected", {"platform": "telegram", "reconnect": is_reconnect})]


@pytest.mark.asyncio
async def test_failed_connect_emits_nothing():
    runner = _runner_with_hooks()

    await runner._connect_adapter_with_timeout(StubAdapter(succeed=False), Platform.TELEGRAM)

    assert await _emitted(runner) == []


@pytest.mark.asyncio
async def test_runtime_fatal_error_emits_platform_fatal():
    runner = _runner_with_hooks()
    runner.stop = AsyncMock()
    adapter = StubAdapter(succeed=True)
    adapter._set_fatal_error("whatsapp_bridge_exited", "WhatsApp bridge process exited unexpectedly", retryable=True)
    runner.adapters[Platform.TELEGRAM] = adapter

    with patch.object(runner, "_ensure_reconnect_watcher_running"):
        await runner._handle_adapter_fatal_error(adapter)

    assert ("platform:fatal", {
        "platform": "telegram",
        "error_code": "whatsapp_bridge_exited",
        "error_message": "WhatsApp bridge process exited unexpectedly",
        "retryable": True,
    }) in await _emitted(runner)


@pytest.mark.asyncio
async def test_needs_attention_emits_once():
    runner = _runner_with_hooks()
    runner._update_platform_runtime_status = MagicMock()
    now = time.monotonic()
    info = {"queued_at": now - 7200, "attempts": 26}

    with patch("gateway.run._reconnect_needs_attention", return_value=True):
        runner._flag_reconnect_needs_attention(Platform.TELEGRAM, info, now)
        runner._flag_reconnect_needs_attention(Platform.TELEGRAM, info, now + 60)

    assert await _emitted(runner) == [("platform:needs_attention", {
        "platform": "telegram",
        "status_key": "telegram",
        "attempts": 26,
        "down_for_seconds": 7200,
    })]


@pytest.mark.asyncio
async def test_runner_without_hooks_emits_nothing():
    runner = _make_runner()

    assert await runner._connect_adapter_with_timeout(StubAdapter(succeed=True), Platform.TELEGRAM) is True
