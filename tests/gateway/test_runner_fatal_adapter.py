import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner


class _FatalAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "telegram_token_lock",
            "Another local Hermes gateway is already using this Telegram bot token.",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _RuntimeRetryableAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.WHATSAPP)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_requests_clean_exit_for_nonretryable_startup_conflict(monkeypatch, tmp_path):
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _FatalAdapter())

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is True
    assert "already using this Telegram bot token" in runner.exit_reason


@pytest.mark.asyncio
async def test_runner_queues_retryable_runtime_fatal_for_reconnection(monkeypatch, tmp_path):
    """Retryable runtime fatal errors queue the platform for reconnection
    AND keep the gateway alive — the background reconnect watcher recovers
    the platform when the underlying issue clears.  (Previously this
    exited-with-failure to trigger a systemd restart; that converted
    transient failures into infinite restart loops.)
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "WhatsApp bridge process exited unexpectedly (code 1).",
        retryable=True,
    )

    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    # Gateway stays alive — watcher will retry in background
    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False
    assert Platform.WHATSAPP in runner._failed_platforms
    assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0


@pytest.mark.asyncio
async def test_concurrent_fatal_notifications_disconnect_same_adapter_once(monkeypatch, tmp_path):
    """
    Two fatal-error notifications for the same still-installed adapter (e.g.
    from two concurrent recovery paths racing on the same underlying outage)
    must result in exactly one disconnect() call.

    Regression test for the TOCTOU race in _handle_adapter_fatal_error: the
    old code only removed the adapter from self.adapters in a `finally` block
    *after* awaiting disconnect(), so a second concurrent call could still see
    itself as "existing" and disconnect() the same object twice — the
    concrete origin of the "'NoneType' object has no attribute 'updater'"
    crash when the adapter's own teardown code re-reads self._app afterwards.
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "WhatsApp bridge process exited unexpectedly (code 1).",
        retryable=True,
    )

    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    disconnect_calls = 0
    release_second_call = asyncio.Event()

    async def slow_disconnect():
        nonlocal disconnect_calls
        disconnect_calls += 1
        # Yield control so the second concurrent notification can run its
        # "existing is adapter" check before this call finishes tearing down.
        release_second_call.set()
        await asyncio.sleep(0)
        adapter._mark_disconnected()

    monkeypatch.setattr(adapter, "disconnect", slow_disconnect)

    await asyncio.gather(
        runner._handle_adapter_fatal_error(adapter),
        runner._handle_adapter_fatal_error(adapter),
    )

    assert disconnect_calls == 1


@pytest.mark.asyncio
async def test_stale_fatal_notification_from_superseded_adapter_is_ignored(monkeypatch, tmp_path):
    """
    A delayed fatal-error notification from an adapter instance that has
    since been replaced by a different, already-installed adapter (e.g. a
    background retry chain on the old instance finally giving up after a
    reconnect on a new instance already succeeded) must be ignored: it must
    not disconnect the new adapter, must not re-queue an already-healthy
    platform for reconnection, and must not shut the gateway down.
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    old_adapter = _RuntimeRetryableAdapter()
    old_adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "stale failure from a superseded adapter instance",
        retryable=True,
    )

    new_adapter = _RuntimeRetryableAdapter()
    new_adapter.disconnect = AsyncMock()
    runner.adapters = {Platform.WHATSAPP: new_adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(old_adapter)

    new_adapter.disconnect.assert_not_awaited()
    assert runner.adapters[Platform.WHATSAPP] is new_adapter
    assert Platform.WHATSAPP not in runner._failed_platforms
    runner.stop.assert_not_awaited()
