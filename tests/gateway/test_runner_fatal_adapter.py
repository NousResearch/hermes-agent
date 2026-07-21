import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


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


class _ReplacementDeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="token", typing_indicator=False),
            Platform.DISCORD,
        )
        self.sent: list[str] = []
        self.connected = True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self.connected = False

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if not self.connected:
            return SendResult(success=False, error="Not connected")
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

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
async def test_retryable_fatal_queues_reconnect_after_cancellation_swallowing_disconnect(
    monkeypatch, tmp_path
):
    """A wedged old adapter cannot block runner-owned reconnect recovery."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("transport_stale", "transport stale", retryable=True)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters

    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def swallow_cancellation():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        finished.set()

    monkeypatch.setattr(adapter, "disconnect", swallow_cancellation)
    operation = asyncio.create_task(runner._handle_adapter_fatal_error(adapter))
    await started.wait()
    done, _pending = await asyncio.wait({operation}, timeout=0.2)
    try:
        assert operation in done
        assert runner.adapters == {}
        assert Platform.WHATSAPP in runner._failed_platforms
        assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0
    finally:
        release.set()
        await asyncio.wait({operation}, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "reviewer"], ids=["primary", "secondary"])
async def test_inflight_final_reply_uses_replacement_adapter_after_reconnect(
    tmp_path, profile
):
    config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    old_adapter = _ReplacementDeliveryAdapter()
    replacement = _ReplacementDeliveryAdapter()
    old_adapter.gateway_runner = runner
    replacement.gateway_runner = runner
    if profile:
        runner.adapters = {}
        runner._profile_adapters = {profile: {Platform.DISCORD: old_adapter}}
    else:
        runner.adapters = {Platform.DISCORD: old_adapter}
    runner.delivery_router.adapters = runner.adapters

    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(_event):
        await old_adapter.send("channel-1", "partial preview")
        handler_started.set()
        await release_handler.wait()
        return "complete final reply"

    old_adapter.set_message_handler(handler)
    event = MessageEvent(
        text="long-running request",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="channel-1",
            chat_type="dm",
            user_id="user-1",
            profile=profile,
        ),
        message_id="inbound-1",
    )
    task = asyncio.create_task(
        old_adapter._process_message_background(event, build_session_key(event.source))
    )
    await handler_started.wait()

    await old_adapter.disconnect()
    if profile:
        runner._profile_adapters[profile][Platform.DISCORD] = replacement
    else:
        runner.adapters = {Platform.DISCORD: replacement}
    runner.delivery_router.adapters = runner.adapters
    release_handler.set()
    await task

    assert old_adapter.sent == ["partial preview"]
    assert replacement.sent == ["complete final reply"]
