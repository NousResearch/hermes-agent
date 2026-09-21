"""Planned-restart online notice survives an offline home-channel transport at boot.

The ``.restart_pending.json`` marker used to be consumed in ``finally`` even when no live transport
existed for the home channel, so the "Gateway online" notice was never sent and never replayed.
See #112109. Runs the real boot pass, marker helpers, home-channel sender and DeliveryTransport.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.delivery as delivery
import gateway.run as gateway_run
import gateway.run_restart_notifications as notices
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.teams.adapter import TeamsAdapter
from plugins.platforms.telegram.adapter import TelegramAdapter

ONLINE_NOTICE = "♻️ Gateway online — Hermes is back and ready."


def _adapter():
    return SimpleNamespace(
        send_path_degraded=False,
        send=AsyncMock(return_value=SendResult(success=True, message_id="unit-test-notice")),
    )


@pytest.fixture
def boot_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    # Await the boot task to completion and propagate failures deterministically.
    monkeypatch.setattr(gateway_run, "_startup_restore_drain_timeout_secs", lambda: 0)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                gateway_restart_notification=True,
                home_channel=HomeChannel(platform=Platform.DISCORD, chat_id="unit-test-home", name="Test home"),
            ),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner.adapters = {}
    runner.delivery_router = SimpleNamespace(adapters=runner.adapters)
    runner._failed_platforms = {}
    runner._sync_voice_mode_state_to_adapter = Mock()
    runner._bind_voice_input_callback = Mock()
    runner._update_platform_runtime_status = Mock()
    runner._redeliver_failed_obligations_for_platform = AsyncMock()
    runner._schedule_resume_pending_sessions = Mock()
    monkeypatch.setattr("gateway.channel_directory.build_channel_directory", AsyncMock())
    # Unrelated conversation recovery and optional account-status text are isolated.
    runner._claim_pending_obligations = AsyncMock(return_value=[])
    runner._redeliver_claimed_obligations = AsyncMock(return_value=0)
    runner._free_tier_startup_line = Mock(return_value=None)
    marker = tmp_path / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")
    return runner, marker


async def _boot(runner):
    await runner._await_startup_boot_sends(
        planned_restart_notification_pending=gateway_run._planned_restart_notification_pending()
    )


async def _reconnect(runner, platform, adapter):
    runner._failed_platforms[platform] = {}
    await runner._install_reconnected_adapter(platform, adapter)
    await asyncio.gather(*runner._background_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True], ids=["offline-at-boot-replayed-on-reconnect", "live-at-boot"])
async def test_planned_restart_notice_reaches_home_channel(boot_notice, live):
    runner, marker = boot_notice
    adapter = _adapter()
    if live:
        runner.adapters[Platform.DISCORD] = adapter
    transport = delivery.resolve_delivery_transport(Platform.DISCORD, runner.config, runner.adapters)
    assert (transport is not None) is live

    await _boot(runner)

    runner._redeliver_claimed_obligations.assert_awaited_once_with([])
    if not live:
        adapter.send.assert_not_called()
        assert marker.exists(), "marker must survive a boot with no live transport"
        await _reconnect(runner, Platform.DISCORD, adapter)
    adapter.send.assert_awaited_once_with("unit-test-home", ONLINE_NOTICE, metadata={"non_conversational": True})
    assert not marker.exists()


@pytest.mark.asyncio
async def test_partial_delivery_is_persisted_and_not_repeated(boot_notice):
    runner, marker = boot_notice
    telegram, discord = _adapter(), _adapter()
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="other-home", thread_id="7", name="Other"),
    )
    runner.config.platforms[Platform.SLACK] = PlatformConfig(
        enabled=True, gateway_restart_notification=False,
        home_channel=HomeChannel(platform=Platform.SLACK, chat_id="muted-home", name="Muted"),
    )
    runner.adapters[Platform.TELEGRAM] = telegram

    await _boot(runner)

    telegram.send.assert_awaited_once()
    discord.send.assert_not_called()
    assert json.loads(marker.read_text(encoding="utf-8"))["delivered_targets"] == [["telegram", "other-home", "7"]]

    # A fresh process has no in-memory history: dedupe must come from the marker. The opted-out
    # Slack home is never owed a notice, so Discord's delivery completes the set.
    recovered = object.__new__(gateway_run.GatewayRunner)
    recovered.__dict__.update(runner.__dict__)
    await _reconnect(recovered, Platform.DISCORD, discord)

    discord.send.assert_awaited_once()
    telegram.send.assert_awaited_once()
    assert not marker.exists()

    # Nothing pending: a later reconnect stays silent.
    await _reconnect(recovered, Platform.DISCORD, discord)
    discord.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["polling", "known-unsent"])
async def test_planned_home_recovers_without_another_reconnect(boot_notice, monkeypatch, recovery):
    runner, marker = boot_notice
    runner._running, runner._restart_requested = True, False
    discord = _adapter()
    runner.adapters[Platform.DISCORD] = discord
    cfg = PlatformConfig(enabled=True, token="test-token", home_channel=HomeChannel(
        platform=Platform.TELEGRAM, chat_id="42", thread_id="7", name="Telegram"))
    runner.config.platforms[Platform.TELEGRAM] = cfg
    telegram = TelegramAdapter(cfg)
    telegram._running = True
    telegram._rich_send_disabled = True
    telegram.send_typing = AsyncMock()
    telegram._write_runtime_status_safe = Mock()
    generation, _ = telegram._begin_polling_generation()
    telegram._send_path_degraded = recovery == "polling"
    telegram._wait_for_reconnection = AsyncMock(return_value=False)
    bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=42)))
    if recovery == "polling":
        telegram._bot = bot
    waiting, release = asyncio.Event(), asyncio.Event()

    async def recover(delay):
        waiting.set()
        await release.wait()
        await asyncio.sleep(0)

    class AsyncioProxy:
        def __getattr__(self, name):
            return recover if name == "sleep" else getattr(asyncio, name)
    monkeypatch.setattr(notices, "asyncio", AsyncioProxy())
    runner._failed_platforms[Platform.TELEGRAM] = {}
    await runner._install_reconnected_adapter(Platform.TELEGRAM, telegram)
    task = asyncio.gather(*runner._background_tasks)
    try:
        await asyncio.wait_for(waiting.wait(), 3)
        discord.send.assert_awaited_once()
        bot.send_message.assert_not_awaited()
        assert json.loads(marker.read_text())["delivered_targets"] == [["discord", "unit-test-home", None]]
        if recovery == "polling":
            assert telegram._record_polling_progress(generation)
        else:
            telegram._wait_for_reconnection.assert_awaited_once()
            telegram._bot = bot
        concurrent = asyncio.create_task(runner._replay_pending_planned_restart_notification())
        release.set()
        await asyncio.wait_for(asyncio.gather(task, concurrent), 3)
        bot.send_message.assert_awaited_once()
        assert bot.send_message.call_args.kwargs["message_thread_id"] == 7
        discord.send.assert_awaited_once()
        assert not marker.exists()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_late_reconnect_progress_is_independent_and_deduplicated(boot_notice, monkeypatch):
    runner, marker = boot_notice
    runner._running, runner._restart_requested = True, False
    runner.config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True, home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="second", name="Second"))
    discord, telegram = _adapter(), _adapter()
    discord.send_path_degraded = True
    runner.adapters[Platform.DISCORD] = discord
    waiting, absent, sending = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release_wait, release_send = asyncio.Event(), asyncio.Event()
    resolve = delivery.resolve_delivery_transport

    def resolve_transport(platform, *args):
        transport = resolve(platform, *args)
        if platform == Platform.TELEGRAM and transport is None:
            absent.set()
        return transport

    async def wait(delay):
        waiting.set()
        await release_wait.wait()

    async def send(*args, **kwargs):
        sending.set()
        await release_send.wait()
        return SendResult(success=True, message_id="accepted")

    class AsyncioProxy:
        def __getattr__(self, name):
            return wait if name == "sleep" else getattr(asyncio, name)

    monkeypatch.setattr(notices, "asyncio", AsyncioProxy())
    monkeypatch.setattr(delivery, "resolve_delivery_transport", resolve_transport)
    telegram.send.side_effect = send
    first = asyncio.create_task(runner._replay_pending_planned_restart_notification())
    try:
        await asyncio.wait_for(asyncio.gather(waiting.wait(), absent.wait()), 3)
        runner._failed_platforms[Platform.TELEGRAM] = {}
        await runner._install_reconnected_adapter(Platform.TELEGRAM, telegram)
        await asyncio.wait_for(sending.wait(), 3)
        # A third replay must neither duplicate the live attempt nor lose its progress.
        await asyncio.wait_for(runner._replay_pending_planned_restart_notification(), 3)
        telegram.send.assert_awaited_once()
        assert json.loads(marker.read_text())["attempted_targets"] == [["telegram", "second", None]]
        release_send.set()
        await asyncio.wait_for(asyncio.gather(*runner._background_tasks), 3)
        assert json.loads(marker.read_text())["delivered_targets"] == [["telegram", "second", None]]
        assert not first.done(), "unrelated degraded home is still waiting"
        discord.send_path_degraded = False
        release_wait.set()
        await asyncio.wait_for(first, 3)
        discord.send.assert_awaited_once()
        telegram.send.assert_awaited_once()
        assert not marker.exists()
    finally:
        release_wait.set()
        release_send.set()
        tasks = [first, *getattr(runner, "_background_tasks", [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_replay_does_not_start_dispatch_and_preserves_debt(boot_notice, monkeypatch):
    runner, marker = boot_notice
    runner._running, runner._restart_requested = True, False
    adapter = _adapter()
    runner.adapters[Platform.DISCORD] = adapter
    cancelled = asyncio.Event()

    def create_task(coro):
        if coro.cr_code.co_name == "dispatch":
            replay.cancel()
            cancelled.set()
        return asyncio.create_task(coro)

    class AsyncioProxy:
        def __getattr__(self, name):
            return create_task if name == "create_task" else getattr(asyncio, name)

    with monkeypatch.context() as dispatch_patch:
        dispatch_patch.setattr(notices, "asyncio", AsyncioProxy())
        replay = asyncio.create_task(runner._replay_pending_planned_restart_notification())
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(replay, 3)
    assert cancelled.is_set()
    adapter.send.assert_not_awaited()
    data = json.loads(marker.read_text())
    assert not data["attempted_targets"]
    assert not data["delivered_targets"]
    # An uncancelled replay still owes the never-dispatched home its notice.
    await runner._replay_pending_planned_restart_notification()
    adapter.send.assert_awaited_once()
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [
    "exhausted", "refused-exhausted", "cancel-wait", "cancel-send", "timeout", "ambiguous", "retryable",
    "new-marker-wait", "new-marker-send", "restart", "boot-replaced", "boot-empty",
])
async def test_planned_home_progress_and_generation_survive_interruption(boot_notice, monkeypatch, outcome):
    runner, marker = boot_notice
    runner._running, runner._restart_requested = True, False
    monkeypatch.setattr(notices, "_RESTART_NOTICE_TIMEOUT", 2.0)
    discord = _adapter()
    runner.adapters[Platform.DISCORD] = discord
    cfg = PlatformConfig(enabled=True, home_channel=HomeChannel(
        platform=Platform("teams"), chat_id="42", name="Teams"))
    adapter = TeamsAdapter(cfg)
    adapter._running = True
    runner.config.platforms[Platform("teams")] = cfg
    runner.adapters[Platform("teams")] = adapter
    waiting, sending, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    accepted = []
    wait_outcomes = {"exhausted", "refused-exhausted", "cancel-wait", "new-marker-wait", "restart"}
    monkeypatch.setattr(type(adapter), "send_path_degraded", property(lambda _: outcome in wait_outcomes - {"refused-exhausted"}))

    async def provider_send(chat_id, content):
        accepted.append((chat_id, content))
        sending.set()
        if outcome in {"cancel-send", "timeout", "new-marker-send"}:
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()  # cancellation-resistant provider must not pin shutdown
        if outcome == "ambiguous":
            raise TimeoutError("provider accepted; response was lost")
        return SimpleNamespace(id="accepted")

    adapter._app = SimpleNamespace(send=provider_send)
    if outcome == "refused-exhausted":
        adapter.send = AsyncMock(return_value=SendResult(
            success=False, retryable=True, known_unsent=True, retry_after=60))
    if outcome == "retryable":
        # Retryable alone is not proof of non-delivery (other adapters can return this).
        adapter.send = AsyncMock(return_value=SendResult(success=False, retryable=True))

    async def wait(delay):
        waiting.set()
        if outcome in {"exhausted", "refused-exhausted"}:
            await asyncio.sleep(delay)
        else:
            await release.wait()
    class AsyncioProxy:
        def __getattr__(self, name):
            return wait if name == "sleep" else getattr(asyncio, name)
    monkeypatch.setattr(notices, "asyncio", AsyncioProxy())

    if outcome == "boot-empty":
        marker.unlink()
    runner._capture_planned_restart_notification()
    newer = '{"requested_at": 999, "via_service": true}'
    if outcome in {"boot-empty", "boot-replaced"}:
        marker.write_text(newer)
    task = asyncio.create_task(runner._replay_pending_planned_restart_notification())
    try:
        if outcome in wait_outcomes | {"cancel-send", "new-marker-send"}:
            await asyncio.wait_for((waiting if outcome in wait_outcomes else sending).wait(), 3)
            if outcome.startswith("new-marker"):
                marker.write_text(newer)
            if outcome == "restart":
                runner._restart_requested = True
            if outcome.startswith("cancel"):
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)
            elif outcome not in {"exhausted", "refused-exhausted"}:
                release.set()
        if not task.cancelled():
            await asyncio.wait_for(task, 5)
        if outcome.startswith("new-marker") or outcome.startswith("boot-"):
            assert marker.read_text() == newer
            await runner._replay_pending_planned_restart_notification()
            assert marker.read_text() == newer
            assert len(accepted) == (1 if outcome == "new-marker-send" else 0)
            return
        discord.send.assert_awaited_once()
        if outcome in wait_outcomes:
            data = json.loads(marker.read_text())
            assert data["delivered_targets"] == [["discord", "unit-test-home", None]]
            assert not data["attempted_targets"]
            assert not accepted
            if outcome == "refused-exhausted":
                adapter.send.assert_awaited_once()  # retry_after exceeds budget: do not retry early
        else:
            # Cancellation may leave a consumed-attempt marker until the next replay.
            await runner._replay_pending_planned_restart_notification()
            assert not marker.exists()
            assert len(accepted) == (0 if outcome == "retryable" else 1)
            if outcome == "retryable":
                adapter.send.assert_awaited_once()
        # A new process retries only the homes still owed, never the delivered/ambiguous ones.
        fresh = object.__new__(gateway_run.GatewayRunner)
        fresh.__dict__.update(runner.__dict__)
        del fresh._planned_restart_notification_payload
        fresh._restart_requested = False
        replacement = _adapter()
        fresh.adapters = {Platform.DISCORD: discord, Platform("teams"): replacement}
        await fresh._replay_pending_planned_restart_notification()
        assert replacement.send.await_count == (1 if outcome in wait_outcomes else 0)
        discord.send.assert_awaited_once()
        assert not marker.exists()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
