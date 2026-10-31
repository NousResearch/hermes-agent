"""Startup and concurrent reconnect attempts must preserve a single live owner."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner


def test_startup_preserves_reconnect_watcher_spawned_by_early_fatal_error():
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._failed_platforms = {Platform.DISCORD: {}}
    runner._scale_to_zero_should_arm = Mock(return_value=False)
    runner._log_scale_to_zero_not_armed = Mock()
    spawned = []

    def spawn(factory, name, **kwargs):
        task = Mock()
        task.done.return_value = False
        spawned.append((name, task))
        if callback := kwargs.get("on_spawn"):
            callback(task)
        return task

    runner._spawn_supervised = spawn
    runner._ensure_reconnect_watcher_running()
    original = runner._reconnect_watcher_task
    runner._start_spawn_background_watchers()

    reconnect_tasks = [task for name, task in spawned if name == "platform_reconnect_watcher"]
    assert reconnect_tasks == [original]
    assert runner._reconnect_watcher_task is original


@pytest.mark.asyncio
async def test_concurrent_watchers_cannot_connect_the_same_queued_platform():
    runner = object.__new__(GatewayRunner)
    info = {"config": PlatformConfig(enabled=True, token="test"), "attempts": 0, "next_retry": 0}
    runner._failed_platforms = {Platform.DISCORD: info}
    runner._flag_reconnect_needs_attention = Mock()
    runner._run_in_executor_with_context = AsyncMock()
    runner._wire_adapter_handlers = Mock()
    adapter = SimpleNamespace(_dedup=None)
    runner._create_adapter = Mock(return_value=adapter)
    entered, release = asyncio.Event(), asyncio.Event()

    async def connect(*args, **kwargs):
        entered.set()
        await release.wait()
        return True

    async def install(platform, candidate):
        runner._failed_platforms.pop(platform)

    runner._connect_adapter_with_timeout = AsyncMock(side_effect=connect)
    runner._install_reconnected_adapter = AsyncMock(side_effect=install)
    first = asyncio.create_task(runner._reconnect_failed_platform(Platform.DISCORD, 1))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        second = asyncio.create_task(runner._reconnect_failed_platform(Platform.DISCORD, 1))
        await asyncio.sleep(0)
        assert runner._connect_adapter_with_timeout.await_count == 1
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second is not None else []))
    runner._install_reconnected_adapter.assert_awaited_once_with(Platform.DISCORD, adapter)
    assert not info.get("connecting")


@pytest.mark.asyncio
@pytest.mark.parametrize("same_adapter", [False, True])
async def test_late_candidate_does_not_replace_or_close_the_installed_adapter(monkeypatch, same_adapter):
    runner = object.__new__(GatewayRunner)
    incumbent = SimpleNamespace(name="incumbent")
    candidate = incumbent if same_adapter else SimpleNamespace(name="late-candidate")
    runner.adapters = {Platform.DISCORD: incumbent}
    runner.delivery_router = SimpleNamespace(adapters=runner.adapters)
    runner._failed_platforms = {}
    runner._publish_primary_adapter = Mock()
    dispose = AsyncMock()
    monkeypatch.setattr("gateway.run._dispose_unused_adapter", dispose)

    await runner._install_reconnected_adapter(Platform.DISCORD, candidate)

    runner._publish_primary_adapter.assert_not_called()
    assert runner.adapters[Platform.DISCORD] is incumbent
    assert runner.delivery_router.adapters[Platform.DISCORD] is incumbent
    if same_adapter:
        dispose.assert_not_awaited()
    else:
        dispose.assert_awaited_once_with(candidate)


@pytest.fixture
def reconnect_attempt(monkeypatch):
    runner = object.__new__(GatewayRunner)
    info = {"config": PlatformConfig(enabled=True, token="test"), "attempts": 0, "next_retry": 0}
    runner._failed_platforms = {Platform.DISCORD: info}
    runner.adapters = {}
    runner.delivery_router = SimpleNamespace(adapters=runner.adapters)
    candidate = SimpleNamespace(send_path_degraded=False)
    runner._flag_reconnect_needs_attention = Mock()
    runner._run_in_executor_with_context = AsyncMock()
    runner._wire_adapter_handlers = Mock()
    runner._create_adapter = Mock(return_value=candidate)
    runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
    runner._sync_voice_mode_state_to_adapter = Mock()
    runner._bind_voice_input_callback = Mock()
    runner._update_platform_runtime_status = Mock()
    runner._schedule_planned_restart_replay = Mock()
    runner._redeliver_failed_obligations_for_platform = AsyncMock()
    runner._schedule_resume_pending_sessions = Mock()
    monkeypatch.setattr("gateway.channel_directory.build_channel_directory", AsyncMock())
    dispose = AsyncMock()
    monkeypatch.setattr("gateway.run._dispose_unused_adapter", dispose)
    return runner, info, candidate, dispose


@pytest.mark.asyncio
async def test_cancelled_connect_disposes_candidate_before_releasing_claim(reconnect_attempt):
    runner, info, candidate, dispose = reconnect_attempt
    runner._connect_adapter_with_timeout.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await runner._reconnect_failed_platform(Platform.DISCORD, 1)
    dispose.assert_awaited_once_with(candidate)
    assert runner._failed_platforms[Platform.DISCORD] is info
    assert not info.get("connecting")
    assert not runner.adapters


@pytest.mark.asyncio
async def test_wiring_failure_keeps_outage_queued_and_disposes_unpublished_candidate(reconnect_attempt):
    runner, info, candidate, dispose = reconnect_attempt
    runner._bind_voice_input_callback.side_effect = RuntimeError("wiring failed")
    await runner._reconnect_failed_platform(Platform.DISCORD, 1)
    assert runner._failed_platforms[Platform.DISCORD] is info
    assert info["attempts"] == 1
    assert not info.get("connecting")
    assert not runner.adapters
    dispose.assert_awaited_once_with(candidate)


@pytest.mark.asyncio
async def test_followup_failure_does_not_dispose_published_adapter(reconnect_attempt):
    runner, info, candidate, dispose = reconnect_attempt
    runner._schedule_planned_restart_replay.side_effect = RuntimeError("follow-up failed")
    await runner._reconnect_failed_platform(Platform.DISCORD, 1)
    assert runner.adapters[Platform.DISCORD] is candidate
    assert Platform.DISCORD not in runner._failed_platforms
    assert info["attempts"] == 0
    assert not info.get("connecting")
    dispose.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_reconnect_publishes_and_replays_once(reconnect_attempt):
    runner, info, candidate, dispose = reconnect_attempt
    await runner._reconnect_failed_platform(Platform.DISCORD, 1)
    assert runner.adapters[Platform.DISCORD] is candidate
    assert runner.delivery_router.adapters[Platform.DISCORD] is candidate
    assert Platform.DISCORD not in runner._failed_platforms
    assert not info.get("connecting")
    runner._schedule_planned_restart_replay.assert_called_once_with()
    runner._redeliver_failed_obligations_for_platform.assert_awaited_once_with(Platform.DISCORD)
    dispose.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [False, True])
async def test_retired_attempt_cannot_consume_a_new_outage(reconnect_attempt, success):
    runner, info, candidate, dispose = reconnect_attempt
    new_info = {"config": info["config"], "attempts": 0, "next_retry": 0}

    async def connect(*args, **kwargs):
        runner._failed_platforms[Platform.DISCORD] = new_info
        return success

    runner._connect_adapter_with_timeout.side_effect = connect
    await runner._reconnect_failed_platform(Platform.DISCORD, 1)
    assert runner._failed_platforms[Platform.DISCORD] is new_info
    assert not runner.adapters
    assert new_info["attempts"] == 0
    assert not info.get("connecting")
    dispose.assert_awaited_once_with(candidate)


@pytest.mark.asyncio
async def test_registry_lookup_cannot_drop_a_replacement_outage(reconnect_attempt):
    runner, info, candidate, dispose = reconnect_attempt
    new_info = {"config": info["config"], "attempts": 0, "next_retry": 0}

    async def registry_lookup(*args, **kwargs):
        runner._failed_platforms[Platform.DISCORD] = new_info

    runner._run_in_executor_with_context.side_effect = registry_lookup
    runner._create_adapter.return_value = None
    runner._adapter_may_heal = Mock(return_value=False)
    await runner._reconnect_failed_platform(Platform.DISCORD, 1)
    assert runner._failed_platforms[Platform.DISCORD] is new_info
    runner._update_platform_runtime_status.assert_not_called()
    dispose.assert_not_awaited()
    assert not info.get("connecting")


@pytest.mark.asyncio
async def test_non_retryable_cleanup_cannot_delete_a_replacement_outage(reconnect_attempt):
    runner, info, candidate, dispose = reconnect_attempt
    new_info = {"config": info["config"], "attempts": 0, "next_retry": 0}
    candidate.has_fatal_error = True
    candidate.fatal_error_retryable = False
    candidate.fatal_error_message = "invalid test credential"
    runner._mark_platform_fatal = Mock()
    runner._connect_adapter_with_timeout.return_value = False

    async def cleanup(*args, **kwargs):
        runner._failed_platforms[Platform.DISCORD] = new_info

    dispose.side_effect = cleanup
    await runner._reconnect_failed_platform(Platform.DISCORD, 1)
    assert runner._failed_platforms[Platform.DISCORD] is new_info
    dispose.assert_awaited_once_with(candidate)
    assert not info.get("connecting")
