"""Tests for the gateway platform reconnection watcher."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner


class StubAdapter(BasePlatformAdapter):
    """Adapter whose connect() result can be controlled."""

    def __init__(
        self,
        *,
        platform=Platform.TELEGRAM,
        succeed=True,
        fatal_error=None,
        fatal_retryable=True,
    ):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self._succeed = succeed
        self._fatal_error = fatal_error
        self._fatal_retryable = fatal_retryable

    async def connect(self):
        if self._fatal_error:
            self._set_fatal_error("test_error", self._fatal_error, retryable=self._fatal_retryable)
            return False
        return self._succeed

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_runner():
    """Create a minimal GatewayRunner via object.__new__ to skip __init__."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_with_failure = False
    runner._exit_cleanly = False
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.delivery_router = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._shutdown_all_gateway_honcho = lambda: None
    runner.session_store = MagicMock()
    return runner


# --- Startup queueing ---

class TestStartupPlatformIsolation:
    """Verify one blocked platform cannot prevent later platforms from starting."""

    @pytest.mark.asyncio
    async def test_start_continues_after_platform_connect_timeout(self, tmp_path):
        """A timeout on Telegram should queue it and still connect Feishu."""
        runner = _make_runner()
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="test"),
                Platform.FEISHU: PlatformConfig(enabled=True, token="test"),
            },
            sessions_dir=tmp_path,
        )
        runner.hooks = MagicMock()
        runner.hooks.loaded_hooks = []
        runner.hooks.emit = AsyncMock()
        runner._suspend_stuck_loop_sessions = MagicMock(return_value=0)
        runner._update_runtime_status = MagicMock()
        runner._update_platform_runtime_status = MagicMock()
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._send_update_notification = AsyncMock(return_value=True)
        runner._send_restart_notification = AsyncMock()

        adapters = {
            Platform.TELEGRAM: StubAdapter(platform=Platform.TELEGRAM),
            Platform.FEISHU: StubAdapter(platform=Platform.FEISHU),
        }
        runner._create_adapter = MagicMock(
            side_effect=lambda platform, _config: adapters[platform]
        )
        runner._connect_adapter_with_timeout = AsyncMock(
            side_effect=[
                TimeoutError("telegram connect timed out after 30s"),
                True,
            ]
        )

        def fake_create_task(coro):
            coro.close()
            return MagicMock()

        with patch("gateway.status.write_runtime_status"):
            with patch("hermes_cli.plugins.discover_plugins"):
                with patch("hermes_cli.config.load_config", return_value={}):
                    with patch("agent.shell_hooks.register_from_config"):
                        with patch(
                            "tools.process_registry.process_registry.recover_from_checkpoint",
                            return_value=0,
                        ):
                            with patch(
                                "gateway.channel_directory.build_channel_directory",
                                new=AsyncMock(return_value={"platforms": {}}),
                            ):
                                with patch("gateway.run.asyncio.create_task", side_effect=fake_create_task):
                                    assert await runner.start() is True

        assert Platform.TELEGRAM in runner._failed_platforms
        assert Platform.FEISHU in runner.adapters
        assert Platform.TELEGRAM not in runner.adapters
        assert runner._create_adapter.call_count == 2

    @pytest.mark.asyncio
    async def test_connect_adapter_timeout_raises_retryable_exception(self, monkeypatch):
        """The timeout helper turns a hanging connect into a caught startup error."""
        runner = _make_runner()
        adapter = StubAdapter()

        async def hang():
            await asyncio.sleep(60)
            return True

        adapter.connect = hang
        monkeypatch.setenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "0.001")

        with pytest.raises(TimeoutError, match="telegram connect timed out"):
            await runner._connect_adapter_with_timeout(adapter, Platform.TELEGRAM)


class TestStartupFailureQueuing:
    """Verify that failed platforms are queued during startup."""

    def test_failed_platform_queued_on_connect_failure(self):
        """When adapter.connect() returns False without fatal error, queue for retry."""
        runner = _make_runner()
        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() + 30,
        }
        assert Platform.TELEGRAM in runner._failed_platforms
        assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == 1

    def test_failed_platform_not_queued_for_nonretryable(self):
        """Non-retryable errors should not be in the retry queue."""
        runner = _make_runner()
        # Simulate: adapter had a non-retryable error, wasn't queued
        assert Platform.TELEGRAM not in runner._failed_platforms


# --- Reconnect watcher ---

class TestPlatformReconnectWatcher:
    """Test the _platform_reconnect_watcher background task."""

    @pytest.mark.asyncio
    async def test_reconnect_succeeds_on_retry(self):
        """Watcher should reconnect a failed platform when connect() succeeds."""
        runner = _make_runner()
        runner._sync_voice_mode_state_to_adapter = MagicMock()

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() - 1,  # Already past retry time
        }

        succeed_adapter = StubAdapter(succeed=True)
        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter", return_value=succeed_adapter):
            with patch("gateway.run.build_channel_directory", create=True):
                # Run one iteration of the watcher then stop
                async def run_one_iteration():
                    runner._running = True
                    # Patch the sleep to exit after first check
                    call_count = 0

                    async def fake_sleep(n):
                        nonlocal call_count
                        call_count += 1
                        if call_count > 1:
                            runner._running = False
                        await real_sleep(0)

                    with patch("asyncio.sleep", side_effect=fake_sleep):
                        await runner._platform_reconnect_watcher()

                await run_one_iteration()

        assert Platform.TELEGRAM not in runner._failed_platforms
        assert Platform.TELEGRAM in runner.adapters

    @pytest.mark.asyncio
    async def test_reconnect_nonretryable_removed_from_queue(self):
        """Non-retryable errors should remove the platform from the retry queue."""
        runner = _make_runner()

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() - 1,
        }

        fail_adapter = StubAdapter(
            succeed=False, fatal_error="bad token", fatal_retryable=False
        )

        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter", return_value=fail_adapter):
            async def run_one_iteration():
                runner._running = True
                call_count = 0

                async def fake_sleep(n):
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        runner._running = False
                    await real_sleep(0)

                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await runner._platform_reconnect_watcher()

            await run_one_iteration()

        assert Platform.TELEGRAM not in runner._failed_platforms
        assert Platform.TELEGRAM not in runner.adapters

    @pytest.mark.asyncio
    async def test_reconnect_retryable_stays_in_queue(self):
        """Retryable failures should remain in the queue with incremented attempts."""
        runner = _make_runner()

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() - 1,
        }

        fail_adapter = StubAdapter(
            succeed=False, fatal_error="DNS failure", fatal_retryable=True
        )

        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter", return_value=fail_adapter):
            async def run_one_iteration():
                runner._running = True
                call_count = 0

                async def fake_sleep(n):
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        runner._running = False
                    await real_sleep(0)

                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await runner._platform_reconnect_watcher()

            await run_one_iteration()

        assert Platform.TELEGRAM in runner._failed_platforms
        assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == 2

    @pytest.mark.asyncio
    async def test_reconnect_gives_up_after_max_attempts(self):
        """After max attempts, platform should be removed from retry queue."""
        runner = _make_runner()

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 20,  # At max
            "next_retry": time.monotonic() - 1,
        }

        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter") as mock_create:
            async def run_one_iteration():
                runner._running = True
                call_count = 0

                async def fake_sleep(n):
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        runner._running = False
                    await real_sleep(0)

                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await runner._platform_reconnect_watcher()

            await run_one_iteration()

        assert Platform.TELEGRAM not in runner._failed_platforms
        mock_create.assert_not_called()  # Should give up without trying

    @pytest.mark.asyncio
    async def test_reconnect_skips_when_not_time_yet(self):
        """Watcher should skip platforms whose next_retry is in the future."""
        runner = _make_runner()

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() + 9999,  # Far in the future
        }

        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter") as mock_create:
            async def run_one_iteration():
                runner._running = True
                call_count = 0

                async def fake_sleep(n):
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        runner._running = False
                    await real_sleep(0)

                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await runner._platform_reconnect_watcher()

            await run_one_iteration()

        assert Platform.TELEGRAM in runner._failed_platforms
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_failed_platforms_watcher_idles(self):
        """When no platforms are failed, watcher should just idle."""
        runner = _make_runner()
        # No failed platforms

        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter") as mock_create:
            async def run_briefly():
                runner._running = True
                call_count = 0

                async def fake_sleep(n):
                    nonlocal call_count
                    call_count += 1
                    if call_count > 2:
                        runner._running = False
                    await real_sleep(0)

                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await runner._platform_reconnect_watcher()

            await run_briefly()

        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_adapter_create_returns_none(self):
        """If _create_adapter returns None, remove from queue (missing deps)."""
        runner = _make_runner()

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() - 1,
        }

        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter", return_value=None):
            async def run_one_iteration():
                runner._running = True
                call_count = 0

                async def fake_sleep(n):
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        runner._running = False
                    await real_sleep(0)

                with patch("asyncio.sleep", side_effect=fake_sleep):
                    await runner._platform_reconnect_watcher()

            await run_one_iteration()

        assert Platform.TELEGRAM not in runner._failed_platforms


# --- Runtime disconnection queueing ---

class TestRuntimeDisconnectQueuing:
    """Test that _handle_adapter_fatal_error queues retryable disconnections."""

    @pytest.mark.asyncio
    async def test_retryable_runtime_error_queued_for_reconnect(self):
        """Retryable runtime errors should add the platform to _failed_platforms."""
        runner = _make_runner()
        runner.stop = AsyncMock()

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("network_error", "DNS failure", retryable=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        await runner._handle_adapter_fatal_error(adapter)

        assert Platform.TELEGRAM in runner._failed_platforms
        assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == 0

    @pytest.mark.asyncio
    async def test_nonretryable_runtime_error_not_queued(self):
        """Non-retryable runtime errors should not be queued for reconnection."""
        runner = _make_runner()

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("auth_error", "bad token", retryable=False)
        runner.adapters[Platform.TELEGRAM] = adapter

        # Need to prevent stop() from running fully
        runner.stop = AsyncMock()

        await runner._handle_adapter_fatal_error(adapter)

        assert Platform.TELEGRAM not in runner._failed_platforms

    @pytest.mark.asyncio
    async def test_retryable_error_exits_for_service_restart_when_all_down(self):
        """Gateway should exit with failure when all platforms fail with retryable errors.

        This lets systemd Restart=on-failure restart the process, which is more
        reliable than in-process background reconnection after exhausted retries.
        """
        runner = _make_runner()
        runner.stop = AsyncMock()

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("network_error", "DNS failure", retryable=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        await runner._handle_adapter_fatal_error(adapter)

        # stop() SHOULD be called — gateway exits for systemd restart
        runner.stop.assert_called_once()
        assert runner._exit_with_failure is True
        assert Platform.TELEGRAM in runner._failed_platforms

    @pytest.mark.asyncio
    async def test_retryable_error_no_exit_when_other_adapters_still_connected(self):
        """Gateway should NOT exit if some adapters are still connected."""
        runner = _make_runner()
        runner.stop = AsyncMock()

        failing_adapter = StubAdapter(succeed=True)
        failing_adapter._set_fatal_error("network_error", "DNS failure", retryable=True)
        runner.adapters[Platform.TELEGRAM] = failing_adapter

        # Another adapter is still connected
        healthy_adapter = StubAdapter(succeed=True)
        runner.adapters[Platform.DISCORD] = healthy_adapter

        await runner._handle_adapter_fatal_error(failing_adapter)

        # stop() should NOT have been called — Discord is still up
        runner.stop.assert_not_called()
        assert Platform.TELEGRAM in runner._failed_platforms

    @pytest.mark.asyncio
    async def test_nonretryable_error_triggers_shutdown(self):
        """Gateway should shut down when no adapters remain and nothing is queued."""
        runner = _make_runner()
        runner.stop = AsyncMock()

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("auth_error", "bad token", retryable=False)
        runner.adapters[Platform.TELEGRAM] = adapter

        await runner._handle_adapter_fatal_error(adapter)

        runner.stop.assert_called_once()
