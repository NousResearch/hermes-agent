"""Tests for the gateway platform reconnection watcher."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
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
        # Records the is_reconnect value of every connect() call so tests can
        # assert that the watcher distinguishes reconnect from cold boot (#46621).
        self.connect_calls: list[bool] = []

    async def connect(self, *, is_reconnect: bool = False):
        self.connect_calls.append(is_reconnect)
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

        async def hang(*, is_reconnect: bool = False):
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
    async def test_reconnect_passes_is_reconnect_true(self):
        """The watcher must connect with is_reconnect=True so adapters preserve
        their server-side update queue across an outage (#46621). Without this,
        bootstrap start_polling(drop_pending_updates=True) silently dropped every
        message queued while the bot was offline."""
        runner = _make_runner()
        runner._sync_voice_mode_state_to_adapter = MagicMock()

        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 1,
            "next_retry": time.monotonic() - 1,
        }

        succeed_adapter = StubAdapter(succeed=True)
        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter", return_value=succeed_adapter):
            with patch("gateway.run.build_channel_directory", create=True):
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

        assert succeed_adapter.connect_calls == [True], (
            f"watcher must pass is_reconnect=True; got {succeed_adapter.connect_calls!r}"
        )
        assert Platform.TELEGRAM in runner.adapters

    @pytest.mark.asyncio
    async def test_cold_connect_defaults_to_is_reconnect_false(self):
        """The cold-start connect path (_connect_adapter_with_timeout with no
        is_reconnect arg) must default to False so a first boot still drops any
        stale queue (#46621)."""
        runner = _make_runner()
        adapter = StubAdapter(succeed=True)

        success = await runner._connect_adapter_with_timeout(adapter, Platform.TELEGRAM)

        assert success is True
        assert adapter.connect_calls == [False], (
            f"cold-start must default to is_reconnect=False; got {adapter.connect_calls!r}"
        )

    @pytest.mark.asyncio
    async def test_reconnect_retries_resume_pending_for_platform(self):
        """A successful reconnect retries the startup auto-resume scoped to
        that platform.

        Regression: a platform offline at gateway startup had its
        restart-interrupted sessions skipped by the one-shot startup pass and
        never rescheduled, so the documented auto-resume silently dropped
        until the user sent a fresh message. The watcher now re-runs the
        platform-scoped auto-resume on reconnect.
        """
        runner = _make_runner()
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._schedule_resume_pending_sessions = MagicMock(return_value=1)

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 1,
            "next_retry": time.monotonic() - 1,
        }

        succeed_adapter = StubAdapter(succeed=True)
        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter", return_value=succeed_adapter):
            with patch("gateway.run.build_channel_directory", create=True):
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

        assert Platform.TELEGRAM in runner.adapters
        runner._schedule_resume_pending_sessions.assert_called_once_with(
            platform=Platform.TELEGRAM
        )

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
    async def test_reconnect_never_auto_pauses_retryable_failures(self):
        """Retryable failures (network/DNS) must keep retrying indefinitely —
        the watcher must NOT auto-pause them. Auto-pausing a transiently-failed
        platform left bots silently dead after a DNS blip (#35284). The pause
        circuit breaker remains available for manual /platform pause only.
        """
        runner = _make_runner()

        platform_config = PlatformConfig(enabled=True, token="test")
        # Far past the old circuit-breaker threshold (10): even after many
        # consecutive retryable failures the platform must stay unpaused.
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 25,
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

        # Platform stays in queue and keeps retrying — never auto-paused.
        assert Platform.TELEGRAM in runner._failed_platforms
        info = runner._failed_platforms[Platform.TELEGRAM]
        assert info.get("paused") is not True
        assert "pause_reason" not in info
        assert info["attempts"] == 26
        # next_retry is pushed out by the backoff (capped at 300s), not inf.
        assert info["next_retry"] != float("inf")
        assert info["next_retry"] > time.monotonic()

    @pytest.mark.asyncio
    async def test_reconnect_skips_paused_platforms(self):
        """A paused platform should not be retried by the watcher tick."""
        runner = _make_runner()

        platform_config = PlatformConfig(enabled=True, token="test")
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": platform_config,
            "attempts": 10,
            "next_retry": time.monotonic() - 1,  # would normally retry now
            "paused": True,
            "pause_reason": "paused via /platform pause",
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

        # Paused platform stays queued and was never touched
        assert Platform.TELEGRAM in runner._failed_platforms
        assert runner._failed_platforms[Platform.TELEGRAM]["paused"] is True
        mock_create.assert_not_called()

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
    async def test_retryable_runtime_error_reconnects_immediately(self):
        """Runtime failures should not wait for the startup retry delay."""
        runner = _make_runner()
        runner.stop = AsyncMock()

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("sidecar_crashed", "bridge exited", retryable=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        before = time.monotonic()
        await runner._handle_adapter_fatal_error(adapter)
        after = time.monotonic()

        info = runner._failed_platforms[Platform.TELEGRAM]
        assert info["attempts"] == 0
        assert before <= info["next_retry"] <= after

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
    async def test_retryable_error_keeps_gateway_alive_when_all_down(self):
        """When all adapters fail at runtime with retryable errors, the
        gateway should stay alive and let the reconnect watcher recover them
        in the background.  (Previously this exited-with-failure to trigger
        a systemd restart — that converted transient outages into infinite
        restart loops and killed in-process state.)
        """
        runner = _make_runner()
        runner.stop = AsyncMock()

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("network_error", "DNS failure", retryable=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        await runner._handle_adapter_fatal_error(adapter)

        # stop() should NOT be called — gateway stays alive for the watcher
        runner.stop.assert_not_called()
        assert runner._exit_with_failure is False
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


# --- Pause / resume circuit breaker ---


class TestPauseResume:
    """Test the per-platform pause/resume helpers and slash command."""

    def test_pause_marks_platform_paused(self):
        runner = _make_runner()
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="t"),
            "attempts": 3,
            "next_retry": time.monotonic() + 30,
        }
        runner._pause_failed_platform(Platform.TELEGRAM, reason="manual")
        info = runner._failed_platforms[Platform.TELEGRAM]
        assert info["paused"] is True
        assert info["pause_reason"] == "manual"
        assert info["next_retry"] == float("inf")

    def test_pause_is_idempotent(self):
        runner = _make_runner()
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="t"),
            "attempts": 3,
            "next_retry": time.monotonic() + 30,
            "paused": True,
            "pause_reason": "first reason",
        }
        runner._pause_failed_platform(Platform.TELEGRAM, reason="second reason")
        # Reason should not be overwritten on a second pause call.
        assert (
            runner._failed_platforms[Platform.TELEGRAM]["pause_reason"]
            == "first reason"
        )

    def test_pause_no_op_when_platform_not_queued(self):
        runner = _make_runner()
        # No exception even when the platform isn't in _failed_platforms.
        runner._pause_failed_platform(Platform.TELEGRAM, reason="x")
        assert Platform.TELEGRAM not in runner._failed_platforms

    def test_resume_clears_paused_and_resets_attempts(self):
        runner = _make_runner()
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="t"),
            "attempts": 10,
            "next_retry": float("inf"),
            "paused": True,
            "pause_reason": "auto-paused",
        }
        assert runner._resume_paused_platform(Platform.TELEGRAM) is True
        info = runner._failed_platforms[Platform.TELEGRAM]
        assert info["paused"] is False
        assert info["attempts"] == 0
        assert info["next_retry"] != float("inf")
        assert "pause_reason" not in info

    def test_resume_returns_false_when_not_paused(self):
        runner = _make_runner()
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="t"),
            "attempts": 1,
            "next_retry": time.monotonic() + 30,
        }
        assert runner._resume_paused_platform(Platform.TELEGRAM) is False

    def test_resume_returns_false_when_not_queued(self):
        runner = _make_runner()
        assert runner._resume_paused_platform(Platform.TELEGRAM) is False


class TestPlatformSlashCommand:
    """Test the /platform list|pause|resume slash command handler."""

    def _make_event(self, content: str):
        ev = MagicMock()
        ev.content = content
        return ev

    @pytest.mark.asyncio
    async def test_list_shows_connected_and_paused(self):
        runner = _make_runner()
        runner.adapters[Platform.DISCORD] = StubAdapter(platform=Platform.DISCORD)
        runner._failed_platforms[Platform.WHATSAPP] = {
            "config": PlatformConfig(enabled=True, token="t"),
            "attempts": 10,
            "next_retry": float("inf"),
            "paused": True,
            "pause_reason": "not paired",
        }
        out = await runner._handle_platform_command(self._make_event("/platform list"))
        assert "discord" in out
        assert "whatsapp" in out
        assert "PAUSED" in out
        assert "not paired" in out

    @pytest.mark.asyncio
    async def test_pause_command_pauses_queued_platform(self):
        runner = _make_runner()
        runner._failed_platforms[Platform.WHATSAPP] = {
            "config": PlatformConfig(enabled=True, token="t"),
            "attempts": 2,
            "next_retry": time.monotonic() + 30,
        }
        out = await runner._handle_platform_command(
            self._make_event("/platform pause whatsapp")
        )
        assert "paused" in out.lower()
        assert runner._failed_platforms[Platform.WHATSAPP]["paused"] is True

    @pytest.mark.asyncio
    async def test_pause_rejects_unqueued_platform(self):
        runner = _make_runner()
        out = await runner._handle_platform_command(
            self._make_event("/platform pause whatsapp")
        )
        assert "not in the retry queue" in out

    @pytest.mark.asyncio
    async def test_resume_command_resumes_paused_platform(self):
        runner = _make_runner()
        runner._failed_platforms[Platform.WHATSAPP] = {
            "config": PlatformConfig(enabled=True, token="t"),
            "attempts": 10,
            "next_retry": float("inf"),
            "paused": True,
            "pause_reason": "x",
        }
        out = await runner._handle_platform_command(
            self._make_event("/platform resume whatsapp")
        )
        assert "resumed" in out.lower()
        assert runner._failed_platforms[Platform.WHATSAPP]["paused"] is False

    @pytest.mark.asyncio
    async def test_unknown_platform_name(self):
        runner = _make_runner()
        out = await runner._handle_platform_command(
            self._make_event("/platform pause notarealplatform")
        )
        assert "Unknown platform" in out

    @pytest.mark.asyncio
    async def test_bare_platform_shows_usage_with_list(self):
        # An empty /platform call defaults to "list".
        runner = _make_runner()
        out = await runner._handle_platform_command(self._make_event("/platform"))
        assert "Gateway platforms" in out


# --- Supervised task wrapper (_spawn_supervised) ---

class TestSpawnSupervised:
    """Verify the task-level supervision wrapper around watcher launches."""

    @pytest.mark.asyncio
    async def test_clean_synchronous_return_is_not_respawned(self):
        # A supervised coro that returns immediately (clean exit) must be
        # invoked EXACTLY ONCE — a clean return means deliberate shutdown or a
        # gated no-op watcher; respawning it would busy-spin the event loop.
        runner = _make_runner()
        calls = {"n": 0}

        async def _coro():
            calls["n"] += 1
            return

        runner._spawn_supervised(lambda: _coro(), "clean_watcher")

        # Drive the loop so the done-callback fires; if it (wrongly) respawned,
        # the count would keep climbing across these ticks.
        for _ in range(50):
            await asyncio.sleep(0)

        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_exception_restart_bounded_by_ceiling(self, monkeypatch):
        # A coro that always raises is restarted with backoff, but the restart
        # chain is capped: initial launch + _MAX_SUPERVISED_RESTARTS respawns.
        runner = _make_runner()
        calls = {"n": 0}

        # Collapse the backoff sleeps to a single loop-yield so the restart
        # chain converges fast. Bind the real sleep BEFORE patching so the
        # replacement still yields control (and doesn't recurse into itself).
        real_sleep = asyncio.sleep

        async def _instant_sleep(_delay):
            await real_sleep(0)

        monkeypatch.setattr("gateway.run.asyncio.sleep", _instant_sleep)

        async def _coro():
            calls["n"] += 1
            raise RuntimeError("boom")

        runner._spawn_supervised(lambda: _coro(), "always_raises")

        expected = runner._MAX_SUPERVISED_RESTARTS + 1
        for _ in range(500):
            await real_sleep(0)
            if calls["n"] >= expected:
                break
        # A few extra ticks to prove the chain has stopped (no over-restart).
        for _ in range(20):
            await real_sleep(0)

        assert calls["n"] == expected

    @pytest.mark.asyncio
    async def test_healthy_run_then_crash_resets_restart_counter(self, monkeypatch):
        # A watcher that runs HEALTHILY (>= _SUPERVISED_HEALTHY_SECS) before
        # each crash must NOT be abandoned at the ceiling: every healthy run
        # resets the consecutive-failure counter, so the daemon keeps
        # restarting it well past _MAX_SUPERVISED_RESTARTS. This is the
        # long-lived-launchd-daemon guarantee — a watcher that crashes a
        # handful of times over days is never permanently dropped.
        runner = _make_runner()

        # Treat every run as "healthy": with the floor at 0s, any positive
        # real elapsed (ran_for >= 0.0) counts as a fresh, isolated failure,
        # so the effective attempt resets to 0 on each crash.
        monkeypatch.setattr(runner, "_SUPERVISED_HEALTHY_SECS", 0.0)

        real_sleep = asyncio.sleep

        async def _instant_sleep(_delay):
            await real_sleep(0)

        monkeypatch.setattr("gateway.run.asyncio.sleep", _instant_sleep)

        # Crash more times than the cumulative cap would ever allow, then
        # return cleanly to terminate the chain.
        crash_budget = runner._MAX_SUPERVISED_RESTARTS + 3
        calls = {"n": 0}

        async def _coro():
            calls["n"] += 1
            if calls["n"] <= crash_budget:
                raise RuntimeError("boom")
            return

        runner._spawn_supervised(lambda: _coro(), "healthy_then_crash")

        target = crash_budget + 1  # crash_budget failures + one final clean run
        for _ in range(2000):
            await real_sleep(0)
            if calls["n"] >= target:
                break
        for _ in range(20):
            await real_sleep(0)

        # Under the OLD cumulative cap this would have stopped at
        # _MAX_SUPERVISED_RESTARTS + 1; the reset lets it run to completion.
        assert calls["n"] == target
        assert calls["n"] > runner._MAX_SUPERVISED_RESTARTS + 1


class TestFatalHandoffCancellationProof:
    """The fatal-error handoff must survive cancellation of the notifying
    task, and a retryable platform must never be silently stranded."""

    @pytest.mark.asyncio
    async def test_caller_cancellation_does_not_strand_platform(self):
        """The fatal notification arrives on the failing adapter's own
        polling task, and adapter.disconnect() inside the handler can cancel
        that task mid-teardown. The platform must still reach the reconnect
        queue (previously the CancelledError killed the handler between the
        fatal log and the queue, stranding the platform until a manual
        restart)."""
        runner = _make_runner()
        runner.stop = AsyncMock()

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("network_error", "DNS failure", retryable=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        release = asyncio.Event()

        async def slow_disconnect():
            await release.wait()

        adapter.disconnect = slow_disconnect  # hold the handler mid-teardown

        caller = asyncio.create_task(runner._handle_adapter_fatal_error(adapter))
        for _ in range(5):
            await asyncio.sleep(0)  # let the handler reach the disconnect await
        caller.cancel()  # what disconnect() does to the notifying task
        with pytest.raises(asyncio.CancelledError):
            await caller
        release.set()  # teardown completes after the caller has died

        for _ in range(200):
            if Platform.TELEGRAM in runner._failed_platforms:
                break
            await asyncio.sleep(0.01)
        assert Platform.TELEGRAM in runner._failed_platforms

    @pytest.mark.asyncio
    async def test_stranded_retryable_platform_exits_for_supervisor_restart(self):
        """If a retryable platform ends up neither reconnected nor queued
        (e.g. its config entry is gone so queueing is skipped), the gateway
        must exit with failure so launchd/systemd KeepAlive restarts it,
        instead of running indefinitely with a dead platform while healthy
        peers mask the loss (#68693)."""
        runner = _make_runner()

        async def _stop():
            runner._shutdown_event.set()

        runner.stop = AsyncMock(side_effect=_stop)
        runner.config = GatewayConfig(platforms={})  # queueing impossible

        adapter = StubAdapter(succeed=True)
        adapter._set_fatal_error("network_error", "DNS failure", retryable=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        # A healthy peer keeps self.adapters non-empty, so the existing
        # "no platforms remain" shutdown branches do not fire.
        runner.adapters[Platform.FEISHU] = StubAdapter(platform=Platform.FEISHU)

        await runner._handle_adapter_fatal_error(adapter)

        assert runner._exit_with_failure is True
        assert runner.stop.await_count == 1
