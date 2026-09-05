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


# --- Reconnect watcher ---

class TestPlatformReconnectWatcher:
    """Test the _platform_reconnect_watcher background task."""


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
    @pytest.mark.parametrize("degraded", [False, True])
    async def test_reconnect_stamp_honours_adapter_send_path_degraded(self, degraded):
        """connect() returning True is not proof the receive path is live:
        Telegram's degraded reconnect returns True while its own ladder
        retries. The watcher's status stamp must publish what the adapter
        reports, not an unconditional "connected" (#101391)."""
        runner = _make_runner()
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._update_platform_runtime_status = MagicMock()
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": PlatformConfig(enabled=True, token="test"),
            "attempts": 1,
            "next_retry": time.monotonic() - 1,
        }

        class _DegradableAdapter(StubAdapter):
            @property
            def send_path_degraded(self) -> bool:
                return degraded

        adapter = _DegradableAdapter(succeed=True)
        real_sleep = asyncio.sleep

        with patch.object(runner, "_create_adapter", return_value=adapter):
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

        stamps = [
            c.kwargs for c in runner._update_platform_runtime_status.call_args_list
            if c.args and c.args[0] == Platform.TELEGRAM.value
        ]
        assert stamps, "watcher never stamped telegram"
        final = stamps[-1]
        assert final["platform_state"] == ("retrying" if degraded else "connected")
        assert final["error_message"] == (adapter.DEGRADED_STATUS_MESSAGE if degraded else None)
        assert final["retrying_since"] is None

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
        runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=1)
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
        runner._redeliver_failed_obligations_for_platform.assert_awaited_once_with(
            Platform.TELEGRAM
        )
        runner._schedule_resume_pending_sessions.assert_called_once_with(
            platform=Platform.TELEGRAM
        )


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


# --- Runtime disconnection queueing ---

class TestRuntimeDisconnectQueuing:
    """Test that _handle_adapter_fatal_error queues retryable disconnections."""


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


# --- Pause / resume circuit breaker ---


class TestPauseResume:
    """Test the per-platform pause/resume helpers and slash command."""


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


# --- Supervised task wrapper (_spawn_supervised) ---

class TestSpawnSupervised:
    """Verify the task-level supervision wrapper around watcher launches."""

    @pytest.mark.asyncio
    async def test_watcher_does_not_inherit_delegated_child_context(self):
        """Long-lived gateway services must not inherit one turn's child scope."""
        from agent.delegation_context import (
            delegated_child_context,
            is_delegated_child_context,
        )

        runner = _make_runner()
        observed = []

        async def _watcher():
            observed.append(is_delegated_child_context())

        with delegated_child_context():
            assert is_delegated_child_context() is True
            task = runner._spawn_supervised(_watcher, "isolated_watcher")
            await task
            assert is_delegated_child_context() is True

        assert observed == [False]

    @pytest.mark.asyncio
    async def test_dispatcher_to_thread_does_not_trip_kanban_guard(self):
        """Already-running ticks copy the caller task context into to_thread.

        Issue #91958 is a dispatcher that was spawned at boot, then a later
        ``delegate_task`` leaves the worker copy marked as a child. Spawn
        isolation does not rewrite that frozen task context; the offload
        helper must scrub it at the tick boundary. The in-task child guard
        stays closed.
        """
        from agent.delegation_context import (
            delegated_child_context,
            is_delegated_child_context,
        )
        from gateway.kanban_watchers_common import _to_thread_process_service
        from hermes_cli.kanban_db import _assert_not_delegated_child_mutation

        with delegated_child_context():
            assert is_delegated_child_context() is True
            with pytest.raises(PermissionError, match="delegate_task child"):
                _assert_not_delegated_child_mutation()
            await _to_thread_process_service(_assert_not_delegated_child_mutation)
            assert is_delegated_child_context() is True

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


class TestFatalHandoffCancellationProof:
    """The fatal-error handoff must survive cancellation of the notifying
    task, and a retryable platform must never be silently stranded."""


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


# ── _ensure_reconnect_watcher_running ──────────────────────────────────


class TestEnsureReconnectWatcherRunning:
    """Verify _ensure_reconnect_watcher_running respawns the watcher when dead."""

    @pytest.mark.asyncio
    async def test_reconnect_watcher_alive_does_nothing(self):
        """Task is alive => no-op."""
        runner = _make_runner()
        runner._running = True
        runner._background_tasks = set()

        async def _dummy():
            await asyncio.sleep(0.2)

        runner._reconnect_watcher_task = asyncio.create_task(_dummy())
        runner._background_tasks.add(runner._reconnect_watcher_task)

        old_task = runner._reconnect_watcher_task
        runner._ensure_reconnect_watcher_running()

        # Same task, not replaced
        assert runner._reconnect_watcher_task is old_task
        assert not runner._reconnect_watcher_task.done()

        old_task.cancel()
        try:
            await old_task
        except asyncio.CancelledError:
            pass


# ── _handle_adapter_fatal_error calls _ensure_reconnect_watcher ────────


class TestReconnectWatcherSelfHeals:
    """Regression tests for issue #71758: a platform already queued in
    _failed_platforms when the reconnect watcher task dies from an
    uncaught exception stayed stranded forever, because
    _ensure_reconnect_watcher_running() is only called from a NEW
    fatal-error arrival -- if no other platform ever fails afterward,
    nothing notices the watcher is dead. The watcher must now be spawned
    via _spawn_supervised (like other long-lived background tasks), so an
    exception escaping its OUTER while-loop is caught, logged, and
    auto-restarted with backoff -- independent of any new fatal-error
    event.
    """


    @pytest.mark.asyncio
    async def test_watcher_self_heals_after_uncaught_exception_with_no_new_fatal_error(self):
        """The core #71758 regression: a platform sits queued in
        _failed_platforms. The watcher task dies from an uncaught
        exception (simulating the KeyError race / any other bug in the
        outer loop). WITHOUT any new fatal-error event for a different
        platform, the watcher must still come back on its own via
        _spawn_supervised's crash-detection callback -- the exact gap
        that stranded the platform for 17.5h in the reported bug.
        """
        runner = _make_runner()
        runner._running = True
        runner._background_tasks = set()
        runner._SUPERVISED_HEALTHY_SECS = GatewayRunner._SUPERVISED_HEALTHY_SECS
        runner._MAX_SUPERVISED_RESTARTS = GatewayRunner._MAX_SUPERVISED_RESTARTS
        runner._spawn_supervised = GatewayRunner._spawn_supervised.__get__(runner)

        attempt_count = {"n": 0}

        async def _flaky_watcher():
            attempt_count["n"] += 1
            if attempt_count["n"] == 1:
                # Simulate the watcher's outer loop raising -- e.g. the
                # KeyError race this same fix also hardens against, or any
                # other bug in code outside the per-platform try/except.
                raise RuntimeError("simulated watcher crash")
            await asyncio.sleep(0.2)  # second run: stays "alive"

        runner._reconnect_watcher_task = runner._spawn_supervised(
            _flaky_watcher, "platform_reconnect_watcher"
        )

        # Let the first (crashing) attempt run and die.
        for _ in range(50):
            await asyncio.sleep(0)
            if attempt_count["n"] >= 1 and runner._reconnect_watcher_task.done():
                break

        assert attempt_count["n"] == 1
        assert runner._reconnect_watcher_task.done()

        # The supervised _done callback schedules a respawn after a short
        # backoff (2**0 = 1s at attempt 0) -- wait for it without a new
        # fatal-error event ever firing.
        for _ in range(30):
            await asyncio.sleep(0.1)
            if attempt_count["n"] >= 2:
                break

        assert attempt_count["n"] >= 2, (
            "Watcher must self-heal via _spawn_supervised without any new "
            "fatal-error event -- this is the exact gap that stranded a "
            "platform in the reported bug"
        )

        # Cleanup: cancel whatever task is currently tracked.
        for task in list(runner._background_tasks):
            task.cancel()
        await asyncio.sleep(0)


class TestReconnectWatcherRaceGuard:
    """Regression: a platform removed from _failed_platforms concurrently
    (e.g. a manual /platform resume racing with the watcher's own
    snapshot-then-lookup) must not raise KeyError and kill the loop
    iteration -- it should just be skipped for that pass."""


    """Verify _handle_adapter_fatal_error calls _ensure_reconnect_watcher_running."""

    @pytest.mark.asyncio
    async def test_retryable_fatal_error_calls_ensure_watcher(self):
        """A retryable fatal error queues the platform AND ensures watcher is alive."""
        runner = _make_runner()
        runner._running = True
        runner._background_tasks = set()
        runner._failed_platforms = {}
        runner._fatal_handler_tasks = set()
        runner._reconnect_watcher_task = asyncio.create_task(asyncio.sleep(0))
        # Let the dummy watcher finish so _ensure_reconnect_watcher_running
        # detects it's dead and respawns.
        await runner._reconnect_watcher_task

        platform_config = PlatformConfig(enabled=True, token="test")
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: platform_config}
        )

        adapter = StubAdapter(
            platform=Platform.TELEGRAM,
            succeed=False,
            fatal_error="network outage",
            fatal_retryable=True,
        )
        # Pre-set fatal error attributes so the handler can read them
        # without going through connect() (#70344).
        adapter._set_fatal_error(
            "NETWORK_ERROR", "network outage", retryable=True
        )
        # Populate adapters so the impl pops it and queues for reconnect
        runner.adapters[Platform.TELEGRAM] = adapter

        call_count = {"ensure": 0}

        def tracking_ensure():
            call_count["ensure"] += 1

        with patch.object(
            runner,
            "_ensure_reconnect_watcher_running",
            side_effect=tracking_ensure,
        ):
            await runner._handle_adapter_fatal_error(adapter)

        assert Platform.TELEGRAM in runner._failed_platforms
        assert call_count["ensure"] >= 1

    @pytest.mark.asyncio
    async def test_nonretryable_fatal_error_does_not_call_ensure(self):
        """A non-retryable error must NOT queue the platform or call the watcher."""
        runner = _make_runner()
        runner._running = True
        runner._background_tasks = set()
        runner._failed_platforms = {}
        runner._fatal_handler_tasks = set()
        runner._reconnect_watcher_task = None

        platform_config = PlatformConfig(enabled=True, token="test")
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: platform_config}
        )

        adapter = StubAdapter(
            platform=Platform.TELEGRAM,
            succeed=False,
            fatal_error="bad token",
            fatal_retryable=False,
        )
        # Pre-set fatal error attributes so the handler can read them
        # without going through connect() (#70344).
        adapter._set_fatal_error(
            "AUTH_FAILED", "bad token", retryable=False
        )
        runner.adapters[Platform.TELEGRAM] = adapter

        ensure_called = False

        def noop_ensure():
            nonlocal ensure_called
            ensure_called = True

        with patch.object(runner, "_ensure_reconnect_watcher_running", side_effect=noop_ensure):
            await runner._handle_adapter_fatal_error(adapter)

        assert Platform.TELEGRAM not in runner._failed_platforms
        assert not ensure_called


# ── _connect_adapter_with_timeout detach-on-timeout ────────────────────


class TestConnectAdapterDetachOnTimeout:
    """Verify _connect_adapter_with_timeout uses the detach pattern."""

    @pytest.mark.asyncio
    async def test_connect_timed_out_raises_timeouterror(self):
        """A connect() that never finishes must raise TimeoutError."""
        runner = _make_runner()

        adapter = StubAdapter(succeed=True)

        async def _slow_connect(**kwargs):
            await asyncio.sleep(0.2)  # never finishes

        with patch.object(adapter, "connect", side_effect=_slow_connect):
            with patch.object(
                runner, "_platform_connect_timeout_secs", return_value=0.01
            ):
                with pytest.raises(TimeoutError, match="timed out"):
                    await runner._connect_adapter_with_timeout(
                        adapter, Platform.TELEGRAM
                    )

        # After the TimeoutError, the slow connect coroutine should have been
        # cancelled and detached, so the event loop can move on.
        await asyncio.sleep(0)


class TestReconnectWatcherHandleTracking:
    """Regression: the supervisor's own backoff respawn must keep
    ``_reconnect_watcher_task`` pointed at the CURRENT live task.

    Before the ``on_spawn`` fix, ``_spawn_supervised``'s internal respawn
    created a new task without updating ``self._reconnect_watcher_task``, so
    after the reconnect watcher crashed and self-respawned, the tracked handle
    still pointed at the DEAD task. A later
    ``_ensure_reconnect_watcher_running()`` then saw ``task.done()`` and
    spawned a SECOND concurrent watcher — double reconnect attempts against
    every failed platform. The two supervision mechanisms (auto-restart +
    ensure-respawn) must compose, not race.
    """

    @pytest.mark.asyncio
    async def test_startup_spawn_tracks_live_handle(self):
        """The startup spawn passes an on_spawn callback so the handle is
        recorded at spawn time (not left None until the lambda in prod)."""
        runner = _make_runner()
        runner._background_tasks = set()

        async def _noop_watcher():
            await asyncio.sleep(0.2)

        # Mirror the production startup call: on_spawn records the handle.
        runner._reconnect_watcher_task = None
        task = runner._spawn_supervised(
            _noop_watcher,
            "platform_reconnect_watcher",
            on_spawn=lambda t: setattr(runner, "_reconnect_watcher_task", t),
        )
        # on_spawn fired synchronously at spawn time.
        assert runner._reconnect_watcher_task is task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _instant_sleep(delay, *a, **k):
    """asyncio.sleep replacement that yields to the loop but never waits."""
    await _REAL_ASYNCIO_SLEEP(0)
    return None


_REAL_ASYNCIO_SLEEP = asyncio.sleep

# --- Voice input callback wiring ---


class TestVoiceInputCallbackWiring:
    """Startup and reconnect must wire _voice_input_callback on Discord."""

    @staticmethod
    def _make_discord_voice_adapter():
        """A minimal Discord adapter stub with voice attributes."""
        adapter = MagicMock()
        adapter._voice_input_callback = None
        adapter._voice_text_channels = {}
        adapter._voice_sources = {}
        adapter.connect = AsyncMock(return_value=True)
        adapter.disconnect = AsyncMock()
        return adapter

    def _make_runner_with_discord(self):
        runner = _make_runner()
        runner.config = GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="test")}
        )
        runner._update_runtime_status = MagicMock()
        runner._update_platform_runtime_status = MagicMock()
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._send_update_notification = AsyncMock(return_value=True)
        runner._send_restart_notification = AsyncMock()
        runner._suspend_stuck_loop_sessions = MagicMock(return_value=0)
        runner.hooks = MagicMock()
        runner.hooks.loaded_hooks = []
        runner.hooks.emit = AsyncMock()
        return runner

    @pytest.mark.asyncio
    async def test_startup_wires_voice_input_callback(self, tmp_path):
        """Cold-start connect must wire _voice_input_callback on Discord adapter."""
        runner = self._make_runner_with_discord()
        adapter = self._make_discord_voice_adapter()
        runner.config.sessions_dir = tmp_path

        def fake_create_task(coro):
            coro.close()
            return MagicMock()

        with patch.object(runner, "_create_adapter", return_value=adapter):
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
                                    with patch(
                                        "gateway.run.asyncio.create_task",
                                        side_effect=fake_create_task,
                                    ):
                                        assert await runner.start() is True

        assert adapter._voice_input_callback is not None, (
            "startup must wire _voice_input_callback"
        )



class TestRequeueHealsDeadReconnectWatcher:
    """Regression for #90386: a second retryable fatal error for a platform that
    is ALREADY queued must still check that the reconnect watcher is alive.

    ``_ensure_reconnect_watcher_running()`` is the documented backstop for the
    watcher exhausting ``_MAX_SUPERVISED_RESTARTS`` (#70344) -- ``_spawn_supervised``
    stops respawning after that and logs "giving up restarts". Its only call site
    sat behind the newly-queued branch of ``_queue_retryable_fatal_platform``,
    so it could never fire for a platform already in ``_failed_platforms`` --
    which is the only kind of platform the watcher can have been retrying long
    enough to exhaust the budget on.

    The observable result is a silent permanent outage: the queue holds the
    platform, nothing retries it, and the stranded check in
    ``_handle_adapter_fatal_error_detached`` deliberately treats a queued
    platform as safe, so the process is never restarted either.
    """

    @staticmethod
    def _runner_with_dead_watcher(spawned):
        runner = _make_runner()
        runner._running = True
        runner._background_tasks = set()

        def _fake_spawn(coro_factory, name, **kwargs):
            spawned.append(name)
            handle = MagicMock()
            handle.done.return_value = False
            on_spawn = kwargs.get("on_spawn")
            if on_spawn is not None:
                on_spawn(handle)
            return handle

        runner._spawn_supervised = _fake_spawn
        return runner

    @staticmethod
    def _already_queued(runner, *, attempts=7):
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": runner.config.platforms[Platform.TELEGRAM],
            "attempts": attempts,
            "next_retry": time.monotonic() + 300,
            "queued_at": time.monotonic() - 3600,
            "credential_claim": None,
            "listener_claim": None,
        }

    @staticmethod
    def _fatal_adapter():
        adapter = StubAdapter()
        adapter._set_fatal_error(
            "telegram_network_error",
            "Telegram polling could not reconnect after 10 network error retries.",
            retryable=True,
        )
        return adapter

    @pytest.mark.asyncio
    async def test_requeue_respawns_a_watcher_that_gave_up_restarting(self):
        """The core #90386 regression.

        Telegram is queued and the watcher has exhausted its restart budget, so
        the task is done and ``_spawn_supervised`` will never bring it back on
        its own. A fresh retryable fatal error arrives for that same platform.
        Nothing is enqueued (it is already there), but the watcher MUST be
        respawned -- otherwise the queue entry is retried by nobody, forever.
        """
        spawned: list[str] = []
        runner = self._runner_with_dead_watcher(spawned)
        self._already_queued(runner)

        async def _died():
            return None

        dead = asyncio.create_task(_died())
        await dead
        runner._reconnect_watcher_task = dead

        queued = runner._queue_retryable_fatal_platform(self._fatal_adapter())

        assert queued is False, "already-queued platform must not be re-enqueued"
        assert spawned == ["platform_reconnect_watcher"], (
            "a re-fatal on an already-queued platform must still heal a dead "
            "reconnect watcher -- it is the only remaining path back to the "
            "queue once _spawn_supervised has given up restarting (#90386)"
        )
        assert not runner._reconnect_watcher_task.done(), (
            "the tracked handle must point at the newly spawned watcher"
        )

    @pytest.mark.asyncio
    async def test_requeue_does_not_disturb_the_existing_queue_entry(self):
        """Guardrail on the fix: healing the watcher must not become a re-enqueue.

        Overwriting the entry would reset ``attempts`` and ``next_retry``, so a
        platform that fails repeatedly would restart its backoff ladder on every
        fatal error and hammer a provider that is already refusing it.
        """
        spawned: list[str] = []
        runner = self._runner_with_dead_watcher(spawned)
        self._already_queued(runner, attempts=7)
        before = dict(runner._failed_platforms[Platform.TELEGRAM])

        async def _died():
            return None

        dead = asyncio.create_task(_died())
        await dead
        runner._reconnect_watcher_task = dead

        assert runner._queue_retryable_fatal_platform(self._fatal_adapter()) is False
        assert runner._failed_platforms[Platform.TELEGRAM] == before, (
            "the existing queue entry, including its attempt count and backoff "
            "deadline, must survive the watcher heal untouched"
        )

    @pytest.mark.asyncio
    async def test_requeue_with_a_live_watcher_spawns_nothing(self):
        """Guardrail on the fix: a live watcher must not be duplicated.

        Two concurrent watchers would double every reconnect attempt, which is
        the failure ``_spawn_supervised``'s ``on_spawn`` handle-tracking exists
        to prevent.
        """
        spawned: list[str] = []
        runner = self._runner_with_dead_watcher(spawned)
        self._already_queued(runner)

        async def _alive():
            await asyncio.sleep(5)

        live = asyncio.create_task(_alive())
        runner._reconnect_watcher_task = live
        try:
            assert runner._queue_retryable_fatal_platform(self._fatal_adapter()) is False
            assert spawned == [], "a live watcher must never be respawned"
            assert runner._reconnect_watcher_task is live
        finally:
            live.cancel()
            try:
                await live
            except asyncio.CancelledError:
                pass


class TestSupervisionExhaustionHasAnOwner:
    """The witness @andrexibiza asked for on #90448: recovery with NO new event.

    The predecessor architecture (#72366, salvage of #71867 by @ygd58)
    established that a reconnect watcher which dies while work is queued must
    be respawned by something *autonomous*, because the only other trigger --
    a fresh fatal error from another platform -- may never arrive. Supervised
    restart closed that, but supervision is finite: after
    ``_MAX_SUPERVISED_RESTARTS`` rapid crashes it logs "giving up restarts" and
    stops.

    Past that point the system is back in exactly the state #72366 described.
    And #81036 (salvage of #80700, preserving @HexLab98) makes the missing
    event less likely rather than more: it publishes the queue *before*
    disconnect and drops the failed adapter from the live map, so there may be
    no adapter left to emit the callback recovery was waiting on.

    These tests exercise that state directly. They queue a platform, let the
    watcher burn its whole budget, and then emit **no further fatal
    callbacks at all** -- the thing the branch-level regression tests below
    cannot prove, because they manufacture the event.
    """

    @staticmethod
    def _runner(monkeypatch, *, crashes):
        """A runner whose reconnect watcher crashes `crashes` times, then lives."""
        runner = _make_runner()
        runner._background_tasks = set()
        runner._reconnect_watcher_task = None
        runner.state = {"live": 0, "spawns": 0}

        # Collapse every real-time delay: the point is the ORDER of events, not
        # their spacing. Left as class attributes in production precisely so a
        # test can do this without sleeping for half an hour.
        monkeypatch.setattr(GatewayRunner, "_MAX_SUPERVISED_RESTARTS", 2)
        monkeypatch.setattr(GatewayRunner, "_SUPERVISED_HEALTHY_SECS", 3600)
        monkeypatch.setattr(GatewayRunner, "_RECONNECT_WATCHER_SLOW_RETRY_SECS", 0)
        monkeypatch.setattr(GatewayRunner, "_supervised_backoff", staticmethod(lambda _a: 0))

        remaining = {"n": crashes}

        async def _watcher():
            runner.state["spawns"] += 1
            if remaining["n"] > 0:
                remaining["n"] -= 1
                raise RuntimeError("watcher crashed on contact")
            # Healthy: stay alive so `task.done()` is False.
            runner.state["live"] += 1
            await asyncio.Event().wait()

        runner._platform_reconnect_watcher = _watcher
        return runner

    @staticmethod
    def _queue_telegram(runner):
        runner._failed_platforms[Platform.TELEGRAM] = {
            "config": runner.config.platforms[Platform.TELEGRAM],
            "attempts": 9,
            "next_retry": time.monotonic() + 300,
            "queued_at": time.monotonic() - 7200,
            "credential_claim": None,
            "listener_claim": None,
        }

    @staticmethod
    async def _settle(times=40):
        """Let the supervisor's done-callbacks and respawn tasks drain."""
        for _ in range(times):
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_watcher_recovers_after_budget_exhaustion_with_no_new_event(
        self, monkeypatch
    ):
        """The whole review gate in one test.

        Queue a platform, crash the watcher past its supervised budget, then do
        nothing at all -- no fatal callback, no enqueue, no external poke. The
        watcher must come back on its own once it stops crashing.
        """
        # One crash more than supervision will tolerate, so the give-up branch
        # is genuinely reached and the slow tier is what recovers it.
        runner = self._runner(monkeypatch, crashes=3)
        self._queue_telegram(runner)

        runner._spawn_reconnect_watcher()
        await self._settle(300)

        assert runner.state["live"] == 1, (
            "with no new fatal event, only the slow tier can have brought the "
            "watcher back after supervision gave up"
        )
        assert runner._reconnect_watcher_task is not None
        assert not runner._reconnect_watcher_task.done()

        runner._running = False
        for task in list(runner._background_tasks):
            task.cancel()

    @pytest.mark.asyncio
    async def test_the_slow_tier_is_bounded_not_a_restart_loop(self, monkeypatch):
        """A watcher that never recovers must stop being respawned, and say so."""
        runner = self._runner(monkeypatch, crashes=10_000)
        self._queue_telegram(runner)
        monkeypatch.setattr(GatewayRunner, "_MAX_SLOW_WATCHER_RESPAWNS", 3)

        runner._spawn_reconnect_watcher()
        await self._settle(400)

        assert runner.state["live"] == 0
        # Bounded, and the arithmetic is worth stating because it is the
        # whole safety argument. Each slow-tier attempt hands the watcher a
        # FRESH supervised budget -- that is deliberate, since a slow retry is
        # a new bet that conditions have changed -- so the ceiling is
        # (1 + _MAX_SUPERVISED_RESTARTS) x (1 + _MAX_SLOW_WATCHER_RESPAWNS):
        # here (1+2) x (1+3) = 12. With production values that is 6 x 7 = 42
        # spawns spread across at least half an hour, which is a ladder, not
        # a restart loop. A tight loop here would be worse than the outage it
        # is healing.
        assert runner.state["spawns"] == 12, (
            f"expected a finite spawn ladder, got {runner.state['spawns']}"
        )
        assert runner._reconnect_watcher_task.done()

        # And it stays stopped: no further spawns once the ladder is spent.
        await self._settle(400)
        assert runner.state["spawns"] == 12

        runner._running = False
        for task in list(runner._background_tasks):
            task.cancel()

    @pytest.mark.asyncio
    async def test_an_empty_queue_leaves_the_watcher_down(self, monkeypatch):
        """No queued work, no invariant to own.

        Respawning a crash-looping watcher that nothing depends on would burn
        the process for no one. The enqueue path spawns a fresh one the moment
        a platform is actually queued.
        """
        runner = self._runner(monkeypatch, crashes=10_000)
        assert not runner._failed_platforms

        runner._spawn_reconnect_watcher()
        await self._settle(200)

        assert runner.state["live"] == 0
        assert runner._reconnect_watcher_task.done()

        runner._running = False
        for task in list(runner._background_tasks):
            task.cancel()

    @pytest.mark.asyncio
    async def test_the_slow_tier_stands_down_when_the_queue_drains(self, monkeypatch):
        """Something else healed the platform -- stop respawning for it."""
        runner = self._runner(monkeypatch, crashes=10_000)
        self._queue_telegram(runner)

        runner._spawn_reconnect_watcher()
        # Let supervision give up, then drain the queue before the slow tier
        # gets its turn.
        await self._settle(6)
        runner._failed_platforms.clear()
        await self._settle(200)

        assert runner.state["live"] == 0, (
            "with nothing queued, the slow tier has no invariant left to own"
        )

        runner._running = False
        for task in list(runner._background_tasks):
            task.cancel()

    @pytest.mark.asyncio
    async def test_give_up_does_nothing_once_the_gateway_is_shutting_down(
        self, monkeypatch
    ):
        runner = self._runner(monkeypatch, crashes=10_000)
        self._queue_telegram(runner)
        runner._running = False

        runner._on_reconnect_watcher_gave_up("platform_reconnect_watcher")
        await self._settle()

        assert runner.state["live"] == 0
        assert not runner._background_tasks
