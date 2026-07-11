"""Runner integration tests for plugin gateway-service lifecycle.

Exercises ``GatewayRunner._start_plugin_gateway_services`` directly:
public API retrieval, reconnect-aware adapter resolver (primary +
secondary profile), exactly-once across reconnect, and sync-callback
rejection at registration time. No network; coordination via asyncio
Events/wait_for.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from gateway.plugin_services import (
    GatewayServiceContext,
    GatewayServiceManager,
    GatewayServiceRegistration,
)


def _make_registration(name: str = "svc", *, service=None) -> GatewayServiceRegistration:
    async def _default(ctx: GatewayServiceContext) -> None:
        pass

    return GatewayServiceRegistration(
        name=name,
        service=service or _default,
        plugin_name="test_plugin",
        plugin_key="test_plugin",
        source="user",
    )


class TestGatewayRunnerIntegration:
    """Runner helper: public API, reconnect-aware resolver, exactly-once."""

    @pytest.mark.asyncio
    async def test_runner_uses_public_api_string_keys_once(self, monkeypatch):
        from gateway import run as gateway_run
        from gateway.config import Platform

        started, captured, count = asyncio.Event(), {}, 0

        async def svc(ctx):
            nonlocal count
            count += 1
            captured.update(dict(ctx.adapters))
            started.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)
        calls = []

        class FakePM:
            def get_gateway_services(self):
                calls.append(1)
                return (reg,)

        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: FakePM())
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.TELEGRAM: MagicMock(name="tg")}
        runner._profile_adapters = {}
        runner._plugin_service_manager = GatewayServiceManager(
            runner._plugin_service_adapters
        )
        await runner._start_plugin_gateway_services()
        await runner._start_plugin_gateway_services()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert count == 1 and set(captured) == {"telegram"} and calls
        await runner._plugin_service_manager.stop_services()

    @pytest.mark.asyncio
    async def test_reconnect_launches_if_not_yet_started(self, monkeypatch):
        from gateway import run as gateway_run
        from gateway.config import Platform

        started, count = asyncio.Event(), 0

        async def svc(ctx):
            nonlocal count
            count += 1
            started.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)

        class FakePM:
            def get_gateway_services(self):
                return (reg,)

        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: FakePM())
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {}
        runner._profile_adapters = {}
        runner._plugin_service_manager = GatewayServiceManager(
            runner._plugin_service_adapters
        )

        await runner._start_plugin_gateway_services()
        assert count == 0

        runner.adapters = {Platform.TELEGRAM: MagicMock(name="tg")}
        await runner._start_plugin_gateway_services()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert count == 1
        await runner._plugin_service_manager.stop_services()

    @pytest.mark.asyncio
    async def test_second_reconnect_does_not_duplicate(self, monkeypatch):
        from gateway import run as gateway_run
        from gateway.config import Platform

        count = 0
        started = asyncio.Event()

        async def svc(ctx):
            nonlocal count
            count += 1
            started.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)

        class FakePM:
            def get_gateway_services(self):
                return (reg,)

        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: FakePM())
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.TELEGRAM: MagicMock(name="tg")}
        runner._profile_adapters = {}
        runner._plugin_service_manager = GatewayServiceManager(
            runner._plugin_service_adapters
        )

        await runner._start_plugin_gateway_services()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert count == 1
        await runner._start_plugin_gateway_services()
        assert count == 1
        await runner._plugin_service_manager.stop_services()

    @pytest.mark.asyncio
    async def test_secondary_profile_adapters_in_snapshot(self, monkeypatch):
        from gateway import run as gateway_run
        from gateway.config import Platform

        captured, started = {}, asyncio.Event()

        async def svc(ctx):
            captured.update(dict(ctx.adapters))
            started.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)

        class FakePM:
            def get_gateway_services(self):
                return (reg,)

        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: FakePM())
        tg = MagicMock(name="tg")
        sec = MagicMock(name="sec_discord")
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.TELEGRAM: tg}
        runner._profile_adapters = {"work": {Platform.DISCORD: sec}}
        runner._plugin_service_manager = GatewayServiceManager(
            runner._plugin_service_adapters
        )

        await runner._start_plugin_gateway_services()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert captured.get("telegram") is tg
        assert captured.get("work/discord") is sec
        await runner._plugin_service_manager.stop_services()


class TestGatewayRunnerReconnectAwareResolver:
    """Reconnect-driven adapter changes become visible to already-running
    services through the runner-provided resolver, without launching a
    duplicate service."""

    @pytest.mark.asyncio
    async def test_running_service_sees_reconnect_change_and_stays_once(
        self, monkeypatch
    ):
        """A currently-running service observes a reconnect-driven adapter
        addition on a fresh .adapters access WHILE STILL RUNNING (before
        any shutdown signal), invocation count stays at one, and the
        task set does not change across reconnect."""
        from gateway import run as gateway_run
        from gateway.config import Platform

        started = asyncio.Event()
        read_now = asyncio.Event()
        read_done = asyncio.Event()
        observed = []
        invocations = [0]

        async def svc(ctx):
            invocations[0] += 1
            started.set()
            observed.append(set(ctx.adapters.keys()))
            await read_now.wait()
            observed.append(set(ctx.adapters.keys()))
            read_done.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise

        reg = _make_registration(service=svc)

        class FakePM:
            def get_gateway_services(self):
                return (reg,)

        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: FakePM())
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.TELEGRAM: MagicMock(name="tg")}
        runner._profile_adapters = {}
        runner._plugin_service_manager = GatewayServiceManager(
            runner._plugin_service_adapters
        )

        await runner._start_plugin_gateway_services()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert observed == [{"telegram"}]
        assert invocations[0] == 1
        initial_tasks = list(runner._plugin_service_manager.tasks)

        # Simulate reconnect: a new primary adapter appears, then the
        # reconnect path re-invokes the start helper.
        runner.adapters[Platform.DISCORD] = MagicMock(name="dc")
        await runner._start_plugin_gateway_services()

        # The running service observes the new adapter before shutdown.
        read_now.set()
        await asyncio.wait_for(read_done.wait(), timeout=2)
        assert observed[-1] == {"telegram", "discord"}
        assert invocations[0] == 1
        assert runner._plugin_service_manager.tasks == initial_tasks

        await runner._plugin_service_manager.stop_services()

    @pytest.mark.asyncio
    async def test_reconnect_does_not_launch_duplicate(self, monkeypatch):
        from gateway import run as gateway_run
        from gateway.config import Platform

        count = 0
        started = asyncio.Event()

        async def svc(ctx):
            nonlocal count
            count += 1
            started.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)

        class FakePM:
            def get_gateway_services(self):
                return (reg,)

        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: FakePM())
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.adapters = {Platform.TELEGRAM: MagicMock(name="tg")}
        runner._profile_adapters = {}
        runner._plugin_service_manager = GatewayServiceManager(
            runner._plugin_service_adapters
        )

        await runner._start_plugin_gateway_services()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert count == 1
        initial_tasks = runner._plugin_service_manager.tasks

        # Reconnect: adapter set changes; helper is called again.
        runner.adapters[Platform.DISCORD] = MagicMock(name="dc")
        runner._profile_adapters["work"] = {Platform.SLACK: MagicMock(name="slk")}
        await runner._start_plugin_gateway_services()

        # Same task set — no duplicate launch.
        assert count == 1
        assert runner._plugin_service_manager.tasks == initial_tasks
        await runner._plugin_service_manager.stop_services()


class TestNoManagerStopCompatibility:
    """Borrowed/fake runners without ``_plugin_service_manager`` must
    survive ``GatewayRunner.stop()`` without ``AttributeError``."""

    @pytest.mark.asyncio
    async def test_stop_safe_without_manager(self):
        """A bare runner created via ``object.__new__`` that inherits the
        class-level ``_plugin_service_manager = None`` default reaches the
        stop line and the ``getattr`` fallback prevents ``AttributeError``."""
        from gateway import run as gateway_run

        runner = object.__new__(gateway_run.GatewayRunner)
        runner._running = False
        runner._draining = False
        runner._restart_requested = False
        runner._restart_detached = False
        runner._restart_via_service = False
        runner._stop_task = None
        runner._exit_cleanly = False
        runner._exit_with_failure = False
        runner._exit_reason = None
        runner._exit_code = None
        runner._restart_drain_timeout = 0.01
        runner._running_agents = {}
        runner._running_agents_ts = {}
        runner.adapters = {}
        runner._background_tasks = set()
        runner._failed_platforms = {}
        runner._shutdown_event = asyncio.Event()
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._busy_ack_ts = {}

        async def _noop(*args, **kwargs):
            pass

        runner._running_agent_count = lambda: 0
        runner._active_cron_job_count = lambda: 0
        runner._update_runtime_status = lambda *a, **k: None
        runner._notify_active_sessions_of_shutdown = _noop

        async def _drain_noop(timeout):
            return {}, False

        runner._drain_active_agents = _drain_noop
        runner._finalize_shutdown_agents = _noop

        await gateway_run.GatewayRunner.stop(runner)
