"""Tests for plugin gateway-service lifecycle seam.

These tests exercise ``gateway.plugin_services`` — the host-owned
lifecycle manager for plugin-registered async gateway services.
No network access; all coordination uses asyncio Events and wait_for.
"""

import asyncio
import gc
import logging
from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from gateway.plugin_services import (
    GatewayServiceContext,
    GatewayServiceManager,
    GatewayServiceRegistration,
)


def _make_registration(
    name: str = "svc",
    *,
    plugin_name: str = "test_plugin",
    plugin_key: str = "test_plugin",
    source: str = "user",
    service=None,
) -> GatewayServiceRegistration:
    async def _default(ctx: GatewayServiceContext) -> None:
        pass

    return GatewayServiceRegistration(
        name=name,
        service=service or _default,
        plugin_name=plugin_name,
        plugin_key=plugin_key,
        source=source,
    )


def _resolver(adapters):
    """Build a resolver closure over the given adapters mapping.

    Tests pass either a static dict or one they mutate to simulate
    reconnect-driven adapter changes.
    """
    return lambda: adapters


def _attach_drained_signal(tasks):
    """Attach a done-callback that signals when every task has finished.

    Callbacks fire in registration order; registering after the
    manager's own overdue callback means our signal fires after the
    manager has cleaned up its overdue/registration tracking.
    """
    signal = asyncio.Event()
    remaining = [len(tasks)]

    def _cb(_task):
        remaining[0] -= 1
        if remaining[0] <= 0:
            signal.set()

    for task in tasks:
        task.add_done_callback(_cb)
    return signal


class TestGatewayServiceContext:
    """Context exposes a fresh read-only adapter view, not the runner."""

    def test_adapters_read_only_view(self):
        tg = MagicMock(name="tg")
        ctx = GatewayServiceContext(_resolver({"telegram": tg}))
        view = ctx.adapters
        assert isinstance(view, MappingProxyType)
        assert set(view.keys()) == {"telegram"}
        assert view["telegram"] is tg

    def test_held_snapshot_is_decoupled_from_later_mutation(self):
        """A held adapters reference is stable; later resolver changes
        are visible only on a new access, never on the held snapshot."""
        primary = {"telegram": MagicMock()}
        ctx = GatewayServiceContext(_resolver(primary))
        held = ctx.adapters
        primary["discord"] = MagicMock()
        assert "discord" not in held
        assert "discord" in ctx.adapters

    def test_adapters_reflects_resolver_changes_across_accesses(self):
        """Reconnect-aware: each access reflects current resolver state."""
        primary = {"telegram": MagicMock(name="tg")}
        ctx = GatewayServiceContext(_resolver(primary))
        assert set(ctx.adapters.keys()) == {"telegram"}
        primary["discord"] = MagicMock(name="dc")
        assert set(ctx.adapters.keys()) == {"telegram", "discord"}

    def test_context_does_not_expose_runner(self):
        ctx = GatewayServiceContext(_resolver({}))
        assert not hasattr(ctx, "runner")
        assert not hasattr(ctx, "_runner")


class TestGatewayServiceManagerNoOp:
    """Manager is a no-op with no registrations or no adapters."""

    @pytest.mark.asyncio
    async def test_noop_no_registrations(self):
        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([])
        assert not mgr.started and not mgr.tasks

    @pytest.mark.asyncio
    async def test_noop_no_adapters(self):
        mgr = GatewayServiceManager(_resolver({}))
        await mgr.start_services([_make_registration()])
        assert not mgr.started and not mgr.tasks

    @pytest.mark.asyncio
    async def test_stop_noop_with_no_tasks(self):
        mgr = GatewayServiceManager(_resolver({}))
        await mgr.stop_services()

    @pytest.mark.asyncio
    async def test_noop_then_real_start_preserves_exact_once(self):
        """A no-op call (no registrations) must not mark started; a later
        real call launches services exactly once."""
        started = asyncio.Event()
        adapters = {"telegram": MagicMock()}

        async def svc(ctx):
            started.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager(_resolver(adapters))
        await mgr.start_services([])
        assert not mgr.started
        await mgr.start_services([reg])
        assert mgr.started
        await asyncio.wait_for(started.wait(), timeout=2)
        await mgr.start_services([reg])
        assert len(mgr.tasks) == 1
        await mgr.stop_services()


class TestGatewayServiceManagerLifecycle:
    """Core lifecycle contracts: start, exactly-once, retention, stop."""

    @pytest.mark.asyncio
    async def test_service_starts_with_connected_adapters(self):
        started = asyncio.Event()

        async def svc(ctx):
            started.set()

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([reg])
        await asyncio.wait_for(started.wait(), timeout=2)
        await mgr.stop_services()

    @pytest.mark.asyncio
    async def test_context_passed_to_service_has_adapters(self):
        seen = asyncio.Event()
        captured = {}

        async def svc(ctx):
            captured["adapters"] = dict(ctx.adapters)
            seen.set()

        adapter = MagicMock(name="my_adapter")
        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager(_resolver({"telegram": adapter}))
        await mgr.start_services([reg])
        await asyncio.wait_for(seen.wait(), timeout=2)
        await mgr.stop_services()
        assert captured.get("adapters") == {"telegram": adapter}

    @pytest.mark.asyncio
    async def test_exactly_once_repeated_start(self):
        call_count = 0
        entered = asyncio.Event()

        async def svc(ctx):
            nonlocal call_count
            call_count += 1
            entered.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([reg])
        await asyncio.wait_for(entered.wait(), timeout=2)
        first_tasks = mgr.tasks
        await mgr.start_services([reg])
        assert mgr.tasks == first_tasks
        assert call_count == 1
        await mgr.stop_services()

    @pytest.mark.asyncio
    async def test_strong_retention(self):
        entered = asyncio.Event()

        async def svc(ctx):
            entered.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([reg])
        await asyncio.wait_for(entered.wait(), timeout=2)
        task_count = len(mgr.tasks)
        assert task_count == 1
        gc.collect()
        assert len(mgr.tasks) == task_count
        assert all(not t.done() for t in mgr.tasks)
        await mgr.stop_services()

    @pytest.mark.asyncio
    async def test_cancellation_awaits_before_return(self):
        cancel_seen = asyncio.Event()
        service_started = asyncio.Event()

        async def svc(ctx):
            service_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                raise

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([reg])
        await asyncio.wait_for(service_started.wait(), timeout=2)
        await mgr.stop_services()
        assert cancel_seen.is_set()
        assert len(mgr.tasks) == 0
        assert len(mgr.overdue_tasks) == 0
        # Provenance ownership is released for normally cancelled tasks.
        assert not mgr._registrations


class TestGatewayServiceFailureIsolation:
    """Startup and runtime failures are isolated and logged per-service."""

    @pytest.mark.asyncio
    async def test_startup_failure_does_not_block_others(self, caplog):
        healthy_started = asyncio.Event()

        async def failing(ctx):
            raise RuntimeError("startup boom")

        async def healthy(ctx):
            healthy_started.set()
            await asyncio.Event().wait()

        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        with caplog.at_level(logging.ERROR, logger="gateway.plugin_services"):
            await mgr.start_services(
                [
                    _make_registration(name="failing", service=failing),
                    _make_registration(name="healthy", service=healthy),
                ],
            )
            await asyncio.wait_for(healthy_started.wait(), timeout=2)
        await mgr.stop_services()
        assert healthy_started.is_set()
        assert any(
            r.exc_info and "startup boom" in str(r.exc_info[1])
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_runtime_failure_isolated_and_logged(self, caplog):
        entered = asyncio.Event()

        async def svc(ctx):
            entered.set()
            raise RuntimeError("runtime boom")

        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        with caplog.at_level(logging.ERROR, logger="gateway.plugin_services"):
            await mgr.start_services(
                [_make_registration(name="boom", service=svc)],
            )
            # Service sets `entered` synchronously before raising; by the
            # time we observe it, _run_service has caught and logged.
            await asyncio.wait_for(entered.wait(), timeout=2)
        await mgr.stop_services()
        assert any(
            "unhandled exception" in r.getMessage()
            for r in caplog.records
        )
        assert any(
            r.exc_info and "runtime boom" in str(r.exc_info[1])
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_startup_failure_logged_with_provenance(self, caplog):
        entered = asyncio.Event()

        async def failing_svc(ctx):
            entered.set()
            raise RuntimeError("oops")

        reg = _make_registration(
            name="prov_svc",
            plugin_name="my_plugin",
            source="entrypoint",
            service=failing_svc,
        )
        mgr = GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        with caplog.at_level(logging.ERROR, logger="gateway.plugin_services"):
            await mgr.start_services([reg])
            await asyncio.wait_for(entered.wait(), timeout=2)
        await mgr.stop_services()
        msgs = [r.getMessage() for r in caplog.records]
        assert any("prov_svc" in m for m in msgs)
        assert any("my_plugin" in m for m in msgs)


class TestGatewayServiceManagerBoundedShutdown:
    """stop_services returns within the host timeout even when a service
    suppresses cancellation; overdue tasks are detached, retained for
    observation, and logged with provenance when they complete."""

    @pytest.mark.asyncio
    async def test_suppress_cancellation_detaches_overdue(
        self, monkeypatch, caplog
    ):
        """Service that swallows CancelledError cannot block shutdown:
        after the timeout it is detached, retained, and provenance-warned."""
        import gateway.plugin_services as ps

        monkeypatch.setattr(ps, "_SHUTDOWN_TIMEOUT_SECS", 0.05)

        started = asyncio.Event()
        suppress_seen = asyncio.Event()

        async def suppressing(ctx):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                suppress_seen.set()
                await asyncio.Event().wait()

        reg = _make_registration(
            name="stuck",
            plugin_name="overdue_plugin",
            source="user",
            service=suppressing,
        )
        mgr = ps.GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([reg])
        await asyncio.wait_for(started.wait(), timeout=2)

        with caplog.at_level(logging.WARNING, logger="gateway.plugin_services"):
            await mgr.stop_services()

        assert suppress_seen.is_set()
        assert not mgr.tasks
        assert len(mgr.overdue_tasks) == 1
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("stuck" in m and "did not stop" in m for m in warnings)
        assert any("overdue_plugin" in m for m in warnings)

        # Cleanup: register a drain signal AFTER the manager's overdue
        # callback so our signal fires once the manager has cleaned up,
        # then force the overdue task to a terminal state.
        overdue_snapshot = list(mgr.overdue_tasks)
        drained = _attach_drained_signal(overdue_snapshot)
        for task in overdue_snapshot:
            task.cancel()
        await asyncio.wait_for(drained.wait(), timeout=2)
        assert not mgr.overdue_tasks
        assert not mgr._registrations

    @pytest.mark.asyncio
    async def test_overdue_failure_logged_via_done_callback(
        self, monkeypatch, caplog
    ):
        """An overdue task that eventually raises must have its failure
        logged with provenance by the done callback, then be cleaned up."""
        import gateway.plugin_services as ps

        monkeypatch.setattr(ps, "_SHUTDOWN_TIMEOUT_SECS", 0.05)

        started = asyncio.Event()
        release = asyncio.Event()

        async def suppressing_then_fail(ctx):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            await release.wait()
            raise RuntimeError("eventual overdue failure")

        reg = _make_registration(
            name="late_boom",
            plugin_name="late_plugin",
            source="user",
            service=suppressing_then_fail,
        )
        mgr = ps.GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([reg])
        await asyncio.wait_for(started.wait(), timeout=2)

        with caplog.at_level(logging.WARNING, logger="gateway.plugin_services"):
            await mgr.stop_services()
        assert len(mgr.overdue_tasks) == 1

        overdue_snapshot = list(mgr.overdue_tasks)
        drained = _attach_drained_signal(overdue_snapshot)
        with caplog.at_level(logging.ERROR, logger="gateway.plugin_services"):
            release.set()
            await asyncio.wait_for(drained.wait(), timeout=2)

        errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and r.exc_info
        ]
        assert any(
            "late_boom" in r.getMessage()
            and "late_plugin" in r.getMessage()
            and "eventual overdue failure" in str(r.exc_info[1])
            for r in errors
        ), [(r.getMessage(), r.exc_info) for r in errors]
        assert not mgr.overdue_tasks
        assert not mgr._registrations

    @pytest.mark.asyncio
    async def test_overdue_normal_completion_logged_and_cleaned(
        self, monkeypatch, caplog
    ):
        """An overdue task that completes normally is observed and removed."""
        import gateway.plugin_services as ps

        monkeypatch.setattr(ps, "_SHUTDOWN_TIMEOUT_SECS", 0.05)

        started = asyncio.Event()
        release = asyncio.Event()

        async def suppressing_then_complete(ctx):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
            await release.wait()

        reg = _make_registration(
            name="late_ok",
            plugin_name="late_plugin",
            source="user",
            service=suppressing_then_complete,
        )
        mgr = ps.GatewayServiceManager(_resolver({"telegram": MagicMock()}))
        await mgr.start_services([reg])
        await asyncio.wait_for(started.wait(), timeout=2)

        with caplog.at_level(logging.WARNING, logger="gateway.plugin_services"):
            await mgr.stop_services()
        assert len(mgr.overdue_tasks) == 1

        overdue_snapshot = list(mgr.overdue_tasks)
        drained = _attach_drained_signal(overdue_snapshot)
        with caplog.at_level(logging.INFO, logger="gateway.plugin_services"):
            release.set()
            await asyncio.wait_for(drained.wait(), timeout=2)

        assert not mgr.overdue_tasks
        assert not mgr._registrations
        assert any(
            "late_ok" in r.getMessage() and "completed after" in r.getMessage()
            for r in caplog.records
        )


class TestGatewayServiceReconnectAwareAdapters:
    """A running service sees reconnect-driven adapter changes through
    the resolver on each .adapters access, without a service restart."""

    @pytest.mark.asyncio
    async def test_running_service_observes_reconnect_change(self):
        """A currently-running service observes a resolver-driven adapter
        addition on a fresh .adapters access WHILE STILL RUNNING (before
        any shutdown signal), and is started exactly once."""
        started = asyncio.Event()
        read_now = asyncio.Event()
        read_done = asyncio.Event()
        release = asyncio.Event()
        observed = []
        invocations = [0]

        async def observing_svc(ctx):
            invocations[0] += 1
            started.set()
            observed.append(set(ctx.adapters.keys()))
            await read_now.wait()
            observed.append(set(ctx.adapters.keys()))
            read_done.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                raise

        adapters = {"telegram": MagicMock(name="tg")}
        reg = _make_registration(service=observing_svc)
        mgr = GatewayServiceManager(_resolver(adapters))
        await mgr.start_services([reg])
        await asyncio.wait_for(started.wait(), timeout=2)
        assert observed == [{"telegram"}]
        assert invocations[0] == 1

        # Simulate reconnect: a new platform appears.
        adapters["discord"] = MagicMock(name="dc")

        # Ask the running service to re-read; verify while still running.
        read_now.set()
        await asyncio.wait_for(read_done.wait(), timeout=2)
        assert observed[-1] == {"telegram", "discord"}
        assert invocations[0] == 1

        await mgr.stop_services()
