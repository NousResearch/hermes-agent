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


class TestGatewayServiceContext:
    """Context exposes a read-only adapter snapshot, not the runner."""

    def test_adapters_read_only_snapshot(self):
        tg = MagicMock(name="tg")
        ctx = GatewayServiceContext({"telegram": tg})
        assert isinstance(ctx.adapters, MappingProxyType)
        assert set(ctx.adapters.keys()) == {"telegram"}
        assert ctx.adapters["telegram"] is tg

    def test_adapters_is_decoupled_from_source(self):
        source = {"telegram": MagicMock()}
        ctx = GatewayServiceContext(source)
        source["discord"] = MagicMock()
        assert "discord" not in ctx.adapters

    def test_context_does_not_expose_runner(self):
        ctx = GatewayServiceContext({})
        assert not hasattr(ctx, "runner")
        assert not hasattr(ctx, "_runner")


class TestGatewayServiceManagerNoOp:
    """Manager is a no-op with no registrations or no adapters."""

    @pytest.mark.asyncio
    async def test_noop_no_registrations_or_adapters(self):
        mgr = GatewayServiceManager()
        await mgr.start_services([], {"telegram": MagicMock()})
        assert not mgr.started and not mgr.tasks
        await mgr.start_services([_make_registration()], {})
        assert not mgr.started and not mgr.tasks

    @pytest.mark.asyncio
    async def test_stop_noop_with_no_tasks(self):
        mgr = GatewayServiceManager()
        await mgr.stop_services()

    @pytest.mark.asyncio
    async def test_noop_then_real_start_preserves_exact_once(self):
        """A no-op call (no registrations) must not mark started; a later
        real call launches services exactly once."""
        started = asyncio.Event()

        async def svc(ctx):
            started.set()
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager()
        await mgr.start_services([], {"telegram": MagicMock()})
        assert not mgr.started
        await mgr.start_services([reg], {"telegram": MagicMock()})
        assert mgr.started
        await asyncio.wait_for(started.wait(), timeout=2)
        await mgr.start_services([reg], {"telegram": MagicMock()})
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
        mgr = GatewayServiceManager()
        await mgr.start_services([reg], {"telegram": MagicMock()})
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
        mgr = GatewayServiceManager()
        await mgr.start_services([reg], {"telegram": adapter})
        await asyncio.wait_for(seen.wait(), timeout=2)
        await mgr.stop_services()
        assert captured.get("adapters") == {"telegram": adapter}

    @pytest.mark.asyncio
    async def test_exactly_once_repeated_start(self):
        call_count = 0

        async def svc(ctx):
            nonlocal call_count
            call_count += 1
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager()
        await mgr.start_services([reg], {"telegram": MagicMock()})
        await asyncio.sleep(0.01)
        first_tasks = mgr.tasks
        await mgr.start_services([reg], {"telegram": MagicMock()})
        assert mgr.tasks == first_tasks
        assert call_count == 1
        await mgr.stop_services()

    @pytest.mark.asyncio
    async def test_strong_retention(self):
        async def svc(ctx):
            await asyncio.Event().wait()

        reg = _make_registration(service=svc)
        mgr = GatewayServiceManager()
        await mgr.start_services([reg], {"telegram": MagicMock()})
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
        mgr = GatewayServiceManager()
        await mgr.start_services([reg], {"telegram": MagicMock()})
        await asyncio.wait_for(service_started.wait(), timeout=2)
        await mgr.stop_services()
        assert cancel_seen.is_set()
        assert len(mgr.tasks) == 0


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

        mgr = GatewayServiceManager()
        with caplog.at_level(logging.ERROR, logger="gateway.plugin_services"):
            await mgr.start_services(
                [
                    _make_registration(name="failing", service=failing),
                    _make_registration(name="healthy", service=healthy),
                ],
                {"telegram": MagicMock()},
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
        async def svc(ctx):
            raise RuntimeError("runtime boom")

        mgr = GatewayServiceManager()
        with caplog.at_level(logging.ERROR, logger="gateway.plugin_services"):
            await mgr.start_services(
                [_make_registration(name="boom", service=svc)],
                {"telegram": MagicMock()},
            )
            await asyncio.sleep(0.1)
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
        async def failing_svc(ctx):
            raise RuntimeError("oops")

        reg = _make_registration(
            name="prov_svc",
            plugin_name="my_plugin",
            source="entrypoint",
            service=failing_svc,
        )
        mgr = GatewayServiceManager()
        with caplog.at_level(logging.ERROR, logger="gateway.plugin_services"):
            await mgr.start_services([reg], {"telegram": MagicMock()})
            await asyncio.sleep(0.1)
        await mgr.stop_services()
        msgs = [r.getMessage() for r in caplog.records]
        assert any("prov_svc" in m for m in msgs)
        assert any("my_plugin" in m for m in msgs)
