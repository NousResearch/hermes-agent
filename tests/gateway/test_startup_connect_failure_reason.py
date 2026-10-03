"""Regression: fatal connect failures must name the reason in the startup log (#110072)."""

import logging

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
import gateway.run_startup as run_startup


class _FatalStartupAdapter:
    """Minimal stand-in: connect() reported a fatal, adapter holds code/message."""

    def __init__(self, *, has_fatal=True):
        self.fatal_error_code = "subscription_permission" if has_fatal else None
        self.fatal_error_message = (
            "Service Account lacks roles/pubsub.subscriber on the subscription"
            if has_fatal
            else None
        )
        self.has_fatal_error = has_fatal
        self.fatal_error_retryable = False
        self.disconnected = False

    async def disconnect(self):
        self.disconnected = True


def _startup_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._running = True
    runner._failed_platforms = {}
    runner._safe_adapter_disconnect = _fake_disconnect
    runner._update_platform_runtime_status = lambda *a, **k: None
    return runner


async def _fake_disconnect(adapter, platform):
    adapter.disconnected = True


@pytest.mark.asyncio
async def test_failed_connect_with_fatal_error_logs_code_and_message(caplog):
    adapter = _FatalStartupAdapter(has_fatal=True)
    runner = _startup_runner()
    raw = [(Platform.DISCORD, adapter, {}, "failed", None)]

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        connected = await runner._start_aggregate_connect_results(raw, [], [])

    assert connected == 0
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "failed to connect" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "subscription_permission" in warnings[0]
    assert "roles/pubsub.subscriber" in warnings[0]


@pytest.mark.asyncio
async def test_failed_connect_without_fatal_error_keeps_bare_line(caplog):
    adapter = _FatalStartupAdapter(has_fatal=False)
    runner = _startup_runner()
    raw = [(Platform.DISCORD, adapter, {}, "failed", None)]

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        connected = await runner._start_aggregate_connect_results(raw, [], [])

    assert connected == 0
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "failed to connect" in r.getMessage()
    ]
    assert warnings == ["✗ discord failed to connect"]


@pytest.mark.asyncio
async def test_fatal_detail_flows_into_nonretryable_startup_errors(caplog):
    """The aggregate list must still receive the detail even when the platform
    is marked fatal — the diagnostic path the gateway exit summary uses."""
    adapter = _FatalStartupAdapter(has_fatal=True)
    runner = _startup_runner()
    raw = [(Platform.DISCORD, adapter, {}, "failed", None)]
    retryable: list = []
    nonretryable: list = []

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._start_aggregate_connect_results(raw, retryable, nonretryable)

    assert nonretryable == [
        "discord: Service Account lacks roles/pubsub.subscriber on the subscription"
    ]
    assert retryable == []
