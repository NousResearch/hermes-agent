"""A configured platform that fails to start must not let the gateway report a normal run.

api_server lost the bind race against the gateway it was replacing (Errno 48 / 10048), logged
``✗ api_server failed to connect`` at WARNING, and carried on serving its messaging platforms. The
startup summary said ``Gateway running with 1 platform(s)`` and the runtime state said ``running``:
status looked healthy, every API client was gone, nothing retried. The startup summary is the SHARED
gate — every platform, not just api_server — so the same failure on any other platform is reported
the same way.
"""
import logging
import time

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import flush_runtime_status_async, read_runtime_status


class _PortTakenAdapter(BasePlatformAdapter):
    """Loses the bind race exactly like api_server: fatal AND non-retryable, so it is parked."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, extra={"port": 8642}), Platform.API_SERVER)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "api_server_port_in_use",
            "Port 8642 already in use. Set platforms.api_server.port in config.yaml to a different "
            "value, then `/platform resume api_server`.",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _HealthyAdapter(BasePlatformAdapter):
    """The messaging platform that keeps the gateway useful — and used to hide the failure."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(monkeypatch, tmp_path, platforms, create_adapter) -> GatewayRunner:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # No plugin registers any platform here, so a None adapter is the "plugin never registered" case.
    monkeypatch.setattr("gateway.platform_registry.platform_registry.is_registered", lambda name: False)
    runner = GatewayRunner(GatewayConfig(platforms=platforms, sessions_dir=tmp_path / "sessions"))
    monkeypatch.setattr(runner, "_create_adapter", create_adapter)

    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)
    return runner


def _runner_with_one_parked_platform(monkeypatch, tmp_path) -> GatewayRunner:
    return _runner(
        monkeypatch, tmp_path,
        {
            Platform.API_SERVER: PlatformConfig(enabled=True, extra={"port": 8642}),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
        },
        lambda platform, platform_config: (
            _PortTakenAdapter() if platform is Platform.API_SERVER else _HealthyAdapter()
        ),
    )


@pytest.mark.asyncio
async def test_parked_platform_is_logged_at_error_and_the_run_is_degraded(monkeypatch, tmp_path, caplog):
    """A configured platform that never came up is reported at ERROR (not the WARNING that hid it)
    and the runtime state says ``degraded`` while the surviving platform keeps the gateway alive."""
    runner = _runner_with_one_parked_platform(monkeypatch, tmp_path)

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True, "the working platform keeps the gateway alive; only reporting changes"
        assert runner.should_exit_cleanly is False
        assert any(
            record.levelno >= logging.ERROR and "api_server" in record.getMessage()
            for record in caplog.records
        )
        state = read_runtime_status()
        assert state["gateway_state"] == "degraded"
        assert state["platforms"]["api_server"]["state"] == "fatal"
        assert state["platforms"]["api_server"]["error_code"] == "api_server_port_in_use"
        assert state["platforms"]["telegram"]["state"] == "connected"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_every_platform_connected_still_reports_a_normal_run(monkeypatch, tmp_path, caplog):
    """Protection: a clean startup keeps the normal ``running`` state and logs no ERROR."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _HealthyAdapter())

    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True
        state = read_runtime_status()
        assert state["gateway_state"] == "running"
        assert state["platforms"]["telegram"]["state"] == "connected"
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
            "a healthy startup must not log ERROR"
        )
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_missing_adapter_degrades_only_the_unserved_enabled_platform(monkeypatch, tmp_path):
    """An enabled missing plugin must appear in status even when a sibling connects."""
    runner = _runner(
        monkeypatch, tmp_path,
        {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
            Platform.SLACK: PlatformConfig(enabled=False, token="***"),
            # A builtin whose probe fails on missing creds needs a config change: flagged once, never
            # queued (else it re-warns forever at the backoff cap on fleet nodes, #5196).
            Platform.BLUEBUBBLES: PlatformConfig(enabled=True),
        },
        lambda platform, platform_config: _HealthyAdapter() if platform is Platform.TELEGRAM else None,
    )
    try:
        assert await runner.start() is True
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "connected"
        assert state["platforms"]["discord"]["state"] == "retrying"
        assert state["platforms"]["discord"]["error_code"] == "adapter_unavailable"
        assert state["platforms"]["discord"]["needs_attention"] is True
        assert "slack" not in state["platforms"]
        bluebubbles = state["platforms"]["bluebubbles"]
        assert (bluebubbles["state"], bluebubbles["error_code"]) == ("fatal", "adapter_unavailable")
        assert bluebubbles["needs_attention"] is True
        assert "Retrying" not in bluebubbles["error_message"]
        assert list(runner._failed_platforms) == [Platform.DISCORD]  # the reconnect watcher owns it
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_adapterless_platform_heals_once_its_adapter_appears(monkeypatch, tmp_path):
    """A plugin that registers late: the queued platform stays queued, then connects. One that turns
    into a registered plugin still returning None needs a config change, so the watcher drops it."""
    platform = Platform.DISCORD
    adapters = {}
    runner = _runner(
        monkeypatch, tmp_path, {platform: PlatformConfig(enabled=True, token="***")},
        lambda platform, platform_config: adapters.get(platform),
    )
    try:
        assert await runner.start() is True
        status = read_runtime_status()["platforms"][platform.value]
        assert status["error_code"] == "adapter_unavailable"
        assert status["needs_attention"] is True
        await runner._reconnect_failed_platform(platform, time.monotonic() + 60)
        assert runner._failed_platforms[platform]["attempts"] == 2  # still missing: kept queued
        assert await flush_runtime_status_async()
        assert "check the plugin" in read_runtime_status()["platforms"][platform.value]["error_message"]

        healthy = _HealthyAdapter()
        healthy.platform = platform
        adapters[platform] = healthy
        await runner._reconnect_failed_platform(platform, time.monotonic() + 3600)
        assert runner.adapters[platform] is healthy
        assert runner._failed_platforms == {}
        assert await flush_runtime_status_async()
        status = read_runtime_status()["platforms"][platform.value]
        assert status["state"] == "connected"
        assert status["needs_attention"] is False

        # Registered plugin whose factory returns None: the watcher marks it fatal and stops retrying.
        slack = Platform.SLACK
        runner._failed_platforms[slack] = runner._startup_retry_entry(
            slack, None, PlatformConfig(enabled=True, token="***"),
        )
        monkeypatch.setattr("gateway.platform_registry.platform_registry.is_registered", lambda name: True)
        await runner._reconnect_failed_platform(slack, time.monotonic() + 3600)
        assert slack not in runner._failed_platforms
        assert await flush_runtime_status_async()  # status writes are queued off-thread
        status = read_runtime_status()["platforms"][slack.value]
        assert (status["state"], status["error_code"]) == ("fatal", "adapter_unavailable")
        assert "Retrying" not in status["error_message"]
    finally:
        await runner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["healthy", "parked", "retrying"])
@pytest.mark.parametrize("fail_wiring", [False, True])
async def test_serving_state_waits_for_startup_wiring(monkeypatch, tmp_path, mode, fail_wiring):
    """An awaited restore must finish before observers can retire startup debt."""
    if mode == "parked":
        runner = _runner_with_one_parked_platform(monkeypatch, tmp_path)
    else:
        runner = _runner(
            monkeypatch, tmp_path,
            {Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
            lambda *_: _HealthyAdapter() if mode == "healthy" else None,
        )
    finish_wiring = runner._start_finish_wiring
    restore = runner._finish_startup_restore
    observed = []

    async def check_starting():
        assert await flush_runtime_status_async()
        state = read_runtime_status()["gateway_state"]
        observed.append(state)
        assert state == "starting"

    async def gated_restore():
        await check_starting()
        await restore()

    async def gated_wiring(count):
        await check_starting()
        await finish_wiring(count)
        await check_starting()
        if fail_wiring:
            raise RuntimeError("startup wiring failed")

    monkeypatch.setattr(runner, "_finish_startup_restore", gated_restore)
    monkeypatch.setattr(runner, "_start_finish_wiring", gated_wiring)
    try:
        if fail_wiring:
            with pytest.raises(RuntimeError, match="startup wiring failed"):
                await runner.start()
        else:
            assert await runner.start()
        expected = "starting" if fail_wiring else "degraded" if mode == "parked" else "running"
        assert read_runtime_status()["gateway_state"] == expected
        assert observed == ["starting"] * 3
    finally:
        await runner.stop()
