"""A configured platform that fails to start must not let the gateway report a normal run.

api_server lost the bind race against the gateway it was replacing (Errno 48 / 10048), logged
``✗ api_server failed to connect`` at WARNING, and carried on serving its messaging platforms. The
startup summary said ``Gateway running with 1 platform(s)`` and the runtime state said ``running``:
status looked healthy, every API client was gone, nothing retried. The startup summary is the SHARED
gate — every platform, not just api_server — so the same failure on any other platform is reported
the same way.
"""
import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


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


def _runner_with_one_parked_platform(monkeypatch, tmp_path) -> GatewayRunner:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.API_SERVER: PlatformConfig(enabled=True, extra={"port": 8642}),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, platform_config: (
            _PortTakenAdapter() if platform is Platform.API_SERVER else _HealthyAdapter()
        ),
    )
    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)
    return runner


@pytest.mark.asyncio
async def test_parked_platform_is_reported_at_error_when_others_serve(monkeypatch, tmp_path, caplog):
    """The failure is logged at ERROR — not only the WARNING that hid it."""
    runner = _runner_with_one_parked_platform(monkeypatch, tmp_path)

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True, "the working platform keeps the gateway alive; only reporting changes"
        assert runner.should_exit_cleanly is False
        assert any(
            record.levelno >= logging.ERROR and "api_server" in record.getMessage()
            for record in caplog.records
        ), "a configured platform that never came up must be reported at ERROR, not only a WARNING"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_parked_platform_is_not_reported_as_a_normal_run(monkeypatch, tmp_path):
    """The runtime state must not claim a normal run while a configured platform is unserved."""
    runner = _runner_with_one_parked_platform(monkeypatch, tmp_path)

    ok = await runner.start()
    try:
        assert ok is True
        assert runner.should_exit_cleanly is False
        state = read_runtime_status()
        assert state["gateway_state"] == "degraded", (
            "the gateway is serving with a configured platform missing — it must not claim a normal run"
        )
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
