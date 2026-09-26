"""A configured platform that fails to start must not let the gateway report a normal run.

api_server lost the bind race against the gateway it was replacing (Errno 48 / 10048), logged
``✗ api_server failed to connect`` at WARNING, and carried on serving its messaging platforms. The
startup summary said ``Gateway running with 1 platform(s)`` and the runtime state said ``running``:
status looked healthy, every API client was gone, nothing retried. The startup summary is the SHARED
gate — every platform, not just api_server — so the same failure on any other platform is reported
the same way.

The api_server case has since been escalated past reporting: a platform that binds a PORT and cannot
is fatal at startup (exit 78), because no external prober can tell an unbound port from a hung host.
The degraded-but-alive contract below is now carried by a platform that binds no port.
"""
import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from gateway.run import GatewayRunner
from gateway.status import flush_runtime_status, read_runtime_status


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


class _UnpairedWhatsAppAdapter(BasePlatformAdapter):
    """A fatal, non-retryable failure that binds NO port: the gateway stays up and degrades."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.WHATSAPP)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "whatsapp_not_paired",
            "WhatsApp is not paired. Run `hermes whatsapp pair`.",
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
async def test_parked_port_binder_takes_the_gateway_down_loudly(monkeypatch, tmp_path, caplog):
    """api_server could not bind: the failure is reported at ERROR and the gateway EXITS 78.

    This test used to assert the opposite (stay alive, report ``degraded``). Reporting was not
    enough: the process stayed up, ``systemctl is-active`` said ``active``, and a provisioner
    polling the PORT saw nothing forever. A listener that is not listening is unobservable from
    outside, so a port binder's non-retryable startup failure is now fatal even with a healthy
    sibling — see ``_startup_fail_unbound_port_binder``. The degraded-and-alive contract still holds
    for platforms that bind no port (next test).
    """
    runner = _runner_with_one_parked_platform(monkeypatch, tmp_path)

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True  # start() returns True; the clean-exit request carries the failure
        assert runner.should_exit_cleanly is True
        assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
        assert any(
            record.levelno >= logging.ERROR and "api_server" in record.getMessage()
            for record in caplog.records
        )
        flush_runtime_status()
        state = read_runtime_status()
        assert state["gateway_state"] == "startup_failed"
        assert state["platforms"]["api_server"]["state"] == "fatal"
        assert state["platforms"]["api_server"]["error_code"] == "api_server_port_in_use"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_parked_non_port_platform_is_logged_at_error_and_the_run_is_degraded(
    monkeypatch, tmp_path, caplog
):
    """A configured platform that never came up is reported at ERROR (not the WARNING that hid it)
    and the runtime state says ``degraded`` while the surviving platform keeps the gateway alive."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, platform_config: (
            _UnpairedWhatsAppAdapter() if platform is Platform.WHATSAPP else _HealthyAdapter()
        ),
    )

    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)

    with caplog.at_level(logging.ERROR):
        ok = await runner.start()
    try:
        assert ok is True, "the working platform keeps the gateway alive; only reporting changes"
        assert runner.should_exit_cleanly is False
        assert any(
            record.levelno >= logging.ERROR and "whatsapp" in record.getMessage()
            for record in caplog.records
        )
        flush_runtime_status()
        state = read_runtime_status()
        assert state["gateway_state"] == "degraded"
        assert state["platforms"]["whatsapp"]["state"] == "fatal"
        assert state["platforms"]["whatsapp"]["error_code"] == "whatsapp_not_paired"
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
