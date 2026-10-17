"""start_gateway exits that never reach running mode must not look like a clean shutdown."""
import asyncio
import threading
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def stub_runtime_listeners(monkeypatch):
    """Keep real startup/stop and database ownership without binding local sockets."""
    monkeypatch.setattr(
        "gateway.control_socket.GatewayControlServer.start",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr("gateway.run_runtime.start_gateway_runtime_api", AsyncMock())


@pytest.mark.asyncio
async def test_refused_startup_keeps_startup_failed_and_crash_recovery(monkeypatch):
    from gateway import run as gateway_run
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run_bootstrap import start_gateway
    from gateway.status import read_runtime_status
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.chmod(0o700)
    monkeypatch.setattr(gateway_run, '_hermes_home', home)
    for name in ('GATEWAY_ALLOW_ALL_USERS', 'WHATSAPP_ALLOW_ALL_USERS', 'WHATSAPP_DM_POLICY'):
        monkeypatch.delenv(name, raising=False)
    # The open-DM refusal: start() writes startup_failed and requests a clean exit before recovery.
    config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(
        enabled=True, extra={'dm_policy': 'open'})})

    assert await asyncio.wait_for(start_gateway(config), 30) is True

    assert not (home / '.clean_shutdown').exists(), 'refused start wrote the clean-shutdown marker'
    assert read_runtime_status()['gateway_state'] == 'startup_failed'
    assert not (home / 'gateway.pid').exists()


@pytest.mark.asyncio
async def test_ready_publication_failure_stops_cron_and_housekeeping(monkeypatch):
    from gateway import run_bootstrap, run_runtime
    from gateway.config import GatewayConfig
    from hermes_constants import get_hermes_home

    get_hermes_home().chmod(0o700)
    stopped = threading.Event()
    threads = []

    def start_owned_threads(runner):
        for _ in range(2):
            thread = threading.Thread(target=stopped.wait, daemon=True)
            thread.start()
            threads.append(thread)
        return stopped, None, *threads

    def refuse_ready(runner):
        raise RuntimeError('gateway stopped before session API readiness')

    monkeypatch.setattr(run_bootstrap, '_start_gateway_start_cron_and_housekeeping', start_owned_threads)
    monkeypatch.setattr(run_runtime, 'publish_gateway_runtime_ready', refuse_ready)
    try:
        with pytest.raises(RuntimeError, match='before session API readiness'):
            await asyncio.wait_for(run_bootstrap.start_gateway(GatewayConfig()), 30)
        assert threads
        assert stopped.is_set(), 'READY failure left the cron ticker running'
        assert all(not thread.is_alive() for thread in threads)
    finally:
        stopped.set()
        for thread in threads:
            thread.join(timeout=2)
