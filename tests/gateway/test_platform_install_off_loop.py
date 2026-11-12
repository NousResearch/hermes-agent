"""A platform's lazy dependency install never runs on the gateway event loop (#125607).

``ensure_deps_fn`` (Telegram: ``check_telegram_requirements`` -> ``pm.ensure_import`` ->
``sync_venv``) can take minutes. Run inline from ``_create_adapter`` it froze the loop until the
liveness watchdog exited the gateway with code 75.
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.platform_registry import PlatformEntry, platform_registry
from hermes_agent.gateway.run import GatewayRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("install_ok", [True, False])
async def test_install_runs_off_loop_and_is_not_retried_on_it(tmp_path, monkeypatch, install_ok):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    loop = asyncio.get_running_loop()
    loop_answered = threading.Event()
    installs = []

    def ensure_deps():
        # The install finishes only once the event loop has run a callback while it was in progress:
        # an install on the loop thread can never see that happen.
        installs.append(threading.get_ident())
        loop.call_soon_threadsafe(loop_answered.set)
        return loop_answered.wait(timeout=5) and install_ok

    adapter = MagicMock()
    monkeypatch.setitem(platform_registry._entries, "slowdeps", PlatformEntry(
        name="slowdeps", label="Slowdeps", adapter_factory=lambda cfg: adapter,
        check_fn=lambda: False, ensure_deps_fn=ensure_deps, source="plugin"))
    platform = Platform("slowdeps")
    runner = GatewayRunner(GatewayConfig(platforms={platform: PlatformConfig(enabled=True, token="t")},
                                         sessions_dir=tmp_path / "sessions"))

    created = await runner._create_adapter_off_loop_deps(platform, PlatformConfig(enabled=True))

    assert loop_answered.is_set(), "the event loop was blocked for the whole dependency install"
    assert len(installs) == 1, "the install was re-run on the event loop by create_adapter"
    assert installs[0] != threading.get_ident()
    assert (created is adapter) is install_ok
