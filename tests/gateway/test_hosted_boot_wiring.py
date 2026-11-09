"""The gateway runs the hosted boot checks before it serves, and freezes the registries at the end
of hosted boot. A local install consults neither (the switch is off), which the fake-runner boot
below drives through the real ``start_gateway``.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock

import pytest

from gateway import run_bootstrap


@pytest.fixture
def _startup(monkeypatch, tmp_path):
    """Drive the real ``start_gateway`` with a fake runner; record startup milestones. Modelled on
    the deferred-integrity-scan boot fixture (a full wiring pass with nothing real started)."""
    from agent import memory_provider
    from gateway import lifecycle_ledger as ledger
    from gateway import run_api, run_runtime
    from gateway import run as run_mod

    events, release = [], threading.Event()

    class _LiveRunner:
        def __init__(self, config):
            from gateway.session import SessionStore
            from hermes_constants import get_hermes_home
            from hermes_state import SessionDB

            self.session_store = SessionStore(get_hermes_home() / "sessions", config)
            self._session_db = SessionDB(get_hermes_home() / "state.db")
            self.session_store._db = self._session_db
            self._draining = False
            self._running = False
            self._closed = False
            self.config = config
            self.should_exit_cleanly = False
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            events.append("runner_started")
            self._running = True
            return True

        def _start_systemd_watchdog(self):
            pass

        async def stop(self):
            self._running = False
            if not self._closed:
                self._closed = True
                self.session_store.close_all_db_handles()
                self._session_db.close()

    monkeypatch.setattr(run_mod, "_host_attach_or_none", AsyncMock(return_value=None))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "GatewayRunner", _LiveRunner)
    monkeypatch.setattr(ledger, "record_startup", lambda: None)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_start_cron_and_housekeeping",
                        lambda runner: (threading.Event(), None, None, None))
    monkeypatch.setattr(run_bootstrap, "_start_gateway_shutdown_tail", AsyncMock(return_value=True))
    monkeypatch.setattr(run_runtime, "publish_gateway_runtime_ready", lambda runner: events.append("ready"))
    monkeypatch.setattr(run_runtime, "wait_gateway_runtime", AsyncMock())
    monkeypatch.setattr(run_api, "warm_gateway_dashboard", lambda runner: None)
    yield events
    release.set()
    from gateway import host_rendezvous as hr

    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.clear_record(hr.ROLE_GATEWAY)


@pytest.mark.asyncio
async def test_a_local_boot_never_consults_the_hosted_checks(_startup, monkeypatch):
    """hosted false (the default): the boot check runs and consults nothing — no refusal check
    fires, no freeze fires — and the local install's gateway starts exactly as before."""
    from hermes_cli import hosted_mode

    def _must_not_run(*args, **kwargs):
        raise AssertionError("a hosted boot check ran on a local install")

    for check in ("_refuse_credential_environment", "_refuse_profile_secret_files",
                  "_refuse_non_sandbox_terminal", "_refuse_direct_platform_adapters",
                  "_refuse_uncurated_inprocess_code", "_require_hosted_backends"):
        monkeypatch.setattr(hosted_mode, check, _must_not_run)
    from gateway.config import GatewayConfig

    assert await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False,
                                             verbosity=None) is True
    assert _startup == ["runner_started", "ready"]
    import registry_freeze

    assert registry_freeze.registries_frozen() is False  # the freeze never fired


@pytest.mark.asyncio
async def test_a_hosted_refusal_stops_the_boot_before_anything_serves(_startup, monkeypatch):
    """The checks run before the host attach, the PID claim or the runner: a refused condition
    exits with the fatal-config code and nothing started."""
    from hermes_cli import hosted_mode
    from hermes_cli.hosted_mode import HostedBootRefused

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)

    def refuse(*args, **kwargs):
        raise HostedBootRefused("a condition the process may not serve with. the fix")

    monkeypatch.setattr(hosted_mode, "assert_hosted_boot", refuse)
    from gateway.config import GatewayConfig
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE

    with pytest.raises(SystemExit) as exit_info:
        await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False, verbosity=None)
    assert exit_info.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert _startup == []  # no runner, no adapters, nothing served


@pytest.mark.asyncio
async def test_the_registries_freeze_at_the_end_of_hosted_boot(_startup, monkeypatch):
    """Hosted boot finishes once everything the process serves is loaded: the freeze runs after
    the runner started (adapters, MCP) and before the runtime is published ready."""
    from hermes_cli import hosted_mode

    events = _startup
    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    monkeypatch.setattr(hosted_mode, "assert_hosted_boot", lambda *a, **k: None)
    monkeypatch.setattr(hosted_mode, "finish_hosted_boot",
                        lambda: events.append("registries_frozen"))
    from gateway.config import GatewayConfig

    assert await run_bootstrap.start_gateway(config=GatewayConfig(), replace=False,
                                             verbosity=None) is True
    assert events == ["runner_started", "registries_frozen", "ready"]
