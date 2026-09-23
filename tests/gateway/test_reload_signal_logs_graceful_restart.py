"""SIGUSR1 (systemd ``ExecReload`` / ``hermes gateway restart``) is a graceful restart, and the
gateway says so in its log when the signal lands (#117267)."""

from __future__ import annotations

import logging
import signal

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig


class _AbortedStartupRunner:
    """Aborts before running mode so start_gateway() returns right after installing signal handlers."""

    def __init__(self, config):
        self.config = config
        self.adapters = {}
        self._running = False
        self._restart_requested = False
        self._restart_via_service = False
        self.should_exit_cleanly = True
        self.should_exit_with_failure = False
        self.exit_reason = None
        self.exit_code = None
        self.restart_calls = []

    async def start(self):
        return False

    async def wait_for_shutdown(self):
        return None

    def request_restart(self, *, detached=False, via_service=False):
        self.restart_calls.append({"detached": detached, "via_service": via_service})
        return True


@pytest.mark.asyncio
async def test_sigusr1_handler_installed_by_start_gateway_logs_graceful_restart(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for target, value in (
        ("gateway.status.get_running_pid", lambda: None),
        ("gateway.status.acquire_gateway_runtime_lock", lambda: True),
        ("gateway.status.write_pid_file", lambda: None),
        ("gateway.status.remove_pid_file", lambda: None),
        ("gateway.status.release_gateway_runtime_lock", lambda: None),
        ("tools.skills_sync.sync_skills", lambda quiet=True: None),
        ("hermes_logging.setup_logging", lambda hermes_home, mode: None),
        ("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda: None),
    ):
        monkeypatch.setattr(target, value)
    runners = []

    def make_runner(config):
        runners.append(_AbortedStartupRunner(config))
        return runners[-1]

    monkeypatch.setattr(gateway_run, "GatewayRunner", make_runner)

    import asyncio
    loop = asyncio.get_running_loop()
    installed = {}
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, handler, *args: installed.update({sig: (handler, args)}))

    await gateway_run.start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    handler, args = installed[signal.SIGUSR1]
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        handler(*args)

    assert runners[0].restart_calls == [{"detached": False, "via_service": True}]
    text = " ".join(r.getMessage() for r in caplog.records if r.name == "gateway.run")
    assert "SIGUSR1 received" in text and "systemctl reload" in text
    assert "graceful gateway restart" in text and "Not an in-process config reload" in text
