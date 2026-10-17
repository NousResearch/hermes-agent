"""Detached replacement must outlast the planned stop drain (#129957)."""

import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.restart import effective_stop_drain_timeout
from gateway.run_shutdown import GatewayShutdownMixin


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("drain,after_turn", [(180.0, 1800.0), (180.0, 0.0)])
@pytest.mark.asyncio
async def test_detached_restart_helper_covers_effective_stop(monkeypatch, drain, after_turn):
    import gateway.run as gateway_run
    import subprocess

    runner = SimpleNamespace(
        _restart_requested=True,
        _restart_drain_timeout=drain,
        _restart_after_turn_timeout=after_turn,
        _detached_restart_helper_started=False,
    )
    popen = Mock()
    monkeypatch.setattr(gateway_run, "_resolve_hermes_bin", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(GatewayShutdownMixin, "_restart_watcher_env", lambda: {})

    await GatewayShutdownMixin._launch_detached_restart_command(runner)

    shell_command = popen.call_args.args[0][-1]
    match = re.search(r"\+ (\d+) \)\)", shell_command)
    assert match is not None
    deadline_seconds = int(match.group(1))
    assert deadline_seconds >= effective_stop_drain_timeout(runner) + 5
    assert shell_command.endswith("hermes gateway restart")
