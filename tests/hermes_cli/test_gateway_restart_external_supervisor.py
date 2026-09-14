"""``hermes gateway restart`` must hand an externally-supervised gateway back to its supervisor.

A custom launchd agent (a plist/label outside the canonical ``ai.hermes.gateway`` path, so
``_installed_service_kind_for`` returns None) fell through to the manual stop + foreground
``run_gateway`` fallback. The foreground run stamps the restart CLI's own PID into
gateway.pid, and every KeepAlive respawn of ``gateway run --external-supervisor`` then refuses
with "Gateway already running (PID <restart>)" — the gateway stays down until the restart
process is killed (#110637). The fix trusts the same ``--external-supervisor`` argv marker
the update path uses (``_prepare_profile_gateway_update_restart``) and SIGUSR1s the gateway
so it drains, exits, and lets the supervisor relaunch it.
"""
from types import SimpleNamespace

import pytest

from gateway import status as gateway_status
from hermes_cli import gateway as gw

SUPERVISED_ARGV = [
    "/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run", "--external-supervisor",
]


@pytest.fixture
def restart_calls(monkeypatch):
    """Drive ``_cmd_restart`` to the manual fallback (no service kind) and record the exits."""
    calls = {"sigusr1": None, "stopped": False, "started": False}
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gw, "_get_restart_exit_wait_budget", lambda: 7.0)
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: 4321)
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda pid: SUPERVISED_ARGV if pid == 4321 else None)

    def _sigusr1(pid, budget, **k):
        calls["sigusr1"] = (pid, budget)
        return calls["sigusr1_returns"]

    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", _sigusr1)
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: calls.__setitem__("stopped", True) or True)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    monkeypatch.setattr(gw, "run_gateway", lambda **k: calls.__setitem__("started", True))
    calls["sigusr1_returns"] = True
    return calls


def _run_restart():
    gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))


def test_external_supervisor_gateway_restarts_via_sigusr1_handback(restart_calls, capsys):
    _run_restart()
    assert restart_calls["sigusr1"] == (4321, 7.0), "must SIGUSR1 the supervised gateway, not stop it"
    assert not restart_calls["stopped"], "the CLI must not SIGTERM a supervisor-owned gateway"
    assert not restart_calls["started"], "a foreground run would stamp the CLI PID and wedge respawns"
    assert "supervisor to relaunch" in capsys.readouterr().out


def test_plain_manual_gateway_still_uses_stop_and_run(restart_calls, monkeypatch):
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda pid: ["/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run"])
    _run_restart()
    assert restart_calls["sigusr1"] is None, "no supervisor marker: the detached fallback is the restart"
    assert restart_calls["started"], "a plain manually-run gateway must still be restarted in-process"


@pytest.mark.parametrize("pid,argv", [(None, None), (4321, None)])
def test_uncapturable_gateway_still_uses_stop_and_run(restart_calls, monkeypatch, pid, argv):
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: pid)
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda p: argv)
    _run_restart()
    assert restart_calls["sigusr1"] is None
    assert restart_calls["started"], "capture failure must degrade to the existing fallback, never a silent no-op"


def test_drain_timeout_falls_back_to_stop_and_run(restart_calls):
    restart_calls["sigusr1_returns"] = False
    _run_restart()
    assert restart_calls["sigusr1"] == (4321, 7.0), "the graceful handback must be attempted first"
    assert restart_calls["stopped"] and restart_calls["started"], \
        "a wedged/timed-out drain must still complete the restart, not strand the gateway down"
