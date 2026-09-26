"""``hermes gateway restart`` must hand an externally-supervised gateway back to its supervisor.

A custom launchd agent (a plist/label outside the canonical ``ai.hermes.gateway`` path, so
``_installed_service_kind_for`` returns None) fell through to the manual stop + foreground
``run_gateway`` fallback. The foreground run stamps the restart CLI's own PID into gateway.pid,
and every KeepAlive respawn of ``gateway run --external-supervisor`` then refuses with
"Gateway already running (PID <restart>)" — the gateway stays down until the restart process is
killed (#110637).
"""
from gateway import signal_restart
from gateway import systemd_restart
from types import SimpleNamespace

import pytest

from gateway import status as gateway_status
from hermes_cli import gateway as gw
from hermes_cli import gateway_supervised_restart as supervised

SUPERVISED_ARGV = [
    "/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run", "--external-supervisor",
]


@pytest.fixture
def restart_calls(monkeypatch):
    """Drive ``_cmd_restart`` to the manual fallback (no service kind) and record the exits."""
    calls = {"sigusr1": None, "stopped": False, "started": False, "sigusr1_returns": True, "replacement": 5555}

    def _running_pid(*a, **k):
        # The supervised gateway holds the pidfile until the drain completes; after a graceful
        # exit only the supervisor's replacement can register a fresh PID.
        if calls["sigusr1"] is not None and calls["sigusr1_returns"]:
            return calls["replacement"]
        return 4321

    def _sigusr1(pid, budget, **k):
        calls["sigusr1"] = (pid, budget)
        return calls["sigusr1_returns"]

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr("gateway.restart.get_restart_exit_wait_budget", lambda: 7.0)
    monkeypatch.setattr(gateway_status, "get_running_pid", _running_pid)
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda pid: SUPERVISED_ARGV if pid == 4321 else None)
    monkeypatch.setattr("gateway.control_socket.identify_gateway", lambda home, **k: None)
    monkeypatch.setattr(supervised, "SUPERVISED_REPLACEMENT_VERIFY_TIMEOUT", 0.5)
    monkeypatch.setattr(signal_restart, "_graceful_restart_via_sigusr1", _sigusr1)
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: calls.__setitem__("stopped", True) or True)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    monkeypatch.setattr(gw, "run_gateway", lambda **k: calls.__setitem__("started", True))
    return calls


def _run_restart():
    gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))


def test_external_supervisor_gateway_restarts_via_sigusr1_handback(restart_calls, capsys):
    _run_restart()
    assert restart_calls["sigusr1"] == (4321, 7.0), "must SIGUSR1 the supervised gateway, not stop it"
    assert not restart_calls["stopped"], "the CLI must not SIGTERM a supervisor-owned gateway"
    assert not restart_calls["started"], "a foreground run would stamp the CLI PID and wedge respawns"
    out = capsys.readouterr().out
    assert "relaunched by its supervisor" in out and "5555" in out, (
        "success must ride on the verified replacement PID, not the bare exit"
    )


def test_custom_systemd_unit_gateway_hands_back_despite_socket_saying_systemd(restart_calls, monkeypatch):
    # A custom (non-canonical) unit running `gateway run --external-supervisor` sets INVOCATION_ID,
    # so the gateway self-identifies as "systemd", not "external"; that unit still owns the respawn,
    # so the socket answer must not demote the argv contract back to the stop + foreground wedge.
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway", lambda home, **k: {"pid": 4321, "supervisor": "systemd"}
    )
    _run_restart()
    assert restart_calls["sigusr1"] == (4321, 7.0)
    assert not restart_calls["stopped"] and not restart_calls["started"]


@pytest.mark.parametrize("sigusr1_returns, replacement", [(True, None), (False, 5555)])
def test_handback_failure_never_takes_ownership(restart_calls, sigusr1_returns, replacement):
    # An unloaded supervisor (clean exit, no replacement) or a drain timeout must fail loudly;
    # SIGTERM + foreground run would recreate the competing-owner wedge (#110637).
    restart_calls["sigusr1_returns"] = sigusr1_returns
    restart_calls["replacement"] = replacement
    with pytest.raises(SystemExit) as exc:
        _run_restart()
    assert exc.value.code == 1
    assert restart_calls["sigusr1"] == (4321, 7.0), "the graceful handback must be attempted first"
    assert not restart_calls["stopped"] and not restart_calls["started"]


def test_plain_manual_gateway_still_uses_stop_and_run(restart_calls, monkeypatch):
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda pid: ["/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run"])
    _run_restart()
    assert restart_calls["sigusr1"] is None, "no supervisor marker: the detached fallback is the restart"
    assert restart_calls["started"], "a plain manually-run gateway must still be restarted in-process"
