"""A supervised gateway's launch-chain wrappers are not unmapped gateways (#135171).

On macOS each launchd-supervised gateway runs as ``osascript`` → ``hermes_cli.stderr_timestamp``
shim → gateway python. The wrappers match ``find_gateway_pids`` but map to no profile, so the
unmapped sweep SIGTERMed them and recorded ``stopped_unmapped`` rows — an obligation
``_marker_owed_gateways`` renders as ``_UNMAPPED_GATEWAY``, which no live fleet can settle when
every mapped gateway covers its own profile, so the post-update warning never cleared. The sweep
must stop a wrapper with its gateway (the supervisor relaunches the whole chain) without owing an
unmapped-restart debt for it — and keep owing it when the chain could not be walked at all.
"""

from __future__ import annotations

import os
import signal

import psutil
import pytest

from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import dashboard_procs


class _ProfileProc:
    def __init__(self, pid: int, profile: str):
        self.pid = pid
        self.profile = profile


# gateway 100 runs under shim 90 under osascript 80 under launchd (pid 1)
_PARENTS = {100: 90, 90: 80, 80: 1}


class _FakeProc:
    def __init__(self, pid: int):
        self.pid = pid

    def parent(self) -> "_FakeProc":
        return _FakeProc(_PARENTS.get(self.pid, 1))


@pytest.fixture
def own_home(monkeypatch, tmp_path):
    home = tmp_path / "homeA" / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", home)])
    return home


def _pid_homes(monkeypatch, mapping: dict):
    monkeypatch.setattr(dashboard_procs, "_hermes_home_for_pid", lambda pid: mapping.get(pid))


def _outcome() -> fleet._GatewayRestartOutcome:
    return fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
    )


def _mock_fleet(monkeypatch, own_home, *, walkable_chain: bool):
    _pid_homes(monkeypatch, {100: str(own_home), 90: str(own_home), 80: str(own_home)})
    monkeypatch.setattr("hermes_cli.gateway._get_service_pids", lambda **k: set())
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids", lambda **k: [100, 90, 80])
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes",
        lambda **k: [_ProfileProc(100, "p1")])
    monkeypatch.setattr(
        "hermes_cli.gateway._prepare_profile_gateway_update_restart", lambda *a: "external-supervisor")
    monkeypatch.setattr(fleet, "_drain_or_signal_gateway_for_update", lambda *a, **k: True)
    monkeypatch.setattr("hermes_cli.gateway._wait_for_gateway_exit", lambda **k: None)
    if walkable_chain:
        monkeypatch.setattr(psutil, "Process", _FakeProc)
    else:
        def _dead(pid):
            raise psutil.NoSuchProcess(pid)

        monkeypatch.setattr(psutil, "Process", _dead)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    return killed


def test_a_walked_launch_chain_is_stopped_without_an_unmapped_debt(monkeypatch, own_home, capsys):
    """The shim and osascript of a mapped gateway are SIGTERMed (the supervisor relaunches the
    chain) but recorded as no unmapped gateway, so the restart warning can clear once the
    mapped gateway is back."""
    killed = _mock_fleet(monkeypatch, own_home, walkable_chain=True)

    out = _outcome()
    fleet._restart_manual_gateways(out, 5.0)

    assert killed == [(90, signal.SIGTERM), (80, signal.SIGTERM)]
    assert out.killed_pids == {100}  # the gateway alone: its drain signalled it
    assert out.externally_supervised_profiles == ["p1"]
    assert out.stopped_unmapped_pids == set()
    assert "Restart manually" not in capsys.readouterr().out


def test_an_unwalkable_launch_chain_keeps_the_unmapped_debt_fail_closed(monkeypatch, own_home):
    """A chain whose gateway died before the walk proves nothing: the wrappers stay recorded
    as unmapped stops, exactly as before (#135171's fix must not widen this fail-closed net)."""
    killed = _mock_fleet(monkeypatch, own_home, walkable_chain=False)

    out = _outcome()
    fleet._restart_manual_gateways(out, 5.0)

    assert killed == [(90, signal.SIGTERM), (80, signal.SIGTERM)]
    assert out.killed_pids == {100, 90, 80}
    assert out.stopped_unmapped_pids == {90, 80}
