"""An update run from one checkout never rewrites or restarts a gateway of another checkout.

A worktree's update (HERMES_HOME a scratch dir under the account's ``~/.hermes``) shares the
account's units and gateway processes. It used to repoint the account's live unit at the
worktree's launcher and SIGUSR1 the live gateway it ran inside, though its own fleet report
already called that gateway "separate checkout, not updated by this run".
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_launch_files
from hermes_cli.update_receipt import _updater_code_root as _maybe_root


def _updater_code_root() -> Path:
    root = _maybe_root()
    assert root is not None
    return root

STALE = "RestartSec=30\n"


def _checkout(path: Path) -> Path:
    (path / "hermes_cli").mkdir(parents=True)
    (path / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    return path


def _unit(tmp_path: Path, launcher_root: Path, home: Path) -> Path:
    unit = tmp_path / "units" / "hermes-gateway.service"
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(f'[Service]\nExecStart="{launcher_root}/.hermes/bin/hermes" "gateway" "run"\n'
                    f'Environment="HERMES_HOME={home}"\n{STALE}', encoding="utf-8")
    return unit


@pytest.fixture
def account(tmp_path, monkeypatch):
    """The incident shape: account root ``<tmp>/.hermes``, the updater's home a scratch dir in it."""
    root = tmp_path / ".hermes"
    scratch = root / "cache" / "scratch" / "worktree-home"
    scratch.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", root)])
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gateway_cli, "generate_systemd_unit", lambda system=False, run_as_user=None: "[Service]\nRestartSec=5\n")
    monkeypatch.setattr(gateway_cli, "_prepare_service_launcher", lambda **kw: None)
    monkeypatch.setattr(update_launch_files.os, "geteuid", lambda: 1000)
    events: list = []
    monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda args, **kw: events.append(("systemctl", *args)))
    return root, scratch, events


@pytest.mark.platforms("linux")
def test_scratch_home_update_never_rewrites_the_account_unit_or_another_checkouts_unit(
        tmp_path, monkeypatch, capsys, account):
    root, scratch, events = account
    other = _checkout(tmp_path / "runtime-checkout")

    # A scratch home nested in the account root is not the account's default profile.
    monkeypatch.setenv("HERMES_HOME", str(scratch))
    unit = _unit(tmp_path, _updater_code_root(), root)
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit)
    update_launch_files.refresh_launch_files()
    assert unit.read_text(encoding="utf-8").endswith(STALE) and events == []

    # The account's own home, but the unit launches another checkout: left alone, named. Both the
    # published-launcher form and the ``python -c "sys.path.insert(0, '<root>')"`` form, whose
    # checkout path may contain spaces.
    monkeypatch.setenv("HERMES_HOME", str(root))
    spaced = _checkout(tmp_path / "Hermes Live")
    unit = _unit(tmp_path, other, root)
    update_launch_files.refresh_launch_files()
    assert unit.read_text(encoding="utf-8").endswith(STALE) and events == []
    assert f"another Hermes checkout ({other.resolve()})" in capsys.readouterr().out
    unit.write_text(f'[Service]\nExecStart="/usr/bin/python3" "-I" "-c" "import sys; sys.path.insert(0, \'{spaced}\')" '
                    f'"gateway" "run"\nEnvironment="HERMES_HOME={root}"\n{STALE}', encoding="utf-8")
    update_launch_files.refresh_launch_files()
    assert unit.read_text(encoding="utf-8").endswith(STALE) and events == []
    assert f"another Hermes checkout ({spaced.resolve()})" in capsys.readouterr().out

    # The launchd restart applies the same checkout rule before ``launchd_restart`` refreshes the plist.
    import plistlib
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_bytes(plistlib.dumps({"Label": "ai.hermes.gateway", "EnvironmentVariables": {"HERMES_HOME": str(root)},
                                      "ProgramArguments": [str(other / ".hermes" / "bin" / "hermes"), "gateway", "run"]}))
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway_cli, "launchd_restart", lambda: events.append(("launchd_restart",)))
    assert fleet._restart_launchd_gateway_after_update(supervision_verify=False) == ([], [])
    assert events == [] and "launchd job of another Hermes checkout" in capsys.readouterr().out

    # Control: this checkout's own stale unit is still rewritten and reloaded.
    unit = _unit(tmp_path, _updater_code_root(), root)
    update_launch_files.refresh_launch_files()
    assert unit.read_text(encoding="utf-8") == "[Service]\nRestartSec=5\n"
    assert events == [("systemctl", "daemon-reload")]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("own", [False, True], ids=["other-checkout", "own-checkout"])
def test_ancestor_gateway_of_another_checkout_is_never_signalled(tmp_path, monkeypatch, own):
    code_root = _updater_code_root() if own else _checkout(tmp_path / "runtime-checkout")
    gateway = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(code_root / "hermes_cli" / "main.py")])
    real_kill = os.kill  # the test records os.kill below; the stand-in gateway still has to die
    try:
        signalled: list[int] = []
        monkeypatch.setattr(gateway_cli, "_is_pid_ancestor_of_current_process", lambda pid: pid == gateway.pid)
        monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: signalled.append(pid) or True)
        monkeypatch.setattr("hermes_cli.update_fleet_scope.gateway_pid_home", lambda pid: None)

        result = fleet._drain_or_signal_gateway_for_update(gateway.pid, 5.0, "hermes-gateway")
        assert (result, signalled) == ((True, [gateway.pid]) if own else (False, []))

        # The manual restart pass: a profile-mapped gateway of another checkout is neither armed for
        # a relaunch nor SIGTERMed.
        signalled.clear()
        armed: list[int] = []
        killed: list[int] = []
        proc = SimpleNamespace(pid=gateway.pid, profile="default")
        monkeypatch.setattr(gateway_cli, "_get_service_pids", lambda **k: set())
        monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda **k: [gateway.pid])
        monkeypatch.setattr(gateway_cli, "find_profile_gateway_processes", lambda **k: [proc])
        monkeypatch.setattr(gateway_cli, "_prepare_profile_gateway_update_restart",
                            lambda profile, pid: armed.append(pid) or "watcher")
        monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **k: None)
        monkeypatch.setattr(fleet.os, "kill", lambda pid, sig: killed.append(pid))
        out = fleet._GatewayRestartOutcome(
            incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
            failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set())
        fleet._restart_manual_gateways(out, 5.0)
        assert (armed, signalled, killed, out.killed_pids) == (
            ([gateway.pid], [gateway.pid], [], {gateway.pid}) if own else ([], [], [], set()))
    finally:
        real_kill(gateway.pid, signal.SIGKILL)
        gateway.wait()
