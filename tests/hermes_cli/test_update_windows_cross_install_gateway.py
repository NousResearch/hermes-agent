"""``hermes update`` on Windows pauses only its own install's gateways (#124659).

The host-wide gateway scan sees every install's gateway. Each process below carries the live
environment a gateway of one install has; ownership is read from it by the same home scope the
POSIX fleet restart uses, so a gateway of another install is never stopped and never suppresses
this install's cold start.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_cli.main_install_repair as main_install_repair
from gateway import process_discovery
from gateway import windows_service as gateway_windows
from hermes_cli import main as cli_main
from hermes_cli import update_cmd
from hermes_cli import update_cmd_windows


@pytest.fixture
def host_gateways(tmp_path):
    """``{name: pid}`` of one long-lived process per install on the host."""
    other = tmp_path / "other-user"
    installs = {
        "own": {"HERMES_HOME": os.environ["HERMES_HOME"]},
        "foreign": {"HERMES_HOME": str(other / ".hermes")},
        # Another install at its platform default home: no HERMES_HOME in its environment.
        "foreign_default": {"HOME": str(other), "USERPROFILE": str(other),
                            "LOCALAPPDATA": str(other / "AppData" / "Local")},
    }
    base = {k: v for k, v in os.environ.items()
            if k not in ("HERMES_HOME", "HOME", "USERPROFILE", "LOCALAPPDATA")}
    procs = {
        name: subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                               env={**base, **extra}, stdin=subprocess.DEVNULL)
        for name, extra in installs.items()
    }
    try:
        yield {name: proc.pid for name, proc in procs.items()}
    finally:
        for proc in procs.values():
            proc.kill()
            proc.wait()


def test_update_pause_takes_only_this_installs_gateways(monkeypatch, host_gateways):
    monkeypatch.setattr(process_discovery, "find_profile_gateway_processes", lambda **_kw: [])
    monkeypatch.setattr(process_discovery, "find_windows_gateway_services", lambda **_kw: [])
    monkeypatch.setattr(process_discovery, "find_gateway_pids", lambda **_kw: list(host_gateways.values()))

    running_pids = update_cmd_windows._discover_windows_gateways()[3]

    assert running_pids == [host_gateways["own"]]


def test_foreign_gateways_do_not_suppress_cold_start(monkeypatch, host_gateways):
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: False)
    monkeypatch.setattr(process_discovery, "find_gateway_pids",
                        lambda **_kw: [host_gateways["foreign"], host_gateways["foreign_default"]])
    spawned = []
    monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda: spawned.append(4242) or 4242)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda *_a, **_kw: [4242])
    monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None)

    assert update_cmd_windows._cold_start_windows_gateway_after_update({"attested_generation": None})
    assert spawned == [4242]
