"""``gateway start --all`` / ``restart --all`` mean "the ONE host multiplexer", not "every gateway".

Base behaviour: ``--all`` SIGTERMed every gateway process on the host — including a multiplexer
serving other profiles — and started a single gateway in its place, so a per-profile command caused
a host-wide outage that left exactly one profile served.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import host_rendezvous as hr
from hermes_cli import gateway as gw


@pytest.fixture
def host_owner(tmp_path, monkeypatch):
    """A REAL live process published as the host gateway serving three profiles."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    record = hr.HostRecord(
        role=hr.ROLE_GATEWAY, pid=child.pid, create_time=hr.process_create_time(child.pid),
        host="", port=None, protocol_version=hr.HOST_PROTOCOL_VERSION, token_fingerprint="",
        profiles=("default", "ops", "coder"), updated_at="", home=str(tmp_path / "root"))
    path = hr.record_path(hr.ROLE_GATEWAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json()), encoding="utf-8")
    try:
        yield SimpleNamespace(pid=child.pid, home=tmp_path / "root")
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_start_all_never_sweeps_a_live_host_multiplexer(host_owner, monkeypatch, capsys):
    def _never(*a, **k):
        raise AssertionError("start --all SIGTERMed the host multiplexer")

    monkeypatch.setattr(gw, "kill_gateway_processes", _never)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda verb: False)
    monkeypatch.setattr(gw, "_service_backend", lambda: None)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda **k: [])

    gw._cmd_start(SimpleNamespace(system=False, all=True, force=False))

    out = capsys.readouterr().out
    assert "already running" in out
    # The served set survives the verb: all three profiles are still reported as served.
    for profile in ("default", "ops", "coder"):
        assert profile in out


def test_restart_all_refuses_to_sweep_a_host_multiplexer_owned_by_another_profile(
        host_owner, monkeypatch, tmp_path):
    def _never(*a, **k):
        raise AssertionError("restart --all SIGTERMed another profile's host multiplexer")

    monkeypatch.setattr(gw, "kill_gateway_processes", _never)
    monkeypatch.setattr(gw, "_stop_installed_service", lambda system: False)
    monkeypatch.setattr("gateway.status._get_process_hermes_home",
                        lambda: Path(tmp_path / "root" / "profiles" / "ops"))

    with pytest.raises(SystemExit) as exc:
        gw._restart_all(system=False)
    assert exc.value.code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE
