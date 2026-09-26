"""Invariants for the detached updater launch path.

A ``/update`` triggered from chat must outlive the gateway restart it causes. Under a
supervisor the gateway's unit restart SIGKILLs every remaining member of its stop cgroup
(``KillMode=mixed``), which killed the plain ``setsid`` child after it printed ``draining``:
no exit-code file, no receipt, and ``fleet_restart_pending`` left behind — so ``hermes doctor``
kept warning "a previous hermes update did not restart running gateways" although the fleet
was on the new code.

These tests pin the launch contract (which argv is chosen), not a snapshot of the command text.
"""

from __future__ import annotations

import sys

import pytest

from gateway import slash_commands


@pytest.fixture
def supervisor(monkeypatch):
    """Pretend the gateway runs under systemd (INVOCATION_ID set in real life)."""
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda *a, **k: True)


@pytest.fixture
def no_supervisor(monkeypatch):
    monkeypatch.setattr("gateway.restart.is_gateway_supervisor_process", lambda *a, **k: False)


def _which(mapping):
    return lambda name: mapping.get(name)


def test_supervised_host_prefers_transient_unit(monkeypatch, supervisor):
    """Own cgroup → the updater survives the gateway restart that follows it."""
    monkeypatch.setattr("shutil.which", _which({"systemd-run": "/usr/bin/systemd-run",
                                                "setsid": "/usr/bin/setsid"}))
    argv = slash_commands._detached_update_argv("hermes update --gateway", now=1789157537)

    assert argv[0] == "/usr/bin/systemd-run"
    assert "--user" in argv and "--collect" in argv
    assert argv[argv.index("--unit") + 1] == "hermes-update-1789157537"
    # the payload still runs through bash, so the redirect + exit-code write stay intact
    assert argv[-3:] == ["bash", "-c", "hermes update --gateway"]


def test_supervised_host_without_systemd_run_falls_back_to_setsid(monkeypatch, supervisor):
    """No user D-Bus (older hosts) → keep the portable path instead of failing the update."""
    monkeypatch.setattr("shutil.which", _which({"setsid": "/usr/bin/setsid"}))
    argv = slash_commands._detached_update_argv("hermes update --gateway", now=1)

    assert argv[0] == "/usr/bin/setsid"
    assert "systemd-run" not in " ".join(argv)


def test_unsupervised_host_keeps_setsid(monkeypatch, no_supervisor):
    """macOS/containers: nothing tears the cgroup down, so setsid stays the default."""
    monkeypatch.setattr("shutil.which", _which({"setsid": "/usr/bin/setsid",
                                                "systemd-run": "/usr/bin/systemd-run"}))
    argv = slash_commands._detached_update_argv("hermes update --gateway", now=1)

    assert argv[0] == "/usr/bin/setsid"


@pytest.mark.skipif(sys.platform == "win32", reason="setsid/systemd-run are POSIX only")
def test_launch_never_hard_fails_without_helpers(monkeypatch, no_supervisor):
    """Neither helper installed → still start something (bash), never raise."""
    monkeypatch.setattr("shutil.which", _which({}))
    argv = slash_commands._detached_update_argv("hermes update --gateway", now=1)

    assert argv[0] == "bash"
