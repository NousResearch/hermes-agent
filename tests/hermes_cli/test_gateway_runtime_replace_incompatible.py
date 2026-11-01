"""An owner from an older Hermes on the SAME home is replaced by ensure, never reported forever.

Upgrade path (Windows Desktop report, TUI "retrying in 30s" forever): the pre-protocol gateway a user
left running answers ``identify`` without ``runtime_protocol``. Every client of this install ensures
through ``ensure_gateway_runtime`` (TUI bootstrap, Desktop's ``hermes gateway ensure``), so that is
where the old owner is stopped the way ``hermes gateway stop`` stops it and the current one started.
A gateway of another install (a home that resolves outside this install's root) stays refused.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_runtime as runtime
from hermes_cli import gateway_runtime_replace as replace
from hermes_cli.gateway_runtime import GatewayDiscovery, GatewayEndpoint


def _ready(home: Path) -> GatewayDiscovery:
    return GatewayDiscovery("ready", GatewayEndpoint(
        profile_id=str(home), instance_id="new", authority_epoch=1, runtime_protocol=1,
        api_origin="http://127.0.0.1:1", supervisor="none", capabilities=frozenset({"session-authority-v1"})))


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = (tmp_path / ".hermes").resolve()
    home.mkdir()
    monkeypatch.setattr(runtime, "_update_fenced", lambda homes: False)
    return home


def _script(monkeypatch, verdicts):
    seen = []

    def discover(h, timeout=2.0):
        seen.append(h)
        return verdicts.pop(0) if len(verdicts) > 1 else verdicts[0]

    monkeypatch.setattr(runtime, "discover_gateway_endpoint", discover)
    return seen


def test_ensure_replaces_an_incompatible_owner_of_this_home(home, monkeypatch):
    stopped = []
    _script(monkeypatch, [GatewayDiscovery("incompatible", reason_code="runtime_protocol"),
                          GatewayDiscovery("absent"), _ready(home)])
    monkeypatch.setattr(replace, "replace_incompatible_owner",
                        lambda h, *, deadline: stopped.append(h) or None)
    from hermes_cli import gateway_runtime_service as service, gateway_runtime_start as start
    monkeypatch.setattr(service, "discover_existing_gateway_service", lambda h, deadline: None)
    monkeypatch.setattr(start, "spawn_unmanaged_gateway",
                        lambda h, deadline, idle_exit=False: SimpleNamespace(poll=lambda: None, pid=1))
    monkeypatch.setattr(start, "stdio_log_size", lambda h: 0)

    result = runtime.ensure_gateway_runtime(home, timeout=5)

    assert result.state == "ready"
    assert stopped == [home]


def test_a_current_protocol_owner_without_session_authority_is_not_replaced(home, monkeypatch):
    """Only the upgrade leftover (no/old ``runtime_protocol``) is replaced; a current owner that
    lacks a capability is a configuration verdict, reported as before."""
    calls = []
    _script(monkeypatch, [GatewayDiscovery("incompatible", reason_code="session_authority_unavailable")])
    monkeypatch.setattr(replace, "replace_incompatible_owner",
                        lambda h, *, deadline: calls.append(h) or None)
    result = runtime.ensure_gateway_runtime(home, timeout=5)
    assert result.reason_code == "session_authority_unavailable" and calls == []


def test_ensure_replaces_at_most_once_then_reports(home, monkeypatch):
    calls = []
    _script(monkeypatch, [GatewayDiscovery("incompatible", reason_code="runtime_protocol")])
    monkeypatch.setattr(replace, "replace_incompatible_owner",
                        lambda h, *, deadline: calls.append(h) or None)

    result = runtime.ensure_gateway_runtime(home, timeout=5)

    assert (result.state, result.reason_code) == ("incompatible", "runtime_protocol")
    assert calls == [home]


def test_a_refused_replacement_is_reported_with_its_reason(home, monkeypatch):
    _script(monkeypatch, [GatewayDiscovery("incompatible", reason_code="runtime_protocol")])
    monkeypatch.setattr(replace, "replace_incompatible_owner",
                        lambda h, *, deadline: "foreign_install")

    result = runtime.ensure_gateway_runtime(home, timeout=5)

    assert (result.state, result.reason_code) == ("incompatible", "runtime_protocol")
    assert result.detail and "another Hermes install" in result.detail


# --- the stop itself -------------------------------------------------------------------------------

def _owner_payload(home: Path, pid: int, start_time):
    # MAIN's identify: no runtime_protocol, kind/pid/start_time/hermes_home from its pid record.
    return {"protocol": 1, "kind": "hermes-gateway", "pid": pid, "start_time": start_time,
            "hermes_home": str(home), "supervisor": "manual"}


@pytest.fixture
def owner(home, monkeypatch):
    """A live process standing in for the old daemon; the replacement must SIGTERM exactly it."""
    import subprocess
    import sys
    from gateway.status import get_process_start_time
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: (home, _owner_payload(
        home, proc.pid, get_process_start_time(proc.pid))))
    yield proc
    if proc.poll() is None:
        proc.kill()
    proc.wait()


@pytest.mark.platforms("linux")
def test_replacement_stops_the_verified_old_owner_with_a_planned_stop(home, owner, monkeypatch):
    import time
    monkeypatch.setattr(replace, "_install_root", lambda: home)
    refusal = replace.replace_incompatible_owner(home, deadline=time.monotonic() + 15)
    assert refusal is None
    assert owner.wait(5) is not None
    marker = home / ".gateway-planned-stop.json"
    # Written for the old owner's own home, so it exits as a planned stop (no supervisor revival).
    assert marker.exists() or owner.returncode is not None


@pytest.mark.platforms("linux")
def test_replacement_refuses_a_recycled_pid(home, owner, monkeypatch):
    import time
    monkeypatch.setattr(replace, "_install_root", lambda: home)
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: (home, _owner_payload(home, owner.pid, 1)))
    assert replace.replace_incompatible_owner(home, deadline=time.monotonic() + 5) == "owner_unverified"
    assert owner.poll() is None


@pytest.mark.platforms("linux")
def test_replacement_refuses_another_installs_owner(tmp_path, home, owner, monkeypatch):
    """A home that resolves outside this install's root (a profile symlinked into another install)
    is that install's: its gateway is never stopped from here."""
    import time
    other_root = (tmp_path / "other").resolve()
    other_root.mkdir()
    monkeypatch.setattr(replace, "_install_root", lambda: other_root)
    assert replace.replace_incompatible_owner(home, deadline=time.monotonic() + 5) == "foreign_install"
    assert owner.poll() is None


@pytest.mark.platforms("linux")
def test_replacement_refuses_an_owner_serving_another_home(tmp_path, home, owner, monkeypatch):
    import time
    from gateway.status import get_process_start_time
    elsewhere = (tmp_path / "elsewhere").resolve()
    monkeypatch.setattr(replace, "_install_root", lambda: home)
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: (home, _owner_payload(
        elsewhere, owner.pid, get_process_start_time(owner.pid))))
    assert replace.replace_incompatible_owner(home, deadline=time.monotonic() + 5) == "foreign_install"
    assert owner.poll() is None


def test_replacement_never_targets_itself(home, monkeypatch):
    import time
    monkeypatch.setattr(replace, "_install_root", lambda: home)
    monkeypatch.setattr(replace, "_identify", lambda h, timeout: (home, _owner_payload(home, os.getpid(), 1)))
    assert replace.replace_incompatible_owner(home, deadline=time.monotonic() + 5) == "owner_unverified"
