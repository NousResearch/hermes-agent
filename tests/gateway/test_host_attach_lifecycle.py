"""The gateway lifecycle verbs mean "the ONE host process", not "this profile's gateway".

Invariants: a second ``gateway run`` for a profile the host process already serves ATTACHES (exit 0,
nothing spawned); one it does not serve yet triggers a rescan and then attaches; and the DEFAULT
profile is reported as served by the host multiplexer (the predicate every lifecycle guard used to
be hard-False on).
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import host_attach, host_rendezvous as hr
from gateway import run as gateway_run


@pytest.fixture
def owner_pid(tmp_path, monkeypatch):
    """A REAL live process standing in for the host gateway (liveness is proved, not stubbed)."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield child.pid
    finally:
        child.terminate()
        child.wait(timeout=10)


def _publish(pid: int, home: Path, profiles: tuple[str, ...]) -> None:
    record = hr.HostRecord(
        role=hr.ROLE_GATEWAY, pid=pid, create_time=hr.process_create_time(pid), host="",
        port=None, protocol_version=hr.HOST_PROTOCOL_VERSION, token_fingerprint="",
        profiles=profiles, updated_at="", home=str(home))
    path = hr.record_path(hr.ROLE_GATEWAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json()), encoding="utf-8")


def test_run_for_a_served_profile_attaches_and_spawns_nothing(tmp_path, monkeypatch, owner_pid):
    owner_home = tmp_path / "root"
    ours = tmp_path / "root" / "profiles" / "other"
    _publish(owner_pid, owner_home, ("default", "other"))
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: ours)

    def _never(*a, **k):
        raise AssertionError("a second gateway was started for an already-served profile")

    monkeypatch.setattr(gateway_run, "_start_gateway_replace_existing_instance", _never)

    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is True


def test_run_for_an_unserved_profile_rescans_then_attaches(tmp_path, monkeypatch, owner_pid):
    owner_home = tmp_path / "root"
    ours = tmp_path / "root" / "profiles" / "other"
    _publish(owner_pid, owner_home, ("default",))
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: ours)
    monkeypatch.setattr(host_attach, "ATTACH_CHANNEL_WAIT_S", 0.0)
    asked: list[Path] = []

    def _rescan(home, *, timeout=8.0):
        asked.append(Path(home))
        return {"multiplex": True, "served_profiles": ["default", "other"]}

    monkeypatch.setattr("gateway.control_socket.rescan_gateway_profiles", _rescan)

    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is True
    assert asked == [owner_home], "the rescan must reach the OWNER's home, not ours"


def test_host_gateway_refuses_when_it_will_not_serve_the_profile(tmp_path, monkeypatch, owner_pid):
    _publish(owner_pid, tmp_path / "root", ("default",))
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: tmp_path / "root" / "profiles" / "other")
    monkeypatch.setattr(host_attach, "ATTACH_CHANNEL_WAIT_S", 0.0)
    monkeypatch.setattr("gateway.control_socket.rescan_gateway_profiles",
                        lambda home, timeout=8.0: {"multiplex": False})

    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is False


def test_default_profile_is_served_by_the_host_multiplexer(tmp_path, owner_pid):
    _publish(owner_pid, tmp_path / "root", ("default", "other"))
    assert host_attach.host_gateway_serving("default") is not None
