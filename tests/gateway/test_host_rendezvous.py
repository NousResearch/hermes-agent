"""Host-wide singleton invariants (multiplex-only): one lock per host, staleness is proved."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gateway import host_rendezvous as hr

_CHILD = """
import json, os, sys, time
sys.path.insert(0, {tree!r})
from gateway.status import acquire_gateway_runtime_lock
from gateway import host_rendezvous as hr
print(json.dumps({{
    "per_home": acquire_gateway_runtime_lock(),
    "host": hr.acquire_host_lock(hr.ROLE_GATEWAY),
}}), flush=True)
time.sleep(60)
"""


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    yield tmp_path
    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.release_host_lock(hr.ROLE_SERVE)


def test_two_homes_take_their_own_per_home_lock_but_only_one_host_lock(host_dir, monkeypatch):
    """The per-home lock is per HERMES_HOME (N profiles = N locks); the host lock is not.

    This is the whole point of the host layer: before it, a second profile's gateway took its
    own ``gateway.lock`` and nothing on the machine noticed.
    """
    tree = str(Path(__file__).resolve().parents[2])
    home_a, home_b = host_dir / "home_a", host_dir / "home_b"
    for home in (home_a, home_b):
        home.mkdir()

    env = {**os.environ, "HERMES_HOME": str(home_a), "HERMES_GATEWAY_LOCK_DIR": str(host_dir / "locks")}
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD.format(tree=tree)], env=env, cwd=tree,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert child.stdout is not None
    try:
        deadline = time.time() + 60
        line = ""
        while not line.strip() and time.time() < deadline:
            line = child.stdout.readline()
        first = json.loads(line)
        assert first == {"per_home": True, "host": True}

        monkeypatch.setenv("HERMES_HOME", str(home_b))
        from gateway import status

        assert status.acquire_gateway_runtime_lock() is True, "second home must get its OWN lock"
        assert hr.acquire_host_lock(hr.ROLE_GATEWAY) is False, "host lock is not per-home"
    finally:
        child.kill()
        child.wait(timeout=10)
        from gateway import status

        status.release_gateway_runtime_lock()


@pytest.mark.parametrize(
    "pid,create_time",
    [(2**22 - 1, 1.0), (os.getpid(), 1.0)],
    ids=["dead-pid", "creation-time-mismatch"],
)
def test_stale_record_is_never_attachable(host_dir, pid, create_time):
    """A dead PID and a live PID from another incarnation are both stale — attaching to either
    dials whatever now owns that port."""
    record = hr.HostRecord(
        role=hr.ROLE_SERVE, pid=pid, create_time=create_time, host="127.0.0.1", port=9119,
        protocol_version=hr.HOST_PROTOCOL_VERSION, token_fingerprint="", profiles=("default",),
        updated_at="2026-01-01T00:00:00+00:00")
    hr.record_path(hr.ROLE_SERVE).parent.mkdir(parents=True, exist_ok=True)
    hr.record_path(hr.ROLE_SERVE).write_text(json.dumps(record.to_json()), encoding="utf-8")

    assert hr.record_is_stale(record) is True
    assert hr.read_record(hr.ROLE_SERVE) is None
    assert hr.read_record(hr.ROLE_SERVE, include_stale=True) is not None
