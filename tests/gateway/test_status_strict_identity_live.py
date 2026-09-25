"""Strict gateway identity against a REAL process holding ``gateway.lock``.

``hermes update`` on Windows maps live gateways to profiles through
``get_running_pid_identity_strict``. A live gateway whose ``gateway.pid`` was
unlinked still holds its runtime lock, and the lock carries the same identity
record, so the updater must identify it from the lock instead of aborting the
whole update. Every other ambiguity (garbled lock, disagreeing PID file, a lock
holder that is not a gateway) still fails closed.

The holder is a real child process that takes the lock through
``acquire_gateway_runtime_lock`` (msvcrt on Windows, flock on POSIX), so the
probe exercises the platform's actual lock semantics, not a stub.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import status

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_HOLDER = """
import os, sys, time
sys.path.insert(0, {root!r})
os.environ["HERMES_HOME"] = {home!r}
from gateway import status
assert status.acquire_gateway_runtime_lock()
status.write_pid_file()
# Print the interpreter's own PID: a Windows venv python.exe is a launcher whose
# child is the real interpreter, so Popen.pid is not the lock holder.
print(os.getpid(), flush=True)
time.sleep(120)
"""


def _spawn_lock_holder(tmp_path: Path, home: Path, *, as_gateway: bool) -> tuple[subprocess.Popen, int]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / ("hermes" if as_gateway else "holder.py")
    script.write_text(_HOLDER.format(root=str(PROJECT_ROOT), home=str(home)), encoding="utf-8")
    argv = [sys.executable, str(script), *(["gateway", "run"] if as_gateway else [])]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = proc.stdout.readline().strip()
    if not line.isdigit():
        proc.kill()
        raise RuntimeError(f"lock holder failed to start: {line!r}")
    return proc, int(line)


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def gateway_holder(tmp_path, home):
    proc, pid = _spawn_lock_holder(tmp_path, home, as_gateway=True)
    yield pid
    proc.kill()
    proc.wait(timeout=10)


@pytest.mark.spawns_gateway_lookalike
class TestStrictIdentityWithLiveLock:
    def test_missing_pid_file_resolves_from_the_lock_record(self, home, gateway_holder):
        with_pid_file = status.get_running_pid_identity_strict(home / "gateway.pid")
        (home / "gateway.pid").unlink()

        identity = status.get_running_pid_identity_strict(home / "gateway.pid")

        assert identity is not None
        assert identity[0] == gateway_holder
        assert identity == with_pid_file

    def test_updater_discovery_maps_the_profile_without_a_pid_file(self, home, gateway_holder):
        from hermes_cli.gateway import find_profile_gateway_processes

        (home / "gateway.pid").unlink()

        procs = find_profile_gateway_processes(strict=True)

        assert [(p.path, p.pid) for p in procs] == [(home, gateway_holder)]

    def test_pid_file_that_disagrees_with_the_lock_still_aborts(self, home, gateway_holder):
        record = json.loads((home / "gateway.pid").read_text(encoding="utf-8"))
        record["pid"] = gateway_holder + 1
        (home / "gateway.pid").write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(RuntimeError, match="disagree"):
            status.get_running_pid_identity_strict(home / "gateway.pid")

    def test_garbled_lock_record_without_pid_file_still_aborts(self, home, gateway_holder):
        (home / "gateway.pid").unlink()
        # The holder's lock is on a byte range (Windows) or advisory (POSIX); the record is writable.
        with open(home / "gateway.lock", "r+", encoding="utf-8") as handle:
            handle.write("not a record" + " " * 200)

        with pytest.raises(RuntimeError, match="malformed"):
            status.get_running_pid_identity_strict(home / "gateway.pid")


@pytest.mark.spawns_gateway_lookalike
def test_lock_holder_that_is_not_a_gateway_still_aborts(tmp_path, home):
    proc, _pid = _spawn_lock_holder(tmp_path, home, as_gateway=False)
    try:
        (home / "gateway.pid").unlink()
        with pytest.raises(RuntimeError, match="does not identify a live gateway"):
            status.get_running_pid_identity_strict(home / "gateway.pid")
    finally:
        proc.kill()
        proc.wait(timeout=10)
