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
import os
import queue
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path

import psutil
import pytest

from gateway import status

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The holder blocks on stdin instead of sleeping, so it exits as soon as the parent's pipe
# closes, even when pytest is killed and teardown never runs.
_HOLDER = """
import os, sys
sys.path.insert(0, {root!r})
from gateway import status
if not status.acquire_gateway_runtime_lock():
    sys.exit("could not acquire gateway.lock")
status.write_pid_file()
# Print the interpreter's own PID: a Windows venv python.exe is a launcher whose
# child is the real interpreter, so Popen.pid is not the lock holder.
print(os.getpid(), flush=True)
sys.stdin.read()
"""


def _readline(stream, timeout: float = 20.0) -> str:
    """``stream.readline()`` bounded by *timeout* (a hung child fails, not hangs)."""
    got: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: got.put(stream.readline()), daemon=True).start()
    try:
        return got.get(timeout=timeout)
    except queue.Empty:
        return ""


def _stop(proc: subprocess.Popen[str], holder: psutil.Process | None = None) -> str:
    """Kill the launcher, its descendants and the self-reported holder, reap, return stderr.

    The holder is the ``psutil.Process`` captured at spawn, so a PID recycled since then is
    refused by psutil instead of killed.
    """
    targets = [holder] if holder else []
    with suppress(psutil.Error):
        launcher = psutil.Process(proc.pid)
        targets += [*launcher.children(recursive=True), launcher]
    for target in targets:
        with suppress(psutil.Error):
            target.kill()
    _out, err = proc.communicate(timeout=10)
    return err


def _spawn_lock_holder(
    bin_dir: Path, home: Path, *, as_gateway: bool,
) -> tuple[subprocess.Popen[str], psutil.Process]:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / ("hermes" if as_gateway else "holder.py")
    script.write_text(_HOLDER.format(root=str(PROJECT_ROOT)), encoding="utf-8")
    argv = [sys.executable, str(script), *(["gateway", "run"] if as_gateway else [])]
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "HERMES_HOME": str(home)},
    )
    assert proc.stdout is not None
    line = _readline(proc.stdout).strip()
    if not line.isdigit():
        pytest.fail(f"lock holder failed to start: {line!r} {_stop(proc)}")
    return proc, psutil.Process(int(line))


def _make_home(root: Path) -> Path:
    home = root / "home"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    return home


@pytest.fixture(scope="class")
def gateway_holder(tmp_path_factory):
    """One live gateway-shaped lock holder for the class; each spawn costs about a second."""
    root = tmp_path_factory.mktemp("strict-identity")
    home = _make_home(root)
    proc, holder = _spawn_lock_holder(root / "bin", home, as_gateway=True)
    yield home, holder.pid
    _stop(proc, holder)


@pytest.fixture
def home(gateway_holder, monkeypatch):
    """The holder's home; the identity files each test mutates are restored afterwards."""
    home, _pid = gateway_holder
    monkeypatch.setenv("HERMES_HOME", str(home))
    pid_bytes = (home / "gateway.pid").read_bytes()
    lock_bytes = (home / "gateway.lock").read_bytes()
    yield home
    (home / "gateway.pid").write_bytes(pid_bytes)
    # The holder's lock is one byte at a far offset (Windows) or advisory (POSIX), so the record
    # stays writable. Overwriting without truncating is safe: a leftover tail is the spaces a
    # test wrote, which the record reader strips.
    with open(home / "gateway.lock", "r+b") as handle:
        handle.write(lock_bytes)


# The class-scoped holder spawns before the per-test live-system guard is armed; the mark
# documents the `hermes gateway run` lookalike and covers it if the fixture is ever narrowed.
@pytest.mark.spawns_gateway_lookalike
class TestStrictIdentityWithLiveLock:
    def test_missing_pid_file_resolves_from_the_lock_record(self, home, gateway_holder):
        with_pid_file = status.get_running_pid_identity_strict(home / "gateway.pid")
        (home / "gateway.pid").unlink()

        identity = status.get_running_pid_identity_strict(home / "gateway.pid")

        assert identity is not None
        assert identity[0] == gateway_holder[1]
        assert identity == with_pid_file

    def test_updater_discovery_maps_the_profile_without_a_pid_file(self, home, gateway_holder, monkeypatch):
        from hermes_cli.gateway import find_profile_gateway_processes

        monkeypatch.setattr("profiles.paths._get_default_hermes_home", lambda: home)
        monkeypatch.setattr("profiles.paths._get_profiles_root", lambda: home / "no-profiles")
        (home / "gateway.pid").unlink()

        procs = find_profile_gateway_processes(strict=True)

        assert [(Path(p.path).resolve(), p.pid) for p in procs] == [(home.resolve(), gateway_holder[1])]

    def test_pid_file_that_disagrees_with_the_lock_still_aborts(self, home, gateway_holder):
        record = json.loads((home / "gateway.pid").read_text(encoding="utf-8-sig"))
        record["pid"] = gateway_holder[1] + 1
        (home / "gateway.pid").write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(RuntimeError, match="disagree"):
            status.get_running_pid_identity_strict(home / "gateway.pid")

    def test_garbled_lock_record_without_pid_file_still_aborts(self, home):
        (home / "gateway.pid").unlink()
        with open(home / "gateway.lock", "r+b") as handle:
            handle.write(b"not a record" + b" " * 100)

        with pytest.raises(RuntimeError, match="malformed"):
            status.get_running_pid_identity_strict(home / "gateway.pid")


def test_lock_holder_that_is_not_a_gateway_still_aborts(tmp_path, monkeypatch):
    home = _make_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    proc, holder = _spawn_lock_holder(tmp_path / "bin", home, as_gateway=False)
    try:
        (home / "gateway.pid").unlink()
        with pytest.raises(RuntimeError, match="does not identify a live gateway"):
            status.get_running_pid_identity_strict(home / "gateway.pid")
    finally:
        _stop(proc, holder)
