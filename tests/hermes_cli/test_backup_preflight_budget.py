"""The pre-flight budget must scale with the database, not a fixed clock cap (#124972).

A healthy 14.40 GB WAL state.db measured ~124 s to copy plus ~47 s to quick-check
on Windows, against the updater's fixed 180 s cap: the cap must not kill
progressing copies, must still kill wedged ones, and staging files left by a
hard kill must be reclaimed once their owner is provably gone.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "hermes_cli" / "backup_sqlite.py"

_LOAD_MODULE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("backup_sqlite_under_test", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
"""

_PLAIN_RUNNER = _LOAD_MODULE + """
import json, sys
from pathlib import Path
print(json.dumps(mod.preflight_state_db(Path(sys.argv[2]))))
"""

_STALLED_COPY_RUNNER = _LOAD_MODULE + """
import sys
from pathlib import Path
real_sqlite3 = mod.sqlite3

class StalledSource(real_sqlite3.Connection):
    # A wedged copy: the backup API never returns and never reports progress.
    def backup(self, *args, **kwargs):
        import time as real_time
        while True:
            real_time.sleep(0.05)

class StalledShim:
    SQLITE_BUSY = real_sqlite3.SQLITE_BUSY
    SQLITE_LOCKED = real_sqlite3.SQLITE_LOCKED
    @staticmethod
    def connect(*args, **kwargs):
        kwargs["factory"] = StalledSource
        return real_sqlite3.connect(*args, **kwargs)

mod.sqlite3 = StalledShim
mod.preflight_state_db(Path(sys.argv[2]), stall_seconds=2.0, copy_budget_seconds=30.0)
"""

_SLOW_BUT_PROGRESSING_RUNNER = _LOAD_MODULE + """
import json, sys
from pathlib import Path

class FakeTime:
    # Injected clock: every 256-page backup step "takes" a minute of wall time,
    # so the copy runs far past the old 180 s fixed cap with no real waiting.
    now = 0.0
    @staticmethod
    def monotonic():
        return FakeTime.now
    @staticmethod
    def sleep(_seconds):
        FakeTime.now += 0.05

real_sqlite3 = mod.sqlite3

class ProgressingSource(real_sqlite3.Connection):
    def backup(self, target, **kwargs):
        real_progress = kwargs.get("progress")
        def wrapping(status, remaining, total):
            FakeTime.now += 60.0
            if real_progress is not None:
                real_progress(status, remaining, total)
        kwargs["progress"] = wrapping
        return real_sqlite3.Connection.backup(self, target, **kwargs)

class ProgressingShim:
    SQLITE_BUSY = real_sqlite3.SQLITE_BUSY
    SQLITE_LOCKED = real_sqlite3.SQLITE_LOCKED
    @staticmethod
    def connect(*args, **kwargs):
        kwargs["factory"] = ProgressingSource
        return real_sqlite3.connect(*args, **kwargs)

mod.sqlite3 = ProgressingShim
mod.time = FakeTime
print(json.dumps(mod.preflight_state_db(
    Path(sys.argv[2]),
    stall_seconds=300.0,
    copy_budget_seconds=100000.0,
    quick_check_budget_seconds=100000.0,
)))
"""

_RECLAIM_RUNNER = _LOAD_MODULE + """
import sys
from pathlib import Path
mod._reclaim_abandoned_staging(Path(sys.argv[2]))
"""

_LIVE_OWNER_RUNNER = _LOAD_MODULE + """
import sys, time as real_time
from pathlib import Path
staging = Path(sys.argv[2]) / sys.argv[3]
staging.write_bytes(b"partial")
fd = mod._acquire_owner_marker(Path(str(staging) + ".owner"))
assert fd is not None, "could not lock the owner marker"
print("locked", flush=True)
real_time.sleep(30)
"""


def _home_with_state_db(tmp_path, pages=1):
    home = tmp_path / "home"
    home.mkdir()
    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE t (x)")
    for _ in range(pages):
        conn.execute("INSERT INTO t VALUES (zeroblob(4000))")
    conn.commit()
    conn.close()
    return home


def _run(home, runner, timeout=60):
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", runner, str(SCRIPT), str(home)],
        capture_output=True, text=True, timeout=timeout,
    )


def test_a_stalled_copy_is_aborted_by_the_watchdog_and_leaves_reclaimable_staging(tmp_path):
    home = _home_with_state_db(tmp_path)
    result = _run(home, _STALLED_COPY_RUNNER, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "no progress" in result.stderr
    partials = list(home.glob("state.db.pre-update-emergency-*.partial"))
    assert len(partials) == 1
    assert Path(str(partials[0]) + ".owner").exists()


def test_a_slow_but_progressing_copy_outlives_the_fixed_cap(tmp_path):
    home = _home_with_state_db(tmp_path, pages=3000)
    result = _run(home, _SLOW_BUT_PROGRESSING_RUNNER, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["path"]).exists()
    assert not list(home.glob("*.partial"))


def test_staging_left_by_a_killed_pre_flight_is_reclaimed_by_the_next_run(tmp_path):
    home = _home_with_state_db(tmp_path)
    killed = _run(home, _STALLED_COPY_RUNNER, timeout=30)
    assert killed.returncode != 0
    assert list(home.glob("*.partial"))
    recovered = _run(home, _PLAIN_RUNNER)
    assert recovered.returncode == 0, recovered.stderr
    payload = json.loads(recovered.stdout)
    assert Path(payload["path"]).exists()
    assert not list(home.glob("*.partial"))
    assert not list(home.glob("*.partial.owner"))


def test_reclaim_spared_a_live_owner_and_aged_markerless_leftovers(tmp_path):
    home = _home_with_state_db(tmp_path)
    staging_name = "state.db.pre-update-emergency-20260101T000000Z-live.partial"
    child = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", _LIVE_OWNER_RUNNER, str(SCRIPT), str(home), staging_name],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        assert _run(home, _RECLAIM_RUNNER).returncode == 0
        assert (home / staging_name).exists()  # the owner is alive: never reclaimed
        stale = home / "state.db.pre-update-emergency-20250101T000000Z-stale.partial"
        stale.write_bytes(b"stale")
        aged = time.time() - 2 * 3600
        os.utime(stale, (aged, aged))
        fresh = home / "state.db.pre-update-emergency-20260101T000000Z-fresh.partial"
        fresh.write_bytes(b"fresh")
        assert _run(home, _RECLAIM_RUNNER).returncode == 0
        assert not stale.exists()  # marker-less and hours old: reclaimed
        assert fresh.exists()  # marker-less but young: an old-runtime copy may still be writing it
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert _run(home, _RECLAIM_RUNNER).returncode == 0
    assert not (home / staging_name).exists()
    assert not (home / f"{staging_name}.owner").exists()


def test_success_closes_owner_before_unlinking_marker(tmp_path, monkeypatch):
    from hermes_cli import backup_sqlite as mod

    home = _home_with_state_db(tmp_path)
    real_acquire = mod._acquire_owner_marker
    real_unlink = Path.unlink
    handles = {}

    def acquire(path):
        handle = real_acquire(path)
        if handle is not None:
            handles[path] = handle
        return handle

    def windows_unlink(path, *args, **kwargs):
        # Simulate Windows' refusal to delete a file with an open owner handle.
        handle = handles.get(path)
        if handle is not None and not handle.closed:
            raise PermissionError("owner marker is still open")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(mod, "_acquire_owner_marker", acquire)
    monkeypatch.setattr(Path, "unlink", windows_unlink)
    result = mod.preflight_state_db(home)
    assert Path(result["path"]).exists()
    assert not list(home.glob("*.partial.owner"))
    assert all(handle.closed for handle in handles.values())


def test_reclaim_discovers_owner_without_partial_and_spares_live_owner(tmp_path):
    home = _home_with_state_db(tmp_path)
    abandoned = home / "state.db.pre-update-emergency-orphan.partial.owner"
    abandoned.write_bytes(b" ")
    staging_name = "state.db.pre-update-emergency-live-orphan.partial"
    child = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", _LIVE_OWNER_RUNNER,
         str(SCRIPT), str(home), staging_name],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "locked"
        (home / staging_name).unlink()
        marker = home / (staging_name + ".owner")
        assert _run(home, _RECLAIM_RUNNER).returncode == 0
        assert not abandoned.exists()
        assert marker.exists()
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert _run(home, _RECLAIM_RUNNER).returncode == 0
    assert not marker.exists()


def test_phase_budgets_scale_with_the_database_size():
    from hermes_cli.backup_sqlite import _copy_budget_seconds, _quick_check_budget_seconds

    gib = 1024 ** 3
    # Never tighter than the historical fixed cap, for any size.
    assert _copy_budget_seconds(0) >= 180
    # A 14.40 GB WAL store measured ~124 s copy + ~47 s quick_check on Windows
    # (#124972); a healthy run gets at least 2x headroom.
    assert _copy_budget_seconds(14.4 * gib) >= 2 * 124
    assert _quick_check_budget_seconds(14.4 * gib) >= 2 * 47
    # Monotone in size; small stores stay at the old fixed-cap behavior.
    assert _copy_budget_seconds(14.4 * gib) >= _copy_budget_seconds(gib)
    assert _copy_budget_seconds(50 * 1024 * 1024) <= 180 + 5
