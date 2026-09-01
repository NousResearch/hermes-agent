"""Cross-process admission for full structural FTS rebuilds (PR #93200 class).

Several independent Hermes processes routinely share one state.db (gateway,
Desktop's ``hermes serve`` backend, CLI sessions, the TUI slash worker). Two
of them detecting FTS corruption at once each ran the full FTS5 'rebuild' on
the same file in parallel, colliding on write and structurally corrupting
state.db (two documented production incidents, 2026-08-15 and 2026-08-23).

The fix: every full structural rebuild entry point — ``rebuild_fts()``, the
``_init_schema`` trigger-repair rebuilds, and ``_recover_stale_fts`` — admits
through one cross-process file lock (``fts_rebuild_admission`` in
hermes_state_common) and FAILS CLOSED: a process that cannot acquire the
authority defers the rebuild instead of racing the holder. These tests use
real spawned processes holding the real lock file, per the review contract
on PR #93200 — the bug is cross-process ownership, so monkeypatched helpers
prove nothing.
"""

import contextlib
import errno
import subprocess
import sqlite3
import sys
import time
from pathlib import Path

import pytest

import hermes_state_common
from hermes_state import FTS_STALE_KEY, SessionDB, _FTS_TRIGGERS

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX flock child-process harness"
)


_HOLD_LOCK_SCRIPT = """
import sys, time, fcntl, pathlib
lock_path = pathlib.Path({lock!r})
handle = lock_path.open("a+b")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep({hold})
"""


def _lock_file(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".fts_rebuild.lock")


@contextlib.contextmanager
def _rebuild_lock_held_by_other_process(db_path: Path, hold_seconds: float = 30.0):
    """Hold the FTS rebuild authority for *db_path* in a real child process."""
    script = _HOLD_LOCK_SCRIPT.format(
        lock=str(_lock_file(db_path)), hold=hold_seconds
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout.readline().strip() == "locked"
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _fts_docsize_count(db_path: Path) -> int:
    raw = sqlite3.connect(str(db_path))
    try:
        return raw.execute("SELECT count(*) FROM messages_fts_docsize").fetchone()[0]
    finally:
        raw.close()


def _base_fts_triggers(db_path: Path) -> set:
    raw = sqlite3.connect(str(db_path))
    try:
        rows = raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            f"AND name IN ({','.join('?' for _ in _FTS_TRIGGERS)})",
            _FTS_TRIGGERS,
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        raw.close()


def _meta_value(db_path: Path, key: str):
    raw = sqlite3.connect(str(db_path))
    try:
        row = raw.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row[0]
    finally:
        raw.close()


@pytest.fixture
def fast_timeout(monkeypatch):
    monkeypatch.setattr(
        hermes_state_common, "_FTS_REBUILD_LOCK_TIMEOUT_SECONDS", 0.5
    )


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    if not d._fts_enabled:
        d.close()
        pytest.skip("FTS5 unavailable in this build")
    d.create_session("s1", source="test")
    for i in range(5):
        d.append_message("s1", "user", f"hello world {i}")
    yield d
    try:
        d.close()
    except Exception:
        pass


class TestRebuildFtsAdmission:
    def test_rebuild_defers_while_another_process_holds_authority(
        self, db, fast_timeout
    ):
        """Fail closed: the contender must NOT rebuild while the lock is held."""
        with _rebuild_lock_held_by_other_process(db.db_path):
            assert db.rebuild_fts() == 0

    def test_rebuild_proceeds_after_holder_releases(self, db, fast_timeout):
        with _rebuild_lock_held_by_other_process(db.db_path):
            assert db.rebuild_fts() == 0
        # Holder killed on context exit → kernel drops the flock → the next
        # caller acquires the authority and the rebuild really runs.
        assert db.rebuild_fts() >= 1

    def test_rebuild_waits_out_a_short_holder(self, db, monkeypatch):
        """A holder that releases within the bounded wait does not cause deferral."""
        monkeypatch.setattr(
            hermes_state_common, "_FTS_REBUILD_LOCK_TIMEOUT_SECONDS", 10.0
        )
        with _rebuild_lock_held_by_other_process(db.db_path, hold_seconds=1.0):
            # Child exits after 1s; deadline is 10s — this must acquire and rebuild.
            assert db.rebuild_fts() >= 1

    def test_admission_yields_true_for_pathless_db(self):
        """In-memory / pathless stores have no cross-process surface."""
        with hermes_state_common.fts_rebuild_admission(None) as admitted:
            assert admitted is True


class TestSchemaPathAdmission:
    def test_startup_trigger_repair_defers_and_fails_closed(
        self, tmp_path, fast_timeout
    ):
        """The _init_schema trigger-repair rebuild is covered by the SAME
        authority — deferral must leave FTS detached with the durable stale
        breadcrumb, never triggers installed over an unrebuilt index gap."""
        db_path = tmp_path / "state.db"
        d = SessionDB(db_path=db_path)
        if not d._fts_enabled:
            d.close()
            pytest.skip("FTS5 unavailable in this build")
        d.create_session("s1", source="test")
        d.append_message("s1", "user", "hello schema path")
        d.close()

        # Drop one sync trigger out-of-band: next open takes the
        # triggers_need_repair branch in _init_schema.
        raw = sqlite3.connect(str(db_path))
        raw.execute(f"DROP TRIGGER IF EXISTS {sorted(_FTS_TRIGGERS)[0]}")
        raw.commit()
        raw.close()

        with _rebuild_lock_held_by_other_process(db_path):
            d2 = SessionDB(db_path=db_path)
            try:
                assert d2._fts_enabled is False
            finally:
                d2.close()

        # Durable state: stale breadcrumb set, no live sync triggers.
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

    def test_stale_recovery_defers_then_succeeds_after_release(
        self, tmp_path, fast_timeout
    ):
        """_recover_stale_fts defers under contention and completes once the
        authority is free (next open)."""
        db_path = tmp_path / "state.db"
        d = SessionDB(db_path=db_path)
        if not d._fts_enabled:
            d.close()
            pytest.skip("FTS5 unavailable in this build")
        d.create_session("s1", source="test")
        d.append_message("s1", "user", "hello recovery path")
        d.close()

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (FTS_STALE_KEY,),
        )
        for trig in _FTS_TRIGGERS:
            raw.execute(f"DROP TRIGGER IF EXISTS {trig}")
        raw.commit()
        raw.close()

        with _rebuild_lock_held_by_other_process(db_path):
            d2 = SessionDB(db_path=db_path)
            try:
                assert d2._fts_enabled is False
            finally:
                d2.close()
        # Deferred: breadcrumb still present, recovery not performed.
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"

        d3 = SessionDB(db_path=db_path)
        try:
            assert d3._fts_enabled is True
        finally:
            d3.close()
        # Recovered: breadcrumb cleared, triggers restored.
        assert _meta_value(db_path, FTS_STALE_KEY) is None
        assert _base_fts_triggers(db_path) == set(_FTS_TRIGGERS)


# ---------------------------------------------------------------------------
# Orphaned-fd staleness break (issue #100108).
#
# flock belongs to the open file DESCRIPTION, which fork() duplicates into
# children. A holder that forks (multiprocessing worker, daemonized helper)
# and then crashes leaves the flock held by the child forever — the kernel's
# holder-death release never fires, and every contender deferred forever
# ("FTS rebuild lock ... held by another process for more than 120s").
# The fix records the acquirer's pid + start time under the lock; a contender
# that times out breaks the lock ONLY when that recorded holder is provably
# dead, and fails closed on any indeterminate state.
# ---------------------------------------------------------------------------

_ORPHANING_HOLDER_SCRIPT = """
import os, sys, time
sys.path.insert(0, {repo!r})
import hermes_state_common

admission = hermes_state_common.fts_rebuild_admission({db!r})
admitted = admission.__enter__()
assert admitted is True
pid = os.fork()
if pid == 0:
    # Forked child: shares the lock fd's open file description. Sleep far
    # beyond the test, never releasing.
    time.sleep(600)
    os._exit(0)
print("child", pid, flush=True)
# Crash WITHOUT releasing (no __exit__): simulates the production holder
# dying mid-rebuild after having forked.
os._exit(1)
"""


@contextlib.contextmanager
def _orphaned_fork_holder(db_path: Path):
    """Real #100108 shape: acquirer records itself, forks, dies."""
    import os
    import signal

    script = _ORPHANING_HOLDER_SCRIPT.format(
        repo=str(Path(hermes_state_common.__file__).parent), db=str(db_path)
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    line = proc.stdout.readline().strip()
    assert line.startswith("child ")
    grandchild = int(line.split()[1])
    proc.wait(timeout=10)  # the acquirer is now dead; grandchild holds the fd
    try:
        yield grandchild
    finally:
        with contextlib.suppress(OSError):
            os.kill(grandchild, signal.SIGKILL)


class TestOrphanedHolderStalenessBreak:
    @pytest.mark.live_system_guard_bypass
    def test_rebuild_breaks_lock_of_dead_forker(self, db, fast_timeout):
        """The #100108 repro: recorded holder dead, forked child holds the
        flock. The contender must break the orphaned lock and rebuild."""
        with _orphaned_fork_holder(db.db_path):
            assert db.rebuild_fts() >= 1

    def test_admission_still_fails_closed_for_live_unrecorded_holder(
        self, db, fast_timeout
    ):
        """A live holder that wrote no record (pre-fix build, non-Hermes
        tool) is indeterminate — must defer, never break."""
        with _rebuild_lock_held_by_other_process(db.db_path):
            assert db.rebuild_fts() == 0

    def test_admission_fails_closed_for_live_recorded_holder(
        self, db, fast_timeout, monkeypatch
    ):
        """A record naming a live pid must defer even after timeout."""
        import json
        import os

        lock = _lock_file(db.db_path)
        with _rebuild_lock_held_by_other_process(db.db_path) as proc:
            record = {
                "pid": proc.pid,
                "start_ticks": hermes_state_common._proc_start_ticks(proc.pid),
                "acquired_at": 0,
            }
            lock.write_bytes(json.dumps(record).encode())
            assert db.rebuild_fts() == 0

    def test_holder_record_cleared_on_normal_release(self, tmp_path):
        lock = tmp_path / "x.db.fts_rebuild.lock"
        with hermes_state_common.fts_rebuild_admission(tmp_path / "x.db") as ok:
            assert ok is True
            assert b"pid" in lock.read_bytes()
        assert lock.read_bytes() == b""

    @pytest.mark.live_system_guard_bypass
    def test_repair_lock_breaks_orphaned_holder(self, tmp_path, monkeypatch):
        """_cross_process_repair_lock shares the same staleness break."""
        import hermes_state

        monkeypatch.setattr(hermes_state, "_REPAIR_LOCK_TIMEOUT_SECONDS", 0.5)
        db_path = tmp_path / "state.db"
        db_path.touch()

        script = """
import os, sys, time
sys.path.insert(0, {repo!r})
from pathlib import Path
import hermes_state

lock_cm = hermes_state._cross_process_repair_lock(Path({db!r}))
assert lock_cm.__enter__() is True
pid = os.fork()
if pid == 0:
    time.sleep(600)
    os._exit(0)
print("child", pid, flush=True)
os._exit(1)
""".format(repo=str(Path(hermes_state_common.__file__).parent), db=str(db_path))
        import os
        import signal

        proc = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
        )
        grandchild = int(proc.stdout.readline().strip().split()[1])
        proc.wait(timeout=10)
        try:
            import hermes_state as hs

            with hs._cross_process_repair_lock(db_path) as holding:
                assert holding is True
        finally:
            with contextlib.suppress(OSError):
                os.kill(grandchild, signal.SIGKILL)


class TestNonContentionErrnoFailsFast:
    def test_non_contention_oserror_does_not_wait_out_timeout(
        self, tmp_path, monkeypatch
    ):
        import fcntl

        monkeypatch.setattr(
            hermes_state_common, "_FTS_REBUILD_LOCK_TIMEOUT_SECONDS", 30.0
        )

        def _flock(*_args, **_kwargs):
            raise OSError(getattr(errno, "ESTALE", errno.EIO), "stale handle")

        monkeypatch.setattr(fcntl, "flock", _flock)
        db_path = tmp_path / "state.db"
        t0 = time.monotonic()
        with hermes_state_common.fts_rebuild_admission(db_path) as admitted:
            assert admitted is False
        assert time.monotonic() - t0 < 2.0

    def test_retry_deferred_fts_recovery_rebuilds_same_instance(
        self, tmp_path, monkeypatch
    ):
        """Gateway-shaped: same SessionDB stays open and retries after deferral."""
        import hermes_state_schema

        monkeypatch.setattr(hermes_state_schema, "_FTS_STALE_RETRY_SECONDS", 0.0)
        db_path = tmp_path / "state.db"
        d = SessionDB(db_path=db_path)
        if not d._fts_enabled:
            d.close()
            pytest.skip("FTS5 unavailable in this build")
        d.create_session("s1", source="test")
        d.append_message("s1", "user", "hello recovery path")
        d.close()

        raw = sqlite3.connect(str(db_path))
        raw.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, '1')",
            (FTS_STALE_KEY,),
        )
        for trig in _FTS_TRIGGERS:
            raw.execute(f"DROP TRIGGER IF EXISTS {trig}")
        raw.commit()
        raw.close()

        holders = [(4242, str(db_path))]
        monkeypatch.setattr(
            SessionDB, "_foreign_state_db_holders", lambda self: list(holders)
        )
        d2 = SessionDB(db_path=db_path)
        try:
            assert d2._fts_stale is True
            d2._fts_stale_retry_after = 0.0
            assert d2.retry_deferred_fts_recovery() is False
            holders.clear()
            d2._fts_stale_retry_after = 0.0
            assert d2.retry_deferred_fts_recovery() is True
            assert d2._fts_stale is False
        finally:
            d2.close()
        assert _meta_value(db_path, FTS_STALE_KEY) is None
        assert _base_fts_triggers(db_path) == set(_FTS_TRIGGERS)
