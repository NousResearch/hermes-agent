"""Regression tests: cron transient writers must not rotate the shared state.db WAL generation (#109824).

Topology (mirrors production):

* A long-lived "gateway" process holds a registry-owned writer on state.db
  (``hermes_state_registry.acquire``) for its whole life.
* Every 30 minutes a standalone cron process (``hermes cron run``) opens the
  SAME state.db through the registry, writes job/session rows, and releases —
  its final ``close()`` runs SQLite's implicit close-time checkpoint when it is
  the last OS connection (gateway down / mid-restart: the issue's restart
  cascade).

On a WAL-reset-vulnerable SQLite build (3.7.0-3.51.2, the 0.21.2 era) that
close-time checkpoint can REPLACE the ``state.db-wal`` inode, orphaning every
live holder's fds to the now-deleted generation → "retired-wal"/"deleted
state.db-wal" fatal. These tests lock the invariant: while a live gateway
holds its handle the ``-wal`` inode must never change, no
``*.retired-wal-*`` capture directory may appear, and after N cron cycles the
gateway still writes and the database passes ``PRAGMA integrity_check``.
"""

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hermes_state
import hermes_state_wal
from hermes_state import SessionDB
from tests.hermes_state._wal_generation_harness import integrity_ok_path

REPO_ROOT = os.path.dirname(os.path.abspath(hermes_state.__file__))
_CHILD_PYTHON = sys.executable
_CHILD_TIMEOUT_S = 30.0

#: Cron fires every 30 minutes; the test compresses the cadence into back-to-back
#: cycles (wall-clock interval is irrelevant to the invariant — what matters is the
#: repeated transient open/write/last-connection-close topology).
CRON_CYCLES = 4


# ── The long-lived "gateway" writer child: registry-owned handle for its whole life. ──
_GATEWAY_CHILD = textwrap.dedent(
    """
    import gc, json, os, sys
    from pathlib import Path
    repo, hermes_home, db_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, repo)
    os.environ["HERMES_HOME"] = hermes_home
    import hermes_state_wal
    if hermes_state_wal.is_sqlite_wal_reset_vulnerable():
        hermes_state_wal.is_sqlite_wal_reset_vulnerable = lambda version_info=None: False
    hermes_state_wal.resolve_journal_mode = lambda: "wal"
    from hermes_state_registry import acquire

    def emit(**e):
        sys.stdout.write(json.dumps(e) + "\\n"); sys.stdout.flush()

    db = acquire(Path(db_path))
    if not db._wal_active:
        emit(event="skip"); sys.exit(3)
    for sid in ("gw-0", "gw-1"):
        db.create_session(sid, "cli")
        db.append_message(sid, role="user", content="seed")
    db._conn.execute("PRAGMA wal_autocheckpoint=0")  # leave frames in the WAL, like N-profile startup
    for sid in ("gw-0", "gw-1"):
        db.append_message(sid, role="assistant", content="uncheckpointed " + "x" * 3000)
    emit(event="ready")
    turn = 0
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "write":
            turn += 1
            db.append_message("gw-0", role="user", content=f"post-cycle turn {turn}")
            emit(event="write", ok=True)
        elif cmd == "probe":
            wal = Path(db_path + "-wal")
            ino = wal.stat().st_ino if wal.exists() else None
            retired = [p.name for p in Path(hermes_home).iterdir()
                       if isinstance(p, Path) and ".retired-wal-" in p.name]
            count = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            emit(event="probe", wal_ino=ino, retired_wal_dirs=retired, messages=count)
        elif cmd == "close":
            db.close()
            del db
            gc.collect()
            emit(event="closed")
        elif cmd == "quit":
            break
    """
)


# ── One transient cron child: exactly what cron/scheduler.py::_open_cron_session_db does. ──
_CRON_CHILD = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path
    repo, hermes_home, db_path, gen = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sys.path.insert(0, repo)
    os.environ["HERMES_HOME"] = hermes_home
    import hermes_state_wal
    if hermes_state_wal.is_sqlite_wal_reset_vulnerable():
        hermes_state_wal.is_sqlite_wal_reset_vulnerable = lambda version_info=None: False
    hermes_state_wal.resolve_journal_mode = lambda: "wal"
    from hermes_state import guard_transient_wal_handle
    from hermes_state_registry import acquire, release_or_close

    db = acquire(Path(db_path))
    guard_transient_wal_handle(db)   # the #109824 preventive guard the scheduler applies
    sid = f"cron-{gen}"
    db.create_session(sid, "cron")
    for i in range(3):
        db.append_message(sid, role="user", content=f"cron {gen} msg {i}")
    release_or_close(db)             # last-connection close inside this process
    wal = Path(db_path + "-wal")
    print(json.dumps({"ok": True, "gen": gen, "cron_wal_ino": wal.stat().st_ino if wal.exists() else None}))
    """
)


def _count_rows_where(db_path: Path, where: str, params: tuple) -> int:
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM messages WHERE {where}", params).fetchone()[0]
    finally:
        conn.close()


def _cron_rows_total(db_path: Path, generations: int) -> int:
    return sum(
        _count_rows_where(db_path, "session_id = ?", (f"cron-{gen}",))
        for gen in range(1, generations + 1)
    )


def _spawn_gateway(tmp_path: Path):
    """Start the gateway child; return a dict of handles, or pytest.skip when WAL is inactive."""
    import queue
    import threading

    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    db_path = tmp_path / "state.db"
    stderr_path = tmp_path / "gateway-stderr.log"
    env = {**os.environ, "HERMES_STATE_DB_GUARD_BYPASS": "1"}
    stderr = stderr_path.open("w")
    proc = subprocess.Popen(
        [_CHILD_PYTHON, "-c", _GATEWAY_CHILD, REPO_ROOT, str(hermes_home), str(db_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
        text=True, encoding="utf-8", bufsize=1, env=env,
    )
    events: "queue.Queue[str]" = queue.Queue()

    def _drain():
        try:
            for line in proc.stdout:
                events.put(line)
        finally:
            events.put(None)

    threading.Thread(target=_drain, daemon=True).start()

    def next_event(name):
        try:
            line = events.get(timeout=_CHILD_TIMEOUT_S)
        except queue.Empty:
            proc.kill()
            pytest.fail(f"gateway child timed out waiting for {name!r}\n{stderr_path.read_text()}")
        if line is None:
            proc.kill()
            pytest.fail(f"gateway child exited early rc={proc.poll()}\n{stderr_path.read_text()}")
        event = json.loads(line)
        if name == "ready" and event.get("event") == "skip":
            proc.kill()
            pytest.skip("WAL not active on this filesystem")
        assert event.get("event") == name, event
        return event

    def probe():
        proc.stdin.write("probe\n")
        proc.stdin.flush()
        return next_event("probe")

    def write():
        proc.stdin.write("write\n")
        proc.stdin.flush()
        return next_event("write")

    def teardown():
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    return {
        "proc": proc,
        "next_event": next_event,
        "probe": probe,
        "write": write,
        "teardown": teardown,
        "hermes_home": hermes_home,
        "db_path": db_path,
    }


def _run_cron_child(db_path: Path, hermes_home: Path, generation: int) -> dict:
    env = {**os.environ, "HERMES_STATE_DB_GUARD_BYPASS": "1"}
    proc = subprocess.run(
        [_CHILD_PYTHON, "-c", _CRON_CHILD, REPO_ROOT, str(hermes_home), str(db_path), str(generation)],
        capture_output=True, text=True, timeout=_CHILD_TIMEOUT_S, env=env,
    )
    assert proc.returncode == 0, (
        f"cron child {generation} failed rc={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _wal_ino(db_path: Path):
    wal = Path(str(db_path) + "-wal")
    return wal.stat().st_ino if wal.exists() else None


# ── Test A: the 30-minute cadence with a live gateway ──────────────────────────────────────────


def test_cron_cycles_keep_wal_inode_stable_with_live_gateway(tmp_path):
    """N transient cron open/write/last-connection-close cycles while a gateway holds its
    registry handle: the -wal inode must never rotate, no retired-wal capture may appear,
    and after every cycle the gateway still writes and the db stays consistent."""
    gw = _spawn_gateway(tmp_path)
    try:
        gw["next_event"]("ready")
        base = gw["probe"]()
        assert base["wal_ino"] is not None, "expected a live -wal sidecar (frames uncheckpointed)"
        if os.name == "nt" and base["wal_ino"] == 0:
            pytest.skip("inode identity unavailable on this filesystem (Windows st_ino == 0)")
        ino0 = base["wal_ino"]
        for gen in range(1, CRON_CYCLES + 1):
            _run_cron_child(gw["db_path"], gw["hermes_home"], gen)
            # The gateway's live view: the transient child's last-connection close must not
            # have rotated anything underneath it.
            snap = gw["probe"]()
            assert snap["wal_ino"] == ino0, (
                f"cron cycle {gen} rotated the -wal inode under a live gateway: "
                f"{ino0} -> {snap['wal_ino']}"
            )
            assert snap["retired_wal_dirs"] == [], (
                f"cron cycle {gen} left retired-wal capture dirs: {snap['retired_wal_dirs']}"
            )
            # The gateway survives every cycle and keeps writing.
            assert gw["write"]()["ok"]
        final = gw["probe"]()
        assert final["wal_ino"] == ino0
        assert final["retired_wal_dirs"] == []
        # Cron rows committed by the transient children are visible from a fresh open.
        assert _cron_rows_total(gw["db_path"], CRON_CYCLES) == CRON_CYCLES * 3
        assert final["messages"] >= base["messages"] + CRON_CYCLES * 3
    finally:
        gw["teardown"]()
    # Final authority: fresh open from the test process, fully quiesced. The gateway's
    # last-connection teardown may legally truncate the now-empty WAL sidecar (standard
    # WAL close semantics) — the invariant is that no generation was ROTATED: all rows,
    # including the cron children's, are intact in the settled database.
    assert integrity_ok_path(gw["db_path"])
    assert _cron_rows_total(gw["db_path"], CRON_CYCLES) == CRON_CYCLES * 3


# ── Test B: the restart cascade — sequential last-writer closes with NO gateway alive ────────


def test_restart_cascade_last_writer_closes_keep_db_consistent(tmp_path):
    """Gateway down: each cron fire is the LAST connection and its close-time checkpoint is the
    only checkpoint at all. The database must stay consistent, all rows stay visible, and no
    retired-wal capture may be needed."""
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    db_path = tmp_path / "state.db"
    for gen in range(1, 3):
        report = _run_cron_child(db_path, hermes_home, gen)
        assert report["ok"]
    assert integrity_ok_path(db_path)
    assert _cron_rows_total(db_path, 2) == 6
    retired = [p.name for p in hermes_home.iterdir() if ".retired-wal-" in p.name]
    assert retired == [], f"restart-cascade closes left retired-wal captures: {retired}"


# ── Test C: the preventive guard itself ────────────────────────────────────────────────────────


def test_guard_noop_on_none_and_read_only():
    from unittest.mock import Mock

    guard = hermes_state.guard_transient_wal_handle
    guard(None)  # must not raise
    ro = Mock(read_only=True, db_path=Path("x"))
    guard(ro)
    ro._disable_close_time_checkpoint.assert_not_called()


def test_guard_disables_close_time_checkpoint_on_writable_handle():
    from unittest.mock import Mock

    guard = hermes_state.guard_transient_wal_handle
    db = Mock(read_only=False, db_path=Path("x"))
    guard(db)
    db._disable_close_time_checkpoint.assert_called_once_with()


def test_guard_swallows_checkpoint_config_failures():
    """Best-effort contract: where setconfig is unavailable (<3.12) or fails, the guard is a
    no-op and the retire-unclosed backstop in SessionDB.close() still covers the handle."""
    from unittest.mock import Mock

    guard = hermes_state.guard_transient_wal_handle
    db = Mock(read_only=False, db_path=Path("state.db"))
    db._disable_close_time_checkpoint.side_effect = RuntimeError("no setconfig on this runtime")
    guard(db)  # must not propagate


def test_open_cron_session_db_applies_guard(monkeypatch, tmp_path):
    """The scheduler wiring: _open_cron_session_db must run the returned handle through
    guard_transient_wal_handle before handing it to the job (the synchronous open path)."""
    import cron.scheduler as sched
    from hermes_state_registry import release_or_close

    db_path = tmp_path / "state.db"
    monkeypatch.setenv("HERMES_STATE_DB_GUARD_BYPASS", "1")
    monkeypatch.setattr("hermes_state._default_db_path", lambda: str(db_path))
    monkeypatch.setattr(sched, "_get_session_db_timeout", lambda: 0)

    captured = {}
    real_guard = hermes_state.guard_transient_wal_handle
    monkeypatch.setattr(
        hermes_state, "guard_transient_wal_handle",
        lambda db: (captured.setdefault("db", db), real_guard(db)),
    )

    handle = sched._open_cron_session_db({"id": "job-1"})
    assert handle is not None and isinstance(handle, SessionDB)
    assert captured["db"] is handle, "_open_cron_session_db must guard the exact handle it returns"
    release_or_close(handle)
