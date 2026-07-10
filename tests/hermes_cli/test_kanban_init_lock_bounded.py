"""Tests for the kanban cross-process init lock.

Two guarantees, both covered here:

1. Fast path (#36644, ``84ba83b09``): once a path is initialized in this
   process, `connect()` skips the cross-process init lock entirely (nothing
   left to serialize), so a held lock cannot block a steady-state connect —
   in particular the long-lived gateway dispatcher's next-tick `connect()`.
2. Blocking first-init: on the first-init path the acquire deliberately
   blocks. The previous bounded acquire ("retry until a deadline, then
   proceed without the lock") timed out spuriously under fresh-board bursts
   of ~20 processes and the losers proceeded against a half-initialized
   database. Losers must wait for the owner, then see a fully initialized
   schema.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return home


def _hold_init_lock(db_path: Path):
    """Return (start_event, release_event, thread) holding the init lock."""
    holding = threading.Event()
    release = threading.Event()

    def _holder():
        with kb._cross_process_init_lock(db_path):
            holding.set()
            release.wait(timeout=30)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert holding.wait(timeout=5), "holder thread never acquired the lock"
    return release, t


def test_initialized_path_connect_skips_init_lock(kanban_home):
    """A connect to an already-initialized path must not block on the init lock."""
    db_path = kb.kanban_db_path(board="default")
    # Initialize once.
    kb.connect().close()
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

    # Hold the init lock; a fast-path connect must return promptly anyway.
    release, t = _hold_init_lock(db_path)
    try:
        start = time.monotonic()
        kb.connect().close()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"fast-path connect blocked on the init lock ({elapsed:.2f}s)"
    finally:
        release.set()
        t.join(timeout=5)


def test_first_init_connect_blocks_until_lock_released(kanban_home):
    """First-init connect must WAIT for the lock holder, not fall through.

    The old bounded acquire proceeded WITHOUT the cross-process lock after a
    timeout, so a loser of a fresh-board race could read a half-initialized
    database. With the blocking acquire, a first-init connect must still be
    waiting while the lock is held, and must complete once it is released.
    """
    db_path = kb.kanban_db_path(board="default")
    release, holder = _hold_init_lock(db_path)

    result: dict = {}

    def _first_init_connect():
        try:
            conn = kb.connect()  # path NOT yet initialized — takes the lock path
            conn.close()
            result["ok"] = True
        except Exception as exc:  # pragma: no cover - failure detail for assert
            result["error"] = exc

    connector = threading.Thread(target=_first_init_connect, daemon=True)
    try:
        connector.start()
        # While the lock is held, the first-init connect must NOT complete.
        connector.join(timeout=1.2)
        assert connector.is_alive(), (
            "first-init connect() returned while the init lock was held — "
            "the blocking acquire fell through"
        )
        assert str(db_path.resolve()) not in kb._INITIALIZED_PATHS

        # Release the holder: the blocked connect must now finish initialization.
        release.set()
        connector.join(timeout=15)
        assert not connector.is_alive(), "first-init connect never completed after release"
        assert result.get("ok"), f"first-init connect failed: {result.get('error')!r}"
        assert str(db_path.resolve()) in kb._INITIALIZED_PATHS
    finally:
        release.set()
        holder.join(timeout=5)
        connector.join(timeout=15)


_FIRST_INIT_WORKER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path

    from hermes_cli import kanban_db as kb

    db_path = Path(sys.argv[1])
    ready_file = Path(sys.argv[2])
    go_file = Path(sys.argv[3])

    # Imports done — report ready, then spin until the parent fires the gun so
    # every worker hits the fresh board's first init as simultaneously as
    # possible.
    ready_file.touch()
    deadline = time.monotonic() + 60
    while not go_file.exists():
        if time.monotonic() >= deadline:
            sys.exit(3)
        time.sleep(0.002)

    conn = kb.connect(db_path)
    try:
        # The schema must be fully present: a loser that fell through the lock
        # mid-init would fail here (missing tables) or in the integrity probe
        # inside connect() itself.
        conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if (row[0] or "").lower() != "ok":
            sys.exit(4)
    finally:
        conn.close()
    sys.exit(0)
    """
)


def test_concurrent_fresh_board_first_init_subprocesses(tmp_path):
    """Simultaneous fresh-board first-inits serialize behind the byte lock.

    Regression test for the fresh-board race: ~20 processes connecting to a
    brand-new board made the old bounded lock wait time out, and the losers
    fell through against a half-initialized database (integrity failures,
    quarantine-backup storms). Every process must succeed, see the complete
    schema, and produce no quarantine backup.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = tmp_path / "board" / "kanban.db"
    go_file = tmp_path / "go"
    repo_root = Path(kb.__file__).resolve().parent.parent
    n_procs = 8

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    procs = []
    ready_files = []
    for i in range(n_procs):
        ready = tmp_path / f"ready-{i}"
        ready_files.append(ready)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _FIRST_INIT_WORKER,
                    str(db_path),
                    str(ready),
                    str(go_file),
                ],
                env=env,
                cwd=str(tmp_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    try:
        # Wait until every worker has finished importing and is spinning on the
        # go file, then start them all at once.
        deadline = time.monotonic() + 60
        while not all(f.exists() for f in ready_files):
            assert time.monotonic() < deadline, "workers never became ready"
            for p in procs:
                if p.poll() is not None and p.returncode != 0:
                    raise AssertionError(
                        f"worker died before start (rc={p.returncode}): "
                        f"{p.communicate()[1]}"
                    )
            time.sleep(0.01)
        go_file.touch()

        failures = []
        for p in procs:
            out, err = p.communicate(timeout=120)
            if p.returncode != 0:
                failures.append((p.returncode, err.strip().splitlines()[-5:]))
        assert not failures, f"concurrent first-init workers failed: {failures}"
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.communicate()

    # No loser may have quarantined the database mid-init.
    corrupt_backups = list(db_path.parent.glob("*.corrupt.*"))
    assert not corrupt_backups, f"quarantine backups created: {corrupt_backups}"

    # The board is fully initialized and healthy after the burst.
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone(), "schema incomplete after concurrent first-init"
        row = conn.execute("PRAGMA integrity_check").fetchone()
        assert (row[0] or "").lower() == "ok"
    finally:
        conn.close()
