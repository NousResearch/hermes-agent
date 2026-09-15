"""An aborted backup must not leave its staging artifacts behind.

The disk report of 15/09/2026 found ``.<name>.<pid>-<tid>.partial`` files (681 MB) and a staged
SQLite copy next to the output: both were written by a run that was killed with SIGTERM, which the
default disposition turns into an immediate death — no ``finally``, no ``ExitStack``. These tests
pin the two guarantees that close it: the termination guard unwinds through the existing cleanup,
and the next run sweeps what no handler could reach (SIGKILL).

The sweep only ever runs on directories where one of our hidden spellings can land — the backup
output directory, the snapshot root, and every restore target directory (``hermes import`` stages
the bytes it is about to publish *inside the home*, next to the file it replaces; the unlink+move
fallback stages beside the live database).
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

import pytest

from hermes_cli.backup import (
    _STAGED_DB_PREFIX,
    _import_members,
    _run_backup_locked,
    _safe_restore_db,
    prune_stale_staging,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")


def _dead_pid() -> int:
    """A pid that certainly no longer exists (spawned, waited for, reaped)."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "this_is_not_a_module"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait(timeout=30)
    return proc.pid


def _touch(path: Path, *, age_seconds: float, size: int = 16) -> Path:
    path.write_bytes(b"x" * size)
    return _age(path, age_seconds=age_seconds)


def _age(path: Path, *, age_seconds: float) -> Path:
    """Backdate *path* (file or directory) so the age-based gate sees it as stale."""
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_sigterm_removes_the_backup_partial(tmp_path) -> None:
    """A real writer, SIGTERM'd mid-write, leaves neither the partial nor a published target."""
    out = tmp_path / "out"
    out.mkdir()
    target = out / "probe-backup.zip"
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(
        """
        import pathlib, sys, time
        from hermes_cli.backup import _atomic_output_path

        out = pathlib.Path(sys.argv[1])
        target = out / "probe-backup.zip"
        with _atomic_output_path(target) as staging:
            with open(staging, "wb") as handle:
                for _ in range(400):
                    handle.write(b"x" * 4096)
                    handle.flush()
                    time.sleep(0.01)
        print("PUBLISHED")
        """
    ), encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(child), str(out)], cwd=os.getcwd())
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not list(out.glob(".*.partial")):
            time.sleep(0.02)
        assert list(out.glob(".*.partial")), "the child never started writing its partial"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    finally:
        if proc.poll() is None:  # pragma: no cover - safety net
            proc.kill()
            proc.wait(timeout=30)

    assert not list(out.glob(".*.partial")), "SIGTERM stranded the hidden partial"
    assert not target.exists(), "an aborted write must not publish the target"


def test_staged_db_copy_is_named_for_the_sweep(tmp_path) -> None:
    """The SQLite staging copy carries the pid, so it is distinguishable from an unrelated file."""
    live = tmp_path / f"{_STAGED_DB_PREFIX}{os.getpid()}-1.abcdef.db"
    live.write_bytes(b"live")
    dead = tmp_path / f"{_STAGED_DB_PREFIX}{_dead_pid()}-1.abcdef.db"
    dead.write_bytes(b"dead")

    prune_stale_staging(tmp_path)

    assert live.exists(), "a live run's staging copy must survive the sweep"
    assert not dead.exists(), "a dead run's staging copy must be swept"


def test_prune_stale_staging_bounds_partial_residue(tmp_path) -> None:
    """Every leftover class is collected; live, recent and non-Hermes artifacts are not."""
    live = tmp_path / f".hermes-backup-2026.zip.{os.getpid()}-7.partial"
    live.write_bytes(b"live")
    dead = tmp_path / f".hermes-backup-2026.zip.{_dead_pid()}-7.partial"
    dead.write_bytes(b"dead")
    orphan_dir = tmp_path / f".20260101-000000.{_dead_pid()}.partial"
    orphan_dir.mkdir()
    (orphan_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    pidless_old = _touch(tmp_path / ".legacy-backup.partial", age_seconds=7 * 60 * 60)
    # mkstemp spelling (restore/extract): pid-less, so only the age gate can reach it.
    pidless_mkstemp = _touch(tmp_path / ".config.yaml.a1b2c3.partial", age_seconds=7 * 60 * 60)
    # Fifth spelling, same pid-less class: the database staged by `hermes import` before the
    # live-safe page copy (`.{db}.XXXXXX.dbimport`).
    pidless_dbimport = _touch(tmp_path / ".state.db.a1b2c3.dbimport", age_seconds=7 * 60 * 60)
    # Sixth spelling: the unlink+move fallback stages the snapshot beside the live database.
    pidless_snap_restore = _touch(tmp_path / ".state.db.snap_restore", age_seconds=7 * 60 * 60)
    pidless_recent = _touch(tmp_path / ".fresh-backup.partial", age_seconds=60)
    dbimport_recent = _touch(tmp_path / ".state.db.d4e5f6.dbimport", age_seconds=60)
    snap_restore_recent = _touch(tmp_path / ".state.db.fresh.snap_restore", age_seconds=60)

    # Not ours, however old. The sweep runs on the output directory — `-o <dir>`, or $HOME when
    # `-o` is absent — so a *visible* name is a user artifact even when it ends in ".partial"
    # (this is what the pre-fix sweep deleted: a user file, a user directory, and a published
    # `<ts>-<label>.partial` snapshot).
    user_file = _touch(tmp_path / "notes.partial", age_seconds=7 * 60 * 60)
    user_dir = tmp_path / "20260915-203000-foo.partial"
    user_dir.mkdir()
    (user_dir / "keepme.txt").write_bytes(b"user file")
    _age(user_dir, age_seconds=7 * 60 * 60)
    # Hidden and old but with no pid: nothing attributes it to a run, so it is never rmtree'd.
    stranger_dir = tmp_path / ".attic.partial"
    stranger_dir.mkdir()
    (stranger_dir / "keepme.txt").write_bytes(b"user dir")
    _age(stranger_dir, age_seconds=7 * 60 * 60)

    removed = prune_stale_staging(tmp_path)

    assert removed == 6
    assert live.exists() and pidless_recent.exists() and dbimport_recent.exists()
    assert snap_restore_recent.exists()
    assert user_file.exists(), "a visible *.partial is the user's, never ours to delete"
    assert (user_dir / "keepme.txt").exists(), "a visible *.partial directory must not be rmtree'd"
    assert (stranger_dir / "keepme.txt").exists(), "no pid: no attribution, no removal"
    assert not dead.exists() and not orphan_dir.exists()
    assert not pidless_old.exists() and not pidless_mkstemp.exists() and not pidless_dbimport.exists()
    assert not pidless_snap_restore.exists()


def test_full_backup_sweeps_a_partial_left_by_a_killed_run(tmp_path) -> None:
    """The run itself is the collector: a fresh backup removes the previous run's residue."""
    from types import SimpleNamespace

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    stale = out / f".hermes-backup-2026-09-14-184032.zip.{_dead_pid()}-9.partial"
    stale.write_bytes(b"y" * 4096)
    # The output directory is the user's: a real run must not touch these (the pre-fix sweep
    # removed all three — the file, the directory, and the published-snapshot spelling).
    user_file = _touch(out / "notes.partial", age_seconds=7 * 60 * 60)
    user_dir = out / "archive.partial"
    user_dir.mkdir()
    (user_dir / "keepme.txt").write_bytes(b"user data")
    _age(user_dir, age_seconds=7 * 60 * 60)

    _run_backup_locked(SimpleNamespace(output=str(out), keep=0, label=None), home)

    assert not stale.exists(), "the next backup left the old partial in place"
    assert user_file.exists(), "a real run deleted a user file in the output directory"
    assert (user_dir / "keepme.txt").exists(), "a real run rmtree'd a user directory"
    assert list(out.glob("hermes-backup-*.zip")), "the backup itself did not run"


def _sqlite_bytes(path: Path) -> bytes:
    """A tiny but valid SQLite image (the import path copies pages, so the bytes must be real)."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, note TEXT)")
        conn.execute("INSERT INTO sessions (note) VALUES ('kept')")
        conn.commit()
    finally:
        conn.close()
    return path.read_bytes()


def test_import_sweeps_the_staging_it_would_leave_beside_each_member(tmp_path) -> None:
    """`hermes import` stages inside the home: every restore target directory is swept first.

    ``_import_db_member`` writes ``.{db}.XXXXXX.dbimport`` next to the *live* database and
    ``_extract_member_atomically`` writes ``.{name}.XXXXXX.partial`` next to any other member — not
    in a backup output directory, which was the only place the sweep ran. The two pre-fix call sites
    never visited this directory: a SIGKILLed (OOM) import left a hidden copy of the database
    (~1.4 GB on the node) that ``ls``, the hygiene cron and the next import all walked past.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    target = home / "state.db"
    payload = _sqlite_bytes(target)
    config = home / "config.yaml"
    config.write_text("model: local-edit\n", encoding="utf-8")
    archive = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("state.db", payload)
        zf.writestr("config.yaml", "model: archived\n")

    stale_import = _touch(home / ".state.db.a1b2c3.dbimport", age_seconds=7 * 60 * 60, size=8192)
    stale_partial = _touch(home / ".config.yaml.a1b2c3.partial", age_seconds=7 * 60 * 60, size=4096)
    fresh_import = _touch(home / ".state.db.d4e5f6.dbimport", age_seconds=60, size=8192)
    fresh_partial = _touch(home / ".config.yaml.d4e5f6.partial", age_seconds=60, size=4096)
    # Visible, old, and not ours: the sweep runs in the user's home, so these must survive.
    user_partial = _touch(home / "notes.partial", age_seconds=7 * 60 * 60)
    user_dir = home / "archive.partial"
    user_dir.mkdir()
    (user_dir / "keepme.txt").write_bytes(b"user data")

    with zipfile.ZipFile(archive) as zf:
        restored, _, errors, _, _ = _import_members(zf, ["state.db", "config.yaml"], "", home, 2)

    assert restored == 2 and not errors, errors
    assert not stale_import.exists(), "the orphaned `.dbimport` next to the live DB survived"
    assert not stale_partial.exists(), "the orphaned `.partial` in the restore target survived"
    assert fresh_import.exists() and fresh_partial.exists(), "a live peer's staging was collected"
    assert user_partial.exists(), "a visible *.partial in the home is the user's"
    assert (user_dir / "keepme.txt").exists(), "a visible *.partial directory must not be rmtree'd"
    assert config.read_text(encoding="utf-8") == "model: archived\n", "the member was not restored"
    conn = sqlite3.connect(str(target))
    try:
        assert conn.execute("SELECT note FROM sessions").fetchone() == ("kept",)
    finally:
        conn.close()


def test_restore_fallback_sweeps_the_snap_restore_residue(tmp_path) -> None:
    """The unlink+move fallback stages beside the live DB, so that directory is swept too.

    ``_unlink_move_restore_db`` writes ``.{dst.name}.snap_restore`` into the database's own
    directory — the Hermes home for ``state.db`` — where neither the backup-output sweep nor the
    snapshot-root sweep ever looks. Two guarantees are pinned here: a copy stranded by an earlier
    abrupt death in that directory is collected (the sweep), and this run's own copy is never left
    behind (the guard + ``finally``). The stale plant is deliberately another database's residue:
    ``.{dst.name}.snap_restore`` is deterministic, so a leftover of a restore *of the same* database
    is overwritten by the next one anyway and would prove nothing.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    dst = home / "state.db"
    _sqlite_bytes(dst)
    with open(dst, "r+b") as handle:  # corrupt header: routes _safe_restore_db to the fallback
        handle.write(b"\x00" * 100)
    src = tmp_path / "snapshot.db"
    _sqlite_bytes(src)

    stale = _touch(home / ".other.db.snap_restore", age_seconds=7 * 60 * 60, size=4096)
    fresh = _touch(home / ".other.db.d4e5f6.snap_restore", age_seconds=60, size=4096)
    user_file = _touch(home / "other.db.snap_restore", age_seconds=7 * 60 * 60)

    assert _safe_restore_db(src, dst) is True, "the fallback did not run"

    assert dst.read_bytes() == src.read_bytes(), "the fallback did not publish the snapshot"
    assert not stale.exists(), "the stranded `.snap_restore` copy in the home survived"
    assert not (home / ".state.db.snap_restore").exists(), "the run stranded its own staged copy"
    assert fresh.exists(), "a recent `.snap_restore` may belong to a live restore"
    assert user_file.exists(), "a visible *.snap_restore is the user's"
