"""Regression: the state.db repair path must never run surgery against a
database another connection is still writing.

Incident (2026-08-18/19): FTS5 shadow-table corruption escalated into b-tree
page damage across `system_prompts`, `session_model_usage` and the `sessions`
index. `repair_state_db_schema` ran its REINDEX/FTS-rebuild strategies while
other connections still held the database open. The caller closes only its own
`self._conn`; the incident process held seven descriptors on state.db.
Rewriting b-tree pages under concurrent writers is what spread the damage out
of the FTS shadow tables and into the canonical tables.

(The companion repair-attempt-ledger fingerprint fix — keying the budget on
something stable across ongoing writes so the cap can actually be reached — is
tracked separately in the fingerprint/repair-loop salvage PR #88425, which
preserves @jirathip-k's #88224 diagnosis and credit. This file covers only the
live-writer guard.)
"""

from __future__ import annotations

import errno
import select
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import hermes_state
import hermes_state_repair
import hermes_state_holders
from hermes_state import SessionDB
from hermes_state_repair import repair_state_db_schema


def _make_wal_db(tmp_path: Path) -> Path:
    """A state.db the repair path will actually work on.

    Built through the real ``SessionDB`` rather than a hand-rolled two-table
    schema. The repair path probes the canonical schema as it goes —
    ``_db_opens_cleanly`` runs ``SELECT COUNT(*) FROM sessions`` and a
    rolled-back ``messages`` write — so a toy schema aborted every repair
    ("no such table: sessions", then "table sessions has no column named id")
    long before reaching the guards these tests exist to cover. The
    assertions below were passing over a code path that never ran.
    """
    db = tmp_path / "state.db"
    handle = SessionDB(db_path=db)
    sid = handle.create_session(session_id=str(uuid.uuid4()), source="cli")
    handle.append_message(sid, role="user", content="seed")
    handle.close()
    return db


# ---------------------------------------------------------------------------
# Repair must refuse to operate under a live writer
# ---------------------------------------------------------------------------


@pytest.mark.requires_wal
def test_repair_refuses_while_another_connection_holds_the_db(tmp_path):
    """Surgery under concurrent writers is what spread the corruption.

    Gated on ``requires_wal``: repair admission
    (``hermes_state_holders.live_writer_holds_db``) first scans for foreign
    holders — deleted sidecar generations, uninspectable Hermes processes —
    and then probes SQLite with ``PRAGMA locking_mode=EXCLUSIVE`` +
    ``BEGIN IMMEDIATE``, which a concurrent connection makes fail with
    SQLITE_BUSY through the WAL index. This test exercises the probe leg: on
    SQLite builds carrying the WAL-reset bug (and on NFS/SMB) Hermes runs
    ``state.db`` in ``journal_mode=DELETE``, where a held reader takes only a
    SHARED lock and ``BEGIN IMMEDIATE`` still acquires RESERVED, so the probe
    alone cannot see the holder. The conftest auto-skips this test where WAL
    is unusable rather than assert a guarantee the probe doesn't make there.
    """
    db = _make_wal_db(tmp_path)

    holder = sqlite3.connect(str(db))
    holder.execute("SELECT count(*) FROM messages").fetchone()
    try:
        report = repair_state_db_schema(db, backup=False)
    finally:
        holder.close()

    assert report["repaired"] is False
    assert "live writer" in (report["error"] or "").lower()


def test_repair_checks_foreign_holders_before_opening_sqlite(tmp_path, monkeypatch):
    """A replacement pathname cannot expose locks on the deleted old inode."""
    db = _make_wal_db(tmp_path)
    monkeypatch.setattr(
        hermes_state_holders,
        "foreign_state_db_holders",
        lambda _path: [(4242, f"{db}-wal (deleted)")],
    )

    def _unexpected_probe(*_args, **_kwargs):
        pytest.fail("repair opened SQLite before excluding foreign holders")

    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", _unexpected_probe)

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert "live writer" in (report["error"] or "").lower()


@pytest.mark.linux_only
def test_linux_holder_scan_does_not_require_psutil(tmp_path, monkeypatch):
    """The Linux safety scan must not make psutil a repair dependency."""
    monkeypatch.setattr(hermes_state_holders, "psutil", None)

    holders = hermes_state_holders.foreign_state_db_holders(
        tmp_path / "absent-state.db"
    )

    assert holders == []


@pytest.mark.linux_only
def test_incomplete_holder_scan_keeps_unknown_sentinel(tmp_path, monkeypatch):
    """A partial scan must not hide uncertainty behind an ordinary holder."""
    db = tmp_path / "state.db"
    db.touch()

    def _listdir(path):
        if path == "/proc":
            return ["4242", "4343"]
        if path == "/proc/4242/fd":
            return ["7"]
        if path == "/proc/4343/fd":
            raise RuntimeError("scan interrupted")
        raise AssertionError(f"unexpected scan path: {path}")

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
    monkeypatch.setattr(hermes_state_holders.os, "readlink", lambda _path: str(db))
    real_stat = hermes_state_holders.os.stat

    def _stat(path, *args, **kwargs):
        if path == "/proc/4242/fd/7":
            return real_stat(db)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

    holders = hermes_state_holders.foreign_state_db_holders(db)

    assert (4242, str(db)) in holders
    assert any(pid < 0 and "scan interrupted" in path for pid, path in holders)


@pytest.mark.linux_only
def test_uninspectable_watched_descriptor_blocks_repair_before_sqlite(
    tmp_path, monkeypatch
):
    """A watched fd whose identity cannot be read is not proven safe."""
    db = _make_wal_db(tmp_path)

    def _listdir(path):
        if path == "/proc":
            return ["4242"]
        if path == "/proc/4242/fd":
            return ["7"]
        raise AssertionError(f"unexpected scan path: {path}")

    real_stat = hermes_state_holders.os.stat

    def _stat(path, *args, **kwargs):
        if path == "/proc/4242/fd/7":
            raise PermissionError(errno.EACCES, "descriptor denied", path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
    monkeypatch.setattr(hermes_state_holders.os, "readlink", lambda _path: str(db))
    monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

    def _unexpected_probe(*_args, **_kwargs):
        pytest.fail("repair opened SQLite with unproven descriptor identity")

    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", _unexpected_probe)

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert "live writer" in (report["error"] or "").lower()


@pytest.mark.linux_only
@pytest.mark.parametrize(
    ("argv", "should_block"),
    (
        (["python3", "backup.py"], False),
        (["python3", "-m", "hermes_cli.main", "gateway"], True),
    ),
)
def test_uninspectable_unknown_descriptor_uses_hermes_identity_at_repair_boundary(
    tmp_path, monkeypatch, argv, should_block
):
    """An unknown fd target blocks only when argv identifies Hermes."""
    db = _make_wal_db(tmp_path)

    def _listdir(path):
        if path == "/proc":
            return ["4242"]
        if path == "/proc/4242/fd":
            return ["7"]
        raise AssertionError(f"unexpected scan path: {path}")

    def _readlink(path):
        raise PermissionError(errno.EACCES, "descriptor denied", path)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
    monkeypatch.setattr(hermes_state_holders.os, "readlink", _readlink)
    monkeypatch.setattr(
        hermes_state_holders,
        "_read_proc_argv",
        lambda _pid: argv,
    )

    if should_block:

        def _unexpected_probe(*_args, **_kwargs):
            pytest.fail("repair opened SQLite with an unproven Hermes descriptor")

        monkeypatch.setattr(
            hermes_state_repair, "_connect_repair_durable",
            _unexpected_probe,
        )
        report = repair_state_db_schema(db, backup=False)

        assert report["repaired"] is False
        assert "live writer" in (report["error"] or "").lower()
    else:
        real_connect = hermes_state_repair._connect_repair_durable
        probe_reached = False

        def _record_probe(*args, **kwargs):
            nonlocal probe_reached
            probe_reached = True
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(
            hermes_state_repair, "_connect_repair_durable",
            _record_probe,
        )
        report = repair_state_db_schema(db, backup=False)

        assert probe_reached is True
        assert "live writer" not in (report["error"] or "").lower()


@pytest.mark.linux_only
def test_uninspectable_watched_identity_blocks_alias_before_sqlite(
    tmp_path, monkeypatch
):
    """A non-disappearance stat error cannot prove an aliased holder safe."""
    db = _make_wal_db(tmp_path)
    alias = tmp_path / "namespace-alias" / "state.db"

    def _listdir(path):
        if path == "/proc":
            return ["4242"]
        if path == "/proc/4242/fd":
            return ["7"]
        raise AssertionError(f"unexpected scan path: {path}")

    real_stat = hermes_state_holders.os.stat

    def _stat(path, *args, **kwargs):
        if str(path) == str(db) and not args and not kwargs:
            raise PermissionError(errno.EACCES, "watched identity denied", path)
        if path == "/proc/4242/fd/7":
            return real_stat(db)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
    monkeypatch.setattr(
        hermes_state_holders.os, "readlink", lambda _path: str(alias)
    )
    monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)

    def _unexpected_probe(*_args, **_kwargs):
        pytest.fail("repair opened SQLite with an unproven watched identity")

    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", _unexpected_probe)

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert "live writer" in (report["error"] or "").lower()


@pytest.mark.linux_only
def test_uninspectable_alias_descriptor_for_hermes_blocks_before_sqlite(
    tmp_path, monkeypatch
):
    """Hermes cannot make an aliased fd safe when its identity is unreadable."""
    db = _make_wal_db(tmp_path)
    alias = tmp_path / "namespace-alias" / "state.db"

    def _listdir(path):
        if path == "/proc":
            return ["4242"]
        if path == "/proc/4242/fd":
            return ["7"]
        raise AssertionError(f"unexpected scan path: {path}")

    real_stat = hermes_state_holders.os.stat

    def _stat(path, *args, **kwargs):
        if path == "/proc/4242/fd/7":
            raise PermissionError(errno.EACCES, "descriptor denied", path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)
    monkeypatch.setattr(
        hermes_state_holders.os, "readlink", lambda _path: str(alias)
    )
    monkeypatch.setattr(hermes_state_holders.os, "stat", _stat)
    monkeypatch.setattr(
        hermes_state_holders,
        "_read_proc_argv",
        lambda _pid: ["python3", "-m", "hermes_cli.main", "gateway"],
    )

    def _unexpected_probe(*_args, **_kwargs):
        pytest.fail("repair opened SQLite with an unproven Hermes alias fd")

    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", _unexpected_probe)

    report = repair_state_db_schema(db, backup=False)

    assert report["repaired"] is False
    assert "live writer" in (report["error"] or "").lower()


@pytest.mark.requires_wal
@pytest.mark.linux_only
def test_repair_refuses_while_foreign_process_holds_deleted_wal(tmp_path):
    """Reproduce the inode split that a pathname lock probe cannot observe."""
    db = _make_wal_db(tmp_path)
    holder_code = """
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("BEGIN IMMEDIATE")
print("ready", flush=True)
sys.stdin.read(1)
conn.rollback()
conn.close()
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code, str(db)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        readable, _, _ = select.select([holder.stdout], [], [], 10)
        assert readable, "holder subprocess did not signal readiness"
        assert holder.stdout.readline().strip() == "ready"
        deleted = []
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{db}{suffix}")
            if sidecar.exists():
                sidecar.unlink()
                deleted.append(sidecar)
        assert deleted

        report = repair_state_db_schema(db, backup=False)

        assert report["repaired"] is False
        assert "live writer" in (report["error"] or "").lower()
    finally:
        if holder.poll() is None and holder.stdin is not None:
            try:
                holder.stdin.write("x")
                holder.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        try:
            holder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=10)


def test_repair_proceeds_once_the_database_is_quiescent(tmp_path):
    """The guard must not deadlock repair on an exclusively-held file."""
    db = _make_wal_db(tmp_path)

    report = repair_state_db_schema(db, backup=False)

    assert "live writer" not in (report["error"] or "").lower()
