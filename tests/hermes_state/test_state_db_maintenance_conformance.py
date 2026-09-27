"""Conformance suite: state.db maintenance must be guard-aware and copy-honest.

Pins the bug CLASS behind #91839 (FTS rebuild corrupting a shared state.db),
#90806 (WAL sidecars replaced under live holders), #90613 (_safe_copy_db
copying a corrupt DB while reporting success) and #88235 (repair ran surgery
under a live writer — repair-specific guard tests already exist in
tests/hermes_state/test_state_db_repair_live_writer_guard.py and are NOT
duplicated here).

Contract, in two sentences:

1. No maintenance entry point that does structural work on state.db may run
   unguarded against a database another live connection still holds — each
   must refuse loudly, degrade to a no-op, or coordinate safely, and the
   database must be ``PRAGMA integrity_check == ok`` afterwards.
2. Any copy/snapshot path must fail closed on a source SQLite cannot read
   (no destination left behind, DB flagged in the snapshot manifest);
   success + unusable copy is the #90613 bug.

All databases are real SQLite files under tmp_path (the production-DB
isolation guard in hermes_state hard-fails on production-shaped paths — never
point these tests at a real profile).  Corruption is real byte damage, not
mocks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest

import hermes_state_repair
from hermes_state import SessionDB
from hermes_cli.backup import (
    _safe_copy_db,
    create_quick_snapshot,
    verify_sqlite_integrity,
)

# ---------------------------------------------------------------------------
# Fixtures — real WAL-mode SessionDB in tmp dirs
# ---------------------------------------------------------------------------


def _make_state_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    handle = SessionDB(db_path=db)
    sid = handle.create_session(session_id=str(uuid.uuid4()), source="cli")
    for i in range(40):
        handle.append_message(sid, role="user",
                              content=f"needle-{i} " + "payload " * 40)
    handle.close()
    return db


def _journal_mode(db: Path) -> str:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    finally:
        conn.close()


def _require_wal(db: Path) -> None:
    mode = _journal_mode(db)
    if mode != "wal":
        pytest.skip(
            f"runtime opened state.db in journal_mode={mode!r} — this "
            "interpreter's sqlite refuses WAL (known venv quirk); the "
            "live-writer contract here is only meaningful under WAL"
        )


def _integrity_ok(db: Path) -> bool:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


class _LiveWriter:
    """A REAL second connection holding a write transaction from a thread."""

    def __init__(self, db: Path):
        self.db = db
        self.ready = threading.Event()
        self.release = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        conn = None
        try:
            conn = sqlite3.connect(str(self.db), timeout=0.0)
            conn.execute("PRAGMA busy_timeout=0")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO state_meta(key, value) VALUES('lw-probe','1') "
                "ON CONFLICT(key) DO UPDATE SET value='1'"
            )
            self.ready.set()
            self.release.wait(timeout=120)
            conn.execute("ROLLBACK")
        except Exception as exc:  # surfaced by __exit__
            self.error = exc
            self.ready.set()
        finally:
            if conn is not None:
                conn.close()

    def __enter__(self) -> "_LiveWriter":
        self.thread.start()
        assert self.ready.wait(timeout=30), "live writer never acquired lock"
        if self.error is not None:
            raise self.error
        return self

    def __exit__(self, *exc) -> None:
        self.release.set()
        self.thread.join(timeout=30)


# ---------------------------------------------------------------------------
# Part 1 — LIVE-WRITER GUARD (in-place ops)
# ---------------------------------------------------------------------------

# In-place structural ops and the kwargs that make them attempt work now.
_IN_PLACE_OPS = {
    "vacuum": {},
    "maybe_auto_prune_and_vacuum": {"retention_days": 0, "min_interval_hours": 0},
    "rebuild_fts": {},
    "optimize_fts": {},
}


def _claims_no_work(result) -> bool:
    """A never-raise maintenance API degraded honestly: it reports zero work.

    ``int`` results count structures rewritten; the auto-maintenance dict
    must not say it vacuumed. Anything else is a claim of success under a
    live writer — the #91839 class.
    """
    if isinstance(result, dict):
        return result.get("vacuumed") is False
    return result == 0


@pytest.mark.requires_wal
@pytest.mark.parametrize("op_name", sorted(_IN_PLACE_OPS))
def test_in_place_op_refuses_or_degrades_under_live_writer(tmp_path, op_name):
    """#91839/#88235 class: structural work must not proceed under a holder.

    With a real second connection holding BEGIN IMMEDIATE, each in-place
    maintenance op must either refuse loudly (``sqlite3.OperationalError``
    naming the lock) or degrade to no work — never report success — and the
    database must be integrity-clean and row-identical afterwards.
    """
    db = _make_state_db(tmp_path)
    _require_wal(db)
    handle = SessionDB(db_path=db)
    try:
        with _LiveWriter(db):
            method = getattr(handle, op_name)
            try:
                result = method(**_IN_PLACE_OPS[op_name])
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                assert "locked" in msg or "busy" in msg, exc
            else:
                assert _claims_no_work(result), (
                    f"{op_name} reported {result!r} while a live writer held "
                    "the database (#91839 class: success claimed under a holder)"
                )
    finally:
        handle.close()
    assert _integrity_ok(db), f"{op_name} left state.db corrupt (#91839)"
    # Canonical rows untouched.
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 40
    finally:
        conn.close()


@pytest.mark.requires_wal
def test_live_writer_probe_detects_real_holder(tmp_path):
    """The shared guard primitive must see a real holder (#88235, #90806)."""
    db = _make_state_db(tmp_path)
    _require_wal(db)
    assert hermes_state_repair._live_writer_holds_db(db) is False
    with _LiveWriter(db):
        assert hermes_state_repair._live_writer_holds_db(db) is True, (
            "guard failed to detect a live write transaction — every "
            "registered op relying on it is now unguarded"
        )
    assert hermes_state_repair._live_writer_holds_db(db) is False


# ---------------------------------------------------------------------------
# Part 2 — COPY HONESTY (#90613)
# ---------------------------------------------------------------------------


def test_copy_honesty_truncated_db(tmp_path):
    """Truncation mid-page: even the raw copier must fail closed.

    ``_safe_copy_db`` routes through sqlite3's backup() API, which reads
    source pages — a file truncated mid-page must yield False and no
    destination, never a truthy return.
    """
    good = _make_state_db(tmp_path)
    raw = good.read_bytes()
    trunc = tmp_path / "trunc.db"
    trunc.write_bytes(raw[: len(raw) - 4096 - 100])
    assert not verify_sqlite_integrity(trunc, run_pragma=True)["valid"]

    dst = tmp_path / "copy.db"
    assert _safe_copy_db(trunc, dst) is False
    assert not dst.exists(), "_safe_copy_db left a partial destination"


def test_quick_snapshot_flags_corrupt_state_db(tmp_path):
    """The quick-snapshot path must flag — never silently absorb — a bad DB.

    A fake HERMES_HOME carries a truncated state.db plus a config.yaml. The
    snapshot must either return None or record state.db in the manifest's
    failed_dbs; a manifest listing state.db as captured is the #90613 class
    surfacing through the snapshot path.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "config.yaml").write_text("test: true\n")
    good = _make_state_db(tmp_path)
    raw = good.read_bytes()
    (home / "state.db").write_bytes(raw[: len(raw) - 4096 - 100])

    snap_id = create_quick_snapshot(label="conformance", hermes_home=home)
    if snap_id is None:
        return  # refused outright: honest
    snap_dir = home / "state-snapshots" / snap_id
    meta = json.loads((snap_dir / "manifest.json").read_text())
    if "state.db" in meta.get("files", {}):
        copied = snap_dir / "state.db"
        assert verify_sqlite_integrity(copied, run_pragma=True)["valid"], (
            "quick snapshot recorded a corrupt state.db as captured"
        )
    else:
        assert "state.db" in meta.get("failed_dbs", []), (
            "state.db neither captured nor flagged as failed — silent loss"
        )
