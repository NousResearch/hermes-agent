"""Conformance suite: state.db maintenance must be guard-aware and copy-honest.

Pins the bug CLASS behind #91839 (FTS rebuild corrupting a shared state.db),
#90806 (WAL sidecars replaced under live holders), #90613 (_safe_copy_db
copying a corrupt DB while reporting success) and #88235 (repair ran surgery
under a live writer — repair-specific guard tests already exist in
tests/test_state_db_repair_live_writer_guard.py and are NOT duplicated here).

Contract, in two sentences:

1. No maintenance entry point that does structural work on state.db may run
   unguarded against a database another live connection still holds — each
   must refuse loudly, degrade to a no-op, or coordinate safely, and the
   database must be ``PRAGMA integrity_check == ok`` afterwards.
2. Any copy/snapshot path that reports success must leave an
   integrity-clean destination; success + corrupt copy is the #90613 bug.

The REGISTRY below enumerates every known maintenance entry point.  A
completeness test greps the source modules so a NEW maintenance function (or
a rename of an existing one) fails this suite loudly instead of silently
shipping without live-writer conformance coverage.

All databases are real SQLite files under tmp_path (the production-DB
isolation guard in hermes_state hard-fails on production-shaped paths — never
point these tests at a real profile).  Corruption is real byte damage, not
mocks.
"""

from __future__ import annotations

import dataclasses
import re
import sqlite3
import threading
import uuid
from pathlib import Path

import pytest

import hermes_state
from hermes_state import SessionDB
from hermes_cli import backup as backup_mod
from hermes_cli.backup import (
    _create_quick_snapshot_locked,
    _safe_copy_db,
    copy_db_and_verify,
    verify_sqlite_integrity,
)

REPO_ROOT = Path(hermes_state.__file__).resolve().parent

# ---------------------------------------------------------------------------
# REGISTRY — every maintenance entry point doing structural work on state.db
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MaintenanceOp:
    """One maintenance entry point and where its live-writer safety lives."""

    module: str          # import path holding the symbol
    qualname: str        # symbol ("func" or "Class.method")
    kind: str            # "in-place" | "copy" | "guard"
    covered_by: str      # which test (here or elsewhere) pins its conformance


REGISTRY: tuple[MaintenanceOp, ...] = (
    # In-place structural ops (part 2 here, except repair — covered elsewhere)
    MaintenanceOp("hermes_state", "SessionDB.vacuum", "in-place",
                  "test_in_place_op_refuses_or_degrades_under_live_writer"),
    MaintenanceOp("hermes_state", "SessionDB.maybe_auto_prune_and_vacuum",
                  "in-place",
                  "test_in_place_op_refuses_or_degrades_under_live_writer"),
    MaintenanceOp("hermes_state", "SessionDB.rebuild_fts", "in-place",
                  "test_in_place_op_refuses_or_degrades_under_live_writer"),
    MaintenanceOp("hermes_state", "SessionDB.optimize_fts", "in-place",
                  "test_in_place_op_refuses_or_degrades_under_live_writer"),
    MaintenanceOp("hermes_state", "repair_state_db_schema", "in-place",
                  "tests/test_state_db_repair_live_writer_guard.py"),
    MaintenanceOp("hermes_state", "SessionDB._try_runtime_fts_rebuild",
                  "in-place", "tests/state/test_fts_runtime_rebuild.py"),
    # The guards themselves — renaming one must break this suite (#88235,
    # #90806: _foreign_state_db_holders is what stops sidecar replacement
    # under a live holder).
    MaintenanceOp("hermes_state", "_live_writer_holds_db", "guard",
                  "test_registry_symbols_exist"),
    MaintenanceOp("hermes_state", "SessionDB._foreign_state_db_holders",
                  "guard", "test_registry_symbols_exist"),
    MaintenanceOp("hermes_cli.backup", "verify_sqlite_integrity", "guard",
                  "test_copy_honesty_page_damage"),
    # Copy/snapshot paths (part 3)
    MaintenanceOp("hermes_cli.backup", "_safe_copy_db", "copy",
                  "test_copy_honesty_truncated_db"),
    MaintenanceOp("hermes_cli.backup", "copy_db_and_verify", "copy",
                  "test_copy_honesty_page_damage"),
    MaintenanceOp("hermes_cli.backup", "_create_quick_snapshot_locked",
                  "copy", "test_quick_snapshot_flags_corrupt_state_db"),
)

# Source files scanned for maintenance-shaped function definitions.
_SCAN_SOURCES = ("hermes_state.py", "hermes_state_search.py",
                 "hermes_cli/backup.py")

# def-names matching this pattern must be in the REGISTRY or the exempt list.
_MAINT_DEF_RE = re.compile(
    r"^\s*def\s+(\w*(?:vacuum|rebuild_fts|repair_state|safe_copy|snapshot"
    r"|checkpoint|optimize_fts|copy_db)\w*)\s*\(", re.M,
)

# Reviewed and deliberately outside the registry: read-only, orchestration
# wrappers around registered ops, or non-structural helpers.
_SCAN_EXEMPT = {
    "_apply_macos_checkpoint_barrier",   # per-connection pragma helper
    "_try_wal_checkpoint",               # best-effort on own conn, no surgery
    "list_quick_snapshots",              # read-only
    "restore_quick_snapshot",            # restore target, not live state.db
    "_prune_quick_snapshots",            # deletes snapshot dirs, not the DB
    "prune_quick_snapshots",             # public wrapper of the above
    "_quick_snapshot_root",              # path helper
    "create_quick_snapshot",             # lock wrapper -> registered _locked
    "create_pre_update_snapshots_all_profiles",  # loops create_quick_snapshot
    "optimize_fts_storage",              # offline rebuild w/ its own tests
    "_repair_state_db_schema_locked",    # inner half of registered repair
}


def _resolve(op: MaintenanceOp):
    mod = {"hermes_state": hermes_state, "hermes_cli.backup": backup_mod}[op.module]
    obj = mod
    for part in op.qualname.split("."):
        obj = getattr(obj, part)
    return obj


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


def _corrupt_page_damage(src: Path, dst: Path) -> None:
    """Real corruption: XOR 512 bytes at the page boundary holding rows."""
    data = bytearray(src.read_bytes())
    idx = data.find(b"needle-10")
    assert idx > 0, "fixture rows not found in raw db bytes"
    page_start = (idx // 4096) * 4096
    for off in range(page_start, page_start + 512):
        data[off] ^= 0xFF
    dst.write_bytes(bytes(data))


# ---------------------------------------------------------------------------
# Part 1 — REGISTRY COMPLETENESS
# ---------------------------------------------------------------------------


def test_registry_symbols_exist():
    """Every registered symbol must still resolve; renames break loudly.

    A silent rename would orphan the coverage below while the suite stayed
    green — the exact failure mode that let #91839/#90806 regress guarded
    paths.
    """
    for op in REGISTRY:
        obj = _resolve(op)
        assert callable(obj), f"{op.module}.{op.qualname} is not callable"


def test_source_scan_finds_no_unregistered_maintenance_ops():
    """Grep the source modules for maintenance-shaped defs.

    Any new function whose name matches the maintenance pattern must be
    added to REGISTRY (with conformance coverage) or to _SCAN_EXEMPT (with a
    reviewed justification). Either way the addition is a conscious act.
    """
    registered = {op.qualname.split(".")[-1] for op in REGISTRY}
    found: set[str] = set()
    for rel in _SCAN_SOURCES:
        src = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        found.update(_MAINT_DEF_RE.findall(src))
    unregistered = found - registered - _SCAN_EXEMPT
    assert not unregistered, (
        "maintenance-shaped functions with no conformance coverage: "
        f"{sorted(unregistered)} — add them to REGISTRY (and write a "
        "live-writer/copy-honesty test) or to _SCAN_EXEMPT with a reason"
    )
    # And the scan itself must still see the registered names it pins,
    # otherwise the regex has rotted into vacuity.
    missing = {"vacuum", "rebuild_fts", "_safe_copy_db",
               "repair_state_db_schema"} - found
    assert not missing, f"source scan no longer sees {sorted(missing)}"


# ---------------------------------------------------------------------------
# Part 2 — LIVE-WRITER GUARD (in-place ops)
# ---------------------------------------------------------------------------

# op name -> callable(handle) -> outcome check. Each documented outcome is
# the op's refusal/degradation contract while a second connection holds a
# write transaction. "raises" pins loud refusal; dict/int outcomes pin the
# never-raise APIs degrading without touching the index/db structure.
_IN_PLACE_CASES = {
    "SessionDB.vacuum": "raises-operational-error",
    "SessionDB.maybe_auto_prune_and_vacuum": "returns-error-dict",
    "SessionDB.rebuild_fts": "returns-zero",
    "SessionDB.optimize_fts": "returns-zero",
}


@pytest.mark.requires_wal
@pytest.mark.parametrize("qualname", sorted(_IN_PLACE_CASES))
def test_in_place_op_refuses_or_degrades_under_live_writer(tmp_path, qualname):
    """#91839/#88235 class: structural work must not proceed under a holder.

    With a real second connection holding BEGIN IMMEDIATE, each in-place
    maintenance op must refuse or degrade per its documented contract, and
    the database must be integrity-clean and row-identical afterwards.
    """
    db = _make_state_db(tmp_path)
    _require_wal(db)
    handle = SessionDB(db_path=db)
    try:
        with _LiveWriter(db):
            outcome = _IN_PLACE_CASES[qualname]
            method = getattr(handle, qualname.split(".")[1])
            if outcome == "raises-operational-error":
                with pytest.raises(sqlite3.OperationalError) as exc_info:
                    method()
                msg = str(exc_info.value).lower()
                assert "locked" in msg or "busy" in msg
            elif outcome == "returns-error-dict":
                result = method(retention_days=0, min_interval_hours=0)
                assert result.get("vacuumed") is False, (
                    "auto-maintenance claims it vacuumed under a live writer"
                )
                assert result.get("error") or result.get("pruned") == 0
            elif outcome == "returns-zero":
                assert method() == 0, (
                    f"{qualname} claims it rewrote FTS structures while a "
                    "live writer held the database (#91839 class)"
                )
    finally:
        handle.close()
    assert _integrity_ok(db), f"{qualname} left state.db corrupt (#91839)"
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
    assert hermes_state._live_writer_holds_db(db) is False
    with _LiveWriter(db):
        assert hermes_state._live_writer_holds_db(db) is True, (
            "guard failed to detect a live write transaction — every "
            "registered op relying on it is now unguarded"
        )
    assert hermes_state._live_writer_holds_db(db) is False


# ---------------------------------------------------------------------------
# Part 3 — COPY HONESTY (#90613)
# ---------------------------------------------------------------------------


def test_copy_honesty_page_damage(tmp_path):
    """#90613 pin: success + corrupt destination is forbidden.

    The merged fix layers verify_sqlite_integrity over the raw copier:
    ``copy_db_and_verify`` must return False AND remove the destination when
    the copy is not integrity-clean. The invariant asserted is the class:
    never (returned True and destination invalid).
    """
    good = _make_state_db(tmp_path)
    corrupt = tmp_path / "corrupt.db"
    _corrupt_page_damage(good, corrupt)
    assert not verify_sqlite_integrity(corrupt, run_pragma=True)["valid"], (
        "fixture failed to actually corrupt the database"
    )

    dst = tmp_path / "copy.db"
    ok = copy_db_and_verify(corrupt, dst)
    dst_valid = dst.exists() and verify_sqlite_integrity(
        dst, run_pragma=True)["valid"]
    assert not (ok and not dst_valid), (
        "copy_db_and_verify returned success with a corrupt destination "
        "(#90613: the exact silent-bad-backup bug)"
    )
    # Pin the merged fix's concrete behavior: refuse + clean up.
    assert ok is False
    assert not dst.exists(), "failed copy left a corrupt destination behind"


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

    dst2 = tmp_path / "copy2.db"
    assert copy_db_and_verify(trunc, dst2) is False
    assert not dst2.exists()


def test_quick_snapshot_flags_corrupt_state_db(tmp_path, monkeypatch):
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

    snap_id = _create_quick_snapshot_locked(label="conformance",
                                            hermes_home=home)
    if snap_id is None:
        return  # refused outright: honest
    snap_dir = home / "state-snapshots" / snap_id
    import json
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
