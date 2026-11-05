"""DoctorAdmin operations of the SQLite store admin: the file's holders, health probes, integrity and journal mode,
the exclusive repair guard and schema repair, each on the one ``state.db`` the handle was opened for."""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from persistence.sqlite._state.dbfile import collect_state_db_stats, count_db_holders, iter_deleted_sqlite_sidecar_holders
from persistence.sqlite._state.holders import describe_holder_pid, foreign_state_db_holders, held_store_refusal, read_only_db_uri
from persistence.sqlite._state.repair import (
    _FTS_TABLES, _connect_repair_durable, _exclusive_repair_db_guard, _live_writer_holds_db, _schema_not_built,
    repair_state_db_schema, state_db_has_structural_damage,
)
from persistence.sqlite._state.wal import (
    _on_disk_journal_mode, _path_on_cross_vm_fs, _set_journal_mode_no_wait, is_sqlite_wal_reset_vulnerable,
)
from persistence import PermanentStoreError, StoreContention
from persistence.core.admin import WRITE_PROBE_SKIPPED, RepairGuard

from .._backend import translated
from .._schema import _is_fts5_unavailable_error
from .._schema_ddl import load_fts5_cjk_extension
from ._base import AdminFile

logger = logging.getLogger("gateway.lifecycle_ledger")

# Above this the snapshot copy a held store needs costs more than the probe is worth; should_fix still probes.
_WRITE_PROBE_SNAPSHOT_MAX_BYTES = 1 << 30

# The integrity check holds a phase-owned startup-watchdog lease renewed from a SQLite progress handler. Without a
# deadline the handler always returns 0 so it never aborts the check, and the ok/absent/first-complaint verdicts
# stay fail-closed. With a deadline the handler returns non-zero once it passes, so SQLite aborts the scan and the
# caller can bound a pathological store.
_INTEGRITY_CHECK_LEASE_PHASE = "state_db_unclean_integrity_check"
_INTEGRITY_CHECK_LEASE_S = 900.0
# VM instructions between progress-handler invocations; each invocation is a
# monotonic-clock compare, renewing at most every _INTEGRITY_CHECK_LEASE_RENEW_S.
_INTEGRITY_CHECK_PROGRESS_OPS = 100_000
_INTEGRITY_CHECK_LEASE_RENEW_S = 60.0

# SQLite file header: bytes 18 (write version) / 19 (read version) are 1 for rollback-journal, 2 for WAL.
_HEADER_VERSION_BYTES = {"delete": (1, 1), "wal": (2, 2)}
_MODE_BY_HEADER_VERSION = {v: k for k, v in _HEADER_VERSION_BYTES.items()}


def _db_opens_cleanly(db_path: Path, home: Path) -> Optional[str]:
    """Probe a DB on a fresh connection. Returns None if healthy, else a reason.

    Runs the first statement that trips the malformed-schema parse (``PRAGMA journal_mode``),
    ``integrity_check``, a ``sessions`` read, FTS5 MATCH probes and a rolled-back ``messages`` write, so FTS5
    index corruption (reads and ``integrity_check`` pass, every ``INSERT INTO messages`` fails through the FTS
    triggers) is reported as unhealthy.
    """
    conn = _connect_repair_durable(db_path)
    try:
        with contextlib.closing(conn):
            # Best-effort tokenizer load: messages_fts_cjk needs cjk_unicode61 before any statement can touch it;
            # tokenizer absence must never classify as corruption.
            load_fts5_cjk_extension(conn, home)
            conn.execute("PRAGMA journal_mode").fetchone()
            rows = conn.execute("PRAGMA integrity_check").fetchall()
            problems = [str(r[0]) for r in rows if r and str(r[0]).lower() != "ok"]
            if problems:
                return "; ".join(problems[:3])
            conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            # FTS5 read probe: partial shadow-table corruption makes MATCH/snippet/rank raise while integrity_check
            # reports healthy. MATCH '""' (empty phrase) parses, scans zero rows and exercises the shadow tables;
            # FTS5 rejects MATCH ''.
            for fts_table in _FTS_TABLES:
                try:
                    conn.execute(f"SELECT 1 FROM {fts_table} WHERE {fts_table} MATCH '\"\"' LIMIT 1").fetchone()
                except sqlite3.DatabaseError as exc:
                    # Builds without fts5/trigram raise "no such module|tokenizer"; calling that corruption would
                    # send the DB into repair, whose final fallback deletes messages_fts%. "no such table/column" =
                    # not built yet.
                    benign = _is_fts5_unavailable_error(exc) or _schema_not_built(exc)
                    if not (isinstance(exc, sqlite3.OperationalError) and benign):
                        # Partial shadow-table damage where MATCH / snippet / rank queries raise
                        # DatabaseError("database disk image is malformed") while reads of the FTS5 table itself
                        # parse fine.
                        return f"fts5 read probe failed on {fts_table}: {exc}"
            # FTS write probe: drive a row through the messages_fts* triggers in a transaction that is always
            # rolled back. The trigger INSERT alone only buffers the row in FTS5's in-memory segment; the
            # ``flush`` command writes that segment to ``<fts>_idx``/``_data`` exactly as a committed append
            # would, so a stale ``_idx`` row at the next segid (IntegrityError "constraint failed": integrity_check
            # and MATCH both clean, every real append fails) is hit here rather than by the user's next message.
            probe_session_id = f"_hermes_fts_health_probe_{time.time_ns()}"
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                             (probe_session_id, "_health_probe", time.time()))
                conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                             (probe_session_id, "user", "_fts_health_probe", time.time()))
                for fts_table in _FTS_TABLES:
                    try:
                        conn.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('flush')")
                    except sqlite3.OperationalError as exc:
                        if not (_is_fts5_unavailable_error(exc) or _schema_not_built(exc)):
                            raise
            except sqlite3.DatabaseError as exc:
                # IntegrityError is a DatabaseError sibling of OperationalError, not a child: catching only the
                # latter let the trigram-segment collision report "healthy".
                # Missing messages/sessions tables = brand new file mid-init, not corruption. "no such tokenizer":
                # this process lacks the cjk extension the DB's index needs: a capability gap; a tokenizer-less
                # store drops the triggers itself.
                if _schema_not_built(exc) or "no such tokenizer: cjk_unicode61" in str(exc).lower():
                    return None
                return f"fts5 write probe failed: {exc}"
            finally:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
            return None
    except sqlite3.DatabaseError as exc:
        return str(exc)


def _install_integrity_check_lease(conn: sqlite3.Connection, *, deadline_at: Optional[float] = None) -> None:
    """Claim the integrity-check lease and renew it from a SQLite progress handler.

    ``report_startup_progress`` is a no-op when no watchdog is armed and never raises.
    Without ``deadline_at`` the handler always returns 0: it must never abort the verdict
    PRAGMA. With ``deadline_at`` (a ``time.monotonic`` timestamp) it returns non-zero once
    that instant is reached, so SQLite raises ``OperationalError: interrupted`` and the
    caller can bound the scan. Synchronous by design: no checker worker thread can outlive
    the check.
    """
    from hermes_startup_watchdog import report_startup_progress

    report_startup_progress(_INTEGRITY_CHECK_LEASE_S, phase=_INTEGRITY_CHECK_LEASE_PHASE)
    last_renew = time.monotonic()

    def _on_progress() -> int:
        nonlocal last_renew
        now = time.monotonic()
        if now - last_renew >= _INTEGRITY_CHECK_LEASE_RENEW_S:
            last_renew = now
            report_startup_progress(_INTEGRITY_CHECK_LEASE_S, phase=_INTEGRITY_CHECK_LEASE_PHASE)
        if deadline_at is not None and now >= deadline_at:
            return 1
        return 0

    conn.set_progress_handler(_on_progress, _INTEGRITY_CHECK_PROGRESS_OPS)


def _header_mode(db_path: Path) -> str:
    """Journal mode as the on-disk header reports it (no SQLite connection is open in this process here)."""
    with open(db_path, "rb") as fh:
        head = fh.read(20)
    if len(head) < 20 or head[:16] != b"SQLite format 3\x00":
        return "not-a-database"
    return _MODE_BY_HEADER_VERSION.get((head[18], head[19]), f"unknown({head[18]}/{head[19]})")


def _refusal(target: str, current: str, *, on_cross_vm_fs: bool) -> Optional[str]:
    """Admission checks independent of holder discovery, which is enforced for every platform."""
    if current == "not-a-database" or current.startswith("unknown("):
        return (f"its file header reads {current}, so it is not a Hermes SQLite store this command can convert "
                "(pass --db PATH to point at one).")
    if target == "wal" and on_cross_vm_fs:
        return ("WAL shared memory silently corrupts on cross-VM filesystems (virtiofs/9p — Docker Desktop, "
                "OrbStack, Lima bind mounts). Keep this store on journal_mode=delete or move it to a native "
                "filesystem.")
    return None


def _watcher_mtime_ns(path: Path):
    """``st_mtime_ns`` of ``path``, or None when it cannot be stat'ed."""
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _newest_mtime_ns(paths) -> int | None:
    """Max ``st_mtime_ns`` across ``paths`` (unstat-able ignored); None when none stat'ed."""
    return max((m for m in map(_watcher_mtime_ns, paths) if m is not None), default=None)


class SQLiteRepairGuard(RepairGuard):
    """:class:`RepairGuard` on the exclusive repair connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def wal_checkpoint_passive(self) -> None:
        """Copy the WAL's committed frames into the database file without waiting on anyone."""
        with translated():
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")


class DoctorAdmin(AdminFile):
    def stats(self) -> dict[str, Any]:
        """Best-effort, strictly read-only stats snapshot of the file; never raises."""
        return collect_state_db_stats(self.db_path)

    def holder_count(self) -> Optional[int]:
        """Distinct PIDs holding the file open (a lower bound); ``None`` on any error or other host."""
        return count_db_holders(self.db_path)

    def deleted_sidecar_holders(self) -> list[tuple[int, str]]:
        """Processes holding an unlinked ``-wal`` / ``-shm`` sidecar of the file."""
        return iter_deleted_sqlite_sidecar_holders(self.db_path)

    def foreign_holders(self) -> list[tuple[int, str]]:
        """Foreign holders of the file or one of its WAL sidecars; a scan failure is an unknown holder."""
        return foreign_state_db_holders(self.db_path)

    def held_refusal(self, *, command: str, force_hint: Optional[str] = "--force") -> Optional[str]:
        """Operator-facing refusal for structural maintenance while another process holds the file; ``None`` when
        the store is provably quiet."""
        return held_store_refusal(self.db_path, command=command, force_hint=force_hint)

    def live_writer_holds(self) -> bool:
        """True when another process (or a connection outside this call) still holds the file, or that cannot
        be ruled out."""
        return _live_writer_holds_db(self.db_path)

    def opens_cleanly(self) -> Optional[str]:
        """None when the file opens and passes the integrity, read and rolled-back write probes, else a reason."""
        return _db_opens_cleanly(self.db_path, self.db_path.parent)

    def has_structural_damage(self) -> bool:
        """True when ``integrity_check`` names damage outside the FTS shadow tables."""
        return state_db_has_structural_damage(self.db_path)

    @contextmanager
    def exclusive_repair_guard(self) -> Iterator[RepairGuard]:
        """Hold one connection that excludes every other opener of the file for the block, yielding the operations
        that run on it; ``StoreContention`` (chaining the SQLite or OS error) when exclusion cannot be taken."""
        with _exclusive_repair_db_guard(self.db_path) as (guard, guard_error):
            if guard is None:
                raise StoreContention(str(guard_error)) from guard_error
            yield SQLiteRepairGuard(guard)

    def session_row_count(self) -> int:
        """``COUNT(*)`` of every session row, read on one ``mode=ro`` connection; a failed read raises the store
        error carrying SQLite's message."""
        with translated(), closing(sqlite3.connect(read_only_db_uri(self.db_path), uri=True)) as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def write_health_reason(self, *, should_fix: bool) -> Optional[str]:
        """FTS/write-health probe (a rolled-back BEGIN IMMEDIATE). Against a store a live writer holds,
        that probe is the second-writer class, so probe a read-only snapshot instead; a quiet
        store is probed in place. Returns the failure reason, None when healthy, or :data:`WRITE_PROBE_SKIPPED` when
        the held store is over 1 GB and ``should_fix`` is False. A failed snapshot copy raises the store error."""
        state_db_path = self.db_path
        if not _live_writer_holds_db(state_db_path):
            return _db_opens_cleanly(state_db_path, state_db_path.parent)
        if not should_fix and state_db_path.stat().st_size > _WRITE_PROBE_SNAPSHOT_MAX_BYTES:
            return WRITE_PROBE_SKIPPED
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, translated():
            snapshot = Path(tmp) / "state.db"
            src = sqlite3.connect(read_only_db_uri(state_db_path), uri=True, timeout=1.0)
            try:
                dest = sqlite3.connect(str(snapshot))
                try:
                    src.backup(dest)
                finally:
                    dest.close()
            finally:
                src.close()
            return _db_opens_cleanly(snapshot, state_db_path.parent)

    def integrity_check(self, *, deadline_s: Optional[float] = None) -> str:
        """``"ok"``, ``"absent"``, or the first ``quick_check`` complaint. Never raises.

        ``quick_check(1)`` stops at the first problem (about 2s on a healthy 500MB store). Opened
        normally: a WAL store needs its -shm sidecar for read-only, and the PRAGMA writes nothing.
        Holds a phase-owned startup-watchdog lease for the check duration so a long healthy check
        on a huge store is not mistaken for a parked deadlock.

        ``deadline_s`` (optional): abort the scan once that many seconds elapse and
        return ``"check-incomplete: exceeded <N>s"``; ``None`` never aborts it.
        """
        path = self.db_path
        if not path.exists():
            return "absent"
        deadline_at: Optional[float] = None
        if deadline_s is not None:
            try:
                deadline_at = time.monotonic() + max(0.0, float(deadline_s))
            except (TypeError, ValueError):
                deadline_at = None
        try:
            with closing(sqlite3.connect(str(path))) as conn:
                _install_integrity_check_lease(conn, deadline_at=deadline_at)
                row = conn.execute("PRAGMA quick_check(1)").fetchone()
        except Exception as exc:  # health: allow BLE001 -- every failure is reported as a string; the check never raises
            if (deadline_at is not None and isinstance(exc, sqlite3.OperationalError)
                    and "interrupt" in str(exc).lower()):
                limit = max(0.0, float(deadline_s)) if deadline_s is not None else 0.0
                return f"check-incomplete: exceeded {limit:g}s"
            return f"check-failed: {exc}"
        return "check-failed: no result" if not row or row[0] is None else str(row[0])

    def journal_mode(self) -> Optional[str]:
        """On-disk journal mode via one ``PRAGMA journal_mode`` read on a short-lived
        connection (no lock outlives it); ``None`` when undeterminable. Never raises."""
        try:
            with closing(sqlite3.connect(str(self.db_path), timeout=1.0)) as conn:
                return _on_disk_journal_mode(conn) or None
        except Exception:
            logger.debug("Could not read state.db journal mode", exc_info=True)
            return None

    def set_journal_mode(self, mode: str, *, force: bool = False) -> tuple[str, str]:
        """Convert the file to journal mode *mode* (``delete`` or ``wal``) offline; returns the mode its header
        reported before and the mode it reports afterwards (the same mode twice when it already was *mode*).

        Refuses with ``StoreContention`` while any other process holds the file or a sidecar (``force`` waives only
        an incomplete holder scan, never a process the scan found), or when an opener appears between the holder
        scan and the switch (the pragma runs without waiting, so a live writer's uncheckpointed commits are never
        switched away); every other refusal (no file, unreadable or foreign header, WAL on a cross-VM filesystem or
        on a SQLite with the WAL-reset bug, a header that disagrees after the switch) is a ``PermanentStoreError``
        carrying the reason."""
        self._require_writable("set_journal_mode")
        target = mode
        db_path = self.db_path
        if not db_path.exists():
            raise PermanentStoreError(f"No database at {db_path} (nothing to convert).")
        try:
            current = _header_mode(db_path)
        except OSError as exc:
            raise PermanentStoreError(f"Cannot read {db_path}: {exc}") from exc
        if current == target:
            return current, current
        refusal = _refusal(
            target, current, on_cross_vm_fs=target == "wal" and _path_on_cross_vm_fs(str(db_path))
        )
        if refusal:
            raise PermanentStoreError(f"Refusing to change the journal mode of {db_path}: {refusal}")
        if target == "wal" and is_sqlite_wal_reset_vulnerable():
            raise PermanentStoreError(
                f"Refusing to enable WAL: the linked SQLite {sqlite3.sqlite_version} has the WAL-reset bug "
                "(https://sqlite.org/wal.html#walresetbug). Upgrade to 3.51.3+ first.")
        holders = foreign_state_db_holders(db_path)
        if force:
            holders = [holder for holder in holders if holder[0] > 0]
        if holders:
            lines = [f"Refusing to change the journal mode of {db_path}: other processes hold it open "
                     "(a live switch would destroy their uncheckpointed commits). Stop them and re-run:"]
            lines += [f"    pid {pid} — {describe_holder_pid(pid) if pid > 0 else 'scan'}: {target_path}"
                      for pid, target_path in holders]
            raise StoreContention("\n".join(lines))
        # timeout=0: any opener that appeared between the scan and the flip makes the pragma fail with
        # 'database is locked' instead of sneaking the switch between a writer's transactions.
        try:
            conn = sqlite3.connect(str(db_path), timeout=0.0, isolation_level=None)
        except (OSError, sqlite3.Error) as exc:
            raise PermanentStoreError(f"Cannot open {db_path}: {exc}") from exc
        try:
            try:
                reported = _set_journal_mode_no_wait(conn, target.upper())
            except sqlite3.OperationalError as exc:
                raise StoreContention(
                    f"journal_mode={target.upper()} refused: {exc} (another opener appeared; nothing was changed)"
                ) from exc
        finally:
            conn.close()
        try:
            after = _header_mode(db_path)
        except OSError as exc:
            raise PermanentStoreError(f"Cannot re-read {db_path} after the switch: {exc}") from exc
        if after != target:
            raise PermanentStoreError(
                f"SQLite reported journal_mode={reported or '?'} but the file header reads {after}; not converted.")
        return current, after

    def repair_schema(self, *, backup: bool = True) -> dict[str, Any]:
        """Repair a malformed ``sqlite_master`` or FTS write corruption on a snapshot and copy the success back;
        returns ``{repaired, strategy, backup_path, error}`` (plus ``journal_mode_before`` on success)."""
        self._require_writable("repair_schema")
        with translated():
            return repair_state_db_schema(self.db_path, backup=backup)

    def change_stamp(self) -> Optional[int]:
        """Newest ``st_mtime_ns`` of the file and its ``-wal``; None when neither can be stat'ed."""
        db_path = self.db_path
        return _newest_mtime_ns((db_path, db_path.with_name(f"{db_path.name}-wal")))
