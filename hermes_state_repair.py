"""state.db repair, backup and writability preflight (split from hermes_state).

Patchable helpers are looked up as module globals at call time, so tests patch ``hermes_state_repair.<name>``.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import itertools
import json
import logging
import os
import shutil
import sqlite3
import stat
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from hermes_startup_watchdog import report_startup_progress
from hermes_state_common import (
    _acquire_db_flock, _clear_lock_holder_record, _describe_lock_holder, _read_lock_holder_record,
    is_advisory_lock_contention,
)

# Log-record parity with the origin module (caplog tests pin "hermes_state").
logger = logging.getLogger("hermes_state")

_REPAIR_LOCK_POLL_SECONDS = 0.1
# Snapshot copies are data transfer, not locking: bounded separately at 10 MiB/s (historical two-minute floor).
_REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND = 10 * 1024 * 1024
# ── Repair-loop bounding + dead-backup hygiene (#86747) ───────────────────── ``_claim_repair_attempt``
# above is an in-memory set: it bounds the loop only WITHIN one process. A corruption class the strategies
# cannot heal (b-tree page damage) failed repair on EVERY process start, and each pass took a fresh ~900MB
# forensic backup — 105 attempts / 89GB of identical dead copies in the reporting install. Two persistent
# bounds fix the class: * a sidecar attempt ledger (``<db>.repair-attempts.json``) that refuses further
# surgery after ``_MAX_PERSISTENT_REPAIR_ATTEMPTS`` failures on the SAME damaged file (fingerprint = size +
# a bounded content sample; any successful repair or replacement changes it and resets the count); * backup
# dedupe + a retention cap in ``_backup_db_file`` — an identical damaged file is never copied twice, and
# only the newest ``_MAX_MALFORMED_BACKUPS`` forensic copies are kept.
_MAX_PERSISTENT_REPAIR_ATTEMPTS = 3
_MAX_MALFORMED_BACKUPS = 3
# Sidecars copied with a damaged DB and pruned with it. ``-journal``: DELETE mode (the NFS/SMB/FUSE/ZFS and
# WAL-reset-bug fallback) leaves a hot journal whenever a transaction was open; without it the copy cannot roll back.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
# Head/tail bytes sampled by ``_db_fingerprint``: changes on any real repair/truncation/restore, O(1) on a multi-GB file.
_FINGERPRINT_SAMPLE_BYTES = 65536
# Header ranges that move on ordinary commits, not on repair — file change counter (24-27), version-valid-for
# (92-95) — masked out of the sample. A malformed-SCHEMA DB still accepts writes and DELETE mode writes the main
# file directly, so without the mask any live write re-keys the ledger and the repair budget resets to 1
# forever. The page-1 sqlite_master b-tree (repair identity) sits after byte 100.
_FINGERPRINT_VOLATILE_HEADER_RANGES = ((24, 28), (92, 96))
# Headroom for the forensic backup (a full raw copy; a repair loop on a large state.db is a disk amplifier).
# Proportional, not a flat multi-GB floor: a refused backup is a HARD STOP, and a big reserve would make repair
# never run on small volumes.
# The backup is a full raw copy of the damaged DB (plus its -wal/-shm sidecars), so a repair loop on a large
# state.db is a disk amplifier: the reporting incident wrote ~98MB every ~10s until the volume was nearly
# full, which would have taken down every agent on the host. Require the copy itself plus a small slice of
# the volume, clamped to a modest floor. See #69603.
_REPAIR_BACKUP_MIN_FREE_BYTES = 256 * 1024 * 1024  # 256 MiB absolute floor
_REPAIR_BACKUP_FREE_FRACTION = 0.02  # plus 2% of the volume
_FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")
_MANUAL_RECOVER_HINT = ("Free disk space, then retry (or recover manually with "
                        "`hermes sessions recover --source {db_path} --inspect-only` first).")


def _sidecars(db_path: Path):
    """The three SQLite sidecar paths for *db_path* (present or not)."""
    return (db_path.with_name(db_path.name + suffix) for suffix in _DB_SIDECAR_SUFFIXES)


def _read_offline(db_path: Path, what: str, reader) -> Optional[str]:
    """Run *reader()* under ``hermes_cli.sqlite_safe_read.offline_file_access``.

    ``close()`` on ANY raw descriptor cancels every POSIX advisory lock this process holds on the file,
    including a peer connection's RESERVED lock (``sqlite_safe_read`` rule 1), so a raw read is only safe with
    no live connection; ``None`` when that makes it unsafe or the file is unreadable. Scaffold/embed installs
    without hermes_cli have no tracked connections."""
    try:
        from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access
    except ImportError:
        offline_file_access, LiveConnectionError = (lambda _p, **_k: contextlib.nullcontext()), OSError
    try:
        with offline_file_access(db_path, what=what):
            return reader()
    except (LiveConnectionError, OSError):
        return None


def _claim_repair_attempt(db_path: Path) -> bool:
    """Claim the one-shot per-process repair attempt for *db_path*: True for the first caller, False
    afterwards (bounds the repair/reopen loop and stops concurrent callers racing surgery on one file)."""
    from hermes_state import _repair_attempt_lock, _repair_attempted_paths
    with _repair_attempt_lock:
        if str(db_path) in _repair_attempted_paths:
            return False
        _repair_attempted_paths.add(str(db_path))
        return True


def _open_lock_file(db_path: Path, suffix: str, what: str, tail: str):
    """Open ``<db>.<suffix>`` for locking; on failure warn and return None."""
    lock_path = db_path.with_name(db_path.name + suffix)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        return lock_path, lock_path.open("a+b")
    except OSError as exc:
        logger.warning(f"Could not open state.db {what} lock %s (%s) — {tail}", lock_path, exc)
        return lock_path, None


def _msvcrt_lock(handle, flag_name: str) -> None:
    import msvcrt
    handle.seek(0)
    msvcrt.locking(handle.fileno(), getattr(msvcrt, flag_name), 1)  # type: ignore[attr-defined]


def _try_lock_nonblocking(handle) -> None:
    """Take the advisory lock on *handle* without waiting (raises on contention)."""
    from hermes_state import _IS_WINDOWS
    if _IS_WINDOWS:
        _msvcrt_lock(handle, "LK_NBLCK")
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_lock_handle(handle, *, clear_record: bool = False) -> None:
    """Drop the advisory lock on *handle* (best effort) and close it."""
    from hermes_state import _IS_WINDOWS
    with contextlib.closing(handle), contextlib.suppress(OSError):  # best-effort release; always close
        if _IS_WINDOWS:
            _msvcrt_lock(handle, "LK_UNLCK")
        else:
            import fcntl
            if clear_record:
                _clear_lock_holder_record(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire_repair_lock_windows(lock_path: Path, handle, timeout: float):
    """Windows counterpart of ``_acquire_db_flock``: True / False (timed out) / None (non-contention error)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            _try_lock_nonblocking(handle)
            return True
        except (BlockingIOError, OSError) as exc:
            if not is_advisory_lock_contention(exc):
                logger.warning("Could not acquire state.db repair lock %s (%s) — skipping schema surgery on a "
                               "non-contention error.", lock_path, exc)
                return None
            if time.monotonic() >= deadline:
                return False
            time.sleep(_REPAIR_LOCK_POLL_SECONDS)


@contextlib.contextmanager
def _cross_process_repair_lock(db_path: Path):
    """Serialize state.db schema surgery across processes.

    Yields True when this process holds the repair lock, False when the bounded acquire timed out or the lock
    file could not be opened; on False the caller must NOT do surgery (unlocked surgery IS the interleaving
    this prevents). ``flock``: the kernel drops it when the holder dies (a pidfile would wedge every future
    repair); a forked child inheriting the fd is the exception, so the acquire records pid + start time and
    breaks a provably dead holder's lock. Bounded because a live repairer can sit in ``VACUUM`` for minutes.
    An unopenable lock file (no space/inodes/descriptors) fails closed too: a sibling that opened ITS handle
    before the disk filled may be inside surgery.

    The acquire is still bounded because a *live* repairer can legitimately sit in ``VACUUM`` for minutes on
    a large DB, and an unbounded wait would hang the caller's open with no traceback (the failure shape of
    #36644).
    """
    from hermes_state import _IS_WINDOWS, _REPAIR_LOCK_TIMEOUT_SECONDS
    lock_path, handle = _open_lock_file(
        db_path, ".repair.lock", "repair", "skipping schema surgery rather than running it without cross-process authority.")
    if handle is None:
        yield False
        return
    acquired = False
    try:
        if _IS_WINDOWS:
            acquired = _acquire_repair_lock_windows(lock_path, handle, _REPAIR_LOCK_TIMEOUT_SECONDS)
        else:
            acquired, handle = _acquire_db_flock(str(lock_path), handle, _REPAIR_LOCK_TIMEOUT_SECONDS,
                                                 _REPAIR_LOCK_POLL_SECONDS, "state.db repair lock")
        if acquired is None:
            acquired = False  # non-contention failure already logged with its errno
        elif not acquired:
            logger.warning("state.db repair lock %s held by another process for more than %.0fs — skipping schema "
                           "surgery in this process to avoid racing the repairer. Recorded holder: %s.",
                           lock_path, _REPAIR_LOCK_TIMEOUT_SECONDS,
                           _describe_lock_holder(None if _IS_WINDOWS else _read_lock_holder_record(handle)))
        yield acquired
    finally:
        if acquired:
            _release_lock_handle(handle, clear_record=True)
        else:
            handle.close()


def _try_acquire_auto_maintenance_lock(db_path: Path) -> Optional[Any]:
    """Non-blocking cross-process lock for one auto-maintenance pass (None = skip the pass: otherwise two startups
    both pass the interval check and the second prunes a row the first has only just closed recoverably)."""
    _lock_path, handle = _open_lock_file(db_path, ".auto-maintenance.lock", "auto-maintenance", "skipping automatic maintenance.")
    try:
        if handle is not None:
            _try_lock_nonblocking(handle)
        return handle
    except (BlockingIOError, OSError):
        handle.close()
        return None


_release_auto_maintenance_lock = _release_lock_handle  # release a _try_acquire_auto_maintenance_lock handle


def _bump_schema_cookie(conn: sqlite3.Connection) -> None:
    """Increment the schema cookie after direct ``sqlite_master`` surgery.

    Ordinary DDL bumps it and peers compare it before running a prepared statement (how they discard a cached
    schema); ``writable_schema=ON`` edits do NOT, so live connections would keep firing triggers into
    ``messages_fts*`` shadow tables that no longer exist. Best-effort, never raises."""
    try:
        current = conn.execute("PRAGMA schema_version").fetchone()[0]
        # Wrap within SQLite's 32-bit signed range; peers compare for equality.
        conn.execute(f"PRAGMA schema_version={(int(current) + 1) & 0x7FFFFFFF}")
    except (sqlite3.DatabaseError, TypeError, IndexError) as exc:
        logger.warning("Could not bump state.db schema cookie: %s", exc)


def _mask_volatile_header(head: bytes) -> bytes:
    """Zero the commit-counter fields so ordinary writes don't re-key the ledger."""
    if len(head) < 96:
        return head
    buf = bytearray(head)
    for start, end in _FINGERPRINT_VOLATILE_HEADER_RANGES:
        buf[start:end] = b"\x00" * (end - start)
    return bytes(buf)


def _repair_backup_headroom_bytes(total_bytes: int) -> int:
    """Free space required *beyond* the copy itself, for a volume of *total_bytes*."""
    return max(_REPAIR_BACKUP_MIN_FREE_BYTES, int(total_bytes * _REPAIR_BACKUP_FREE_FRACTION))


def _disk_budget(db_path: Path, refusal: str) -> "Tuple[Optional[str], int, int, int]":
    """``(error, bundle_bytes, free_bytes, headroom_bytes)`` for *db_path*'s volume (main file plus every PRESENT
    sidecar); *error* is set (and the sizes zero) on stat()/disk_usage() failure. Fails CLOSED: the nearly-full
    volume these guards exist for is exactly where they are most likely to fail."""
    try:
        need = db_path.stat().st_size + sum(p.stat().st_size for p in _sidecars(db_path) if p.exists())
        usage = shutil.disk_usage(db_path.parent)
    except OSError as exc:
        return (f"could not determine free space on {db_path.parent} ({exc}); refusing the {refusal} rather than risk "
                "filling the volume"), 0, 0, 0
    return None, need, usage.free, _repair_backup_headroom_bytes(usage.total)


def _repair_scratch_space_error(db_path: Path) -> Optional[str]:
    """Return an error unless snapshot, VACUUM and promotion can fit safely."""
    error, snapshot_bytes, free, headroom = _disk_budget(db_path, "repair snapshot")
    if error is not None:
        return error
    # VACUUM on the staged DB may need up to 2x the database size (SQLite docs); the same reserve then covers
    # transactional promotion into the live DB.
    if free >= snapshot_bytes + (2 * snapshot_bytes) + headroom:
        return None
    return (f"only {free / 1e9:.2f}GB free on {db_path.parent}; the repair snapshot needs up to "
            f"{snapshot_bytes / 1e9:.2f}GB, VACUUM may need another {(2 * snapshot_bytes) / 1e9:.2f}GB, and "
            f"{headroom / 1e9:.2f}GB must remain as headroom. Free disk space, then retry.")


def _backup_free_space_error(db_path: Path) -> Optional[str]:
    """Disk guard for the forensic copy: reason to refuse, or None. A full raw copy on a nearly-full volume (which a
    preceding repair loop may itself have caused) can finish off the disk and every process on the machine."""
    hint = _MANUAL_RECOVER_HINT.format(db_path=db_path)
    error, need, free, headroom = _disk_budget(db_path, "forensic copy")
    if error is not None:
        return f"{error}. {hint}"
    if free - need >= headroom:
        return None
    return (f"only {free / 1e9:.2f}GB free on {db_path.parent}; copying the damaged DB needs {need / 1e9:.2f}GB and must "
            f"leave {headroom / 1e9:.2f}GB headroom. {hint}")


def _repair_snapshot_timeout_seconds(source_path: Path) -> float:
    """Bound one SQLite snapshot by source size incl. sidecars (a WAL can hold committed rows not yet in the
    main file), so a healthy large-database copy is not cut off by the repair-lock timeout."""
    from hermes_state import _REPAIR_LOCK_TIMEOUT_SECONDS
    source_bytes = 0
    for candidate in (source_path, *_sidecars(source_path)):
        with contextlib.suppress(FileNotFoundError):  # a sidecar may vanish mid-walk
            source_bytes += candidate.stat().st_size
    return max(_REPAIR_LOCK_TIMEOUT_SECONDS, source_bytes / _REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND)


def _repair_failure_consumes_attempt(exc: BaseException) -> bool:
    """Whether a pre-strategy SQLite failure proves deterministic corruption.

    Lock contention, timeouts, disk-full and I/O failures are environmental — a retry may succeed, so they
    must not burn the repair ledger. Only SQLite's corruption/image result codes prove deterministic damage."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    if isinstance(error_code := getattr(exc, "sqlite_errorcode", None), int):
        # Extended result codes keep the primary code in the low byte.
        return (error_code & 0xFF) in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)
    # Older sqlite3 without result-code attributes: narrow message match only, never turning generic
    # "disk is full"/"readonly" into permanent failures.
    return any(m in str(exc).lower() for m in ("file is not a database", "database disk image is malformed"))


def _repair_ledger_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".repair-attempts.json")


def _db_fingerprint(db_path: Path) -> "Optional[str]":
    """Cheap identity for a damaged DB file: size + a bounded content sample.

    EXCLUDES mtime: a malformed-schema DB still accepts writes, so live writers, checkpoints and the
    strategies move mtime between passes; keyed on mtime every pass looked like a NEW file and the attempt
    counter reset forever. Hashing a multi-GB file per open is the cost this ledger avoids, so sample the
    head/tail slices any real repair, truncation or restore must change.

    ``None`` = identity unavailable (:func:`_read_offline`; a live peer is expected here since this runs
    BEFORE ``_backup_db_file``'s live-connection guard). Callers MUST NOT substitute a differently-shaped key:
    the ledger compares for equality, so alternating shapes never match and the unbounded loop returns."""
    def _sample() -> str:
        st = db_path.stat()
        with open(db_path, "rb") as fh:
            head = fh.read(_FINGERPRINT_SAMPLE_BYTES)
            fh.seek(max(0, st.st_size - _FINGERPRINT_SAMPLE_BYTES))
            tail = fh.read(_FINGERPRINT_SAMPLE_BYTES) if st.st_size > _FINGERPRINT_SAMPLE_BYTES else b""
        return f"{st.st_size}:{hashlib.sha256(_mask_volatile_header(head) + tail).hexdigest()[:32]}"
    return _read_offline(db_path, "fingerprint", _sample)


def _backup_content_identity(db_path: Path) -> "Optional[str]":
    """Recovery-image identity for forensic-backup dedupe: whole file + sidecars.

    A DIFFERENT relation from :func:`_db_fingerprint` (never conflate them): the fingerprint answers "same
    repair epoch?" from head+tail only, and a live writer can commit into an *interior* page while preserving
    size and both 64 KiB slices — reusing a backup on that basis hands the operator a snapshot predating real
    user data. A forensic copy must claim byte identity, so this digests the ENTIRE main file plus every
    present sidecar (the WAL can hold uncheckpointed frames). ``None`` when a live connection makes the read
    unsafe (:func:`_read_offline`) — the caller then takes a fresh backup, never a false reuse.

    See #87409.
    """
    def _digest() -> str:
        hasher = hashlib.sha256()
        members = [("main", db_path), *((sfx, p) for sfx, p in zip(_DB_SIDECAR_SUFFIXES, _sidecars(db_path)) if p.exists())]
        for label, path in members:
            # Length-delimit every member (main file included) so the concatenation is prefix-free; otherwise a
            # main-file tail could coincide with a main+sidecar split and dedupe two images together.
            hasher.update(f"\0{label}:{path.stat().st_size}\0".encode())
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    hasher.update(chunk)
        return hasher.hexdigest()
    return _read_offline(db_path, "backup-identity", _digest)


def _read_repair_ledger(db_path: Path) -> "Dict[str, Any]":
    with contextlib.suppress(OSError, ValueError):
        raw = json.loads(_repair_ledger_path(db_path).read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    return {}


def _persistent_repair_attempts_exhausted(db_path: Path) -> bool:
    """Whether *db_path* has already burned its cross-restart repair budget.

    True only when the ledger records ``_MAX_PERSISTENT_REPAIR_ATTEMPTS`` failures against the CURRENT
    fingerprint. Never raises; a missing/corrupt ledger or unstatable DB reads as "not exhausted" (the
    in-process claim and cross-process lock still bound one run). Fingerprint unavailable (live connection) ->
    compare the SIZE the ledger recorded, otherwise a peer connection hides an exhausted budget on every pass."""
    ledger = _read_repair_ledger(db_path)
    recorded = ledger.get("fingerprint")
    fp = _db_fingerprint(db_path)
    try:  # size is the one key component that needs no raw read
        same = recorded == fp if fp is not None else (
            isinstance(recorded, str) and recorded.startswith(f"{db_path.stat().st_size}:"))
    except OSError:
        return False
    return same and int(ledger.get("failed_attempts", 0)) >= _MAX_PERSISTENT_REPAIR_ATTEMPTS


def _persistent_repair_exhausted_error(db_path: Path) -> str:
    """The stable operator-facing diagnostic for an exhausted repair budget."""
    return (f"automatic repair has already failed {_MAX_PERSISTENT_REPAIR_ATTEMPTS} times on this exact file — the "
            f"corruption is beyond the schema/FTS repair strategies (likely b-tree page damage). Manual recovery "
            f"required: restore a backup, or salvage with `hermes sessions recover --source {db_path} "
            f"--inspect-only`, then (if it reports recoverable) `hermes sessions recover --source {db_path} "
            f"--output recovered-state.db` (recovery snapshots the damaged file first, then runs the page-level "
            f"`.recover` lane on the copy; do NOT point a raw `sqlite3` shell at the live database). "
            f"Delete {_repair_ledger_path(db_path).name} to force another automatic attempt.")


def _record_repair_outcome(db_path: Path, *, repaired: bool, fingerprint: "Optional[str]" = None) -> None:
    """Update the persistent attempt ledger after a repair pass. Never raises.

    Keys on the post-attempt fingerprint (what the NEXT exhaustion probe sees). If a live connection makes it
    unavailable, keep the recorded key and still increment — dropping the pass lets a peer reset the budget
    every time. Never write a differently shaped key."""
    ledger_path = _repair_ledger_path(db_path)
    try:
        if repaired:
            ledger_path.unlink(missing_ok=True)
            return
        ledger = _read_repair_ledger(db_path)
        recorded = ledger.get("fingerprint")
        fp = fingerprint if fingerprint is not None else _db_fingerprint(db_path)
        if fp is None:
            if not isinstance(recorded, str):
                # No prior key to extend and no safe way to mint one; the in-process claim and cross-process lock
                # still bound this run.
                return
            fp = recorded
        attempts = int(ledger.get("failed_attempts", 0)) + 1 if recorded == fp else 1
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        ledger_path.write_text(json.dumps({"fingerprint": fp, "failed_attempts": attempts, "last_attempt": stamp}), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not update state.db repair ledger: %s", exc)


def _existing_malformed_backups(db_path: Path) -> "List[Path]":
    """Timestamped forensic backups of *db_path*, newest first."""
    prefix = f"{db_path.name}.malformed-backup-"
    try:
        found = [p for p in db_path.parent.iterdir() if p.name.startswith(prefix) and not p.name.endswith(_DB_SIDECAR_SUFFIXES)]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.name, reverse=True)


def _prune_malformed_backups(db_path: Path, keep: int = _MAX_MALFORMED_BACKUPS) -> None:
    """Delete all but the *keep* newest forensic backups (and sidecars)."""
    for stale in _existing_malformed_backups(db_path)[keep:]:
        for victim in (stale, *_sidecars(stale)):
            try:
                victim.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning("Could not prune stale DB backup %s: %s", victim, exc)


def _publish_backup_bundle(db_path: Path, staging: Path, backup_path: Path) -> None:
    """Copy DB + sidecars to *staging* names, then rename each into place.

    ORDER MATTERS: the main DB name is the bundle's commit marker (what ``_existing_malformed_backups``
    counts), so sidecars publish FIRST and the main DB LAST — a failure partway never leaves a countable
    backup over a missing sidecar. On failure, staging files AND anything promoted are removed."""
    main = (db_path, staging, backup_path)
    sidecars = [(sidecar, staging.with_name(staging.name + suffix), backup_path.with_name(backup_path.name + suffix))
                for suffix, sidecar in zip(_DB_SIDECAR_SUFFIXES, _sidecars(db_path)) if sidecar.exists()]
    published: "List[Path]" = []
    try:
        for src, staged, _dst in (main, *sidecars):
            shutil.copy2(src, staged)
        for _src, staged, dst in (*sidecars, main):
            os.replace(staged, dst)
            published.append(dst)
    except Exception:
        for victim in (*(staged for _s, staged, _d in (*sidecars, main)), *published):
            with contextlib.suppress(OSError):  # best effort; a failure here is never the caller's error
                victim.unlink(missing_ok=True)
        raise


def _backup_db_file(db_path: Path) -> "Tuple[Optional[Path], Optional[str]]":
    """Raw-copy a (possibly malformed) DB plus sidecars to a timestamped backup.

    Raw bytes on purpose: the DB won't open cleanly, so preserve them exactly for forensics. Returns ``(backup_path,
    None)`` or ``(None, reason)``; repair treats a refused backup as a HARD STOP because the bundle is the recovery
    path when every strategy fails. Refuses while a connection to this DB is live in the process: the raw read would
    ``close()`` a descriptor and cancel that connection's POSIX advisory locks (``hermes_cli.sqlite_safe_read``) —
    real case: one SessionDB enters repair while the gateway holds others.

    Dedupe: reuse the newest backup when byte-identical to the current recovery image (``_backup_content_identity``
    — NOT mtime, NOT ``_db_fingerprint``); a repair loop once re-copied the same bytes on every restart. Staging
    names live OUTSIDE the ``.malformed-backup-`` prefix: inside it they count as a backup, sort NEWEST (prune kept
    partials, deleted intact copies) and dedupe could return one with no real forensic copy on disk.

    Refusal reasons (``_backup_free_space_error`` / ``_MANUAL_RECOVER_HINT``) point operators at the safe lane,
    `hermes sessions recover --source <db> --inspect-only`, never at a raw sqlite3 shell on the live file.

    See #69603.
    """
    with contextlib.suppress(ImportError):  # scaffold/embed installs without hermes_cli track no connections
        from hermes_cli.sqlite_safe_read import has_live_connection
        if has_live_connection(db_path):
            reason = (f"a connection to {db_path} is still open in this process; raw-copying it would cancel that "
                      "connection's POSIX advisory locks. Close all SessionDB handles first.")
            logger.error("Refusing to raw-copy %s for backup: %s", db_path, reason)
            return None, reason
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.malformed-backup-{stamp}")
    for seq in itertools.count(1):  # same-second collision must not overwrite the earlier forensic copy
        if not backup_path.exists():
            break
        backup_path = db_path.with_name(f"{db_path.name}.malformed-backup-{stamp}_{seq}")
    try:
        # Sweep staging debris from an interrupted pass BEFORE the dedupe (it is byte-identical to the damaged
        # DB, so dedupe would hand it back as a backup). Also the old ``.incomplete`` spelling, which
        # prefix-matches as a backup, sorts NEWEST and would otherwise survive prune forever.
        for pattern in (f"{db_path.name}.backup-staging-*", f"{db_path.name}.malformed-backup-*.incomplete*"):
            for old in db_path.parent.glob(pattern):
                with contextlib.suppress(OSError):
                    old.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            # Hash the source only when there is a candidate: hashing a multi-GB file before copying it is waste.
            if newest := _existing_malformed_backups(db_path)[:1]:
                src_id = _backup_content_identity(db_path)
                if src_id is not None and _backup_content_identity(newest[0]) == src_id:
                    logger.info("Reusing existing forensic backup %s (identical to the damaged DB).", newest[0])
                    return newest[0], None
        if (reason := _backup_free_space_error(db_path)) is not None:
            logger.error("Refusing forensic backup of %s: %s", db_path, reason)
            return None, reason
        _publish_backup_bundle(db_path, db_path.with_name(f"{db_path.name}.backup-staging-{stamp}"), backup_path)
        # Retention cap (#86747): keep only the newest few forensic copies.
        _prune_malformed_backups(db_path)
        return backup_path, None
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not back up malformed DB %s: %s", db_path, exc)
        return None, f"backup copy failed: {exc}"


def preflight_db_writability(db_path: Path, *, db_label: str = "state.db") -> None:
    """Refuse-or-repair read-only DB files BEFORE the first connection opens.

    A stray read-only ``state.db`` / ``-wal`` / ``-shm`` (sudo run, restored backup, copied dotfiles) otherwise
    surfaces as an opaque "attempt to write a readonly database" inside ``_init_schema``, and the obvious wrong
    "fix" (deleting the ``-wal``) loses committed transactions. ``chmod u+rw`` repair only inside the Hermes home
    tree (Hermes owns those files; ``chmod`` fails on files the user doesn't own, bounding the repair exactly);
    otherwise fail fast naming the file and command. Never deletes/truncates a WAL sidecar — once writable, the
    normal open checkpoints it. ``:memory:``/``file:`` skipped. Shared with ``kanban_db``.

    Port of Kilo-Org/kilocode#12508's startup preflight. This preflight:
    """
    if str(db_path) == ":memory:" or str(db_path).startswith("file:"):
        return
    home: Optional[Path] = None
    with contextlib.suppress(Exception):  # pragma: no cover - defensive
        home = Path(get_hermes_home()).resolve()
    # SQLite needs a writable directory in every journal mode (WAL/SHM sidecars, or the DELETE-mode journal).
    sidecars = (db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm"))
    for p, is_dir in [(db_path.parent, True), *((p, False) for p in (db_path, *sidecars) if p.is_file())]:
        if (is_dir and not p.is_dir()) or os.access(p, os.R_OK | os.W_OK):
            continue
        x = "x" if is_dir else ""
        in_scope = False
        with contextlib.suppress(OSError, ValueError):
            in_scope = home is not None and p.resolve().is_relative_to(home)
            if in_scope:
                os.chmod(p, p.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if is_dir else 0))
        if in_scope and os.access(p, os.R_OK | os.W_OK):
            logger.info("%s preflight: repaired read-only %s (chmod u+rw%s)", db_label, p, x)
            continue
        wal_note = (" Do NOT delete the -wal file — it contains committed data that "
                    "will be merged into the database once it is writable." if p.name.endswith("-wal") else "")
        raise sqlite3.OperationalError(
            f"{db_label} is not writable: {'directory' if is_dir else 'file'} {p} is read-only for this user. Hermes "
            f"needs read-write access to open the database. Fix with: chmod u+rw{x} '{p}' (files owned by another "
            f"user may need sudo/chown).{wal_note}")


def _connect_repair_durable(db_path: Path, *, timeout: float = 5.0) -> sqlite3.Connection:
    """``sqlite3.connect`` for the repair/probe paths, with macOS write barriers.

    These paths bypass ``SessionDB``/:func:`apply_wal_with_fallback`, so they inherited ``synchronous=NORMAL`` and
    no ``checkpoint_fullfsync`` — on Darwin an interrupted ``REINDEX``/``VACUUM``/``writable_schema`` rewrite leaves
    half-written b-tree pages. Autocommit (``isolation_level=None``): DDL and ``VACUUM`` are illegal inside an
    implicit transaction. Barriers are best-effort: on a malformed schema even ``PRAGMA synchronous=FULL`` raises,
    so whole-file rewrites call :func:`_reapply_durability_barriers` once the schema parses again."""
    conn = sqlite3.connect(str(db_path), timeout=timeout, isolation_level=None)
    _reapply_durability_barriers(conn)
    return conn


def _repair_conn(db_path: Path, *, timeout: float = 5.0):
    """A :func:`_connect_repair_durable` connection as a context manager, closed on exit."""
    return contextlib.closing(_connect_repair_durable(db_path, timeout=timeout))


def _reapply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Best-effort (re)application of the macOS write barriers; True if accepted.
    Call before ``VACUUM``/``REINDEX`` once the schema parses: a connection opened
    on a malformed schema could not take them at open time. Never raises."""
    from hermes_state_wal import _apply_macos_checkpoint_barrier, _enforce_macos_synchronous_full
    try:
        _apply_macos_checkpoint_barrier(conn)
        _enforce_macos_synchronous_full(conn)
        return True
    except Exception:  # schema still unparseable, or anything else — pragmas cannot be set yet
        return False


def apply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Durability barriers for guest users of ``state.db`` that must inherit its owner's journal mode. Also
    applies the configured ``database.synchronous`` level, a per-connection pragma that otherwise only
    rides on the journal-mode setup path guests must not run."""
    from hermes_state_wal import _apply_synchronous_pragma
    ok = _reapply_durability_barriers(conn)
    with contextlib.suppress(Exception):
        from hermes_cli.config import cfg_get, load_config_readonly  # local: avoids an import cycle
        if (raw_synchronous := cfg_get(load_config_readonly(), "database", "synchronous", default=None)) is not None:
            _apply_synchronous_pragma(conn, raw_synchronous, db_label="state.db (guest)")
    return ok


def _close_unpinned(conn: sqlite3.Connection) -> None:
    """Leave EXCLUSIVE locking mode (so the file is never left pinned) and close."""
    with contextlib.suppress(Exception):
        conn.execute("PRAGMA locking_mode=NORMAL")
    conn.close()


def _open_exclusive(db_path: Path, begin: str) -> sqlite3.Connection:
    """Zero-timeout connection holding ``locking_mode=EXCLUSIVE`` after a rolled-back
    *begin*; closed (unpinned) and re-raised when exclusion cannot be taken."""
    conn = _connect_repair_durable(db_path, timeout=0.0)
    try:
        for statement in ("PRAGMA locking_mode=EXCLUSIVE", begin, "ROLLBACK"):
            conn.execute(statement)
    except BaseException:
        _close_unpinned(conn)
        raise
    return conn


@contextlib.contextmanager
def _exclusive_repair_db_guard(db_path: Path):
    """Yield ``(conn, None)`` — one live connection that excludes writers for repair surgery — or ``(None,
    exc)`` when exclusion could not be taken.

    ``locking_mode=EXCLUSIVE`` retains file-level exclusion after the short ``BEGIN EXCLUSIVE`` is rolled
    back; the rollback is essential because ``Connection.backup`` uses this connection as *source* and later
    as the promotion *destination*, both transaction-free. It stays open across the snapshot -> strategies ->
    promotion window so no writer can commit a change promotion would overwrite. Existing readers make
    acquisition fail rather than being disturbed (fail closed). Timeout 0: the cross-process lock already
    serializes repairers, and a partial repair is less safe than "stop the gateway and retry"."""
    try:
        guard = _open_exclusive(db_path, "BEGIN EXCLUSIVE")
    except (sqlite3.Error, OSError) as exc:
        yield None, exc
        return
    try:
        yield guard, None
    finally:
        # Releasing the exclusive locks before close keeps a close-time checkpoint from being mistaken for a
        # repair write by callers that reopen immediately.
        _close_unpinned(guard)


def _copy_database_snapshot(source_path: Path, destination_path: Path, *,
                            source_connection: Optional[sqlite3.Connection] = None,
                            destination_connection: Optional[sqlite3.Connection] = None) -> None:
    """Copy one complete SQLite snapshot without replacing either file inode: the online backup API folds
    committed WAL frames into the source snapshot and writes the destination in one transaction (rolled
    back if interrupted), so ``state.db`` is never swapped out from under handles that refer to it."""
    # Deadline first: a sidecar vanishing mid-stat must not leak a just-opened descriptor.
    deadline_seconds = _repair_snapshot_timeout_seconds(source_path)
    deadline = time.monotonic() + deadline_seconds

    def _check_deadline(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out copying SQLite repair snapshot after {deadline_seconds:.0f}s")

    with contextlib.ExitStack() as owned:  # closes only connections opened here, destination first
        source = source_connection or owned.enter_context(_repair_conn(source_path))
        if destination_connection is not None and destination_connection.in_transaction:
            # sqlite3_backup needs a transaction-free destination (the guard holds exclusion via locking_mode).
            raise sqlite3.ProgrammingError("SQLite repair backup destination has an active transaction")
        destination = destination_connection or owned.enter_context(_repair_conn(destination_path))
        source.backup(destination, pages=256, progress=_check_deadline, sleep=_REPAIR_LOCK_POLL_SECONDS)


def _schema_not_built(exc: BaseException) -> bool:
    """``no such table/column``: FTS5 / core tables not created yet (brand new file mid-init)."""
    return any(m in str(exc).lower() for m in ("no such table", "no such column"))


def _db_opens_cleanly(db_path: Path) -> Optional[str]:
    """Probe a DB on a fresh connection. Returns None if healthy, else a reason.

    Runs the first statement that trips the malformed-schema parse (``PRAGMA journal_mode``),
    ``integrity_check``, a ``sessions`` read, FTS5 MATCH probes and a rolled-back ``messages`` write — so FTS5
    index corruption (reads and ``integrity_check`` pass, every ``INSERT INTO messages`` fails through the FTS
    triggers) is reported as unhealthy.

    See #50502.
    """
    from hermes_state import SessionDB, load_fts5_cjk_extension
    # ── Strategy 0.5: rebuild stale B-tree indexes (#63386) ── PRAGMA integrity_check can report "wrong #
    # of entries in index" when a B-tree index (e.g. idx_sessions_handoff_state) falls out of sync with its
    # base table. REINDEX rewrites the index b-tree from the canonical table rows using the existing index
    # definition, fixing the mismatch without touching data or FTS schema.
    conn = _connect_repair_durable(db_path)
    try:
        with contextlib.closing(conn):
            # Best-effort tokenizer load: messages_fts_cjk needs cjk_unicode61 before any statement can touch it;
            # tokenizer absence must never classify as corruption.
            load_fts5_cjk_extension(conn)
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
                    benign = SessionDB._is_fts5_unavailable_error(exc) or _schema_not_built(exc)
                    if not (isinstance(exc, sqlite3.OperationalError) and benign):
                        # This is the corruption class #66724 actually wants caught: partial shadow-table
                        # damage where MATCH / snippet / rank queries raise DatabaseError("database disk
                        # image is malformed") while reads of the FTS5 table itself parse fine.
                        return f"fts5 read probe failed on {fts_table}: {exc}"
            # FTS write probe: drive a row through the messages_fts* triggers in a transaction that is always
            # rolled back.
            probe_session_id = f"_hermes_fts_health_probe_{time.time_ns()}"
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                             (probe_session_id, "_health_probe", time.time()))
                conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                             (probe_session_id, "user", "_fts_health_probe", time.time()))
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError as exc:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                # Missing messages/sessions tables = brand new file mid-init, not corruption. "no such tokenizer":
                # this process lacks the cjk extension the DB's index needs — capability gap; a tokenizer-less
                # SessionDB drops the triggers itself.
                if _schema_not_built(exc) or "no such tokenizer: cjk_unicode61" in str(exc).lower():
                    return None
                return str(exc)
            return None
    except sqlite3.DatabaseError as exc:
        return str(exc)


def _live_writer_holds_db(db_path: Path) -> bool:
    """True when a connection outside this call still holds ``db_path`` open.

    Asks SQLite for what a repair needs and a live holder cannot grant: ``locking_mode=EXCLUSIVE`` then
    ``BEGIN IMMEDIATE`` — in WAL mode that needs exclusive WAL-index locks, so any other open connection fails
    it with SQLITE_BUSY; neither statement parses the schema, so it works on malformed DBs. Fails **open**
    (False) on anything but a positive busy/locked signal: refusing to repair a DB nobody holds would strand
    the self-heal path. In ``journal_mode=DELETE`` a held reader takes only SHARED and this returns False;
    repair is then serialised only by the cross-process repairer lock. Before probing, the foreign-holder scan
    (``hermes_state_holders``) fails closed on deleted-WAL-generation, uninspectable, or unknown holders."""
    import hermes_state_holders as _state_holders
    return _state_holders.live_writer_holds_db(db_path, connect_repair_durable=_connect_repair_durable)


def _repair_skip(report: Dict[str, Any], verb: str, error: str, exc: Optional[BaseException] = None) -> Dict[str, Any]:
    """Record *error* on *report* and log it as ``state.db repair <verb>``. An
    *exc* proving deterministic corruption consumes the persistent repair budget
    (private ``_repair_attempted`` marker, popped by the caller)."""
    if exc is not None and _repair_failure_consumes_attempt(exc):
        report["_repair_attempted"] = True
    report["error"] = error
    logger.error(f"state.db repair {verb}: %s", error)
    return report


def repair_state_db_schema(db_path: Path, *, backup: bool = True) -> Dict[str, Any]:
    """Repair a state.db whose ``sqlite_master`` is malformed or whose FTS indexes reject writes.

    Two corruption classes: malformed schema / "duplicate object definition" (even ``PRAGMA`` fails), and FTS
    write-corruption (reads and ``integrity_check`` pass, writes fail through ``messages_fts*`` triggers).
    ``_REPAIR_STRATEGIES`` run least-destructive first on a complete snapshot; a success is copied back
    transactionally, so canonical rows are never modified by a failed attempt. A raw backup is taken first
    unless ``backup=False``. Serialised across processes (gateway, Desktop backend and CLI open the same file;
    concurrent ``writable_schema`` surgery is itself a corruption source). Returns ``{repaired, strategy,
    backup_path, error}``.

    See #50502.
    """
    report: Dict[str, Any] = {"repaired": False, "strategy": None, "backup_path": None, "error": None}
    # Startup-watchdog lease: repair is I/O-bound (near-zero CPU), which the watchdog's CPU fallback would
    # misread as a parked deadlock. One lease (clamped to _MAX_LEASE_S=900) beats per-chunk renewal complexity.
    report_startup_progress(900.0, phase="state_db_repair")
    db_path = Path(db_path)
    if not db_path.exists():
        report["error"] = f"{db_path} does not exist"
        return report
    # Cross-restart cap: the in-memory claim bounds one process, but unhealable b-tree damage used to re-run
    # surgery + a fresh backup on EVERY restart.
    # Cross-restart attempt cap (#86747): the in-memory claim bounds one process, but a corruption class the
    # strategies below cannot heal (b-tree page damage) previously re-ran the whole surgery — and took a
    # fresh multi-hundred-MB forensic backup — on EVERY restart, forever. After
    # _MAX_PERSISTENT_REPAIR_ATTEMPTS failures against the same damaged file, stop retrying and surface a
    # terminal, actionable error.
    if _persistent_repair_attempts_exhausted(db_path):
        return _repair_skip(report, "skipped", _persistent_repair_exhausted_error(db_path))

    with _cross_process_repair_lock(db_path) as holding_lock:
        if not holding_lock:
            # Another process holds the lock (or the lock file was unopenable); it may have healed the file
            # already, so re-probe before failing.
            if _db_opens_cleanly(db_path) is None:
                report["repaired"], report["strategy"] = True, "repaired_by_other_process"
            else:
                report["error"] = ("could not obtain the state.db repair lock (held by another process, or the lock "
                                   "file was unopenable); skipped schema surgery to avoid racing a concurrent repairer")
            return report

        result = report
        # Recheck exhaustion after acquisition: a queued repairer can have recorded the final failure while
        # this process waited.
        if _persistent_repair_attempts_exhausted(db_path):
            _repair_skip(report, "skipped", _persistent_repair_exhausted_error(db_path))
        # WAL-holder preflight: fail closed for active readers before a backup is taken. Not the race defence — the
        # exclusive guard in the locked routine excludes writers through promotion and sees DELETE-mode readers too.
        elif _live_writer_holds_db(db_path):
            _repair_skip(report, "skipped", "a live writer still holds state.db; skipped schema surgery to avoid tearing "
                         "b-tree pages under a concurrent writer. Stop the gateway (hermes gateway stop) and retry.")
        else:
            # Probe journal mode BEFORE surgery: a rebuilt file comes back in the default (delete) mode and nothing
            # else records the flip. Unprobeable (damaged file) -> database.journal_mode is the restore target.
            # The open-time WAL-reset gate never sees this flip because it happens inside the repair path
            # (distinct from the open-time flip #89393 warns about).
            before_mode = _probe_journal_mode_for_repair(db_path)
            result = _repair_state_db_schema_locked(db_path, backup=backup, report=report,
                                                    journal_mode_before=before_mode)
            if result.get("repaired"):
                result["journal_mode_before"] = before_mode
        # Environmental aborts (before a strategy mutates the snapshot) are retriable, not proof of exhaustion;
        # the private marker stays out of the public report. The ledger update stays under the cross-process
        # lock so two repairers cannot lose each other's updates; a queued loser must not record at all.
        if result.pop("_repair_attempted", False) or result.get("repaired"):
            _record_repair_outcome(db_path, repaired=bool(result.get("repaired")))
    return result


def _probe_journal_mode_for_repair(db_path: Path) -> Optional[str]:
    """Best-effort journal-mode probe: ``wal``/``delete``, or ``None`` when the file cannot be opened or
    probed (malformed header, concurrent opener's locks — both expected on the repair path); callers then
    fall back to ``database.journal_mode``."""
    from hermes_state_wal import _on_disk_journal_mode
    try:
        with _repair_conn(db_path) as conn:
            return _on_disk_journal_mode(conn)
    except (sqlite3.Error, OSError):
        return None


def _restore_journal_mode_after_repair(db_path: Path, before_mode: Optional[str], *, conn=None) -> None:
    """Re-apply the journal mode after schema surgery.

    ``conn`` must be the exclusive repair guard connection when called from the repair path: opening a fresh
    connection AFTER the guard released let a writer still holding the unlinked old ``-wal`` inode coexist with
    a brand-new ``state.db-wal`` — two generations of one store. The reopen is the hazard, not the mode.

    A rebuilt file comes back in the default (delete) mode; without this a corruption event silently moves a
    WAL store out of WAL (the open-time WAL-reset gate never sees a flip made inside repair). Routed through
    :func:`apply_wal_with_fallback`, not a direct pragma, so it inherits the WAL-reset gate (a vulnerable
    runtime deliberately keeps DELETE; the journal_mode-changed WARNING is expected there), the macOS-NFS
    silent-refusal handling and the WAL companions. ``before_mode`` is only for the log comparison; the target
    is ``database.journal_mode``. Best-effort: the repair already succeeded, so failures log at WARNING.

    See #89674.
    See #89393.
    The transactional promotion already leaves the destination in its pre-repair mode, so on that path this
    is mostly the WAL-companion re-assertion; the reopen is the hazard, not the mode. See #101064.
    """
    from hermes_state_wal import apply_wal_with_fallback
    try:
        if conn is None:
            with _repair_conn(db_path) as owned:
                after = apply_wal_with_fallback(owned, db_label=db_path.name)
        else:
            after = apply_wal_with_fallback(conn, db_label=db_path.name)
        if before_mode and after != before_mode:
            logger.warning("state.db repair changed journal_mode %r -> %r (pre-surgery probe %r; restore resolved "
                           "through apply_wal_with_fallback per database.journal_mode and the WAL-reset gate)",
                           before_mode, after, before_mode)
    except (sqlite3.Error, OSError) as exc:
        logger.warning("state.db repair at %s: post-surgery journal-mode restore failed (%s); verify with PRAGMA "
                       "journal_mode on the next open", db_path, exc)


def _repair_state_db_schema_locked(
    db_path: Path, *, backup: bool, report: Dict[str, Any], journal_mode_before: Optional[str] = None,
) -> Dict[str, Any]:
    """Repair strategies for :func:`repair_state_db_schema`; caller holds the cross-process repair lock.

    Strategies run on a SCRATCH COPY, copied back through SQLite's transactional backup API only once proven to open
    cleanly, so a failed repair cannot modify or lose committed data (a WAL checkpoint of committed frames on guard
    release is not a repair mutation). WHY not in place: the final strategy ends in ``VACUUM``, which rebuilds the
    file from the schema SQLite can still parse — when the damage IS in the schema b-tree (the ``malformed database
    schema ()`` class) every table hanging off the unreadable part is silently dropped, the probe still reports
    malformed, and repair returned ``repaired=False`` having destroyed what it was asked to save.

    The pre-repair backup (#69603) does not close this: it is a forensic artefact that nothing reads back,
    so recovery still depends on a human noticing a ``.malformed-backup-*`` file and knowing what to do with
    it. Not mutating the original in the first place is the property that holds without a human in the loop.
    """
    scratch = db_path.with_name(f"{db_path.name}.repair-scratch")
    if (cleanup_error := _unlink_db_triple(scratch)) is not None:
        return _repair_skip(report, "aborted", f"could not remove a stale repair snapshot before probing state.db: {cleanup_error}")
    # Re-probe under the lock: a process we queued behind may have just repaired the file; redoing surgery
    # would undo it (the repair/re-corrupt cascade).
    if _db_opens_cleanly(db_path) is None:
        report["repaired"], report["strategy"] = True, "already_healthy"
        return report
    if backup:
        bpath, backup_error = _backup_db_file(db_path)
        report["backup_path"] = str(bpath) if bpath else None
        if bpath is None:  # HARD STOP: the forensic image is the recovery path when every strategy fails.
            return _repair_skip(report, "aborted", "pre-repair backup refused; aborting schema repair to avoid "
                                f"mutating the only copy of the damaged DB: {backup_error}")
    # The forensic copy precedes this guard on purpose: its live-holder checks would be poisoned by our own
    # exclusive connection. Everything touching the repair image or live promotion happens under writer exclusion.
    with _exclusive_repair_db_guard(db_path) as (live_guard, guard_error):
        if live_guard is None:
            return _repair_skip(report, "skipped", "could not acquire exclusive state.db repair ownership; skipped "
                                f"schema surgery to avoid overwriting a concurrent writer. Stop the gateway and retry: "
                                f"{guard_error}", exc=guard_error)
        if (space_error := _repair_scratch_space_error(db_path)) is not None:
            return _repair_skip(report, "aborted", space_error)
        try:
            # Source = live_guard: it owns the exclusion, and a second connection could be blocked by our own
            # EXCLUSIVE lock on some SQLite builds.
            _copy_database_snapshot(db_path, scratch, source_connection=live_guard)
        except (OSError, sqlite3.Error, TimeoutError) as exc:
            _unlink_db_triple(scratch)
            return _repair_skip(report, "aborted", f"could not stage a complete SQLite repair snapshot of {db_path}: {exc}", exc=exc)
        try:
            # Private marker for the outer wrapper: a strategy failure consumes the persistent budget; a
            # promotion failure is classified separately.
            report["_repair_attempted"] = True
            _run_repair_strategies(scratch, report)
            if report.get("repaired"):
                # Never ``os.replace`` the live DB: Windows rejects replacement under open handles and POSIX would
                # leave those handles on the old inode. The guard keeps writer exclusion throughout.
                try:
                    _copy_database_snapshot(scratch, db_path, destination_connection=live_guard)
                except (OSError, sqlite3.Error, TimeoutError) as exc:
                    report.update(repaired=False, strategy=None,
                                  _repair_attempted=_repair_failure_consumes_attempt(exc))
                    report["error"] = f"repaired snapshot could not be promoted transactionally: {exc}"
                    logger.error("state.db repair promotion failed: %s", exc)
                else:
                    logger.warning("state.db repaired via '%s' and promoted transactionally: %s",
                                   report.get("strategy"), db_path)
                    _restore_journal_mode_after_repair(db_path, journal_mode_before, conn=live_guard)
            if not report.get("repaired"):
                # Logged HERE, not in the strategies: they see the scratch copy, and the message a human acts on
                # must name a path that still exists.
                logger.error("state.db schema repair could not recover %s automatically (no committed canonical data "
                             "was modified or lost; backup: %s); manual restore from backup may be required.",
                             db_path, report["backup_path"])
            return report
        finally:
            # Never leave a half-repaired file beside the DB to be mistaken for the real thing.
            if (cleanup_error := _unlink_db_triple(scratch)) is not None:
                logger.warning("Could not remove state.db repair snapshot after repair: %s", cleanup_error)


def _unlink_db_triple(path: Path) -> Optional[str]:
    """Remove *path* and every SQLite sidecar; return any cleanup failure."""
    from hermes_state import _IS_WINDOWS
    failures: List[str] = []
    for victim in (path, *_sidecars(path)):
        for attempt in range(10):
            try:
                victim.unlink(missing_ok=True)
            except OSError as exc:
                # Windows may retain a just-closed SQLite handle for a few scheduler ticks; bounded retry.
                if isinstance(exc, PermissionError) and _IS_WINDOWS and attempt < 9:
                    time.sleep(0.05)
                    continue
                failures.append(f"{victim}: {exc}")
            break
    return "; ".join(failures) or None


# ── Repair strategies, least destructive first (each mutates its connection's DB) ──

def _edit_sqlite_master(conn: sqlite3.Connection, edit) -> None:
    """Run *edit* under ``writable_schema=ON``; bump the schema cookie when it
    reports a change so live peers discard their cached schema."""
    conn.execute("PRAGMA writable_schema=ON")
    if edit():
        _bump_schema_cookie(conn)
    conn.execute("PRAGMA writable_schema=OFF")
    conn.commit()


def _strategy_rebuild_fts(conn: sqlite3.Connection) -> None:
    """FTS5 'rebuild' rewrites each index from the content table: the least-
    destructive fix for an index that rejects writes while reads work."""
    from hermes_state import load_fts5_cjk_extension
    # The cjk index can only be rebuilt with its tokenizer loaded (best-effort).
    load_fts5_cjk_extension(conn)
    for table_name in _FTS_TABLES:
        with contextlib.suppress(sqlite3.OperationalError):  # table absent (FTS disabled / trigram off / cjk absent)
            conn.execute(f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')")


def _strategy_reindex(conn: sqlite3.Connection) -> None:
    """integrity_check reports "wrong # of entries in index" when a B-tree
    index drifts from its base table; REINDEX rewrites it from canonical rows."""
    # REINDEX rewrites every index b-tree; take the barriers now that the schema parses.
    _reapply_durability_barriers(conn)
    conn.execute("REINDEX")
    conn.commit()


def _strategy_dedup_schema(conn: sqlite3.Connection) -> None:
    """De-duplicate sqlite_master (lowest rowid per type/name wins), keeping FTS."""
    def _dedup() -> bool:
        dupes = conn.execute(
            "SELECT type, name, COUNT(*) AS c, MIN(rowid) AS keep FROM sqlite_master GROUP BY type, name HAVING c > 1"
        ).fetchall()
        for type_, name, _count, keep in dupes:
            conn.execute("DELETE FROM sqlite_master WHERE type IS ? AND name IS ? AND rowid <> ?",
                         (type_, name, keep))
        return bool(dupes)

    _edit_sqlite_master(conn, _dedup)


def _strategy_drop_fts_vacuum(conn: sqlite3.Connection) -> None:
    """Drop all FTS schema and VACUUM; indexes rebuild on the next open. The
    destructive one, and why strategies run on a scratch copy: on a damaged
    schema b-tree VACUUM silently drops every table hanging off the unreadable part."""
    _edit_sqlite_master(conn, lambda: conn.execute("DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'") or True)
    # The schema parses now, so the barriers can stick — VACUUM rewrites the whole file.
    _reapply_durability_barriers(conn)
    conn.execute("VACUUM")


# (name, body, success log, failure log) in escalation order. failure log None =
# final strategy: its failure lands in report["error"] instead of logged-and-skipped.
_REPAIR_STRATEGIES = (
    ("rebuild_fts", _strategy_rebuild_fts, "state.db FTS indexes rebuilt in place (schema preserved): %s",
     "state.db FTS in-place rebuild pass failed: %s"),
    ("reindex_btree", _strategy_reindex, "state.db B-tree indexes rebuilt via REINDEX: %s",
     "state.db REINDEX pass failed: %s"),
    ("dedup_schema", _strategy_dedup_schema,
     "state.db schema repaired by de-duplicating sqlite_master (FTS index preserved): %s",
     "state.db dedup repair pass failed: %s"),
    ("drop_fts_rebuild", _strategy_drop_fts_vacuum,
     "state.db schema repaired by dropping FTS schema; indexes will rebuild from messages on next open: %s", None),
)


def _run_repair_strategies(db_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    """Escalating repair attempts, applied to *db_path* IN PLACE — only ever a scratch copy nothing else holds open,
    never the user's database. The "could not recover" log lives in the caller so it names the user's database."""

    for name, body, success_msg, failure_msg in _REPAIR_STRATEGIES:
        try:
            with _repair_conn(db_path) as conn:
                body(conn)
            reason = _db_opens_cleanly(db_path)
        except sqlite3.DatabaseError as exc:
            reason = str(exc)
            if failure_msg is not None:
                logger.warning(failure_msg, exc)
        if reason is None:
            report["repaired"], report["strategy"] = True, name
            logger.warning(success_msg, db_path)
            return report
        if failure_msg is None:
            report["error"] = reason
    return report
