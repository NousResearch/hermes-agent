"""Quiescent, explicit installation of a verified session-recovery candidate.

This is intentionally a separate step from ordinary offline recovery. It reuses
SessionDB's cross-process repair lock, the existing fail-closed holder scan, and
SQLite's exclusive repair connection for the whole candidate-publication window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from hermes_cli import session_recovery as recovery
from hermes_cli.session_recovery import SessionRecoveryError, SessionRecoverySafetyError
from hermes_state import SessionDB
from hermes_state_health import reset_storage_state
from hermes_state_holders import foreign_state_db_holders


_VOLATILE_SIDECARS = ("-shm",)
_INSTALL_SIDECARS = ("", "-wal", "-shm", "-journal")
logger = logging.getLogger(__name__)


def _backup_root_paths() -> tuple[Path, Path]:
    """Location of the preserved source bundles, without creating anything.

    The symlink refusal lives here so both the space preflight and the copy
    observe it before any directory is created.
    """
    home = Path(get_hermes_home()).resolve()
    backups_root = home / "backups"
    backup_root = backups_root / "session-recovery"
    if backups_root.is_symlink() or backup_root.is_symlink():
        raise SessionRecoverySafetyError(f"Refusing recovery backup through a symlink: {backup_root}")
    return backups_root, backup_root


def _nearest_existing_dir(path: Path) -> Path:
    """Nearest existing ancestor of *path* (itself when it exists)."""
    probe = path
    while True:
        try:
            if probe.exists():
                return probe
        except OSError as exc:
            raise SessionRecoverySafetyError(
                f"Could not determine the filesystem for {path} ({exc}); refusing recovery "
                "installation rather than risk filling the volume. The active database was not replaced."
            ) from exc
        parent = probe.parent
        if parent == probe:
            raise SessionRecoverySafetyError(
                f"Could not determine the filesystem for {path}; refusing recovery installation "
                "rather than risk filling the volume. The active database was not replaced."
            )
        probe = parent


def _volume_key(path: Path) -> int:
    """``st_dev`` for the filesystem holding *path* (via its nearest existing ancestor)."""
    probe = _nearest_existing_dir(path)
    try:
        return int(os.stat(probe).st_dev)
    except OSError as exc:
        raise SessionRecoverySafetyError(
            f"Could not determine free space on {probe} ({exc}); refusing recovery installation "
            "rather than risk filling the volume. The active database was not replaced."
        ) from exc


def _volume_free_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding *path*; fails closed when undeterminable."""
    probe = _nearest_existing_dir(path)
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError as exc:
        raise SessionRecoverySafetyError(
            f"Could not determine free space on {probe} ({exc}); refusing recovery installation "
            "rather than risk filling the volume. The active database was not replaced."
        ) from exc


def _raw_bundle_bytes(source: Path) -> int:
    """Total bytes of the durable source bundle (main file plus present sidecars)."""
    try:
        return int(sum(info["size"] for info in recovery._source_fingerprint(source).values()))
    except OSError as exc:
        raise SessionRecoverySafetyError(
            f"Could not determine the source bundle size for {source} ({exc}); refusing recovery "
            "installation rather than risk filling the volume. The active database was not replaced."
        ) from exc


def _install_disk_space_preflight(
    source: Path, backup_root: Path, work_root: Path, output_parent: Path,
) -> dict[str, Any]:
    """Reserve headroom for the backup, disposable copy, and candidate before any copy.

    The original install copied the full source bundle into
    ``<HERMES_HOME>/backups/session-recovery`` before any space check, while the
    later ``recover_session_database`` preflight budgets only its work/output
    locations. On a nearly full profile volume the first copy alone can exhaust
    the disk. This preflight runs first and budgets all three writes, grouping
    by filesystem so co-located destinations share one requirement and separate
    volumes are checked independently. Undeterminable usage fails closed.
    """
    bundle_bytes = _raw_bundle_bytes(source)
    backup_need, work_need, output_need = bundle_bytes, bundle_bytes, bundle_bytes
    total = backup_need + work_need + output_need
    headroom = max(recovery._MINIMUM_SPACE_HEADROOM, int(total * 0.05))
    demands = (
        ("backup", backup_root, backup_need),
        ("work", work_root, work_need),
        ("output", output_parent, output_need),
    )
    groups: dict[int, dict[str, Any]] = {}
    for label, location, need in demands:
        key = _volume_key(location)
        group = groups.get(key)
        if group is None:
            groups[key] = {"representative": location, "total_bytes": need, "members": [label]}
        else:
            group["total_bytes"] = int(group["total_bytes"]) + need
            group["members"].append(label)
    details: list[dict[str, Any]] = []
    shortages: list[str] = []
    for group in groups.values():
        representative = group["representative"]
        required = int(group["total_bytes"]) + headroom
        free = _volume_free_bytes(representative)
        group.update(required_bytes=required, free_bytes=free, headroom_bytes=headroom,
                     representative=str(representative))
        details.append({**group, "members": list(group["members"])})
        if free < required:
            members = "+".join(group["members"])
            shortages.append(
                f"{representative} [{members}]: "
                f"{recovery._format_bytes(free)} available, {recovery._format_bytes(required)} required "
                f"({recovery._format_bytes(int(group['total_bytes']))} writes + "
                f"{recovery._format_bytes(headroom)} headroom)"
            )
    report: dict[str, Any] = {
        "source_bundle_bytes": bundle_bytes,
        "backup_bytes": backup_need,
        "work_bytes": work_need,
        "output_bytes": output_need,
        "headroom_bytes": headroom,
        "groups": details,
    }
    if shortages:
        raise SessionRecoverySafetyError(
            "Not enough free disk space for a safe recovery installation: "
            + "; ".join(shortages)
            + ". Free disk space or use --work-dir/--output on a filesystem with more free space. "
            "The active database was not replaced and no source bundle was preserved."
        )
    return report


def _live_table_row_counts(conn) -> dict[str, int]:
    """User-table row counts on a live connection (sqlite internals excluded)."""
    names = [
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]
    counts = {}
    for name in names:
        try:
            counts[name] = int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        except Exception:
            continue  # unreadable under corruption: recovery already refused or salvaged it
    return counts


def _nonempty_tables_missing_from_candidate(live_conn, candidate_path: Path) -> dict[str, int]:
    """Live tables the recovered candidate does not carry, with their row counts.

    Installation replaces the whole file, so a live-only table with rows would
    vanish on an exit-0 install. The candidate only copies recovery-known
    tables; anything else (a lazily created gateway table recovery never
    registered, a newer-schema table) must block the install, not be dropped.
    """
    candidate_conn = sqlite3.connect(str(candidate_path))
    try:
        candidate_tables = {
            row[0] for row in candidate_conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        candidate_conn.close()
    return {
        name: rows for name, rows in _live_table_row_counts(live_conn).items()
        if name not in candidate_tables and rows > 0
    }


def _path_signature(db_path: Path) -> dict[str, dict[str, int]]:
    """Fingerprint the durable database files, ignoring regenerable SHM and empty journals.

    The exclusive SQLite guard may create an empty WAL while taking ownership. It contains
    no committed frames and is not a source-generation change.
    """
    signature: dict[str, dict[str, int]] = {}
    for suffix in _INSTALL_SIDECARS:
        if suffix in _VOLATILE_SIDECARS:
            # -shm is process-volatile WAL peer state with no durable content, and the
            # exclusive guard itself creates/drops it while taking ownership — fingerprinting
            # it would flag a generation change the install did to itself. Do not "simplify"
            # this exclusion away.
            continue
        path = Path(f"{db_path}{suffix}") if suffix else db_path
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if suffix and stat.st_size == 0:
            continue
        signature[suffix or "main"] = {
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return signature


def _copy_metadata(db_path: Path) -> dict[str, dict[str, int]]:
    """Size/mtime view used to prove ``copy2`` captured the same durable source files."""
    return {
        name: {"size": row["size"], "mtime_ns": row["mtime_ns"]}
        for name, row in _path_signature(db_path).items()
    }


def _require_no_foreign_holders(db_path: Path, *, phase: str) -> None:
    """Refuse on any foreign holder or incomplete scan; a PID is not an ownership token."""
    holders = foreign_state_db_holders(db_path)
    if not holders:
        return
    details = []
    for pid, _target in holders:
        details.append(f"PID {pid} holds a database handle" if pid > 0 else "a holder scan was incomplete")
    joined = "; ".join(details[:4])
    raise SessionRecoverySafetyError(
        f"Refusing recovery installation during {phase}: state.db or a sidecar is open in another "
        f"process ({joined}). Stop the profile's gateway, Desktop/serve backend, cron workers and "
        "other Hermes processes, then retry. The active database was not replaced."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preserve_source_bundle(source: Path) -> tuple[Path, Path, dict[str, Any], dict[str, dict[str, int]]]:
    """Keep a private, raw DB/WAL/SHM bundle before SQLite opens the active source."""
    _, backup_root = _backup_root_paths()
    backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(backup_root, 0o700)

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    staging = backup_root / f".{run_id}.partial"
    published = backup_root / run_id
    source_dir = staging / "source"
    staging.mkdir(mode=0o700)
    source_dir.mkdir(mode=0o700)
    if os.name != "nt":
        os.chmod(staging, 0o700)
        os.chmod(source_dir, 0o700)

    try:
        # Check before and after the raw copy. offline_file_access holds the same-process
        # lifecycle mutex across open/copy/close; the foreign-holder scans cover other processes.
        _require_no_foreign_holders(source, phase="source preservation")
        before = _path_signature(source)
        snapshot, copied = recovery._copy_source_bundle(source, source_dir)
        after = _path_signature(source)
        copied_metadata = _copy_metadata(snapshot)
        expected_metadata = {
            name: {"size": row["size"], "mtime_ns": row["mtime_ns"]}
            for name, row in before.items()
        }
        if before != after or copied_metadata != expected_metadata:
            raise SessionRecoverySafetyError(
                "The active source changed while its raw recovery bundle was being preserved. "
                "No candidate was installed; stop all writers and retry."
            )
        _require_no_foreign_holders(source, phase="source preservation")
        files = {
            path.name: {"size": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(source_dir.iterdir()) if path.is_file()
        }
        manifest = {
            "format_version": 1,
            "source_path": str(source),
            "source_signature": before,
            "copied_files": copied,
            "files": files,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        staging.rename(published)
        return published, published / "source" / source.name, manifest, before
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _application_id(path: Path) -> int:
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=1.0)
    try:
        return int(conn.execute("PRAGMA application_id").fetchone()[0] or 0)
    finally:
        conn.close()


def _set_application_id(path: Path, value: int) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=1.0)
    try:
        conn.execute(f"PRAGMA application_id={int(value)}")
    finally:
        conn.close()


def _fresh_application_id(previous: int) -> int:
    while True:
        value = secrets.randbelow(0x7FFFFFFF) + 1
        if value != previous:
            return value


def _ensure_candidate_generation(source: Path, candidate: Path) -> dict[str, int]:
    """Give the candidate a header generation different from existing SessionDB handles."""
    source_id = _application_id(source)
    candidate_id = _application_id(candidate)
    if candidate_id == 0 or candidate_id == source_id:
        candidate_id = _fresh_application_id(source_id)
        _set_application_id(candidate, candidate_id)
    return {"source_application_id": source_id, "candidate_application_id": candidate_id}


def _sessiondb_canary(db_path: Path, *, label: str) -> None:
    """Exercise ordinary SessionDB create/write/read/FTS/delete against a candidate or install."""
    token = f"recoverycanary{uuid.uuid4().hex}"
    session_id = f"recovery-{label}-{uuid.uuid4().hex}"
    db = SessionDB(db_path=db_path)
    created = False
    cleanup_error: Optional[Exception] = None
    try:
        db.create_session(session_id=session_id, source="system")
        created = True
        db.append_message(session_id, role="user", content=token)
        rows = db.get_messages(session_id)
        if not any(row.get("content") == token for row in rows):
            raise SessionRecoveryError(f"SessionDB {label} canary did not read back its message")
        hits = db.search_messages(token)
        if not any(row.get("session_id") == session_id for row in hits):
            raise SessionRecoveryError(f"SessionDB {label} canary did not find its FTS entry")
    finally:
        if created:
            try:
                db.delete_session(session_id)
            except Exception as exc:
                cleanup_error = exc
                logger.warning("Could not remove temporary SessionDB %s canary %s", label, session_id, exc_info=True)
        db.close()
    if cleanup_error is not None:
        raise SessionRecoveryError(f"SessionDB {label} canary cleanup failed: {cleanup_error}") from cleanup_error


def _active_state_db(source_path: Path) -> Path:
    source = source_path.expanduser().resolve(strict=True)
    active = (Path(get_hermes_home()) / "state.db").resolve(strict=False)
    if source != active:
        raise SessionRecoverySafetyError(
            "--install is restricted to the active profile's state.db. Recover backups to a separate "
            "--output without --install; installation will not guess which live profile to replace."
        )
    return source


def recover_and_install_session_database(
    source_path: Path,
    output_path: Path,
    *,
    work_dir: Optional[Path] = None,
    chunk_size: int = 1_000,
    progress_cb: Optional[recovery.ProgressCallback] = None,
) -> dict[str, Any]:
    """Recover a complete candidate and install it only under cross-process writer exclusion.

    The raw source bundle is preserved first. A candidate is built from that immutable
    bundle, then the current source generation is revalidated under the existing repair
    lock + exclusive SQLite guard. Installation uses SQLite's transactional backup API,
    not pathname replacement, so a live generation is never unlinked under a waiter.
    """
    source = _active_state_db(Path(source_path))
    source, output, work_root = recovery._validate_paths(
        source, output_path=Path(output_path), work_dir=work_dir,
    )
    assert output is not None
    _, backup_root = _backup_root_paths()
    install_space = _install_disk_space_preflight(source, backup_root, work_root, output.parent)
    backup_dir, snapshot_source, manifest, source_signature = _preserve_source_bundle(source)
    report = recovery.recover_session_database(
        snapshot_source, output, work_dir=work_root, chunk_size=chunk_size,
        progress_cb=progress_cb, allow_partial=False,
    )
    report.update(
        install_requested=True,
        install_target=str(source),
        preserved_source_bundle=str(backup_dir),
        preserved_source_manifest=str(backup_dir / "manifest.json"),
        installed=False,
        install_disk_space=install_space,
    )
    if not report.get("complete") or not report.get("verified") or report.get("partial"):
        report["install_refusal"] = (
            "Only a complete, fully verified recovery can be installed. The candidate and preserved source bundle "
            "were kept for review; the active database was not replaced."
        )
        return report

    generation = _ensure_candidate_generation(snapshot_source, output)
    try:
        _sessiondb_canary(output, label="candidate")
    except Exception as exc:
        report["install_refusal"] = (
            f"The candidate did not pass the normal SessionDB write/read/FTS canary: "
            f"{type(exc).__name__}: {exc}. It was not installed."
        )
        report.setdefault("verification", {})["candidate_sessiondb_canary"] = False
        report["generation_fence"] = generation
        return report
    report["generation_fence"] = generation
    report.setdefault("verification", {})["candidate_sessiondb_canary"] = True

    from hermes_state_repair import (
        _copy_database_snapshot,
        _cross_process_repair_lock,
        _exclusive_repair_db_guard,
        _live_writer_holds_db,
        _restore_journal_mode_after_repair,
    )

    promoted = False
    with _cross_process_repair_lock(source) as lock_acquired:
        if not lock_acquired:
            report["install_refusal"] = (
                "Could not obtain the state.db repair lock. The active database was not replaced; "
                "stop profile writers and retry."
            )
            return report
        _require_no_foreign_holders(source, phase="installation preflight")
        if _live_writer_holds_db(source):
            report["install_refusal"] = (
                "A live state.db connection still holds the profile. The candidate was not installed. "
                "Stop the profile's gateway, Desktop/serve backend, cron workers and other Hermes processes, "
                "then retry."
            )
            return report
        with _exclusive_repair_db_guard(source) as (guard, guard_error):
            if guard is None:
                report["install_refusal"] = (
                    f"Could not obtain exclusive state.db writer ownership ({guard_error}). The candidate was "
                    "not installed; stop the profile's writers and retry."
                )
                return report
            _require_no_foreign_holders(source, phase="exclusive installation")
            if _path_signature(source) != source_signature:
                report["install_refusal"] = (
                    "The active database generation changed after its recovery snapshot was made. The candidate "
                    "was not installed; retry recovery from the current source."
                )
                return report
            source_application_id = int(guard.execute("PRAGMA application_id").fetchone()[0] or 0)
            if source_application_id != generation["source_application_id"]:
                report["install_refusal"] = (
                    "The active database generation no longer matches the preserved source bundle. The candidate "
                    "was not installed; retry recovery from the current source."
                )
                return report
            journal_mode = guard.execute("PRAGMA journal_mode").fetchone()[0]
            unknown_live = _nonempty_tables_missing_from_candidate(guard, output)
            if unknown_live:
                names = ", ".join(f"{name} ({rows} rows)" for name, rows in sorted(unknown_live.items()))
                report["promoted"] = False
                report["verification"]["unknown_live_tables"] = sorted(unknown_live)
                report["install_refusal"] = (
                    f"The live database holds tables the recovered candidate does not carry: {names}. "
                    "Installing would silently drop them. Register the owning tables in "
                    "hermes_cli/session_recovery.py (_AUXILIARY_TABLE_SCHEMAS) or back them up separately, "
                    "then retry. The active database was not replaced."
                )
                return report
            _copy_database_snapshot(output, source, destination_connection=guard)
            promoted = True
            _restore_journal_mode_after_repair(source, journal_mode, conn=guard)
            try:
                integrity = [str(row[0]).lower() for row in guard.execute("PRAGMA integrity_check").fetchall()]
                foreign_keys = guard.execute("PRAGMA foreign_key_check").fetchall()
                fts_rows = int(guard.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
                report.setdefault("verification", {}).update(
                    installed_integrity_check=integrity,
                    installed_foreign_key_check=[list(row) for row in foreign_keys],
                    installed_fts_rows=fts_rows,
                )
                checks_ok = integrity == ["ok"] and not foreign_keys
            except Exception as check_error:
                report.setdefault("verification", {})["installed_check_error"] = (
                    f"{type(check_error).__name__}: {check_error}"
                )
                checks_ok = False
            if not checks_ok:
                try:
                    _copy_database_snapshot(snapshot_source, source, destination_connection=guard)
                    _restore_journal_mode_after_repair(source, journal_mode, conn=guard)
                    report["promoted"] = False
                    report["verification"]["install_rollback_restored"] = True
                    report["install_refusal"] = (
                        "The guarded post-copy checks failed; the preserved source snapshot was restored and "
                        "the candidate was not installed."
                    )
                except Exception as rollback_error:
                    report["promoted"] = True
                    report["verification"]["install_rollback_restored"] = False
                    report["install_refusal"] = (
                        "The post-copy checks failed and automatic restoration from the preserved source bundle "
                        f"also failed ({type(rollback_error).__name__}: {rollback_error}). Do not resume writers; "
                        "the source bundle is preserved for manual recovery."
                    )
                return report

    if promoted:
        report["promoted"] = True
        try:
            _sessiondb_canary(source, label="installed")
        except Exception as exc:
            report["install_error"] = f"Post-install SessionDB canary failed: {type(exc).__name__}: {exc}"
            report.setdefault("verification", {})["installed_sessiondb_canary"] = False
            return report
        reset_storage_state(source)
        report.setdefault("verification", {})["installed_sessiondb_canary"] = True
        report["installed"] = True
        report["install_refusal"] = None
    return report
