"""Profile-local volatile runtime state for cron job definitions.

Cron job definitions live in ``jobs.json``. Scheduler-owned timestamps, status,
counters, leases, and cross-store recovery metadata live here so ordinary runs
do not rewrite operator intent.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence


_RUNTIME_DB_NAME = "runtime.db"


def runtime_db_path(cron_dir: Path) -> Path:
    """Return the runtime database path for one profile-local cron directory."""
    return cron_dir / _RUNTIME_DB_NAME


def _owner_tuple(path: Path) -> Optional[tuple[int, int]]:
    """Return uid/gid when the platform exposes POSIX ownership."""
    try:
        stat_result = path.stat()
    except OSError:
        return None
    uid = getattr(stat_result, "st_uid", None)
    gid = getattr(stat_result, "st_gid", None)
    if uid is None or gid is None:
        return None
    return int(uid), int(gid)


def _secure_runtime_files(path: Path, owner: Optional[tuple[int, int]]) -> None:
    """Best-effort owner/mode repair for the database and SQLite sidecars."""
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not candidate.exists():
            continue
        try:
            os.chmod(candidate, 0o600)
        except OSError:
            pass
        if owner is None or not hasattr(os, "chown"):
            continue
        try:
            current = _owner_tuple(candidate)
            if current != owner:
                os.chown(candidate, owner[0], owner[1])
        except OSError:
            pass


def _connect(cron_dir: Path) -> sqlite3.Connection:
    """Open and initialize one profile's cron runtime database."""
    path = runtime_db_path(cron_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = _owner_tuple(path) or _owner_tuple(path.parent)
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="cron/runtime.db")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS job_runtime (
                 job_id TEXT PRIMARY KEY,
                 state_json TEXT NOT NULL
               )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pending_definitions (
                 singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                 definitions_json TEXT NOT NULL
               )"""
        )
        conn.commit()
        _secure_runtime_files(path, owner)
        return conn
    except BaseException:
        if conn is not None:
            conn.close()
        raise


@contextmanager
def _transaction(cron_dir: Path) -> Iterator[sqlite3.Connection]:
    """Open one transaction and close its connection deterministically."""
    conn = _connect(cron_dir)
    try:
        with conn:
            yield conn
    finally:
        conn.close()
        owner = _owner_tuple(runtime_db_path(cron_dir)) or _owner_tuple(cron_dir)
        _secure_runtime_files(runtime_db_path(cron_dir), owner)


def _serialize(state: Mapping[str, Any]) -> str:
    """Serialize runtime state canonically for deterministic updates."""
    return json.dumps(
        dict(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _serialize_definitions(definitions: Sequence[Mapping[str, Any]]) -> str:
    """Serialize a pending definition snapshot canonically."""
    return json.dumps(
        [dict(definition) for definition in definitions],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _deserialize(job_id: str, payload: str) -> Dict[str, Any]:
    """Decode one runtime row and fail closed on corrupt state."""
    try:
        state = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cron runtime state for job {job_id!r} is corrupted"
        ) from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"Cron runtime state for job {job_id!r} must be an object")
    return state


def _replace_rows(
    conn: sqlite3.Connection,
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Replace the complete runtime snapshot without SQL variable limits."""
    normalized = {str(job_id): dict(state) for job_id, state in states.items()}
    conn.execute("DELETE FROM job_runtime")
    if normalized:
        conn.executemany(
            "INSERT INTO job_runtime(job_id, state_json) VALUES (?, ?)",
            [(job_id, _serialize(state)) for job_id, state in normalized.items()],
        )


def load_runtime_states(cron_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all volatile job state for one profile."""
    path = runtime_db_path(cron_dir)
    if not path.exists():
        return {}
    with _transaction(cron_dir) as conn:
        rows = conn.execute(
            "SELECT job_id, state_json FROM job_runtime ORDER BY job_id"
        ).fetchall()
    return {
        str(row["job_id"]): _deserialize(str(row["job_id"]), row["state_json"])
        for row in rows
    }


def load_pending_definitions(cron_dir: Path) -> Optional[list[Dict[str, Any]]]:
    """Load a journaled definition snapshot awaiting materialization."""
    path = runtime_db_path(cron_dir)
    if not path.exists():
        return None
    with _transaction(cron_dir) as conn:
        row = conn.execute(
            "SELECT definitions_json FROM pending_definitions WHERE singleton = 1"
        ).fetchone()
    if row is None:
        return None
    try:
        definitions = json.loads(row["definitions_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cron pending definition journal is corrupted") from exc
    if not isinstance(definitions, list) or not all(
        isinstance(definition, dict) for definition in definitions
    ):
        raise RuntimeError("Cron pending definition journal must be a list of objects")
    return definitions


def list_live_claims(
    cron_dir: Path,
    *,
    fire_claim_ttl_seconds: float,
    run_claim_ttl_seconds: float,
) -> list[Dict[str, str]]:
    """Return fresh fire/run claims that make destructive restore unsafe.

    Malformed, future-dated, and expired claims are stale by the same bounded
    age rule used by scheduler acquisition.  The caller supplies both TTLs so
    this storage module remains independent of scheduler policy/configuration.
    """
    from hermes_time import now

    current = now()
    live: list[Dict[str, str]] = []
    for job_id, state in load_runtime_states(cron_dir).items():
        for field, ttl in (
            ("fire_claim", fire_claim_ttl_seconds),
            ("run_claim", run_claim_ttl_seconds),
        ):
            claim = state.get(field)
            if not isinstance(claim, dict):
                continue
            try:
                claimed_at = datetime.fromisoformat(str(claim["at"]))
                if claimed_at.tzinfo is None:
                    claimed_at = claimed_at.replace(tzinfo=current.tzinfo)
                age = (current - claimed_at).total_seconds()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= age < ttl:
                live.append({
                    "job_id": job_id,
                    "kind": field,
                    "at": str(claim["at"]),
                    "owner": str(claim.get("id") or claim.get("by") or "unknown"),
                })
    return live


def clear_pending_definitions(cron_dir: Path) -> None:
    """Clear the forward-recovery journal after jobs.json is durable."""
    with _transaction(cron_dir) as conn:
        conn.execute("DELETE FROM pending_definitions WHERE singleton = 1")


def merge_legacy_runtime_states(
    cron_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Seed migrated legacy state without overwriting a newer runtime row.

    Runtime is committed before the combined definition artifact is stripped.
    If the process dies between those steps, the next migration sees the
    existing authoritative row and safely retries the definition rewrite.
    """
    if not states:
        return
    with _transaction(cron_dir) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO job_runtime(job_id, state_json) VALUES (?, ?)",
            [(str(job_id), _serialize(state)) for job_id, state in states.items()],
        )


def replace_runtime_states(
    cron_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically replace runtime rows for the current definition set."""
    with _transaction(cron_dir) as conn:
        _replace_rows(conn, states)


def stage_runtime_and_definitions(
    cron_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
    definitions: Sequence[Mapping[str, Any]],
) -> None:
    """Commit runtime plus a forward-recovery definition journal atomically."""
    with _transaction(cron_dir) as conn:
        _replace_rows(conn, states)
        conn.execute(
            """INSERT INTO pending_definitions(singleton, definitions_json)
               VALUES (1, ?)
               ON CONFLICT(singleton) DO UPDATE
               SET definitions_json=excluded.definitions_json""",
            (_serialize_definitions(definitions),),
        )
