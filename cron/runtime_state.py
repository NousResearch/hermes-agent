"""Cron: profile-local runtime state, kept out of the declarative ``jobs.json`` (#75607).

``jobs.json`` holds what an operator declares (prompt, schedule, delivery, enabled, ...). Scheduler
bookkeeping (``next_run_at``, ``last_*``, claims, ``pending_slot``, repeat progress, ...) lives in
``<cron dir>/runtime.db`` so an ordinary fire never rewrites the artifact operators back up, review
or keep in source control. ``cron.jobs`` owns the field split and the merge; this module is only the
SQLite store, plus a one-row journal that makes a save touching both artifacts crash-recoverable:
runtime rows and the next definition snapshot commit in ONE transaction, then ``jobs.json`` is
replaced, then the journal row is acknowledged. A crash in between leaves a journal the next load
finishes (see ``cron.jobs._recover_pending_definitions``).

Every call takes the cron dir explicitly: profile scoping (``use_cron_store``) is resolved by the
caller, never by ``get_hermes_home()`` here.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Collection, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

RUNTIME_DB_NAME = "runtime.db"


def runtime_db_path(cron_dir: Path) -> Path:
    return cron_dir / RUNTIME_DB_NAME


_PENDING_COLUMNS = frozenset({"definitions_json", "generation_id", "base_definitions_digest"})


def _schema_current(conn: sqlite3.Connection) -> bool:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not {"job_runtime", "pending_definitions"} <= tables:
        return False
    columns = {r[1] for r in conn.execute("PRAGMA table_info(pending_definitions)")}
    return _PENDING_COLUMNS <= columns


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    if _schema_current(conn):  # the common case: no write lock just to open the store
        return
    # IMMEDIATE so two first-openers cannot both see the pre-column shape and race the ALTER.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS job_runtime (job_id TEXT PRIMARY KEY, state_json TEXT NOT NULL)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pending_definitions (
                 singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                 definitions_json TEXT NOT NULL,
                 generation_id TEXT,
                 base_definitions_digest TEXT
               )""")
        # Tolerates a runtime.db written by an earlier revision of this store.
        add_column_if_missing(conn, "pending_definitions", "generation_id", "generation_id TEXT")
        add_column_if_missing(
            conn, "pending_definitions", "base_definitions_digest", "base_definitions_digest TEXT")
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise


def _secure(cron_dir: Path) -> None:
    """Owner-only mode on the db and its sidecars (same helper as jobs.json)."""
    from hermes_cli.config import _secure_file

    path = runtime_db_path(cron_dir)
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            with contextlib.suppress(OSError):
                _secure_file(candidate)


@contextlib.contextmanager
def _transaction(cron_dir: Path, *, write: bool = True) -> Iterator[sqlite3.Connection]:
    """One transaction on a fresh connection, always closed. Writes use IMMEDIATE (a plain ``SELECT``
    opens no transaction in sqlite3, leaving a read-to-write window for a sibling writer) and apply
    the journal-mode policy; reads run on the hot ``load_jobs()`` path, and WAL mode persists in the
    file once a writer set it, so they skip that (config-reading) setup."""
    from hermes_cli.sqlite_util import open_db, transaction

    conn = open_db(
        runtime_db_path(cron_dir), db_label="cron/runtime.db", synchronous_full=True, wal=write,
        wal_lock_retries=3, initialize=_initialize_schema)
    try:
        with transaction(conn, immediate=write):
            yield conn
    finally:
        if write:
            _secure(cron_dir)


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_state(job_id: str, payload: str) -> Dict[str, Any]:
    """Decode one row; fail closed on corruption (silently dropping state could re-fire a job)."""
    try:
        state = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cron runtime state for job {job_id!r} is corrupted") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"Cron runtime state for job {job_id!r} must be a JSON object")
    return state


def serialize_state(state: Mapping[str, Any]) -> str:
    """The canonical stored form of one row; equal text means an unchanged row."""
    return _dumps(dict(state))


def read_store(cron_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], bool]:
    """``(every job's runtime row, each row's stored text, whether an interrupted save's journal is
    pending)`` in one read snapshot; empty when the store was never created. The stored text lets a
    caller detect changes without keeping a (mutable, aliasable) copy of the decoded row."""
    if not runtime_db_path(cron_dir).exists():
        return {}, {}, False
    with _transaction(cron_dir, write=False) as conn:
        conn.execute("BEGIN")  # one snapshot for both statements
        rows = conn.execute("SELECT job_id, state_json FROM job_runtime").fetchall()
        pending = conn.execute(
            "SELECT 1 FROM pending_definitions WHERE singleton = 1").fetchone() is not None
    states = {str(r["job_id"]): _decode_state(str(r["job_id"]), r["state_json"]) for r in rows}
    return states, {str(r["job_id"]): str(r["state_json"]) for r in rows}, pending


def load_runtime_states(cron_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Every job's runtime row for one profile; ``{}`` when the store was never created."""
    return read_store(cron_dir)[0]


def load_pending_definitions(
    cron_dir: Path,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
    """``(definitions, generation_id, base_digest)`` of an unacknowledged save, else all None."""
    if not runtime_db_path(cron_dir).exists():
        return None, None, None
    with _transaction(cron_dir, write=False) as conn:
        row = conn.execute(
            "SELECT definitions_json, generation_id, base_definitions_digest "
            "FROM pending_definitions WHERE singleton = 1").fetchone()
    if row is None:
        return None, None, None
    try:
        definitions = json.loads(row["definitions_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cron pending-definition journal is corrupted") from exc
    if not isinstance(definitions, list):
        raise RuntimeError("Cron pending-definition journal must be a JSON list")
    # A pre-generation row gets a synthetic id so it can still be acknowledged exactly once.
    generation_id = str(row["generation_id"] or "").strip() or "legacy"
    base_digest = str(row["base_definitions_digest"] or "").strip() or None
    return definitions, generation_id, base_digest


def acknowledge_pending_definitions(cron_dir: Path, generation_id: str) -> bool:
    """Drop the journal only if it is still ``generation_id``: an older materializer must never
    erase a newer writer's recovery record."""
    with _transaction(cron_dir) as conn:
        if generation_id == "legacy":
            cur = conn.execute(
                "DELETE FROM pending_definitions WHERE singleton = 1 "
                "AND (generation_id IS NULL OR generation_id = '')")
        else:
            cur = conn.execute(
                "DELETE FROM pending_definitions WHERE singleton = 1 AND generation_id = ?",
                (generation_id,))
    return cur.rowcount == 1


def _write_rows(
    conn: sqlite3.Connection, states: Mapping[str, Mapping[str, Any]], *,
    removed_ids: Collection[str], replace: bool,
) -> None:
    """Upsert ``states`` and delete ``removed_ids``. Rows for other jobs are left alone — under the
    degraded (flock-timeout) path a sibling may have committed them after this writer's load — unless
    ``replace`` asks for the wholesale rewrite ``save_jobs(replace=True)`` promises."""
    if replace:
        keep = {str(k) for k in states}
        existing = [str(r[0]) for r in conn.execute("SELECT job_id FROM job_runtime").fetchall()]
        removed_ids = set(removed_ids) | {job_id for job_id in existing if job_id not in keep}
    if states:
        conn.executemany(
            "INSERT INTO job_runtime(job_id, state_json) VALUES (?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET state_json = excluded.state_json",
            [(str(job_id), serialize_state(state)) for job_id, state in states.items()])
    removed = [(str(job_id),) for job_id in removed_ids if job_id and str(job_id) not in states]
    if removed:
        conn.executemany("DELETE FROM job_runtime WHERE job_id = ?", removed)


def write_runtime_states(
    cron_dir: Path, states: Mapping[str, Mapping[str, Any]], *,
    removed_ids: Collection[str] = (), replace: bool = False,
) -> None:
    """Runtime-only save: nothing declarative changed, so ``jobs.json`` is not touched."""
    if not states and not removed_ids and not replace:
        return
    with _transaction(cron_dir) as conn:
        _write_rows(conn, states, removed_ids=removed_ids, replace=replace)


def stage_runtime_and_definitions(
    cron_dir: Path, states: Mapping[str, Mapping[str, Any]],
    definitions: Sequence[Mapping[str, Any]], *, base_definitions_digest: Optional[str],
    removed_ids: Collection[str] = (), replace: bool = False,
) -> str:
    """Commit runtime rows and the journal of the ``jobs.json`` about to be written, atomically;
    return the generation id to acknowledge once that file is in place."""
    generation_id = uuid.uuid4().hex
    with _transaction(cron_dir) as conn:
        _write_rows(conn, states, removed_ids=removed_ids, replace=replace)
        conn.execute(
            "INSERT INTO pending_definitions"
            "(singleton, definitions_json, generation_id, base_definitions_digest) "
            "VALUES (1, ?, ?, ?) ON CONFLICT(singleton) DO UPDATE SET "
            "definitions_json = excluded.definitions_json, generation_id = excluded.generation_id, "
            "base_definitions_digest = excluded.base_definitions_digest",
            (_dumps(list(definitions)), generation_id, base_definitions_digest))
    return generation_id
