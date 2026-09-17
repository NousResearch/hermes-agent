"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue. Interrupted attempts
become ``unknown`` only after their exact owner process is proved gone. Terminal states are
immutable.
"""

from __future__ import annotations

import os
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override. Production resolves the path at transaction time so dashboard operations
# that temporarily enter another profile cannot leak that profile's records into the import-time
# home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
HANDOFF_ADOPTION_GRACE_SECONDS = 30.0
_TERMINAL_STATES = ("completed", "failed", "unknown")
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


# --- executions ledger --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/executions.db", synchronous_full=True, initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    conn.execute(
        """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             handoff_pending INTEGER NOT NULL DEFAULT 0,
             handoff_started_at REAL,
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""
    )
    add_column_if_missing(
        conn, "executions", "handoff_pending",
        "handoff_pending INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, "executions", "handoff_started_at", "handoff_started_at REAL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    add_column_if_missing(conn, "executions", "delivery_manifest", "delivery_manifest TEXT")
    add_column_if_missing(conn, "executions", "incident_id", "incident_id TEXT")
    add_column_if_missing(conn, "executions", "incident_generation", "incident_generation INTEGER")
    add_column_if_missing(conn, "executions", "delivery_projection_settled", "delivery_projection_settled INTEGER NOT NULL DEFAULT 0")
    add_column_if_missing(conn, "executions", "delivery_outcome", "delivery_outcome TEXT")
    add_column_if_missing(conn, "executions", "scheduled_instant", "scheduled_instant TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_occurrence "
        "ON executions(job_id, scheduled_instant) WHERE status='completed'"
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def _fetch(conn: sqlite3.Connection, execution_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[Dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    return current is not None and current == started_at


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
               AND (delivery_outcome IS NULL OR delivery_outcome != 'queued')
               AND (delivery_manifest IS NULL OR delivery_projection_settled=1)
             ORDER BY finished_at DESC, claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (max(0, int(MAX_TERMINAL_EXECUTIONS)),),
    )


def create_execution(
    job_id: str, *, source: str, scheduled_instant: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    from cron.occurrences import scheduled_instant as canonical_instant

    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, scheduled_instant)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now, canonical_instant(scheduled_instant)),
        )
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def set_execution_occurrence(execution_id: str, instant: Optional[str]) -> None:
    """Bind the store-claimed snapshot before a provider hands it to a worker."""
    from cron.occurrences import scheduled_instant

    with _transaction() as conn:
        cur = conn.execute(
            "UPDATE executions SET scheduled_instant=? WHERE id=? AND status='claimed' "
            "AND handoff_pending=0 AND process_id=? AND pid=?",
            (scheduled_instant(instant), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Cron occurrence could not be bound before dispatch")


def mark_execution_handoff_pending(execution_id: str) -> Optional[Dict[str, Any]]:
    """Fence restart recovery while an external worker is adopting a claim."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET handoff_pending=1, handoff_started_at=?
               WHERE id=? AND status='claimed'
                 AND process_id=? AND pid=?""",
            (time.time(), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def adopt_claimed_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Atomically transfer and start an attempt in its worker process.

    The dispatching gateway creates the row before spawning a restart-safe
    worker.  Adoption is the single ``claimed`` → ``running`` gate: only the
    winner may acknowledge ownership or run side effects.
    """
    pid = os.getpid()
    process_started_at = _process_start_time(pid)
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET process_id=?, pid=?, process_started_at=?,
                   status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=1""",
            (_PROCESS_ID, pid, process_started_at, now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def mark_execution_running(execution_id: str) -> Optional[Dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=0
                 AND process_id=? AND pid=?""",
            (now, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten."""
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    # A worker may observe a terminal receipt during its wait. Use the same durable
    # classification as restart reconciliation, in particular unknown != failed.
    with _transaction() as conn:
        pending = _fetch(conn, execution_id)
        if pending and pending.get("delivery_manifest"):
            projection = _delivery_projection(pending)
            if projection is not None:
                delivery_outcome = projection[0]
        cur = conn.execute(
            """UPDATE executions
               SET status=?, finished_at=?, error=?, handoff_pending=0,
                   handoff_started_at=NULL, delivery_outcome=?
               WHERE id=? AND status IN ('claimed','running')
                 AND process_id=? AND pid=?""",
            (status, now, detail, delivery_outcome, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _fetch(conn, execution_id)
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    if record and record.get("delivery_manifest"):
        reconcile_delivery_projections()
    return record


def bind_delivery_incident(execution_id: Optional[str], incident_id: str) -> None:
    """Persist the producer's actual incident association, never infer it from errors."""
    if execution_id:
        with _transaction() as conn:
            from cron.incidents import _initialize_schema as init_incidents
            init_incidents(conn)
            _bind_delivery_incident_unlocked(conn, execution_id, incident_id)


def _bind_delivery_incident_unlocked(conn, execution_id: str, incident_id: str) -> None:
    """Called in the producer's incident upsert transaction; no later generation lookup."""
    conn.execute("UPDATE executions SET incident_id=?, incident_generation="
                 "(SELECT generation FROM cron_incidents WHERE id=?) "
                 "WHERE id=? AND incident_id IS NULL "
                 "AND status IN ('claimed','running') AND process_id=? AND pid=?",
                 (incident_id, incident_id, execution_id, _PROCESS_ID, os.getpid()))


def record_delivery_manifest(execution_id: Optional[str], manifest: dict) -> None:
    """Retain receipt identities and sibling disposition separately from the run result.

    The worker may first bind the external handoff; its gateway replaces that placeholder
    with the resolved target receipts. No payload or raw output is copied here.
    """
    if not execution_id:
        return
    with _transaction() as conn:
        _store_manifest(conn, execution_id, manifest)


def _store_manifest(conn, execution_id: str, manifest: dict) -> None:
    """CAS the manifest onto a row that has none (or only the external placeholder)."""
    placeholder = json.dumps({"external": True}, sort_keys=True)
    row = _fetch(conn, execution_id)
    if row and row.get("delivery_manifest") == placeholder:
        manifest = {**manifest, "external": True}
    conn.execute("UPDATE executions SET delivery_manifest=? WHERE id=? AND "
                 "(delivery_manifest IS NULL OR delivery_manifest=?)",
                 (json.dumps(manifest, sort_keys=True), execution_id, placeholder))


def _journal_root() -> Path:
    return (EXECUTIONS_FILE.parent if EXECUTIONS_FILE else get_hermes_home().resolve() / "cron") / "manifest_journal"


def journal_manifest(execution_id: str, manifest: dict) -> None:
    """Durably record a delivery manifest the ledger could not store (post-send fault).

    Source-owned: independent of which receipt store (deferred file or live-owner mailbox)
    holds the children, and never mutated by receipt consumers. Replayed by
    :func:`_recover_unrecorded_manifests`; deleted only after the ledger holds it.
    """
    from utils import atomic_json_write
    root = _journal_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_json_write(root / f"{execution_id}.json", {"execution_id": execution_id, "manifest": manifest},
                      fsync_dir=True, mode=0o600)


def journaled_manifests() -> Dict[str, dict]:
    root = _journal_root()
    if not root.is_dir():
        return {}
    out: Dict[str, dict] = {}
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            logging.getLogger(__name__).error("Unreadable cron manifest journal entry %s", path)
            continue
        if record.get("execution_id") and record.get("manifest"):
            out[record["execution_id"]] = record["manifest"]
    return out


def _recover_unrecorded_manifests() -> Dict[str, dict]:
    """Replay journaled manifests into the ledger; return what is STILL outstanding.

    Idempotent through the same CAS as the primary write. An entry is removed only once
    the ledger row actually carries a manifest with the children; while outstanding, the
    row's placeholder manifest must not be projected or pruned (see callers).
    """
    outstanding: Dict[str, dict] = {}
    for execution_id, manifest in journaled_manifests().items():
        try:
            with _transaction() as conn:
                row = _fetch(conn, execution_id)
                if row is None:
                    outstanding[execution_id] = manifest
                    continue
                _store_manifest(conn, execution_id, manifest)
                row = _fetch(conn, execution_id)
            stored = json.loads(row["delivery_manifest"]) if row and row.get("delivery_manifest") else {}
            if stored.get("bot") == manifest.get("bot"):
                (_journal_root() / f"{execution_id}.json").unlink(missing_ok=True)
            else:
                outstanding[execution_id] = manifest
        except Exception:
            logging.getLogger(__name__).exception("Could not replay journaled cron manifest %s", execution_id)
            outstanding[execution_id] = manifest
    return outstanding


def _delivery_projection(record: dict) -> Optional[tuple[str, dict]]:
    from cron import delivery_queue, bot_chat_delivery
    from tools.bot_live_delivery import read_delivery_result

    manifest = json.loads(record["delivery_manifest"])
    external_outcome = None
    external_error = None
    if manifest.get("external"):
        receipt = delivery_queue.get_status(record["id"])
        if not receipt or receipt["status"] in ("pending", "delivering"):
            return None
        outcome = receipt["status"]
        external_outcome, external_error = outcome, receipt.get("error")
        if not manifest.get("bot"):
            return outcome, {"last_delivery_queued": None, "last_delivery_unverified": None,
                             "last_delivery_error": receipt.get("error")}

    queued = {}
    states = []
    for target, ref in manifest.get("bot", {}).items():
        receipt = read_delivery_result(Path(ref["home"]), ref["delivery_id"])
        if receipt is None:
            receipt = bot_chat_delivery.read_pending(ref["delivery_id"])
        status = (receipt or ref)["status"]
        if status in ("queued", "claimed", "transferred"):
            queued[target] = {**ref, "status": status}
        states.append(status)
    error = manifest.get("error") or external_error
    unverified = manifest.get("unverified")
    if queued:
        outcome = "queued"
    elif external_outcome == "unknown" or "ambiguous" in states or "unknown" in states or unverified:
        outcome = "unknown"
    elif external_outcome == "failed" or "failed" in states or error:
        outcome = "failed"
    elif manifest.get("delivered") or "settled" in states:
        outcome = "delivered"
    else:
        outcome = "suppressed"
    if outcome in ("failed", "unknown") and not error:
        error = "Deferred delivery completion unverified; do not resend"
    return outcome, {"last_delivery_queued": queued or None,
                     "last_delivery_unverified": unverified or None,
                     "last_delivery_error": error}


def reconcile_delivery_projections() -> None:
    """Replay durable receipts, never sends. Safe after either store write or restart.

    Only delivery_outcome can change after run completion. The queued CAS and manifest
    comparison fence racing reconcilers; execution status/error/timestamps are untouched.
    Retaining manifests also repairs a crash between the ledger and jobs/incident writes.
    """
    import logging
    from cron.jobs import update_delivery_projection
    from cron.incidents import _initialize_schema as init_incidents

    try:
        outstanding = _recover_unrecorded_manifests()
    except Exception:
        logging.getLogger(__name__).exception("Could not replay journaled cron delivery manifests")
        outstanding = journaled_manifests()
    with _transaction() as conn:
        records = [dict(row) for row in conn.execute(
            "SELECT * FROM executions WHERE delivery_manifest IS NOT NULL "
            "AND status IN ('completed','failed','unknown')").fetchall()]
    for record in records:
        if record["id"] in outstanding:
            # The ledger holds at most the external placeholder for this run while its real
            # manifest (with pending children) is journaled: an incomplete manifest must not be
            # projected as a terminal disposition. Stay queued until the replay succeeds.
            continue
        try:
            projection = _delivery_projection(record)
            if projection is None:
                continue
            outcome, values = projection
            with _transaction() as conn:
                conn.execute("UPDATE executions SET delivery_outcome=? WHERE id=? "
                             "AND delivery_outcome='queued' AND delivery_manifest=?",
                             (outcome, record["id"], record["delivery_manifest"]))
                current = _fetch(conn, record["id"])
                if not current or current["delivery_outcome"] != outcome:
                    continue
                if outcome == "delivered" and record.get("incident_id"):
                    init_incidents(conn)
                    # Late completion cannot resurrect resolved/acked incidents.
                    conn.execute("UPDATE cron_incidents SET state='alerted' WHERE id=? "
                                 "AND state='detected' AND generation=?",
                                 (record["incident_id"], record.get("incident_generation")))
            update_delivery_projection(record["job_id"], record["id"], values)
            if outcome != "queued":
                # Pin the receipt association until BOTH durable projections have
                # completed. A crash above leaves it replayable even under retention.
                with _transaction() as conn:
                    conn.execute("UPDATE executions SET delivery_projection_settled=1 "
                                 "WHERE id=? AND delivery_outcome=? AND delivery_manifest=?",
                                 (record["id"], outcome, record["delivery_manifest"]))
        except Exception:
            logging.getLogger(__name__).exception(
                "Could not reconcile cron delivery %s; durable receipt retained", record["id"])


def recover_interrupted_executions() -> int:
    """Mark provably abandoned attempts unknown without scheduling retries."""
    now = _hermes_now().isoformat()
    changed = 0
    recovered: List[Dict[str, Any]] = []
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, status, process_id, pid, process_started_at,
                      handoff_pending, handoff_started_at
               FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
        for row in rows:
            if row["process_id"] == _PROCESS_ID:
                continue
            if _owner_is_live(int(row["pid"]), row["process_started_at"]):
                continue
            handoff_started_at = row["handoff_started_at"]
            if (
                row["handoff_pending"]
                and handoff_started_at is not None
                and time.time() - float(handoff_started_at)
                < HANDOFF_ADOPTION_GRACE_SECONDS
            ):
                continue
            cur = conn.execute(
                """UPDATE executions
                   SET status='unknown', finished_at=?, error=?,
                       handoff_pending=0, handoff_started_at=NULL
                   WHERE id=? AND status=? AND process_id=? AND pid=?
                     AND handoff_pending=?
                     AND handoff_started_at IS ?""",
                (now,
                 "Scheduler restarted after this execution's owner exited before a durable "
                 "terminal state; whether side effects ran is unknown.",
                 row["id"], row["status"], row["process_id"], row["pid"],
                 row["handoff_pending"], row["handoff_started_at"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                record = _fetch(conn, row["id"])
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50, before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(str(before_claimed_at))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Return one exact execution attempt, or ``None`` when it is absent."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?",
            (str(execution_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def latest_execution(job_id: str) -> Optional[Dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def latest_executions(job_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load latest execution for many jobs in one indexed query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e
                WHERE e.job_id IN ({placeholders})
                  AND e.id=(SELECT e2.id FROM executions e2
                            WHERE e2.job_id=e.job_id
                            ORDER BY e2.claimed_at DESC, e2.id DESC LIMIT 1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
