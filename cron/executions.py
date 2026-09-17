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
    # Durable intent written BEFORE any deferred (Bot Chat) send: while set, the row's manifest is
    # incomplete and no reader may project/settle/prune from it. Cleared by the manifest write itself.
    add_column_if_missing(conn, "executions", "delivery_manifest_pending", "delivery_manifest_pending INTEGER NOT NULL DEFAULT 0")
    add_column_if_missing(conn, "executions", "delivery_outcome", "delivery_outcome TEXT")
    add_column_if_missing(conn, "executions", "scheduled_instant", "scheduled_instant TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_occurrence "
        "ON executions(job_id, scheduled_instant) WHERE status='completed'"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    # Column existence is NOT the migration authority (DDL commits on its own). The fence row
    # below is: until it exists, readers hold pre-flag rows conservatively and every
    # initialization retries adoption.
    _try_adopt_legacy_intent(conn)


_ADOPTION_KEY = "manifest_intent_adopted"
_adoption_complete: set = set()  # ledger paths whose fence row was observed (monotonic: it never disappears)


def _ledger_key() -> str:
    return str(EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db"))


def _legacy_intent_adopted(conn: Optional[sqlite3.Connection] = None) -> bool:
    """True once the one-time adoption of pre-flag journaled intent has durably completed."""
    key = _ledger_key()
    if key in _adoption_complete:
        return True
    if conn is None:
        with _transaction() as own:
            return _legacy_intent_adopted(own)
    row = conn.execute("SELECT 1 FROM schema_meta WHERE key=?", (_ADOPTION_KEY,)).fetchone()
    if row is not None:
        _adoption_complete.add(key)
    return row is not None


def _journal_inventory_strict() -> Dict[str, dict]:
    """Journal inventory that RAISES on any unreadable directory or entry.

    Adoption must distinguish a genuinely empty journal from one it could not read: an
    incomplete inventory is not proof that no outstanding intent exists.
    """
    root = _journal_root()
    if not root.exists():
        return {}
    out: Dict[str, dict] = {}
    for path in sorted(root.iterdir()):  # raises PermissionError on an unreadable directory
        if path.suffix != ".json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))  # raises on an unreadable entry
        if record.get("execution_id") and record.get("manifest"):
            out[record["execution_id"]] = record["manifest"]
    return out


def _try_adopt_legacy_intent(conn: sqlite3.Connection) -> bool:
    """One-time, retryable, atomic adoption of pre-flag outstanding journals.

    In ONE write transaction: every row whose real manifest is still only journaled gets
    ``delivery_manifest_pending=1``, then the fence row is written. Any failure (unreadable
    journal, rejected write) rolls back everything, logs, and leaves the fence absent so the
    next initialization retries and readers keep holding legacy rows conservatively.
    """
    if _legacy_intent_adopted(conn):
        return True
    if conn.in_transaction:
        # A caller (e.g. incident upsert) owns an open transaction; adoption must not nest a
        # BEGIN inside it nor piggyback on its commit. Defer to the next ordinary ledger open.
        logging.getLogger(__name__).debug("Cron ledger: legacy intent adoption deferred (caller transaction open)")
        return False
    try:
        inventory = _journal_inventory_strict()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute("SELECT 1 FROM schema_meta WHERE key=?", (_ADOPTION_KEY,)).fetchone():
                conn.execute("ROLLBACK")  # a concurrent initializer won the race
                _adoption_complete.add(_ledger_key())
                return True
            for execution_id in inventory:
                conn.execute(
                    "UPDATE executions SET delivery_manifest_pending=1 WHERE id=? AND "
                    "(delivery_manifest IS NULL OR instr(delivery_manifest, '\"bot\"')=0)",
                    (execution_id,))
            conn.execute("INSERT INTO schema_meta(key, value) VALUES (?, ?)", (_ADOPTION_KEY, _hermes_now().isoformat()))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    except Exception:
        logging.getLogger(__name__).error(
            "Cron ledger: legacy delivery-intent adoption incomplete; pre-flag rows are held until it succeeds",
            exc_info=True)
        return False
    _adoption_complete.add(_ledger_key())
    return True


def _placeholder_only(record: Optional[dict]) -> bool:
    manifest = record.get("delivery_manifest") if record else None
    return bool(manifest) and "bot" not in json.loads(manifest)


def _journal_entry_state(execution_id: str) -> Optional[bool]:
    """True: this row has a journal entry. False: verified absent. None: could not tell."""
    path = _journal_root() / f"{execution_id}.json"
    try:
        return path.is_file()
    except OSError:
        return None


def _held_by_incomplete_adoption(record: Optional[dict], conn: Optional[sqlite3.Connection] = None) -> bool:
    """Conservative fence while adoption is incomplete: a row carrying only the external
    placeholder may still have an outstanding journaled child we cannot see."""
    if not record or _legacy_intent_adopted(conn):
        return False
    return _placeholder_only(record)


def _late_legacy_intent(record: Optional[dict], conn: Optional[sqlite3.Connection] = None) -> bool:
    """Old-writer accounting at terminalization (K2c).

    A pre-flag process that outlives adoption publishes its child manifest only to the
    journal. A placeholder-only row with pending=0 is therefore complete ONLY if its own
    journal entry is verifiably absent. When present, the intent is adopted here (in the
    caller's transaction when given) so every later reader sees the flag; when unknowable,
    the row is held. Rows that already carry "bot" are past this window and never probed.
    """
    if not record or record.get("delivery_manifest_pending") or not _placeholder_only(record):
        return False
    state = _journal_entry_state(record["id"])
    if state is False:
        return False
    if state is True and conn is not None:
        # Write-time recheck: `record` is a snapshot taken before the stat. A replay that landed
        # the complete manifest in between must not be re-armed with a flag nothing can clear.
        cur = conn.execute(
            "UPDATE executions SET delivery_manifest_pending=1 WHERE id=? AND delivery_manifest_pending=0 "
            "AND delivery_manifest IS NOT NULL AND instr(delivery_manifest, '\"bot\"')=0",
            (record["id"],))
        if cur.rowcount == 0:
            # The row moved on (complete manifest landed, or already flagged): defer to the
            # authoritative row state rather than the stale snapshot.
            current = _fetch(conn, record["id"])
            return bool(current and (current.get("delivery_manifest_pending") or _placeholder_only(current)))
    return True


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
    # While legacy-intent adoption is incomplete, rows carrying only a placeholder manifest may
    # still anchor an unseen journaled child: never prune them.
    hold = "" if _legacy_intent_adopted(conn) else " AND (delivery_manifest IS NULL OR instr(delivery_manifest, '\"bot\"')>0)"
    conn.execute(
        f"""DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
               AND (delivery_outcome IS NULL OR delivery_outcome != 'queued')
               AND (delivery_manifest IS NULL OR delivery_projection_settled=1)
               AND delivery_manifest_pending=0{hold}
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
    caller_outcome = delivery_outcome
    with _transaction() as conn:
        cur = None
        for _attempt in range(2):
            # Derive the terminal delivery decision from the AUTHORITATIVE row contents and
            # write it with a CAS on the exact manifest it was derived from. A concurrent
            # replay that lands the complete manifest between derivation and write fails the
            # CAS; the second pass derives again from what is there now. The row is ours
            # (process/pid), so this never double-terminalizes.
            row = _fetch(conn, execution_id)
            derived_from = row.get("delivery_manifest") if row else None
            delivery_outcome = caller_outcome
            if row and (row.get("delivery_manifest_pending") or _held_by_incomplete_adoption(row, conn)
                        or _late_legacy_intent(row, conn)):
                # Deferred children were sent but their manifest is not in the ledger yet: the
                # caller's classification (and any placeholder) is incomplete by construction.
                delivery_outcome = "queued"
            elif row and derived_from:
                projection = _delivery_projection(row)
                if projection is not None:
                    delivery_outcome = projection[0]
                elif manifest_pending(row):
                    delivery_outcome = "queued"
            cur = conn.execute(
                """UPDATE executions
                   SET status=?, finished_at=?, error=?, handoff_pending=0,
                       handoff_started_at=NULL, delivery_outcome=?
                   WHERE id=? AND status IN ('claimed','running')
                     AND process_id=? AND pid=? AND delivery_manifest IS ?""",
                (status, now, detail, delivery_outcome, execution_id, _PROCESS_ID, os.getpid(), derived_from),
            )
            if cur.rowcount == 1:
                break
        if cur is None or cur.rowcount != 1:
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
    """CAS the manifest onto a row that has none (or only the external placeholder).

    Clearing ``delivery_manifest_pending`` is part of the SAME statement: a storage fault that
    rejects the manifest leaves the intent flag set, so no reader can misread the placeholder.
    """
    row = _fetch(conn, execution_id)
    existing = row.get("delivery_manifest") if row else None
    if existing and "bot" not in json.loads(existing):
        # Any placeholder-only manifest (canonical or not) is incomplete by the same predicate
        # every reader uses; the complete manifest replaces it and keeps its external marker.
        manifest = {**manifest, "external": True}
    conn.execute("UPDATE executions SET delivery_manifest=?, delivery_manifest_pending=0 WHERE id=? AND "
                 "(delivery_manifest IS NULL OR instr(delivery_manifest, '\"bot\"')=0)",
                 (json.dumps(manifest, sort_keys=True), execution_id))


def mark_delivery_manifest_pending(execution_id: Optional[str]) -> None:
    """Record, BEFORE any deferred send, that this run will own child receipts.

    Raises when the ledger cannot take the write: nothing has been sent yet, so the caller
    may truthfully report a pre-send failure instead of sending without durable intent.
    """
    if not execution_id:
        return
    with _transaction() as conn:
        conn.execute("UPDATE executions SET delivery_manifest_pending=1 WHERE id=?", (execution_id,))


def manifest_pending(record: Optional[dict]) -> bool:
    """True while the row's deferred-child manifest is not yet in the ledger."""
    if not record or not record.get("delivery_manifest_pending"):
        return False
    manifest = record.get("delivery_manifest")
    return not (manifest and "bot" in json.loads(manifest))


def _journal_root() -> Path:
    return (EXECUTIONS_FILE.parent if EXECUTIONS_FILE else get_hermes_home().resolve() / "cron") / "manifest_journal"


def journal_manifest(execution_id: str, manifest: dict) -> bool:
    """Best-effort durable copy of a manifest the ledger could not store (post-send fault).

    The row's ``delivery_manifest_pending`` flag is the authoritative invariant; this journal
    only lets the reconciler recover the manifest once the ledger accepts writes again.
    Never raises: a second storage failure is logged, and the row simply stays pending.
    """
    try:
        # The row flag is the authoritative fence; re-assert it here (idempotent) so a journal
        # entry never exists for a row that readers would treat as complete.
        with _transaction() as conn:
            conn.execute("UPDATE executions SET delivery_manifest_pending=1 WHERE id=?", (execution_id,))
    except Exception:
        logging.getLogger(__name__).debug("Could not re-assert manifest intent for %s", execution_id, exc_info=True)
    try:
        from utils import atomic_json_write
        root = _journal_root()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_json_write(root / f"{execution_id}.json", {"execution_id": execution_id, "manifest": manifest},
                          fsync_dir=True, mode=0o600)
        return True
    except Exception:
        logging.getLogger(__name__).exception(
            "Cron delivery manifest for %s could not be journaled; row stays pending", execution_id)
        return False


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

    Idempotent through the same CAS as the primary write (which also clears the row's pending
    flag). An entry is removed only once the ledger row actually carries the children.
    Callers must NOT use the return value as an authority: the row flag is authoritative.
    """
    outstanding: Dict[str, dict] = {}
    for execution_id, manifest in journaled_manifests().items():
        try:
            # A journal entry IS outstanding intent (legacy rows predating the flag, or a mark
            # whose re-assert failed). Assert it in its OWN committed transaction first: if the
            # replay below is rejected by storage, the fence must survive the rollback.
            with _transaction() as conn:
                row = _fetch(conn, execution_id)
                if row is None:
                    outstanding[execution_id] = manifest
                    continue
                conn.execute("UPDATE executions SET delivery_manifest_pending=1 WHERE id=? AND "
                             "(delivery_manifest IS NULL OR instr(delivery_manifest, '\"bot\"')=0)",
                             (execution_id,))
                # A terminal classification derived from a placeholder-only manifest was, by
                # construction, computed without the children this journal now supplies (a
                # pre-flag writer publishing late). Reopen exactly that row, in the same
                # transaction as the intent, so the real manifest is projected once it lands.
                conn.execute("UPDATE executions SET delivery_outcome='queued', delivery_projection_settled=0 "
                             "WHERE id=? AND delivery_manifest_pending=1 AND delivery_outcome IN ('delivered','failed') "
                             "AND (delivery_manifest IS NULL OR instr(delivery_manifest, '\"bot\"')=0)",
                             (execution_id,))
            with _transaction() as conn:
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

    if manifest_pending(record) or _held_by_incomplete_adoption(record) or _late_legacy_intent(record):
        return None  # children were sent; their manifest is not in the ledger yet
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
        _recover_unrecorded_manifests()
    except Exception:
        logging.getLogger(__name__).exception("Could not replay journaled cron delivery manifests")
    with _transaction() as conn:
        records = [dict(row) for row in conn.execute(
            "SELECT * FROM executions WHERE delivery_manifest IS NOT NULL "
            "AND status IN ('completed','failed','unknown')").fetchall()]
    for record in records:
        try:
            # A pending row (children sent, manifest not yet in the ledger) projects to None
            # inside _delivery_projection; the CAS below re-checks the flag at write time so a
            # reader that loaded the row before the flag flipped cannot settle it either.
            projection = _delivery_projection(record)
            if projection is None:
                continue
            outcome, values = projection
            with _transaction() as conn:
                conn.execute("UPDATE executions SET delivery_outcome=? WHERE id=? "
                             "AND delivery_outcome='queued' AND delivery_manifest=? "
                             "AND delivery_manifest_pending=0",
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
                                 "WHERE id=? AND delivery_outcome=? AND delivery_manifest=? "
                                 "AND delivery_manifest_pending=0",
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
                       handoff_pending=0, handoff_started_at=NULL,
                       delivery_outcome=CASE WHEN delivery_manifest_pending=1 OR (? AND delivery_manifest IS NOT NULL
                                                  AND instr(delivery_manifest, '"bot"')=0)
                                             THEN 'queued' ELSE delivery_outcome END
                   WHERE id=? AND status=? AND process_id=? AND pid=?
                     AND handoff_pending=?
                     AND handoff_started_at IS ?""",
                (now,
                 "Scheduler restarted after this execution's owner exited before a durable "
                 "terminal state; whether side effects ran is unknown.",
                 0 if _legacy_intent_adopted(conn) else 1,
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
