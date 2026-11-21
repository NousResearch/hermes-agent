"""Profile-local durable handoff for cron delivery through live gateway adapters.

A restart-safe cron worker executes outside the gateway cgroup.  It cannot own
relay/E2EE adapter objects, so it queues the final send here.  A gateway claims
each row before sending. Only typed pre-admission Bot Chat refusals restore
pending with bounded backoff. If the gateway dies after claiming, the outcome
is marked unknown and never retried: losing a delivery is safer than duplicating
a possibly-completed send.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.deadline import poll_until
from agent.redact import redact_sensitive_text
from cron.executions import _owner_is_live, _process_start_time
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

DELIVERY_DB: Optional[Path] = None
_PROCESS_ID = uuid.uuid4().hex
_lock = threading.RLock()
_ACTIVE_DELIVERIES: set[str] = set()
_TERMINAL = ("delivered", "failed", "unknown", "suppressed")
MAX_TERMINAL_DELIVERIES = 1000
DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS = 300.0
MAX_REFUSALS = 12


class KnownNotAdmitted(str):
    """Typed proof of pre-admission refusal; never derive this from error prose."""


def _specialized_reference(job: dict, execution_id: str) -> dict:
    reference = {"execution_id": job.get("execution_id"), "job_id": job.get("id"),
                 "delivery_id": job.get("_bot_chat_delivery_id") or execution_id,
                 "home": job.get("_bot_chat_destination") or job["_bot_chat_pending"]}
    if not reference["execution_id"]:
        reference["reference_error"] = "original execution identity unavailable"
    return reference


def _backfill_bot_chat_references(conn: sqlite3.Connection) -> None:
    """Keep attribution before old specialized payloads disappear; never invert a digest."""
    from cron.bot_chat_legacy import _CANONICAL_KEY
    for table in ("deliveries", "delivery_tombstones"):
        fields = "execution_id, job_json, status" if table == "deliveries" else "execution_id"
        rows = conn.execute(f"SELECT {fields} FROM {table} WHERE bot_chat_receipt IS NULL").fetchall()
        for row in rows:
            reference = None
            job = {}
            if table == "deliveries":
                try:
                    job = json.loads(row["job_json"])
                except (ValueError, TypeError):
                    continue  # Malformed evidence cannot supply an execution identity.
                if isinstance(job, dict) and "_bot_chat_pending" in job:
                    reference = _specialized_reference(job, row["execution_id"])
            if (reference is None and job == {} and _CANONICAL_KEY.fullmatch(row["execution_id"])
                    and (table == "delivery_tombstones" or row["status"] in _TERMINAL)):
                reference = {"execution_id": None,
                             "reference_error": "original execution identity unavailable"}
            if reference is not None:
                conn.execute(f"UPDATE {table} SET bot_chat_receipt=? "
                             "WHERE execution_id=? AND bot_chat_receipt IS NULL",
                             (json.dumps(reference), row["execution_id"]))


def _prune_terminal_unlocked(conn: sqlite3.Connection) -> None:
    """Redact terminal payloads and retain only bounded outcome metadata."""
    _backfill_bot_chat_references(conn)
    conn.execute(
        """UPDATE deliveries SET job_json='{}', content=''
           WHERE status IN ('delivered','failed','unknown','suppressed')
             AND (job_json != '{}' OR content != '')"""
    )
    keep = max(0, int(MAX_TERMINAL_DELIVERIES))
    terminal_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM deliveries "
            "WHERE status IN ('delivered','failed','unknown','suppressed')"
        ).fetchone()[0]
    )
    excess = terminal_count - keep
    if excess > 0:
        conn.execute(
            """INSERT OR IGNORE INTO delivery_tombstones
               (execution_id, terminal_status, finished_at, payload_hash, bot_chat_receipt)
               SELECT execution_id, status, finished_at, payload_hash, bot_chat_receipt FROM deliveries
               WHERE status IN ('delivered','failed','unknown','suppressed')
               ORDER BY julianday(finished_at), finished_at,
                        julianday(created_at), created_at, execution_id
               LIMIT ?""",
            (excess,),
        )
        conn.execute(
            """DELETE FROM deliveries WHERE execution_id IN (
                 SELECT execution_id FROM deliveries
                 WHERE status IN ('delivered','failed','unknown','suppressed')
                 ORDER BY julianday(finished_at), finished_at,
                          julianday(created_at), created_at, execution_id
                 LIMIT ?
               )""",
            (excess,),
        )


def queue_path(home: Optional[Path] = None) -> Path:
    """The queue file of ``home`` (the active home when None); a test override wins."""
    if DELIVERY_DB is not None:
        return DELIVERY_DB
    root = Path(home) if home is not None else get_hermes_home()
    return root.resolve() / "cron" / "deliveries.db"


def _path() -> Path:
    return queue_path()


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    # SQLite cannot widen a CHECK in place. Preserve all old rows atomically,
    # including claimed sends, while admitting a distinct never-sent disposition.
    for table in ("deliveries", "delivery_tombstones"):
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if row and "'suppressed'" not in row[0]:
            conn.execute("SAVEPOINT notification_disposition")
            try:
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
                conn.execute(row[0].replace("'unknown'", "'unknown','suppressed'"))
                conn.execute(f"INSERT INTO {table} SELECT * FROM {table}_old")
                conn.execute(f"DROP TABLE {table}_old")
                conn.execute("RELEASE notification_disposition")
            except BaseException:
                conn.execute("ROLLBACK TO notification_disposition")
                conn.execute("RELEASE notification_disposition")
                raise

    conn.execute(
        """CREATE TABLE IF NOT EXISTS deliveries (
             execution_id TEXT PRIMARY KEY,
             job_json TEXT NOT NULL,
             content TEXT NOT NULL,
             for_failure INTEGER NOT NULL DEFAULT 0,
             status TEXT NOT NULL CHECK(status IN
               ('pending','delivering','delivered','failed','unknown','suppressed')),
             owner_process_id TEXT,
             owner_pid INTEGER,
             owner_started_at INTEGER,
             created_at TEXT NOT NULL,
             finished_at TEXT,
             error TEXT
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_tombstones (
             execution_id TEXT PRIMARY KEY,
             terminal_status TEXT NOT NULL CHECK(terminal_status IN
               ('delivered','failed','unknown','suppressed')),
             finished_at TEXT
           )"""
    )
    add_column_if_missing(
        conn, "deliveries", "for_failure",
        "for_failure INTEGER NOT NULL DEFAULT 0",
    )
    for table in ("deliveries", "delivery_tombstones"):
        add_column_if_missing(conn, table, "payload_hash", "payload_hash TEXT")
        add_column_if_missing(conn, table, "bot_chat_receipt", "bot_chat_receipt TEXT")
    add_column_if_missing(
        conn, "deliveries", "retry_count", "retry_count INTEGER NOT NULL DEFAULT 0")
    add_column_if_missing(
        conn, "deliveries", "next_attempt_at", "next_attempt_at REAL DEFAULT 0")
    # Also repair databases opened by an earlier upgrade that added the nullable
    # column but never populated it. Release the migration before callers BEGIN.
    conn.execute("SAVEPOINT bot_chat_references")
    try:
        _backfill_bot_chat_references(conn)
        conn.execute("RELEASE bot_chat_references")
    except BaseException:
        conn.execute("ROLLBACK TO bot_chat_references")
        conn.execute("RELEASE bot_chat_references")
        raise


def _connect(home: Optional[Path] = None) -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from hermes_cli.sqlite_util import open_db

    path = queue_path(home)
    conn = open_db(path, db_label="cron/deliveries.db", synchronous_full=True, initialize=_initialize_schema)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return conn


@contextmanager
def _transaction(home: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    # Pruning is done explicitly by the paths that create terminal
    # rows (_finish / recover_abandoned / _terminalize_wait_timeout);
    # read-only polls must not pay for a full-table UPDATE + COUNT.
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect() if home is None else _connect(home)) as conn:
        yield conn


def canonical_payload_hash(message: str, destination: Path, category: str = "result") -> str:
    """Hash the final redacted render, resolved destination, and notification category.

    Rendering/redaction belongs to the producer, before its first persistence.
    This format deliberately matches the approved pre-v0.23 Bot Chat receipts.
    """
    canonical = {"message": message, "destination": str(Path(destination).resolve()),
                 "category": category}
    return hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _validate_payload_hash(existing: Optional[str], incoming: Optional[str]) -> None:
    if existing != incoming:
        raise ValueError("delivery id already belongs to a different payload")


def enqueue(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    payload_hash: Optional[str] = None,
    rendered_message: Optional[str] = None,
    destination: Optional[Path] = None,
    category: Optional[str] = None,
    origin_home: Optional[Path] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict:
    """Persist one idempotent request, validating Bot Chat collision metadata.

    Generic multi-target jobs have no resolved Bot Chat render here; their legacy
    hash-less contract is retained. Bot Chat producers supply the canonical hash
    or final render + destination. A hash-bearing owner cannot be bypassed by
    omitting these inputs on a later call. To compose admission with an existing
    ownership_arbitration context, pass its conn (the caller owns the commit).
    """
    if rendered_message is not None or destination is not None:
        if rendered_message is None or destination is None:
            raise ValueError("rendered_message and destination are required together")
        computed = canonical_payload_hash(
            rendered_message, destination, category or ("diagnostic" if for_failure else "result"))
        if payload_hash is not None:
            _validate_payload_hash(payload_hash, computed)
        payload_hash = computed
    elif payload_hash is None and job.get("_bot_chat_pending"):
        # The approved specialized envelope pins its own-profile destination.
        # Generic multi-target rows must instead supply their resolved render.
        from cron.bot_chat_legacy import payload_hash as bot_chat_payload_hash
        payload_hash = bot_chat_payload_hash(
            job, content, Path(job.get("_bot_chat_destination") or job["_bot_chat_pending"]),
            for_failure=for_failure)
    from cron import bot_chat_legacy
    delivery_key = job.get("_bot_chat_delivery_id") or str(execution_id)
    bot_chat_key = bot_chat_legacy._CANONICAL_KEY.fullmatch(delivery_key)
    own_transaction = conn is None
    reservation = (bot_chat_legacy.ownership_arbitration(delivery_key, origin_home=origin_home)
                   if own_transaction and bot_chat_key else
                   _transaction(origin_home) if own_transaction else nullcontext(conn))
    with reservation as conn:
        # Bot Chat takes the predecessor stripe before SQLite; generic rows
        # need only serialize their tombstone check and insert with retention.
        if own_transaction and not bot_chat_key:
            conn.execute("BEGIN IMMEDIATE")
        elif not conn.in_transaction:
            raise ValueError("enqueue requires an ownership write reservation")
        if bot_chat_key:
            decision = bot_chat_legacy.arbitrate(
                delivery_key, payload_hash, origin_home=origin_home, conn=conn,
                execution_id=str(execution_id))
            if decision["owned"]:
                return dict(decision["record"], ownership_store=decision["owner"])
        tombstone = conn.execute(
            "SELECT terminal_status, finished_at, payload_hash FROM delivery_tombstones "
            "WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if tombstone is not None:
            _validate_payload_hash(tombstone["payload_hash"], payload_hash)
            return {
                "execution_id": str(execution_id),
                "status": tombstone["terminal_status"],
                "finished_at": tombstone["finished_at"],
                "payload_hash": tombstone["payload_hash"],
            }
        conn.execute(
            """INSERT OR IGNORE INTO deliveries
               (execution_id, job_json, content, for_failure, status, created_at, payload_hash)
               VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
            (
                str(execution_id),
                json.dumps(job, ensure_ascii=False, sort_keys=True),
                str(content),
                int(bool(for_failure)),
                _hermes_now().isoformat(),
                payload_hash,
            ),
        )
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
        _validate_payload_hash(row["payload_hash"], payload_hash)
    return dict(row)


def get_status(execution_id: str, *, origin_home: Optional[Path] = None) -> Optional[dict]:
    with _transaction(origin_home) as conn:
        return _get_status_unlocked(conn, execution_id)


def _get_status_unlocked(conn: sqlite3.Connection, execution_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
    ).fetchone()
    if row is not None:
        return dict(row)
    tombstone = conn.execute(
        "SELECT execution_id, terminal_status, finished_at, payload_hash, bot_chat_receipt "
        "FROM delivery_tombstones WHERE execution_id=?",
        (str(execution_id),),
    ).fetchone()
    if tombstone is None:
        return None
    return {
        "execution_id": tombstone["execution_id"],
        "status": tombstone["terminal_status"],
        "finished_at": tombstone["finished_at"],
        "payload_hash": tombstone["payload_hash"],
        "bot_chat_receipt": tombstone["bot_chat_receipt"],
        "error": None,
    }


def claim_next() -> Optional[dict]:
    """Atomically claim one pending send before touching the transport."""
    pid = os.getpid()
    started = _process_start_time(pid)
    with _transaction() as conn:
        # created_at carries a DST-varying offset: order by instant, not text.
        row = conn.execute(
            "SELECT execution_id FROM deliveries WHERE status='pending' "
            "AND retry_count < ? AND next_attempt_at IS NOT NULL AND next_attempt_at <= ? "
            "ORDER BY julianday(created_at), created_at, execution_id LIMIT 1",
            (MAX_REFUSALS, time.time()),
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            """UPDATE deliveries SET status='delivering', owner_process_id=?,
               owner_pid=?, owner_started_at=?
               WHERE execution_id=? AND status='pending'
                 AND retry_count < ? AND next_attempt_at IS NOT NULL AND next_attempt_at <= ?""",
            (_PROCESS_ID, pid, started, row["execution_id"], MAX_REFUSALS, time.time()),
        )
        if cur.rowcount != 1:
            return None
        claimed = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (row["execution_id"],)
        ).fetchone()
        job = json.loads(claimed["job_json"])
        if "_bot_chat_pending" not in job:
            # Upgrade the actual clean-upstream full-job shape before dispatch.
            from cron.scheduler_delivery import _prepare_bot_chat_queue, bot_chat_message
            job, content = _prepare_bot_chat_queue(job, claimed["content"],
                                                   for_failure=bool(claimed["for_failure"]))
            if "_bot_chat_pending" in job:
                digest = canonical_payload_hash(
                    bot_chat_message(job, content),
                    Path(job["_bot_chat_destination"]),
                    "diagnostic" if claimed["for_failure"] else "result")
                conn.execute("UPDATE deliveries SET job_json=?, content=?, payload_hash=? WHERE execution_id=?",
                             (json.dumps(job), content, digest, row["execution_id"]))
        reference = None
        if "_bot_chat_pending" in job:
            reference = _specialized_reference(job, row["execution_id"])
        else:
            from cron.scheduler_delivery import _resolve_delivery_targets, _bot_chat_envelope, BOT_CHAT_PLATFORM
            targets = _resolve_delivery_targets(job, for_failure=bool(claimed["for_failure"]))
            bot_targets = []
            for target in targets:
                if target["platform"] == BOT_CHAT_PLATFORM:
                    envelope, _ = _bot_chat_envelope(job, claimed["content"], target["chat_id"],
                                                     for_failure=bool(claimed["for_failure"]))
                    bot_targets.append({"delivery_id": envelope["_bot_chat_delivery_id"],
                                        "home": envelope["_bot_chat_destination"]})
            if bot_targets:
                reference = {"execution_id": job.get("execution_id") or row["execution_id"],
                             "job_id": job.get("id"), "targets": bot_targets,
                             "other_targets": len(targets) != len(bot_targets)}
        if reference is not None:
            conn.execute("UPDATE deliveries SET bot_chat_receipt=? WHERE execution_id=?",
                         (json.dumps(reference), row["execution_id"]))
        claimed = conn.execute("SELECT * FROM deliveries WHERE execution_id=?", (row["execution_id"],)).fetchone()
        _ACTIVE_DELIVERIES.add(row["execution_id"])
    result = dict(claimed)
    result["job"] = json.loads(result.pop("job_json"))
    return result


def _record_bot_chat_suppression(conn: sqlite3.Connection, execution_id: Optional[str],
                                 delivery_id: str) -> None:
    """Retain a target's local disposition under the dispatch ownership reservation."""
    if not execution_id:
        return
    row = conn.execute(
        "SELECT bot_chat_receipt FROM deliveries WHERE execution_id=? AND status='delivering' "
        "AND owner_process_id=? AND owner_pid=?",
        (execution_id, _PROCESS_ID, os.getpid())).fetchone()
    if row is None or not row["bot_chat_receipt"]:
        return
    reference = json.loads(row["bot_chat_receipt"])
    for target in reference.get("targets", [reference]):
        if target.get("delivery_id") == delivery_id:
            target["local_disposition"] = "suppressed"
    conn.execute("UPDATE deliveries SET bot_chat_receipt=? WHERE execution_id=?",
                 (json.dumps(reference), execution_id))


def _park_bot_chat_handoff(execution_id: str) -> bool:
    """Release transport ownership, but never claim or replay an admitted turn."""
    with _transaction() as conn:
        row = conn.execute("SELECT bot_chat_receipt FROM deliveries WHERE execution_id=?",
                           (execution_id,)).fetchone()
        if row is None or not row["bot_chat_receipt"]:
            return False
        reference = json.loads(row["bot_chat_receipt"])
        reference["handoff"] = True
        cur = conn.execute(
            "UPDATE deliveries SET status='pending', next_attempt_at=NULL, job_json='{}', content='', "
            "owner_process_id=NULL, owner_pid=NULL, owner_started_at=NULL, bot_chat_receipt=? "
            "WHERE execution_id=? AND status='delivering' AND owner_process_id=? AND owner_pid=?",
            (json.dumps(reference), execution_id, _PROCESS_ID, os.getpid()))
    return cur.rowcount == 1


def get_execution_status(execution_id: str) -> Optional[dict]:
    """Resolve a retained imported digest through its original execution reference."""
    with _transaction() as conn:
        row = _get_status_unlocked(conn, execution_id)
        if row is not None:
            return row
        keys = []
        for table in ("deliveries", "delivery_tombstones"):
            keys.extend(r[0] for r in conn.execute(
                f"SELECT execution_id FROM {table} WHERE "
                "CASE WHEN json_valid(bot_chat_receipt) THEN "
                "json_extract(bot_chat_receipt, '$.execution_id') END = ?", (execution_id,)))
        # Multiple retained owners are conflicting evidence, not a choice of winner.
        return _get_status_unlocked(conn, keys[0]) if len(keys) == 1 else None


def _refresh_bot_chat_handoffs() -> None:
    from cron.delivery_outcome import settle_quietly
    with _transaction() as conn:
        references = [r[0] for r in conn.execute(
            "SELECT bot_chat_receipt FROM deliveries WHERE status='pending' AND next_attempt_at IS NULL "
            "AND CASE WHEN json_valid(bot_chat_receipt) THEN json_extract(bot_chat_receipt, '$.handoff') END = 1")]
    for encoded in references:
        reference = json.loads(encoded)
        settle_quietly(reference.get("job_id"), reference.get("execution_id"))


def _finish(execution_id: str, *, error: Optional[str], suppressed: bool = False,
            unknown: bool = False) -> bool:
    status = "unknown" if unknown else "failed" if error else "suppressed" if suppressed else "delivered"
    safe_error = (
        redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
        if error
        else None
    )
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE deliveries SET status=?, finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'
                 AND owner_process_id=? AND owner_pid=?""",
            (
                status,
                _hermes_now().isoformat(),
                safe_error,
                execution_id,
                _PROCESS_ID,
                os.getpid(),
            ),
        )
        _prune_terminal_unlocked(conn)
    return cur.rowcount == 1


def _defer_known_refusal(execution_id: str, error: KnownNotAdmitted) -> bool:
    """Restore only this process/PID's proven refusal; exhausted rows stay parked."""
    if not isinstance(error, KnownNotAdmitted):
        return False
    safe_error = redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
    with _transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT retry_count FROM deliveries WHERE execution_id=? AND status='delivering' "
            "AND owner_process_id=? AND owner_pid=?",
            (execution_id, _PROCESS_ID, os.getpid()),
        ).fetchone()
        if row is None:
            return False
        count = row["retry_count"] + 1
        due = time.time() + min(30 * 2 ** (count - 1), 300) if count < MAX_REFUSALS else None
        cur = conn.execute(
            "UPDATE deliveries SET status='pending', retry_count=?, next_attempt_at=?, "
            "owner_process_id=NULL, owner_pid=NULL, owner_started_at=NULL, error=? "
            "WHERE execution_id=? AND status='delivering' AND owner_process_id=? AND owner_pid=?",
            (count, due, safe_error, execution_id, _PROCESS_ID, os.getpid()),
        )
    return cur.rowcount == 1


def recover_abandoned() -> int:
    """Fence dead delivery owners as unknown; never replay uncertain sends.

    The firer settled long ago against a ``queued`` row, so each fenced row's outcome is written
    back here (job ``delivery_failed`` + error, ledger ``failed``) instead of leaving the run at
    ``delivery_queued`` forever."""
    fenced: list[tuple[Optional[str], str]] = []
    changed = 0
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT execution_id, job_json, bot_chat_receipt, owner_process_id, owner_pid, owner_started_at "
            "FROM deliveries WHERE status='delivering'"
        ).fetchall()
        for row in rows:
            same_process = row["owner_process_id"] == _PROCESS_ID
            if same_process:
                with _lock:
                    if row["execution_id"] in _ACTIVE_DELIVERIES:
                        continue
            elif _owner_is_live(int(row["owner_pid"]), row["owner_started_at"]):
                continue
            error = (
                "Gateway finished delivery but could not persist its outcome; "
                "send was not retried."
                if same_process
                else "Gateway exited during delivery; send outcome is unknown and was not retried."
            )
            cur = conn.execute(
                """UPDATE deliveries SET status='unknown', finished_at=?, error=?
                   WHERE execution_id=? AND status='delivering'""",
                (
                    _hermes_now().isoformat(),
                    error,
                    row["execution_id"],
                ),
            )
            if cur.rowcount:
                changed += 1
                # Capture the original run before pruning scrubs the envelope.
                # A delivery digest is never a substitute for a lost execution id.
                job = json.loads(row["job_json"] or "{}")
                reference = json.loads(row["bot_chat_receipt"] or "null")
                if reference is None and "_bot_chat_pending" in job:
                    reference = _specialized_reference(job, row["execution_id"])
                    conn.execute("UPDATE deliveries SET bot_chat_receipt=? WHERE execution_id=?",
                                 (json.dumps(reference), row["execution_id"]))
                if reference is not None:
                    if reference.get("execution_id"):
                        fenced.append((reference.get("job_id"), reference["execution_id"]))
                else:
                    fenced.append((job.get("id"), row["execution_id"]))
        _prune_terminal_unlocked(conn)
    from cron.delivery_outcome import settle_quietly

    for job_id, execution_id in fenced:
        settle_quietly(job_id, execution_id)
    return changed


def drain(
    send: Callable[[dict, str, bool], Optional[str]], *, limit: int = 20
) -> int:
    """Drain due rows; only typed pre-admission Bot Chat refusals may defer."""
    recover_abandoned()
    _refresh_bot_chat_handoffs()
    processed = 0
    for _ in range(max(0, limit)):
        row = claim_next()
        if row is None:
            break
        with _lock:
            _ACTIVE_DELIVERIES.add(row["execution_id"])
        try:
            try:
                if row.get("bot_chat_receipt"):
                    row["job"]["_delivery_queue_id"] = row["execution_id"]
                error = send(
                    row["job"], row["content"], bool(row["for_failure"])
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            bot_chat = "_bot_chat_pending" in row["job"]
            if isinstance(error, KnownNotAdmitted) and bot_chat:
                _defer_known_refusal(row["execution_id"], error)
            elif not error and row["job"].get("_bot_chat_handoff"):
                _park_bot_chat_handoff(row["execution_id"])
            elif _finish(row["execution_id"], error=error,
                         suppressed=bool(row["job"].get("_notification_all_targets_suppressed")),
                         **({"unknown": True} if bot_chat and error else {})):
                from cron.delivery_outcome import settle_quietly

                execution = row["job"].get("execution_id") if bot_chat else row["execution_id"]
                if execution:
                    settle_quietly(row["job"].get("id"), execution)
        finally:
            with _lock:
                _ACTIVE_DELIVERIES.discard(row["execution_id"])
        processed += 1
    return processed


def _terminalize_wait_timeout(execution_id: str) -> str:
    """Fence a delivery whose worker can no longer wait for confirmation.

    A row still ``pending`` was provably never attempted, so it is left queued
    for whichever gateway comes up next (a restart that includes an update can
    easily exceed the worker's wait budget).  That is a deferral, not a
    failure: report success so the job is not recorded ``delivery_failed`` for
    a message the drain will still send.  Only a row caught mid-send is
    uncertain and gets fenced ``unknown``.
    """
    now = _hermes_now().isoformat()
    uncertain_error = (
        "timed out while gateway delivery was in progress; outcome is unknown and "
        "was not retried"
    )
    with _transaction() as conn:
        row = conn.execute(
            "SELECT status FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if row is not None and row["status"] == "pending":
            logger.warning(
                "Cron delivery %s: no live gateway within the wait budget; "
                "left queued for the next gateway",
                execution_id,
            )
            return ""
        conn.execute(
            """UPDATE deliveries SET status='unknown', finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'""",
            (now, uncertain_error, str(execution_id)),
        )
        row = conn.execute(
            "SELECT status, error FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        _prune_terminal_unlocked(conn)
    if row is None:
        return "timed out waiting for live gateway delivery"
    if row["status"] in {"delivered", "suppressed"}:
        return ""
    return str(row["error"] or f"delivery {row['status']}")


def enqueue_and_wait(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Queue delivery and wait for a gateway's terminal at-most-once outcome."""
    queued = enqueue(execution_id, job, content, for_failure=for_failure)
    if queued["status"] in _TERMINAL:
        return None if queued["status"] in {"delivered", "suppressed"} else str(
            queued.get("error") or f"delivery {queued['status']}"
        )
    wait_timeout = (
        DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    )
    def _terminal_row() -> Optional[dict]:
        row = get_status(execution_id)
        return row if row and row["status"] in _TERMINAL else None

    row = poll_until(_terminal_row, wait_timeout, 1.0)
    if row is None:
        return _terminalize_wait_timeout(execution_id) or None
    return None if row["status"] in {"delivered", "suppressed"} else str(
        row.get("error") or f"delivery {row['status']}"
    )
