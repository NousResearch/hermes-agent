"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue.
Interrupted attempts become ``unknown`` only after their exact owner process is
proved gone. Terminal states are immutable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import unicodedata
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override. Production resolves the path at transaction time so
# dashboard operations that temporarily enter another profile cannot leak that
# profile's execution records into the import-time home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
_RECEIPT_SCHEMA_VERSION = 5
HANDOFF_ADOPTION_GRACE_SECONDS = 30.0
_TERMINAL_STATES = ("completed", "failed", "unknown")
_EXECUTION_ERROR_KINDS = frozenset({
    "blocked_config",
    "claim_lost",
    "dispatch_failed",
    "execution_failed",
    "interrupted",
    "legacy_redacted",
    "unknown",
})
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


def scheduled_fire_identity(job_id: str, scheduled_for: str) -> str:
    """Return a stable opaque identity for one job's scheduled occurrence."""
    normalized_job_id = _bounded_text(job_id, field="job_id", limit=256)
    normalized_time = _bounded_text(scheduled_for, field="scheduled_for", limit=128)
    parsed = datetime.fromisoformat(normalized_time.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    canonical = parsed.astimezone(timezone.utc).isoformat()
    return hashlib.sha256(f"{normalized_job_id}\0{canonical}".encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    from cron.jobs import _ensure_cron_dir

    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    _ensure_cron_dir(path.parent)
    return sqlite3.connect(path, timeout=5)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    apply_wal_with_fallback(conn, db_label="cron/executions.db")
    conn.execute("PRAGMA synchronous=FULL")
    # Keep every schema migration and privacy scrub failure-atomic.
    conn.execute("BEGIN IMMEDIATE")
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
             error TEXT,
             error_kind TEXT,
             receipt_state TEXT
           )"""
    )
    from hermes_cli.sqlite_util import add_column_if_missing

    add_column_if_missing(conn, "executions", "fire_identity", "fire_identity TEXT")
    add_column_if_missing(conn, "executions", "error_kind", "error_kind TEXT")
    # NULL identifies executions written before the receipt contract.
    add_column_if_missing(conn, "executions", "receipt_state", "receipt_state TEXT")
    add_column_if_missing(
        conn, "executions", "handoff_pending",
        "handoff_pending INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, "executions", "handoff_started_at", "handoff_started_at REAL"
    )
    # Old execution rows may contain provider text, message fragments, paths,
    # or other sensitive diagnostics. Preserve only a bounded category.
    conn.execute(
        """UPDATE executions SET error_kind='legacy_redacted', error=NULL
           WHERE error IS NOT NULL"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    # A singleton metadata row avoids the old MAX(version) ambiguity. Migrate
    # the prior multi-row table transactionally; its only data is the version.
    receipt_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(receipt_schema)").fetchall()
    }
    if receipt_columns and "singleton" not in receipt_columns:
        conn.execute("DROP TABLE receipt_schema")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS receipt_schema (
             singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
             version INTEGER NOT NULL CHECK(version > 0)
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_targets (
             id INTEGER PRIMARY KEY,
             execution_id TEXT NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
             platform TEXT NOT NULL CHECK(length(platform) BETWEEN 1 AND 64),
             chat_id TEXT NOT NULL CHECK(length(chat_id) BETWEEN 1 AND 512),
             thread_id TEXT NOT NULL DEFAULT '' CHECK(length(thread_id) <= 512),
             UNIQUE(execution_id, platform, chat_id, thread_id)
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_components (
             id INTEGER PRIMARY KEY,
             target_id INTEGER NOT NULL REFERENCES delivery_targets(id) ON DELETE CASCADE,
             component TEXT NOT NULL CHECK(length(component) BETWEEN 1 AND 64),
             ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
             content_hash TEXT NOT NULL CHECK(length(content_hash) = 64),
             UNIQUE(target_id, component, ordinal, content_hash)
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_attempts (
             id TEXT PRIMARY KEY,
             component_id INTEGER NOT NULL REFERENCES delivery_components(id) ON DELETE CASCADE,
             attempt_no INTEGER NOT NULL CHECK(attempt_no >= 1),
             outcome TEXT NOT NULL CHECK(outcome IN ('delivered','unknown','failed')),
             provider_message_id TEXT CHECK(provider_message_id IS NULL OR length(provider_message_id) BETWEEN 1 AND 1024),
             actual_platform TEXT CHECK(actual_platform IS NULL OR length(actual_platform) BETWEEN 1 AND 64),
             actual_chat_id TEXT CHECK(actual_chat_id IS NULL OR length(actual_chat_id) BETWEEN 1 AND 512),
             actual_thread_id TEXT CHECK(actual_thread_id IS NULL OR length(actual_thread_id) <= 512),
             failure_kind TEXT CHECK(failure_kind IS NULL OR length(failure_kind) BETWEEN 1 AND 64),
             observed_at TEXT,
             created_at TEXT NOT NULL,
             UNIQUE(component_id, attempt_no)
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_delivery_attempts_component "
        "ON delivery_attempts(component_id, attempt_no)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_delivery_targets_execution "
        "ON delivery_targets(execution_id, id)"
    )
    # Legacy executions have no provider-target evidence. Do not manufacture
    # fake targets/attempt ids: receipt_summary() projects them as one bounded
    # attempted_unconfirmed/unknown result at read time instead.
    conn.execute(
        """INSERT INTO receipt_schema(singleton, version) VALUES (1, ?)
           ON CONFLICT(singleton) DO UPDATE SET version=excluded.version""",
        (_RECEIPT_SCHEMA_VERSION,),
    )
    conn.commit()


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback on exit, always close.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commit or roll back
    the transaction; it does not close the connection. Relying on that alone
    leaks a connection (and its WAL/SHM file descriptors) on every call,
    since closing then depends on the garbage collector. Schema init runs
    inside the ``try`` too, so a PRAGMA/DDL failure after a successful
    ``connect()`` still closes the connection instead of leaking it.
    """
    with _lock:
        conn = _connect()
        try:
            _initialize_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()


def _record(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
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
    limit = max(0, int(MAX_TERMINAL_EXECUTIONS))
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
             ORDER BY finished_at DESC, claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (limit,),
    )


def _bounded_text(value: Any, *, field: str, limit: int, allow_empty: bool = False) -> str:
    if value is None and allow_empty:
        return ""
    if type(value) is not str:
        raise ValueError(f"{field} must be a string")
    text = unicodedata.normalize("NFC", value)
    if (
        (not allow_empty and not text)
        or len(text) > limit
        or text != value
        or any(char.isspace() or unicodedata.category(char).startswith("C") for char in text)
    ):
        raise ValueError(f"{field} must be {'non-empty and ' if not allow_empty else ''}at most {limit} characters")
    return text


def _component_hash(content: Any) -> str:
    """Hash canonical logical text without placing the content in SQLite."""
    if type(content) is not str:
        raise ValueError("content must be a string")
    canonical = unicodedata.normalize("NFC", content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def preregister_receipt_plan(
    execution_id: str,
    *,
    fire_identity: str,
    components: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Atomically create unknown receipt attempts before any external dispatch.

    ``components`` is intentionally a tiny internal plan shape.  It accepts
    logical content only long enough to calculate its SHA-256; no body, media
    path, raw provider response, exception, or credential reaches the DB.
    Callers must abort dispatch if this raises.
    """
    _bounded_text(execution_id, field="execution_id", limit=128)
    _bounded_text(fire_identity, field="fire_identity", limit=256)
    if type(components) is not list:
        raise ValueError("components must be a list")
    prepared = []
    for item in components:
        if type(item) is not dict:
            raise ValueError("every receipt component must be an object")
        target_value = item.get("target")
        if type(target_value) is not dict:
            raise ValueError("every receipt component requires a target")
        target = target_value
        ordinal = item.get("ordinal", 0)
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("component ordinal must be a non-negative integer")
        prepared.append(
            (
                _bounded_text(target.get("platform"), field="platform", limit=64),
                _bounded_text(target.get("chat_id"), field="chat_id", limit=512),
                _bounded_text(target.get("thread_id"), field="thread_id", limit=512, allow_empty=True),
                _bounded_text(item.get("component"), field="component", limit=64),
                ordinal,
                _component_hash(item.get("content", "")),
            )
        )
    if not prepared:
        return []


    created_at = _hermes_now().isoformat()
    created: List[Dict[str, Any]] = []
    with _transaction() as conn:
        execution = conn.execute(
            "SELECT job_id, fire_identity, receipt_state FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
        if execution is None:
            raise ValueError("unknown execution_id")
        existing_fire = execution["fire_identity"]
        if existing_fire is None:
            conn.execute(
                """UPDATE executions SET fire_identity=?, receipt_state='planned'
                   WHERE id=? AND fire_identity IS NULL""",
                (fire_identity, execution_id),
            )
        elif existing_fire != fire_identity:
            if existing_fire == execution_id and execution["receipt_state"] == "not_planned":
                conn.execute(
                    "UPDATE executions SET fire_identity=? WHERE id=?",
                    (fire_identity, execution_id),
                )
            else:
                raise ValueError("conflicting receipt plan fire_identity")
        existing_rows = conn.execute(
            """SELECT a.id, a.outcome, c.component, c.ordinal, c.content_hash,
                      t.platform, t.chat_id, t.thread_id
               FROM delivery_attempts a
               JOIN delivery_components c ON c.id=a.component_id
               JOIN delivery_targets t ON t.id=c.target_id
               WHERE t.execution_id=? AND a.attempt_no=1
               ORDER BY t.id, c.component, c.ordinal""",
            (execution_id,),
        ).fetchall()
        if existing_rows:
            existing_plan = {
                (row["platform"], row["chat_id"], row["thread_id"], row["component"],
                 row["ordinal"], row["content_hash"])
                for row in existing_rows
            }
            if existing_plan != set(prepared) or len(existing_rows) != len(prepared):
                raise ValueError("conflicting receipt plan")
            return [
                {"id": row["id"], "execution_id": execution_id,
                 "platform": row["platform"], "chat_id": row["chat_id"],
                 "thread_id": row["thread_id"], "component": row["component"],
                 "ordinal": row["ordinal"], "outcome": row["outcome"]}
                for row in existing_rows
            ]
        # A fire identity is single-dispatch. Once any component was
        # preregistered, an unknown outcome cannot authorize replay: the
        # provider write may have completed before acknowledgement was lost.
        # A different fire identity remains independent, even for identical
        # content.
        cutoff = (_hermes_now() - timedelta(hours=24)).isoformat()
        prior_rows = conn.execute(
            """SELECT t.platform, t.chat_id, t.thread_id,
                      c.component, c.ordinal, c.content_hash
               FROM delivery_attempts a
               JOIN delivery_components c ON c.id=a.component_id
               JOIN delivery_targets t ON t.id=c.target_id
               JOIN executions e ON e.id=t.execution_id
               WHERE e.id<>? AND e.job_id=? AND e.fire_identity=?
                 AND e.claimed_at>=?
               ORDER BY t.platform, t.chat_id, t.thread_id, c.component, c.ordinal""",
            (execution_id, execution["job_id"], fire_identity, cutoff),
        ).fetchall()
        if prior_rows:
            prior_plan = {
                (
                    row["platform"], row["chat_id"], row["thread_id"],
                    row["component"], row["ordinal"], row["content_hash"],
                )
                for row in prior_rows
            }
            if prior_plan != set(prepared) or len(prior_rows) != len(prepared):
                raise ValueError("conflicting receipt plan for existing fire_identity")
            raise ValueError("delivery receipt already attempted for this fire identity")
        for platform, chat_id, thread_id, component, ordinal, content_hash in prepared:
            conn.execute(
                """INSERT OR IGNORE INTO delivery_targets
                   (execution_id, platform, chat_id, thread_id) VALUES (?, ?, ?, ?)""",
                (execution_id, platform, chat_id, thread_id),
            )
            target_row = conn.execute(
                """SELECT id FROM delivery_targets WHERE execution_id=? AND platform=?
                   AND chat_id=? AND thread_id=?""",
                (execution_id, platform, chat_id, thread_id),
            ).fetchone()
            assert target_row is not None
            component_row = conn.execute(
                """INSERT INTO delivery_components(target_id, component, ordinal, content_hash)
                   VALUES (?, ?, ?, ?) RETURNING id""",
                (target_row["id"], component, ordinal, content_hash),
            ).fetchone()
            attempt_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO delivery_attempts
                   (id, component_id, attempt_no, outcome, created_at)
                   VALUES (?, ?, 1, 'unknown', ?)""",
                (attempt_id, component_row["id"], created_at),
            )
            created.append({
                "id": attempt_id,
                "execution_id": execution_id,
                "platform": platform,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "component": component,
                "ordinal": ordinal,
                "outcome": "unknown",
            })
    return created


def receipt_summary(execution_id: str) -> Dict[str, int]:
    """Return bounded receipt counts; never expose component content or errors."""
    with _transaction() as conn:
        counts = {"delivered": 0, "failed": 0, "unknown": 0, "targets_delivered": 0}
        for row in conn.execute(
            """SELECT outcome, COUNT(*) AS count FROM delivery_attempts a
               JOIN delivery_components c ON c.id=a.component_id
               JOIN delivery_targets t ON t.id=c.target_id
               WHERE t.execution_id=? GROUP BY outcome""",
            (execution_id,),
        ):
            counts[row["outcome"]] = int(row["count"])
        target_row = conn.execute(
            """SELECT e.receipt_state, COUNT(t.id) AS count
               FROM executions e LEFT JOIN delivery_targets t ON t.execution_id=e.id
               WHERE e.id=? GROUP BY e.id""",
            (execution_id,),
        ).fetchone()
        if (
            target_row is not None
            and int(target_row["count"]) == 0
            and target_row["receipt_state"] is None
        ):
            # A pre-receipt/legacy execution has an attempted side effect but
            # no provider proof. Surface that honestly without creating rows
            # during a read operation.
            counts["unknown"] = 1
        target_row = conn.execute(
            """SELECT COUNT(*) AS count FROM (
                 SELECT t.id FROM delivery_targets t
                 JOIN delivery_components c ON c.target_id=t.id
                 JOIN delivery_attempts a ON a.component_id=c.id
                 WHERE t.execution_id=?
                 GROUP BY t.id
                 HAVING COUNT(DISTINCT c.id) = COUNT(DISTINCT CASE
                   WHEN a.outcome='delivered'
                    AND a.actual_platform=t.platform
                    AND a.actual_chat_id=t.chat_id
                    AND COALESCE(a.actual_thread_id, '')=t.thread_id
                   THEN c.id END)
               )""",
            (execution_id,),
        ).fetchone()
        counts["targets_delivered"] = int(target_row["count"] if target_row else 0)
    return counts


def record_transport_receipt(attempt_id: str, receipt: Any) -> bool:
    """Commit a typed provider acknowledgement once, preserving unknown on error.

    This function intentionally has no retry behavior.  If the caller receives
    an acknowledgement but this transaction fails, the attempt remains unknown
    and the caller must not infer that a resend is safe.
    """
    from gateway.platforms.base import TransportReceipt, TransportTarget

    if type(receipt) is not TransportReceipt:
        raise ValueError("receipt must be a TransportReceipt")
    if type(receipt.ordinal) is not int or receipt.ordinal < 0:
        raise ValueError("receipt ordinal must be a non-negative integer")
    requested = receipt.requested_target
    actual = receipt.actual_target
    if type(requested) is not TransportTarget:
        raise TypeError("requested_target must be a TransportTarget")
    if actual is not None and type(actual) is not TransportTarget:
        raise TypeError("actual_target must be a TransportTarget")

    # ``frozen=True`` emulates immutability but can still be bypassed with
    # ``object.__setattr__``. Reconstruct the complete typed contract at this
    # persistence boundary so every target and receipt invariant is re-run
    # immediately before durable state can change. Keep the explicit outcome
    # checks above first so their established error categories remain stable.
    requested = TransportTarget(
        platform=requested.platform,
        chat_id=requested.chat_id,
        thread_id=requested.thread_id,
    )
    if actual is not None:
        actual = TransportTarget(
            platform=actual.platform,
            chat_id=actual.chat_id,
            thread_id=actual.thread_id,
        )
    receipt = TransportReceipt(
        outcome=receipt.outcome,
        requested_target=requested,
        actual_target=actual,
        provider_message_id=receipt.provider_message_id,
        observed_at=receipt.observed_at,
        failure_kind=receipt.failure_kind,
        component=receipt.component,
        ordinal=receipt.ordinal,
    )
    outcome = receipt.outcome
    if outcome not in {"delivered", "failed"}:
        return False
    provider_message_id = receipt.provider_message_id
    failure_kind = receipt.failure_kind
    actual_platform = getattr(actual, "platform", None) if actual is not None else None
    actual_chat_id = getattr(actual, "chat_id", None) if actual is not None else None
    actual_thread_id = getattr(actual, "thread_id", None) if actual is not None else None
    if provider_message_id is not None:
        provider_message_id = _bounded_text(provider_message_id, field="provider_message_id", limit=1024)
    if actual_platform is not None:
        actual_platform = _bounded_text(actual_platform, field="actual_platform", limit=64)
        actual_chat_id = _bounded_text(actual_chat_id, field="actual_chat_id", limit=512)
        actual_thread_id = _bounded_text(actual_thread_id, field="actual_thread_id", limit=512, allow_empty=True) or None
    if failure_kind is not None:
        failure_kind = _bounded_text(failure_kind, field="failure_kind", limit=64)
    observed_at = receipt.observed_at
    observed_text = observed_at.isoformat()
    normalized_attempt_id = _bounded_text(
        attempt_id, field="attempt_id", limit=128,
    )
    with _transaction() as conn:
        binding = conn.execute(
            """SELECT c.component, c.ordinal, t.platform, t.chat_id, t.thread_id
               FROM delivery_attempts a
               JOIN delivery_components c ON c.id=a.component_id
               JOIN delivery_targets t ON t.id=c.target_id WHERE a.id=?""",
            (normalized_attempt_id,),
        ).fetchone()
        if binding is None:
            return False
        if (
            requested.platform != binding["platform"]
            or requested.chat_id != binding["chat_id"]
            or (requested.thread_id or "") != binding["thread_id"]
            or receipt.component != binding["component"]
            or receipt.ordinal != binding["ordinal"]
        ):
            raise ValueError("receipt requested_target/component does not match preregistered attempt")
        cur = conn.execute(
            """UPDATE delivery_attempts SET outcome=?, provider_message_id=?,
                   actual_platform=?, actual_chat_id=?, actual_thread_id=?,
                   failure_kind=?, observed_at=?
               WHERE id=? AND outcome='unknown'""",
            (
                outcome,
                provider_message_id,
                actual_platform,
                actual_chat_id,
                actual_thread_id,
                failure_kind,
                observed_text,
                normalized_attempt_id,
            ),
        )
    return cur.rowcount == 1


def observe_transport_unknown(attempt_id: str, receipt: Any) -> bool:
    """Record a typed ambiguous observation without upgrading its outcome."""
    from gateway.platforms.base import TransportReceipt, TransportTarget

    if type(receipt) is not TransportReceipt:
        raise ValueError("receipt must be a typed unknown TransportReceipt")
    requested = receipt.requested_target
    actual = receipt.actual_target
    if type(requested) is not TransportTarget:
        raise TypeError("requested_target must be a TransportTarget")
    if actual is not None and type(actual) is not TransportTarget:
        raise TypeError("actual_target must be a TransportTarget")
    # Reconstruct at the persistence boundary so frozen-dataclass bypasses and
    # malformed target fields cannot reach SQLite.
    requested = TransportTarget(
        platform=requested.platform,
        chat_id=requested.chat_id,
        thread_id=requested.thread_id,
    )
    receipt = TransportReceipt(
        outcome=receipt.outcome,
        requested_target=requested,
        actual_target=actual,
        provider_message_id=receipt.provider_message_id,
        observed_at=receipt.observed_at,
        failure_kind=receipt.failure_kind,
        component=receipt.component,
        ordinal=receipt.ordinal,
    )
    if receipt.outcome != "unknown":
        raise ValueError("receipt must be a typed unknown TransportReceipt")
    observed_text = receipt.observed_at.isoformat()
    normalized_attempt_id = _bounded_text(
        attempt_id, field="attempt_id", limit=128,
    )
    with _transaction() as conn:
        binding = conn.execute(
            """SELECT c.component, c.ordinal, t.platform, t.chat_id, t.thread_id
               FROM delivery_attempts a
               JOIN delivery_components c ON c.id=a.component_id
               JOIN delivery_targets t ON t.id=c.target_id WHERE a.id=?""",
            (normalized_attempt_id,),
        ).fetchone()
        if binding is None:
            return False
        if (
            requested.platform != binding["platform"]
            or requested.chat_id != binding["chat_id"]
            or (requested.thread_id or "") != binding["thread_id"]
            or receipt.component != binding["component"]
            or receipt.ordinal != binding["ordinal"]
        ):
            raise ValueError(
                "receipt requested_target/component does not match preregistered attempt"
            )
        cur = conn.execute(
            """UPDATE delivery_attempts SET observed_at=?
               WHERE id=? AND outcome='unknown' AND observed_at IS NULL""",
            (observed_text, normalized_attempt_id),
        )
    return cur.rowcount == 1


def create_execution(
    job_id: str, *, source: str, fire_identity: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a claimed attempt and its fire identity before dispatch."""
    normalized_job_id = _bounded_text(job_id, field="job_id", limit=256)
    normalized_source = _bounded_text(source, field="source", limit=32)
    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    normalized_fire_identity = _bounded_text(
        execution_id if fire_identity is None else fire_identity,
        field="fire_identity",
        limit=256,
    )
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, receipt_state, fire_identity)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, 'not_planned', ?)""",
            (execution_id, normalized_job_id, normalized_source, _PROCESS_ID, pid,
             _process_start_time(pid), now, normalized_fire_identity),
        )
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone()
    record = _record(row)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def bind_execution_fire_identity(
    execution_id: str, fire_identity: str,
) -> Dict[str, Any]:
    """Bind a claimed execution to its acquired fire exactly once before planning."""
    normalized_execution_id = _bounded_text(
        execution_id, field="execution_id", limit=128,
    )
    normalized_fire_identity = _bounded_text(
        fire_identity, field="fire_identity", limit=256,
    )
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?", (normalized_execution_id,),
        ).fetchone()
        if row is None:
            raise ValueError("unknown execution_id")
        if row["fire_identity"] == normalized_fire_identity:
            return _record(row)  # type: ignore[return-value]
        if (
            row["status"] != "claimed"
            or row["receipt_state"] != "not_planned"
            or row["fire_identity"] != normalized_execution_id
        ):
            raise ValueError("execution fire_identity is already bound")
        conn.execute(
            "UPDATE executions SET fire_identity=? WHERE id=?",
            (normalized_fire_identity, normalized_execution_id),
        )
        bound = conn.execute(
            "SELECT * FROM executions WHERE id=?", (normalized_execution_id,),
        ).fetchone()
    return _record(bound)  # type: ignore[return-value]


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
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
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
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
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
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    error_kind: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten."""
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    del error  # Free-form diagnostics must never enter the durable ledger.
    category = None if success else (error_kind or "execution_failed")
    if category is not None and category not in _EXECUTION_ERROR_KINDS:
        raise ValueError("execution error_kind is invalid")
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, finished_at=?, error=NULL, error_kind=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status IN ('claimed','running')
                 AND process_id=? AND pid=?""",
            (status, now, category, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _record(conn.execute(
            "SELECT * FROM executions WHERE id=?", (execution_id,)
        ).fetchone())
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    return record


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
                   SET status='unknown', finished_at=?, error=NULL,
                       error_kind='interrupted',
                       handoff_pending=0, handoff_started_at=NULL
                   WHERE id=? AND status=? AND process_id=? AND pid=?
                     AND handoff_pending=?
                     AND handoff_started_at IS ?""",
                (now, row["id"], row["status"], row["process_id"], row["pid"],
                 row["handoff_pending"], row["handoff_started_at"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                record = _record(conn.execute(
                    "SELECT * FROM executions WHERE id=?", (row["id"],)
                ).fetchone())
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50,
    before_claimed_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: List[str] = []
    params: List[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(_bounded_text(job_id, field="job_id", limit=256))
    if before_claimed_at is not None:
        clauses.append("claimed_at < ?")
        params.append(_bounded_text(
            before_claimed_at, field="before_claimed_at", limit=128,
        ))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    if type(limit) is not int:
        raise ValueError("limit must be an integer")
    params.append(max(1, min(limit, 500)))
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
    if type(job_ids) is not list:
        raise ValueError("job_ids must be a list")
    clean = []
    seen = set()
    for job_id in job_ids:
        normalized = _bounded_text(job_id, field="job_id", limit=256)
        if normalized not in seen:
            seen.add(normalized)
            clean.append(normalized)
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
