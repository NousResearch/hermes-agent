"""Transactional durable delivery receipts for same-board task projections.

This module stores transport metadata alongside the authoritative Kanban board.  A
receipt never changes ``tasks`` or ``task_events``; its source revision is an
existing ``task_events.id`` and its incarnation is the task's immutable
``created`` event id.  The public mutators join an existing caller transaction
through a savepoint and commit only when called standalone.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

from hermes_cli import kanban_db as _kb


DELIVERY_STATES = frozenset({"pending", "sent", "unknown", "failed", "deleted"})
_RETRYABLE_DISPOSITIONS = frozenset({"retry", "retryable", "safe_retry"})
_RETRY_DISPOSITIONS = frozenset({
    "delivered", "retry", "retryable", "safe_retry", "no_retry",
    "exhausted", "reconcile_required", "replacement_authorized",
    "replacement_available",
})


def migrate_delivery_receipts(conn):
    """Migrate transport metadata, never guess legacy attempted revisions."""
    columns = [r[1] for r in conn.execute("PRAGMA table_info(kanban_delivery_receipts)")]
    if columns and "attempted_revision" not in columns:
        with _write_scope(conn):
            ddl = _kb.SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS kanban_delivery_receipts (", 1)[1].split(";", 1)[0]
            conn.execute("CREATE TABLE kanban_delivery_receipts_v2 (" + ddl)
            names = ", ".join(columns)
            conn.execute(f"INSERT INTO kanban_delivery_receipts_v2 ({names}) SELECT {names} FROM kanban_delivery_receipts")
            conn.execute("UPDATE kanban_delivery_receipts_v2 SET profile_key = COALESCE(notifier_profile, '')")
            conn.execute("UPDATE kanban_delivery_receipts_v2 SET state='unknown', owner_id=NULL, "
                         "lease_expires_at=NULL, retry_disposition='reconcile_required' "
                         "WHERE state='pending' AND attempt_id IS NOT NULL")
            conn.execute("DROP TABLE kanban_delivery_receipts")
            conn.execute("ALTER TABLE kanban_delivery_receipts_v2 RENAME TO kanban_delivery_receipts")
            conn.execute("CREATE INDEX idx_receipts_lane ON kanban_delivery_receipts(task_id, task_incarnation, platform, chat_id, thread_id)")
            conn.execute("CREATE INDEX idx_receipts_state ON kanban_delivery_receipts(state, lease_expires_at)")
    from hermes_cli.sqlite_util import add_column_if_missing
    if columns:
        add_column_if_missing(conn, "kanban_delivery_receipts", "control_hash", "control_hash TEXT")
    if conn.execute("PRAGMA table_info(kanban_notify_subs)").fetchone() is not None:
        add_column_if_missing(conn, "kanban_notify_subs", "binding_token", "binding_token TEXT")
        conn.execute("UPDATE kanban_notify_subs SET binding_token=lower(hex(randomblob(16))) WHERE binding_token IS NULL")


def reconcile_delivery_outcome(conn, receipt_id, **evidence):
    """Host-only exact-attempt settlement, not writer authority or a plugin API.

    The host transport is the only production caller. Fencing/expiry does not
    destroy a returned message ID. A successor attempt rejects stale evidence.
    """
    return record_delivery_outcome(conn, receipt_id, _reconcile=True, **evidence)


def quarantine_delivery(conn, lease):
    with _write_scope(conn):
        conn.execute("UPDATE kanban_delivery_receipts SET state='unknown', owner_id=NULL, "
                     "lease_expires_at=NULL, retry_disposition='reconcile_required', "
                     "failure_count=failure_count+1, renderer_hash=NULL, control_hash=NULL "
                     "WHERE id=? AND state='pending' AND attempt_id=? AND owner_epoch=?",
                     (lease.receipt.id, lease.attempt_id, lease.owner_epoch))


class DeliveryReceiptError(RuntimeError):
    """Base class for fail-closed receipt errors."""


class DeliveryReceiptNotFound(DeliveryReceiptError):
    pass


class DeliveryReceiptIdentityError(DeliveryReceiptError):
    pass


class DeliveryReceiptRevisionError(DeliveryReceiptError):
    pass


class DeliveryReceiptLeaseLost(DeliveryReceiptError):
    pass


class DeliveryReceiptUnknown(DeliveryReceiptError):
    pass


class DeliveryReceiptNotDue(DeliveryReceiptError):
    pass


class DeliveryReceiptNotRetryable(DeliveryReceiptError):
    pass


@dataclass(frozen=True)
class TaskSource:
    """Canonical task identity and event revision used by projections."""

    task_id: str
    task_incarnation: int
    current_revision: int
    status: str


@dataclass(frozen=True)
class DeliveryReceipt:
    id: int
    task_id: str
    task_incarnation: int
    desired_revision: int
    delivered_revision: Optional[int]
    platform: str
    chat_id: str
    thread_id: str
    notifier_profile: Optional[str]
    routing_metadata: dict[str, Any]
    destination_message_id: Optional[str]
    destination_profile: Optional[str]
    renderer_version: Optional[str]
    renderer_hash: Optional[str]
    control_hash: Optional[str]
    owner_epoch: int
    owner_id: Optional[str]
    lease_expires_at: Optional[int]
    attempt_id: Optional[str]
    attempt_count: int
    state: str
    retry_disposition: Optional[str]
    last_error: Optional[str]
    replacement_budget: int
    created_at: int
    updated_at: int
    attempted_revision: Optional[int]
    binding_token: Optional[str]
    retry_at: float
    failure_count: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DeliveryReceipt":
        return cls(
            id=int(row["id"]),
            task_id=str(row["task_id"]),
            task_incarnation=int(row["task_incarnation"]),
            desired_revision=int(row["desired_revision"]),
            delivered_revision=(
                int(row["delivered_revision"])
                if row["delivered_revision"] is not None
                else None
            ),
            platform=str(row["platform"]),
            chat_id=str(row["chat_id"]),
            thread_id=str(row["thread_id"] or ""),
            notifier_profile=row["notifier_profile"],
            routing_metadata=_decode_json_object(row["routing_metadata"]),
            destination_message_id=(
                str(row["destination_message_id"])
                if row["destination_message_id"] is not None
                else None
            ),
            destination_profile=row["destination_profile"],
            renderer_version=row["renderer_version"],
            renderer_hash=row["renderer_hash"],
            control_hash=row["control_hash"],
            owner_epoch=int(row["owner_epoch"]),
            owner_id=row["owner_id"],
            lease_expires_at=(
                int(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
            attempt_id=row["attempt_id"],
            attempt_count=int(row["attempt_count"]),
            state=str(row["state"]),
            retry_disposition=row["retry_disposition"],
            last_error=row["last_error"],
            replacement_budget=int(row["replacement_budget"]),
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
            attempted_revision=row["attempted_revision"],
            binding_token=row["binding_token"],
            retry_at=float(row["retry_at"]),
            failure_count=int(row["failure_count"]),
        )


@dataclass(frozen=True)
class DeliveryLease:
    receipt: DeliveryReceipt
    owner_id: str
    owner_epoch: int
    attempt_id: str
    desired_revision: int


def _decode_json_object(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _json_object(value: Optional[Mapping[str, Any]], *, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise ValueError(f"{field} is too large")
    return encoded


def _text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if value is None:
        raise ValueError(f"{field} is required")
    result = str(value) if not isinstance(value, str) else value
    if not allow_empty and not result:
        raise ValueError(f"{field} is required")
    return result


def _now(value: Optional[int]) -> int:
    return int(time.time()) if value is None else int(value)


@contextlib.contextmanager
def _write_scope(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit standalone calls, but never commit a caller-owned transaction."""
    with _kb.write_txn(conn, allow_nested=bool(conn.in_transaction)):
        yield conn


def _source_row(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        raise DeliveryReceiptIdentityError(f"unknown task source: {task_id}")
    return row


def get_task_source(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_revision: Optional[int] = None,
    expected_status: Optional[str] = None,
    task_incarnation: Optional[int] = None,
) -> TaskSource:
    """Read the current canonical task incarnation and event revision.

    A task without its creation event is intentionally unusable by this
    foundation.  ``created_at`` is not treated as an incarnation because it is
    not unique and can be reused by imported/legacy rows.
    """
    task_id = _text(task_id, field="task_id")
    row = _source_row(conn, task_id)
    created = conn.execute(
        "SELECT id FROM task_events WHERE task_id = ? AND kind = 'created' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if created is None:
        raise DeliveryReceiptIdentityError(
            f"task {task_id!r} has no immutable created event"
        )
    incarnation = int(created["id"])
    if task_incarnation is not None and int(task_incarnation) != incarnation:
        raise DeliveryReceiptIdentityError("task incarnation does not match the live task")
    current = conn.execute(
        "SELECT MAX(id) AS revision FROM task_events WHERE task_id = ?", (task_id,)
    ).fetchone()
    current_revision = int(current["revision"] or 0)
    if current_revision <= 0:
        raise DeliveryReceiptIdentityError("task has no event revision")
    if expected_revision is not None:
        expected_revision = int(expected_revision)
        if expected_revision <= 0:
            raise DeliveryReceiptRevisionError("task event revision must be positive")
        event = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND id = ?",
            (task_id, expected_revision),
        ).fetchone()
        if event is None or expected_revision < incarnation:
            raise DeliveryReceiptRevisionError(
                "expected revision is not an event for this task"
            )
    if expected_status is not None and str(row["status"]) != str(expected_status):
        raise DeliveryReceiptRevisionError("task status no longer matches the source")
    return TaskSource(
        task_id=task_id,
        task_incarnation=incarnation,
        current_revision=current_revision,
        status=str(row["status"]),
    )


def _receipt_row(conn: sqlite3.Connection, receipt_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM kanban_delivery_receipts WHERE id = ?", (int(receipt_id),)
    ).fetchone()
    if row is None:
        raise DeliveryReceiptNotFound(f"delivery receipt {receipt_id} does not exist")
    return row


def get_delivery_receipt(
    conn: sqlite3.Connection, receipt_id: int
) -> Optional[DeliveryReceipt]:
    row = conn.execute(
        "SELECT * FROM kanban_delivery_receipts WHERE id = ?", (int(receipt_id),)
    ).fetchone()
    return DeliveryReceipt.from_row(row) if row is not None else None


def list_delivery_receipts(
    conn: sqlite3.Connection,
    *,
    task_id: Optional[str] = None,
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> list[DeliveryReceipt]:
    clauses = []
    params: list[Any] = []
    for column, value in (
        ("task_id", task_id),
        ("platform", platform),
        ("chat_id", chat_id),
        ("thread_id", None if thread_id is None else thread_id or ""),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    sql = "SELECT * FROM kanban_delivery_receipts"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id ASC"
    return [DeliveryReceipt.from_row(row) for row in conn.execute(sql, params)]


def ensure_delivery_receipt(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    notifier_profile: Optional[str] = None,
    routing_metadata: Optional[Mapping[str, Any]] = None,
    renderer_version: Optional[str] = None,
    renderer_hash: Optional[str] = None,
    desired_revision: int,
    task_incarnation: Optional[int] = None,
    replacement_budget: int = 1,
    surface_kind: str = "task_card",
    now: Optional[int] = None,
) -> DeliveryReceipt:
    """Durably create/update one task-to-lane receipt intent.

    The call is idempotent for the same task incarnation and exact lane.  A
    lower desired revision is rejected rather than silently regressing a card.
    The receipt is committed before a caller performs remote I/O when invoked
    standalone; inside a caller transaction it remains part of that transaction.
    """
    platform = _text(platform, field="platform")
    chat_id = _text(chat_id, field="chat_id")
    thread_id = "" if thread_id is None else str(thread_id)
    desired_revision = int(desired_revision)
    if desired_revision <= 0:
        raise DeliveryReceiptRevisionError("desired_revision must be positive")
    if int(replacement_budget) < 0:
        raise ValueError("replacement_budget cannot be negative")
    metadata_json = _json_object(routing_metadata, field="routing_metadata")
    stamp = _now(now)
    with _write_scope(conn):
        source = get_task_source(
            conn, task_id, expected_revision=desired_revision,
            task_incarnation=task_incarnation,
        )
        lane = conn.execute(
            "SELECT * FROM kanban_delivery_receipts WHERE task_id = ? "
            "AND task_incarnation = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND profile_key = ? AND surface_kind = ?",
            (task_id, source.task_incarnation, platform, chat_id, thread_id, notifier_profile or "", surface_kind),
        ).fetchone()
        if lane is None:
            conn.execute(
                """
                INSERT INTO kanban_delivery_receipts (
                    task_id, task_incarnation, desired_revision, platform, chat_id,
                    thread_id, notifier_profile, routing_metadata, renderer_version,
                    renderer_hash, replacement_budget, created_at, updated_at, profile_key, surface_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, source.task_incarnation, desired_revision, platform,
                    chat_id, thread_id, notifier_profile, metadata_json,
                    renderer_version, renderer_hash, int(replacement_budget), stamp, stamp,
                    notifier_profile or "", surface_kind,
                ),
            )
            receipt_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        else:
            receipt_id = int(lane["id"])
            if notifier_profile and lane["notifier_profile"] not in (None, "", notifier_profile):
                raise DeliveryReceiptIdentityError("notifier profile does not match receipt lane")
            if desired_revision < int(lane["desired_revision"]):
                raise DeliveryReceiptRevisionError("desired revision regressed")
            conn.execute(
                """
                UPDATE kanban_delivery_receipts
                   SET desired_revision = MAX(desired_revision, ?),
                       notifier_profile = COALESCE(notifier_profile, ?),
                       routing_metadata = COALESCE(?, routing_metadata),
                       renderer_version = COALESCE(?, renderer_version),
                       renderer_hash = COALESCE(?, renderer_hash),

                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    desired_revision, notifier_profile, metadata_json,
                    renderer_version, renderer_hash, stamp,
                    receipt_id,
                ),
            )
    receipt = get_delivery_receipt(conn, receipt_id)
    assert receipt is not None
    return receipt


def claim_delivery_receipt(
    conn: sqlite3.Connection,
    receipt_id: int,
    *,
    owner_id: str,
    lease_seconds: int = 120,
    now: Optional[int] = None,
    expected_revision: Optional[int] = None,
    refresh: bool = False,
) -> DeliveryLease:
    """Claim one due receipt with an owner epoch and attempt identity.

    An expired ``pending`` owner is converted to ``unknown``.  It is never
    silently claimed by a new worker because the remote create may have
    succeeded before the process crashed.
    """
    owner_id = _text(owner_id, field="owner_id")
    lease_seconds = int(lease_seconds)
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    stamp = _now(now)
    attempt_id = secrets.token_urlsafe(12)
    expired_unknown = False
    new_epoch = 0
    receipt = None
    with _write_scope(conn):
        row = _receipt_row(conn, receipt_id)
        state = str(row["state"])
        expires = row["lease_expires_at"]
        get_task_source(conn, row["task_id"], task_incarnation=row["task_incarnation"])
        if expected_revision is not None and row["desired_revision"] != expected_revision:
            raise DeliveryReceiptNotDue("newer demand superseded this snapshot")
        if float(row["retry_at"]) > stamp:
            raise DeliveryReceiptNotDue("receipt retry_after has not elapsed")
        if (state == "failed" and row["failure_count"] >= 1
                and row["retry_disposition"] == "exhausted"
                and row["last_error"] == "known message unchanged"
                and row["destination_message_id"] is not None
                and row["attempted_revision"] is not None
                and int(row["desired_revision"]) > int(row["attempted_revision"])):
            # A deterministic known-message rejection cannot have created a
            # duplicate. A later rendered revision gets one fresh bounded
            # attempt budget; the same revision can never reset itself.
            conn.execute(
                "UPDATE kanban_delivery_receipts SET failure_count=0, "
                "retry_disposition='safe_retry' WHERE id=?",
                (receipt_id,),
            )
            row = _receipt_row(conn, receipt_id)
            state = "failed"
        if state in {"unknown", "failed"} and row["failure_count"] >= 3:
            raise DeliveryReceiptNotRetryable("delivery recovery budget exhausted")
        if state == "unknown" and row["destination_message_id"] is not None:
            # Repeating an edit of a known ID cannot create a second card. The
            # old attempt is fenced by the next claim epoch. Unknown creates,
            # including replacements, have no ID and remain quarantined.
            conn.execute("UPDATE kanban_delivery_receipts SET state='failed', "
                         "owner_id=NULL, lease_expires_at=NULL, retry_disposition='safe_retry', "
                         "renderer_hash=NULL, control_hash=NULL WHERE id=?",
                         (receipt_id,))
            row = _receipt_row(conn, receipt_id)
            state = "failed"
        if state == "pending" and row["owner_id"]:
            if expires is not None and int(expires) <= stamp:
                conn.execute(
                    """
                    UPDATE kanban_delivery_receipts
                       SET state = 'unknown', owner_id = NULL,
                           lease_expires_at = NULL,
                           failure_count = failure_count + 1,
                           retry_disposition = 'reconcile_required',
                           last_error = 'delivery owner lease expired; remote outcome is unknown',
                           renderer_hash = NULL, control_hash = NULL,
                           updated_at = ?
                     WHERE id = ? AND state = 'pending' AND owner_epoch = ?
                    """,
                    (stamp, int(receipt_id), int(row["owner_epoch"])),
                )
                expired_unknown = True
            if expired_unknown:
                pass
            elif row["owner_id"]:
                raise DeliveryReceiptLeaseLost("delivery receipt is owned by a live lease")
        if not expired_unknown:
            if state == "unknown":
                raise DeliveryReceiptUnknown("delivery outcome is unknown; reconcile before retry")
            if state == "deleted":
                raise DeliveryReceiptNotRetryable("deleted delivery needs explicit replacement authorization")
            delivered = row["delivered_revision"]
            if state == "sent" and not refresh and delivered is not None and int(delivered) >= int(row["desired_revision"]):
                raise DeliveryReceiptNotDue("receipt already delivered at the desired revision")
            if state == "failed" and row["retry_disposition"] not in _RETRYABLE_DISPOSITIONS:
                raise DeliveryReceiptNotRetryable("receipt failure is not classified retryable")
            if state not in {"pending", "failed", "sent"}:
                raise DeliveryReceiptNotRetryable(f"receipt state {state!r} cannot be claimed")
            old_epoch = int(row["owner_epoch"])
            new_epoch = old_epoch + 1
            lease_until = stamp + lease_seconds
            cur = conn.execute(
                """
                UPDATE kanban_delivery_receipts
                   SET owner_epoch = ?, owner_id = ?, lease_expires_at = ?,
                       attempt_id = ?, attempt_count = attempt_count + 1,
                       attempted_revision = desired_revision, attempt_owner_id = ?,
                       state = 'pending', retry_disposition = NULL,
                       last_error = NULL, updated_at = ?
                 WHERE id = ? AND owner_epoch = ? AND state IN ('pending', 'failed', 'sent')
                   AND (owner_id IS NULL OR owner_id = '')
                   AND (state != 'failed' OR retry_disposition IN ('retry', 'retryable', 'safe_retry'))
                   AND (state != 'sent' OR delivered_revision IS NULL OR delivered_revision < desired_revision OR ?)
                """,
                (
                    new_epoch, owner_id, lease_until, attempt_id, owner_id, stamp,
                    int(receipt_id), old_epoch, bool(refresh),
                ),
            )
            if cur.rowcount != 1:
                raise DeliveryReceiptLeaseLost("receipt claim lost a compare-and-swap race")
            # Freeze the claimed authority before COMMIT releases the competing writer.
            receipt = get_delivery_receipt(conn, receipt_id)
            assert receipt is not None
    if expired_unknown:
        raise DeliveryReceiptUnknown("expired delivery lease requires reconciliation")
    assert receipt is not None and receipt.attempted_revision is not None
    return DeliveryLease(
        receipt=receipt, owner_id=owner_id, owner_epoch=new_epoch,
        attempt_id=attempt_id, desired_revision=receipt.attempted_revision,
    )


def confirm_equivalent_delivery(
    conn: sqlite3.Connection,
    receipt_id: int,
    *,
    desired_revision: int,
    renderer_hash: str,
    message_id: str,
    now: Optional[int] = None,
) -> DeliveryReceipt:
    """Advance a known message when its full rendered payload is unchanged.

    This is local equivalence to an earlier confirmed send, not settlement of
    the newer transport attempt.  Unknown, pending, deleted and never-sent
    receipts remain fenced.
    """
    desired_revision = int(desired_revision)
    renderer_hash = _text(renderer_hash, field="renderer_hash")
    message_id = _text(message_id, field="message_id")
    stamp = _now(now)
    with _write_scope(conn):
        row = _receipt_row(conn, receipt_id)
        get_task_source(
            conn, row["task_id"], expected_revision=desired_revision,
            task_incarnation=row["task_incarnation"],
        )
        if (int(row["desired_revision"]) != desired_revision
                or row["state"] != "sent"
                or row["owner_id"] not in {None, ""}
                or row["destination_message_id"] != message_id
                or row["delivered_revision"] is None
                or row["renderer_hash"] != renderer_hash):
            raise DeliveryReceiptNotDue("receipt has no equivalent confirmed payload")
        conn.execute(
            "UPDATE kanban_delivery_receipts SET state='sent', delivered_revision=?, "
            "retry_disposition='delivered', last_error=NULL, failure_count=0, updated_at=? "
            "WHERE id=?",
            (desired_revision, stamp, int(receipt_id)),
        )
    receipt = get_delivery_receipt(conn, receipt_id)
    assert receipt is not None
    return receipt


def record_delivery_outcome(
    conn: sqlite3.Connection,
    receipt_id: int,
    *,
    owner_id: str,
    owner_epoch: int,
    attempt_id: str,
    desired_revision: int,
    state: str,
    message_id: Optional[str] = None,
    destination_profile: Optional[str] = None,
    delivered_revision: Optional[int] = None,
    retry_disposition: Optional[str] = None,
    error: Optional[str] = None,
    now: Optional[int] = None,
    _reconcile: bool = False,
) -> DeliveryReceipt:
    """CAS-record a remote result from the exact live receipt lease."""
    owner_id = _text(owner_id, field="owner_id")
    attempt_id = _text(attempt_id, field="attempt_id")
    state = _text(state, field="state")
    if state not in DELIVERY_STATES - {"pending"}:
        raise ValueError("outcome state must be sent, unknown, failed, or deleted")
    desired_revision = int(desired_revision)
    stamp = _now(now)
    if state == "sent":
        if not message_id or not destination_profile:
            raise ValueError("sent outcome requires message_id and destination_profile")
        if delivered_revision is None or int(delivered_revision) != desired_revision:
            raise DeliveryReceiptRevisionError("sent outcome must acknowledge the claimed revision")
        retry_disposition = retry_disposition or "delivered"
    elif state == "deleted":
        if not message_id:
            raise ValueError("deleted outcome requires the known message_id")
    elif state == "failed" and not retry_disposition:
        raise ValueError("failed outcome requires retry_disposition")
    if state == "unknown":
        retry_disposition = retry_disposition or "reconcile_required"
    if state != "deleted" and retry_disposition not in _RETRY_DISPOSITIONS:
        raise ValueError("retry_disposition is not a supported bounded disposition")
    with _write_scope(conn):
        row = _receipt_row(conn, receipt_id)
        if desired_revision != row["attempted_revision"]:
            raise DeliveryReceiptLeaseLost("attempt revision does not match recorded intent")
        if _reconcile and row["state"] == "sent":
            if (row["attempt_id"] == attempt_id and row["owner_epoch"] == owner_epoch
                    and row["attempt_owner_id"] == owner_id and state == "sent"
                    and row["destination_message_id"] == str(message_id)):
                return DeliveryReceipt.from_row(row)
            raise DeliveryReceiptLeaseLost("conflicting duplicate evidence")
        if state == "deleted" and row["destination_message_id"] != str(message_id):
            raise DeliveryReceiptIdentityError("deletion message does not match receipt destination")
        if state == "deleted" and retry_disposition is None:
            retry_disposition = retry_disposition or (
                "replacement_available"
                if int(row["replacement_budget"]) > 0
                else "no_retry"
            )
        if retry_disposition not in _RETRY_DISPOSITIONS:
            raise ValueError("retry_disposition is not a supported bounded disposition")
        if state == "sent" and row["destination_message_id"] not in (None, str(message_id)):
            raise DeliveryReceiptIdentityError("destination message changed unexpectedly")
        # pending -> unresolved is the durable once-per-attempt charge. Expiry
        # and quarantine already charged unknown; exact late evidence must not
        # charge it again. Only verified success (or confirmed deletion) resets
        # consecutive failures. The CAS below fences successor-attempt evidence.
        failures = (int(row["failure_count"]) + int(row["state"] == "pending")
                    if state in {"failed", "unknown"} else 0)
        if state == "failed" and failures >= 3 and retry_disposition in _RETRYABLE_DISPOSITIONS:
            retry_disposition = "exhausted"
        delivered_sql = "?" if state == "sent" else "delivered_revision"
        params: list[Any] = [state]
        if state == "sent":
            assert delivered_revision is not None
            params.append(int(delivered_revision))
        params.extend([
            str(message_id) if message_id is not None else None,
            destination_profile, retry_disposition, error, stamp, failures,
            state, state,
            int(receipt_id), owner_id, int(owner_epoch), attempt_id,
            desired_revision,
        ])
        guard = ("state IN ('pending', 'unknown') AND attempt_owner_id = ?" if _reconcile
                 else "state = 'pending' AND owner_id = ?")
        expiry_guard = "" if _reconcile else " AND lease_expires_at > ?"
        if not _reconcile:
            params.append(stamp)
        cur = conn.execute(
            f"""
            UPDATE kanban_delivery_receipts
               SET state = ?, delivered_revision = {delivered_sql},
                   destination_message_id = COALESCE(?, destination_message_id),
                   destination_profile = COALESCE(?, destination_profile),
                   owner_id = NULL, lease_expires_at = NULL,
                   retry_disposition = ?, last_error = ?, updated_at = ?, failure_count = ?,
                   renderer_hash = CASE WHEN ? = 'unknown' THEN NULL ELSE renderer_hash END,
                   control_hash = CASE WHEN ? = 'unknown' THEN NULL ELSE control_hash END
             WHERE id = ? AND {guard}
               AND owner_epoch = ? AND attempt_id = ?
               AND attempted_revision = ? {expiry_guard}
            """,
            params,
        )
        if cur.rowcount != 1:
            raise DeliveryReceiptLeaseLost("stale or expired delivery lease cannot record outcome")
    receipt = get_delivery_receipt(conn, receipt_id)
    assert receipt is not None
    return receipt


def authorize_delivery_replacement(
    conn: sqlite3.Connection,
    receipt_id: int,
    *,
    desired_revision: int,
    renderer_version: Optional[str] = None,
    renderer_hash: Optional[str] = None,
    now: Optional[int] = None,
) -> DeliveryReceipt:
    """Spend one explicit replacement budget for a known deleted message."""
    stamp = _now(now)
    desired_revision = int(desired_revision)
    if desired_revision <= 0:
        raise DeliveryReceiptRevisionError("desired_revision must be positive")
    with _write_scope(conn):
        row = _receipt_row(conn, receipt_id)
        source = get_task_source(
            conn, row["task_id"], expected_revision=desired_revision,
            task_incarnation=int(row["task_incarnation"]),
        )
        if desired_revision < int(row["desired_revision"]):
            raise DeliveryReceiptRevisionError("replacement revision regressed")
        if row["state"] != "deleted" or int(row["replacement_budget"]) <= 0:
            raise DeliveryReceiptNotRetryable("replacement is not authorized or budget is exhausted")
        cur = conn.execute(
            """
            UPDATE kanban_delivery_receipts
               SET desired_revision = ?, state = 'pending',
                   destination_message_id = NULL, destination_profile = NULL,
                   owner_id = NULL, lease_expires_at = NULL, attempt_id = NULL,
                   retry_disposition = 'replacement_authorized', last_error = NULL,
                   renderer_version = COALESCE(?, renderer_version),
                   renderer_hash = COALESCE(?, renderer_hash),
                   replacement_budget = replacement_budget - 1, updated_at = ?
             WHERE id = ? AND state = 'deleted' AND replacement_budget > 0
            """,
            (
                max(desired_revision, int(row["desired_revision"])), renderer_version,
                renderer_hash, stamp, int(receipt_id),
            ),
        )
        if cur.rowcount != 1:
            raise DeliveryReceiptLeaseLost("replacement authorization lost a race")
    receipt = get_delivery_receipt(conn, receipt_id)
    assert receipt is not None
    return receipt
