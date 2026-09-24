"""Opaque, durable Telegram action records on the Kanban board.

Record helpers remain inert. ``execute_blocker_choice`` is the bounded canonical
execution caller: it composes claim, the existing unblock handler, audit and
durable outcome within one caller-owned ``kanban_db.write_txn``.
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
from hermes_cli import kanban_db_surface as _surface


ACTION_STATES = frozenset({
    "pending", "claimed", "completed", "failed", "unknown", "expired", "rejected",
})
_TERMINAL_STATES = frozenset({"completed", "failed", "unknown", "expired", "rejected"})


class ActionRecordError(RuntimeError):
    """Base class for action-record failures."""


class ActionAuthorizationError(ActionRecordError):
    """Identity or route did not match; no cached result is exposed."""


class ActionIdempotencyConflict(ActionRecordError):
    pass


class ActionStaleSource(ActionRecordError):
    pass


class ActionExpired(ActionRecordError):
    pass


class ActionAlreadyClaimed(ActionRecordError):
    pass


class ActionUnknown(ActionRecordError):
    pass


class ActionLeaseLost(ActionRecordError):
    pass


@dataclass(frozen=True)
class ActionRecord:
    id: int
    token: str
    task_id: str
    task_incarnation: int
    expected_revision: int
    expected_task_status: str
    board_identity: str
    profile: str
    telegram_principal: int
    origin_chat_id: str
    origin_thread_id: str
    origin_message_id: str
    action_kind: str
    action_payload: dict[str, Any]
    conflict_key: Optional[str]
    expires_at: int
    idempotency_key: str
    state: str
    claim_epoch: int
    claim_owner: Optional[str]
    claim_attempt_id: Optional[str]
    claim_expires_at: Optional[int]
    result: Optional[dict[str, Any]]
    created_at: int
    updated_at: int

    @classmethod
    def from_row(cls, row: sqlite3.Row, *, expose_token: bool = True) -> "ActionRecord":
        return cls(
            id=int(row["id"]),
            token=str(row["token"]) if expose_token else "",
            task_id=str(row["task_id"]),
            task_incarnation=int(row["task_incarnation"]),
            expected_revision=int(row["expected_revision"]),
            expected_task_status=str(row["expected_task_status"]),
            board_identity=str(row["board_identity"]),
            profile=str(row["profile"]),
            telegram_principal=int(row["telegram_principal"]),
            origin_chat_id=str(row["origin_chat_id"]),
            origin_thread_id=str(row["origin_thread_id"] or ""),
            origin_message_id=str(row["origin_message_id"]),
            action_kind=str(row["action_kind"]),
            action_payload=_decode_object(row["action_payload"]),
            conflict_key=row["conflict_key"],
            expires_at=int(row["expires_at"]),
            idempotency_key=str(row["idempotency_key"]),
            state=str(row["state"]),
            claim_epoch=int(row["claim_epoch"]),
            claim_owner=row["claim_owner"],
            claim_attempt_id=row["claim_attempt_id"],
            claim_expires_at=(
                int(row["claim_expires_at"])
                if row["claim_expires_at"] is not None
                else None
            ),
            result=_decode_object(row["result"]) if row["result"] else None,
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
        )


@dataclass(frozen=True)
class ActionClaim:
    record: ActionRecord
    claimed: bool
    attempt_id: Optional[str]


def _decode_object(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _json_object(value: Optional[Mapping[str, Any]], *, field: str) -> str:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON values") from exc
    if len(encoded.encode("utf-8")) > 16 * 1024:
        raise ValueError(f"{field} is too large")
    return encoded


def _text(value: Any, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} is required")
    result = value if isinstance(value, str) else str(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _principal(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("telegram_principal must be a positive integer")
    return value


def _now(value: Optional[int]) -> int:
    return int(time.time()) if value is None else int(value)


@contextlib.contextmanager
def _write_scope(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    with _kb.write_txn(conn, allow_nested=bool(conn.in_transaction)):
        yield conn


def _route_values(
    *,
    board_identity: str,
    profile: str,
    telegram_principal: Any,
    origin_chat_id: str,
    origin_thread_id: Optional[str],
    origin_message_id: str,
) -> tuple[str, str, int, str, str, str]:
    return (
        _text(board_identity, field="board_identity"),
        _text(profile, field="profile"),
        _principal(telegram_principal),
        _text(origin_chat_id, field="origin_chat_id"),
        "" if origin_thread_id is None else str(origin_thread_id),
        _text(origin_message_id, field="origin_message_id"),
    )


def _row_for_token(conn: sqlite3.Connection, token: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM kanban_action_records WHERE token = ?", (token,)
    ).fetchone()
    if row is None:
        raise ActionAuthorizationError("action token is not authorized")
    return row


def _verify_route(row: sqlite3.Row, route: tuple[str, str, int, str, str, str]) -> None:
    board, profile, principal, chat, thread, message = route
    if (
        row["board_identity"] != board
        or row["profile"] != profile
        or int(row["telegram_principal"]) != principal
        or row["origin_chat_id"] != chat
        or (row["origin_thread_id"] or "") != thread
        or row["origin_message_id"] != message
    ):
        # Do not return the row, its state, or its prior result on this path.
        raise ActionAuthorizationError("action token is not authorized")


def _validate_live_source(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    try:
        source = _surface.get_task_source(
            conn,
            row["task_id"],
            expected_revision=int(row["expected_revision"]),
            expected_status=row["expected_task_status"],
            task_incarnation=int(row["task_incarnation"]),
        )
    except _surface.DeliveryReceiptError as exc:
        raise ActionStaleSource("action source is no longer current") from exc
    if source.current_revision != int(row["expected_revision"]):
        raise ActionStaleSource("action source revision is stale")


def _same_immutable_action(row: sqlite3.Row, values: tuple[Any, ...]) -> bool:
    (
        task_id, incarnation, revision, task_status, board, profile, principal,
        chat, thread, message, kind, payload, conflict_key, expires_at,
    ) = values
    return (
        row["task_id"] == task_id
        and int(row["task_incarnation"]) == incarnation
        and int(row["expected_revision"]) == revision
        and row["expected_task_status"] == task_status
        and row["board_identity"] == board
        and row["profile"] == profile
        and int(row["telegram_principal"]) == principal
        and row["origin_chat_id"] == chat
        and (row["origin_thread_id"] or "") == thread
        and row["origin_message_id"] == message
        and row["action_kind"] == kind
        and _decode_object(row["action_payload"]) == _decode_object(payload)
        and row["conflict_key"] == conflict_key
        and int(row["expires_at"]) == expires_at
    )


def issue_action(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    expected_revision: int,
    expected_task_status: str,
    board_identity: str,
    profile: str,
    telegram_principal: int,
    origin_chat_id: str,
    origin_thread_id: Optional[str] = None,
    origin_message_id: str,
    action_kind: str,
    action_payload: Optional[Mapping[str, Any]] = None,
    expires_at: int,
    idempotency_key: str,
    conflict_key: Optional[str] = None,
    task_incarnation: Optional[int] = None,
    now: Optional[int] = None,
) -> ActionRecord:
    """Issue or repeat one opaque action record, without executing its payload."""
    board, profile, principal, chat, thread, message = _route_values(
        board_identity=board_identity, profile=profile,
        telegram_principal=telegram_principal, origin_chat_id=origin_chat_id,
        origin_thread_id=origin_thread_id, origin_message_id=origin_message_id,
    )
    task_id = _text(task_id, field="task_id")
    action_kind = _text(action_kind, field="action_kind")
    idempotency_key = _text(idempotency_key, field="idempotency_key")
    conflict_key = str(conflict_key) if conflict_key is not None else None
    expected_status = _text(expected_task_status, field="expected_task_status")
    expected_revision = int(expected_revision)
    expires_at = int(expires_at)
    stamp = _now(now)
    if expires_at <= stamp:
        raise ValueError("expires_at must be in the future")
    payload_json = _json_object(action_payload, field="action_payload")
    immutable = (
        task_id, None, expected_revision, expected_status, board, profile, principal,
        chat, thread, message, action_kind, payload_json, conflict_key, expires_at,
    )
    with _write_scope(conn):
        source = _surface.get_task_source(
            conn, task_id, expected_revision=expected_revision,
            expected_status=expected_status, task_incarnation=task_incarnation,
        )
        immutable = (
            task_id, source.task_incarnation, expected_revision, expected_status,
            board, profile, principal, chat, thread, message, action_kind,
            payload_json, conflict_key, expires_at,
        )
        existing = conn.execute(
            "SELECT * FROM kanban_action_records WHERE board_identity = ? AND idempotency_key = ?",
            (board, idempotency_key),
        ).fetchone()
        if existing is not None:
            if not _same_immutable_action(existing, immutable):
                raise ActionIdempotencyConflict("idempotency key is bound to another action")
            # The issuer is the authorized actor for the callback.  Returning
            # this row is safe only after every immutable route field matched.
            return ActionRecord.from_row(existing)
        token = ""
        for _ in range(4):
            token = secrets.token_urlsafe(18)
            try:
                conn.execute(
                    """
                    INSERT INTO kanban_action_records (
                        token, task_id, task_incarnation, expected_revision,
                        expected_task_status, board_identity, profile,
                        telegram_principal, origin_chat_id, origin_thread_id,
                        origin_message_id, action_kind, action_payload,
                        conflict_key, expires_at, idempotency_key,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        token, task_id, source.task_incarnation, expected_revision,
                        expected_status, board, profile, principal, chat, thread,
                        message, action_kind, payload_json, conflict_key, expires_at,
                        idempotency_key, stamp, stamp,
                    ),
                )
                break
            except sqlite3.IntegrityError:
                if conn.execute(
                    "SELECT 1 FROM kanban_action_records WHERE board_identity = ? AND idempotency_key = ?",
                    (board, idempotency_key),
                ).fetchone() is not None:
                    raise ActionIdempotencyConflict("idempotency key raced another action")
        else:  # pragma: no cover - secrets collision is extraordinarily unlikely
            raise ActionRecordError("could not allocate opaque action token")
        action_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    row = conn.execute("SELECT * FROM kanban_action_records WHERE id = ?", (action_id,)).fetchone()
    assert row is not None
    return ActionRecord.from_row(row)


def claim_action(
    conn: sqlite3.Connection,
    token: str,
    *,
    board_identity: str,
    profile: str,
    telegram_principal: int,
    origin_chat_id: str,
    origin_thread_id: Optional[str] = None,
    origin_message_id: str,
    claim_owner: str,
    lease_seconds: int = 120,
    now: Optional[int] = None,
) -> ActionClaim:
    """Authorize and CAS-claim an action token exactly once."""
    token = _text(token, field="token")
    claim_owner = _text(claim_owner, field="claim_owner")
    lease_seconds = int(lease_seconds)
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    stamp = _now(now)
    route = _route_values(
        board_identity=board_identity, profile=profile,
        telegram_principal=telegram_principal, origin_chat_id=origin_chat_id,
        origin_thread_id=origin_thread_id, origin_message_id=origin_message_id,
    )
    expired_claim = False
    expired_token = False
    attempt_id: Optional[str] = None
    with _write_scope(conn):
        row = _row_for_token(conn, token)
        _verify_route(row, route)
        state = str(row["state"])
        if state in _TERMINAL_STATES:
            return ActionClaim(ActionRecord.from_row(row), claimed=False, attempt_id=None)
        if state == "claimed":
            if row["claim_expires_at"] is not None and int(row["claim_expires_at"]) <= stamp:
                conn.execute(
                    """
                    UPDATE kanban_action_records
                       SET state = 'unknown', result = ?, updated_at = ?
                     WHERE id = ? AND state = 'claimed' AND claim_epoch = ?
                    """,
                    (json.dumps({"ok": False, "reason": "uncertain_claim"}), stamp,
                     int(row["id"]), int(row["claim_epoch"])),
                )
                expired_claim = True
            else:
                raise ActionAlreadyClaimed("action is already claimed")
        elif state != "pending":
            raise ActionRecordError(f"action state {state!r} cannot be claimed")
        if not expired_claim:
            if int(row["expires_at"]) <= stamp:
                conn.execute(
                    "UPDATE kanban_action_records SET state = 'expired', result = ?, updated_at = ? WHERE id = ? AND state = 'pending'",
                    (json.dumps({"ok": False, "reason": "expired"}), stamp, int(row["id"])),
                )
                expired_token = True
            else:
                _validate_live_source(conn, row)
                attempt_id = secrets.token_urlsafe(12)
                new_epoch = int(row["claim_epoch"]) + 1
                cur = conn.execute(
                    """
                    UPDATE kanban_action_records
                       SET state = 'claimed', claim_epoch = ?, claim_owner = ?,
                           claim_attempt_id = ?, claim_expires_at = ?, updated_at = ?
                     WHERE id = ? AND state = 'pending' AND claim_epoch = ?
                       AND expires_at > ?
                    """,
                    (
                        new_epoch, claim_owner, attempt_id, stamp + lease_seconds, stamp,
                        int(row["id"]), int(row["claim_epoch"]), stamp,
                    ),
                )
                if cur.rowcount != 1:
                    raise ActionLeaseLost("action claim lost a compare-and-swap race")
                # A callback choice group is exclusive.  Mark sibling choices as
                # rejected before the winner's lease leaves this transaction.
                if row["conflict_key"] is not None:
                    conn.execute(
                        """
                        UPDATE kanban_action_records
                           SET state = 'rejected',
                               result = ?, updated_at = ?
                         WHERE board_identity = ? AND task_id = ? AND task_incarnation = ?
                           AND expected_revision = ? AND conflict_key = ?
                           AND id != ? AND state = 'pending'
                        """,
                        (
                            json.dumps({"ok": False, "reason": "competing_action_won"}), stamp,
                            row["board_identity"], row["task_id"], int(row["task_incarnation"]),
                            int(row["expected_revision"]), row["conflict_key"], int(row["id"]),
                        ),
                    )
    if expired_claim:
        raise ActionUnknown("expired claim has an uncertain execution outcome")
    if expired_token:
        raise ActionExpired("action token has expired")
    fresh = conn.execute("SELECT * FROM kanban_action_records WHERE id = ?", (int(row["id"]),)).fetchone()
    assert fresh is not None
    return ActionClaim(
        record=ActionRecord.from_row(fresh), claimed=True, attempt_id=attempt_id,
    )


def record_action_outcome(
    conn: sqlite3.Connection,
    token: str,
    *,
    board_identity: str,
    profile: str,
    telegram_principal: int,
    origin_chat_id: str,
    origin_thread_id: Optional[str] = None,
    origin_message_id: str,
    claim_owner: str,
    claim_epoch: int,
    attempt_id: str,
    outcome_state: str,
    result: Optional[Mapping[str, Any]] = None,
    now: Optional[int] = None,
) -> ActionRecord:
    """Record a canonical handler result under the exact live claim.

    The payload is inert JSON.  This function never changes the task, event
    stream, comment log, or any other authority.
    """
    token = _text(token, field="token")
    claim_owner = _text(claim_owner, field="claim_owner")
    attempt_id = _text(attempt_id, field="attempt_id")
    outcome_state = _text(outcome_state, field="outcome_state")
    if outcome_state not in {"completed", "failed", "unknown"}:
        raise ValueError("outcome_state must be completed, failed, or unknown")
    result_json = _json_object(result, field="result")
    stamp = _now(now)
    route = _route_values(
        board_identity=board_identity, profile=profile,
        telegram_principal=telegram_principal, origin_chat_id=origin_chat_id,
        origin_thread_id=origin_thread_id, origin_message_id=origin_message_id,
    )
    with _write_scope(conn):
        row = _row_for_token(conn, token)
        _verify_route(row, route)
        cur = conn.execute(
            """
            UPDATE kanban_action_records
               SET state = ?, result = ?, updated_at = ?
             WHERE token = ? AND state = 'claimed' AND claim_owner = ?
               AND claim_epoch = ? AND claim_attempt_id = ?
               AND claim_expires_at > ?
            """,
            (
                outcome_state, result_json, stamp, token, claim_owner,
                int(claim_epoch), attempt_id, stamp,
            ),
        )
        if cur.rowcount != 1:
            raise ActionLeaseLost("stale or expired action claim cannot record a result")
    fresh = conn.execute("SELECT * FROM kanban_action_records WHERE token = ?", (token,)).fetchone()
    assert fresh is not None
    return ActionRecord.from_row(fresh)


def get_authorized_action(
    conn: sqlite3.Connection,
    token: str,
    *,
    board_identity: str,
    profile: str,
    telegram_principal: int,
    origin_chat_id: str,
    origin_thread_id: Optional[str] = None,
    origin_message_id: str,
) -> ActionRecord:
    """Read an action only after exact actor and origin-route validation."""
    token = _text(token, field="token")
    route = _route_values(
        board_identity=board_identity, profile=profile,
        telegram_principal=telegram_principal, origin_chat_id=origin_chat_id,
        origin_thread_id=origin_thread_id, origin_message_id=origin_message_id,
    )
    row = _row_for_token(conn, token)
    _verify_route(row, route)
    return ActionRecord.from_row(row)


def blocker_choice_applicable(conn, task_id):
    """Only an idle needs-input blocker; no review, provider reset or worker power."""
    row = conn.execute(
        "SELECT status, block_kind, current_run_id, worker_pid FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    return bool(row and row["status"] == "blocked" and row["block_kind"] == "needs_input"
                and row["current_run_id"] is None and row["worker_pid"] is None
                and _kb._resume_status_from_events(conn, task_id) == "ready")


def execute_blocker_choice(conn, token, *, authorize, **route):
    """One local transaction: current host policy, claim, canonical transition, audit, result.

    ``authorize(record, conn)`` is a host-supplied live authorization check, never
    callback payload data. It runs after acquiring SQLite's writer lock, including
    on duplicates. No external effects or awaits may occur within this boundary.
    """
    with _kb.write_txn(conn):
        record = get_authorized_action(conn, token, **route)
        authorize(record, conn)
        stamp = int(time.time())
        if not record.created_at <= stamp < record.expires_at:
            raise ActionExpired("action is outside its validity interval")
        if record.action_kind != "unblock_needs_input":
            raise ActionAuthorizationError("unsupported action")
        source = _surface.get_task_source(conn, record.task_id,
                                          task_incarnation=record.task_incarnation)
        if record.state == "completed":
            return record.result
        if (source.current_revision != record.expected_revision
                or not blocker_choice_applicable(conn, record.task_id)):
            raise ActionStaleSource("blocker choice is no longer applicable")
        claim = claim_action(conn, token, claim_owner="canonical-blocker-choice", **route)
        if not claim.claimed:
            raise ActionStaleSource("choice did not execute")
        if not _kb.unblock_task(conn, record.task_id, allow_nested=True):
            raise ActionStaleSource("canonical transition refused")
        task = _kb.get_task(conn, record.task_id)
        assert task is not None and claim.attempt_id is not None
        _kb._append_event(conn, record.task_id, "task_decision", {
            "action_id": record.id, "actor": record.telegram_principal,
            "kind": record.action_kind, "status": task.status,
            "grant_id": record.action_payload["grant"]["id"],
        })
        result = {"ok": True, "status": task.status, "action_id": record.id,
                  "revision": _surface.get_task_source(conn, record.task_id).current_revision}
        record_action_outcome(conn, token, claim_owner="canonical-blocker-choice",
                              claim_epoch=claim.record.claim_epoch, attempt_id=claim.attempt_id,
                              outcome_state="completed", result=result, **route)
        # Policy may change in another process while the transaction runs. Recheck
        # the original pending capability so denial rolls back every local effect.
        authorize(record, conn)
        return result
