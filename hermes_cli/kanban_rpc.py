"""Remote-worker board RPC — owner-side protocol, receipts, and execution.

A remote worker never opens the board. It sends versioned JSON-RPC frames over a
persistent stdio channel (`hermes kanban rpc --stdio`, reached by SSH) carrying:

    {version, board_id, actor_task_id, run_id, request_id, operation,
     target_task_id, payload}

The owner authenticates the connection, validates the frame, and executes the
operation against its **local** board. Retryable mutations are deduplicated by a
receipt row committed **in the same transaction as the mutation**.

Crash argument (the reason the receipt shares the mutation's transaction):

    before commit        neither mutation nor receipt survives -> retry executes once
    after commit, no reply  both survive                      -> retry returns stored result
    after reply          both survive                         -> retry returns stored result

The two arrangements that would break it — committing a "processed" marker before
the mutation, or committing the mutation and then inserting its receipt in a
separate transaction — are both forbidden.

Transport is deliberately absent from this module: everything here operates on an
already-authenticated request. See `kanban_rpc_stdio` for the framing loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_log = logging.getLogger(__name__)

#: Protocol version. A frame with any other value is rejected before touching the DB.
PROTOCOL_VERSION = 1

#: Hard cap on a single frame, enforced before JSON parsing.
MAX_FRAME_BYTES = 1 << 20  # 1 MiB

#: Operations the owner will execute. Deliberately small: this is increment one.
OP_CLAIM_STATUS = "claim_status"
OP_COMMENT = "comment"

#: Read-only operations, which do not consume a receipt.
READ_ONLY_OPERATIONS = frozenset({OP_CLAIM_STATUS})

#: Mutating operations, which MUST commit a receipt alongside their effect.
MUTATING_OPERATIONS = frozenset({OP_COMMENT})


class RpcError(Exception):
    """A request the owner refuses. ``code`` is stable and machine-readable."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_frame(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message,
                                       "retryable": self.retryable}}


# --------------------------------------------------------------------------
# Request envelope
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardRequest:
    """One validated request. ``request_hash`` is computed over the canonical form."""

    version: int
    board_id: str
    actor_task_id: str
    run_id: int
    request_id: str
    operation: str
    target_task_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_wire(cls, frame: dict[str, Any]) -> BoardRequest:
        """Validate a decoded frame. Raises :class:`RpcError` on any defect.

        Validation is strict and happens before the database is touched: an
        unparseable or incomplete frame must never reach a transaction.
        """
        if not isinstance(frame, dict):
            raise RpcError("malformed_frame", "frame must be a JSON object")

        version = frame.get("version")
        # Exact type check: ``True == 1`` and ``1.0 == 1`` in Python, so an
        # equality test would accept a bool or float as a protocol version.
        if isinstance(version, bool) or not isinstance(version, int):
            raise RpcError("malformed_frame", "'version' must be an integer")
        if version != PROTOCOL_VERSION:
            raise RpcError(
                "unsupported_version",
                f"protocol version {version!r} is not supported (expected {PROTOCOL_VERSION})",
            )

        def _require_str(key: str) -> str:
            value = frame.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RpcError("malformed_frame", f"{key!r} must be a non-empty string")
            return value

        operation = _require_str("operation")
        if operation not in (READ_ONLY_OPERATIONS | MUTATING_OPERATIONS):
            raise RpcError("unsupported_operation", f"operation {operation!r} is not supported")

        run_id = frame.get("run_id")
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
            raise RpcError("malformed_frame", "'run_id' must be a positive integer")

        payload = frame.get("payload") or {}
        if not isinstance(payload, dict):
            raise RpcError("malformed_frame", "'payload' must be an object")

        target = frame.get("target_task_id")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise RpcError("malformed_frame", "'target_task_id' must be a non-empty string or absent")

        return cls(
            version=version,
            board_id=_require_str("board_id"),
            actor_task_id=_require_str("actor_task_id"),
            run_id=run_id,
            request_id=_require_str("request_id"),
            operation=operation,
            target_task_id=target,
            payload=payload,
        )

    # -- canonical form ----------------------------------------------------

    def canonical(self) -> bytes:
        """Canonical bytes for hashing.

        Covers the whole validated envelope EXCEPT any credential: the capability
        token is not a field here by construction, so a token rotation must not
        change a request's identity. Sorted keys + separators make the encoding
        independent of dict ordering and incidental whitespace.
        """
        body = {
            "version": self.version,
            "board_id": self.board_id,
            "actor_task_id": self.actor_task_id,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "operation": self.operation,
            "target_task_id": self.target_task_id,
            "payload": self.payload,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")

    def request_hash(self) -> bytes:
        return hashlib.sha256(self.canonical()).digest()


# --------------------------------------------------------------------------
# Receipts — the durable dedup
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Receipt:
    result: dict[str, Any]


def _lookup_receipt(conn: sqlite3.Connection, req: BoardRequest) -> Optional[Receipt]:
    """Return the stored receipt for this request, or None.

    Reusing a request id for a *different* envelope is a hard conflict: the
    caller has a bug or is replaying with altered content, and neither executing
    nor silently returning the old result is safe.
    """
    row = conn.execute(
        "SELECT request_hash, result_json FROM board_rpc_receipts "
        "WHERE board_id = ? AND run_id = ? AND request_id = ?",
        (req.board_id, req.run_id, req.request_id),
    ).fetchone()
    if row is None:
        return None
    stored_hash = row["request_hash"]
    if isinstance(stored_hash, str):
        stored_hash = bytes.fromhex(stored_hash)
    if stored_hash != req.request_hash():
        raise RpcError(
            "request_id_conflict",
            f"request_id {req.request_id!r} was already used for a different request",
        )
    try:
        result = json.loads(row["result_json"])
    except (TypeError, ValueError):
        raise RpcError("receipt_unreadable", "stored receipt could not be decoded")
    if not isinstance(result, dict):
        raise RpcError("receipt_unreadable", "stored receipt is not an object")
    return Receipt(result=result)


def _store_receipt(conn: sqlite3.Connection, req: BoardRequest,
                   result: dict[str, Any], *, now: Optional[int] = None) -> None:
    """Insert the receipt. MUST be called inside the mutation's own transaction."""
    conn.execute(
        "INSERT INTO board_rpc_receipts "
        "(board_id, run_id, request_id, actor_task_id, target_task_id, operation, "
        " request_hash, result_json, committed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (req.board_id, req.run_id, req.request_id, req.actor_task_id,
         req.target_task_id, req.operation, req.request_hash(),
         json.dumps(result, sort_keys=True, ensure_ascii=False),
         int(now if now is not None else time.time())),
    )


# --------------------------------------------------------------------------
# Authorization — verified inside the mutation's transaction
# --------------------------------------------------------------------------

def _authority_row(conn: sqlite3.Connection, req: BoardRequest) -> sqlite3.Row:
    """Fetch the task + run rows that decide authority."""
    row = conn.execute(
        "SELECT t.status AS task_status, t.current_run_id, t.claim_lock AS task_claim_lock, "
        "       t.claim_expires AS task_claim_expires, t.last_heartbeat_at, "
        "       r.id AS run_id, r.task_id AS run_task_id, r.status AS run_status, "
        "       r.claim_lock AS run_claim_lock, r.claim_expires AS run_claim_expires, "
        "       r.ended_at AS run_ended_at "
        "  FROM tasks t LEFT JOIN task_runs r ON r.id = ? "
        " WHERE t.id = ?",
        (req.run_id, req.actor_task_id),
    ).fetchone()
    if row is None:
        raise RpcError("unknown_actor_task", f"actor task {req.actor_task_id!r} does not exist")
    return row


def authority_problems(conn: sqlite3.Connection, req: BoardRequest,
                       *, now: Optional[int] = None) -> list[str]:
    """Every reason this request currently lacks mutation authority, in order.

    Deliberately reports the FULL list rather than the first failure: an operator
    debugging a rejected remote worker needs to see all of it, and the checks are
    cheap and already loaded.

    Mirrors the ``worker_claim_is_live`` contract: an expired lease means mutation
    authority has lapsed *even before the reaper processes the row*, and a closed
    run is not authority either. Reporting an expired or nonexistent run as live
    is worse than reporting nothing, because a caller acts on it.
    """
    ts = int(now if now is not None else time.time())
    row = _authority_row(conn, req)
    problems: list[str] = []

    if row["task_status"] != "running":
        problems.append(f"task is {row['task_status']!r}, not 'running'")
    if row["current_run_id"] != req.run_id:
        problems.append(f"run {req.run_id} is not the live run (current_run_id={row['current_run_id']})")

    # -- run row ----------------------------------------------------------
    if row["run_id"] is None:
        problems.append(f"run {req.run_id} does not exist")
        return problems
    if row["run_task_id"] != req.actor_task_id:
        problems.append(f"run {req.run_id} belongs to {row['run_task_id']!r}, not the actor task")
    if row["run_status"] != "running":
        problems.append(f"run status is {row['run_status']!r}, not 'running'")
    if row["run_ended_at"] is not None:
        problems.append(f"run was closed at {row['run_ended_at']}")

    # -- leases (both rows must be unexpired) ------------------------------
    for label, expires in (("task", row["task_claim_expires"]),
                           ("run", row["run_claim_expires"])):
        if expires is None:
            problems.append(f"{label} has no claim lease (claim_expires is NULL)")
        elif int(expires) <= ts:
            problems.append(f"{label} claim lease expired at {expires} (now {ts})")

    return problems


def verify_run_authority(conn: sqlite3.Connection, req: BoardRequest) -> None:
    """Raise unless the actor task currently holds live authority for THIS run.

    Re-checked under the caller's write lock rather than trusted from connection
    setup: between authenticating the connection and taking the write
    transaction, the card can be reclaimed and a second run started.

    NOTE: this is *authorization*. It is not authentication — the identity triple
    (task, run, claim lock) is a concurrency coordinate, not a credential.
    """
    problems = authority_problems(conn, req)
    if problems:
        raise RpcError("no_authority", "; ".join(problems))


# --------------------------------------------------------------------------
# Operation implementations — transaction-owned, shared with local callers
# --------------------------------------------------------------------------

def _op_claim_status(conn: sqlite3.Connection, req: BoardRequest) -> dict[str, Any]:
    """Read-only: report the authoritative claim state for the actor run.

    This replaces the remote branch of ``worker_claim_is_live()``, which reads the
    board directly and therefore cannot work off-owner.

    ``is_live`` is derived from the SAME predicate that authorizes mutation, so it
    can never report a run as live that the mutation path would then reject. An
    expired lease or a closed run is reported NOT live, with the reasons attached —
    a caller acts on this answer, so a false positive is worse than no answer.
    """
    row = _authority_row(conn, req)
    problems = authority_problems(conn, req)
    return {
        "actor_task_id": req.actor_task_id,
        "run_id": req.run_id,
        "task_status": row["task_status"],
        "current_run_id": row["current_run_id"],
        "run_status": row["run_status"],
        "run_ended_at": row["run_ended_at"],
        "task_claim_expires": row["task_claim_expires"],
        "run_claim_expires": row["run_claim_expires"],
        "last_heartbeat_at": row["last_heartbeat_at"],
        "is_live": not problems,
        "problems": problems,
    }


def _op_comment(conn: sqlite3.Connection, req: BoardRequest, *,
                author: str) -> dict[str, Any]:
    """Mutating: append a comment to ``target_task_id`` (default: the actor task).

    The author is a REQUIRED parameter sourced from the authenticated grant, never
    read from the payload — the model controls the payload.

    Writes through ``insert_comment_txn``, which fires no observer. The caller
    fires the lifecycle hook after its commit; firing here would let an external
    effect escape a transaction that may still roll back.
    """
    body = req.payload.get("body")
    if not isinstance(body, str) or not body.strip():
        raise RpcError("invalid_payload", "'body' must be a non-empty string")
    target = req.target_task_id or req.actor_task_id
    comment_id = kb.insert_comment_txn(conn, target, author, body)
    return {"comment_id": comment_id, "target_task_id": target, "author": author}


_OPERATIONS: dict[str, Callable[..., dict[str, Any]]] = {
    OP_CLAIM_STATUS: _op_claim_status,
    OP_COMMENT: _op_comment,
}


# --------------------------------------------------------------------------
# The owner execution algorithm
# --------------------------------------------------------------------------

def execute_request(req: BoardRequest, *, conn: Optional[sqlite3.Connection] = None,
                    author: Optional[str] = None) -> dict[str, Any]:
    """Execute one authenticated request against the local board.

    ``author`` MUST come from the connection's authenticated grant. It is passed
    to operations as a separate argument — never injected into the payload — so a
    model-supplied ``_author`` cannot influence it, and the request object stays
    immutable so ``request_hash()`` keeps describing what the caller actually sent.

    An absent ``author`` is a refusal, not a fallback: this method is only ever
    reached from an authenticated channel, so no author means the channel did not
    establish one.
    """
    handler = _OPERATIONS.get(req.operation)
    if handler is None:
        raise RpcError("unsupported_operation", f"operation {req.operation!r} is not supported")

    if req.operation in MUTATING_OPERATIONS and not (author or "").strip():
        raise RpcError("no_authenticated_author",
                       "a mutating operation requires an authenticated author")

    own_conn = conn is None
    if own_conn:
        conn = kbc.connect()

    try:
        if req.operation in READ_ONLY_OPERATIONS:
            # No receipt: a read has no effect to deduplicate, and committing a
            # receipt for one would make every poll a write.
            return handler(conn, req)

        # The receipt is committed in the SAME transaction as the mutation. This
        # is the whole correctness argument; see the module docstring.
        with kbc.write_txn(conn):
            cached = _lookup_receipt(conn, req)
            if cached is not None:
                return cached.result
            verify_run_authority(conn, req)
            result = handler(conn, req, author=author)
            _store_receipt(conn, req, result)
        # ONLY now — after the commit — may an observer run. Firing inside the
        # transaction would let an external effect escape a rollback, and an
        # observer on another connection could not yet see the row.
        _fire_post_commit_obligations(conn, req, result)
        return result
    finally:
        if own_conn:
            conn.close()


def _fire_post_commit_obligations(conn: sqlite3.Connection, req: BoardRequest,
                                  result: dict[str, Any]) -> None:
    """Run the lifecycle observers a committed operation owes.

    NOTE: this is deliberately the SIMPLE form for now. If the process dies
    between the commit and this call, the observer never fires and the comment
    notification is lost — see the recoverable outbox in the design spec. Kept
    separate and named so that swap is local.
    """
    if req.operation == OP_COMMENT:
        try:
            kb._fire_task_hook_by_id(
                conn, "kanban_task_commented", result["target_task_id"],
                author=result["author"], body=req.payload.get("body"),
                comment_id=result["comment_id"],
            )
        except Exception as exc:
            _log.warning("kanban rpc: post-commit comment hook failed: %s", exc)


def new_request_id() -> str:
    """A fresh, opaque request id."""
    return secrets.token_hex(16)
