"""Transaction-bound Triage release contracts for the Kanban database facade.

Trust model
-----------
A release guard is **trusted first-party code**. The only caller today is the
Office's fixed guard in ``server/collect.mjs`` (writerflow/hermes-office). The
scoped SQLite authorizer and the read-only connection facade are defence in depth
against *mistakes* in such a guard (an accidental write, commit or DDL); they are
**not a sandbox** against a hostile callable. A guard that deliberately reaches
past the Hermes wrapper (for example ``sqlite3.Connection.set_authorizer(conn,
None)`` through the base descriptor) can lift the policy; the post-guard
change/transaction checks then roll the release back, but that is a backstop, not
isolation. Never pass a guard that is untrusted or not fixed in source. Running
guards out of process is deliberately out of scope; the follow-up is
``FOLLOWUP_REF`` below, and it triggers if any caller ever passes an untrusted or
non-fixed guard.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol

FOLLOWUP_REF = "https://linear.app/writerflow/issue/WRI-2435"


@dataclass(frozen=True)
class TriageReleaseGuardResult:
    """Decision returned by a guarded Triage release precondition callback."""

    allowed: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class TriageReleaseReceipt:
    """Stable result of an attempted guarded Triage release."""

    released: bool
    outcome: str
    status: Optional[str]
    changed_fields: tuple[str, ...] = ()
    guard_reason: Optional[str] = None


class TriageReleaseGuardConnection(Protocol):
    """Read-only, transaction-bound connection surface supplied to a guard."""

    @property
    def in_transaction(self) -> bool: ...

    def execute(self, sql: str, parameters: Any = ()) -> Any: ...

    def cursor(self) -> Any: ...


class _GuardDenied(Exception):
    def __init__(self, receipt: TriageReleaseReceipt) -> None:
        super().__init__(receipt.guard_reason)
        self.receipt = receipt


def _require_read_statement(sql: str) -> None:
    """Reject every SQL class except a plain SELECT before SQLite sees it."""
    tokens = sql.split(None, 1)
    keyword = tokens[0].upper() if tokens else ""
    if keyword != "SELECT":
        raise RuntimeError("Triage release guard connection is read-only")


class _ReadOnlyCursor:
    """Cursor view that cannot turn a previously allowed read into a write."""

    __slots__ = ("__connection", "__cursor")

    def __init__(self, cursor: sqlite3.Cursor, connection: _ReadOnlyConnection) -> None:
        self.__cursor = cursor
        self.__connection = connection

    @property
    def connection(self) -> _ReadOnlyConnection:
        return self.__connection

    @property
    def description(self) -> Any:
        return self.__cursor.description

    @property
    def rowcount(self) -> int:
        return self.__cursor.rowcount

    def execute(self, sql: str, parameters: Any = ()) -> _ReadOnlyCursor:
        _require_read_statement(sql)
        self.__cursor.execute(sql, parameters)
        return self

    def executemany(self, _sql: str, _parameters: Any) -> None:
        raise RuntimeError("Triage release guard connection is read-only")

    def executescript(self, _sql_script: str) -> None:
        raise RuntimeError("Triage release guard connection is read-only")

    def fetchone(self) -> Optional[sqlite3.Row]:
        return self.__cursor.fetchone()

    def fetchmany(self, size: Optional[int] = None) -> list[sqlite3.Row]:
        return self.__cursor.fetchmany(size) if size is not None else self.__cursor.fetchmany()

    def fetchall(self) -> list[sqlite3.Row]:
        return self.__cursor.fetchall()

    def close(self) -> None:
        self.__cursor.close()

    def __iter__(self) -> Iterator[sqlite3.Row]:
        return iter(self.__cursor)


class _ReadOnlyConnection:
    """Transaction-bound connection view exposed to release guards."""

    __slots__ = ("__connection",)

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.__connection = connection

    @property
    def in_transaction(self) -> bool:
        return self.__connection.in_transaction

    @property
    def total_changes(self) -> int:
        return self.__connection.total_changes

    def execute(self, sql: str, parameters: Any = ()) -> _ReadOnlyCursor:
        _require_read_statement(sql)
        return _ReadOnlyCursor(self.__connection.execute(sql, parameters), self)

    def cursor(self) -> _ReadOnlyCursor:
        return _ReadOnlyCursor(self.__connection.cursor(), self)

    def executemany(self, _sql: str, _parameters: Any) -> None:
        raise RuntimeError("Triage release guard connection is read-only")

    def executescript(self, _sql_script: str) -> None:
        raise RuntimeError("Triage release guard connection is read-only")

    def commit(self) -> None:
        raise RuntimeError("Triage release guard connection is read-only")

    def rollback(self) -> None:
        raise RuntimeError("Triage release guard connection is read-only")


_GUARD_DENIED_ACTIONS = frozenset(
    getattr(sqlite3, name)
    for name in (
        "SQLITE_ALTER_TABLE",
        "SQLITE_ANALYZE",
        "SQLITE_ATTACH",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_CREATE_VTABLE",
        "SQLITE_DELETE",
        "SQLITE_DETACH",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_DROP_VTABLE",
        "SQLITE_INSERT",
        "SQLITE_PRAGMA",
        "SQLITE_REINDEX",
        "SQLITE_SAVEPOINT",
        "SQLITE_TRANSACTION",
        "SQLITE_UPDATE",
    )
)


# Built-in, side-effect-free SQL functions a read-only guard may call. Any other
# function (user-defined ones registered with ``create_function`` included) stays
# denied, since a user function could run arbitrary Python inside the transaction.
_GUARD_ALLOWED_FUNCTIONS = frozenset(
    {
        "abs", "avg", "coalesce", "count", "glob", "group_concat", "ifnull", "iif",
        "instr", "length", "like", "lower", "ltrim", "max", "min", "nullif",
        "replace", "round", "rtrim", "substr", "substring", "sum", "total",
        "trim", "typeof", "upper",
    }
)


def _require_known_authorizer(conn: sqlite3.Connection) -> None:
    """Refuse connections whose current authorizer cannot be read back.

    SQLite exposes no getter, so the prior authorizer can only be restored when the
    connection records it (``TrackedConnection`` from ``kanban_db_connect.connect``).
    On a plain ``sqlite3.Connection`` the call would replace the caller's authorizer
    with no way to put it back (failing open), so it is refused before ``BEGIN``.
    """
    if not hasattr(conn, "_hermes_authorizer_locked"):
        raise TypeError(
            "guarded_specify_triage_task needs a connection that records its authorizer "
            "(use hermes_cli.kanban_db_connect.connect); a plain sqlite3.Connection's "
            "existing authorizer could not be restored"
        )


@contextmanager
def _guard_authorizer(conn: sqlite3.Connection) -> Iterator[None]:
    """Deny recursive writes/txn control during a guard, then restore the exact prior policy."""
    previous = conn._hermes_authorizer_callback  # type: ignore[attr-defined]

    def authorize(action, _arg1, arg2, _database, _source):
        if action == sqlite3.SQLITE_FUNCTION:
            if isinstance(arg2, str) and arg2.lower() in _GUARD_ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action in _GUARD_DENIED_ACTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    sqlite3.Connection.set_authorizer(conn, authorize)
    conn._hermes_authorizer_callback = authorize  # type: ignore[attr-defined]
    conn._hermes_authorizer_locked = True  # type: ignore[attr-defined]
    try:
        yield
    finally:
        try:
            sqlite3.Connection.set_authorizer(conn, previous)
        finally:
            conn._hermes_authorizer_callback = previous  # type: ignore[attr-defined]
            conn._hermes_authorizer_locked = False  # type: ignore[attr-defined]


def guarded_specify_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    guard: Callable[[TriageReleaseGuardConnection, str], TriageReleaseGuardResult],
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
) -> TriageReleaseReceipt:
    """Validate ``guard`` and release one Triage task in the same write transaction.

    The callback receives the transaction-bound connection for read-only precondition
    checks. Returning a denial rolls the transaction back before returning its receipt.
    Callback exceptions and malformed results propagate with the transaction rolled back.
    After a released outcome the global ``recompute_ready`` pass runs, as the unguarded
    path always did, so dependents of already-finished parents are promoted.

    Trust model: ``guard`` must be trusted first-party code (the only caller is the
    Office's fixed guard in ``server/collect.mjs``). The scoped authorizer and read-only
    facade protect against accidental writes; they are defence in depth, not a sandbox
    against a hostile callable, and a guard can lift the policy through the base
    ``sqlite3.Connection.set_authorizer`` descriptor. Never pass an untrusted or
    non-fixed guard. Out-of-process isolation is a tracked follow-up
    (``FOLLOWUP_REF`` in this module), triggered by any such caller.

    Authorizer: while the guard runs, a scoped authorizer denies writes, DDL, PRAGMA,
    transaction control and non-allow-listed SQL functions (built-in aggregates and
    scalars such as ``count`` and ``max`` are allowed). On exit, including on error, the
    caller's previous authorizer is restored exactly. That requires a connection that
    records its authorizer (``kanban_db_connect.connect``); any other connection is
    refused with ``TypeError`` before ``BEGIN``, with nothing written, because its
    authorizer could not be restored. An authorizer installed through the base
    ``sqlite3.Connection.set_authorizer`` descriptor bypasses that record and cannot be
    detected or restored (the same trusted-caller boundary).
    """
    _require_known_authorizer(conn)
    receipt = _release_triage_task(
        conn, task_id, guard=guard, title=title, body=body, assignee=assignee, author=author
    )
    if receipt.released:
        from hermes_cli import kanban_db as kb

        kb.recompute_ready(conn)
    return receipt


def _release_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    guard: Optional[Callable[[TriageReleaseGuardConnection, str], TriageReleaseGuardResult]],
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
) -> TriageReleaseReceipt:
    """Release one Triage task; run ``guard`` (when given) inside the write transaction."""
    from hermes_cli import kanban_db as kb

    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = kb._canonical_assignee(assignee)
    try:
        with kb.write_txn(conn):
            existing = conn.execute(
                "SELECT title, body, assignee FROM tasks WHERE id = ? AND status = 'triage'",
                (task_id,),
            ).fetchone()
            if existing is None:
                status = kb._task_status(conn, task_id)
                return TriageReleaseReceipt(False, "not_in_triage", status)

            if guard is not None:
                changes_before_guard = conn.total_changes
                with _guard_authorizer(conn):
                    decision = guard(_ReadOnlyConnection(conn), task_id)
                if not conn.in_transaction:
                    raise RuntimeError("Triage release guard ended its transaction")
                if conn.total_changes != changes_before_guard:
                    raise RuntimeError("Triage release guard must be read-only")
                if not isinstance(decision, TriageReleaseGuardResult):
                    raise TypeError("guard must return TriageReleaseGuardResult")
                reason = (decision.reason or "").strip() or None
                if not decision.allowed:
                    if reason is None:
                        raise ValueError("denied Triage release guard requires a reason")
                    raise _GuardDenied(
                        TriageReleaseReceipt(
                            False, "precondition_failed", "triage", guard_reason=reason,
                        )
                    )

            sets: list[str] = ["status = 'todo'"]
            params: list[object] = []
            changed_fields: list[str] = []
            if title is not None and title.strip() != (existing["title"] or ""):
                sets.append("title = ?")
                params.append(title.strip())
                changed_fields.append("title")
            if body is not None and (body or "") != (existing["body"] or ""):
                sets.append("body = ?")
                params.append(body)
                changed_fields.append("body")
            if assignee is not None and assignee != (existing["assignee"] or None):
                sets.append("assignee = ?")
                params.append(assignee)
                changed_fields.append("assignee")
            params.append(task_id)
            cur = conn.execute(
                f"UPDATE tasks SET {', '.join(sets)} WHERE id = ? AND status = 'triage'",
                tuple(params),
            )
            if cur.rowcount != 1:
                return TriageReleaseReceipt(False, "not_in_triage", kb._task_status(conn, task_id))
            if changed_fields and author and author.strip():
                kb._insert_comment(
                    conn,
                    task_id,
                    author.strip(),
                    "Specified — updated " + ", ".join(changed_fields) + " and promoted to todo.",
                    int(time.time()),
                )
            kb._append_event(
                conn,
                task_id,
                "specified",
                {"changed_fields": changed_fields} if changed_fields else None,
            )
            final_status = "todo"
            if kb._parents_satisfied(conn, task_id):
                final_status = kb._resume_status_from_events(conn, task_id)
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'",
                    (final_status, task_id),
                )
                kb._append_event(
                    conn,
                    task_id,
                    "promoted",
                    {"status": final_status} if final_status != "ready" else None,
                )
            return TriageReleaseReceipt(
                True,
                "released",
                final_status,
                tuple(changed_fields),
            )
    except _GuardDenied as denied:
        return denied.receipt
