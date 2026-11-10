"""Transaction-bound Triage release contracts for the Kanban database facade."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol


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
    """
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

            changes_before_guard = conn.total_changes
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
