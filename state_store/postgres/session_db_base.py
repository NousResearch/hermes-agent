"""Operation-free PostgreSQL SessionDB composition base."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Optional, TypeVar

from state_store.postgres.core import PostgresReadOnlyError, PostgresStateStore
from state_store.session_api import json_dumps, json_loads, normalize_row, normalize_rows


T = TypeVar("T")


class PostgresSessionDBBase:
    """Shared PostgreSQL primitives; operation mixins own concrete SQL later."""

    _READ_BATCH_SIZE = 256
    _MAX_READ_ROWS = 10_000

    def __init__(
        self,
        state_store: PostgresStateStore,
        *,
        capabilities: Mapping[str, bool],
    ) -> None:
        self._state_store = state_store
        self.spec = state_store.spec
        self.read_only = self.spec.read_only
        self.capabilities = MappingProxyType(dict(capabilities))
        # Deliberately unlike SQLite SessionDB.db_path: PostgreSQL has no path.
        self.db_path = None

    @contextmanager
    def _tx(self, *, read_only: Optional[bool] = None) -> Iterator[Any]:
        """Yield a transaction-scoped connection only to internal helpers."""

        effective_read_only = self.read_only if read_only is None else read_only
        if self.read_only and not effective_read_only:
            raise PostgresReadOnlyError(
                "Cannot request a writable PostgreSQL SessionDB transaction"
            )
        with self._state_store.transaction(read_only=effective_read_only) as connection:
            yield connection

    def _run(
        self,
        operation: Callable[[Any], T],
        *,
        read_only: Optional[bool] = False,
    ) -> T:
        """Run one internal operation without exposing a connection to callers."""

        with self._tx(read_only=read_only) as connection:
            return operation(connection)

    def _read_one(
        self,
        query: str,
        params: Sequence[Any] = (),
    ) -> Optional[dict[str, Any]]:
        """Run one bounded read and normalize mapping or tuple driver rows."""

        def operation(connection: Any) -> Optional[dict[str, Any]]:
            cursor = connection.execute(query, params)
            return normalize_row(cursor.fetchone(), columns=self._cursor_columns(cursor))

        return self._run(operation, read_only=True)

    def _read_many(
        self,
        query: str,
        params: Sequence[Any] = (),
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Run a bounded streaming read; never call ``fetchall``."""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("read limit must be a positive integer")
        if limit > self._MAX_READ_ROWS:
            raise ValueError("read limit exceeds PostgreSQL SessionDB maximum")

        def operation(connection: Any) -> list[dict[str, Any]]:
            cursor = connection.execute(query, params)
            columns = self._cursor_columns(cursor)
            rows: list[Any] = []
            remaining = limit
            while remaining:
                batch = cursor.fetchmany(min(self._READ_BATCH_SIZE, remaining))
                if not batch:
                    break
                rows.extend(batch)
                remaining -= len(batch)
            return normalize_rows(rows, columns=columns, limit=limit)

        return self._run(operation, read_only=True)

    def _write(self, query: str, params: Sequence[Any] = ()) -> int:
        """Run a write without allowing a cursor to escape this base class."""

        def operation(connection: Any) -> int:
            cursor = connection.execute(query, params)
            rowcount = getattr(cursor, "rowcount", 0)
            return int(rowcount) if rowcount is not None else 0

        return self._run(operation, read_only=False)

    def _write_returning(
        self,
        query: str,
        params: Sequence[Any] = (),
    ) -> Optional[dict[str, Any]]:
        """Run a write with one normalized ``RETURNING`` row."""

        def operation(connection: Any) -> Optional[dict[str, Any]]:
            cursor = connection.execute(query, params)
            return normalize_row(cursor.fetchone(), columns=self._cursor_columns(cursor))

        return self._run(operation, read_only=False)

    @staticmethod
    def _cursor_columns(cursor: Any) -> tuple[str, ...]:
        description = getattr(cursor, "description", None) or ()
        columns: list[str] = []
        for column in description:
            name = getattr(column, "name", None)
            columns.append(str(name if name is not None else column[0]))
        return tuple(columns)

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json_dumps(value)

    @staticmethod
    def _json_loads(value: Optional[str], *, default: Any = None) -> Any:
        return json_loads(value, default=default)

    def close(self) -> None:
        """Release the underlying bounded PostgreSQL state-store pool."""

        self._state_store.close()
