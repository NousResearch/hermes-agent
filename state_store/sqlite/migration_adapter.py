"""Read-only SQLite source adapter for state-store migration.

This module deliberately does not modify live data.  Its writer fence holds a
SQLite ``BEGIN IMMEDIATE`` transaction, which is enforceable by every process
that writes through SQLite's normal locking protocol.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3
import tempfile
import threading
from typing import Any, ContextManager, Optional


_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)
_MIGRATION_ATTEMPT_LOCK = threading.RLock()
_WRITER_FENCE_HOOK: Optional[Callable[[str], ContextManager[None]]] = None


def register_writer_fence_hook(
    hook: Optional[Callable[[str], ContextManager[None]]],
) -> None:
    """Register an optional runtime-specific quiescence hook.

    The SQLite transaction remains the authoritative cross-process writer
    fence.  A runtime may add this hook to stop non-SQLite side effects before
    the migration begins, but registration is not required for safety.
    """

    global _WRITER_FENCE_HOOK
    _WRITER_FENCE_HOOK = hook


def _quote_identifier(value: str) -> str:
    if not value or any(character not in _IDENTIFIER_CHARS for character in value):
        raise ValueError("SQLite migration identifier is invalid")
    return f'"{value}"'


def _readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _readwrite_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=rw"


def _lexicographic_predicate(
    columns: Sequence[str], after_key: tuple[Any, ...]
) -> tuple[str, tuple[Any, ...]]:
    if len(columns) != len(after_key):
        raise ValueError("SQLite keyset cursor does not match key columns")
    clauses: list[str] = []
    parameters: list[Any] = []
    for index, column in enumerate(columns):
        prefix = [
            f"{_quote_identifier(previous)} = ?"
            for previous in columns[:index]
        ]
        clauses.append(
            "(" + " AND ".join(prefix + [f"{_quote_identifier(column)} > ?"]) + ")"
        )
        parameters.extend(after_key[:index])
        parameters.append(after_key[index])
    return "(" + " OR ".join(clauses) + ")", tuple(parameters)


class SQLiteBackupSnapshot:
    """Read-only, removable SQLite backup created with ``Connection.backup``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection = sqlite3.connect(
            _readonly_uri(path),
            uri=True,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA query_only=ON")

    def available_tables(self) -> list[str]:
        cursor = self._connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        return [str(row["name"]) for row in cursor.fetchmany(10_000)]

    def columns(self, table: str) -> tuple[str, ...]:
        cursor = self._connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")
        return tuple(str(row["name"]) for row in cursor.fetchmany(10_000))

    def fetchmany_keyset(
        self,
        table: str,
        key_columns: Sequence[str],
        after_key: Optional[tuple[Any, ...]],
        batch_size: int,
    ) -> list[Mapping[str, Any]]:
        if batch_size < 1:
            raise ValueError("SQLite migration batch_size must be positive")
        columns = self.columns(table)
        if not columns:
            raise ValueError(f"SQLite source table {table!r} does not exist")
        if not key_columns:
            raise ValueError("SQLite migration keyset requires at least one column")
        selected = ", ".join(_quote_identifier(column) for column in columns)
        order_by = ", ".join(_quote_identifier(column) for column in key_columns)
        parameters: tuple[Any, ...] = ()
        where = ""
        if after_key is not None:
            predicate, parameters = _lexicographic_predicate(key_columns, after_key)
            where = f" WHERE {predicate}"
        cursor = self._connection.execute(
            f"SELECT {selected} FROM {_quote_identifier(table)}{where} "
            f"ORDER BY {order_by} LIMIT ?",
            (*parameters, batch_size),
        )
        return [dict(row) for row in cursor.fetchmany(batch_size)]

    def close(self) -> None:
        try:
            self._connection.close()
        finally:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass


class SQLiteMigrationSourceAdapter:
    """Production source adapter for one existing SQLite ``state.db``."""

    def __init__(
        self,
        sqlite_path: Path,
        *,
        writer_fence_hook: Optional[Callable[[str], ContextManager[None]]] = None,
        snapshot_directory: Optional[Path] = None,
        fence_timeout_s: float = 5.0,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self._writer_fence_hook = writer_fence_hook
        self._snapshot_directory = snapshot_directory
        if fence_timeout_s <= 0:
            raise ValueError("SQLite migration fence timeout must be positive")
        self._fence_timeout_s = float(fence_timeout_s)
        self._fence_connection: Optional[sqlite3.Connection] = None
        self._fence_owner_thread: Optional[int] = None

    @contextmanager
    def writer_fence(self, run_id: str) -> Iterator[None]:
        """Hold a SQLite writer lock throughout the migration lifetime.

        ``BEGIN IMMEDIATE`` waits for an existing writer to commit, then holds
        SQLite's database-wide writer reservation.  This blocks writers in
        other Hermes processes without copying WAL files or changing live data.
        """

        with _MIGRATION_ATTEMPT_LOCK:
            if self._fence_connection is not None:
                raise RuntimeError("SQLite state-postgres writer fence is already held")
            connection = self._connect_writer_fence()
            hook = self._writer_fence_hook or _WRITER_FENCE_HOOK
            try:
                connection.execute(
                    f"PRAGMA busy_timeout = {int(self._fence_timeout_s * 1_000)}"
                )
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.DatabaseError:
                connection.close()
                raise RuntimeError(
                    "could not acquire cross-process SQLite writer fence; "
                    "stop active writers and retry the offline migration"
                ) from None
            self._fence_connection = connection
            self._fence_owner_thread = threading.get_ident()
            try:
                if hook is None:
                    yield
                else:
                    with hook(run_id):
                        yield
            finally:
                self._fence_connection = None
                self._fence_owner_thread = None
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.DatabaseError:
                    pass
                connection.close()

    def available_tables(self) -> list[str]:
        with self._connect_readonly() as connection:
            cursor = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            return [str(row["name"]) for row in cursor.fetchmany(10_000)]

    def sqlite_integrity_errors(self) -> list[str]:
        with self._connect_readonly() as connection:
            rows = connection.execute("PRAGMA integrity_check").fetchmany(100)
        return [str(row[0]) for row in rows if str(row[0]).lower() != "ok"]

    def sqlite_foreign_key_violations(self) -> list[str]:
        with self._connect_readonly() as connection:
            rows = connection.execute("PRAGMA foreign_key_check").fetchmany(100)
        return [
            f"{row[0]} rowid={row[1]} parent={row[2]} fk={row[3]}"
            for row in rows
        ]

    def active_writer_reasons(self) -> list[str]:
        # The actual exclusion is owned by writer_fence().  Keeping this
        # separate lets the engine recheck database-backed admission facts.
        return []

    def active_compression_leases(self, now: float) -> list[str]:
        return self._limited_values(
            "compression_locks",
            "session_id",
            "expires_at >= ?",
            (now,),
        )

    def active_delegations(self) -> list[str]:
        return self._limited_values(
            "async_delegations",
            "delegation_id",
            (
                "COALESCE(delivery_state, 'pending') NOT IN ('delivered', 'failed', "
                "'cancelled') OR completed_at IS NULL"
            ),
            (),
        )

    def active_handoffs(self) -> list[str]:
        return self._limited_values(
            "sessions",
            "id",
            "handoff_state IN ('pending', 'running')",
            (),
        )

    def snapshot_via_sqlite_backup(self, run_id: str) -> SQLiteBackupSnapshot:
        """Snapshot through SQLite's backup API; never copy WAL/SHM files."""

        del run_id
        self._require_writer_fence()
        directory = self._snapshot_directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="hermes-state-postgres-",
            suffix=".sqlite",
            dir=str(directory) if directory is not None else None,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with self._connect_readonly() as source:
                destination = sqlite3.connect(str(temporary_path), isolation_level=None)
                try:
                    source.backup(destination)
                finally:
                    destination.close()
            return SQLiteBackupSnapshot(temporary_path)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    @contextmanager
    def _connect_readonly(self) -> Iterator[sqlite3.Connection]:
        if not self.sqlite_path.is_file():
            raise RuntimeError("SQLite state database does not exist")
        connection = sqlite3.connect(
            _readonly_uri(self.sqlite_path),
            uri=True,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            yield connection
        finally:
            connection.close()

    def _connect_writer_fence(self) -> sqlite3.Connection:
        if not self.sqlite_path.is_file():
            raise RuntimeError("SQLite state database does not exist")
        return sqlite3.connect(
            _readwrite_uri(self.sqlite_path),
            uri=True,
            timeout=self._fence_timeout_s,
            isolation_level=None,
        )

    def _require_writer_fence(self) -> None:
        if (
            self._fence_connection is None
            or self._fence_owner_thread != threading.get_ident()
        ):
            raise RuntimeError(
                "SQLite snapshot requires the cross-process writer fence to be held"
            )

    def _limited_values(
        self,
        table: str,
        column: str,
        predicate: str,
        parameters: tuple[Any, ...],
    ) -> list[str]:
        with self._connect_readonly() as connection:
            if not self._table_exists(connection, table):
                return []
            cursor = connection.execute(
                f"SELECT {_quote_identifier(column)} FROM {_quote_identifier(table)} "
                f"WHERE {predicate} ORDER BY {_quote_identifier(column)} LIMIT 50",
                parameters,
            )
            values = [str(row[0]) for row in cursor.fetchmany(50)]
        if len(values) == 50:
            values.append("additional active records")
        return values

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone() is not None
