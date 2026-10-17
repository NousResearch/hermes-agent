"""Existing-only SQLite readers for the stock cron ledgers.

The producer constructors are intentionally not used.  Each database is opened
through an existing-only ``mode=rw`` URI, its real connection takes
``BEGIN IMMEDIATE`` for the complete consumption window, and the qualified WAL
guard is released before SQLite closes the connection.  No journal transition,
checkpoint, migration, or sidecar manipulation is performed here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator, Mapping


_SCHEMA_ROW_LIMIT = 256


class SQLiteCaptureRefused(RuntimeError):
    """A cron SQLite source cannot be safely attached or has unknown schema."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise SQLiteCaptureRefused("cron_capture_deadline_exceeded")


@dataclass(frozen=True, slots=True)
class SQLiteColumnSpec:
    name: str
    declared_type: str
    not_null: int
    primary_key: int
    default: str | None = None


@dataclass(frozen=True, slots=True)
class SQLiteIndexSpec:
    name: str
    columns: tuple[str, ...]
    descending: tuple[bool, ...] = ()
    partial: bool = False
    unique: bool = False
    collations: tuple[str, ...] = ()
    predicate: str | None = None


@dataclass(frozen=True, slots=True)
class SQLiteTableSpec:
    name: str
    columns: tuple[SQLiteColumnSpec, ...]
    indexes: tuple[SQLiteIndexSpec, ...] = ()
    ddl_fragments: tuple[str, ...] = ()
    foreign_keys: tuple[tuple[str, str, str], ...] = ()
    ddl_exact: str | None = None


def _existing_rw_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=rw"


def _set_no_close_checkpoint(conn: sqlite3.Connection) -> bool:
    flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
    setter = getattr(conn, "setconfig", None)
    if flag is None or not callable(setter):
        return False
    try:
        setter(flag, True)
    except (AttributeError, TypeError, ValueError, sqlite3.Error):
        return False
    return True


def _norm_sql(value: object) -> str:
    normalized = " ".join(str(value or "").replace("\n", " ").split()).lower()
    return normalized.replace("create table if not exists ", "create table ")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _schema_rows(
    conn: sqlite3.Connection,
    statement: str,
    parameters: tuple[Any, ...] = (),
    *,
    deadline: float | None = None,
) -> Iterator[Any]:
    """Read one bounded schema row at a time with a cooperative deadline."""

    _check_deadline(deadline)
    try:
        cursor = conn.execute(statement, parameters)
    except sqlite3.Error as exc:
        raise SQLiteCaptureRefused("cron_capture_sqlite_schema_unreadable") from exc
    rows = 0
    while True:
        _check_deadline(deadline)
        try:
            row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise SQLiteCaptureRefused("cron_capture_sqlite_schema_unreadable") from exc
        if row is None:
            return
        rows += 1
        if rows > _SCHEMA_ROW_LIMIT:
            raise SQLiteCaptureRefused("cron_capture_sqlite_schema_too_large")
        yield row


def _set_length_limit(conn: sqlite3.Connection, max_bytes: int) -> None:
    """Fail closed unless SQLite can bound one cell before it is materialized."""

    category = getattr(sqlite3, "SQLITE_LIMIT_LENGTH", None)
    setter = getattr(conn, "setlimit", None)
    if not isinstance(category, int) or not callable(setter):
        raise SQLiteCaptureRefused("cron_capture_sqlite_row_limit_unavailable")
    try:
        setter(category, max_bytes)
        effective = setter(category, -1)
    except (AttributeError, OverflowError, TypeError, ValueError, sqlite3.Error) as exc:
        raise SQLiteCaptureRefused("cron_capture_sqlite_row_limit_unavailable") from exc
    if not isinstance(effective, int) or effective <= 0 or effective > max_bytes:
        raise SQLiteCaptureRefused("cron_capture_sqlite_row_limit_unavailable")


def _begin_immediate(conn: sqlite3.Connection, deadline: float) -> None:
    """Take the actual SQLite writer fence with bounded non-blocking attempts."""

    try:
        conn.execute("PRAGMA busy_timeout=0")
    except sqlite3.Error as exc:
        raise SQLiteCaptureRefused("cron_capture_sqlite_probe_failed") from exc
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise SQLiteCaptureRefused("cron_capture_sqlite_begin_failed") from exc
            if time.monotonic() >= deadline:
                raise SQLiteCaptureRefused("cron_capture_sqlite_busy") from exc
            delay = min(0.02, max(0.0, deadline - time.monotonic()))
            if delay <= 0:
                raise SQLiteCaptureRefused("cron_capture_sqlite_busy") from exc
            time.sleep(delay)


def _column_specs(
    conn: sqlite3.Connection,
    table: str,
    *,
    deadline: float | None = None,
) -> tuple[SQLiteColumnSpec, ...]:
    result: list[SQLiteColumnSpec] = []
    for row in _schema_rows(conn, f"PRAGMA table_xinfo({_quote(table)})", deadline=deadline):
        if len(row) >= 7 and int(row[6] or 0) != 0:
            raise SQLiteCaptureRefused("cron_capture_sqlite_hidden_column")
        result.append(
            SQLiteColumnSpec(
                name=str(row[1]),
                declared_type=str(row[2] or ""),
                not_null=int(row[3] or 0),
                primary_key=int(row[5] or 0),
                default=None if row[4] is None else str(row[4]),
            )
        )
    return tuple(result)


def _index_specs(
    conn: sqlite3.Connection,
    table: str,
    *,
    deadline: float | None = None,
) -> tuple[SQLiteIndexSpec, ...]:
    result: list[SQLiteIndexSpec] = []
    for row in _schema_rows(conn, f"PRAGMA index_list({_quote(table)})", deadline=deadline):
        name = str(row[1])
        origin = str(row[3]) if len(row) > 3 else ""
        if name.startswith("sqlite_autoindex_") or origin == "pk":
            continue
        columns: list[str] = []
        descending: list[bool] = []
        collations: list[str] = []
        info = _schema_rows(conn, f"PRAGMA index_xinfo({_quote(name)})", deadline=deadline)
        for item in sorted(info, key=lambda value: int(value[0])):
            if len(item) >= 6 and int(item[5] or 0) == 0:
                continue
            if len(item) < 4 or item[2] is None:
                continue
            columns.append(str(item[2]))
            descending.append(bool(item[3]))
            collations.append(str(item[4] or "BINARY"))
        sql_row = next(
            _schema_rows(
                conn,
                "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                (name,),
                deadline=deadline,
            ),
            None,
        )
        normalized = _norm_sql(sql_row[0] if sql_row else "")
        predicate = None
        if " where " in normalized:
            predicate = normalized.split(" where ", 1)[1]
        result.append(
            SQLiteIndexSpec(
                name=name,
                columns=tuple(columns),
                descending=tuple(descending),
                partial=bool(row[4]) if len(row) > 4 else False,
                unique=bool(row[2]) if len(row) > 2 else False,
                collations=tuple(collations),
                predicate=predicate,
            )
        )
    return tuple(sorted(result, key=lambda value: value.name))


def _validate_schema(
    conn: sqlite3.Connection,
    specs: Mapping[str, SQLiteTableSpec],
    *,
    deadline: float | None = None,
) -> tuple[tuple[str, str, str | None], ...]:
    table_names: set[str] = set()
    table_sql: dict[str, str] = {}
    evidence: list[tuple[str, str, str | None]] = []
    for kind, name, table_name, sql in _schema_rows(
        conn,
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name",
        deadline=deadline,
    ):
        _check_deadline(deadline)
        kind = str(kind)
        name = str(name)
        table_name = str(table_name or "")
        if kind == "table":
            if name not in specs:
                raise SQLiteCaptureRefused("cron_capture_sqlite_unknown_table")
            table_names.add(name)
            table_sql[name] = _norm_sql(sql)
        elif kind == "index":
            if table_name not in specs:
                raise SQLiteCaptureRefused("cron_capture_sqlite_unknown_index")
            allowed = {item.name for item in specs[table_name].indexes}
            if name not in allowed:
                raise SQLiteCaptureRefused("cron_capture_sqlite_unknown_index")
        elif kind not in {"view", "trigger"}:
            raise SQLiteCaptureRefused("cron_capture_sqlite_unknown_object")
        else:
            raise SQLiteCaptureRefused("cron_capture_sqlite_unknown_object")
        evidence.append((kind, name, None if sql is None else str(sql)))

    if not table_names:
        raise SQLiteCaptureRefused("cron_capture_sqlite_no_known_table")
    for name in sorted(table_names):
        spec = specs[name]
        actual_columns = _column_specs(conn, name, deadline=deadline)
        if actual_columns != spec.columns:
            raise SQLiteCaptureRefused("cron_capture_sqlite_columns_changed")
        actual_indexes = _index_specs(conn, name, deadline=deadline)
        expected_indexes = tuple(sorted(spec.indexes, key=lambda value: value.name))
        if actual_indexes != expected_indexes:
            raise SQLiteCaptureRefused("cron_capture_sqlite_indexes_changed")
        if spec.ddl_exact is not None and table_sql[name] != spec.ddl_exact:
            raise SQLiteCaptureRefused("cron_capture_sqlite_ddl_changed")
        if any(fragment not in table_sql[name] for fragment in spec.ddl_fragments):
            raise SQLiteCaptureRefused("cron_capture_sqlite_ddl_changed")
        try:
            foreign = tuple(
                (str(item[3]), str(item[2]), str(item[4]))
                for item in _schema_rows(
                    conn,
                    f"PRAGMA foreign_key_list({_quote(name)})",
                    deadline=deadline,
                )
            )
        except sqlite3.Error as exc:
            raise SQLiteCaptureRefused("cron_capture_sqlite_schema_unreadable") from exc
        if foreign != spec.foreign_keys:
            raise SQLiteCaptureRefused("cron_capture_sqlite_foreign_keys_changed")
    return tuple(evidence)


def _schema_values(
    conn: sqlite3.Connection,
    tables: tuple[str, ...],
    *,
    deadline: float | None = None,
) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for table in tables:
        _check_deadline(deadline)
        columns = tuple(
            dict(
                name=item[1],
                declared_type=item[2],
                not_null=int(item[3] or 0),
                default=item[4],
                primary_key=int(item[5] or 0),
                hidden=int(item[6] or 0) if len(item) > 6 else 0,
            )
            for item in _schema_rows(
                conn,
                f"PRAGMA table_xinfo({_quote(table)})",
                deadline=deadline,
            )
        )
        indexes = tuple(
            dict(
                name=item.name,
                columns=item.columns,
                descending=item.descending,
                partial=item.partial,
                unique=item.unique,
                collations=item.collations,
                predicate=item.predicate,
            )
            for item in _index_specs(conn, table, deadline=deadline)
        )
        values.append(dict(table=table, columns=columns, indexes=indexes))
    _check_deadline(deadline)
    return tuple(values)


@dataclass(slots=True)
class SQLiteCaptureHandle:
    path: Path
    conn: sqlite3.Connection
    expected_identity: tuple[int, int] | None
    expected_sidecars: dict[str, tuple[int, int]]
    capture_empty_wal: tuple[int, int] | None
    capture_created_shm: tuple[int, int] | None
    wal_guard: Any
    tables: tuple[str, ...]
    schema_sql: tuple[tuple[str, str, str | None], ...]
    journal_mode: str
    schema_version: int
    user_version: int
    application_id: int
    expected_journal: tuple[int, int] | None
    closed: bool = False

    def assert_current(self, *, deadline: float | None = None) -> None:
        _check_deadline(deadline)
        if self.closed:
            raise SQLiteCaptureRefused("cron_capture_sqlite_released")
        from hermes_state_dbfile import _stat_db_file_identity, _stat_sqlite_sidecar_identity

        current = _stat_db_file_identity(self.path)
        if current != self.expected_identity:
            raise SQLiteCaptureRefused("cron_capture_sqlite_changed")
        _check_deadline(deadline)
        current_sidecars = _stat_sqlite_sidecar_identity(self.path)
        expected_sidecars = {
            key: value
            for key, value in self.expected_sidecars.items()
        }
        if self.capture_empty_wal is not None:
            expected_sidecars["-wal"] = self.capture_empty_wal
            _require_empty_wal(self.path, current_sidecars, self.capture_empty_wal)
        if self.capture_created_shm is not None:
            expected_sidecars["-shm"] = self.capture_created_shm
        if current_sidecars != expected_sidecars:
            raise SQLiteCaptureRefused("cron_capture_sqlite_sidecar_changed")
        _check_deadline(deadline)
        journal_path = Path(str(self.path) + "-journal")
        try:
            metadata = journal_path.stat()
            current_journal = (int(metadata.st_dev), int(metadata.st_ino))
        except FileNotFoundError:
            current_journal = None
        except OSError as exc:
            raise SQLiteCaptureRefused("cron_capture_sqlite_journal_changed") from exc
        if current_journal != self.expected_journal:
            raise SQLiteCaptureRefused("cron_capture_sqlite_journal_changed")
        _check_deadline(deadline)

    def schema_values(self, *, deadline: float | None = None) -> tuple[dict[str, Any], ...]:
        return _schema_values(self.conn, self.tables, deadline=deadline)

    def iter_rows(
        self,
        table: str,
        *,
        page_size: int = 128,
        deadline: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        _check_deadline(deadline)
        if table not in self.tables:
            raise SQLiteCaptureRefused("cron_capture_sqlite_unknown_table")
        if type(page_size) is not int or page_size <= 0:
            raise SQLiteCaptureRefused("cron_capture_page_size_invalid")
        columns = _column_specs(self.conn, table, deadline=deadline)
        primary = tuple(item.name for item in sorted(columns, key=lambda item: item.primary_key) if item.primary_key)
        if not primary:
            raise SQLiteCaptureRefused("cron_capture_sqlite_primary_key_missing")
        order = ", ".join(f"{_quote(item)} ASC" for item in primary)
        try:
            cursor = self.conn.execute(f"SELECT * FROM {_quote(table)} ORDER BY {order}")
        except sqlite3.DataError as exc:
            raise SQLiteCaptureRefused("cron_capture_byte_budget_exceeded") from exc
        except sqlite3.Error as exc:
            raise SQLiteCaptureRefused("cron_capture_sqlite_row_read_failed") from exc
        while True:
            _check_deadline(deadline)
            try:
                row = cursor.fetchone()
            except sqlite3.DataError as exc:
                raise SQLiteCaptureRefused("cron_capture_byte_budget_exceeded") from exc
            except sqlite3.Error as exc:
                raise SQLiteCaptureRefused("cron_capture_sqlite_row_read_failed") from exc
            if row is None:
                return
            _check_deadline(deadline)
            yield {str(key): row[key] for key in row.keys()}

    def close(self) -> None:
        if self.closed:
            return
        error: BaseException | None = None
        try:
            if self.conn.in_transaction:
                self.conn.rollback()
        except sqlite3.Error as exc:
            error = exc
        try:
            if self.wal_guard:
                import hermes_state_lockguard as lockguard

                lockguard.release(self.wal_guard)
                self.wal_guard = {}
        except OSError as exc:
            if error is None:
                error = exc
        try:
            self.conn.close()
        except sqlite3.Error as exc:
            if error is None:
                error = exc
        self.closed = True
        if error is not None:
            raise SQLiteCaptureRefused("cron_capture_sqlite_close_failed") from error


def _open_handle(
    path: Path,
    specs: Mapping[str, SQLiteTableSpec],
    *,
    deadline: float,
    max_bytes: int | None = None,
) -> SQLiteCaptureHandle:
    import hermes_state_dbfile as dbfile
    import hermes_state_lockguard as lockguard
    from hermes_state_wal import _on_disk_journal_mode

    expected_identity = dbfile._stat_db_file_identity(path)
    if expected_identity is None or not path.is_file():
        raise SQLiteCaptureRefused("cron_capture_sqlite_missing")
    try:
        dbfile.refuse_deleted_wal_generation(path)
        conn = dbfile._connect_tracked_db(
            _existing_rw_uri(path),
            tracking_path=path,
            uri=True,
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
    except SQLiteCaptureRefused:
        raise
    except BaseException as exc:
        raise SQLiteCaptureRefused("cron_capture_sqlite_open_failed") from exc
    guard: Any = {}
    journal_path = Path(str(path) + "-journal")
    try:
        if dbfile._stat_db_file_identity(path) != expected_identity:
            raise SQLiteCaptureRefused("cron_capture_sqlite_changed")
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
                raise SQLiteCaptureRefused("cron_capture_budget_invalid")
            _set_length_limit(conn, max_bytes)
        if not _set_no_close_checkpoint(conn):
            raise SQLiteCaptureRefused("cron_capture_sqlite_no_close_guard_unavailable")
        initial_sidecars = dbfile._stat_sqlite_sidecar_identity(path)
        try:
            metadata = journal_path.stat()
            expected_journal = (int(metadata.st_dev), int(metadata.st_ino))
        except FileNotFoundError:
            expected_journal = None
        except OSError as exc:
            raise SQLiteCaptureRefused("cron_capture_sqlite_journal_unavailable") from exc
        if initial_sidecars:
            guard = lockguard.hold(path, guard)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        mode = _on_disk_journal_mode(conn)
        if mode not in {"wal", "delete", "truncate", "persist", "memory", "off"}:
            raise SQLiteCaptureRefused("cron_capture_sqlite_journal_unknown")
        if mode == "wal":
            guard = lockguard.hold(path, guard)
            if lockguard.supported() and not guard:
                raise SQLiteCaptureRefused("cron_capture_sqlite_wal_guard_unavailable")
        # Validate an existing schema before BEGIN IMMEDIATE.  A refused unknown
        # object must not mint a new WAL generation merely as a side effect of
        # the capture probe; accepted captures still repeat the validation under
        # the real writer fence below.
        _validate_schema(conn, specs, deadline=deadline)
        _begin_immediate(conn, deadline)
        schema_sql = _validate_schema(conn, specs, deadline=deadline)
        capture_empty_wal = _capture_empty_wal(path, initial_sidecars, dbfile)
        capture_created_shm = _capture_created_shm(path, initial_sidecars, dbfile)
        tables = tuple(sorted({item[1] for item in schema_sql if item[0] == "table"}))
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        application_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
        return SQLiteCaptureHandle(
            path=path,
            conn=conn,
            expected_identity=expected_identity,
            expected_sidecars=initial_sidecars,
            capture_empty_wal=capture_empty_wal,
            capture_created_shm=capture_created_shm,
            wal_guard=guard,
            tables=tables,
            schema_sql=schema_sql,
            journal_mode=mode,
            schema_version=schema_version,
            user_version=user_version,
            application_id=application_id,
            expected_journal=expected_journal,
        )
    except BaseException as exc:
        try:
            if conn.in_transaction:
                conn.rollback()
        finally:
            if guard:
                lockguard.release(guard)
            conn.close()
        if isinstance(exc, SQLiteCaptureRefused):
            raise
        raise SQLiteCaptureRefused("cron_capture_sqlite_open_failed") from exc


def _capture_empty_wal(
    path: Path,
    initial_sidecars: Mapping[str, tuple[int, int]],
    dbfile: Any,
) -> tuple[int, int] | None:
    if "-wal" in initial_sidecars:
        return None
    current_wal = dbfile._stat_sqlite_sidecar_identity(path).get("-wal")
    if current_wal is None:
        return None
    try:
        wal_size = Path(str(path) + "-wal").stat().st_size
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SQLiteCaptureRefused("cron_capture_sqlite_sidecar_unavailable") from exc
    return current_wal if wal_size == 0 else None


def _capture_created_shm(
    path: Path,
    initial_sidecars: Mapping[str, tuple[int, int]],
    dbfile: Any,
) -> tuple[int, int] | None:
    if "-shm" in initial_sidecars:
        return None
    return dbfile._stat_sqlite_sidecar_identity(path).get("-shm")


def _require_empty_wal(
    path: Path,
    current_sidecars: Mapping[str, tuple[int, int]],
    expected: tuple[int, int],
) -> None:
    if current_sidecars.get("-wal") != expected:
        raise SQLiteCaptureRefused("cron_capture_sqlite_sidecar_changed")
    try:
        wal_size = Path(str(path) + "-wal").stat().st_size
    except OSError as exc:
        raise SQLiteCaptureRefused("cron_capture_sqlite_sidecar_changed") from exc
    if wal_size != 0:
        raise SQLiteCaptureRefused("cron_capture_sqlite_sidecar_changed")


@contextmanager
def open_fenced_sqlite(
    path: Path,
    specs: Mapping[str, SQLiteTableSpec],
    *,
    deadline: float,
    max_bytes: int | None = None,
) -> Iterator[SQLiteCaptureHandle]:
    handle = _open_handle(path, specs, deadline=deadline, max_bytes=max_bytes)
    try:
        yield handle
    finally:
        handle.close()


__all__ = [
    "SQLiteCaptureHandle",
    "SQLiteCaptureRefused",
    "SQLiteColumnSpec",
    "SQLiteIndexSpec",
    "SQLiteTableSpec",
    "open_fenced_sqlite",
]
