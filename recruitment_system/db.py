"""MySQL querying for RecruitmentSystem."""

from __future__ import annotations

import queue
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from .config import RecruitmentDBConfig
from .models import QueryResult


class RecruitmentQueryError(RuntimeError):
    """Base error for RecruitmentSystem query failures."""


class DatabaseConfigurationError(RecruitmentQueryError):
    """Raised when database configuration or dependencies are missing."""


class DatabaseExecutionError(RecruitmentQueryError):
    """Raised when a SQL query fails."""


class QueryExecutor(Protocol):
    def query(self, sql: str, params: dict[str, Any] | None = None) -> QueryResult:
        ...


@dataclass
class _PooledConnection:
    conn: Any
    created_at: float


class MySQLQueryClient:
    def __init__(self, config: RecruitmentDBConfig):
        self.config = config
        self._pool: queue.Queue[_PooledConnection] = queue.Queue(maxsize=max(0, config.max_idle_conns))
        self._open_connections = 0

    @classmethod
    def from_env(cls) -> "MySQLQueryClient":
        return cls(RecruitmentDBConfig.from_env())

    def health_check(self) -> tuple[bool, str]:
        try:
            result = self.query("SELECT 1 AS ok", {})
            ok = bool(result.rows and result.rows[0].get("ok") == 1)
            return ok, "ok" if ok else "unexpected health check result"
        except RecruitmentQueryError as exc:
            return False, str(exc)

    def query(self, sql: str, params: dict[str, Any] | None = None) -> QueryResult:
        if not self.config.is_complete():
            missing = ", ".join(self.config.missing_keys())
            raise DatabaseConfigurationError(f"RecruitmentSystem MySQL config is incomplete: {missing}")

        start = time.perf_counter()
        with self._connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params or {})
                    rows = cursor.fetchall()
            except Exception as exc:
                raise DatabaseExecutionError(f"MySQL query failed: {type(exc).__name__}: {exc}") from exc
        duration_ms = (time.perf_counter() - start) * 1000
        return QueryResult(rows=[dict(row) for row in rows], duration_ms=duration_ms)

    def describe_schema(self, tables: list[str]) -> QueryResult:
        placeholders = ", ".join(f"%(table_{idx})s" for idx, _ in enumerate(tables))
        params = {f"table_{idx}": table.split(".")[-1] for idx, table in enumerate(tables)}
        sql = (
            "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT "
            "FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() "
            f"AND TABLE_NAME IN ({placeholders}) "
            "ORDER BY TABLE_NAME, ORDINAL_POSITION LIMIT 500"
        )
        return self.query(sql, params)

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        pooled = self._get_connection()
        keep = True
        try:
            yield pooled.conn
        except Exception:
            keep = False
            self._close_connection(pooled)
            raise
        finally:
            if keep:
                self._return_connection(pooled)

    def _get_connection(self) -> _PooledConnection:
        while True:
            try:
                pooled = self._pool.get_nowait()
            except queue.Empty:
                break
            if not self._expired(pooled):
                return pooled
            self._close_connection(pooled)

        if self._open_connections >= self.config.max_open_conns > 0:
            try:
                pooled = self._pool.get(timeout=self.config.connect_timeout_seconds)
            except queue.Empty as exc:
                raise DatabaseExecutionError("MySQL connection pool exhausted") from exc
            if not self._expired(pooled):
                return pooled
            self._close_connection(pooled)

        conn = self._connect()
        self._open_connections += 1
        return _PooledConnection(conn=conn, created_at=time.monotonic())

    def _return_connection(self, pooled: _PooledConnection) -> None:
        if self._expired(pooled):
            self._close_connection(pooled)
            return
        try:
            self._pool.put_nowait(pooled)
        except queue.Full:
            self._close_connection(pooled)

    def _expired(self, pooled: _PooledConnection) -> bool:
        lifetime = self.config.conn_max_lifetime_seconds
        return lifetime > 0 and time.monotonic() - pooled.created_at > lifetime

    def _close_connection(self, pooled: _PooledConnection) -> None:
        try:
            pooled.conn.close()
        except Exception:
            pass
        self._open_connections = max(0, self._open_connections - 1)

    def _connect(self) -> Any:
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            raise DatabaseConfigurationError(
                "PyMySQL is required for RecruitmentSystem MySQL queries. "
                "Install with: pip install 'hermes-agent[recruitment]'"
            ) from exc

        try:
            return pymysql.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                connect_timeout=self.config.connect_timeout_seconds,
                read_timeout=self.config.query_timeout_seconds,
                write_timeout=self.config.query_timeout_seconds,
                charset="utf8mb4",
                cursorclass=DictCursor,
                autocommit=True,
            )
        except Exception as exc:
            raise DatabaseExecutionError(f"MySQL connection failed: {type(exc).__name__}: {exc}") from exc
