"""Dedicated PostgreSQL run-idempotency store; the API server keeps its SQLite module path.

Mirrors the tenant-scoped, atomic check-and-claim semantics of
``gateway.platforms.api_server_run_idempotency.RunIdempotencyStore``.  Each operation
owns its transaction and connection; no database handle escapes this adapter.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator

from gateway.platforms.api_server_run_idempotency import (
    TERMINAL_STATUSES,
    _encode_status,
    _outcome,
    _record,
)
from hermes_constants import get_hermes_home, profile_name_for_home

_ROOT_SCHEMA = "hermes_run_idempotency"
_SCHEMA_RE = re.compile(r"^hermes_run_idempotency_tenant_[0-9a-f]{32}$")
_VERSION = 1

# Columns returned in the exact order ``_outcome``/``_record`` expect.
_SELECT_BY_KEY = (
    "SELECT fingerprint, run_id, status_json, owner_pid, owner_started, updated_at "
    "FROM run_idempotency WHERE scope=%s AND idempotency_key=%s")


class RunIdempotencyStoreConfigurationError(ValueError):
    pass


def postgresql_run_idempotency_schema() -> str:
    """Derive tenant only from canonical resolved home/profile, never adapter metadata."""
    home = get_hermes_home().resolve()
    profile = profile_name_for_home(home)
    if profile == "default":
        return _ROOT_SCHEMA
    digest = hashlib.sha256(f"{home}\0{profile}".encode()).hexdigest()[:32]
    return f"hermes_run_idempotency_tenant_{digest}"


@dataclass(frozen=True)
class RunIdempotencyPostgreSQLConfig:
    connect_timeout_seconds: int = 5
    pool_max_size: int = 4


class PostgreSQLRunIdempotencyStore:
    """Tenant-scoped, atomic PostgreSQL implementation of the run-idempotency contract."""

    RETENTION_SECONDS = 24 * 60 * 60
    ACKNOWLEDGED_RETENTION_SECONDS = 24 * 60 * 60

    @property
    def durable(self) -> bool:
        return True

    def __init__(self, dsn: str, *, schema: str | None = None,
                 settings: RunIdempotencyPostgreSQLConfig | None = None) -> None:
        try:
            self._psycopg = importlib.import_module("psycopg")
        except ImportError as exc:
            raise RunIdempotencyStoreConfigurationError(
                "PostgreSQL run-idempotency store requires psycopg") from exc
        self._schema = schema or postgresql_run_idempotency_schema()
        if self._schema != _ROOT_SCHEMA and not _SCHEMA_RE.fullmatch(self._schema):
            raise RunIdempotencyStoreConfigurationError(
                "PostgreSQL run-idempotency store received an invalid trusted tenant schema")
        self._dsn = dsn
        self._settings = settings or RunIdempotencyPostgreSQLConfig()
        self._closed = False
        connection = self._connect()
        try:
            self._migrate(connection)
        finally:
            connection.close()

    def _connect(self):
        if self._closed:
            raise RuntimeError("PostgreSQL run-idempotency store is closed")
        return self._psycopg.connect(
            self._dsn, connect_timeout=self._settings.connect_timeout_seconds, autocommit=False)

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'SET LOCAL search_path TO "{self._schema}", pg_catalog')
                yield cursor
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self, connection: Any) -> None:
        try:
            with connection.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"{self._schema}:run-idempotency-migration",))
                cur.execute("SHOW server_version_num")
                if int(cur.fetchone()[0]) < 140000:
                    raise RunIdempotencyStoreConfigurationError(
                        "PostgreSQL run-idempotency store requires PostgreSQL 14 or newer")
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._schema}.run_idempotency_schema_migrations "
                    f"(version integer PRIMARY KEY, applied_at double precision NOT NULL)")
                cur.execute(f"SELECT version FROM {self._schema}.run_idempotency_schema_migrations")
                versions = {int(row[0]) for row in cur.fetchall()}
                if versions - {_VERSION}:
                    raise RunIdempotencyStoreConfigurationError(
                        "unsupported PostgreSQL run-idempotency migration version")
                if 1 not in versions:
                    cur.execute(f"""CREATE TABLE {self._schema}.run_idempotency (
                      scope text NOT NULL,
                      idempotency_key text NOT NULL,
                      fingerprint text NOT NULL,
                      run_id text NOT NULL,
                      status_json text NOT NULL,
                      owner_pid integer NOT NULL DEFAULT 0,
                      owner_started integer NOT NULL DEFAULT 0,
                      retention_until double precision NOT NULL DEFAULT 0,
                      acknowledged_at double precision,
                      created_at double precision NOT NULL,
                      updated_at double precision NOT NULL,
                      PRIMARY KEY (scope, idempotency_key))""")
                    cur.execute(
                        f"CREATE UNIQUE INDEX run_idempotency_run_id "
                        f"ON {self._schema}.run_idempotency (run_id)")
                    cur.execute(
                        f"INSERT INTO {self._schema}.run_idempotency_schema_migrations "
                        f"VALUES (1, extract(epoch from clock_timestamp()))")
                self._validate(cur)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _validate(self, cur: Any) -> None:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema=%s AND table_name='run_idempotency'", (self._schema,))
        required = {'scope', 'idempotency_key', 'fingerprint', 'run_id', 'status_json',
                    'owner_pid', 'owner_started', 'retention_until', 'acknowledged_at',
                    'created_at', 'updated_at'}
        missing = required - {row[0] for row in cur.fetchall()}
        if missing:
            raise RunIdempotencyStoreConfigurationError(
                f"PostgreSQL run-idempotency store schema drift: missing columns {sorted(missing)}")
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname='run_idempotency_run_id'",
            (self._schema,))
        if cur.fetchone() is None:
            raise RunIdempotencyStoreConfigurationError(
                "PostgreSQL run-idempotency store schema drift: missing run_idempotency_run_id")

    def _prune_stale_terminal(self, cur: Any, now: float) -> None:
        """Prune aged replay records only once their stored run is terminal, mirroring the
        SQLite ``_prune_stale_terminal_locked`` predicate (inside the caller's transaction)."""
        cur.execute(
            "SELECT scope, idempotency_key, status_json FROM run_idempotency "
            "WHERE acknowledged_at <= %s "
            "OR (retention_until > 0 AND retention_until <= %s) "
            "OR (retention_until <= 0 AND updated_at < %s)",
            (now - self.ACKNOWLEDGED_RETENTION_SECONDS, now, now - self.RETENTION_SECONDS))
        for stale_scope, stale_key, stale_status in cur.fetchall():
            try:
                terminal = json.loads(stale_status).get("status") in TERMINAL_STATUSES
            except Exception:
                terminal = False
            if terminal:
                cur.execute(
                    "DELETE FROM run_idempotency WHERE scope=%s AND idempotency_key=%s",
                    (stale_scope, stale_key))

    def reserve(self, scope: str, key: str, fingerprint: str, run_id: str,
                status: Dict[str, Any], *, owner_pid: int = 0, owner_started: int = 0,
                retention_until: float = 0):
        """Atomically reserve a key; return ``(outcome, stored_record)``."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        encoded = _encode_status(status)
        with self._transaction() as cur:
            self._prune_stale_terminal(cur, now)
            cur.execute(
                "INSERT INTO run_idempotency (scope, idempotency_key, fingerprint, run_id, "
                "status_json, owner_pid, owner_started, retention_until, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (scope, idempotency_key) DO NOTHING",
                (scope, key, fingerprint, run_id, encoded, int(owner_pid or 0),
                 int(owner_started or 0), retention_until, now, now))
            created = cur.rowcount == 1
            if retention_until:
                cur.execute(
                    "UPDATE run_idempotency SET retention_until = GREATEST(retention_until, %s) "
                    "WHERE scope=%s AND idempotency_key=%s AND fingerprint=%s",
                    (retention_until, scope, key, fingerprint))
            if created:
                return "created", _record(run_id, encoded, owner_pid, owner_started, now) | {"status": status}
            cur.execute(_SELECT_BY_KEY, (scope, key))
            row = cur.fetchone()
            return _outcome(row, fingerprint)

    def lookup(self, scope: str, key: str, fingerprint: str, *, retention_until: float = 0):
        """Return ``missing``, ``reused`` or ``conflict`` without reserving."""
        now = time.time()
        retention_until = max(0.0, float(retention_until or 0))
        with self._transaction() as cur:
            if retention_until:
                cur.execute(
                    "UPDATE run_idempotency SET retention_until = GREATEST(retention_until, %s) "
                    "WHERE scope=%s AND idempotency_key=%s AND fingerprint=%s",
                    (retention_until, scope, key, fingerprint))
            self._prune_stale_terminal(cur, now)
            cur.execute(_SELECT_BY_KEY, (scope, key))
            row = cur.fetchone()
        return ("missing", None) if row is None else _outcome(row, fingerprint)

    def status_for_run(self, scope: str, run_id: str, *, retention_until: float = 0) -> Dict[str, Any] | None:
        """Load one durable run status inside its authenticated scope."""
        retention_until = max(0.0, float(retention_until or 0))
        with self._transaction() as cur:
            if retention_until:
                cur.execute(
                    "UPDATE run_idempotency SET retention_until = GREATEST(retention_until, %s) "
                    "WHERE scope=%s AND run_id=%s", (retention_until, scope, run_id))
            cur.execute(
                "SELECT status_json, owner_pid, owner_started, updated_at "
                "FROM run_idempotency WHERE scope=%s AND run_id=%s", (scope, run_id))
            row = cur.fetchone()
        if row is None:
            return None
        return {k: v for k, v in _record(None, *row).items() if k != "run_id"}

    def extend_retention(self, scope: str, run_id: str, until: float) -> bool:
        """Persist the latest verified recovery horizon for an active grant."""
        checked_until = max(0.0, float(until or 0))
        if not checked_until:
            return False
        with self._transaction() as cur:
            cur.execute(
                "UPDATE run_idempotency SET retention_until = GREATEST(retention_until, %s) "
                "WHERE scope=%s AND run_id=%s", (checked_until, scope, run_id))
            return cur.rowcount == 1

    def owns_run(self, scope: str, run_id: str) -> bool:
        with self._transaction() as cur:
            cur.execute(
                "SELECT 1 FROM run_idempotency WHERE scope=%s AND run_id=%s", (scope, run_id))
            return cur.fetchone() is not None

    def update_status(self, run_id: str, status: Dict[str, Any]) -> None:
        with self._transaction() as cur:
            cur.execute(
                "UPDATE run_idempotency SET status_json=%s, updated_at=%s WHERE run_id=%s",
                (_encode_status(status), time.time(), run_id))

    def close(self) -> None:
        self._closed = True
