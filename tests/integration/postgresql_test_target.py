"""Ownership-validated disposable PostgreSQL targets for integration tests.

The marker capability is intentionally required for every destructive operation.
A UUID-shaped name alone is not proof that the target belongs to this test.
"""
from __future__ import annotations

import importlib
import re
import secrets
import uuid
from dataclasses import dataclass, field
from collections.abc import Generator, Iterable
from typing import Any

import pytest

TEST_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SCHEMA_PREFIX = "hermes_state_store_tenant_"
_DELIVERY_PREFIX = "hermes_delivery_ledger_tenant_"
_ASYNC_PREFIX = "hermes_async_delegation_tenant_"
_RUN_IDEMPOTENCY_PREFIX = "hermes_run_idempotency_tenant_"
_MARKER_TABLE = "__hermes_owned_test_target"
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class PostgreSQLTestTargetOwnershipError(RuntimeError):
    """Raised instead of mutating a target that this fixture cannot prove it owns."""


def _psycopg() -> Any:
    return importlib.import_module("psycopg")


def _quote(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise PostgreSQLTestTargetOwnershipError(f"unsafe PostgreSQL identifier: {identifier!r}")
    return f'"{identifier}"'


@dataclass
class OwnedPostgreSQLTestTarget:
    """A per-test UUID schema whose destructive lifecycle requires its marker."""

    dsn: str
    prefix: str = _SCHEMA_PREFIX
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    _closed: bool = False

    @property
    def schema(self) -> str:
        return f"{self.prefix}{self.identity}"

    def allocate(self) -> "OwnedPostgreSQLTestTarget":
        schema = _quote(self.schema)
        marker = _quote(_MARKER_TABLE)
        with _psycopg().connect(self.dsn, autocommit=True) as connection, connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
            cursor.execute(f"CREATE TABLE {schema}.{marker} (token text PRIMARY KEY, creator_scope text NOT NULL)")
            cursor.execute(
                f"INSERT INTO {schema}.{marker} (token, creator_scope) VALUES (%s, %s)",
                (self.token, f"pytest:{self.identity}"),
            )
        return self

    def verify(self) -> None:
        if self._closed or self.prefix not in {_SCHEMA_PREFIX, _DELIVERY_PREFIX, _ASYNC_PREFIX, _RUN_IDEMPOTENCY_PREFIX} or len(self.identity) != 32:
            raise PostgreSQLTestTargetOwnershipError("target is not an active fixture-created UUID target")
        try:
            uuid.UUID(hex=self.identity)
        except ValueError as exc:
            raise PostgreSQLTestTargetOwnershipError("target has an invalid UUID identity") from exc
        schema, marker = _quote(self.schema), _quote(_MARKER_TABLE)
        with _psycopg().connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT token, creator_scope FROM {schema}.{marker}")
            rows = cursor.fetchall()
        if rows != [(self.token, f"pytest:{self.identity}")]:
            raise PostgreSQLTestTargetOwnershipError("target marker is absent, changed, or non-singleton")

    def connect(self, **kwargs: Any) -> Any:
        self.verify()
        return _psycopg().connect(self.dsn, **kwargs)

    def execute(self, statement: str, parameters: Iterable[Any] | None = None) -> None:
        """Run a fault-injection mutation only after marker verification.

        Test SQL must schema-qualify relations; this helper rejects search_path
        changes so tests cannot redirect a target after construction.
        """
        if re.search(r"\bSET(?:\s+LOCAL)?\s+search_path\b", statement, re.IGNORECASE):
            raise PostgreSQLTestTargetOwnershipError("test target never permits post-hoc search_path routing")
        self.verify()
        with _psycopg().connect(self.dsn, autocommit=True) as connection, connection.cursor() as cursor:
            cursor.execute(statement, tuple(parameters or ()))

    def drop(self) -> None:
        """Drop only this exact marked schema; fail closed on any mismatch."""
        self.verify()
        schema = _quote(self.schema)
        with _psycopg().connect(self.dsn, autocommit=True) as connection, connection.cursor() as cursor:
            # Re-read in the same destructive connection to close TOCTOU gaps.
            marker = _quote(_MARKER_TABLE)
            cursor.execute(f"SELECT token, creator_scope FROM {schema}.{marker}")
            if cursor.fetchall() != [(self.token, f"pytest:{self.identity}")]:
                raise PostgreSQLTestTargetOwnershipError("marker changed before teardown")
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")
        self._closed = True

    def reset(self) -> None:
        """Recreate this exact marked target after marker-validated destruction."""
        self.drop()
        self._closed = False
        self.allocate()


@pytest.fixture
def postgresql_test_target() -> Generator[OwnedPostgreSQLTestTarget, None, None]:
    target = OwnedPostgreSQLTestTarget(TEST_DSN).allocate()
    try:
        yield target
    finally:
        target.drop()


@pytest.fixture
def postgresql_delivery_target() -> OwnedPostgreSQLTestTarget:
    target = OwnedPostgreSQLTestTarget(TEST_DSN, prefix=_DELIVERY_PREFIX).allocate()
    try:
        yield target
    finally:
        target.drop()


@pytest.fixture
def postgresql_async_delegation_target() -> OwnedPostgreSQLTestTarget:
    target = OwnedPostgreSQLTestTarget(TEST_DSN, prefix=_ASYNC_PREFIX).allocate()
    try:
        yield target
    finally:
        target.drop()


@pytest.fixture
def postgresql_run_idempotency_target() -> OwnedPostgreSQLTestTarget:
    target = OwnedPostgreSQLTestTarget(TEST_DSN, prefix=_RUN_IDEMPOTENCY_PREFIX).allocate()
    try:
        yield target
    finally:
        target.drop()
