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
_OWNERSHIP_SCHEMA_PREFIX = "hermes_owned_pg_fixture_"
_OWNERSHIP_MARKER_TABLE = "__hermes_owned_postgresql_test_target"
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
_MUTABLE_SQL = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|MERGE|TRUNCATE|CREATE|ALTER|DROP|REINDEX|COPY|GRANT|REVOKE|VACUUM|DO|CALL)\b",
    re.IGNORECASE,
)
_UNSAFE_GLOBAL_SQL = re.compile(
    r"\b(?:CREATE|ALTER|DROP)\s+(?:DATABASE|SCHEMA)\b|^\s*(?:SET|RESET|DISCARD)\b",
    re.IGNORECASE,
)
_UNSAFE_DYNAMIC_SQL = re.compile(
    r"^\s*(?:COPY|DO)\b|\bEXECUTE\b(?!\s+FUNCTION\b)",
    re.IGNORECASE,
)
_DML_RELATION = re.compile(
    r"^\s*(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO|TRUNCATE(?:\s+TABLE)?)\s+"
    r"(?:ONLY\s+)?(?P<relation>[^\s(]+)",
    re.IGNORECASE,
)
_SCHEMA_REFERENCE = re.compile(r'(?<![\w"])(?P<schema>"?[a-z_][a-z0-9_]*"?)\s*\.', re.IGNORECASE)


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
    """A per-test UUID schema whose owned companion marker authorizes teardown.

    The tenant itself remains empty after allocation so production bootstrap sees
    no test marker and can use its strict empty-schema preflight unchanged.
    """

    dsn: str
    prefix: str = _SCHEMA_PREFIX
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    _closed: bool = False

    @property
    def schema(self) -> str:
        schema = f"{self.prefix}{self.identity}"
        # State-store constructors accept an explicit trusted-schema capability,
        # while raw SQL fixtures still need normal string formatting.  A str
        # subclass satisfies both without granting arbitrary caller strings.
        if self.prefix == _SCHEMA_PREFIX:
            from state_store_alembic.migration_helpers import TrustedTenantSchema

            return TrustedTenantSchema(schema)
        return schema

    @property
    def ownership_schema(self) -> str:
        """A distinct, UUID-scoped companion namespace for fixture-only ownership proof."""
        return f"{_OWNERSHIP_SCHEMA_PREFIX}{self.identity}"

    def allocate(self) -> "OwnedPostgreSQLTestTarget":
        schema = _quote(self.schema)
        ownership_schema = _quote(self.ownership_schema)
        marker = _quote(_OWNERSHIP_MARKER_TABLE)
        with _psycopg().connect(self.dsn) as connection, connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
            cursor.execute(f"CREATE SCHEMA {ownership_schema}")
            cursor.execute(
                f"CREATE TABLE {ownership_schema}.{marker} "
                "(schema_name text PRIMARY KEY, token text NOT NULL, creator_scope text NOT NULL)"
            )
            cursor.execute(
                f"INSERT INTO {ownership_schema}.{marker} (schema_name, token, creator_scope) VALUES (%s, %s, %s)",
                (self.schema, self.token, f"pytest:{self.identity}"),
            )
            connection.commit()
        return self

    def _verify_identity(self) -> None:
        if self._closed or self.prefix not in {_SCHEMA_PREFIX, _DELIVERY_PREFIX, _ASYNC_PREFIX, _RUN_IDEMPOTENCY_PREFIX} or len(self.identity) != 32:
            raise PostgreSQLTestTargetOwnershipError("target is not an active fixture-created UUID target")
        try:
            uuid.UUID(hex=self.identity)
        except ValueError as exc:
            raise PostgreSQLTestTargetOwnershipError("target has an invalid UUID identity") from exc

    def _verify_cursor(self, cursor: Any, *, lock_marker: bool = False) -> None:
        """Verify this target's complete durable proof on an existing transaction."""
        ownership_schema = _quote(self.ownership_schema)
        marker = _quote(_OWNERSHIP_MARKER_TABLE)
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s), "
            "EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s)",
            (self.schema, self.ownership_schema),
        )
        target_exists, ownership_exists = cursor.fetchone()
        if not target_exists or not ownership_exists:
            raise PostgreSQLTestTargetOwnershipError("target or ownership companion schema is absent")
        try:
            cursor.execute(
                f"SELECT token, creator_scope FROM {ownership_schema}.{marker} WHERE schema_name=%s"
                + (" FOR UPDATE" if lock_marker else ""),
                (self.schema,),
            )
            rows = cursor.fetchall()
        except Exception as exc:
            raise PostgreSQLTestTargetOwnershipError("target ownership marker is absent") from exc
        if rows != [(self.token, f"pytest:{self.identity}")]:
            raise PostgreSQLTestTargetOwnershipError("target marker is absent, changed, or non-singleton")

    def verify(self) -> None:
        self._verify_identity()
        with _psycopg().connect(self.dsn) as connection, connection.cursor() as cursor:
            self._verify_cursor(cursor)

    def connect(self, **kwargs: Any) -> Any:
        self.verify()
        return _psycopg().connect(self.dsn, **kwargs)

    def _assert_safe_fixture_sql(self, statement: str) -> None:
        """Allow only one explicitly owned, schema-qualified test mutation."""
        normalized = statement.strip()
        if not normalized:
            raise PostgreSQLTestTargetOwnershipError("test target requires a SQL statement")
        # Split only ordinary test SQL.  Comments are denied so each component has
        # an unambiguous target; every component is validated independently.
        if "--" in normalized or "/*" in normalized:
            raise PostgreSQLTestTargetOwnershipError("test target permits uncommented SQL only")
        statements = [part.strip() for part in normalized.split(";") if part.strip()]
        if not statements:
            raise PostgreSQLTestTargetOwnershipError("test target requires a SQL statement")
        allowed_schemas = {str(self.schema), self.ownership_schema}
        for component in statements:
            if _UNSAFE_GLOBAL_SQL.search(component):
                raise PostgreSQLTestTargetOwnershipError(
                    "test target never permits schema/database or connection-context mutation"
                )
            # Standalone COPY and DO commands can execute an external program
            # or defer the target relation to runtime.  Dynamic EXECUTE has the
            # same problem, but CREATE TRIGGER ... EXECUTE FUNCTION is static
            # schema-qualified DDL and remains an intentional test capability.
            if _UNSAFE_DYNAMIC_SQL.search(component):
                raise PostgreSQLTestTargetOwnershipError(
                    "test target never permits COPY or dynamic SQL execution"
                )
            if not _MUTABLE_SQL.search(component):
                continue
            referenced_schemas = {
                match.group("schema").strip('"')
                for match in _SCHEMA_REFERENCE.finditer(component)
            }
            if any(schema not in allowed_schemas | {"pg_catalog"} for schema in referenced_schemas):
                raise PostgreSQLTestTargetOwnershipError(
                    "test target mutation references a schema not owned by this invocation"
                )
            if not (referenced_schemas & allowed_schemas):
                raise PostgreSQLTestTargetOwnershipError(
                    "test target mutable SQL requires an explicitly schema-qualified owned relation"
                )
            for match in _DML_RELATION.finditer(component):
                relation = match.group("relation")
                if "." not in relation or relation.split(".", 1)[0].strip('"') not in allowed_schemas:
                    raise PostgreSQLTestTargetOwnershipError(
                        "test target mutable SQL requires an explicitly schema-qualified owned relation"
                    )

    def execute(self, statement: str, parameters: Iterable[Any] | None = None) -> None:
        """Run a single ownership-bounded fault-injection mutation.

        The helper establishes its own target-only search path before execution,
        but mutable SQL still has to name an owned schema explicitly.  This makes
        accidental unqualified DML and shared-schema DDL fail closed even when a
        caller tries to rely on connection defaults.
        """
        self._assert_safe_fixture_sql(statement)
        self.verify()
        with _psycopg().connect(self.dsn, autocommit=True) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.set_config('search_path', %s, false)",
                (f'{_quote(str(self.schema))}, pg_catalog',),
            )
            cursor.execute(statement, tuple(parameters or ()))

    def drop(self) -> None:
        """Atomically drop only this exact marker-owned schema pair."""
        self._verify_identity()
        schema = _quote(self.schema)
        ownership_schema = _quote(self.ownership_schema)
        # PostgreSQL schema DDL is transactional.  Verify and drop both namespaces
        # through one connection/transaction so a changed marker or either DROP
        # failure leaves the pair intact for reconciliation.
        with _psycopg().connect(self.dsn) as connection, connection.cursor() as cursor:
            # Lock the exact durable proof until the two destructive DDL statements
            # commit. A concurrent marker mutation cannot cross verification and
            # turn a verified target into a different target before teardown.
            self._verify_cursor(cursor, lock_marker=True)
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")
            cursor.execute(f"DROP SCHEMA {ownership_schema} CASCADE")
            connection.commit()
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
