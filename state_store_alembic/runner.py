"""Programmatic PostgreSQL-only runner for the immutable core and child head.

The runtime supplies the DSN connection and tenant identity.  There is no
alembic.ini, URL fallback, SQLite path, stamping, legacy-ledger replay, or
self-repair path.
"""

from __future__ import annotations


from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any

from state_store_alembic.errors import (
    BaselineMigrationContractError,
    SchemaNotEmptyReinitializationRequired,
    StateStoreAlembicError,
    StateStoreMigrationReinitializationRequired,
    UnsupportedMigrationDialectError,
)
from state_store_alembic.migration_helpers import (
    TrustedTenantSchema,
    V25_CORE_REVISION,
    _issue_owned_target_tenant_schema,
    _issue_runtime_tenant_schema,
    require_trusted_tenant_schema,
)
from state_store_alembic.semantic_catalog import (
    V26_SQLITE_IMPORT_MANIFEST_REVISION,
    validate_current_catalog,
    validate_v25_core_catalog,
)

CURRENT_STATE_STORE_REVISION = V26_SQLITE_IMPORT_MANIFEST_REVISION

if TYPE_CHECKING:
    import psycopg


def _runtime_state_store_schema(schema_name: str) -> TrustedTenantSchema:
    """Issue the capability used by the canonical profile resolver only."""
    return _issue_runtime_tenant_schema(schema_name)


def _owned_target_state_store_schema(schema_name: str) -> TrustedTenantSchema:
    """Issue the capability for a newly allocated importer target only."""
    return _issue_owned_target_tenant_schema(schema_name)


class _BorrowedPsycopgConnection:
    """DBAPI facade that prevents SQLAlchemy from closing the store-owned connection."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


@dataclass(frozen=True, slots=True)
class BaselineMigrationResult:
    schema: TrustedTenantSchema
    revision: str


def _require_postgresql_psycopg_connection(connection: Any) -> None:
    if getattr(getattr(connection, "info", None), "vendor", None) != "PostgreSQL":
        raise UnsupportedMigrationDialectError(
            "State-store Alembic accepts only an existing psycopg PostgreSQL connection"
        )


def _require_postgresql_14(connection: Any) -> None:
    try:
        version = int(connection.exec_driver_sql("SHOW server_version_num").scalar_one())
    except (TypeError, ValueError) as exc:
        raise UnsupportedMigrationDialectError("PostgreSQL State Store could not verify its server version") from exc
    if version < 140000:
        raise UnsupportedMigrationDialectError("PostgreSQL State Store requires PostgreSQL 14 or newer")


def _relations(connection: Any, schema: TrustedTenantSchema) -> list[str]:
    return list(connection.exec_driver_sql(
        "SELECT relation.relname FROM pg_catalog.pg_class AS relation "
        "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace "
        "WHERE namespace.nspname = %s AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f') "
        "ORDER BY relation.relname",
        (schema.name,),
    ).scalars().all())


def _has_legacy_ledger(connection: Any, schema: TrustedTenantSchema) -> bool:
    return bool(connection.exec_driver_sql(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class AS relation "
        "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace "
        "WHERE namespace.nspname = %s AND relation.relname = 'schema_migrations')",
        (schema.name,),
    ).scalar_one())


def _require_version_table_contract(connection: Any, schema: TrustedTenantSchema) -> None:
    valid = connection.exec_driver_sql(
        "SELECT EXISTS ("
        "SELECT 1 FROM pg_catalog.pg_class AS relation "
        "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace "
        "JOIN pg_catalog.pg_attribute AS attribute ON attribute.attrelid = relation.oid "
        "WHERE namespace.nspname = %s AND relation.relname = 'alembic_version' "
        "AND relation.relkind = 'r' AND attribute.attname = 'version_num' "
        "AND attribute.attnum > 0 AND NOT attribute.attisdropped AND attribute.attnotnull "
        "AND format_type(attribute.atttypid, attribute.atttypmod) = 'character varying(32)' "
        "AND (SELECT count(*) FROM pg_catalog.pg_attribute AS live_column "
        "     WHERE live_column.attrelid = relation.oid AND live_column.attnum > 0 "
        "     AND NOT live_column.attisdropped) = 1 "
        "AND EXISTS (SELECT 1 FROM pg_catalog.pg_constraint AS primary_key "
        "            WHERE primary_key.conrelid = relation.oid AND primary_key.contype = 'p' "
        "            AND primary_key.conkey = ARRAY[attribute.attnum]::smallint[])"
        ")",
        (schema.name,),
    ).scalar_one()
    if not valid:
        raise BaselineMigrationContractError(
            "PostgreSQL State Store Alembic metadata is malformed; formal reinitialization is required"
        )


def _read_single_revision(connection: Any, schema: TrustedTenantSchema) -> str:
    try:
        revisions = list(connection.exec_driver_sql(
            f'SELECT version_num FROM "{schema.name}".alembic_version'
        ).scalars().all())
    except Exception as exc:
        raise BaselineMigrationContractError(
            "PostgreSQL State Store Alembic metadata is malformed; formal reinitialization is required"
        ) from exc
    if len(revisions) != 1 or revisions[0] not in {V25_CORE_REVISION, CURRENT_STATE_STORE_REVISION}:
        raise BaselineMigrationContractError(
            "PostgreSQL State Store Alembic metadata must contain exactly one supported revision"
        )
    return str(revisions[0])


def _read_exact_head(connection: Any, schema: TrustedTenantSchema) -> str:
    revision = _read_single_revision(connection, schema)
    if revision != CURRENT_STATE_STORE_REVISION:
        raise BaselineMigrationContractError(
            "PostgreSQL State Store Alembic metadata must contain exactly the current "
            "state_store_v26_sqlite_import head"
        )
    return revision


def _preflight(connection: Any, schema: TrustedTenantSchema) -> bool:
    """Classify a locked tenant before any catalog mutation."""

    if _has_legacy_ledger(connection, schema):
        raise StateStoreMigrationReinitializationRequired()
    relations = _relations(connection, schema)
    if "alembic_version" in relations:
        _require_version_table_contract(connection, schema)
        revision = _read_single_revision(connection, schema)
        if revision == V25_CORE_REVISION:
            validate_v25_core_catalog(connection, schema.name)
        return False
    if not relations:
        return True
    raise SchemaNotEmptyReinitializationRequired(
        "PostgreSQL State Store tenant schema has unknown objects; formal reinitialization is required"
    )


def upgrade_new_tenant_to_current(
    connection: "psycopg.Connection[Any]", schema: TrustedTenantSchema,
) -> BaselineMigrationResult:
    """Upgrade an empty or valid-v25 tenant to the v26 non-topic importer head.

    The tenant-scoped advisory lock, PostgreSQL 14 gate, legacy-ledger detection,
    metadata checks, and Alembic upgrade run on the connection supplied by the
    state-store owner.  No caller-controlled URL or fallback backend exists.
    """

    # Reject this before inspecting the connection, taking a lock, or issuing
    # SQL/DDL: matching strings are not runtime tenant capabilities.
    schema = require_trusted_tenant_schema(schema)
    _require_postgresql_psycopg_connection(connection)
    try:
        from alembic import command
        from alembic.config import Config
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
    except ImportError as exc:
        raise BaselineMigrationContractError(
            "PostgreSQL State Store Alembic bootstrap requires optional dependencies; "
            "install with: pip install 'hermes-agent[state-store]'"
        ) from exc

    engine = create_engine(
        "postgresql+psycopg://",
        creator=lambda: _BorrowedPsycopgConnection(connection),
        poolclass=StaticPool,
        use_native_hstore=False,
    )
    sqlalchemy_connection = None
    try:
        sqlalchemy_connection = engine.connect()
        if sqlalchemy_connection.dialect.name != "postgresql":
            raise UnsupportedMigrationDialectError("State-store Alembic requires PostgreSQL dialect support")
        try:
            with sqlalchemy_connection.begin():
                _require_postgresql_14(sqlalchemy_connection)
                sqlalchemy_connection.exec_driver_sql(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"{schema.name}:migration",),
                )
                if _preflight(sqlalchemy_connection, schema):
                    sqlalchemy_connection.exec_driver_sql(
                        f'CREATE SCHEMA IF NOT EXISTS "{schema.name}"'
                    )
                config = Config()
                config.attributes["connection"] = sqlalchemy_connection
                config.attributes["tenant_schema"] = schema
                with resources.as_file(resources.files("state_store_alembic")) as script_location:
                    config.set_main_option("script_location", str(script_location))
                    command.upgrade(config, "head")
                _require_version_table_contract(sqlalchemy_connection, schema)
                _read_exact_head(sqlalchemy_connection, schema)
                validate_current_catalog(sqlalchemy_connection, schema.name)
        except StateStoreAlembicError:
            raise
        except Exception as exc:
            raise BaselineMigrationContractError(
                "PostgreSQL State Store Alembic migration failed closed; formal reinitialization is required"
            ) from exc
    finally:
        if sqlalchemy_connection is not None:
            sqlalchemy_connection.close()
        engine.dispose()
    return BaselineMigrationResult(schema=schema, revision=CURRENT_STATE_STORE_REVISION)


def upgrade_new_tenant_to_v25(
    connection: "psycopg.Connection[Any]", schema: TrustedTenantSchema,
) -> BaselineMigrationResult:
    """Compatibility entrypoint; use :func:`upgrade_new_tenant_to_current`.

    The v25 identifier names the immutable baseline accepted as input, not the
    head produced by this function.  Keeping this wrapper avoids widening the
    migration API change while callers move to the unambiguous name.
    """
    return upgrade_new_tenant_to_current(connection, schema)
