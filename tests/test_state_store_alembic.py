"""Unit contracts for the PostgreSQL-only Alembic bootstrap boundary."""

from __future__ import annotations

from contextlib import contextmanager
import pickle
from types import SimpleNamespace

import pytest

from state_store_alembic import (
    LegacySchemaMigrationsReinitializationRequired,
    TrustedTenantSchema,
    UntrustedTenantSchemaError,
    V25_CORE_REVISION,
    upgrade_new_tenant_to_current,
    upgrade_new_tenant_to_v25,
)
from state_store_alembic.semantic_catalog import _PKS, _expected_index_signatures


def test_core_v25_revision_is_the_single_public_baseline_identifier():
    assert V25_CORE_REVISION == "state_store_v25_core"


def test_current_bootstrap_name_distinguishes_the_v25_baseline_from_v27_head():
    from state_store_alembic import CURRENT_STATE_STORE_REVISION

    assert CURRENT_STATE_STORE_REVISION == "state_store_v27_message_identity"
    assert upgrade_new_tenant_to_current is not upgrade_new_tenant_to_v25


def test_semantic_catalog_includes_alembic_version_primary_key():
    assert _PKS["alembic_version"] == ("version_num",)


def test_semantic_catalog_constructs_fresh_valid_index_signatures_in_catalog_order():
    expected = _expected_index_signatures()

    assert expected["messages_session_id_id"] == (
        "messages", "btree", (("session_id", 0), ("id", 0)), False,
        True, True, True, False, True, True, True, (), "",
    )
    assert expected["sessions_title_unique"] == (
        "sessions", "btree", (("title", 0),), True,
        True, True, True, False, True, True, True, (), "titleisnotnull",
    )
    assert expected["sessions_visibility_started_at"][2][-1] == ("started_at", 3)


def test_public_tenant_capability_constructors_and_factory_fail_closed():
    from state_store_alembic import runner

    tenant = "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"
    with pytest.raises(UntrustedTenantSchemaError, match="issued only"):
        TrustedTenantSchema(tenant)
    assert not hasattr(runner, "trusted_state_store_schema")
    assert not hasattr(__import__("state_store_alembic"), "trusted_state_store_schema")


def test_issued_tenant_capability_survives_pickle_without_admitting_forged_construction():
    """Spawn may restore issued capabilities, but constructors remain fail-closed."""
    from state_store_alembic.migration_helpers import require_trusted_tenant_schema
    from state_store_alembic.runner import _runtime_state_store_schema

    tenant = "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"
    issued = _runtime_state_store_schema(tenant)
    restored = pickle.loads(pickle.dumps(issued))

    assert type(restored) is TrustedTenantSchema
    assert restored.name == tenant
    assert require_trusted_tenant_schema(restored) is restored
    with pytest.raises(UntrustedTenantSchemaError, match="issued only"):
        TrustedTenantSchema(tenant)
    forged = str.__new__(TrustedTenantSchema, tenant)
    with pytest.raises(UntrustedTenantSchemaError, match="runtime-derived trusted tenant schema capability"):
        pickle.dumps(forged)


@pytest.mark.parametrize(
    "schema",
    (
        "hermes_state_store_tenant_0123456789abcdef0123456789abcdef",
        object(),
        str.__new__(TrustedTenantSchema, "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"),
    ),
)
def test_runner_rejects_non_capability_before_connection_lock_or_ddl(schema):
    """Invalid schema input must not inspect the supplied connection at all."""

    class NoSqlConnection:
        @property
        def info(self):
            raise AssertionError("runner inspected a connection for an invalid schema capability")

    with pytest.raises(UntrustedTenantSchemaError, match="runtime-derived trusted tenant schema capability"):
        upgrade_new_tenant_to_v25(NoSqlConnection(), schema)  # type: ignore[arg-type]


def test_former_public_schema_construction_and_forged_values_reject_before_runner_sql():
    """Neither construction nor forged str values reach the runner."""
    tenant = "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"

    class NoSqlConnection:
        @property
        def info(self):
            raise AssertionError("a forged capability reached runner SQL")

    with pytest.raises(UntrustedTenantSchemaError, match="issued only"):
        TrustedTenantSchema(tenant)
    for forged in (
        str.__new__(TrustedTenantSchema, tenant),
        type("Lookalike", (), {"name": tenant})(),
    ):
        with pytest.raises(UntrustedTenantSchemaError, match="runtime-derived trusted tenant schema capability"):
            upgrade_new_tenant_to_v25(NoSqlConnection(), forged)  # type: ignore[arg-type]


def test_pg13_version_rejection_precedes_advisory_lock_or_catalog_mutation(monkeypatch):
    """Mocked PG13 seam proves the version gate fires before any tenant mutation."""
    import sqlalchemy
    from state_store_alembic import UnsupportedMigrationDialectError
    from state_store_alembic import runner

    statements: list[str] = []

    class ScalarResult:
        def scalar_one(self):
            return "130000"

    class VersionOnlyConnection:
        dialect = SimpleNamespace(name="postgresql")

        @contextmanager
        def begin(self):
            yield self

        def exec_driver_sql(self, statement, *_args):
            statements.append(statement)
            if statement == "SHOW server_version_num":
                return ScalarResult()
            raise AssertionError(f"PG13 rejection reached a mutation/lock statement: {statement}")

        def close(self):
            return None

    class VersionOnlyEngine:
        def __init__(self):
            self.connection = VersionOnlyConnection()
            self.disposed = False

        def connect(self):
            return self.connection

        def dispose(self):
            self.disposed = True

    engine = VersionOnlyEngine()
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *_args, **_kwargs: engine)
    raw_connection = SimpleNamespace(info=SimpleNamespace(vendor="PostgreSQL"))

    from state_store import _resolve_postgresql_tenant_schema

    with pytest.raises(UnsupportedMigrationDialectError, match="requires PostgreSQL 14 or newer"):
        runner.upgrade_new_tenant_to_v25(
            raw_connection, _resolve_postgresql_tenant_schema(),  # type: ignore[arg-type]
        )

    assert statements == ["SHOW server_version_num"]
    assert engine.disposed is True


@pytest.mark.parametrize("revision", (
    V25_CORE_REVISION, "state_store_v26_sqlite_import", "state_store_v27_message_identity",
))
def test_runner_preflight_dispatches_only_the_matching_revision_catalog(monkeypatch, revision):
    from state_store_alembic import runner

    schema = runner._runtime_state_store_schema(
        "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"
    )
    checks: list[str] = []

    class RevisionConnection:
        def exec_driver_sql(self, statement):
            assert statement == f'SELECT version_num FROM "{schema.name}".alembic_version'
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [revision]))

    monkeypatch.setattr(runner, "_has_legacy_ledger", lambda *_args: False)
    monkeypatch.setattr(runner, "_relations", lambda *_args: ["alembic_version"])
    monkeypatch.setattr(runner, "_require_version_table_contract", lambda *_args: checks.append("metadata"))
    monkeypatch.setattr(runner, "validate_v25_core_catalog", lambda *_args: checks.append("v25"))
    monkeypatch.setattr(runner, "validate_v26_catalog", lambda *_args: checks.append("v26"))

    connection = RevisionConnection()
    assert runner._preflight(connection, schema) is False
    assert checks == ["metadata", *({V25_CORE_REVISION: ["v25"],
                                    "state_store_v26_sqlite_import": ["v26"]}.get(revision, []))]
    if revision == "state_store_v27_message_identity":
        assert runner._read_exact_head(connection, schema) == revision
    else:
        with pytest.raises(runner.BaselineMigrationContractError, match="state_store_v27_message_identity head"):
            runner._read_exact_head(connection, schema)


def test_v26_drift_refuses_upgrade_before_ddl_then_recovers_after_repair(monkeypatch):
    import sqlalchemy
    from alembic import command
    from state_store_alembic import runner

    schema = runner._runtime_state_store_schema(
        "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"
    )
    events: list[str] = []
    drifted = True

    class FakeConnection:
        dialect = SimpleNamespace(name="postgresql")
        revision = "state_store_v26_sqlite_import"

        @contextmanager
        def begin(self):
            yield self

        def exec_driver_sql(self, statement, *_args):
            events.append(statement)
            if statement == "SHOW server_version_num":
                return SimpleNamespace(scalar_one=lambda: "180000")
            if statement.startswith("SELECT version_num FROM"):
                return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [self.revision]))
            assert statement.startswith("SELECT pg_advisory_xact_lock(")
            return None

        def close(self):
            return None

    class FakeEngine:
        def connect(self):
            return connection

        def dispose(self):
            return None

    connection = FakeConnection()
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *_args, **_kwargs: FakeEngine())
    monkeypatch.setattr(runner, "_has_legacy_ledger", lambda *_args: False)
    monkeypatch.setattr(runner, "_relations", lambda *_args: ["alembic_version", "messages"])
    monkeypatch.setattr(runner, "_require_version_table_contract", lambda *_args: events.append("metadata"))

    def validate_v26(_connection, _schema):
        events.append("validate_v26")
        if drifted:
            raise runner.BaselineMigrationContractError("v26 catalog drift")

    def upgrade(_config, target):
        assert target == "head"
        events.append("upgrade")
        connection.revision = runner.CURRENT_STATE_STORE_REVISION

    monkeypatch.setattr(runner, "validate_v26_catalog", validate_v26)
    monkeypatch.setattr(runner, "validate_current_catalog", lambda *_args: events.append("validate_v27"))
    monkeypatch.setattr(command, "upgrade", upgrade)
    raw_connection = SimpleNamespace(info=SimpleNamespace(vendor="PostgreSQL"))

    with pytest.raises(runner.BaselineMigrationContractError, match="v26 catalog drift"):
        runner.upgrade_new_tenant_to_current(raw_connection, schema)
    assert [event for event in events if event in {"metadata", "validate_v26", "upgrade", "validate_v27"}] == [
        "metadata", "validate_v26",
    ]
    assert "upgrade" not in events
    assert not any(statement.startswith("CREATE ") for statement in events)
    assert connection.revision == "state_store_v26_sqlite_import"

    drifted = False
    result = runner.upgrade_new_tenant_to_current(raw_connection, schema)
    assert result.revision == "state_store_v27_message_identity"
    assert [event for event in events if event in {"metadata", "validate_v26", "upgrade", "validate_v27"}] == [
        "metadata", "validate_v26", "metadata", "validate_v26", "upgrade", "metadata", "validate_v27",
    ]
    assert not any(statement.startswith("CREATE ") for statement in events)


def test_public_state_store_and_operations_boundaries_reject_raw_schema_names():
    """Only the resolver's explicit TrustedTenantSchema capability crosses these APIs."""
    from postgresql_state_store_operations import PostgreSQLSandboxOperations, PostgreSQLSandboxOperationsError
    from state_store import PostgreSQLStateStoreConfig, StateStoreConfigurationError
    from state_store_postgresql import PostgreSQLStateStore

    settings = PostgreSQLStateStoreConfig(dsn_env="TEST_DSN", connect_timeout_seconds=1, pool_max_size=1)
    valid_but_unproven = "hermes_state_store_tenant_0123456789abcdef0123456789abcdef"
    forged_capability = str.__new__(TrustedTenantSchema, valid_but_unproven)
    for raw_schema in ("public", valid_but_unproven, forged_capability):
        with pytest.raises(StateStoreConfigurationError, match="runtime-derived trusted tenant schema"):
            PostgreSQLStateStore(settings, "postgresql://unused", schema=raw_schema)  # type: ignore[arg-type]
        with pytest.raises(PostgreSQLSandboxOperationsError, match="runtime-derived trusted tenant schema"):
            PostgreSQLSandboxOperations(settings, "postgresql://unused", schema=raw_schema)  # type: ignore[arg-type]


def test_owned_postgresql_fixture_supplies_the_explicit_tenant_capability():
    from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

    fixture = OwnedPostgreSQLTestTarget("postgresql://unused")
    assert isinstance(fixture.schema, TrustedTenantSchema)


def test_runtime_search_repair_contains_no_index_ddl():
    """The compatibility entrypoint can observe or reject drift, never mutate the catalog."""
    import inspect
    from state_store_postgresql import PostgreSQLStateStore

    source = inspect.getsource(PostgreSQLStateStore.rebuild_search_index)
    assert "cursor.execute" not in source


def test_legacy_ledger_error_requires_formal_reinitialization():
    error = LegacySchemaMigrationsReinitializationRequired()

    assert "schema_migrations" in str(error)
    assert "formal reinitialization" in str(error)
