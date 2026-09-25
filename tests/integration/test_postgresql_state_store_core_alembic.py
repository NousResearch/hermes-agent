"""PostgreSQL-only Alembic core and child-head integration contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from threading import Barrier
from typing import Any, cast

import pytest

from state_store import PostgreSQLStateStoreConfig, StateStoreConfigurationError, open_state_store
from state_store_alembic import (
    CURRENT_STATE_STORE_REVISION,
    V25_CORE_REVISION,
    V26_SQLITE_IMPORT_MANIFEST_REVISION,
    upgrade_new_tenant_to_v25,
)
from state_store_alembic.semantic_catalog import _CURRENT_TABLES
from state_store_postgresql import PostgreSQLStateStore

_DSN_ENV = "HERMES_STATE_STORE_TEST_DSN"


def _config() -> dict[str, object]:
    return {
        "state_store": {
            "backend": "postgresql",
            "postgresql": {"dsn_env": _DSN_ENV, "connect_timeout_seconds": 5, "pool_max_size": 1},
        },
    }


def _open_owned_store(monkeypatch, target):
    monkeypatch.setenv(_DSN_ENV, target.dsn)
    import state_store

    # The owned fixture replaces only the canonical default binding; the
    # factory still evaluates the historic guard in fail-closed production form.
    monkeypatch.setattr(state_store, "_resolve_postgresql_tenant_schema", lambda: target.schema)
    return open_state_store(_config())


def _upgrade_target_to_v25(target) -> None:
    """Create an authentic v25-only catalog without running the child migration."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine

    engine = create_engine(target.dsn.replace("postgresql://", "postgresql+psycopg://", 1))
    try:
        with engine.begin() as connection:
            config = Config()
            config.attributes["connection"] = connection
            config.attributes["tenant_schema"] = target.schema
            with resources.as_file(resources.files("state_store_alembic")) as script_location:
                config.set_main_option("script_location", str(script_location))
                command.upgrade(config, V25_CORE_REVISION)
    finally:
        engine.dispose()


def _assert_version(target, revision: str) -> None:
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f'SELECT version_num FROM "{target.schema}".alembic_version')
        assert cursor.fetchall() == [(revision,)]


def test_owned_fixture_keeps_tenant_empty_before_production_bootstrap(postgresql_test_target):
    with postgresql_test_target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT relation.relname FROM pg_catalog.pg_class AS relation "
            "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname=%s",
            (postgresql_test_target.schema,),
        )
        assert cursor.fetchall() == []


def test_fresh_owned_tenant_has_current_head_and_no_legacy_ledger(monkeypatch, postgresql_test_target):
    store = cast(Any, _open_owned_store(monkeypatch, postgresql_test_target))
    try:
        assert CURRENT_STATE_STORE_REVISION == V26_SQLITE_IMPORT_MANIFEST_REVISION
        _assert_version(postgresql_test_target, V26_SQLITE_IMPORT_MANIFEST_REVISION)
        with store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class AS relation "
                "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace "
                "WHERE namespace.nspname = %s AND relation.relname = 'schema_migrations')",
                (store.tenant_schema,),
            )
            assert cursor.fetchone() == (False,)
    finally:
        store.close()


def test_open_state_store_rejects_historic_default_schema_even_when_empty_before_tenant_bootstrap(
    monkeypatch, postgresql_test_target,
):
    """Namespace existence covers collations and every text-search object class."""
    import importlib
    import state_store

    historic_schema = "hermes_state_store_slice"

    psycopg = importlib.import_module("psycopg")
    created_by_test = False
    with psycopg.connect(postgresql_test_target.dsn, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s)",
            (historic_schema,),
        )
        if not cursor.fetchone()[0]:
            cursor.execute(f'CREATE SCHEMA "{historic_schema}"')
            created_by_test = True

    monkeypatch.setattr(state_store, "_is_default_state_store_profile", lambda: True)
    try:
        with pytest.raises(StateStoreConfigurationError, match="formal reinitialization/cutover"):
            open_state_store(_config(), secret_lookup=lambda _name: postgresql_test_target.dsn)
    finally:
        if created_by_test:
            with psycopg.connect(postgresql_test_target.dsn, autocommit=True) as connection, connection.cursor() as cursor:
                cursor.execute(f'DROP SCHEMA IF EXISTS "{historic_schema}" CASCADE')


def _cold_tenant_opener(target, barrier: Barrier) -> PostgreSQLStateStore:
    """Construct a separate store/pool after both competing callers are ready."""
    barrier.wait(timeout=15)
    return PostgreSQLStateStore(
        PostgreSQLStateStoreConfig(dsn_env=_DSN_ENV, connect_timeout_seconds=5, pool_max_size=1),
        target.dsn,
        schema=target.schema,
    )


def test_two_independent_cold_tenant_openers_converge_on_one_head_and_catalog(postgresql_test_target):
    """The real advisory-lock bootstrap permits two simultaneous fresh-tenant openers."""
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_cold_tenant_opener, postgresql_test_target, barrier) for _ in range(2)]
        stores = [future.result(timeout=60) for future in futures]
    try:
        with postgresql_test_target.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f'SELECT version_num FROM "{postgresql_test_target.schema}".alembic_version')
            assert cursor.fetchall() == [(CURRENT_STATE_STORE_REVISION,)]
            cursor.execute(
                "SELECT relation.relname, relation.relkind FROM pg_catalog.pg_class AS relation "
                "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid=relation.relnamespace "
                "WHERE namespace.nspname=%s AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')",
                (postgresql_test_target.schema,),
            )
            actual_catalog = dict(cursor.fetchall())
        expected_catalog = {**{name: "r" for name in _CURRENT_TABLES}, "alembic_version": "r", "messages_id_seq": "S"}
        assert actual_catalog == expected_catalog
    finally:
        for store in stores:
            store.close()


def test_current_core_catalog_excludes_later_session_topic_ddl(monkeypatch, postgresql_test_target):
    """v26 session-topic DDL is intentionally reserved for the later topic branch."""
    store = cast(Any, _open_owned_store(monkeypatch, postgresql_test_target))
    try:
        with postgresql_test_target.connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (f'"{postgresql_test_target.schema}".session_topics',))
            assert cursor.fetchone() == (None,)
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_schema=%s AND table_name='messages' AND column_name='topic_id')",
                (postgresql_test_target.schema,),
            )
            assert cursor.fetchone() == (False,)
    finally:
        store.close()


def test_valid_v25_catalog_upgrades_to_the_current_head(postgresql_test_target):
    _upgrade_target_to_v25(postgresql_test_target)
    _assert_version(postgresql_test_target, V25_CORE_REVISION)
    with postgresql_test_target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class relation "
            "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname=%s AND relation.relname='sqlite_import_manifests')",
            (postgresql_test_target.schema,),
        )
        assert cursor.fetchone() == (False,)

    with postgresql_test_target.connect() as connection:
        result = upgrade_new_tenant_to_v25(connection, postgresql_test_target.schema)

    assert result.revision == V26_SQLITE_IMPORT_MANIFEST_REVISION
    _assert_version(postgresql_test_target, V26_SQLITE_IMPORT_MANIFEST_REVISION)
    with postgresql_test_target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class relation "
            "JOIN pg_catalog.pg_namespace namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname=%s AND relation.relname='sqlite_import_manifests')",
            (postgresql_test_target.schema,),
        )
        assert cursor.fetchone() == (True,)


def test_legacy_numeric_ledger_fails_closed_before_alembic_bootstrap(monkeypatch, postgresql_test_target):
    postgresql_test_target.execute(
        f'CREATE TABLE "{postgresql_test_target.schema}".schema_migrations '
        "(version integer PRIMARY KEY, applied_at double precision NOT NULL)"
    )

    with pytest.raises(StateStoreConfigurationError, match="formal reinitialization"):
        _open_owned_store(monkeypatch, postgresql_test_target)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("ALTER TABLE {schema}.sessions ALTER COLUMN hidden SET DEFAULT true", "default/generated contract"),
        ("ALTER TABLE {schema}.sessions DROP CONSTRAINT sessions_parent_session_id_fkey", "foreign-key semantics"),
        ("ALTER TABLE {schema}.session_runtime_owners DROP CONSTRAINT session_runtime_owners_fence_check", "check constraint"),
        ("ALTER TABLE {schema}.session_runtime_owners DROP CONSTRAINT session_runtime_owners_fence_check; ALTER TABLE {schema}.session_runtime_owners ADD CONSTRAINT session_runtime_owners_fence_check CHECK (fence >= 0)", "check constraint"),
        ("DROP INDEX {schema}.sessions_visibility_started_at; CREATE INDEX sessions_visibility_started_at ON {schema}.sessions (archived, hidden, started_at ASC)", "index method"),
        ("ALTER TABLE {schema}.sessions ALTER COLUMN display_name TYPE text COLLATE \"C\"", "collation differs"),
        ("ALTER SEQUENCE {schema}.messages_id_seq INCREMENT BY 2", "identity sequence ownership or options"),
        ("DROP INDEX {schema}.sessions_source_session_key; CREATE INDEX sessions_source_session_key ON {schema}.sessions (source text_pattern_ops, session_key)", "opclasses"),
        ("DROP INDEX {schema}.sessions_source_session_key; CREATE INDEX sessions_source_session_key ON {schema}.sessions (source COLLATE \"C\", session_key)", "collations"),
        ("DROP INDEX {schema}.sessions_title_unique; CREATE UNIQUE INDEX sessions_title_unique ON {schema}.sessions (title) NULLS NOT DISTINCT WHERE title IS NOT NULL", "NULLS NOT DISTINCT"),
        ("ALTER INDEX {schema}.messages_search_document_gin SET (fastupdate = off)", "reloptions"),
        ("ALTER TABLE {schema}.messages DROP CONSTRAINT messages_session_id_fkey; ALTER TABLE {schema}.messages ADD CONSTRAINT messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES {schema}.sessions(id) ON UPDATE CASCADE", "foreign-key semantics"),
        ("ALTER TABLE {schema}.messages DROP CONSTRAINT messages_session_id_fkey; ALTER TABLE {schema}.messages ADD CONSTRAINT messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES {schema}.sessions(id) MATCH FULL", "foreign-key semantics"),
        ("ALTER TABLE {schema}.messages DROP CONSTRAINT messages_session_id_fkey; ALTER TABLE {schema}.messages ADD CONSTRAINT messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES {schema}.sessions(id) DEFERRABLE INITIALLY DEFERRED", "foreign-key semantics"),
        ("CREATE INDEX unexpected_expression_index ON {schema}.sessions ((lower(id)))", "index method"),
        ("CREATE TABLE {schema}.unexpected_populated_relation (id bigint PRIMARY KEY)", "relations"),
    ),
)
def test_existing_current_catalog_drift_fails_closed_before_store_use(
    monkeypatch, postgresql_test_target, mutation: str, match: str,
):
    store = cast(Any, _open_owned_store(monkeypatch, postgresql_test_target))
    store.close()
    postgresql_test_target.execute(mutation.format(schema=f'"{postgresql_test_target.schema}"'))

    with pytest.raises(StateStoreConfigurationError, match=match):
        _open_owned_store(monkeypatch, postgresql_test_target)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("CREATE FUNCTION {schema}.foreign_function() RETURNS integer LANGUAGE sql AS 'SELECT 1'", "functions"),
        ("CREATE TRIGGER foreign_trigger BEFORE UPDATE ON {schema}.sessions FOR EACH ROW EXECUTE FUNCTION pg_catalog.suppress_redundant_updates_trigger()", "triggers"),
        ("CREATE RULE foreign_rule AS ON UPDATE TO {schema}.sessions DO ALSO NOTIFY foreign_rule_channel", "rules"),
        ("ALTER TABLE {schema}.sessions ENABLE ROW LEVEL SECURITY; CREATE POLICY foreign_policy ON {schema}.sessions USING (true)", "RLS policies"),
        ("CREATE TYPE {schema}.foreign_type AS ENUM ('foreign')", "types"),
        ("CREATE COLLATION {schema}.foreign_collation (provider = libc, locale = 'C')", "collations"),
        ("CREATE TEXT SEARCH CONFIGURATION {schema}.foreign_configuration (COPY = pg_catalog.simple)", "text-search configurations"),
    ),
)
def test_foreign_behavior_bearing_catalog_objects_fail_closed(
    monkeypatch, postgresql_test_target, mutation: str, match: str,
):
    store = cast(Any, _open_owned_store(monkeypatch, postgresql_test_target))
    store.close()
    postgresql_test_target.execute(mutation.format(schema=f'"{postgresql_test_target.schema}"'))

    with pytest.raises(StateStoreConfigurationError, match=match):
        _open_owned_store(monkeypatch, postgresql_test_target)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("ALTER TABLE {schema}.alembic_version DROP CONSTRAINT alembic_version_pkc", "metadata is malformed"),
        ("ALTER TABLE {schema}.alembic_version ALTER COLUMN version_num TYPE text", "metadata is malformed"),
        ("ALTER TABLE {schema}.alembic_version DROP CONSTRAINT alembic_version_pkc; ALTER TABLE {schema}.alembic_version ALTER COLUMN version_num DROP NOT NULL", "metadata is malformed"),
        ("ALTER TABLE {schema}.alembic_version ADD COLUMN unexpected integer", "metadata is malformed"),
        ("INSERT INTO {schema}.alembic_version (version_num) VALUES ('another_supported_revision')", "exactly one supported revision"),
    ),
)
def test_malformed_alembic_version_contract_fails_closed(
    monkeypatch, postgresql_test_target, mutation: str, match: str,
):
    store = cast(Any, _open_owned_store(monkeypatch, postgresql_test_target))
    store.close()
    postgresql_test_target.execute(mutation.format(schema=f'"{postgresql_test_target.schema}"'))

    with pytest.raises(StateStoreConfigurationError, match=match):
        _open_owned_store(monkeypatch, postgresql_test_target)
