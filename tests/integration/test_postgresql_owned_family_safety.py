"""Sentinel and source-inventory proof for migrated PostgreSQL fixture families."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

from gateway.delivery_ledger_postgresql import PostgreSQLDeliveryLedger
from state_store import open_state_store
from tests.integration.postgresql_test_target import TEST_DSN, OwnedPostgreSQLTestTarget


_FAMILY = (
    "test_postgresql_state_store_slice.py",
    "test_postgresql_session_runtime_ownership.py",
    "test_postgresql_delivery_ledger.py",
    "test_postgresql_cli_session_store.py",
    "test_postgresql_compression_rotation_acceptance.py",
    "test_postgresql_compression_coordination.py",
    "test_postgresql_phase12_fault_harness.py",
    "test_postgresql_state_store_sqlite_import.py",
    "test_postgresql_state_store_operations.py",
)
# Store-pool search_path poisoning is a deliberate lifecycle assertion: the facade
# must reset it on checkout. Catalog/data reads are also safe; direct DDL/DML is not.
_SAFE_DIRECT_EXECUTE_EXEMPTIONS = ("SET search_path TO public", "SHOW search_path", "SHOW server_version_num", "SELECT ")
_DANGEROUS_SQL = ("CREATE ", "ALTER ", "DROP ", "INSERT ", "UPDATE ", "DELETE ", "REINDEX ", "TRUNCATE ", "MERGE ", "COPY ", "GRANT ", "REVOKE ", "VACUUM ", "DO ", "CALL ")


def _psycopg():
    return importlib.import_module("psycopg")


def _public_catalog_fingerprint() -> list[tuple[str, str]]:
    """Read-only shared-namespace sentinel; this test never creates shared DDL."""
    with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT relation.relname, relation.relkind FROM pg_catalog.pg_class AS relation "
            "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname='public' ORDER BY relation.relname, relation.relkind"
        )
        return list(cursor.fetchall())


def test_owned_family_migrations_cleanup_only_owned_targets_and_leave_public_untouched(monkeypatch):
    """CLI/runtime/session/ledger constructors mutate only allocated UUID targets."""
    public_before = _public_catalog_fingerprint()
    target = OwnedPostgreSQLTestTarget(TEST_DSN).allocate()
    delivery = OwnedPostgreSQLTestTarget(TEST_DSN, prefix="hermes_delivery_ledger_tenant_").allocate()
    try:
        import state_store
        monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", TEST_DSN)
        monkeypatch.setattr(state_store, "_resolve_postgresql_tenant_schema", lambda *_args, **_kwargs: target.schema)
        config = {"state_store": {"backend": "postgresql", "postgresql": {"dsn_env": "HERMES_STATE_STORE_TEST_DSN"}}}
        store = open_state_store(config)
        store.close()
        # Session-runtime migrations live in the StateStore catalog and are covered by the same factory path.
        runtime_store = open_state_store(config)
        assert getattr(runtime_store, "_schema") == target.schema
        runtime_store.close()
        ledger = PostgreSQLDeliveryLedger(TEST_DSN, schema=delivery.schema)
        ledger.close()
        target.verify()
        delivery.verify()
    finally:
        delivery.drop()
        target.drop()
    with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT nspname FROM pg_catalog.pg_namespace "
            "WHERE nspname IN (%s, %s, %s, %s) ORDER BY nspname",
            (target.schema, target.ownership_schema, delivery.schema, delivery.ownership_schema),
        )
        assert cursor.fetchall() == []
    assert _public_catalog_fingerprint() == public_before


def _store_internal_line_numbers(source: str, tree: ast.AST) -> set[int]:
    """Line numbers covered by a ``with store._store._connection() ... as cursor:`` block.

    Store-internal connections are NOT a bypass: the store factory already performs the
    allocate/migrate/verify lifecycle over its own tenant schema, so seeding or
    forced-failure fixtures through ``store._store._connection()`` are legitimate. The
    inventory's real target is a raw ``psycopg.connect(...)`` that sidesteps
    ``OwnedPostgreSQLTestTarget.execute`` on the TEST's separately-owned target schema.
    """
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "_connection"
            for item in node.items
        ):
            continue
        for child in node.body:
            covered.update(range(child.lineno, getattr(child, "end_lineno", child.lineno) + 1))
    return covered


def test_migrated_family_has_no_direct_dangerous_postgresql_execute_calls():
    """Regression inventory: mutations require OwnedPostgreSQLTestTarget.execute."""
    root = Path(__file__).parent
    violations: list[str] = []
    for filename in _FAMILY:
        source = (root / filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
        store_internal = _store_internal_line_numbers(source, tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "execute":
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "cursor":
                continue
            if node.lineno in store_internal:
                continue
            statement = ast.get_source_segment(source, node) or ""
            normalized = statement.upper()
            if any(token in normalized for token in _DANGEROUS_SQL):
                violations.append(f"{filename}:{node.lineno}: {statement}")
            elif not any(exemption.upper() in normalized for exemption in _SAFE_DIRECT_EXECUTE_EXEMPTIONS):
                violations.append(f"{filename}:{node.lineno}: unlisted direct execute exemption: {statement}")
    assert not violations, "\n".join(violations)
