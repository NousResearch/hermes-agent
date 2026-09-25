"""Safety contract for ownership-validated PostgreSQL test targets."""
from __future__ import annotations

import importlib
from threading import Event, Thread

import pytest

from tests.integration.postgresql_test_target import (
    TEST_DSN,
    OwnedPostgreSQLTestTarget,
    PostgreSQLTestTargetOwnershipError,
)


def _psycopg():
    return importlib.import_module("psycopg")


def _public_catalog_fingerprint() -> list[tuple[str, str]]:
    """Read-only sentinel for the pre-existing shared ``public`` namespace."""
    with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT relation.relname, relation.relkind FROM pg_catalog.pg_class AS relation "
            "JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid=relation.relnamespace "
            "WHERE namespace.nspname='public' ORDER BY relation.relname, relation.relkind"
        )
        return list(cursor.fetchall())


def test_owned_target_cleanup_is_exact_and_keeps_public_untouched() -> None:
    """Fixture DDL lives only in a UUID target and its UUID ownership companion."""
    public_before = _public_catalog_fingerprint()
    target = OwnedPostgreSQLTestTarget(TEST_DSN).allocate()
    try:
        target.execute(f'CREATE TABLE "{target.schema}".probe (id integer)')
        target.verify()
    finally:
        target.drop()

    with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT nspname FROM pg_catalog.pg_namespace WHERE nspname IN (%s, %s) ORDER BY nspname",
            (target.schema, target.ownership_schema),
        )
        assert cursor.fetchall() == []
    assert _public_catalog_fingerprint() == public_before


def _schema_pair(target: OwnedPostgreSQLTestTarget) -> set[str]:
    with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT nspname FROM pg_catalog.pg_namespace WHERE nspname IN (%s, %s)",
            (target.schema, target.ownership_schema),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def test_owned_target_drop_fails_closed_and_preserves_both_schemas_on_changed_marker() -> None:
    """A changed marker prevents the one-transaction teardown from dropping either schema."""
    target = OwnedPostgreSQLTestTarget(
        TEST_DSN, identity="1" * 32, token="expected-marker-token"
    ).allocate()
    try:
        with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
            cursor.execute(
                f'UPDATE "{target.ownership_schema}".__hermes_owned_postgresql_test_target '
                "SET token=%s WHERE schema_name=%s",
                ("changed-marker-token", target.schema),
            )
            connection.commit()

        with pytest.raises(PostgreSQLTestTargetOwnershipError, match="marker"):
            target.drop()
        assert _schema_pair(target) == {str(target.schema), target.ownership_schema}
    finally:
        # Restore the durable proof solely to clean up this test's owned pair.
        with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
            cursor.execute(
                f'UPDATE "{target.ownership_schema}".__hermes_owned_postgresql_test_target '
                "SET token=%s WHERE schema_name=%s",
                (target.token, target.schema),
            )
            connection.commit()
        target.drop()


def test_owned_target_drop_removes_both_schemas_after_same_transaction_verification() -> None:
    target = OwnedPostgreSQLTestTarget(
        TEST_DSN, identity="2" * 32, token="valid-marker-token"
    ).allocate()
    target.drop()
    assert _schema_pair(target) == set()


def test_owned_target_drop_locks_marker_across_verification_and_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker mutation cannot interleave after destructive verification."""
    target = OwnedPostgreSQLTestTarget(
        TEST_DSN, identity="3" * 32, token="locked-marker-token"
    ).allocate()
    marker_locked = Event()
    continue_drop = Event()
    drop_errors: list[BaseException] = []
    original_verify_cursor = target._verify_cursor

    def verify_then_pause(cursor, *, lock_marker: bool = False) -> None:
        original_verify_cursor(cursor, lock_marker=lock_marker)
        if lock_marker:
            marker_locked.set()
            assert continue_drop.wait(timeout=5)

    def drop_target() -> None:
        try:
            target.drop()
        except BaseException as exc:  # surfaced in the owning test thread
            drop_errors.append(exc)

    monkeypatch.setattr(target, "_verify_cursor", verify_then_pause)
    drop_thread = Thread(target=drop_target)
    drop_thread.start()
    try:
        assert marker_locked.wait(timeout=5)
        with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '250ms'")
            with pytest.raises(_psycopg().errors.LockNotAvailable):
                cursor.execute(
                    f'UPDATE "{target.ownership_schema}".__hermes_owned_postgresql_test_target '
                    "SET token=%s WHERE schema_name=%s",
                    ("changed-marker-token", target.schema),
                )
        assert _schema_pair(target) == {str(target.schema), target.ownership_schema}
    finally:
        continue_drop.set()
        drop_thread.join(timeout=5)
        if _schema_pair(target):
            target.drop()
    assert not drop_thread.is_alive()
    assert drop_errors == []
    assert _schema_pair(target) == set()


def test_owned_target_fails_closed_when_marker_is_not_exact(
    postgresql_test_target: OwnedPostgreSQLTestTarget,
) -> None:
    postgresql_test_target.execute(
        f'UPDATE "{postgresql_test_target.ownership_schema}".__hermes_owned_postgresql_test_target '
        "SET token=%s WHERE schema_name=%s",
        ("tampered", postgresql_test_target.schema),
    )
    with pytest.raises(PostgreSQLTestTargetOwnershipError, match="marker"):
        postgresql_test_target.verify()
    # Rebind this test's local capability so the fixture can perform verified teardown.
    postgresql_test_target.token = "tampered"
    postgresql_test_target.verify()


def test_owned_target_execute_rejects_shared_and_unqualified_mutation(
    postgresql_test_target: OwnedPostgreSQLTestTarget,
) -> None:
    """The raw SQL fixture cannot escape into a shared or search-path target."""
    target = postgresql_test_target
    public_before = _public_catalog_fingerprint()
    target.execute(f'CREATE TABLE "{target.schema}".fixture_probe (id integer PRIMARY KEY)')

    for statement, match in (
        ("DROP SCHEMA public CASCADE", "schema/database"),
        ("DELETE FROM pg_catalog.pg_class", "schema-qualified"),
        ("INSERT INTO fixture_probe (id) VALUES (1)", "schema-qualified"),
        ("SET search_path TO public", "schema/database"),
    ):
        with pytest.raises(PostgreSQLTestTargetOwnershipError, match=match):
            target.execute(statement)

    target.execute(f'INSERT INTO "{target.schema}".fixture_probe (id) VALUES (1)')
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f'SELECT id FROM "{target.schema}".fixture_probe')
        assert cursor.fetchall() == [(1,)]
    assert _public_catalog_fingerprint() == public_before


@pytest.mark.parametrize(
    "statement",
    (
        'COPY "{schema}".fixture_probe TO PROGRAM \'true\'',
        'COPY "{schema}".fixture_probe FROM PROGRAM \'printf 1\'',
        "DO $$ BEGIN EXECUTE 'DELETE FROM \\\"{schema}\\\".fixture_probe'; END $$",
        "EXECUTE 'DELETE FROM \\\"{schema}\\\".fixture_probe'",
        "CREATE FUNCTION \"{schema}\".dynamic_fixture() RETURNS void LANGUAGE plpgsql AS $$ "
        "BEGIN EXECUTE 'DELETE FROM \\\"{schema}\\\".fixture_probe'; END $$",
    ),
)
def test_owned_target_execute_rejects_copy_and_dynamic_sql_even_when_owned(
    postgresql_test_target: OwnedPostgreSQLTestTarget, statement: str,
) -> None:
    """Owned relation text cannot make program or dynamic SQL safe for this fixture."""
    target = postgresql_test_target
    target.execute(f'CREATE TABLE "{target.schema}".fixture_probe (id integer PRIMARY KEY)')
    with pytest.raises(PostgreSQLTestTargetOwnershipError, match="COPY or dynamic SQL"):
        target.execute(statement.format(schema=target.schema))


def test_owned_target_execute_allows_owned_ddl_with_non_dynamic_keywords(
    postgresql_test_target: OwnedPostgreSQLTestTarget,
) -> None:
    """Keyword occurrences in owned DDL are not standalone dynamic commands."""
    target = postgresql_test_target
    target.execute(f'CREATE TABLE "{target.schema}".fixture_probe (id integer PRIMARY KEY)')
    target.execute(
        f'CREATE FUNCTION "{target.schema}".fixture_trigger() RETURNS trigger LANGUAGE plpgsql AS $$ '
        "BEGIN RETURN NEW; END $$"
    )
    target.execute(
        f'CREATE TRIGGER fixture_trigger BEFORE UPDATE ON "{target.schema}".fixture_probe '
        f'FOR EACH ROW EXECUTE FUNCTION "{target.schema}".fixture_trigger()'
    )
    target.execute(
        f'CREATE RULE fixture_rule AS ON UPDATE TO "{target.schema}".fixture_probe '
        "DO ALSO NOTIFY fixture_rule_channel"
    )
    target.execute(
        f'CREATE TEXT SEARCH CONFIGURATION "{target.schema}".fixture_configuration '
        "(COPY = pg_catalog.simple)"
    )
