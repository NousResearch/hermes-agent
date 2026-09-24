"""Safety contract for ownership-validated PostgreSQL test targets."""
from __future__ import annotations

import importlib

import pytest

from tests.integration.postgresql_test_target import (
    TEST_DSN,
    OwnedPostgreSQLTestTarget,
    PostgreSQLTestTargetOwnershipError,
)


def _psycopg():
    return importlib.import_module("psycopg")


def test_owned_target_requires_exact_marker_and_never_mutates_shared_sentinel(
    postgresql_test_target: OwnedPostgreSQLTestTarget,
) -> None:
    """A pre-existing shared schema is neither selected nor touched by a target."""
    shared = "hermes_shared_fixture_sentinel"
    with _psycopg().connect(TEST_DSN, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{shared}"')
        cursor.execute(f'CREATE TABLE IF NOT EXISTS "{shared}".sentinel (value text PRIMARY KEY)')
        cursor.execute(f'INSERT INTO "{shared}".sentinel (value) VALUES (%s) ON CONFLICT DO NOTHING', ("keep",))
    try:
        postgresql_test_target.execute(f'CREATE TABLE "{postgresql_test_target.schema}".probe (id integer)')
        postgresql_test_target.verify()
        with _psycopg().connect(TEST_DSN) as connection, connection.cursor() as cursor:
            cursor.execute(f'SELECT value FROM "{shared}".sentinel')
            assert cursor.fetchall() == [("keep",)]
    finally:
        with _psycopg().connect(TEST_DSN, autocommit=True) as connection, connection.cursor() as cursor:
            cursor.execute(f'DROP TABLE "{shared}".sentinel')
            cursor.execute(f'DROP SCHEMA "{shared}"')


def test_owned_target_fails_closed_when_marker_is_not_exact(
    postgresql_test_target: OwnedPostgreSQLTestTarget,
) -> None:
    with _psycopg().connect(TEST_DSN, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            f'UPDATE "{postgresql_test_target.schema}"."__hermes_owned_test_target" SET token=%s',
            ("tampered",),
        )
    with pytest.raises(PostgreSQLTestTargetOwnershipError, match="marker"):
        postgresql_test_target.verify()
    # Restore only so the fixture's independently verified teardown can execute.
    with _psycopg().connect(TEST_DSN, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute(
            f'UPDATE "{postgresql_test_target.schema}"."__hermes_owned_test_target" SET token=%s',
            (postgresql_test_target.token,),
        )
