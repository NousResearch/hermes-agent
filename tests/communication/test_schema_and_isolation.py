from __future__ import annotations

import sqlite3

import pytest

from communication_core.errors import DatabaseMissingError
from communication_core.repository import CommunicationRepository
from communication_core.schema import _apply, connect, rollback


def test_read_does_not_create_missing_database(tmp_path):
    path = tmp_path / "nested" / "communication.db"
    repository = CommunicationRepository(path)

    with pytest.raises(DatabaseMissingError):
        repository.list_accounts()

    assert not path.exists()


def test_migrations_are_reversible_and_failed_script_is_atomic(tmp_path):
    path = tmp_path / "communication.db"
    repository = CommunicationRepository(path)
    assert repository.initialize() == 2
    assert rollback(path, 1) == 1
    assert rollback(path, 0) == 0

    connection = connect(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            _apply(
                connection,
                "CREATE TABLE must_rollback(id INTEGER); INVALID SQL;",
                0,
            )
        found = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='must_rollback'"
        ).fetchone()
        assert found is None
    finally:
        connection.close()


def test_identical_external_ids_are_isolated_by_connected_account(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    first = repository.add_account(
        provider="fake",
        account_namespace="one",
        label="one",
        owner_profile="test",
    )
    second = repository.add_account(
        provider="fake",
        account_namespace="two",
        label="two",
        owner_profile="test",
    )

    first_identity, first_endpoint = repository.upsert_identity(
        connected_account_id=first["id"],
        external_id="same-external-id",
        display_name="First",
    )
    second_identity, second_endpoint = repository.upsert_identity(
        connected_account_id=second["id"],
        external_id="same-external-id",
        display_name="Second",
    )

    assert first_identity["id"] != second_identity["id"]
    assert first_endpoint["id"] != second_endpoint["id"]
    assert first_endpoint["connected_account_id"] == first["id"]
    assert second_endpoint["connected_account_id"] == second["id"]


def test_storage_trigger_rejects_cross_account_endpoint(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    first = repository.add_account(
        provider="fake", account_namespace="one", label="one", owner_profile="test"
    )
    second = repository.add_account(
        provider="fake", account_namespace="two", label="two", owner_profile="test"
    )
    identity, _ = repository.upsert_identity(
        connected_account_id=first["id"], external_id="contact", display_name="Contact"
    )

    with repository.transaction() as connection, pytest.raises(
        sqlite3.IntegrityError, match="endpoint account/identity mismatch"
    ):
        connection.execute(
            """INSERT INTO contact_endpoints(
                   id, connected_account_id, platform_identity_id, created_at, updated_at
               ) VALUES ('bad', ?, ?, 'now', 'now')""",
            (second["id"], identity["id"]),
        )
