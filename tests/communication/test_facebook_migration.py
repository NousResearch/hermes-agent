from __future__ import annotations

import hashlib
import sqlite3

from communication_core.migrations import FacebookMigrationBridge
from communication_core.adapters import FacebookCommunicationAdapter
from communication_core.repository import CommunicationRepository
from communication_core.service import CommunicationService
from universal_browser_manager.facebook_core.schema import create_schema


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_database(path):
    connection = sqlite3.connect(path)
    try:
        create_schema(connection, applied_at="2026-01-01T00:00:00Z")
        connection.execute(
            "INSERT OR REPLACE INTO facebook_settings(key, value) VALUES ('write_actions_enabled', '0')"
        )
        friend_id = connection.execute(
            """INSERT INTO friends(canonical_key, name, profile_url, thread_url, added_at, updated_at)
               VALUES ('facebook:user:42', 'Fixture Person',
                       'https://facebook.invalid/42',
                       'https://facebook.invalid/messages/t/42',
                       '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z')"""
        ).lastrowid
        connection.execute(
            """INSERT INTO messages(
                   friend_id, message_key, message_id, sender_name, message_text,
                   timestamp, direction, text, sent_at, source_system, source_record_id
               ) VALUES (?, 'message-key', 'message-id', 'Fixture Person', 'hello',
                         '2026-01-03T00:00:00Z', 'received', 'hello',
                         '2026-01-03T00:00:00Z', 'messenger_dom', 'thread-42')""",
            (friend_id,),
        )
        connection.execute(
            """INSERT INTO interactions(
                   interaction_key, friend_id, type, details, interacted_at,
                   source_system, source_record_id
               ) VALUES ('interaction-key', ?, 'profile_update', 'fixture',
                         '2026-01-04T00:00:00Z', 'fixture', 'profile-42')""",
            (friend_id,),
        )
        connection.execute(
            """INSERT INTO outreach_campaigns(
                   friend_id, goal, relationship_type, current_stage, pending_draft
               ) VALUES (?, 'follow up', 'friend', 'review', 'legacy draft')""",
            (friend_id,),
        )
        connection.execute(
            """INSERT INTO birthday_wishes(
                   friend_name, profile_url, proposed_text, status, created_at
               ) VALUES ('Fixture Person', 'https://facebook.invalid/42',
                         'legacy birthday draft', 'pending', '2026-01-05T00:00:00Z')"""
        )
        connection.execute(
            """INSERT INTO write_approvals(
                   token_hash, actor, action, recipient_key, payload_sha256,
                   created_at, expires_at
               ) VALUES ('legacy-token-hash', 'legacy', 'message.send', '42',
                         'legacy-payload-hash', '2026-01-01T00:00:00Z',
                         '2026-01-02T00:00:00Z')"""
        )
        connection.execute(
            """INSERT INTO facebook_write_outbox(
                   idempotency_key, action, recipient_key, payload, payload_sha256,
                   permission_tier, status, available_at, created_at, authorized_at
               ) VALUES ('legacy-outbox', 'message.send', '42', 'legacy payload',
                         'legacy-payload-hash', 1, 'pending',
                         '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                         '2026-01-01T00:00:00Z')"""
        )
        connection.commit()
    finally:
        connection.close()


def test_facebook_migration_is_readonly_idempotent_reconciled_and_reversible(tmp_path):
    source = tmp_path / "facebook.db"
    _legacy_database(source)
    source_before = _file_hash(source)

    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    first_account = repository.add_account(
        provider="facebook", account_namespace="first", label="first", owner_profile="test"
    )
    second_account = repository.add_account(
        provider="facebook", account_namespace="second", label="second", owner_profile="test"
    )
    bridge = FacebookMigrationBridge(repository)

    first = bridge.migrate(source, first_account["id"])
    replay = bridge.migrate(source, first_account["id"])
    second = bridge.migrate(source, second_account["id"])

    assert first["status"] == "succeeded"
    assert first["reconciliation"]["ok"] is True
    assert replay["idempotent_replay"] is True
    assert replay["run_id"] == first["run_id"]
    assert second["run_id"] != first["run_id"]
    assert second["reconciliation"]["ok"] is True
    assert _file_hash(source) == source_before
    with sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True) as connection:
        assert connection.execute(
            "SELECT value FROM facebook_settings WHERE key = 'write_actions_enabled'"
        ).fetchone()[0] == "0"

    first_identity = repository.get_identity_by_external(first_account["id"], "facebook:user:42")
    second_identity = repository.get_identity_by_external(second_account["id"], "facebook:user:42")
    assert first_identity["id"] != second_identity["id"]
    with repository.read_connection() as connection:
        archived_types = {
            row[0]
            for row in connection.execute(
                """SELECT entity_type FROM legacy_records
                   WHERE connected_account_id = ?""",
                (first_account["id"],),
            ).fetchall()
        }
        assert connection.execute("SELECT COUNT(*) FROM approvals").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM outbox_items").fetchone()[0] == 0
    assert {
        "facebook_settings",
        "outreach_campaigns",
        "birthday_wishes",
        "write_approvals",
        "facebook_write_outbox",
    }.issubset(archived_types)

    rolled_back = bridge.rollback(first["run_id"])
    assert rolled_back["status"] == "rolled_back"
    assert repository.get_identity_by_external(first_account["id"], "facebook:user:42") is None
    assert repository.get_identity_by_external(second_account["id"], "facebook:user:42") is not None

    restored = bridge.migrate(source, first_account["id"])
    restored_identity = repository.get_identity_by_external(first_account["id"], "facebook:user:42")
    assert restored["status"] == "succeeded"
    assert restored_identity["id"] == first_identity["id"]


def test_facebook_adapter_wraps_verified_local_read_contract(tmp_path):
    source = tmp_path / "facebook.db"
    _legacy_database(source)
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    account = repository.add_account(
        provider="facebook", account_namespace="adapter", label="adapter", owner_profile="test"
    )
    service = CommunicationService(repository, register_builtin_adapters=False)
    service.register_adapter(FacebookCommunicationAdapter(source))

    result = service.sync(account["id"], mode="full")

    assert result["status"] == "succeeded"
    assert result["stats"] == {
        "contacts": 1,
        "conversations": 1,
        "messages_inserted": 1,
        "messages_observed": 1,
        "events": 1,
        "issues": 0,
    }
    assert service.account_health(account["id"])["write_actions_enabled"] is False
