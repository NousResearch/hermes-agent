"""A SQLite conversation moves intact, or the destination remains untouched."""

import json
import sqlite3
from contextlib import closing

import pytest

from hermes_state import SessionDB
from scripts.migrate_sqlite_to_postgres import migrate_sqlite_to_postgres


@pytest.fixture
def sqlite_source(postgres_home):
    path = postgres_home / "sqlite-source" / "state.db"
    with SessionDB(path) as source:
        assert source.backend == "sqlite"
        source.create_session("child", "cli", model_config={"_branched_from": "parent"})
        source.create_session(
            "parent", "telegram", model="test-model", model_config={"temperature": 0.25},
            system_prompt="Unchanged system prompt 世界", user_id="user", session_key="peer",
            chat_id="chat", thread_id="topic", profile_name="work",
            origin_json=json.dumps({"platform": "telegram", "chat_id": "chat"}),
        )
        source._execute_write(lambda connection: connection.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("parent", "child")
        ))
        source.append_message("parent", "user", "Original question")
        source.append_message("parent", "assistant", "Original answer")
        source.archive_and_compact("parent", [{"role": "user", "content": "Earlier summary"}])
        source.append_message(
            "parent", "user", [{"type": "text", "text": "A picture 世界"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}],
            timestamp=1_800_000_000.123456, platform_message_id="platform-message",
            display_metadata={"gateway_input_owner": "owner"}, api_content="Exact API content",
        )
        source.append_message(
            "parent", "assistant", "Visible answer", reasoning_content="Preserved reasoning",
            reasoning_details=[{"type": "reasoning.text", "text": "Details"}],
            codex_message_items=[{"type": "message", "id": "item"}], finish_reason="stop",
        )
        rewind_id = source.append_message("parent", "user", "Rewound question")
        source.append_message("parent", "assistant", "Rewound answer")
        source.rewind_to_message("parent", rewind_id)
        source.append_message("child", "user", "Embedded\x00NUL")
        source.append_message("child", "assistant", b"legacy\x00bytes")
        source.update_token_counts(
            "parent", input_tokens=31, output_tokens=9, api_call_count=2,
            model="test-model", billing_provider="test-provider", actual_cost_usd=None,
        )
        source.touch_session_activity("parent", 1_900_000_000.125, description="Saved activity")
        source.set_meta("migration-note", "copied")
        source.save_gateway_routing_entry("peer", json.dumps({"session_id": "parent"}), scope="work")
        source.increment_hygiene_failure_streak("peer")
        source.end_session("parent", "session_reset")
        source.enable_telegram_topic_mode(chat_id="chat", user_id="user", profile_name="work")
        source.bind_telegram_topic(
            chat_id="chat", thread_id="topic", user_id="user", session_key="peer",
            session_id="parent", profile_name="work",
        )
        source.try_acquire_session_turn_lease("child", "old-process")
        source.try_acquire_compression_lock("child", "old-process")
        source.register_backend_heartbeat(backend_id="old-process", pid=123, started_at=1.0)
    return path


def test_migration_preserves_history_and_metadata_across_small_batches(sqlite_source, postgres_db):
    source_bytes = sqlite_source.read_bytes()
    with closing(sqlite3.connect(sqlite_source)) as source:
        source.row_factory = sqlite3.Row
        expected = {
            table: [dict(row) for row in source.execute(f'SELECT * FROM "{table}"')]
            for table in (
                "system_prompts", "sessions", "messages", "session_model_usage", "state_meta",
                "gateway_routing", "gateway_hygiene_state", "conversation_generations",
                "telegram_dm_topic_mode", "telegram_dm_topic_bindings",
            )
        }
    assert expected["sessions"][0]["parent_session_id"] == expected["sessions"][1]["id"]
    assert any(row["compacted"] for row in expected["messages"])
    assert any(not row["active"] and not row["compacted"] for row in expected["messages"])

    counts = migrate_sqlite_to_postgres(sqlite_source, postgres_db, batch_size=1)

    assert counts == {table: len(rows) for table, rows in expected.items()}
    with postgres_db._read_ctx() as connection:
        for table, rows in expected.items():
            actual = [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]
            if table == "messages":
                for row in rows:
                    row["content"] = SessionDB._decode_content(row["content"])
                for row in actual:
                    row["content"] = postgres_db._decode_content(row["content"])
                for row in [*rows, *actual]:
                    for column in ("display_identity", "display_order", "search_vector"):
                        row.pop(column, None)
            assert sorted(actual, key=repr) == sorted(rows, key=repr), table
        for table in ("compression_locks", "session_turn_leases", "gateway_heartbeats"):
            assert connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0

    assert postgres_db.get_session("parent")["system_prompt"] == "Unchanged system prompt 世界"
    assert postgres_db.is_telegram_topic_mode_enabled(chat_id="chat", user_id="user", profile_name="work")
    assert postgres_db.get_telegram_topic_binding_by_session(session_id="parent")["thread_id"] == "topic"
    assert postgres_db.get_messages("parent", include_compacted=True)
    next_id = postgres_db.append_message("child", "user", "New PostgreSQL message")
    assert next_id > max(row["id"] for row in expected["messages"])
    assert sqlite_source.read_bytes() == source_bytes


def test_migration_refuses_a_populated_destination_without_changing_either_store(sqlite_source, postgres_db):
    postgres_db.create_session("existing", "cli")
    postgres_db.append_message("existing", "user", "Already stored")
    source_bytes = sqlite_source.read_bytes()
    original_revision = postgres_db.get_change_revision()

    with pytest.raises(ValueError, match="not empty; migration refused"):
        migrate_sqlite_to_postgres(sqlite_source, postgres_db)

    assert postgres_db.session_count() == 1
    assert postgres_db.get_messages("existing")[0]["content"] == "Already stored"
    assert postgres_db.get_change_revision() == original_revision
    assert sqlite_source.read_bytes() == source_bytes


def test_later_copy_failure_rolls_back_earlier_tables_and_allows_retry(sqlite_source, postgres_db):
    import psycopg

    source_bytes = sqlite_source.read_bytes()
    postgres_db._execute_write(lambda connection: connection.execute(
        "ALTER TABLE state_meta ADD CONSTRAINT reject_migration_note CHECK (key <> 'migration-note')"
    ))
    original_revision = postgres_db.get_change_revision()

    with pytest.raises(psycopg.errors.CheckViolation, match="reject_migration_note"):
        migrate_sqlite_to_postgres(sqlite_source, postgres_db, batch_size=1)

    with postgres_db._read_ctx() as connection:
        for table in ("sessions", "messages", "system_prompts", "session_model_usage", "state_meta"):
            assert connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 0
        assert connection.execute("SELECT to_regclass('telegram_dm_topic_mode')").fetchone()[0] is None
    assert postgres_db.get_change_revision() == original_revision
    assert sqlite_source.read_bytes() == source_bytes

    postgres_db._execute_write(lambda connection: connection.execute(
        "ALTER TABLE state_meta DROP CONSTRAINT reject_migration_note"
    ))
    counts = migrate_sqlite_to_postgres(sqlite_source, postgres_db, batch_size=1)
    assert postgres_db.session_count() == counts["sessions"]
    assert sqlite_source.read_bytes() == source_bytes


def test_unknown_source_columns_refuse_migration_without_partial_copy(sqlite_source, postgres_db):
    with closing(sqlite3.connect(sqlite_source)) as source:
        source.execute("ALTER TABLE sessions ADD COLUMN future_metadata TEXT")
    source_bytes = sqlite_source.read_bytes()

    with pytest.raises(ValueError, match="Unsupported sessions columns: future_metadata"):
        migrate_sqlite_to_postgres(sqlite_source, postgres_db)

    assert postgres_db.session_count() == 0
    assert postgres_db._read_one("SELECT COUNT(*) FROM system_prompts")[0] == 0
    assert sqlite_source.read_bytes() == source_bytes
