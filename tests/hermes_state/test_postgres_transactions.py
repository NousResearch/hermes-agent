"""PostgreSQL failure recovery across read transactions and competing writers."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from hermes_state import SessionDB
from hermes_state_backend import DatabaseSettings


def test_unlinked_topic_lookup_recovers_when_optional_tables_are_absent(postgres_db):
    db = postgres_db
    db.create_session("unlinked", "telegram", user_id="42")
    db.append_message("unlinked", "user", "topic receipt")

    rows = db.list_unlinked_telegram_sessions_for_user(chat_id="42", user_id="42")

    assert [row["id"] for row in rows] == ["unlinked"]
    assert db._read_one("SELECT to_regclass('telegram_dm_topic_bindings')")[0] is None
    db.bind_telegram_topic(chat_id="42", thread_id="7", user_id="42", session_key="topic:7", session_id="unlinked")
    assert db.list_unlinked_telegram_sessions_for_user(chat_id="42", user_id="42") == []


def test_turn_lease_retries_postgres_writer_lock_timeout(postgres_db):
    db = postgres_db
    db.create_session("leased", "cli")
    waiting = Event()
    with SessionDB(db.db_path) as other, db._pool.connection() as blocker:
        blocker.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (db.schema,))
        with ThreadPoolExecutor(max_workers=1) as executor:
            acquired = executor.submit(
                other.acquire_session_turn_lease, "leased", "worker", wait_seconds=5,
                poll_interval_seconds=0.01, acquire_patience_s=0.01, on_wait=lambda elapsed: waiting.set())
            try:
                assert waiting.wait(10), "the lease caller did not retry the PostgreSQL lock timeout"
            finally:
                blocker.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (db.schema,))
            assert acquired.result(timeout=10) is True
        db.append_message("leased", "user", "owned receipt", turn_lease_holder="worker")


def test_topic_binding_deletion_commits_when_optional_mode_table_is_missing(postgres_db, tmp_path):
    with SessionDB(tmp_path / "sqlite.db", database_settings=DatabaseSettings()) as sqlite_db:
        for db in (sqlite_db, postgres_db):
            db.create_session("topic", "telegram", user_id="42")
            db.bind_telegram_topic(
                chat_id="42", thread_id="7", user_id="42", session_key="topic:7", session_id="topic")
            db._execute_write(lambda conn: conn.execute("DROP TABLE telegram_dm_topic_mode"))

            assert db.delete_telegram_topic_binding(chat_id="42", thread_id="7") == 1

            assert db.get_telegram_topic_binding(chat_id="42", thread_id="7") is None
            assert db.get_session("topic") is not None
