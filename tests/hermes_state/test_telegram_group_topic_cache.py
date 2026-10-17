"""SQLite contracts for persistent forum names (salvaged from #51839)."""

import pytest

from hermes_state import SessionDB


def test_existing_database_reconciles_group_cache_without_dm_opt_in(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        version = db._read_one("SELECT version FROM schema_version")[0]
        db._execute_write(lambda conn: conn.execute("DROP TABLE telegram_group_topic_cache"))
    scope = dict(profile_name="receiving-profile", bot_id="123456", chat_id="-100123", thread_id="42")
    with SessionDB(path) as db:
        db.set_telegram_group_topic_name(**scope, name="Planning")
        db.set_telegram_group_topic_name(**scope, name="Renamed", source="forum_topic_edited")
        db.set_telegram_group_topic_name(**scope, name="Old creation", source="creation_reply", only_if_unknown=True)
        db.set_telegram_group_topic_name(**scope, name="", source="forum_topic_edited")
        assert db.get_telegram_group_topic_name(**scope) == "Renamed"
        assert db._read_one("SELECT version FROM schema_version")[0] == version
        assert not db._read_one("SELECT name FROM sqlite_master WHERE name='telegram_dm_topic_mode'")
    with SessionDB(path) as db:
        assert db.get_telegram_group_topic_name(**scope) == "Renamed"
        for key, value in (("profile_name", "other"), ("bot_id", "654321"), ("chat_id", "-100999"), ("thread_id", "99")):
            assert db.get_telegram_group_topic_name(**{**scope, key: value}) is None
        row = db._read_one("SELECT source, updated_at FROM telegram_group_topic_cache")
        assert row["source"] == "forum_topic_edited"
        assert row["updated_at"] > 0


def test_group_cache_requires_explicit_namespace(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        with pytest.raises(TypeError):
            db.set_telegram_group_topic_name(chat_id="-100123", thread_id="42", name="No namespace")
        with pytest.raises(ValueError, match="namespace"):
            db.set_telegram_group_topic_name(profile_name="", bot_id="123456", chat_id="-100123", thread_id="42", name="Invalid")


def test_unknown_observation_is_stable_and_reply_seed_cannot_replace_rename(tmp_path):
    scope = dict(profile_name="receiver", bot_id="123456", chat_id="-100123", thread_id="42")
    with SessionDB(tmp_path / "state.db") as db:
        db.record_telegram_group_topic_observation(**scope)
        assert db.get_telegram_group_topic_name(**scope) is None
        assert db.list_telegram_group_topic_names(profile_name="receiver", bot_id="123456") == [
            {"chat_id": "-100123", "thread_id": "42", "name": ""}]
        changes = db._conn.total_changes
        db.record_telegram_group_topic_observation(**scope)
        assert db._conn.total_changes == changes
        db.set_telegram_group_topic_name(**scope, name="Creation", source="creation_reply", only_if_unknown=True)
        assert db.get_telegram_group_topic_name(**scope) == "Creation"
        db.set_telegram_group_topic_name(**scope, name="Renamed", source="forum_topic_edited")
        changes = db._conn.total_changes
        db.record_telegram_group_topic_observation(**scope)
        db.set_telegram_group_topic_name(**scope, name="Creation", source="creation_reply", only_if_unknown=True)
        assert db._conn.total_changes == changes
        assert db.get_telegram_group_topic_name(**scope) == "Renamed"
