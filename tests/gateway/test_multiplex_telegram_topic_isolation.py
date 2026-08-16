"""Telegram DM topic lanes must not leak between multiplexed profiles.

A Telegram private chat reports the *user's own id* as ``chat.id``. Every bot
in a multiplexed gateway therefore sees the identical ``chat_id`` for the same
human. Before the ``profile`` column existed, ``telegram_dm_topic_bindings``
was keyed on ``(chat_id, thread_id)`` alone, so:

  1. the default bot bound DM topic 1343 for chat 5550001111;
  2. a Career Ops message arrived "lobby" shaped (no message_thread_id);
  3. ``_recover_telegram_topic_thread_id`` looked up bindings by chat_id,
     found the default bot's topic 1343 as the newest, and pinned the
     Career Ops turn to it;
  4. the Career Ops reply was sent with ``message_thread_id=1343`` — a topic
     that exists only in the *default* bot's chat — and Telegram dropped it.

The user saw no reply in Telegram at all, while the response showed up in the
Hermes app filed under the default profile.
"""

import pytest

from hermes_state import SessionDB

CHAT = "5550001111"  # a DM chat_id == the user's own Telegram id
USER = "5550001111"


def _make_runner(*, session_db, multiplex=True, primary="default"):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.config.multiplex_profiles = multiplex
    runner._session_db = session_db
    runner._primary_profile_name = primary
    return runner


def _source(thread_id=None, profile=None):
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=USER,
        chat_id=CHAT,
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
        profile=profile,
    )


def _bind(db, *, profile, thread_id, session_id):
    db.create_session(session_id=session_id, source="telegram", user_id=USER)
    db.bind_telegram_topic(
        chat_id=CHAT,
        thread_id=thread_id,
        user_id=USER,
        session_key=f"agent:{profile}:telegram:dm:{CHAT}:{thread_id}",
        session_id=session_id,
        profile=profile,
    )


class TestBindingIsolation:
    def test_same_chat_and_thread_can_bind_under_two_profiles(self, tmp_path):
        """The identical (chat, thread) pair is a distinct lane per profile."""
        db = SessionDB(db_path=tmp_path / "state.db")
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")
        _bind(db, profile="career-ops", thread_id="1343", session_id="sess-career")

        default_binding = db.get_telegram_topic_binding(
            chat_id=CHAT, thread_id="1343", profile="default"
        )
        career_binding = db.get_telegram_topic_binding(
            chat_id=CHAT, thread_id="1343", profile="career-ops"
        )
        assert default_binding["session_id"] == "sess-default"
        assert career_binding["session_id"] == "sess-career"

    def test_list_for_chat_does_not_return_another_profiles_topics(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")

        assert db.list_telegram_topic_bindings_for_chat(
            chat_id=CHAT, profile="career-ops"
        ) == []
        assert len(
            db.list_telegram_topic_bindings_for_chat(chat_id=CHAT, profile="default")
        ) == 1

    def test_topic_mode_enabled_is_per_profile(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.enable_telegram_topic_mode(chat_id=CHAT, user_id=USER, profile="default")

        assert db.is_telegram_topic_mode_enabled(
            chat_id=CHAT, user_id=USER, profile="default"
        ) is True
        assert db.is_telegram_topic_mode_enabled(
            chat_id=CHAT, user_id=USER, profile="career-ops"
        ) is False

    def test_disable_does_not_clear_another_profiles_bindings(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")
        _bind(db, profile="career-ops", thread_id="636", session_id="sess-career")

        db.disable_telegram_topic_mode(chat_id=CHAT, profile="default")

        assert db.list_telegram_topic_bindings_for_chat(
            chat_id=CHAT, profile="default"
        ) == []
        assert len(
            db.list_telegram_topic_bindings_for_chat(chat_id=CHAT, profile="career-ops")
        ) == 1

    def test_prune_only_removes_the_owning_profiles_binding(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")
        _bind(db, profile="career-ops", thread_id="1343", session_id="sess-career")

        removed = db.delete_telegram_topic_binding(
            chat_id=CHAT, thread_id="1343", profile="career-ops"
        )

        assert removed == 1
        assert db.get_telegram_topic_binding(
            chat_id=CHAT, thread_id="1343", profile="default"
        ) is not None


class TestLobbyRecovery:
    """The actual dropped-reply path."""

    def test_career_ops_lobby_message_does_not_inherit_default_topic(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.enable_telegram_topic_mode(chat_id=CHAT, user_id=USER, profile="default")
        db.enable_telegram_topic_mode(chat_id=CHAT, user_id=USER, profile="career-ops")
        # Only the default bot has ever bound a DM topic.
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")

        runner = _make_runner(session_db=db)

        # The default profile still recovers its own lane.
        assert runner._recover_telegram_topic_thread_id(
            _source(thread_id=None, profile="default")
        ) == "1343"

        # Career Ops must NOT be pinned to the default bot's topic 1343.
        assert runner._recover_telegram_topic_thread_id(
            _source(thread_id=None, profile="career-ops")
        ) is None

    def test_each_profile_recovers_its_own_newest_topic(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.enable_telegram_topic_mode(chat_id=CHAT, user_id=USER, profile="default")
        db.enable_telegram_topic_mode(chat_id=CHAT, user_id=USER, profile="career-ops")
        _bind(db, profile="career-ops", thread_id="636", session_id="sess-career")
        # Bound later, so it is the newest row overall — the pre-fix lookup
        # ordered by updated_at across every profile and would return this.
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")

        runner = _make_runner(session_db=db)

        assert runner._recover_telegram_topic_thread_id(
            _source(thread_id=None, profile="career-ops")
        ) == "636"
        assert runner._recover_telegram_topic_thread_id(
            _source(thread_id=None, profile="default")
        ) == "1343"

    def test_topic_mode_off_for_secondary_profile_stands_recovery_down(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        # Topic mode enabled by the default bot only.
        db.enable_telegram_topic_mode(chat_id=CHAT, user_id=USER, profile="default")
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")

        runner = _make_runner(session_db=db)

        assert runner._telegram_topic_mode_enabled(
            _source(thread_id=None, profile="career-ops")
        ) is False
        assert runner._recover_telegram_topic_thread_id(
            _source(thread_id=None, profile="career-ops")
        ) is None


class TestSingleProfileGatewayUnchanged:
    def test_unmultiplexed_gateway_resolves_to_default(self, tmp_path):
        """Pre-v3 rows were backfilled to 'default'; a single-profile gateway
        must keep reading them even when it runs under a named profile home."""
        db = SessionDB(db_path=tmp_path / "state.db")
        db.enable_telegram_topic_mode(chat_id=CHAT, user_id=USER, profile="default")
        _bind(db, profile="default", thread_id="1343", session_id="sess-default")

        runner = _make_runner(session_db=db, multiplex=False, primary="career-ops")

        assert runner._topic_profile_for_source(_source(profile="career-ops")) == "default"
        assert runner._recover_telegram_topic_thread_id(
            _source(thread_id=None, profile="career-ops")
        ) == "1343"


class TestMigration:
    def test_v2_rows_are_backfilled_to_default(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="sess-legacy", source="telegram", user_id=USER)

        def _install_v2(conn):
            conn.executescript(
                """
                DROP TABLE IF EXISTS telegram_dm_topic_bindings;
                DROP TABLE IF EXISTS telegram_dm_topic_mode;
                CREATE TABLE telegram_dm_topic_mode (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    activated_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    has_topics_enabled INTEGER,
                    allows_users_to_create_topics INTEGER,
                    capability_checked_at REAL,
                    intro_message_id TEXT,
                    pinned_message_id TEXT
                );
                CREATE TABLE telegram_dm_topic_bindings (
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    managed_mode TEXT NOT NULL DEFAULT 'auto',
                    linked_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, thread_id)
                );
                INSERT INTO telegram_dm_topic_mode
                    VALUES ('5550001111', '5550001111', 1, 1.0, 1.0,
                            NULL, NULL, NULL, NULL, NULL);
                INSERT INTO telegram_dm_topic_bindings
                    VALUES ('5550001111', '1343', '5550001111',
                            'agent:main:telegram:dm:5550001111:1343',
                            'sess-legacy', 'auto', 1.0, 1.0);
                """
            )
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("telegram_dm_topic_schema_version", "2"),
            )

        db._execute_write(_install_v2)
        db.apply_telegram_topic_migration()

        binding = db.get_telegram_topic_binding(
            chat_id=CHAT, thread_id="1343", profile="default"
        )
        assert binding is not None
        assert binding["profile"] == "default"
        assert binding["session_id"] == "sess-legacy"
        assert db.is_telegram_topic_mode_enabled(
            chat_id=CHAT, user_id=USER, profile="default"
        ) is True
        # And the migrated shape now isolates a second profile.
        assert db.list_telegram_topic_bindings_for_chat(
            chat_id=CHAT, profile="career-ops"
        ) == []

    def test_migration_is_idempotent(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.apply_telegram_topic_migration()
        db.apply_telegram_topic_migration()
        _bind(db, profile="career-ops", thread_id="636", session_id="sess-career")
        assert len(
            db.list_telegram_topic_bindings_for_chat(chat_id=CHAT, profile="career-ops")
        ) == 1
