"""Regression tests for profile-name-keyed state migration on `hermes profile rename`.

When a profile is renamed, the directory move carries the row data, but the profile name is also
baked into session keys (``agent:<name>:*``), ``sessions.profile_name``, ``gateway_heartbeats.profile``,
``delivery_obligations`` and the ``gateway_routing`` index. Left stale, an inbound event on a chat keyed
to the old name resolves to a profile that no longer exists and floods errors.log. See the profile-rename
identity-migration fix.
"""
import json
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


class TestRekeyProfileState:
    def test_rekeys_session_key_namespace_and_profile_columns(self, db):
        # A session owned by the old profile, keyed in its namespace.
        db.create_session(
            "sess_old", "feishu", session_key="agent:oldname:feishu:dm:chatA",
            profile_name="oldname", chat_id="chatA", chat_type="dm",
        )
        # An unrelated profile's session must be left untouched.
        db.create_session(
            "sess_other", "feishu", session_key="agent:keepme:feishu:dm:chatB",
            profile_name="keepme", chat_id="chatB", chat_type="dm",
        )

        counts = db.rekey_profile_state("oldname", "newname")

        assert counts["sessions_session_key"] == 1
        assert counts["sessions_profile_name"] == 1
        # Renamed row now lives under the new namespace + owner.
        row = db._read_one(
            "SELECT session_key, profile_name FROM sessions WHERE id = ?", ("sess_old",))
        assert row["session_key"] == "agent:newname:feishu:dm:chatA"
        assert row["profile_name"] == "newname"
        # Bystander untouched.
        other = db._read_one(
            "SELECT session_key, profile_name FROM sessions WHERE id = ?", ("sess_other",))
        assert other["session_key"] == "agent:keepme:feishu:dm:chatB"
        assert other["profile_name"] == "keepme"

    def test_rekeys_routing_index_key_and_embedded_profile(self, db):
        entry = {
            "session_key": "agent:oldname:feishu:dm:chatA",
            "session_id": "sess_old",
            "origin": {"platform": "feishu", "chat_id": "chatA", "profile": "oldname"},
        }
        db.save_gateway_routing_entry(
            "agent:oldname:feishu:dm:chatA", json.dumps(entry), scope="/root/sessions")

        counts = db.rekey_profile_state("oldname", "newname")
        assert counts["gateway_routing"] == 1

        rows = db.load_gateway_routing_entries(scope="/root/sessions")
        assert "agent:oldname:feishu:dm:chatA" not in rows
        assert "agent:newname:feishu:dm:chatA" in rows
        payload = json.loads(rows["agent:newname:feishu:dm:chatA"])
        assert payload["session_key"] == "agent:newname:feishu:dm:chatA"
        assert payload["origin"]["profile"] == "newname"

    def test_rekeys_heartbeats_and_delivery_obligations(self, db, tmp_path, monkeypatch):
        db.register_backend_heartbeat(
            backend_id="be1", pid=123, started_at=time.time(),
            profile="oldname", host="h")
        # delivery_obligations is created lazily by the delivery ledger against the same state.db.
        monkeypatch.setenv("HERMES_HOME", str(db.db_path.parent))
        from gateway import delivery_ledger
        monkeypatch.setattr(delivery_ledger, "_db_path", lambda: db.db_path)
        with delivery_ledger._connect() as conn:
            now = time.time()
            conn.execute(
                "INSERT INTO delivery_obligations (obligation_id, session_key, platform, chat_id, "
                "content, state, created_at, updated_at, adapter_profile) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("ob1", "agent:oldname:feishu:dm:chatA", "feishu", "chatA",
                 "hi", "pending", now, now, "oldname"),
            )
            conn.commit()

        counts = db.rekey_profile_state("oldname", "newname")

        assert counts["gateway_heartbeats_profile"] == 1
        assert counts["delivery_obligations_adapter_profile"] == 1
        assert counts["delivery_obligations_session_key"] == 1
        hb = db._read_one("SELECT profile FROM gateway_heartbeats WHERE backend_id = ?", ("be1",))
        assert hb["profile"] == "newname"
        ob = db._read_one(
            "SELECT session_key, adapter_profile FROM delivery_obligations WHERE obligation_id = ?",
            ("ob1",))
        assert ob["session_key"] == "agent:newname:feishu:dm:chatA"
        assert ob["adapter_profile"] == "newname"

    def test_noop_when_names_equal_or_empty(self, db):
        assert db.rekey_profile_state("x", "x") == {}
        assert db.rekey_profile_state("", "y") == {}
        assert db.rekey_profile_state("x", "") == {}

    def test_idempotent(self, db):
        db.create_session(
            "sess_old", "feishu", session_key="agent:oldname:feishu:dm:chatA",
            profile_name="oldname", chat_id="chatA", chat_type="dm",
        )
        first = db.rekey_profile_state("oldname", "newname")
        assert first["sessions_session_key"] == 1
        second = db.rekey_profile_state("oldname", "newname")
        # Nothing left under the old name.
        assert second["sessions_session_key"] == 0
        assert second["sessions_profile_name"] == 0

    def test_underscore_in_profile_name_is_not_a_like_wildcard(self, db):
        db.create_session(
            "literal", "telegram", session_key="agent:foo_bar:telegram:dm:a",
            profile_name="foo_bar", chat_id="a", chat_type="dm")
        db.create_session(
            "bystander", "telegram", session_key="agent:fooXbar:telegram:dm:b",
            profile_name="fooXbar", chat_id="b", chat_type="dm")
        db.rekey_profile_state("foo_bar", "renamed")
        assert db._read_one(
            "SELECT session_key FROM sessions WHERE id = ?", ("literal",)
        )["session_key"] == "agent:renamed:telegram:dm:a"
        assert db._read_one(
            "SELECT session_key FROM sessions WHERE id = ?", ("bystander",)
        )["session_key"] == "agent:fooXbar:telegram:dm:b"

    def test_rekeys_origin_json_and_telegram_topic_state(self, db):
        db.create_session(
            "sess_old", "telegram", session_key="agent:oldname:telegram:dm:chatA",
            profile_name="oldname", chat_id="chatA", chat_type="dm")
        db._write_sql(
            "UPDATE sessions SET origin_json = ? WHERE id = ?",
            (json.dumps({"platform": "telegram", "profile": "oldname"}), "sess_old"))
        db.enable_telegram_topic_mode(
            chat_id="chatA", user_id="userA", profile_name="oldname")
        db.bind_telegram_topic(
            chat_id="chatA", thread_id="threadA", user_id="userA",
            session_key="agent:oldname:telegram:dm:chatA", session_id="sess_old",
            profile_name="oldname")
        counts = db.rekey_profile_state("oldname", "newname")
        row = db._read_one("SELECT origin_json FROM sessions WHERE id = ?", ("sess_old",))
        assert json.loads(row["origin_json"])["profile"] == "newname"
        assert counts["sessions_origin_json"] == 1
        binding = db._read_one(
            "SELECT profile_name, session_key FROM telegram_dm_topic_bindings "
            "WHERE chat_id = ? AND thread_id = ?", ("chatA", "threadA"))
        assert binding["profile_name"] == "newname"
        assert binding["session_key"] == "agent:newname:telegram:dm:chatA"

