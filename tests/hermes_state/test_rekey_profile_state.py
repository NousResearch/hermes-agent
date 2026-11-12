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

from hermes_agent.hermes_state import SessionDB
from hermes_agent.hermes_state_local import API_BINDING_PREFIX, POLICY_PREFIX


def _session_receipt(name, sid, platform, **extra):
    """A session receipt minted by profile *name* for *sid* on *platform*: its route and entry key
    sit in the ``agent:<name>:`` namespace, its origin names the profile."""
    route = f"agent:{name}:{platform}:dm:{sid}"
    entry_sid = extra["lineage"][-1] if "lineage" in extra else sid
    return {"profile_id": f"/homes/{name}", "session_id": sid, "route": route, **extra,
            "entry": {"session_id": entry_sid, "session_key": route,
                      "origin": {"platform": platform, "chat_id": sid, "profile": name}}}


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


def _create_legacy_v2_topic_tables(db):
    """Create the supported pre-profile-name topic schema without migrating it."""
    db._write_sql("""
        CREATE TABLE telegram_dm_topic_mode (
            chat_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            activated_at REAL NOT NULL, updated_at REAL NOT NULL,
            has_topics_enabled INTEGER, allows_users_to_create_topics INTEGER,
            capability_checked_at REAL, intro_message_id TEXT, pinned_message_id TEXT
        )
    """)
    db._write_sql("""
        CREATE TABLE telegram_dm_topic_bindings (
            chat_id TEXT NOT NULL, thread_id TEXT NOT NULL, user_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            managed_mode TEXT NOT NULL DEFAULT 'auto',
            linked_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (chat_id, thread_id)
        )
    """)


def _create_legacy_ledger_without_adapter_profile(db):
    """The delivery_obligations shape before ``adapter_profile`` was added (ledger never reopened)."""
    db._write_sql("""
        CREATE TABLE delivery_obligations (
            obligation_id TEXT PRIMARY KEY, session_key TEXT NOT NULL,
            platform TEXT NOT NULL, chat_id TEXT NOT NULL, thread_id TEXT, content TEXT NOT NULL,
            state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            owner_pid INTEGER, owner_started_at INTEGER, last_error TEXT
        )
    """)
    for oid, key in (("ob_gone", "agent:gone:feishu:dm:chatA"), ("ob_keep", "agent:keepme:feishu:dm:chatB")):
        db._write_sql(
            "INSERT INTO delivery_obligations (obligation_id, session_key, platform, chat_id, "
            "content, state, created_at, updated_at) VALUES (?, ?, 'feishu', 'c', 'hi', 'pending', 1, 1)",
            (oid, key))


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
        from hermes_agent.gateway import delivery_ledger
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

    def test_rekeys_legacy_v2_topic_binding_by_session_key(self, db):
        """Legacy v2 tables have no profile_name, but bindings retain profile namespaces."""
        db.create_session(
            "sess_old", "telegram", session_key="agent:oldname:telegram:dm:chatA",
            profile_name="oldname", chat_id="chatA", chat_type="dm")
        db.create_session(
            "sess_keep", "telegram", session_key="agent:keepme:telegram:dm:chatB",
            profile_name="keepme", chat_id="chatB", chat_type="dm")
        _create_legacy_v2_topic_tables(db)
        db._write_sql(
            "INSERT INTO telegram_dm_topic_mode "
            "(chat_id, user_id, enabled, activated_at, updated_at) VALUES (?, ?, 1, 1, 1)",
            ("chatA", "userA"))
        db._write_sql(
            "INSERT INTO telegram_dm_topic_bindings "
            "(chat_id, thread_id, user_id, session_key, session_id, linked_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1, 1)",
            ("chatA", "threadA", "userA", "agent:oldname:telegram:dm:chatA", "sess_old"))
        db._write_sql(
            "INSERT INTO telegram_dm_topic_bindings "
            "(chat_id, thread_id, user_id, session_key, session_id, linked_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1, 1)",
            ("chatB", "threadB", "userB", "agent:keepme:telegram:dm:chatB", "sess_keep"))

        counts = db.rekey_profile_state("oldname", "newname")

        assert counts["telegram_dm_topic_bindings_session_key"] == 1
        assert db._read_one(
            "SELECT session_key FROM telegram_dm_topic_bindings WHERE chat_id = ?", ("chatA",)
        )["session_key"] == "agent:newname:telegram:dm:chatA"
        assert db._read_one(
            "SELECT session_key FROM telegram_dm_topic_bindings WHERE chat_id = ?", ("chatB",)
        )["session_key"] == "agent:keepme:telegram:dm:chatB"

    def test_rekeys_api_binding_receipts(self, db):
        """The renamed profile's API receipts move to the new name; others and every id stay."""
        bound = _session_receipt("oldname", "sess_api", "api_server")
        # Its name starts with the renamed one: only the whole ``agent:<name>:`` namespace may match.
        bystander = _session_receipt("oldname2", "sess_keep", "api_server")
        db.set_meta(API_BINDING_PREFIX + "sess_api", json.dumps(bound))
        db.set_meta(API_BINDING_PREFIX + "sess_keep", json.dumps(bystander))

        counts = db.rekey_profile_state("oldname", "newname")

        saved = json.loads(db.get_meta(API_BINDING_PREFIX + "sess_api"))
        assert saved["route"] == saved["entry"]["session_key"] == "agent:newname:api_server:dm:sess_api"
        assert saved["entry"]["origin"]["profile"] == "newname"
        # The id and the minting home are the store's, not the name's: both stay as stored.
        assert saved["session_id"] == saved["entry"]["session_id"] == "sess_api"
        assert saved["profile_id"] == "/homes/oldname"
        assert json.loads(db.get_meta(API_BINDING_PREFIX + "sess_keep")) == bystander
        assert counts.get("api_binding_receipts") == 1

    def test_leaves_unparseable_receipt_rows_alone(self, db):
        """Control: a receipt the store cannot parse is skipped, not rewritten into garbage. The key
        carries the real prefix, so the rekey scan does read it."""
        db.set_meta(API_BINDING_PREFIX + "broken", "not json")

        counts = db.rekey_profile_state("oldname", "newname")

        assert counts.get("api_binding_receipts", 0) == 0
        assert db.get_meta(API_BINDING_PREFIX + "broken") == "not json"

    def test_rekeys_legacy_ledger_without_adapter_profile_by_session_key(self, db):
        """A ledger created before adapter_profile existed is rekeyed on its namespace alone."""
        _create_legacy_ledger_without_adapter_profile(db)

        counts = db.rekey_profile_state("gone", "newname")

        assert "delivery_obligations_adapter_profile" not in counts
        assert counts["delivery_obligations_session_key"] == 1
        keys = {row["obligation_id"]: row["session_key"] for row in db._read_all(
            "SELECT obligation_id, session_key FROM delivery_obligations")}
        assert keys == {"ob_gone": "agent:newname:feishu:dm:chatA",
                        "ob_keep": "agent:keepme:feishu:dm:chatB"}

    def test_rekeys_local_policy_receipts(self, db):
        """The renamed profile's local-chat receipts move to the new name (route, entry key, origin);
        the minting home, the id and the lineage stay, and no legacy marker is written."""
        from hermes_agent.hermes_state_local import local_receipt
        from hermes_agent.hermes_state_local_migration import LEGACY_PREFIX

        moved = _session_receipt("oldname", "local-a", "local", lineage=["local-a", "local-a-tip"])
        bystander = _session_receipt("oldname2", "local-b", "local")
        db.set_meta(POLICY_PREFIX + "local-a", json.dumps(moved))
        db.set_meta(POLICY_PREFIX + "local-b", json.dumps(bystander))

        counts = db.rekey_profile_state("oldname", "newname")

        saved = local_receipt(db, "local-a")
        assert saved["route"] == saved["entry"]["session_key"] == "agent:newname:local:dm:local-a"
        assert saved["entry"]["origin"]["profile"] == "newname"
        # The minting home is the session id's hash input: it, the id and the lineage stay as stored.
        assert saved["profile_id"] == "/homes/oldname"
        assert saved["session_id"] == "local-a"
        assert saved["lineage"] == ["local-a", "local-a-tip"]
        assert "legacy_session_id" not in saved
        assert db.get_meta(LEGACY_PREFIX + "local-a") is None
        assert local_receipt(db, "local-b") == bystander
        assert counts["local_policy_receipts"] == 1
