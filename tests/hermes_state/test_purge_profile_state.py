"""Regression tests for profile-keyed state purge on `hermes profile delete`.

`delete_profile` removes the profile directory and the multiplexer tears its runtime down, but the
profile name is also baked into session keys (``agent:<name>:*``), ``gateway_heartbeats.profile``,
``delivery_obligations`` and the ``gateway_routing`` index. Left stale, an inbound event on a chat
keyed to the deleted name resolves to a profile that no longer exists and floods errors.log — the
delete-side sibling of the rename rekey in #111926.
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


def _write_obligation(db, monkeypatch, obligation_id, session_key, chat_id, profile,
                      state="pending"):
    """delivery_obligations is created lazily by the delivery ledger against the same state.db."""
    monkeypatch.setenv("HERMES_HOME", str(db.db_path.parent))
    from gateway import delivery_ledger
    monkeypatch.setattr(delivery_ledger, "_db_path", lambda: db.db_path)
    with delivery_ledger._connect() as conn:
        now = time.time()
        conn.execute(
            "INSERT INTO delivery_obligations (obligation_id, session_key, platform, chat_id, "
            "content, state, created_at, updated_at, adapter_profile) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (obligation_id, session_key, "feishu", chat_id, "hi", state, now, now, profile),
        )
        conn.commit()


class TestPurgeProfileState:
    def test_purges_routing_and_heartbeats_but_keeps_bystanders(self, db):
        db.save_gateway_routing_entry(
            "agent:gone:feishu:dm:chatA", json.dumps({"session_key": "agent:gone:feishu:dm:chatA"}),
            scope="/root/sessions")
        db.save_gateway_routing_entry(
            "agent:keepme:feishu:dm:chatB",
            json.dumps({"session_key": "agent:keepme:feishu:dm:chatB"}), scope="/root/sessions")
        db.register_backend_heartbeat(
            backend_id="be-gone", pid=1, started_at=time.time(), profile="gone", host="h")
        db.register_backend_heartbeat(
            backend_id="be-keep", pid=2, started_at=time.time(), profile="keepme", host="h")

        counts = db.purge_profile_state("gone")

        assert counts["gateway_routing"] == 1
        assert counts["gateway_heartbeats"] == 1
        routing = db.load_gateway_routing_entries(scope="/root/sessions")
        assert "agent:gone:feishu:dm:chatA" not in routing
        assert "agent:keepme:feishu:dm:chatB" in routing
        assert db._read_one(
            "SELECT profile FROM gateway_heartbeats WHERE backend_id = ?",
            ("be-keep",))["profile"] == "keepme"
        assert db._read_one(
            "SELECT COUNT(*) AS n FROM gateway_heartbeats WHERE profile = ?", ("gone",))["n"] == 0

    def test_abandons_pending_deliveries_instead_of_deleting_them(self, db, monkeypatch):
        """A pending obligation is delivery state, not identity: terminalize it, never drop it."""
        _write_obligation(db, monkeypatch, "ob-gone", "agent:gone:feishu:dm:chatA", "chatA", "gone")
        _write_obligation(
            db, monkeypatch, "ob-gone-done", "agent:gone:feishu:dm:chatB", "chatB", "gone",
            state="delivered")
        _write_obligation(
            db, monkeypatch, "ob-keep", "agent:keepme:feishu:dm:chatC", "chatC", "keepme")

        counts = db.purge_profile_state("gone")

        assert counts["delivery_obligations"] == 1
        pending = db._read_one(
            "SELECT state FROM delivery_obligations WHERE obligation_id = ?", ("ob-gone",))
        assert pending["state"] == "abandoned"
        # A delivered row for the same profile is history, not pending state: left exactly as it was.
        delivered = db._read_one(
            "SELECT state FROM delivery_obligations WHERE obligation_id = ?", ("ob-gone-done",))
        assert delivered["state"] == "delivered"
        bystander = db._read_one(
            "SELECT state FROM delivery_obligations WHERE obligation_id = ?", ("ob-keep",))
        assert bystander["state"] == "pending"

    def test_purge_does_not_delete_session_rows(self, db):
        """Pins this helper's scope only: the purge settles identity, it never deletes rows itself.

        This is NOT a claim that a deleted profile's conversation record outlives the delete — the
        delete path removes the profile's own ``state.db`` together with ``profiles/<name>/``. It
        says the identity purge adds no deletion of its own, so it cannot be the step that loses
        history, and nothing here pins the row's stale ownership either.
        """
        db.create_session(
            "sess_gone", "feishu", session_key="agent:gone:feishu:dm:chatA",
            profile_name="gone", chat_id="chatA", chat_type="dm",
        )

        db.purge_profile_state("gone")

        assert db._read_one(
            "SELECT id FROM sessions WHERE id = ?", ("sess_gone",)) is not None

    def test_purges_topic_bindings_matched_by_session_key_namespace(self, db):
        """The rekey rewrites a binding's ``profile_name`` AND its ``session_key`` namespace, so the
        purge has to match both — a binding whose ``profile_name`` names somebody else still belongs
        to the deleted profile by key, and matching on ``profile_name`` alone leaves it routing
        events for a profile that no longer exists."""
        db.create_session(
            "sess_gone", "telegram", session_key="agent:gone:telegram:dm:chatA",
            profile_name="gone", chat_id="chatA", chat_type="dm",
        )
        db.create_session(
            "sess_keep", "telegram", session_key="agent:keepme:telegram:dm:chatB",
            profile_name="keepme", chat_id="chatB", chat_type="dm",
        )
        db.bind_telegram_topic(
            chat_id="chatA", thread_id="threadA", user_id="userA",
            session_key="agent:gone:telegram:dm:chatA", session_id="sess_gone",
            profile_name="otherholder")
        db.bind_telegram_topic(
            chat_id="chatB", thread_id="threadB", user_id="userB",
            session_key="agent:keepme:telegram:dm:chatB", session_id="sess_keep",
            profile_name="keepme")

        counts = db.purge_profile_state("gone")

        assert db._read_one(
            "SELECT COUNT(*) AS n FROM telegram_dm_topic_bindings WHERE chat_id = ?",
            ("chatA",))["n"] == 0
        assert db._read_one(
            "SELECT COUNT(*) AS n FROM telegram_dm_topic_bindings WHERE chat_id = ?",
            ("chatB",))["n"] == 1
        assert counts["telegram_dm_topic_bindings"] == 1

    def test_underscore_in_profile_name_is_not_a_like_wildcard(self, db):
        db.save_gateway_routing_entry(
            "agent:foo_bar:feishu:dm:chatA",
            json.dumps({"session_key": "agent:foo_bar:feishu:dm:chatA"}), scope="/root/sessions")
        db.save_gateway_routing_entry(
            "agent:fooXbar:feishu:dm:chatB",
            json.dumps({"session_key": "agent:fooXbar:feishu:dm:chatB"}), scope="/root/sessions")

        db.purge_profile_state("foo_bar")

        routing = db.load_gateway_routing_entries(scope="/root/sessions")
        assert "agent:foo_bar:feishu:dm:chatA" not in routing
        assert "agent:fooXbar:feishu:dm:chatB" in routing

    def test_purge_is_idempotent(self, db):
        db.register_backend_heartbeat(
            backend_id="be1", pid=1, started_at=time.time(), profile="gone", host="h")

        assert db.purge_profile_state("gone")["gateway_heartbeats"] == 1
        assert db.purge_profile_state("gone")["gateway_heartbeats"] == 0
