"""Delegate rows and gateway routing in the SQLite backend: peer recovery never resolves a chat to a delegate child (#138210) unless the row is the chat's own /new continuation (``_reset_from``, #109073) or a delegate the user resumed the chat onto (``_chat_adopted``); the create path keeps a stray delegate marker off a reset row; a compression rotation carries the adoption onto the new tip."""

from __future__ import annotations

import json
import uuid

import pytest

from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.conversation._store.compression import publish_compression_child
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session, end_session, get_session, list_sessions_rich,
)

THREAD_ID = "54287"
PEER = dict(
    source="telegram",
    user_id="u1",
    session_key="agent:main:telegram:dm:42",
    chat_id="42",
    chat_type="dm",
    thread_id=THREAD_ID,
)
ROUTE = {k: v for k, v in PEER.items() if k != "source"}


@pytest.fixture
def store(tmp_path):
    scope = ProfileScope(profile_dir=tmp_path, name="test")
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / "test_state.db"})
    try:
        yield backend.open(scope)
    finally:
        backend.close()


@pytest.fixture
def db(store):
    return store.holder()


def _sid(stamp: str) -> str:
    return f"{stamp}_{uuid.uuid4().hex[:6]}"


def _parent(db) -> str:
    sid = _sid("20261014_020931")
    create_session(db, sid, "telegram", **ROUTE)
    append_message(db, sid, "user", "host apps on the cluster with CLI access")
    return sid


def _keyed_child(db, parent_id: str, model_config: dict, stamp: str = "20261014_040222") -> str:
    sid = _sid(stamp)
    create_session(db, sid, "telegram", parent_session_id=parent_id, model_config=model_config, **ROUTE)
    append_message(db, sid, "assistant", "a row on the chat's route")
    return sid


def _route(store) -> str | None:
    recovered = store.routing.latest_session_for_peer(**PEER)
    return str(recovered.session.session_id) if recovered else None


def _config(db, sid: str) -> dict:
    raw = get_session(db, sid)["model_config"]
    return json.loads(raw) if isinstance(raw, str) else (raw or {})


class TestPeerRecoveryExcludesDelegateChildren:
    def test_delegate_child_holding_the_thread_never_wins_the_route(self, db, store):
        parent = _parent(db)
        _keyed_child(db, parent, {"_delegate_from": parent})
        assert _route(store) == parent

    def test_parent_ended_session_switch_does_not_hand_the_thread_to_the_child(self, db, store):
        parent = _parent(db)
        child = _keyed_child(db, parent, {"_delegate_from": parent})
        end_session(db, parent, "session_switch")
        assert _route(store) is None
        assert _route(store) != child

    def test_route_falls_back_to_the_parent_when_the_child_ended(self, db, store):
        parent = _parent(db)
        child = _keyed_child(db, parent, {"_delegate_from": parent})
        end_session(db, child, "agent_close")
        assert _route(store) == parent

    def test_peer_tuple_fallback_skips_a_delegate_child_too(self, db, store):
        parent = _parent(db)
        _keyed_child(db, parent, {"_delegate_from": parent})
        recovered = store.routing.latest_session_for_peer(**{**PEER, "session_key": "agent:main:telegram:dm:other"})
        assert recovered is not None and str(recovered.session.session_id) == parent

    def test_fresh_conversation_after_reset_is_still_reachable(self, db, store):
        parent = _parent(db)
        end_session(db, parent, "session_reset")
        successor = _sid("20261014_050000")
        create_session(db, successor, "telegram", **ROUTE)
        append_message(db, successor, "user", "a brand new conversation")
        assert _route(store) == successor

    def test_a_branch_child_still_owns_the_route_it_was_created_on(self, db, store):
        parent = _parent(db)
        end_session(db, parent, "session_switch")
        branch = _keyed_child(db, parent, {"_branched_from": parent}, stamp="20261014_050000")
        assert _route(store) == branch

    def test_a_reset_continuation_carrying_a_delegate_marker_keeps_its_route(self, db, store):
        parent = _parent(db)
        end_session(db, parent, "session_reset")
        live = _keyed_child(db, parent, {"_delegate_from": parent, "_reset_from": parent}, stamp="20261014_060000")
        assert _route(store) == live

    def test_a_delegate_row_the_chat_was_resumed_onto_keeps_its_route(self, db, store):
        parent = _parent(db)
        end_session(db, parent, "session_switch")
        adopted = _keyed_child(db, parent, {"_delegate_from": parent, "_chat_adopted": True})
        assert _route(store) == adopted


class TestCreatePathKeepsTheMarkerOffResetRows:
    OLD, LIVE = "20260819_180933_c53aa488", "20260829_143124_a736cc"

    def _old(self, db) -> None:
        create_session(db, self.OLD, "telegram", **ROUTE)
        append_message(db, self.OLD, "user", "old chat")
        end_session(db, self.OLD, "session_reset")

    def test_a_config_arriving_with_a_delegate_marker_never_lands_on_a_reset_row(self, db):
        self._old(db)
        create_session(db, self.LIVE, "telegram", parent_session_id=self.OLD,
                       model_config={"_reset_from": self.OLD}, **ROUTE)
        create_session(db, self.LIVE, "telegram",
                       model_config={"_delegate_from": self.OLD, "max_iterations": 90}, **ROUTE)
        assert _config(db, self.LIVE) == {"_reset_from": self.OLD, "max_iterations": 90}
        create_session(db, "child", "subagent", parent_session_id=self.LIVE,
                       model_config={"_delegate_from": self.LIVE})
        assert _config(db, "child") == {"_delegate_from": self.LIVE}

    def test_a_new_reset_row_with_a_delegate_marker_is_visible_before_reopen(self, db):
        self._old(db)
        create_session(db, self.LIVE, "telegram", parent_session_id=self.OLD,
                       model_config={"_reset_from": self.OLD, "_delegate_from": self.OLD}, **ROUTE)
        append_message(db, self.LIVE, "user", "new chat")
        assert _config(db, self.LIVE) == {"_reset_from": self.OLD}
        assert [s["id"] for s in list_sessions_rich(db, source="telegram", limit=50)][0] == self.LIVE

    def test_a_new_keyed_row_with_only_a_delegate_marker_stays_hidden(self, db):
        create_session(db, self.OLD, "telegram", **ROUTE)
        create_session(db, self.LIVE, "telegram", parent_session_id=self.OLD,
                       model_config={"_delegate_from": self.OLD}, **ROUTE)
        append_message(db, self.LIVE, "user", "subagent work")
        assert _config(db, self.LIVE) == {"_delegate_from": self.OLD}
        assert self.LIVE not in [s["id"] for s in list_sessions_rich(db, source="telegram", limit=50)]


def test_a_resumed_delegate_keeps_its_adoption_through_a_compression_rotation(db):
    create_session(db, "chat", "telegram", **ROUTE)
    child = "chat-worker"
    create_session(db, child, "subagent", parent_session_id="chat",
                   model_config={"_delegate_from": "chat", "_chat_adopted": True})
    tip = child + "-c1"
    publish_compression_child(
        db, parent_session_id=child, child_session_id=tip, source="subagent", system_prompt="p",
        messages=[{"role": "user", "content": "summary"}], require_compression_lease=False,
        model_config={"_delegate_from": "chat"})
    assert _config(db, tip) == {"_chat_adopted": True, "_delegate_from": "chat"}
