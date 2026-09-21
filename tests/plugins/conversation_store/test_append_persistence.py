from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from acp_adapter.session import SessionManager
from agent.session_persistence import _db_flush_collect, _db_flush_write
from conversation_store import (
    ConversationConflictError, ConversationMutationResult, ConversationRevision, ConversationStore,
)
from hermes_state import SessionDB


class AppendStore(ConversationStore):
    def __init__(self):
        self.revisions = {}
        self.conversations = {}
        self.messages = {}
        self.idempotent = {}
        self.next_id = 100
        self.fail = None
        self.conflict_on_append = False

    @property
    def name(self):
        return "append-store"

    def is_available(self):
        return True

    def ensure_conversation(self, conversation):
        sid = conversation["id"]
        self.revisions.setdefault(sid, 1)
        self.messages.setdefault(sid, [])
        self.conversations.setdefault(sid, deepcopy(conversation))
        return ConversationRevision(self.revisions[sid])

    def get_conversation(self, conversation_id):
        row = self.conversations.get(conversation_id)
        return deepcopy(row) if row is not None else None

    def conversation_history(self, conversation_id, **_filters):
        return [deepcopy(row) for row in self.messages.get(conversation_id, [])]

    def get_revision(self, conversation_id):
        return ConversationRevision(self.revisions[conversation_id])

    def append_messages(self, conversation_id, messages, *, expected_revision, idempotency_key=None):
        if self.fail is not None:
            raise self.fail
        if idempotency_key and idempotency_key in self.idempotent:
            prior = self.idempotent[idempotency_key]
            return ConversationMutationResult(
                revision=self.get_revision(conversation_id), affected_count=0,
                message_ids=prior.message_ids, canonical_messages=prior.canonical_messages)
        if self.conflict_on_append:
            self.revisions[conversation_id] += 1
            self.conflict_on_append = False
        actual = self.get_revision(conversation_id)
        if expected_revision != actual:
            raise ConversationConflictError("stale append")

        canonical, ids = [], []
        tool_calls = 0
        for message in messages:
            self.next_id += 1
            row = deepcopy(message)
            row["_row_id"] = self.next_id
            if row.get("role") == "assistant" and row.get("content") == "draft":
                row["content"] = "canonical"
            canonical.append(row)
            ids.append(self.next_id)
            tool_calls += len(row.get("tool_calls") or [])
        self.messages[conversation_id].extend(deepcopy(canonical))
        self.revisions[conversation_id] += 1
        result = ConversationMutationResult(
            revision=self.get_revision(conversation_id), affected_count=len(messages),
            message_ids=tuple(ids), canonical_messages=tuple(canonical),
            tool_call_count_delta=tool_calls)
        if idempotency_key:
            self.idempotent[idempotency_key] = result
        return result


def _db(tmp_path):
    store = AppendStore()
    db = SessionDB(db_path=tmp_path / "shadow.db", conversation_store=store)
    db.create_session("s1", source="cli")
    return db, store


def _local_counts(db):
    messages = db._read_one("SELECT COUNT(*) FROM messages")[0]
    session = db._read_one("SELECT message_count, tool_call_count FROM sessions WHERE id = ?", ("s1",))
    return messages, session["message_count"], session["tool_call_count"]


def test_turn_flush_persists_only_to_external_store_and_stamps_live_messages(tmp_path):
    db, store = _db(tmp_path)
    try:
        batch_rows = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "draft", "tool_calls": [{"id": "call-1"}]},
        ]
        live = [deepcopy(row) for row in batch_rows]
        agent = SimpleNamespace(_session_db=db, session_id="s1")

        _db_flush_write(agent, batch_rows, live, live)

        assert len(store.messages["s1"]) == 2
        assert _local_counts(db) == (0, 2, 1)
        assert all(message["_db_persisted"] is True for message in live)
        assert [message["_row_id"] for message in live] == [101, 102]
        assert live[1]["content"] == "canonical"
    finally:
        db.close()


def test_provider_failure_or_conflict_never_falls_back_to_sqlite(tmp_path):
    for mode in ("failure", "conflict"):
        db, store = _db(tmp_path / mode)
        try:
            if mode == "failure":
                store.fail = RuntimeError("provider offline")
                expected = RuntimeError
            else:
                store.conflict_on_append = True
                expected = ConversationConflictError
            row = {"role": "user", "content": mode}
            with pytest.raises(expected):
                db.append_messages_batch("s1", [row])
            assert _local_counts(db) == (0, 0, 0)
            assert "_row_id" not in row
        finally:
            db.close()


def test_single_append_uses_external_canonical_id(tmp_path):
    db, store = _db(tmp_path)
    try:
        row_id = db.append_message("s1", "assistant", "draft")
        assert row_id == 101
        assert store.messages["s1"][0]["content"] == "canonical"
        assert _local_counts(db) == (0, 1, 0)
    finally:
        db.close()


def test_reopened_provider_history_is_not_reappended(tmp_path):
    db, store = _db(tmp_path)
    db.append_messages_batch("s1", [{"role": "user", "content": "hello"}])
    db.close()

    reopened = SessionDB(db_path=tmp_path / "shadow.db", conversation_store=store)
    try:
        history = reopened.get_messages_as_conversation("s1")
        assert history[0]["_db_persisted"] is True
        agent = SimpleNamespace(
            session_id="s1", _last_flushed_db_idx=0,
            _flushed_db_message_session_id=None, _flushed_db_message_ids=set(),
            _db_flush_scan_prefix=None,
        )
        rows, messages = _db_flush_collect(agent, history, history)
        assert rows == []
        assert messages == []
        assert reopened._read_one("SELECT COUNT(*) FROM messages")[0] == 0
    finally:
        reopened.close()


def test_acp_owner_does_not_double_write_external_transcript(tmp_path):
    db, store = _db(tmp_path)
    calls = []
    try:
        agent = SimpleNamespace(
            model="test-model", provider=None, base_url=None, api_mode=None,
            _session_db=db, _session_db_created=True,
        )
        manager = SessionManager(agent_factory=lambda **_kwargs: agent, db=db)
        state = manager.create_session(cwd=".")
        state.history.append({"role": "user", "content": "hello"})
        db.create_session(state.session_id, source="acp")
        db.append_messages_batch(state.session_id, [deepcopy(state.history[0])])
        db.replace_messages = lambda *_args, **_kwargs: calls.append("replace")

        manager.save_session(state.session_id)

        assert calls == []
        assert len(store.messages[state.session_id]) == 1
    finally:
        db.close()


def test_delegation_delivery_is_idempotent_without_local_transcript(tmp_path):
    db, store = _db(tmp_path)
    metadata = {"delegation_id": "d1", "delivery_notice": "done"}
    try:
        first = db.append_delegation_delivery("s1", "result", metadata)
        second = db.append_delegation_delivery("s1", "result", metadata)
        assert first == second == 101
        assert len(store.messages["s1"]) == 1
        assert _local_counts(db) == (0, 1, 0)
    finally:
        db.close()
