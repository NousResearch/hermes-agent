from __future__ import annotations

from copy import deepcopy

import pytest

from conversation_store import (
    ConversationConflictError, ConversationMutationResult, ConversationRevision,
    ConversationSnapshot, ConversationStore,
)
from hermes_state import SessionDB


class RevisionStore(ConversationStore):
    def __init__(self):
        self.rev = 1
        self.meta = {"id": "s1", "title": "one"}
        self.messages = [{"id": 1, "role": "user", "content": "hello"}]

    @property
    def name(self):
        return "revision-store"

    def is_available(self):
        return True

    def get_revision(self, conversation_id):
        assert conversation_id == "s1"
        return ConversationRevision(self.rev)

    def snapshot(self, conversation_id, *, include_messages=False):
        return ConversationSnapshot(
            revision=self.get_revision(conversation_id),
            conversation=deepcopy(self.meta),
            messages=tuple(deepcopy(self.messages)) if include_messages else (),
        )

    def _commit(self, expected_revision, mutate):
        actual = self.get_revision("s1")
        if expected_revision != actual:
            raise ConversationConflictError(
                f"stale conversation revision: expected {expected_revision.value!r}, actual {actual.value!r}"
            )
        affected, ids = mutate()
        self.rev += 1
        return ConversationMutationResult(
            revision=ConversationRevision(self.rev), affected_count=affected, message_ids=tuple(ids))

    def append_messages(self, conversation_id, messages, *, expected_revision):
        def mutate():
            start = max((m["id"] for m in self.messages), default=0) + 1
            ids = list(range(start, start + len(messages)))
            self.messages.extend({"id": i, **deepcopy(m)} for i, m in zip(ids, messages))
            return len(messages), ids
        return self._commit(expected_revision, mutate)

    def update_conversation(self, conversation_id, changes, *, expected_revision):
        def mutate():
            self.meta.update(deepcopy(changes))
            return 1, ()
        return self._commit(expected_revision, mutate)

    def replace_messages(self, conversation_id, messages, *, expected_revision, **_kwargs):
        def mutate():
            self.messages = [{"id": i + 1, **deepcopy(m)} for i, m in enumerate(messages)]
            return len(messages), [m["id"] for m in self.messages]
        return self._commit(expected_revision, mutate)

    def rewind_to_message(self, conversation_id, message_id, *, expected_revision, **_kwargs):
        def mutate():
            before = len(self.messages)
            self.messages = [m for m in self.messages if m["id"] <= message_id]
            return before - len(self.messages), ()
        return self._commit(expected_revision, mutate)

    def publish_compaction(self, conversation_id, messages, *, expected_revision, **_kwargs):
        return self.replace_messages(
            conversation_id, messages, expected_revision=expected_revision)


@pytest.fixture
def db_store(tmp_path):
    store = RevisionStore()
    db = SessionDB(db_path=tmp_path / "shadow.db", conversation_store=store)
    try:
        yield db, store
    finally:
        db.close()


def test_session_db_exposes_opaque_revision_and_atomic_snapshot(db_store):
    db, _store = db_store
    assert db.conversation_revision("s1") == ConversationRevision(1)
    snap = db.conversation_snapshot("s1", include_messages=True)
    assert snap.revision == ConversationRevision(1)
    assert snap.conversation["title"] == "one"
    assert snap.messages[0]["content"] == "hello"


def test_sqlite_mode_has_no_external_revision(tmp_path):
    db = SessionDB(db_path=tmp_path / "sqlite.db")
    try:
        assert db.conversation_revision("missing") is None
        assert db.conversation_snapshot("missing") is None
    finally:
        db.close()


@pytest.mark.parametrize("mutation", ["append", "metadata", "replace", "rewind", "compact"])
def test_stale_mutation_is_rejected_without_state_change(db_store, mutation):
    db, store = db_store
    stale = db.conversation_snapshot("s1", include_messages=True)
    store.update_conversation("s1", {"external": True}, expected_revision=stale.revision)
    before = (deepcopy(store.meta), deepcopy(store.messages), store.rev)

    calls = {
        "append": lambda: store.append_messages("s1", [{"role": "assistant", "content": "x"}], expected_revision=stale.revision),
        "metadata": lambda: store.update_conversation("s1", {"title": "stale"}, expected_revision=stale.revision),
        "replace": lambda: store.replace_messages("s1", [{"role": "user", "content": "stale"}], expected_revision=stale.revision),
        "rewind": lambda: store.rewind_to_message("s1", 1, expected_revision=stale.revision),
        "compact": lambda: store.publish_compaction("s1", [{"role": "user", "content": "summary"}], expected_revision=stale.revision),
    }
    with pytest.raises(ConversationConflictError):
        calls[mutation]()
    assert (store.meta, store.messages, store.rev) == before


def test_successful_mutation_advances_revision(db_store):
    db, store = db_store
    revision = db.conversation_revision("s1")
    result = store.append_messages(
        "s1", [{"role": "assistant", "content": "world"}], expected_revision=revision)
    assert result.revision == ConversationRevision(2)
    assert result.message_ids == (2,)
    assert db.conversation_revision("s1") == ConversationRevision(2)
