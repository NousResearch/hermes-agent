from __future__ import annotations

from copy import deepcopy

import pytest

from conversation_store import (
    ConversationConflictError,
    ConversationMutationResult,
    ConversationRevision,
    ConversationSnapshot,
    ConversationStore,
)


class CasStore(ConversationStore):
    def __init__(self):
        self.record = {"id": "s1", "title": "Original"}
        self.messages = [{"id": 1, "role": "user", "content": "hello"}]
        self.revision = 1

    @property
    def name(self):
        return "cas"

    def is_available(self):
        return True

    def get_revision(self, conversation_id):
        assert conversation_id == "s1"
        return ConversationRevision(self.revision)

    def snapshot(self, conversation_id, *, include_messages=False):
        assert conversation_id == "s1"
        return ConversationSnapshot(
            revision=self.get_revision(conversation_id),
            conversation=deepcopy(self.record),
            messages=tuple(deepcopy(self.messages)) if include_messages else (),
        )

    def _commit(self, expected_revision, mutate):
        if expected_revision != ConversationRevision(self.revision):
            raise ConversationConflictError("stale conversation revision")
        affected, message_ids = mutate()
        self.revision += 1
        return ConversationMutationResult(
            ConversationRevision(self.revision), affected, tuple(message_ids))

    def append_messages(self, conversation_id, messages, *, expected_revision):
        def mutate():
            ids = []
            for message in messages:
                row = dict(message)
                row["id"] = max((m["id"] for m in self.messages), default=0) + 1
                self.messages.append(row)
                ids.append(row["id"])
            return len(ids), ids
        return self._commit(expected_revision, mutate)

    def update_conversation(self, conversation_id, changes, *, expected_revision):
        return self._commit(expected_revision, lambda: (self.record.update(changes) is None, ()))

    def replace_messages(
        self, conversation_id, messages, *, expected_revision, active_only=False, archive_dropped=False,
    ):
        def mutate():
            self.messages = [dict(message, id=i + 1) for i, message in enumerate(messages)]
            return len(self.messages), [m["id"] for m in self.messages]
        return self._commit(expected_revision, mutate)

    def rewind_to_message(
        self, conversation_id, message_id, *, expected_revision, preserve_compaction_handoff=False,
    ):
        def mutate():
            before = len(self.messages)
            self.messages = [m for m in self.messages if m["id"] < message_id]
            return before - len(self.messages), ()
        return self._commit(expected_revision, mutate)

    def publish_compaction(
        self, conversation_id, messages, *, expected_revision, model_config_patch=None, tail_count=0,
    ):
        return self.replace_messages(
            conversation_id, messages, expected_revision=expected_revision)


def test_snapshot_carries_one_atomic_revision():
    store = CasStore()
    snapshot = store.snapshot("s1", include_messages=True)

    assert snapshot.revision == ConversationRevision(1)
    assert snapshot.conversation["title"] == "Original"
    assert snapshot.messages[0]["content"] == "hello"


def test_successful_mutation_advances_revision_and_reports_ids():
    store = CasStore()
    result = store.append_messages(
        "s1", [{"role": "assistant", "content": "hi"}],
        expected_revision=ConversationRevision(1),
    )

    assert result.revision == ConversationRevision(2)
    assert result.affected_count == 1
    assert result.message_ids == (2,)
    assert store.get_revision("s1") == ConversationRevision(2)


def test_stale_append_fails_without_changing_state():
    store = CasStore()
    first = store.append_messages(
        "s1", [{"role": "assistant", "content": "winner"}],
        expected_revision=ConversationRevision(1),
    )
    before = deepcopy(store.messages)

    with pytest.raises(ConversationConflictError):
        store.append_messages(
            "s1", [{"role": "assistant", "content": "stale"}],
            expected_revision=ConversationRevision(1),
        )

    assert first.revision == ConversationRevision(2)
    assert store.messages == before
    assert store.get_revision("s1") == ConversationRevision(2)


@pytest.mark.parametrize("operation", ["metadata", "replace", "rewind", "compact"])
def test_every_destructive_contract_rejects_stale_revision(operation):
    store = CasStore()
    stale = ConversationRevision(0)
    before_record = deepcopy(store.record)
    before_messages = deepcopy(store.messages)

    with pytest.raises(ConversationConflictError):
        if operation == "metadata":
            store.update_conversation("s1", {"title": "stale"}, expected_revision=stale)
        elif operation == "replace":
            store.replace_messages("s1", [], expected_revision=stale)
        elif operation == "rewind":
            store.rewind_to_message("s1", 1, expected_revision=stale)
        else:
            store.publish_compaction("s1", [], expected_revision=stale)

    assert store.record == before_record
    assert store.messages == before_messages
    assert store.get_revision("s1") == ConversationRevision(1)
