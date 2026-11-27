"""Contract cases for ``NewMessage.restored_from_store``: a batch row the writer already saw durable is not inserted again while an active row carries its uid and replays exactly like it; without the mark, or edited, it inserts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from persistence.conversation.messages import NewMessage, RowDisposition
from persistence.conversation.sessions import NewSession

SID = "restored-contract"


@pytest.fixture
def store(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(SID, "cli"))
    return store


def _seed(store) -> tuple:
    result = store.messages.append_batch(SID, [
        NewMessage("user", "first turn", timestamp=1000.0),
        NewMessage("assistant", "first reply", timestamp=1001.0, api_content="first reply [wire]"),
    ])
    return tuple(outcome.id for outcome in result.outcomes)


def _handed_back(store, *, restored: bool) -> list[NewMessage]:
    """The stored history as a restore hands it back across a wire: uids and payload kept, no row address or version."""
    live = [{k: v for k, v in m.as_openai_message().items() if not k.startswith("_")} for m in store.messages.read_conversation(SID)]
    assert all(m.get("message_uid") for m in live)
    return [replace(NewMessage.from_live(m), restored_from_store=restored) for m in live]


def test_a_restored_copy_of_an_active_row_is_not_inserted_again(store):
    ids = _seed(store)
    result = store.messages.append_batch(SID, _handed_back(store, restored=True))
    assert [(o.id, o.disposition) for o in result.outcomes] == [(i, RowDisposition.REPAIRED) for i in ids]
    assert result.inserted == 0
    assert [(m.role, m.content) for m in store.messages.read_conversation(SID)] == [("user", "first turn"), ("assistant", "first reply")]


def test_the_same_copy_without_the_mark_is_inserted(store):
    _seed(store)
    result = store.messages.append_batch(SID, _handed_back(store, restored=False))
    assert result.inserted == 2
    assert [m.content for m in store.messages.read_conversation(SID)] == ["first turn", "first reply"] * 2


def test_an_edited_restored_copy_is_inserted(store):
    ids = _seed(store)
    copies = _handed_back(store, restored=True)
    copies[-1] = replace(copies[-1], content="first reply (edited)")
    result = store.messages.append_batch(SID, copies)
    assert result.outcomes[0].id == ids[0] and result.outcomes[0].disposition is RowDisposition.REPAIRED
    assert result.outcomes[1].disposition is RowDisposition.APPENDED
    assert [m.content for m in store.messages.read_conversation(SID)] == ["first turn", "first reply", "first reply (edited)"]


def test_a_restored_copy_with_a_new_api_content_is_inserted(store):
    _seed(store)
    copies = _handed_back(store, restored=True)
    copies[-1] = replace(copies[-1], api_content="first reply [wire v2]")
    assert store.messages.append_batch(SID, copies).inserted == 1


def test_a_restored_copy_of_a_rewound_row_is_inserted(store):
    ids = _seed(store)
    copies = _handed_back(store, restored=True)
    assert store.messages.deactivate_message(SID, ids[1]) is True
    result = store.messages.append_batch(SID, copies)
    assert [o.disposition for o in result.outcomes] == [RowDisposition.REPAIRED, RowDisposition.APPENDED]
    assert [m.content for m in store.messages.read_conversation(SID)] == ["first turn", "first reply"]


def test_from_live_reads_the_restored_mark():
    assert NewMessage.from_live({"role": "user", "content": "x", "_restored_from_store": True}).restored_from_store is True
    assert NewMessage.from_live({"role": "user", "content": "x", "_restored_from_store": 1}).restored_from_store is False
    assert NewMessage.from_live({"role": "user", "content": "x"}).restored_from_store is False
