"""The SQLite batch append skips a restored-from-store copy of an active row that replays exactly like it, and stores one
whose content or replay sidecar changed.

The mark is a live-dict key the agent's flush sets on a row whose uid it already saw durable; the backend's batch
append reads it off the dicts it receives.
"""

import sqlite3

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.sqlite.conversation._store.messages import append_messages_batch
from persistence.sqlite.conversation._store.rules import RESTORED_FROM_STORE
from tests.persistence_sqlite.conftest import store_for

SID = "restored"


@pytest.fixture
def store(backend, scope_a):
    store = backend.open(scope_a)
    store.sessions.create(NewSession(SID, "cli"))
    return store


def _restored(store) -> list[dict]:
    """The stored history as a restore hands it back across a wire: uids kept, every underscore key gone, marked."""
    return [{**{k: v for k, v in m.as_openai_message().items() if not k.startswith("_")}, RESTORED_FROM_STORE: True}
            for m in store.messages.read_conversation(SID)]


def _active(scope) -> int:
    with sqlite3.connect(store_for(scope)) as conn:
        return conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1", (SID,)).fetchone()[0]


def test_a_restored_copy_of_an_active_row_is_not_inserted_again(store, scope_a):
    store.messages.append_batch(SID, [NewMessage("user", "first turn", timestamp=1000.0),
                                      NewMessage("assistant", "first reply", timestamp=1001.0)])
    restored = _restored(store)
    assert all(m.get("message_uid") for m in restored)

    append_messages_batch(store.holder(), SID, restored)

    assert _active(scope_a) == 2


def test_an_edited_restored_copy_lands(store):
    store.messages.append_batch(SID, [NewMessage("user", "q", timestamp=1.0), NewMessage("assistant", "answer", timestamp=2.0)])
    restored = _restored(store)
    restored[-1]["content"] = "answer (edited)"

    append_messages_batch(store.holder(), SID, restored)

    assert "answer (edited)" in [m.content for m in store.messages.read_conversation(SID)]


@pytest.mark.parametrize("field, before, after", [
    ("api_content", "wire bytes v1", "wire bytes v2"),
    ("codex_reasoning_items", [{"type": "reasoning", "encrypted_content": "enc-v1"}],
     [{"type": "reasoning", "encrypted_content": "enc-v2"}]),
], ids=["api_content", "codex_reasoning_items"])
def test_a_restored_copy_with_a_new_replay_sidecar_is_stored(store, field, before, after):
    store.messages.append_batch(SID, [NewMessage("user", "q", timestamp=1.0),
                                      NewMessage("assistant", "answer", timestamp=2.0, **{field: before})])
    restored = _restored(store)
    assert restored[-1].get(field) == before
    restored[-1][field] = after

    append_messages_batch(store.holder(), SID, restored)

    assert after in [getattr(m, field) for m in store.messages.read_conversation(SID)]
