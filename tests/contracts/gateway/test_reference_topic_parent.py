"""Topic bindings require an existing session and refuse atomically."""

import pytest

from persistence import SessionRowMissing
from persistence.conversation.sessions import NewSession


def test_topic_binding_does_not_create_its_parent(backend, scope_a):
    store_a = backend.open(scope_a)
    bind = {"chat_id": "chat", "thread_id": "thread", "user_id": "user", "session_key": "key", "session_id": "s"}
    # A missing session row refuses the bind (classified session_row_missing).
    with pytest.raises(SessionRowMissing):
        store_a.routing.bind_topic(**bind)
    assert store_a.sessions.get("s") is None
    assert store_a.routing.get_topic_binding(chat_id="chat", thread_id="thread") is None
    store_a.sessions.create(NewSession("s", "telegram"))
    store_a.routing.bind_topic(**bind)
    assert store_a.routing.get_topic_binding(chat_id="chat", thread_id="thread").session_id == "s"
