"""An explicit ``/resume <delegate-child>`` sticks: the next message stays on the resumed session.

The #138210 route guard reads a ``_delegate_from`` row as a stale capture and moves the chat off it.
A session the user deliberately switched the chat onto is the conversation, so ``switch_session``
adopts it (drops the marker) and the route self-heal keeps it. Real ``SessionStore`` + ``SessionDB``.
"""

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


def test_a_resumed_delegate_child_keeps_the_route(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    db.create_session(child, "subagent", parent_session_id=entry.session_id,
                      model_config={"_delegate_from": entry.session_id})

    assert store.switch_session(entry.session_key, child) is not None
    assert not store._is_session_ended_in_db(child, entry.session_key), "self-heal would evict the resumed chat"
    assert store.get_or_create_session(source).session_id == child
    db.close()
