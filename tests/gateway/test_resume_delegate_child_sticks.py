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


def test_adoption_keeps_the_childs_delegate_lineage(tmp_path):
    """``_delegate_from`` is lineage, not a routing flag: the still-running worker's compression publish
    re-sends it and a compression-ended parent would otherwise read the child as its continuation."""
    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    db.create_session(child, "subagent", parent_session_id=entry.session_id,
                      model_config={"_delegate_from": entry.session_id})
    db.end_session(entry.session_id, "compression")

    assert store.switch_session(entry.session_key, child) is not None

    assert db.get_session_model_config_value(child, "_delegate_from") == entry.session_id
    assert not db._is_compression_child_row(db.get_session(child))
    assert store.get_or_create_session(source).session_id == child
    db.close()


def test_a_failed_adoption_leaves_the_old_route_in_place(tmp_path, monkeypatch):
    """The marker write fails (a busy writer): the chat must stay on its own conversation, not point
    at a still-marked child the next message would reject."""
    import pytest

    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    db.create_session(child, "subagent", parent_session_id=entry.session_id,
                      model_config={"_delegate_from": entry.session_id})

    def locked(*_a, **_k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "patch_session_model_config", locked)
    with pytest.raises(RuntimeError):
        store.switch_session(entry.session_key, child)

    assert store.get_or_create_session(source).session_id == entry.session_id
    assert db.get_session(entry.session_id)["end_reason"] is None
    db.close()


def test_a_resumed_delegate_keeps_the_route_through_a_compression_rotation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    db.create_session(child, "subagent", parent_session_id=entry.session_id,
                      model_config={"_delegate_from": entry.session_id})
    assert store.switch_session(entry.session_key, child) is not None

    tip = child + "-c1"
    assert db.try_acquire_compression_lock(child, "holder", ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=child, child_session_id=tip, source="subagent", system_prompt="p",
        messages=[{"role": "user", "content": "summary"}], compression_lock_holder="holder",
        model_config={"_delegate_from": entry.session_id})
    from gateway.session_lifecycle import is_delegate_child_row

    # The route guard reads the tip compression published, not a row switch_session touched again.
    row = db.get_session(tip)
    assert db.get_session_model_config_value(tip, "_delegate_from") == entry.session_id
    assert not is_delegate_child_row(row), "the rotated tip lost the user's choice"
    db.close()
