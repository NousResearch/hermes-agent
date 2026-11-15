"""An explicit ``/resume <delegate-child>`` sticks: the next message stays on the resumed session.

The #138210 route guard reads a ``_delegate_from`` row as a stale capture and moves the chat off it.
A session the user deliberately switched the chat onto is the conversation, so ``switch_session``
adopts it (drops the marker) and the route self-heal keeps it. Real ``SessionStore`` + the store it
opens through the storage interface.
"""

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


def test_a_resumed_delegate_child_keeps_the_route(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store.persistence
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    seed_session(db, child, "subagent", parent_session_id=entry.session_id,
                 model_config={"_delegate_from": entry.session_id})

    assert store.switch_session(entry.session_key, child) is not None
    assert not store._is_session_ended_in_db(child, entry.session_key), "self-heal would evict the resumed chat"
    assert store.get_or_create_session(source).session_id == child
    db.close()


def test_adoption_keeps_the_childs_delegate_lineage(tmp_path):
    """``_delegate_from`` is lineage, not a routing flag: the still-running worker's compression publish
    re-sends it and a compression-ended parent would otherwise read the child as its continuation."""
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store.persistence
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    seed_session(db, child, "subagent", parent_session_id=entry.session_id,
                 model_config={"_delegate_from": entry.session_id})
    db.sessions.end(entry.session_id, "compression")

    assert store.switch_session(entry.session_key, child) is not None

    assert db.sessions.get_model_config_value(child, "_delegate_from") == entry.session_id
    assert not db.lineage.is_compression_child(db.sessions.get(child))
    assert store.get_or_create_session(source).session_id == child
    db.close()


def test_a_failed_adoption_leaves_the_old_route_in_place(tmp_path, monkeypatch):
    """The marker write fails (a busy writer): the chat must stay on its own conversation, not point
    at a still-marked child the next message would reject."""
    import pytest

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store.persistence
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    seed_session(db, child, "subagent", parent_session_id=entry.session_id,
                 model_config={"_delegate_from": entry.session_id})

    def locked(*_a, **_k):
        raise RuntimeError("database is locked")

    persistence = store._persistence_for_key(entry.session_key)
    assert persistence is not None
    monkeypatch.setattr(persistence.sessions, "patch_model_config", locked)
    with pytest.raises(RuntimeError):
        store.switch_session(entry.session_key, child)

    assert store.get_or_create_session(source).session_id == entry.session_id
    assert session_row(db, entry.session_id)["end_reason"] is None
    db.close()


def test_a_resumed_delegate_keeps_the_route_through_a_compression_rotation(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store.persistence
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    seed_session(db, child, "subagent", parent_session_id=entry.session_id,
                 model_config={"_delegate_from": entry.session_id})
    assert store.switch_session(entry.session_key, child) is not None

    tip = child + "-c1"
    assert db.coordination.try_acquire_compression_lock(session_id=child, holder="holder", ttl_seconds=60)
    db.lineage.publish_compression_child(
        child, CompressionChild(session_id=tip, source="subagent", system_prompt="p",
                                model_config={"_delegate_from": entry.session_id}),
        [NewMessage.from_live({"role": "user", "content": "summary"})], compression_lock_holder="holder")
    from gateway.session_lifecycle import is_delegate_child_row

    # The route guard reads the tip compression published, not a row switch_session touched again.
    row = session_row(db, tip)
    assert db.sessions.get_model_config_value(tip, "_delegate_from") == entry.session_id
    assert not is_delegate_child_row(row), "the rotated tip lost the user's choice"
    db.close()


def test_a_refused_switch_never_strips_an_overlapping_successful_adoption(tmp_path, monkeypatch):
    """Two switches onto one delegate row overlap: a stale-snapshot switch that is refused must leave
    the marker the successful one relies on, or the next message moves the chat off the resumed row."""
    import threading

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store.persistence
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="1", chat_type="dm")
    entry = store.get_or_create_session(source)
    child = entry.session_id + "-worker"
    seed_session(db, child, "subagent", parent_session_id=entry.session_id,
                 model_config={"_delegate_from": entry.session_id})
    real_adopt = store._adopt_delegate_row
    armed = {"on": True}
    winner = []

    def overlapping_adopt(key, sid):
        adopted = real_adopt(key, sid)
        if armed.pop("on", False):  # the refused switch's adoption: the winner runs in between
            thread = threading.Thread(target=lambda: winner.append(store.switch_session(key, child)))
            thread.start()
            thread.join(timeout=1)
        return adopted

    monkeypatch.setattr(store, "_adopt_delegate_row", overlapping_adopt)
    assert store.switch_session(entry.session_key, child, expected_session_id="a-stale-snapshot") is None
    armed.pop("on", None)
    if not winner:
        winner.append(store.switch_session(entry.session_key, child))

    assert winner[0] is not None and winner[0].session_id == child
    assert store.get_or_create_session(source).session_id == child
    db.close()
