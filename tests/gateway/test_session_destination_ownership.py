import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionOwnershipConflict,
    SessionSource,
    SessionStore,
)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    result = SessionStore(
        sessions_dir=tmp_path / "sessions",
        config=GatewayConfig(write_sessions_json=False),
    )
    if result._db is not None:
        result._db.close()
    result._db = None
    result._loaded = True
    return result


def _source(*, chat_type="dm", user_id="8650058832", thread_id=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8650058832",
        chat_type=chat_type,
        user_id=user_id,
        thread_id=thread_id,
    )


def test_unthreaded_dm_rejects_synthetic_group_owner(store):
    dm = store.get_or_create_session(_source())

    with pytest.raises(SessionOwnershipConflict, match="distinct thread/topic"):
        store.get_or_create_session(_source(chat_type="group"))

    assert list(store._entries) == [dm.session_key]


def test_existing_unthreaded_group_rejects_second_dm_owner(store):
    group = store.get_or_create_session(_source(chat_type="group"))

    with pytest.raises(SessionOwnershipConflict, match="destination ownership conflict"):
        store.get_or_create_session(_source())

    assert list(store._entries) == [group.session_key]


def test_group_per_user_sessions_remain_allowed_without_dm_owner(store):
    alice = store.get_or_create_session(_source(chat_type="group", user_id="alice"))
    bob = store.get_or_create_session(_source(chat_type="group", user_id="bob"))

    assert alice.session_key != bob.session_key
    assert len(store._entries) == 2


def test_distinct_thread_can_own_separate_session(store):
    root = store.get_or_create_session(_source())
    topic = store.get_or_create_session(
        _source(chat_type="dm", thread_id="topic-77")
    )

    assert root.session_key != topic.session_key
    assert len(store._entries) == 2


def test_same_dm_routing_key_reuses_owner(store):
    first = store.get_or_create_session(_source())
    second = store.get_or_create_session(_source())

    assert first is second
    assert len(store._entries) == 1


def test_concurrent_conflicting_publications_create_exactly_one_owner(store):
    barrier = threading.Barrier(2)

    def synchronized_recovery(**_kwargs):
        barrier.wait(timeout=5)
        return None

    store._query_recoverable_session = synchronized_recovery
    sources = [_source(), _source(chat_type="group")]

    outcomes = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.get_or_create_session, src) for src in sources]
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except BaseException as exc:  # assert the exact conflict below
                outcomes.append(exc)

    entries = [value for value in outcomes if not isinstance(value, BaseException)]
    errors = [value for value in outcomes if isinstance(value, BaseException)]
    assert len(entries) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], SessionOwnershipConflict)
    assert len(store._entries) == 1
