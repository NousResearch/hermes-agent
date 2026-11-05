"""The agent keeps the storage interface handle it is given and closes it with a store it owns."""
from types import SimpleNamespace

from hermes_constants import get_hermes_home
from run_agent import AIAgent


class _Handle:
    closed = 0
    read_only = False

    def close(self):
        self.closed += 1


def _agent(**kwargs):
    return AIAgent(api_key="test-key", base_url="http://127.0.0.1:9/v1", model="test/model", quiet_mode=True,
                   skip_context_files=True, skip_memory=True, skip_background_review=True, **kwargs)


def test_constructed_agent_keeps_the_handle_and_opens_nothing():
    handle = _Handle()
    agent = _agent(persistence=handle)
    try:
        assert agent.persistence is handle
        assert not (get_hermes_home() / "state.db").exists()
    finally:
        agent.close()
    assert handle.closed == 0  # a caller-supplied store and its handle stay the caller's


def test_owned_store_releases_its_handle_with_it():
    store, handle = _Handle(), _Handle()
    owner = SimpleNamespace(_session_db=store, session_id=None, _end_session_on_close=False,
                            _owns_session_db=True, persistence=handle)
    AIAgent._finalize_owned_session_row(owner)
    AIAgent._finalize_owned_session_row(owner)
    assert store.closed == 1 and handle.closed == 1 and owner.persistence is None


def test_a_failed_release_still_closes_the_handle():
    class _FailingStore:
        def close(self):
            raise OSError("disk I/O error")

    handle = _Handle()
    owner = SimpleNamespace(_session_db=_FailingStore(), session_id=None, _end_session_on_close=False,
                            _owns_session_db=True, persistence=handle)
    try:
        AIAgent._finalize_owned_session_row(owner)
    except OSError:
        pass
    else:
        raise AssertionError("the store's release error is the agent's to raise")
    assert handle.closed == 1 and owner.persistence is None


def test_recall_opens_a_writer_handle_and_holds_it_as_its_store():
    agent = _agent()
    try:
        store = agent._get_session_db_for_recall()
        assert store is agent.persistence and agent._owns_session_db is True and not store.read_only
        assert store.scope.profile_dir == get_hermes_home().resolve()
        assert (get_hermes_home() / "state.db").exists()  # opened eagerly, so a failure answers at the open
    finally:
        agent.close()
    assert agent.persistence is None and agent._owns_session_db is False


def test_recall_has_no_store_when_the_handle_cannot_open(monkeypatch):
    import persistence_assembly

    def fail(*args, **kwargs):
        raise ValueError("unknown persistence provider 'nope'")

    monkeypatch.setattr(persistence_assembly, "open_persistence", fail)
    agent = _agent()
    try:
        assert agent._get_session_db_for_recall() is None
        assert agent._session_db is None and agent._owns_session_db is False and agent.persistence is None
    finally:
        agent.close()


def test_lookup_reader_yields_the_label_reader_when_its_handle_cannot_open(monkeypatch):
    import persistence_assembly
    from hermes_cli import main

    persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=get_hermes_home()), eager=True).close()

    def fail(*args, **kwargs):
        raise ValueError("unknown persistence provider 'nope'")

    monkeypatch.setattr(persistence_assembly, "open_persistence_reader", fail)
    with main._session_db() as reader:
        assert reader is not None and reader.persistence is None
        assert reader.list_sessions_rich(limit=5) == []
