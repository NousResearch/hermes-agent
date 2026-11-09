"""A plugin context engine binds the deprecated ``hermes_state.SessionDB`` over the agent's profile, the object main
handed it as ``session_db``, and gets the agent's store as ``persistence`` in each hook that accepts that keyword; the
built-in compressor keeps the agent's store handle."""

import warnings
from unittest.mock import patch

import pytest

import hermes_state
from agent.context_engine import ContextEngine
from agent.conversation_compression import _adopt_live_compression_child
from persistence_assembly import open_persistence, profile_scope
from run_agent import AIAgent
from tests.fixtures.storage_seed import seed_messages, seed_session


class _PluginEngine(ContextEngine):
    """A plugin engine that reads its bound session through main's ``SessionDB.get_session``."""

    def __init__(self):
        super().__init__()
        self.bound = []
        self.started = []

    @property
    def name(self):
        return "stub-plugin"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None):
        return messages

    def bind_session_state(self, session_db=None, session_id=""):
        self.bound.append((session_db, session_db.get_session(session_id)))

    def on_session_start(self, session_id, **kwargs):
        self.started.append((session_id, kwargs))


class _StorePluginEngine(_PluginEngine):
    """A plugin engine whose bind also names ``persistence``."""

    def __init__(self):
        super().__init__()
        self.stores = []

    def bind_session_state(self, session_db=None, session_id="", persistence=None):
        super().bind_session_state(session_db, session_id)
        self.stores.append(persistence)


def _agent(store, session_id, engine=None):
    kwargs = dict(api_key="test-key", base_url="http://127.0.0.1:9/v1", model="test/model", quiet_mode=True,
                  skip_context_files=True, skip_memory=True, skip_background_review=True,
                  session_db=store, persistence=store, session_id=session_id)
    if engine is None:
        return AIAgent(**kwargs)
    with patch("agent.agent_init._select_context_engine", return_value=engine):
        return AIAgent(**kwargs)


@pytest.fixture
def store():
    handle = open_persistence(profile_scope(), eager=True)
    seed_session(handle, "s1", "cli", model="m1", title="alpha plan")
    yield handle
    handle.close()


def test_a_plugin_engine_binds_the_deprecated_session_db_and_reads_mains_answer(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", False)
    engine = _PluginEngine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = _agent(store, "s1", engine)
    [warning] = [w for w in caught if w.category is DeprecationWarning]
    text = str(warning.message)
    assert f"{__name__}._PluginEngine" in text and "persistence=" in text and "v0.27.0" in text
    try:
        [(session_db, row)] = engine.bound
        assert isinstance(session_db, hermes_state.SessionDB)
        assert row == store.sessions.get("s1").as_session_row()
        assert (row["id"], row["source"], row["model"], row["title"]) == ("s1", "cli", "m1", "alpha plan")
    finally:
        agent.close()


def test_compression_adoption_hands_a_plugin_engine_the_same_session_db(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", True)
    store.sessions.end("s1", "compression")
    seed_session(store, "s1-child", "cli", parent_session_id="s1")
    seed_messages(store, "s1-child", {"role": "user", "content": "compacted tail"})
    engine = _PluginEngine()
    agent = _agent(store, "s1", engine)
    try:
        [(session_db, _row)] = engine.bound
        assert _adopt_live_compression_child(agent, store, "s1", store)
        [(child, kwargs)] = engine.started[-1:]
        assert child == "s1-child"
        assert kwargs["session_db"] is session_db
        assert kwargs["persistence"] is store
        assert kwargs["session_db"].get_session("s1-child")["parent_session_id"] == "s1"
    finally:
        agent.close()


def test_compression_adoption_binds_an_engine_without_on_session_start_with_the_agents_store(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", True)
    store.sessions.end("s1", "compression")
    seed_session(store, "s1-child", "cli", parent_session_id="s1")
    seed_messages(store, "s1-child", {"role": "user", "content": "compacted tail"})
    engine = _StorePluginEngine()
    monkeypatch.setattr(engine, "on_session_start", None)  # adoption falls back to bind_session_state
    agent = _agent(store, "s1", engine)
    try:
        assert _adopt_live_compression_child(agent, store, "s1", store)
        assert engine.bound[-1][1]["id"] == "s1-child"
        assert engine.stores[-1] is store
    finally:
        agent.close()


def test_a_plugin_engine_whose_hooks_take_persistence_is_not_warned(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", False)
    engine = _StorePluginEngine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = _agent(store, "s1", engine)
    try:
        assert [w for w in caught if w.category is DeprecationWarning] == []
        assert hermes_state._warned is False
        assert isinstance(engine.bound[0][0], hermes_state._SessionDB)
    finally:
        agent.close()


def test_a_plugin_engine_whose_bind_names_persistence_gets_the_agents_store(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", True)
    engine = _StorePluginEngine()
    agent = _agent(store, "s1", engine)
    try:
        [(session_db, row)] = engine.bound
        assert isinstance(session_db, hermes_state.SessionDB)
        assert engine.stores == [store]
        assert row == store.sessions.get("s1").as_session_row()
    finally:
        agent.close()


def test_the_built_in_compressor_keeps_the_agents_store_handle(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", False)
    agent = _agent(store, "s1")
    try:
        assert agent.context_compressor._session_db is store
        assert agent.context_compressor._persistence is store
        assert hermes_state._warned is False
    finally:
        agent.close()


def test_a_reset_rebinds_a_plugin_engine_to_the_same_session_db(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", True)
    seed_session(store, "s2", "cli", model="m2")
    engine = _PluginEngine()
    agent = _agent(store, "s1", engine)
    try:
        [(session_db, _row)] = engine.bound
        agent.session_id = "s2"
        agent.reset_session_state()
        assert engine.bound[-1][0] is session_db
        assert engine.bound[-1][1]["model"] == "m2"
    finally:
        agent.close()


def test_a_reset_hands_a_plugin_engine_whose_bind_names_persistence_the_agents_store(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", True)
    seed_session(store, "s2", "cli", model="m2")
    engine = _StorePluginEngine()
    agent = _agent(store, "s1", engine)
    try:
        [(session_db, _row)] = engine.bound
        agent.session_id = "s2"
        agent.reset_session_state()
        assert engine.bound[-1][0] is session_db
        assert engine.bound[-1][1]["model"] == "m2"
        assert engine.stores == [store, store]
    finally:
        agent.close()


def test_the_plugin_engines_session_db_closes_with_the_agent(monkeypatch, store):
    monkeypatch.setattr(hermes_state, "_warned", True)
    engine = _PluginEngine()
    agent = _agent(store, "s1", engine)
    [(session_db, _row)] = engine.bound
    closes = []
    monkeypatch.setattr(session_db, "close", lambda close=session_db.close: closes.append(close()))
    agent.close()
    agent.close()
    assert closes == [None]
    assert store.sessions.get("s1") is not None
