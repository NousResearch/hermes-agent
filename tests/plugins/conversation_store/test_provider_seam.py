from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from conversation_store import ConversationStore, ConversationStoreUnavailableError
import hermes_state_registry as registry
import plugins.conversation_store as stores


class FakeEntryPoint:
    group = stores.ENTRY_POINTS_GROUP

    def __init__(self, name, loaded):
        self.name = name
        self._loaded = loaded

    def load(self):
        return self._loaded


class FakeEntryPoints(list):
    def select(self, *, group):
        return [ep for ep in self if ep.group == group]


class FakeStore(ConversationStore):
    created = []

    def __init__(self, name="fake", available=True):
        self._name = name
        self.available = available
        self.initialized_home = None
        self.closed = False
        type(self).created.append(self)

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def unavailable_reason(self):
        return "not configured"

    def initialize(self, *, hermes_home: Path):
        self.initialized_home = Path(hermes_home)

    def close(self):
        self.closed = True


@pytest.fixture
def entry_points(monkeypatch):
    points = FakeEntryPoints()
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: points)
    FakeStore.created.clear()
    return points


def _configure(home: Path, name="fake"):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"sessions:\n  store: {name}\n", encoding="utf-8")


def test_sqlite_is_the_zero_config_builtin(tmp_path, monkeypatch, entry_points):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert stores.configured_conversation_store_name() == "sqlite"
    assert stores.load_configured_conversation_store() is None
    assert stores.list_conversation_store_names() == ["sqlite"]


def test_named_entry_point_initializes_for_active_profile(tmp_path, monkeypatch, entry_points):
    _configure(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry_points.append(FakeEntryPoint("fake", FakeStore))

    store = stores.load_configured_conversation_store()

    assert isinstance(store, FakeStore)
    assert store.initialized_home == tmp_path
    assert not store.closed


def test_standard_register_callback_shape_is_supported(tmp_path, monkeypatch, entry_points):
    _configure(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def register(ctx):
        ctx.register_conversation_store(FakeStore())

    entry_points.append(FakeEntryPoint("fake", SimpleNamespace(register=register)))
    assert isinstance(stores.load_configured_conversation_store(), FakeStore)


def test_named_missing_or_unavailable_store_fails_closed(tmp_path, monkeypatch, entry_points):
    _configure(tmp_path, "missing")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(ConversationStoreUnavailableError, match="not installed"):
        stores.load_configured_conversation_store()

    _configure(tmp_path, "fake")
    entry_points.append(FakeEntryPoint("fake", lambda: FakeStore(available=False)))
    with pytest.raises(ConversationStoreUnavailableError, match="not configured"):
        stores.load_configured_conversation_store()


def test_bare_default_session_db_uses_configured_store(tmp_path, monkeypatch, entry_points):
    import hermes_state
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _configure(tmp_path)
    entry_points.append(FakeEntryPoint("fake", FakeStore))
    token = set_hermes_home_override(tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    try:
        db = hermes_state.SessionDB()
        try:
            assert isinstance(db.conversation_store, FakeStore)
            assert db.conversation_store.initialized_home == tmp_path
        finally:
            store = db.conversation_store
            db.close()
            assert store.closed
    finally:
        reset_hermes_home_override(token)


def test_shared_canonical_db_owns_provider_lifetime(tmp_path, monkeypatch, entry_points):
    _configure(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry_points.append(FakeEntryPoint("fake", FakeStore))
    registry.close_all()

    db = registry.acquire()
    store = db.conversation_store
    assert isinstance(store, FakeStore)
    assert db.uses_external_conversation_store
    assert store.initialized_home == tmp_path

    db.close()
    assert store.closed


def test_shared_acquire_fails_closed_when_configured_store_is_missing(tmp_path, monkeypatch, entry_points):
    import hermes_state
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _configure(tmp_path, "missing")
    registry.close_all()
    token = set_hermes_home_override(tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    try:
        with pytest.raises(ConversationStoreUnavailableError, match="not installed"):
            registry.acquire()
        assert registry.stats()["live_generations"] == 0
    finally:
        reset_hermes_home_override(token)


def test_explicit_db_path_stays_sqlite_only(tmp_path, monkeypatch, entry_points):
    home = tmp_path / "home"
    _configure(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry_points.append(FakeEntryPoint("fake", FakeStore))
    registry.close_all()

    db = registry.acquire(tmp_path / "recovery.db")
    try:
        assert db.conversation_store is None
        assert not db.uses_external_conversation_store
        assert FakeStore.created == []
    finally:
        registry.release(db)


def test_provider_instances_follow_profile_scope(tmp_path, monkeypatch, entry_points):
    import hermes_state
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    _configure(home_a)
    _configure(home_b)
    entry_points.append(FakeEntryPoint("fake", FakeStore))
    registry.close_all()

    token = set_hermes_home_override(home_a)
    try:
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home_a / "state.db")
        db_a = registry.acquire()
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(home_b)
    try:
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home_b / "state.db")
        db_b = registry.acquire()
    finally:
        reset_hermes_home_override(token)

    try:
        assert db_a is not db_b
        assert db_a.conversation_store.initialized_home == home_a
        assert db_b.conversation_store.initialized_home == home_b
    finally:
        registry.release(db_a)
        registry.release(db_b)
