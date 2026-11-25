"""Storage interface readers handed to a surface open no connection beyond the reader's own."""
import sqlite3

import pytest

import persistence_assembly
from hermes_constants import get_hermes_home
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope


@pytest.fixture
def connects(monkeypatch):
    calls = []
    real = sqlite3.connect

    def counting(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("database"))
        return real(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting)
    return calls


@pytest.fixture
def failing_opens(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unknown persistence provider 'nope'")

    monkeypatch.setattr(persistence_assembly, "open_persistence", fail)
    monkeypatch.setattr(persistence_assembly, "open_persistence_reader", fail)


def _seeded_store():
    path = get_hermes_home() / "state.db"
    open_persistence(profile_scope(), eager=True).close()
    return path


def test_gateway_session_store_reader_carries_an_unopened_handle(connects):
    from hermes_cli.gateway_session_store import GatewaySessionStore

    path = _seeded_store()
    before = len(connects)
    with open_persistence_reader(path) as bare:
        bare.ready()
    reader_only = len(connects) - before

    store = GatewaySessionStore(connection=object())
    before = len(connects)
    reader = store._db()
    assert len(connects) - before == reader_only
    assert reader.persistence is store.persistence
    assert store.persistence.scope.profile_dir == get_hermes_home().resolve()

    store.close_reader()
    assert store.persistence is None


def test_label_reader_and_insights_keep_the_handle_they_are_given(connects):
    from agent.insights import InsightsEngine
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    path = _seeded_store()
    with open_persistence_reader(path) as persistence:
        assert InsightsEngine(None, persistence=persistence).persistence is persistence
        before = len(connects)
        assert SurfaceLabelReader(persistence).persistence is persistence
        assert connects[before:] == []


def test_the_label_reader_reads_nothing_when_its_handle_cannot_open(failing_opens):
    from hermes_cli.gateway_session_store import GatewaySessionStore

    # The name bound at import is the real open: the store file exists before the patched reader fails.
    open_persistence(profile_scope(home=get_hermes_home()), eager=True).close()
    store = GatewaySessionStore(connection=object())
    assert store._db() is None and store.persistence is None
    assert store.list_sessions_rich(limit=5) is None
    assert store.get_session("any") is None
    store.close_reader()
