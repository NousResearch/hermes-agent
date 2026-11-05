"""Storage interface readers handed to a surface open no connection beyond the reader's own."""
import sqlite3

import pytest

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
