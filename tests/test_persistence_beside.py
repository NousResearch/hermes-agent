"""A storage interface handle beside a store never changes what the store's path does."""
from contextlib import ExitStack

import pytest

import persistence_assembly
from hermes_constants import get_hermes_home
from persistence_assembly import open_persistence, profile_scope
from persistence_beside import close_beside, open_beside, open_reader_beside


@pytest.fixture
def failing_opens(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("unknown persistence provider 'nope'")

    monkeypatch.setattr(persistence_assembly, "open_persistence", fail)
    monkeypatch.setattr(persistence_assembly, "open_persistence_reader", fail)


def test_open_beside_opens_a_handle_on_the_named_home(tmp_path):
    handle = open_beside(tmp_path)
    try:
        assert handle.scope.profile_dir == tmp_path.resolve()
    finally:
        close_beside(handle)
    assert not (tmp_path / "state.db").exists()


def test_a_failed_open_answers_none(failing_opens, tmp_path):
    assert open_beside() is None
    handles = ExitStack()
    assert open_reader_beside(tmp_path / "state.db", handles) is None
    close_beside(handles)


def test_close_beside_keeps_a_failed_close_to_itself():
    class Raising:
        closed = 0

        def close(self):
            self.closed += 1
            raise OSError("disk I/O error")

    handle = Raising()
    close_beside(handle)
    close_beside(None)
    assert handle.closed == 1


def test_the_label_reader_reads_nothing_when_its_handle_cannot_open(failing_opens):
    from hermes_cli.gateway_session_store import GatewaySessionStore

    # The name bound at import is the real open: the store file exists before the patched reader fails.
    open_persistence(profile_scope(home=get_hermes_home()), eager=True).close()
    store = GatewaySessionStore(connection=object())
    assert store._db() is None and store.persistence is None
    assert store.list_sessions_rich(limit=5) is None
    assert store.get_session("any") is None
    store.close_reader()
