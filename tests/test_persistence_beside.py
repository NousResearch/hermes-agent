"""A storage interface handle beside a store never changes what the store's path does."""
from contextlib import ExitStack

import pytest

import persistence_assembly
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
