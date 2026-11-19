"""Storage interface handles for fake authorities: opened on a profile home, closed after each test.

A test module activates the closing fixture by importing ``close_storage_handles`` beside
``storage_handle``."""
import pytest

_HANDLES = []


@pytest.fixture(autouse=True)
def close_storage_handles():
    yield
    while _HANDLES:
        _HANDLES.pop().close()


def storage_handle(home):
    """A storage interface handle on *home*'s state.db, closed after the test."""
    from persistence_assembly import open_persistence, profile_scope
    _HANDLES.append(open_persistence(profile_scope(home=home)))
    return _HANDLES[-1]
