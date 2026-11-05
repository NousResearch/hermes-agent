"""A tracked connection collected without close() releases its registry entry (#149291).

Only ``close()`` used to untrack, so a dropped connection left its path "live" for the life of the
process and every ``offline_file_access`` on it was refused. Each connection must still be untracked
exactly once, whether it is closed, collected, or both.
"""

import gc
import sqlite3

import pytest

import hermes_cli.sqlite_safe_read as sr
from hermes_cli.sqlite_safe_read import (
    LiveConnectionError,
    connect_tracked,
    has_live_connection,
    offline_file_access,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "_live_connections", {})
    path = tmp_path / "state.db"
    sqlite3.connect(path).close()
    return path


def _offline_allowed(path) -> bool:
    try:
        with offline_file_access(path):
            return True
    except LiveConnectionError:
        return False


def _open_and_drop(path, **kwargs) -> None:
    conn = connect_tracked(str(path), tracking_path=path, **kwargs)
    conn.execute("SELECT 1")


def test_collected_connection_releases_its_entry(db):
    _open_and_drop(db)
    gc.collect()
    assert not has_live_connection(db)
    assert _offline_allowed(db)


def test_collecting_one_of_two_keeps_the_path_live(db):
    kept = connect_tracked(str(db), tracking_path=db)
    _open_and_drop(db)
    gc.collect()
    assert has_live_connection(db), "the open connection's locks are still at stake"
    assert not _offline_allowed(db)
    kept.close()
    assert not has_live_connection(db)


def test_close_then_collect_untracks_once(db):
    other = connect_tracked(str(db), tracking_path=db)
    closed = connect_tracked(str(db), tracking_path=db)
    closed.close()
    del closed
    gc.collect()
    assert has_live_connection(db), "a second untrack would have released the other connection"
    other.close()
    assert not has_live_connection(db)


def test_retrofitted_connection_is_released_when_collected(db):
    class OwnFactory(sqlite3.Connection):
        pass

    def opener(path, **kwargs):
        kwargs["factory"] = OwnFactory  # discards the tracking factory, forcing the retrofit
        return sqlite3.connect(path, **kwargs)

    _open_and_drop(db, connect_fn=opener)
    gc.collect()
    assert not has_live_connection(db)


def test_failed_finalizer_attach_leaves_no_entry(db, monkeypatch):
    def broken(conn, key):
        raise TypeError("cannot attach")

    monkeypatch.setattr(sr, "_untrack_when_collected", broken)
    with pytest.raises(TypeError):
        connect_tracked(str(db), tracking_path=db)
    assert not has_live_connection(db)
