"""A named store can be homed apart from ``get_hermes_home()`` in one context.

An embedder that binds a turn to one profile home may keep the auth store and the
background-work ledger in another; ``set_store_home_override`` is that door.
"""

from hermes_cli.auth import _auth_file_path
from hermes_constants import (
    get_hermes_home,
    get_store_home,
    reset_store_home_override,
    set_store_home_override,
)
from tools.async_delegation import _db_path


def test_store_follows_its_override_while_hermes_home_does_not(tmp_path, monkeypatch):
    profile, head = tmp_path / "profile", tmp_path / "head"
    monkeypatch.setenv("HERMES_HOME", str(profile))
    auth = set_store_home_override("auth", head)
    work = set_store_home_override("background_work", head / "work")
    try:
        assert get_hermes_home() == profile
        assert _auth_file_path() == head / "auth.json"
        assert _db_path() == head / "work" / "state.db"
    finally:
        reset_store_home_override(work)
        reset_store_home_override(auth)
    assert get_store_home("auth") == profile
    assert _auth_file_path() == profile / "auth.json"
    assert _db_path() == profile / "state.db"
