"""Callers name a store by its profile folder, never by a ``<home>/state.db`` path they build themselves.

The fixture keeps the SQLite store under another file name in the profile folder, so a caller that builds
``<home>/state.db`` by hand names a file that does not exist, while a caller that names the profile reaches the store.
"""
import pytest

import persistence_assembly


@pytest.fixture
def store_elsewhere(monkeypatch):
    monkeypatch.setattr(persistence_assembly, "_state_db", lambda scope: scope.profile_dir / "store-elsewhere.db")


def test_the_session_store_keeps_one_handle_per_profile_however_its_folder_is_spelled(tmp_path, monkeypatch,
                                                                                       store_elsewhere):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    real = tmp_path / "home"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        token = set_hermes_home_override(real.resolve())  # a profile turn's scope names the canonical folder
        try:
            assert store._scoped_persistence() is store._routing_persistence
        finally:
            reset_hermes_home_override(token)
        assert list(store._db_handles) == [real.resolve()]
    finally:
        store.close_all_db_handles()
