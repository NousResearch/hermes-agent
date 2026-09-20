"""Behavior contracts for the session-store provider construction seam."""

from pathlib import Path

import pytest

import hermes_state_registry as registry
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.close_all()
    yield
    registry.close_all()


@pytest.fixture
def profile_homes(tmp_path, monkeypatch):
    import hermes_state

    # The suite's live-DB guard pins DEFAULT_DB_PATH to hermes_test/state.db. Restore the
    # production call-time behavior so argless acquire() follows each active profile scope.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = (tmp_path / "profile-a", tmp_path / "profile-b")
    for home in homes:
        home.mkdir()
    return homes


def _acquire_for_home(home: Path):
    token = set_hermes_home_override(home)
    try:
        return registry.acquire()
    finally:
        reset_hermes_home_override(token)


def test_default_provider_opens_sqlite_for_active_hermes_home(profile_homes):
    home, _ = profile_homes
    db = _acquire_for_home(home)
    try:
        assert isinstance(db, SessionDB)
        assert Path(db.db_path) == (home / "state.db").resolve()
    finally:
        assert registry.release(db) is True


def test_registered_provider_tracks_profile_scope_a_b_a(profile_homes, monkeypatch):
    import hermes_state_provider as providers

    home_a, home_b = profile_homes
    (home_a / "config.yaml").write_text(
        "sessiondb:\n  provider: recording-a\n",
        encoding="utf-8",
    )
    (home_b / "config.yaml").write_text(
        "sessiondb:\n  provider: recording-b\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(providers, "_PROVIDERS", providers._PROVIDERS.copy())

    opened = []

    def recording_factory(label):
        def open_recording(db_path: Path):
            db = SessionDB(db_path=db_path)
            opened.append((label, db_path, db))
            return db

        return open_recording

    providers.register_session_store_provider("recording-a", recording_factory("a"))
    providers.register_session_store_provider("recording-b", recording_factory("b"))

    returned = []
    for label, home in (("a", home_a), ("b", home_b), ("a", home_a)):
        db = _acquire_for_home(home)
        returned.append(db)
        try:
            assert opened[-1] == (label, (home / "state.db").resolve(), db)
        finally:
            assert registry.release(db) is True

    assert [label for label, _, _ in opened] == ["a", "b", "a"]
    assert [db for _, _, db in opened] == returned
