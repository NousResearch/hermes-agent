"""HERMES_STATE_DB relocates state.db alone, leaving the rest of HERMES_HOME in place."""
from pathlib import Path

import pytest

import hermes_state


@pytest.fixture
def default_path(monkeypatch):
    """``_default_db_path`` with the monkeypatch hook neutralised, so only the env var varies."""
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    return hermes_state._default_db_path


def test_unset_falls_back_to_hermes_home(default_path, monkeypatch):
    monkeypatch.delenv("HERMES_STATE_DB", raising=False)
    assert default_path() == hermes_state.get_hermes_home() / "state.db"


def test_blank_is_treated_as_unset(default_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_DB", "   ")
    assert default_path() == hermes_state.get_hermes_home() / "state.db"


def test_explicit_file_path_is_used_verbatim(default_path, monkeypatch, tmp_path):
    target = tmp_path / "elsewhere" / "state.db"
    monkeypatch.setenv("HERMES_STATE_DB", str(target))
    assert default_path() == target


def test_existing_directory_gets_state_db_appended(default_path, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_STATE_DB", str(tmp_path))
    assert default_path() == tmp_path / "state.db"


def test_trailing_separator_is_treated_as_a_directory(default_path, monkeypatch, tmp_path):
    missing = tmp_path / "not-created-yet"
    monkeypatch.setenv("HERMES_STATE_DB", str(missing) + "/")
    assert default_path() == missing / "state.db"


def test_tilde_is_expanded(default_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_DB", "~/hermes-state/state.db")
    assert default_path() == Path("~/hermes-state/state.db").expanduser()


def test_repointed_default_wins_over_env(monkeypatch, tmp_path):
    """An in-process redirect must not be overridden by an inherited env var."""
    redirect = tmp_path / "redirected.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", redirect)
    monkeypatch.setenv("HERMES_STATE_DB", str(tmp_path / "from-env.db"))
    assert hermes_state._default_db_path() == redirect


def test_session_db_opens_at_the_override(default_path, monkeypatch, tmp_path):
    """End to end: a bare SessionDB() lands on the override and creates the parent directory."""
    target = tmp_path / "other-volume" / "state.db"
    monkeypatch.setenv("HERMES_STATE_DB", str(target))
    with hermes_state.SessionDB() as db:
        db.create_session(session_id="s1", source="tui", model="alpha")
        assert db.db_path == target
    assert target.exists()
