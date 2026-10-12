"""Setup hands off to the profile it was made or reset from, even when every live home is the setup profile
(the setup profile's own pooled backend, launched under it)."""

from pathlib import Path

import pytest

from hermes_cli import profiles, setup_profile


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _setup_from(monkeypatch, owner: str) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(profiles.create_profile(owner, no_alias=True)))
    return setup_profile.ensure_setup_profile().path


def test_handoff_from_the_setup_backend_names_the_profile_setup_was_made_from(root, monkeypatch):
    setup = _setup_from(monkeypatch, "work")
    monkeypatch.setenv("HERMES_HOME", str(setup))

    assert setup_profile.primary_profile(setup) == "work"


def test_reset_from_the_setup_backend_keeps_the_handoff_owner(root, monkeypatch):
    setup = _setup_from(monkeypatch, "work")
    monkeypatch.setenv("HERMES_HOME", str(setup))

    setup_profile.reset_setup_profile(setup)

    assert setup_profile.primary_profile(setup) == "work"
    assert "owner_profile" not in setup_profile.read_state()


def test_handoff_follows_a_renamed_owner_profile(root, monkeypatch):
    setup = _setup_from(monkeypatch, "work")
    monkeypatch.setenv("HERMES_HOME", str(setup))

    profiles.rename_profile("work", "personal")

    assert setup_profile.primary_profile(setup) == "personal"
