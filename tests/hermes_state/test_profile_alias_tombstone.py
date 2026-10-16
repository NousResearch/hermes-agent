"""Every spelling of one named profile directory shares one retirement fence.

Regression from the PR #93508 review: the tombstone was keyed to the lexical home, so once
the real home retired, ``SessionDB`` refused the canonical path but opened and wrote through
a symlink alias of the same directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_state
from hermes_cli.profile_incarnation import ensure_profile_incarnation
from hermes_cli.profile_lifecycle import mark_profile_deleting, profile_lifecycle_lease
from hermes_state import SessionDB


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    return root


def _retire(home: Path, token: str) -> None:
    with profile_lifecycle_lease(home):
        mark_profile_deleting(home, token)


@pytest.mark.platforms("posix")
def test_alias_spellings_share_the_real_homes_tombstone(root, tmp_path):
    home = root / "profiles" / "alpha"
    home.mkdir()
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    outside = tmp_path / "alias"
    outside.symlink_to(home, target_is_directory=True)
    sibling = root / "profiles" / "beta"
    sibling.symlink_to(home, target_is_directory=True)
    token = ensure_profile_incarnation(home)
    spellings = (home, outside, sibling)
    for spelling in spellings:
        SessionDB(db_path=spelling / "state.db", expected_profile_incarnation=token).close()

    _retire(home, token)

    for spelling in spellings:
        with pytest.raises(FileNotFoundError):
            SessionDB(db_path=spelling / "state.db", expected_profile_incarnation=token)


@pytest.mark.platforms("posix")
def test_named_entry_linked_outside_hermes_keeps_its_own_fence(root, tmp_path):
    """Canonical keying must not strip the fence from a profile stored on another disk."""
    storage = tmp_path / "external" / "work"
    storage.mkdir(parents=True)
    (storage / "config.yaml").write_text("{}", encoding="utf-8")
    home = root / "profiles" / "work"
    home.symlink_to(storage, target_is_directory=True)
    token = ensure_profile_incarnation(home)
    SessionDB(db_path=home / "state.db", expected_profile_incarnation=token).close()

    _retire(home, token)

    with pytest.raises(FileNotFoundError):
        SessionDB(db_path=home / "state.db", expected_profile_incarnation=token)
