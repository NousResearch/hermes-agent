"""A named profile stored outside the Hermes root is admitted by the shared registry.

The profile's identity marker and lifecycle lock live on its ``profiles/<name>`` path, not in the
junction or symlink target, so ``acquire`` checks the incarnation, takes the lease and opens the
store through that path while it still shares one handle per resolved database file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import hermes_state
import hermes_state_registry as registry
from hermes_cli.profile_incarnation import ensure_profile_incarnation
from hermes_cli.profile_lifecycle import mark_profile_deleting, profile_lifecycle_lease


def link_directory(link: Path, target: Path) -> None:
    """The host's native directory link: a junction on Windows, a symlink elsewhere."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.fixture
def linked_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    storage = tmp_path / "other-disk" / "work"
    storage.mkdir(parents=True)
    (storage / "config.yaml").write_text("{}", encoding="utf-8")
    home = root / "profiles" / "work"
    link_directory(home, storage)
    yield home
    registry.close_all()


def test_registry_admits_named_profile_linked_outside_hermes(linked_home):
    token = ensure_profile_incarnation(linked_home)
    assert token

    cold = registry.acquire(linked_home / "state.db", expected_profile_incarnation=token)
    cold.create_session("linked-row", "tui")
    warm = registry.acquire(linked_home / "state.db", expected_profile_incarnation=token)

    assert warm is cold
    assert warm.get_session("linked-row")["id"] == "linked-row"
    assert registry.release(warm) is True
    assert registry.release(cold) is True
    assert registry.stats()["live_generations"] == 0


def test_registry_keeps_the_fence_on_a_linked_profile_after_retirement(linked_home):
    token = ensure_profile_incarnation(linked_home)
    registry.release(registry.acquire(linked_home / "state.db", expected_profile_incarnation=token))
    with profile_lifecycle_lease(linked_home):
        mark_profile_deleting(linked_home, token)

    # The deletion marker belongs to the lexical home, so the lease must check it there.
    with pytest.raises(FileNotFoundError):
        registry.acquire(linked_home / "state.db", expected_profile_incarnation=token)
