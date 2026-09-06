"""First-owner Discord thread map: unowned threads stay participation-gated."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.platforms.helpers import ThreadOwnerTracker, _thread_owner_state_home


@pytest.fixture
def owner_home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_mark_owner_is_first_writer_wins(owner_home):
    tracker = ThreadOwnerTracker("discord")
    assert tracker.mark_owner("555", "999") is True
    assert tracker.mark_owner("555", "999") is True
    assert tracker.mark_owner("555", "111") is False
    assert tracker.owner_for("555") == "999"
    saved = json.loads((owner_home / "discord_thread_owners.json").read_text())
    assert saved == {"555": "999"}


def test_invalid_ids_are_rejected(owner_home):
    tracker = ThreadOwnerTracker("discord")
    assert tracker.mark_owner("not-a-snowflake", "999") is False
    assert tracker.mark_owner("555", "bot-name") is False
    assert tracker.owner_for("555") is None


def test_corrupt_file_is_treated_as_unowned(owner_home):
    (owner_home / "discord_thread_owners.json").write_text("{not json", encoding="utf-8")
    tracker = ThreadOwnerTracker("discord")
    assert tracker.owner_for("555") is None
    assert tracker.mark_owner("555", "999") is True


def test_non_dict_file_is_treated_as_unowned(owner_home):
    (owner_home / "discord_thread_owners.json").write_text("[]", encoding="utf-8")
    tracker = ThreadOwnerTracker("discord")
    assert tracker.owner_for("555") is None


def test_owners_are_shared_across_profile_homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    alpha = root / "profiles" / "alpha"
    beta = root / "profiles" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(alpha))
    assert _thread_owner_state_home() == root
    first = ThreadOwnerTracker("discord")
    assert first.mark_owner("555", "999") is True

    monkeypatch.setenv("HERMES_HOME", str(beta))
    assert _thread_owner_state_home() == root
    second = ThreadOwnerTracker("discord")
    assert second.owner_for("555") == "999"
    assert second.mark_owner("555", "111") is False
    assert (root / "discord_thread_owners.json").exists()
    assert not (alpha / "discord_thread_owners.json").exists()
    assert not (beta / "discord_thread_owners.json").exists()


def test_default_profile_stores_next_to_hermes_home(owner_home):
    tracker = ThreadOwnerTracker("discord")
    tracker.mark_owner("1", "2")
    assert _thread_owner_state_home() == owner_home
    assert (owner_home / "discord_thread_owners.json").exists()


def test_capacity_leaves_thread_unowned(owner_home):
    tracker = ThreadOwnerTracker("discord", max_tracked=1)
    assert tracker.mark_owner("1", "9") is True
    assert tracker.mark_owner("2", "8") is False
    assert tracker.owner_for("2") is None
