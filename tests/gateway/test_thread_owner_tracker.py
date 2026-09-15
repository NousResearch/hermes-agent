"""First-owner Discord thread map: unowned threads stay participation-gated."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
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


def test_mark_owner_accepts_int_snowflakes(owner_home):
    tracker = ThreadOwnerTracker("discord")
    assert tracker.mark_owner(555, 999) is True
    assert tracker.owner_for(555) == "999"
    assert tracker.owner_for("555") == "999"


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


def test_stale_in_memory_cache_does_not_steal_owner(owner_home):
    """Two trackers in one process: the later claim must reload, not overwrite."""
    first = ThreadOwnerTracker("discord")
    second = ThreadOwnerTracker("discord")
    assert first.mark_owner("555", "111") is True
    assert second.mark_owner("555", "999") is False
    assert second.owner_for("555") == "111"


def test_mark_owner_first_writer_wins_across_processes(owner_home):
    repo_root = str(Path(__file__).resolve().parents[2])
    child = (
        "import os, sys\n"
        "os.environ['HERMES_HOME'] = sys.argv[1]\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from gateway.platforms.helpers import ThreadOwnerTracker\n"
        "ok = ThreadOwnerTracker('discord').mark_owner('555', sys.argv[3])\n"
        "sys.stdout.write('1' if ok else '0')\n"
    )

    def _claim(owner: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", child, str(owner_home), repo_root, owner],
            capture_output=True,
            text=True,
            check=False,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_claim, "111")
        second = pool.submit(_claim, "999")
        results = [first.result(), second.result()]

    assert all(r.returncode == 0 for r in results), [r.stderr for r in results]
    wins = "".join(r.stdout for r in results)
    assert wins.count("1") == 1
    assert wins.count("0") == 1
    assert ThreadOwnerTracker("discord").owner_for("555") in {"111", "999"}
