"""Tests for Discord thread participation persistence helpers."""

import json

from gateway.platforms.discord_impl import state as discord_state


def test_thread_state_path_uses_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert discord_state.thread_state_path() == tmp_path / "discord_threads.json"


def test_load_participated_threads_returns_empty_without_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert discord_state.load_participated_threads() == set()


def test_load_participated_threads_reads_json_list(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "discord_threads.json").write_text(json.dumps(["111", "222"]), encoding="utf-8")

    assert discord_state.load_participated_threads() == {"111", "222"}


def test_load_participated_threads_tolerates_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "discord_threads.json").write_text("not-json", encoding="utf-8")

    assert discord_state.load_participated_threads() == set()


def test_save_participated_threads_persists_trimmed_set(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    saved_threads = discord_state.save_participated_threads({"1", "2", "3", "4"}, max_threads=2)

    assert saved_threads <= {"1", "2", "3", "4"}
    assert len(saved_threads) == 2
    persisted = set(json.loads((tmp_path / "discord_threads.json").read_text(encoding="utf-8")))
    assert persisted == saved_threads


def test_track_thread_adds_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    updated_threads = discord_state.track_thread(set(), "111")

    assert updated_threads == {"111"}
    assert json.loads((tmp_path / "discord_threads.json").read_text(encoding="utf-8")) == ["111"]


def test_track_thread_duplicate_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    existing = discord_state.track_thread(set(), "111")

    updated_threads = discord_state.track_thread(existing, "111")

    assert updated_threads == {"111"}
    assert json.loads((tmp_path / "discord_threads.json").read_text(encoding="utf-8")) == ["111"]
