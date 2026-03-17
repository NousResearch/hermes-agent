"""Tests for Discord thread participation persistence.

Verifies that _bot_participated_threads survives adapter restarts by
being persisted to ~/.hermes/discord_threads.json.
"""

import json
import os
from unittest.mock import patch

import pytest


class TestDiscordThreadPersistence:
    """Thread IDs are saved to disk and reloaded on init."""

    def _make_adapter(self, tmp_path):
        """Build a minimal DiscordAdapter with HERMES_HOME pointed at tmp_path."""
        from gateway.config import PlatformConfig
        from gateway.platforms.discord import DiscordAdapter

        config = PlatformConfig(enabled=True, token="test-token")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            return DiscordAdapter(config=config)

    def test_starts_empty_when_no_state_file(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        assert adapter._bot_participated_threads == set()

    def test_track_thread_persists_to_disk(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            adapter._track_thread("111")
            adapter._track_thread("222")

        state_file = tmp_path / "discord_threads.json"
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert set(saved) == {"111", "222"}

    def test_threads_survive_restart(self, tmp_path):
        """Threads tracked by one adapter instance are visible to the next."""
        adapter1 = self._make_adapter(tmp_path)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            adapter1._track_thread("aaa")
            adapter1._track_thread("bbb")

        adapter2 = self._make_adapter(tmp_path)
        assert "aaa" in adapter2._bot_participated_threads
        assert "bbb" in adapter2._bot_participated_threads

    def test_duplicate_track_does_not_double_save(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            adapter._track_thread("111")
            adapter._track_thread("111")  # no-op

        saved = json.loads((tmp_path / "discord_threads.json").read_text())
        assert saved.count("111") == 1

    def test_caps_at_max_tracked_threads(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        adapter._MAX_TRACKED_THREADS = 5
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            for i in range(10):
                adapter._track_thread(str(i))

        assert len(adapter._bot_participated_threads) == 5

    def test_corrupted_state_file_falls_back_to_empty(self, tmp_path):
        state_file = tmp_path / "discord_threads.json"
        state_file.write_text("not valid json{{{")
        adapter = self._make_adapter(tmp_path)
        assert adapter._bot_participated_threads == set()

    def test_missing_hermes_home_does_not_crash(self, tmp_path):
        """Load/save tolerate missing directories."""
        fake_home = tmp_path / "nonexistent" / "deep"
        with patch.dict(os.environ, {"HERMES_HOME": str(fake_home)}):
            from gateway.platforms.discord import DiscordAdapter
            # _load should return empty set, not crash
            threads = DiscordAdapter._load_participated_threads()
            assert threads == set()
