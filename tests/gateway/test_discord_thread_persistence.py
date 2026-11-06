"""Tests for Discord thread participation persistence.

Verifies that _threads (ThreadParticipationTracker) survives adapter restarts by
being persisted to ~/.hermes/discord_threads.json.
"""

import os
from unittest.mock import patch


class TestDiscordThreadPersistence:
    """Thread IDs are saved to disk and reloaded on init."""

    def _make_adapter(self, tmp_path):
        """Build a minimal DiscordAdapter with HERMES_HOME pointed at tmp_path."""
        from gateway.config import PlatformConfig
        from plugins.platforms.discord.adapter import DiscordAdapter

        config = PlatformConfig(enabled=True, token="test-token")
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            return DiscordAdapter(config=config)



    def test_threads_survive_restart(self, tmp_path):
        """Threads tracked by one adapter instance are visible to the next."""
        adapter1 = self._make_adapter(tmp_path)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            adapter1._threads.mark("aaa")
            adapter1._threads.mark("bbb")

        adapter2 = self._make_adapter(tmp_path)
        assert "aaa" in adapter2._threads
        assert "bbb" in adapter2._threads


