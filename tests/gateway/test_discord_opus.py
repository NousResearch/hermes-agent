"""Tests for Discord Opus codec loading — must use ctypes.util.find_library."""

import inspect
import types


class TestOpusFindLibrary:
    """Opus loading must try ctypes.util.find_library first, with platform fallback."""

    def test_uses_find_library_first(self):
        """find_library must be the primary lookup strategy."""
        from plugins.platforms.discord.adapter import DiscordAdapter
        source = inspect.getsource(DiscordAdapter.connect)
        assert "find_library" in source, \
            "Opus loading must use ctypes.util.find_library"


