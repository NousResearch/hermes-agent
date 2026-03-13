"""Tests for Discord Opus codec loading — must use ctypes.util.find_library."""

import inspect


class TestOpusFindLibrary:
    """Opus loading must use ctypes.util.find_library, not hardcoded paths."""

    def test_no_hardcoded_opus_path(self):
        from gateway.platforms.discord import DiscordAdapter
        source = inspect.getsource(DiscordAdapter.connect)
        assert "/opt/homebrew" not in source, \
            "Opus loading must not use hardcoded /opt/homebrew path"
        assert "libopus.so.0" not in source, \
            "Opus loading must not use hardcoded libopus.so.0 path"

    def test_uses_find_library(self):
        from gateway.platforms.discord import DiscordAdapter
        source = inspect.getsource(DiscordAdapter.connect)
        assert "find_library" in source, \
            "Opus loading must use ctypes.util.find_library"

    def test_opus_decode_error_logged(self):
        """Opus decode failure must log the error, not silently return."""
        from gateway.platforms.discord import VoiceReceiver
        source = inspect.getsource(VoiceReceiver._on_packet)
        assert "logger" in source, \
            "_on_packet must log Opus decode errors"
        # Must not have bare `except Exception:\n            return`
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "except Exception" in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                assert next_line != "return", \
                    f"_on_packet has bare 'except Exception: return' at line {i+1}"
