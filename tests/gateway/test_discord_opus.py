"""Tests for Discord Opus codec loading.

The loader logic lives in ``plugins.platforms.discord.opus_loader`` and
is shared between the runtime adapter and the doctor script. These
tests assert source-level invariants only:

  * The shared helper is the resolution path used by ``DiscordAdapter.connect``.
  * The helper continues to consult ``ctypes.util.find_library`` and
    carry a macOS Homebrew fallback (preserves pre-#30723 behaviour).
  * The voice-decode error path still logs instead of swallowing
    exceptions silently.

Behavioural assertions (env-var priority, NixOS scenarios, error
messaging) live in ``tests/plugins/platforms/discord/test_opus_loader_30723.py``
where they can drive the helper with mocks.
"""

import inspect


class TestOpusFindLibrary:
    """Opus loading must use ctypes.util.find_library, with platform fallback."""

    def test_uses_find_library_first(self):
        """find_library must be the primary lookup strategy used by the
        shared loader. The adapter delegates to the loader (#30723), so
        we assert against the loader source rather than ``connect()``."""
        from plugins.platforms.discord import opus_loader
        source = inspect.getsource(opus_loader)
        assert "find_library" in source, \
            "Opus loading must use ctypes.util.find_library"

    def test_homebrew_fallback_is_conditional(self):
        """Homebrew paths must only be tried on Darwin, AFTER
        find_library has been consulted. The check inspects the
        loader's platform-specific candidate function."""
        from plugins.platforms.discord import opus_loader
        source = inspect.getsource(opus_loader)
        assert "/opt/homebrew" in source or "homebrew" in source, \
            "Opus loading should have macOS Homebrew fallback"
        # find_library must appear BEFORE any Homebrew path in the
        # combined module source — the loader uses find_library in
        # ensure_discord_opus_loaded() (top) and the Homebrew paths
        # live in the _DARWIN_FALLBACK_PATHS tuple (bottom).
        fl_idx = source.index("find_library")
        hb_idx = source.index("/opt/homebrew")
        assert fl_idx < hb_idx, \
            "find_library must be tried before Homebrew fallback paths"
        # Fallback must be guarded by a platform check — the
        # _candidates_for_platform helper routes only Darwin into the
        # Homebrew bucket.
        assert "darwin" in source, \
            "Homebrew fallback must be guarded by macOS platform check"

    def test_adapter_delegates_to_shared_loader(self):
        """DiscordAdapter.connect must route Opus loading through the
        shared helper, not re-implement it inline (#30723)."""
        from plugins.platforms.discord.adapter import DiscordAdapter
        source = inspect.getsource(DiscordAdapter.connect)
        assert "ensure_discord_opus_loaded" in source, (
            "connect() must call ensure_discord_opus_loaded so the "
            "doctor script and the runtime stay in sync"
        )

    def test_opus_decode_error_logged(self):
        """Opus decode failure must log the error, not silently return."""
        from plugins.platforms.discord.adapter import VoiceReceiver
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
