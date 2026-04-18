"""Behavior tests for Discord Opus loading and decode error handling."""

import asyncio
import struct
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from gateway.config import PlatformConfig


class _FakeOpusModule:
    def __init__(self):
        self.loaded = False
        self.load_calls = []

    def is_loaded(self):
        return self.loaded

    def load_opus(self, path):
        self.load_calls.append(path)
        self.loaded = True


class TestOpusFindLibrary:
    """Opus loading must prefer library lookup and log decode failures."""

    def test_connect_uses_find_library_before_fallback_paths(self):
        import gateway.platforms.discord as discord_mod

        fake_opus = _FakeOpusModule()
        adapter = discord_mod.DiscordAdapter(PlatformConfig(enabled=True))

        with (
            patch.object(discord_mod, "DISCORD_AVAILABLE", True),
            patch.object(discord_mod, "discord", SimpleNamespace(opus=fake_opus)),
            patch("ctypes.util.find_library", return_value="/usr/lib/libopus.so"),
            patch.object(discord_mod.os.path, "isfile") as isfile_mock,
        ):
            assert asyncio.run(adapter.connect()) is False

        assert fake_opus.load_calls == ["/usr/lib/libopus.so"]
        isfile_mock.assert_not_called()

    def test_connect_uses_homebrew_fallback_only_when_lookup_fails(self):
        import gateway.platforms.discord as discord_mod

        fake_opus = _FakeOpusModule()
        adapter = discord_mod.DiscordAdapter(PlatformConfig(enabled=True))

        with (
            patch.object(discord_mod, "DISCORD_AVAILABLE", True),
            patch.object(discord_mod, "discord", SimpleNamespace(opus=fake_opus)),
            patch("ctypes.util.find_library", return_value=None),
            patch.object(sys, "platform", "darwin"),
            patch.object(
                discord_mod.os.path,
                "isfile",
                side_effect=lambda path: path == "/opt/homebrew/lib/libopus.dylib",
            ) as isfile_mock,
        ):
            assert asyncio.run(adapter.connect()) is False

        assert fake_opus.load_calls == ["/opt/homebrew/lib/libopus.dylib"]
        assert [call.args[0] for call in isfile_mock.call_args_list] == [
            "/opt/homebrew/lib/libopus.dylib"
        ]

    def test_opus_decode_error_is_logged(self):
        import gateway.platforms.discord as discord_mod

        class _Decoder:
            def decode(self, _payload):
                raise RuntimeError("bad opus frame")

        class _Aead:
            def __init__(self, _secret_key):
                pass

            def decrypt(self, encrypted, header, nonce):
                assert encrypted == b""
                assert len(header) == 12
                assert len(nonce) == 24
                return b"opus-frame"

        nacl_module = ModuleType("nacl")
        nacl_secret_module = ModuleType("nacl.secret")
        nacl_secret_module.Aead = _Aead
        nacl_module.secret = nacl_secret_module

        receiver = discord_mod.VoiceReceiver(
            SimpleNamespace(_connection=SimpleNamespace(secret_key=b"0" * 32, dave_session=None, ssrc=9999))
        )
        receiver._running = True
        receiver._secret_key = b"0" * 32
        receiver._bot_ssrc = 9999

        packet = struct.pack(">BBHII", 0x80, 0x78, 1, 2, 1234) + b"\x00\x00\x00\x00"

        with (
            patch.object(
                discord_mod,
                "discord",
                SimpleNamespace(opus=SimpleNamespace(Decoder=lambda: _Decoder())),
            ),
            patch.dict(sys.modules, {"nacl": nacl_module, "nacl.secret": nacl_secret_module}),
            patch.object(discord_mod.logger, "debug") as debug_mock,
        ):
            receiver._on_packet(packet)

        assert any(
            call.args and "Opus decode error" in call.args[0]
            for call in debug_mock.call_args_list
        )
