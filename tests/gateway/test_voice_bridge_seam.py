"""Discord adapter ↔ MisaMisaVoiceBridge integration seam tests.

Phase 2a AC #1: one approved integration seam connects live Misa-Misa
Discord voice I/O to ``kensei-voice`` without duplicate receive/STT/
playback ownership.

These tests pin the contract between the Discord adapter
(``plugins/platforms/discord/adapter.py``) and the bridge
(``kensei_voice.misa_misa_seam``) so the seam cannot regress to
"two STT pipelines on one Discord client" or "legacy VoiceReceiver
running in parallel with the bridge".

A fake :class:`VoiceRecvClient` stands in for the live Discord
voice client; the real ``MisaMisaVoiceBridge`` runs against it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The bridge and its numeric audio runtime are optional together. Gate both
# before importing either so a base install reports a module skip rather than
# failing collection before the existing kensei_voice guard can run.
_HAS_VOICE_BRIDGE_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "kensei_voice", "pocket_tts")
)
pytestmark = pytest.mark.skipif(
    not _HAS_VOICE_BRIDGE_STACK,
    reason="voice bridge integration requires numpy, kensei_voice, and pocket_tts",
)

# kensei_voice is an intentionally separate, optional package.  When it
# is not installed the entire integration seam has nothing to test, so
# skip the whole module cleanly instead of raising a collection error.
# Do NOT add kensei-voice as a core dependency — the adapter runtime
# already guards the import with KENSEI_VOICE_BRIDGE_AVAILABLE.
if _HAS_VOICE_BRIDGE_STACK:
    import numpy as np
    from kensei_voice.discord_transport import _StreamingPCMSource
    from kensei_voice.misa_misa_seam import LiveLatencyLog
else:
    np = None
    _StreamingPCMSource = None
    LiveLatencyLog = None

from gateway.config import Platform, PlatformConfig


def _make_minimal_adapter(*, use_voice_bridge: bool = True):
    """Build a DiscordAdapter shell with just the voice-bridge fields."""
    from plugins.platforms.discord import adapter as adapter_mod

    adapter = object.__new__(adapter_mod.DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(
        enabled=True, token="t", extra={"use_kensei_voice_bridge": use_voice_bridge}
    )
    adapter._voice_bridges = {}
    adapter._voice_bridge_latency_logs = {}
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._use_voice_bridge = use_voice_bridge and adapter_mod.KENSEI_VOICE_BRIDGE_AVAILABLE
    adapter._voice_timeout_seconds = adapter_mod.DiscordAdapter.VOICE_TIMEOUT
    # Stub the legacy path that the adapter would otherwise start.
    adapter._voice_fx_cfg = {"enabled": False}
    return adapter


class _FakeVoiceClient:
    """Minimal stand-in for ``discord.ext.voice_recv.VoiceRecvClient``.

    Mirrors the shape the MisaMisaVoiceBridge exercises: ``listen``,
    ``play``, ``stop``, ``stop_listening``, ``is_connected``,
    ``is_playing``, and ``disconnect`` (async).
    """

    def __init__(self, *, channel=None):
        self.channel = channel or MagicMock(id=100)
        self._sink = None
        self._playing = False
        self._played_source = None
        self._play_after = None
        self._play_calls: List[dict] = []
        self._stop_calls: List[float] = []
        self._connected = True
        self._listen_calls = 0
        self._disconnect_calls = 0

    def is_connected(self) -> bool:
        return self._connected

    def is_playing(self) -> bool:
        return self._playing

    def listen(self, sink) -> None:
        self._listen_calls += 1
        self._sink = sink
        sink._voice_client = self  # type: ignore[attr-defined]

    def stop_listening(self) -> None:
        self._sink = None

    def play(self, source, *, after=None, **kwargs) -> None:
        if self._playing:
            raise RuntimeError("Already playing audio")
        self._play_calls.append({"ts": time.perf_counter(), "source": source})
        self._playing = True
        self._played_source = source
        self._play_after = after

    def stop(self) -> None:
        self._stop_calls.append(time.perf_counter())
        self._playing = False
        if self._play_after:
            self._play_after(None)

    async def disconnect(self) -> None:
        self._connected = False
        self._playing = False
        self._disconnect_calls += 1


async def _wait_for(predicate, timeout_s: float = 5.0, poll_s: float = 0.02) -> bool:
    start = time.perf_counter()
    while not predicate():
        if time.perf_counter() - start > timeout_s:
            return False
        await asyncio.sleep(poll_s)
    return True


# ---------------------------------------------------------------------------
# AC #1: integration seam is wired without duplicate ownership
# ---------------------------------------------------------------------------


class TestAdapterBridgeSeam:
    def test_bridge_constants_are_imported(self):
        """The adapter must expose the bridge types so callers can introspect."""
        from plugins.platforms.discord import adapter as adapter_mod

        assert adapter_mod.KENSEI_VOICE_BRIDGE_AVAILABLE is True
        assert adapter_mod.MisaMisaVoiceBridge is not None
        assert adapter_mod.LiveLatencyLog is not None

    @pytest.mark.asyncio
    async def test_voice_connect_for_bridge_returns_voicerecv(self):
        """The bridge connect path asks Discord for a VoiceRecvClient."""
        from plugins.platforms.discord import adapter as adapter_mod
        from discord.ext import voice_recv as real_voice_recv

        adapter = _make_minimal_adapter()
        fake_vc = _FakeVoiceClient()

        async def _connect(cls=None, **_kw):
            # Adapter passes cls=voice_recv.VoiceRecvClient; just record and return.
            adapter._last_connect_cls = cls
            return fake_vc

        channel = MagicMock()
        channel.connect = _connect
        result = await adapter._voice_connect_for_bridge(channel)

        assert result is fake_vc
        # Verify the adapter asked Discord for a VoiceRecvClient (not a plain
        # VoiceClient — the bridge subscribes via vc.listen, which only the
        # recv extension provides).
        assert adapter._last_connect_cls is not None
        assert adapter._last_connect_cls is real_voice_recv.VoiceRecvClient

    @pytest.mark.asyncio
    async def test_voice_connect_for_bridge_returns_none_on_import_failure(self):
        """If discord-ext-voice-recv is unavailable the helper returns None."""
        from plugins.platforms.discord import adapter as adapter_mod

        adapter = _make_minimal_adapter()

        channel = MagicMock()
        # Make channel.connect raise to simulate a connect failure; the
        # helper catches the failure path and returns None.
        async def _explode(**_kw):
            raise RuntimeError("connect failed")
        channel.connect = _explode

        # Block the import-by-name so the helper short-circuits.
        import builtins
        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "discord.ext.voice_recv" or name.startswith("discord.ext.voice_recv."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            result = await adapter._voice_connect_for_bridge(channel)
        # The helper swallowed the import error and returned None.
        assert result is None

    @pytest.mark.asyncio
    async def test_start_voice_bridge_initialises_state(self):
        """The bridge constructor is called with a real brain and log."""
        from plugins.platforms.discord import adapter as adapter_mod
        from kensei_voice.misa_misa_seam import MisaMisaVoiceBridge

        adapter = _make_minimal_adapter()
        fake_vc = _FakeVoiceClient()
        adapter._voice_clients[42] = fake_vc

        # The bridge's start is what subscribes the voice client. We patch
        # MisaMisaVoiceBridge.start to a no-op so we don't have to spin a
        # real voice loop; the seam invariants are independent of the loop.
        with patch.object(
            MisaMisaVoiceBridge, "start", new=AsyncMock()
        ), patch.object(
            MisaMisaVoiceBridge, "started", new=True
        ):
            ok = await adapter._start_voice_bridge(42, fake_vc)

        assert ok is True
        assert 42 in adapter._voice_bridges
        assert 42 in adapter._voice_bridge_latency_logs

        log = adapter._voice_bridge_latency_logs[42]
        assert isinstance(log, LiveLatencyLog)
        # Config context must be captured at start time (host, python, etc.).
        ctx = log.config_context
        assert ctx["guild_id"] == 42
        assert "host" in ctx
        assert "python" in ctx

    def test_voice_bridge_active_reports_state(self):
        from plugins.platforms.discord import adapter as adapter_mod
        from kensei_voice.misa_misa_seam import MisaMisaVoiceBridge

        adapter = _make_minimal_adapter()
        # No bridge — returns False.
        assert adapter.voice_bridge_active(42) is False

        # Fake a started bridge.
        fake_bridge = MagicMock(spec=MisaMisaVoiceBridge)
        fake_bridge.started = True
        adapter._voice_bridges[42] = fake_bridge
        assert adapter.voice_bridge_active(42) is True

        # Stopped bridge.
        fake_bridge.started = False
        assert adapter.voice_bridge_active(42) is False

    def test_voice_bridge_latency_report_returns_log(self):
        from plugins.platforms.discord import adapter as adapter_mod

        adapter = _make_minimal_adapter()
        # No log yet — returns None.
        assert adapter.voice_bridge_latency_report(42) is None

        log = LiveLatencyLog(config_context={"guild_id": 42})
        adapter._voice_bridge_latency_logs[42] = log
        report = adapter.voice_bridge_latency_report(42)
        assert report is not None
        assert report["config_context"]["guild_id"] == 42

    @pytest.mark.asyncio
    async def test_join_voice_channel_starts_bridge_not_receiver(self):
        """Bridge path: ``join_voice_channel`` starts the bridge and
        does NOT instantiate the legacy ``VoiceReceiver``."""
        from plugins.platforms.discord import adapter as adapter_mod
        from kensei_voice.misa_misa_seam import MisaMisaVoiceBridge

        adapter = _make_minimal_adapter(use_voice_bridge=True)
        fake_vc = _FakeVoiceClient()

        # Track VoiceReceiver instantiation to verify it does NOT happen.
        with patch.object(
            adapter_mod, "VoiceReceiver"
        ) as receiver_mock, patch.object(
            adapter_mod.DiscordAdapter, "_voice_connect_for_bridge",
            new=AsyncMock(return_value=fake_vc),
        ), patch.object(
            adapter_mod.DiscordAdapter, "_install_voice_mixer",
            new=AsyncMock(),
        ), patch.object(
            MisaMisaVoiceBridge, "start", new=AsyncMock()
        ), patch.object(
            MisaMisaVoiceBridge, "started", new=True, create=True
        ):
            # adapter._client is checked at the top of join_voice_channel
            adapter._client = MagicMock()
            channel = MagicMock()
            channel.guild.id = 42
            ok = await adapter.join_voice_channel(channel)

        assert ok is True
        # The bridge path is taken.
        assert adapter._voice_bridges[42] is not None
        # VoiceReceiver must NOT have been instantiated.
        receiver_mock.assert_not_called()
        # Voice client is the one the bridge helper returned.
        assert adapter._voice_clients[42] is fake_vc

    @pytest.mark.asyncio
    async def test_join_voice_channel_legacy_path_still_works(self):
        """Bridge disabled: ``join_voice_channel`` uses the legacy path.

        This guards against the integration seam accidentally removing
        the legacy fallback operators depend on for non-Misa-Misa
        deploys.
        """
        from plugins.platforms.discord import adapter as adapter_mod

        adapter = _make_minimal_adapter(use_voice_bridge=False)
        fake_vc = _FakeVoiceClient()

        async def _plain_connect(**_kw):
            return fake_vc

        channel = MagicMock()
        channel.id = 100
        channel.guild.id = 42
        channel.connect = _plain_connect

        # Track coroutines the adapter schedules so we can close them
        # cleanly and avoid "coroutine was never awaited" RuntimeWarnings.
        scheduled_coros: list[asyncio.Future] = []

        def _ensure_future(coro):
            # Wrap the real coroutine in a sleep task (we don't want the
            # legacy listen-loop/timeout-handler actually running) but
            # close the original coroutine so it is not left unawaited.
            coro.close()
            task = asyncio.create_task(asyncio.sleep(0))
            scheduled_coros.append(task)
            return task

        with patch.object(
            adapter_mod, "VoiceReceiver",
            MagicMock(return_value=MagicMock(start=lambda: None)),
        ) as receiver_mock, patch.object(
            adapter_mod.asyncio, "ensure_future", _ensure_future,
        ):
            adapter._client = MagicMock()
            ok = await adapter.join_voice_channel(channel)

        assert ok is True
        # Legacy VoiceReceiver path was taken.
        receiver_mock.assert_called_once()
        # No bridge started.
        assert 42 not in adapter._voice_bridges

    @pytest.mark.asyncio
    async def test_leave_voice_channel_stops_bridge(self):
        """``leave_voice_channel`` must stop the bridge before disconnecting."""
        from plugins.platforms.discord import adapter as adapter_mod
        from kensei_voice.misa_misa_seam import MisaMisaVoiceBridge

        adapter = _make_minimal_adapter(use_voice_bridge=True)
        fake_vc = _FakeVoiceClient()
        adapter._voice_clients[42] = fake_vc

        # Drop in a fake bridge whose .stop() we can observe.
        fake_bridge = MagicMock(spec=MisaMisaVoiceBridge)
        fake_bridge.stop = AsyncMock()
        fake_bridge.started = True
        adapter._voice_bridges[42] = fake_bridge
        adapter._voice_bridge_latency_logs[42] = LiveLatencyLog()

        await adapter.leave_voice_channel(42)

        fake_bridge.stop.assert_awaited_once()
        assert 42 not in adapter._voice_bridges
        assert 42 not in adapter._voice_bridge_latency_logs
        # Voice client disconnected.
        assert fake_vc._disconnect_calls == 1


# ---------------------------------------------------------------------------
# AC #1 follow-up: end-to-end with the real bridge (no real Discord)
# ---------------------------------------------------------------------------


class TestAdapterBridgeEndToEnd:
    """Drive a real MisaMisaVoiceBridge through the adapter's seam.

    Confirms the bridge subscribes the fake VoiceRecvClient exactly
    once (the seam's load-bearing invariant: legacy VoiceReceiver
    must not run in parallel) and that the live latency log captures
    the timestamps the live ACs require.
    """

    @pytest.mark.asyncio
    async def test_bridge_starts_via_seam_and_does_not_double_subscribe(self):
        from plugins.platforms.discord import adapter as adapter_mod
        from kensei_voice.misa_misa_seam import MisaMisaVoiceBridge

        adapter = _make_minimal_adapter(use_voice_bridge=True)
        fake_vc = _FakeVoiceClient()

        with patch.object(
            adapter_mod.DiscordAdapter, "_install_voice_mixer",
            new=AsyncMock(),
        ):
            # Don't mock MisaMisaVoiceBridge.start this time — we want the
            # real bridge to run so the seam contract is exercised end-to-end.
            adapter._client = MagicMock()
            channel = MagicMock()
            channel.guild.id = 7
            channel.id = 700
            # channel.connect returns the fake VoiceRecvClient (no real Discord).
            async def _connect(cls=None, **_kw):
                return fake_vc
            channel.connect = _connect

            ok = await adapter.join_voice_channel(channel)

        assert ok is True
        bridge = adapter._voice_bridges[7]
        assert isinstance(bridge, MisaMisaVoiceBridge)
        # The seam's invariant: the bridge subscribed the voice client
        # exactly once.  A second subscribe would compete for the same
        # Opus-decoded frames and is forbidden by design.
        assert fake_vc._listen_calls == 1
        # No legacy VoiceReceiver was started.
        assert 7 not in adapter._voice_receivers

        # Clean up.
        await adapter.leave_voice_channel(7)
        assert fake_vc._connected is False
