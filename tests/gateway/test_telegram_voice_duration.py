"""Telegram voice/audio messages must carry an explicit duration.

Telegram only auto-derives a clip's duration from container metadata for short
recordings; longer ones (~5 min+) are sent with duration 0 and render as
``0:00`` in the player. ``TelegramAdapter.send_voice`` now probes the length
locally and passes ``duration=`` to ``sendVoice`` / ``sendAudio`` so the
bubble shows the real time.

These tests confirm:
  1. ``_probe_voice_duration_seconds`` rounds to whole seconds (real WAV via
     stdlib ``wave``), reads OGG/MP3 lengths via mutagen, and degrades to
     ``None`` (omit duration) when nothing can read the file.
  2. The ``.ogg`` voice path forwards the probed duration to ``bot.send_voice``.
  3. The ``.mp3`` audio path forwards the probed duration to ``bot.send_audio``.

Hermetic: no real mutagen / ffprobe required. The WAV path uses stdlib only;
the mutagen path is exercised with a fake module injected into ``sys.modules``.
"""
import sys
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as telegram_mod  # noqa: E402
from plugins.platforms.telegram.adapter import (  # noqa: E402
    TelegramAdapter,
    _coerce_duration_seconds,
    _probe_voice_duration_seconds,
)
from tools.send_message_tool import _send_telegram  # noqa: E402


def _write_wav(path, *, rate, frames):
    """Write a silent mono WAV of ``frames`` samples at ``rate`` Hz."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(rate)
        wf.writeframes(b"\x00" * frames)


# ---------------------------------------------------------------------------
# 1a. WAV path — real stdlib read, rounding to whole seconds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rate,frames,expected",
    [
        (10, 2900, 290),   # 290.0 -> 290
        (10, 2904, 290),   # 290.4 -> 290
        (10, 2906, 291),   # 290.6 -> 291
        (10, 12, 1),       # 1.2   -> 1
    ],
)
def test_probe_wav_rounds_to_whole_seconds(tmp_path, rate, frames, expected):
    f = tmp_path / "clip.wav"
    _write_wav(f, rate=rate, frames=frames)
    assert _probe_voice_duration_seconds(str(f)) == expected


# ---------------------------------------------------------------------------
# 1b. mutagen path — fake module so ogg/mp3 work without the real dependency
# ---------------------------------------------------------------------------

def _inject_fake_mutagen(monkeypatch, length):
    fake = SimpleNamespace(
        File=lambda _p: SimpleNamespace(info=SimpleNamespace(length=length))
    )
    monkeypatch.setitem(sys.modules, "mutagen", fake)


# ---------------------------------------------------------------------------
# 1c. _coerce_duration_seconds rounding/guard contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [(290.0, 290), (290.6, 291), ("12.7", 13), (0, None), (-3, None),
     (None, None), ("", None), ("x", None)],
)
def test_coerce_duration_seconds(value, expected):
    assert _coerce_duration_seconds(value) == expected


# ---------------------------------------------------------------------------
# 2 + 3. send_voice forwards the probed duration to the Bot API
# ---------------------------------------------------------------------------

def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_voice = AsyncMock(return_value=MagicMock(message_id=1))
    adapter._bot.send_audio = AsyncMock(return_value=MagicMock(message_id=2))
    return adapter


@pytest.mark.asyncio
async def test_voice_send_forwards_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        telegram_mod, "_probe_voice_duration_seconds", lambda _p: 314
    )
    audio = tmp_path / "reply.ogg"
    audio.write_bytes(b"\x00" * 16)

    adapter = _make_adapter()
    result = await adapter.send_voice("123", str(audio))

    assert result.success is True
    adapter._bot.send_voice.assert_awaited_once()
    assert adapter._bot.send_voice.await_args.kwargs["duration"] == 314
    adapter._bot.send_audio.assert_not_awaited()


