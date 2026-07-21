"""Regression test for #68728 — Discord voice replies drop message reference.

Voice sends previously posted without a `message_reference` (native path) or
without `reference=` (file-fallback path) even when `reply_to` was supplied,
breaking thread continuity vs. text sends.
"""

import io
import json
import sys
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    # discord.http.Route is the real class — keep it but ignore its arguments.
    discord_mod.http = SimpleNamespace(Route=lambda *args, **kwargs: SimpleNamespace(name=args, kwargs=kwargs))

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _make_ogg(tmp_path) -> str:
    """Write a minimal valid ogg-opus-ish file (just bytes, no header check)."""
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"OggS" + b"\x00" * 64)
    return str(audio)


def _make_adapter(reply_to_mode: str = "first") -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._reply_to_mode = reply_to_mode
    return adapter


def _make_ref_message(message_id: int = 99, channel_id: int = 555):
    reference_obj = SimpleNamespace(
        message_id=message_id,
        channel_id=channel_id,
        to_message_reference_dict=lambda: {
            "message_id": str(message_id),
            "channel_id": str(channel_id),
            "fail_if_not_exists": False,
        },
    )
    ref_msg = SimpleNamespace(id=message_id, to_reference=MagicMock(return_value=reference_obj))
    return ref_msg, reference_obj


@pytest.mark.asyncio
async def test_send_voice_native_path_includes_message_reference(tmp_path):
    """Native voice-message REST path must include `message_reference` in the payload."""
    audio_path = _make_ogg(tmp_path)
    adapter = _make_adapter()
    ref_msg, reference_obj = _make_ref_message(99, 555)

    http_calls = []

    async def fake_request(route, *, form):
        http_calls.append({"route": route, "form": form})
        return {"id": 7777}

    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=fake_request),
    )

    result = await adapter.send_voice(chat_id="555", audio_path=audio_path, reply_to="99")

    assert result.success is True
    assert result.message_id == "7777"
    assert channel.fetch_message.await_count == 1
    ref_msg.to_reference.assert_called_once_with(fail_if_not_exists=False)

    # The native path is a single POST /channels/{id}/messages with a form
    # whose first entry is the JSON payload containing message_reference.
    assert len(http_calls) == 1
    form = http_calls[0]["form"]
    payload_entry = next(f for f in form if f["name"] == "payload_json")
    payload = json.loads(payload_entry["value"])
    assert "message_reference" in payload
    assert payload["message_reference"]["message_id"] == "99"
    assert payload["message_reference"]["channel_id"] == "555"
    assert payload["flags"] == 8192


@pytest.mark.asyncio
async def test_send_voice_fallback_passes_reference_to_channel_send(tmp_path):
    """When the native voice-flag path raises, the file fallback must forward `reference`."""
    audio_path = _make_ogg(tmp_path)
    adapter = _make_adapter()
    ref_msg, reference_obj = _make_ref_message(99, 555)

    async def fake_request(route, *, form):
        raise RuntimeError("simulated voice flag failure")

    sent_msg = SimpleNamespace(id=8888)
    send_calls = []

    async def fake_send(*, file, reference=None):
        send_calls.append({"file": file, "reference": reference})
        return sent_msg

    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=fake_request),
    )

    result = await adapter.send_voice(chat_id="555", audio_path=audio_path, reply_to="99")

    assert result.success is True
    assert result.message_id == "8888"
    assert len(send_calls) == 1
    assert send_calls[0]["reference"] is reference_obj


@pytest.mark.asyncio
async def test_send_voice_without_reply_to_sends_unthreaded(tmp_path):
    """`reply_to=None` must not raise, and the payload must omit message_reference."""
    audio_path = _make_ogg(tmp_path)
    adapter = _make_adapter()

    http_calls = []

    async def fake_request(route, *, form):
        http_calls.append({"route": route, "form": form})
        return {"id": 5555}

    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=fake_request),
    )

    result = await adapter.send_voice(chat_id="555", audio_path=audio_path, reply_to=None)

    assert result.success is True
    assert result.message_id == "5555"
    assert channel.fetch_message.await_count == 0
    assert len(http_calls) == 1
    payload = json.loads(http_calls[0]["form"][0]["value"])
    assert "message_reference" not in payload


@pytest.mark.asyncio
async def test_send_voice_reply_mode_off_sends_unthreaded(tmp_path):
    """`reply_to_mode='off'` must skip reference resolution even when reply_to is set."""
    audio_path = _make_ogg(tmp_path)
    adapter = _make_adapter(reply_to_mode="off")

    http_calls = []

    async def fake_request(route, *, form):
        http_calls.append({"route": route, "form": form})
        return {"id": 6666}

    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=fake_request),
    )

    result = await adapter.send_voice(chat_id="555", audio_path=audio_path, reply_to="99")

    assert result.success is True
    assert channel.fetch_message.await_count == 0
    payload = json.loads(http_calls[0]["form"][0]["value"])
    assert "message_reference" not in payload