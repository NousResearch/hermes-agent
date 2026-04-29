import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_discord_mock():
    """Install a lightweight discord mock when discord.py isn't available."""
    import sys

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
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True, load_opus=lambda *_args, **_kwargs: None)
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter, _probe_audio_duration_seconds


def test_probe_audio_duration_prefers_ffprobe():
    completed = MagicMock(stdout="6.4865\n")
    with patch("gateway.platforms.discord.subprocess.run", return_value=completed) as run_mock:
        duration = _probe_audio_duration_seconds("/tmp/sample.ogg", file_size_bytes=52651)

    assert duration == pytest.approx(6.4865)
    run_mock.assert_called_once()


@pytest.mark.asyncio
async def test_send_voice_uses_probed_duration_in_discord_payload(tmp_path):
    audio_path = tmp_path / "sample.ogg"
    audio_path.write_bytes(b"OggS" + b"\x00" * 128)

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")

    channel = SimpleNamespace(id=123)
    captured = {}

    async def fake_request(route, *, form):
        captured["form"] = form
        return {"id": "msg-123"}

    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(return_value=channel),
        http=SimpleNamespace(request=fake_request),
    )

    with patch("gateway.platforms.discord._probe_audio_duration_seconds", return_value=6.4865), \
         patch("gateway.platforms.discord.discord.http.Route", return_value=object()):
        result = await DiscordAdapter.send_voice(adapter, chat_id="123", audio_path=str(audio_path))

    assert result.success is True
    payload_json = next(item["value"] for item in captured["form"] if item["name"] == "payload_json")
    payload = json.loads(payload_json)
    assert payload["attachments"][0]["duration_secs"] == pytest.approx(6.49)
