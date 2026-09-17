"""Salt R3 B4/S1/S3: refactor regressions reproduced as behavior contracts."""
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from gateway.config import Platform


def _cfg(tmp_path, monkeypatch, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path)); monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_discord_forum_caption_identical_to_notice_still_posts(tmp_path, monkeypatch, setting):
    """B4: text equality is not a policy signal; a requested caption equal to the generated notice travels when hidden."""
    _cfg(tmp_path, monkeypatch, setting)
    from plugins.platforms.discord.adapter_media import DiscordMediaMixin
    from gateway.platforms.base import BasePlatformAdapter
    class A(DiscordMediaMixin):
        name = "discord"; platform = Platform.DISCORD
        warning_notifications_enabled = BasePlatformAdapter.warning_notifications_enabled
        _discord_upload_limit_bytes = staticmethod(lambda ch: 1)
        _is_forum_parent = staticmethod(lambda ch: True)
        _send_to_forum = AsyncMock()
    a = A(); channel = MagicMock(); channel.send = AsyncMock()
    big = tmp_path / "big.bin"; big.write_bytes(b"xx")
    limit_mb = 1 / (1024 * 1024)
    notice = (f"⚠️ Could not attach `big.bin` — {2/(1024*1024):.1f} MB exceeds Discord's "
              f"{limit_mb:.0f} MB upload limit for this channel. Compress the file or share a link instead.")
    res = await a._reject_oversized_upload(channel, str(big), "big.bin", caption=notice)
    assert res.success is False
    if setting is True:
        A._send_to_forum.assert_awaited_once_with(channel, notice)  # the CAPTION (identical text) is requested content
    else:
        A._send_to_forum.assert_not_awaited(); channel.send.assert_not_awaited()  # legacy: no bare notice on forum parents


@pytest.mark.asyncio
async def test_slack_interim_send_never_seals_prefix_matching_stream(tmp_path, monkeypatch):
    """S3: an explicit interim warning whose text extends the streamed prefix leaves the final stream open."""
    _cfg(tmp_path, monkeypatch, None)
    from plugins.platforms.slack.adapter import SlackAdapter
    a = SlackAdapter.__new__(SlackAdapter)
    a._active_streams = {"C1": {"ts": "1.0", "sent": "⚠️"}}
    a._seal_stream = AsyncMock(return_value=True)
    a._outbound_blocked = lambda *args: None
    a._dm_target = AsyncMock(return_value="C1")
    a._metadata_team_id = lambda m: None
    a._pop_slash_context = lambda c, t: None
    a.format_message = lambda c: c
    a._post_message = AsyncMock(return_value={"ok": True, "ts": "2.0"})
    a._clear_thread_status_quietly = AsyncMock()
    a._send_plain = AsyncMock(return_value=MagicMock(success=True))
    try:
        await a.send("C1", "⚠️ Couldn't deliver the image attachment.", metadata={"_interim_send": True})
    except Exception:
        pass  # transport internals beyond the seal decision are not under test
    a._seal_stream.assert_not_awaited()
    assert "C1" in a._active_streams


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False])
async def test_matrix_shown_fallback_routing_is_legacy_bytes(tmp_path, monkeypatch, setting):
    """S1: legacy shown fallback passed NO metadata; the shared helper must not add thread routing."""
    _cfg(tmp_path, monkeypatch, setting)
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    calls = []
    class A(BasePlatformAdapter):
        name = "matrix"; platform = Platform.MATRIX
        def __init__(self): pass
        async def connect(self): pass
        async def disconnect(self): pass
        async def get_chat_info(self, chat_id): return {}
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            calls.append((content, reply_to, metadata)); return SendResult(success=True)
    a = A()
    await a.emit_media_warning("!r", "⚠️ Couldn't deliver the attachment.", caption="cap", reply_to="$e", metadata={"thread_id": "$t"})
    assert calls == [("cap\n⚠️ Couldn't deliver the attachment.", "$e", None)]
