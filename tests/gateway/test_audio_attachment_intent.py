"""Audio file intent survives adapter normalization and pending-event merges."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.weixin import WeixinAdapter
from gateway.run import GatewayRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [[4], [3], [4, 3], [2, 4]])
async def test_weixin_only_native_voice_items_enter_stt(items, monkeypatch):
    adapter = WeixinAdapter(PlatformConfig(enabled=True, token="test", extra={"account_id": "bot"}))
    adapter._poll_session = object()
    adapter._token = ""
    monkeypatch.setattr(adapter, "_is_dm_intake_allowed", lambda _: True)
    paths = {2: ("/cache/image.jpg", "image/jpeg"), 3: ("/cache/voice.silk", "audio/silk"),
             4: ("/cache/song.m4a", "audio/mp4")}

    async def download(item, spec):
        return paths[item["type"]]

    monkeypatch.setattr(adapter, "_download_media", download)
    adapter.handle_message = AsyncMock()
    await adapter._process_message({"from_user_id": "user", "message_id": "one",
                                    "item_list": [{"type": kind} for kind in items]})
    event = adapter.handle_message.call_args.args[0]
    _, voice, files, _ = GatewayRunner._classify_inbound_media(event, False)
    assert voice == ([paths[3][0]] if 3 in items else [])
    assert files == ([paths[4][0]] if 4 in items else [])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("with_mime", [False, True])
def test_pending_merge_preserves_voice_and_audio_file_intent(reverse, with_mime):
    events = [MessageEvent(text="", message_type=MessageType.AUDIO,
                           media_urls=["/cache/song.mp3"], media_types=["audio/mpeg"]),
              MessageEvent(text="", message_type=MessageType.VOICE,
                           media_urls=["/cache/voice.ogg"], media_types=["audio/ogg"])]
    if not with_mime:
        for event in events:
            event.media_types = []
    if reverse:
        events.reverse()
    pending = {"session": events[0]}
    merge_pending_message_event(pending, "session", events[1])
    _, voice, files, _ = GatewayRunner._classify_inbound_media(pending["session"], False)
    assert voice == ["/cache/voice.ogg"]
    assert files == ["/cache/song.mp3"]


@pytest.mark.asyncio
async def test_native_envelope_preserves_mixed_audio_intent(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session_authority import initialize_session_authority
    from gateway.session_envelope import snapshot_native, restore_native

    monkeypatch.setenv("WEIXIN_ALLOWED_USERS", "user")
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = WeixinAdapter(PlatformConfig(enabled=True, token="test", extra={"account_id": "bot"}))
    runner.adapters = {Platform.WEIXIN: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="audio")
    paths = [tmp_path / "song.mp3", tmp_path / "voice.ogg"]
    for path in paths:
        path.write_bytes(b"audio fixture " + path.name.encode())
    event = MessageEvent(text="compare", source=adapter.build_source(chat_id="user", user_id="user"),
                         message_type=MessageType.VOICE, media_urls=list(map(str, paths)),
                         media_types=["audio/mpeg", "audio/ogg"], media_is_voice=[False, True])
    try:
        restored = restore_native(snapshot_native(runner, event))
        _, voice, files, _ = runner._classify_inbound_media(restored, False)
        assert len(voice) == len(files) == 1
        assert Path(voice[0]).read_bytes() == paths[1].read_bytes()
        assert Path(files[0]).read_bytes() == paths[0].read_bytes()
    finally:
        authority.db.close()
