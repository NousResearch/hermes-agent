"""An unavailable voice reply must not look like voice mode was switched off."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.session import build_session_key
from tests.gateway.test_base_auto_tts_output_format import (
    _DummyAdapter, _hold_typing, _make_voice_event,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["requirements", "result", "exception", "empty"])
async def test_voice_failure_preserves_text_and_notifies_once(failure):
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _: True
    adapter.play_tts = AsyncMock()
    adapter.set_message_handler(lambda _: asyncio.sleep(0, result="reply text"))
    event = _make_voice_event(Platform.TELEGRAM)
    kwargs = {"return_value": json.dumps({"success": False, "error": "private-secret"})}
    if failure == "exception":
        kwargs = {"side_effect": RuntimeError("private-secret")}
    elif failure == "empty":
        kwargs = {"return_value": json.dumps({"success": True, "file_paths": []})}
    with patch("tools.tts_tool.check_tts_requirements", return_value=failure != "requirements"), patch(
        "tools.tts_tool.text_to_speech_tool", **kwargs
    ):
        for _ in range(2):
            await adapter._process_message_background(event, build_session_key(event.source))
    assert [item["content"] for item in adapter.sent].count("reply text") == 2
    notices = [item["content"] for item in adapter.sent if item["content"] != "reply text"]
    assert len(notices) == 1, "failed voice synthesis must explain the text-only fallback once"
    assert "voice" in notices[0].lower()
    assert "private-secret" not in notices[0]
    assert adapter.sent[0]["content"] == "reply text"
    adapter.play_tts.assert_not_awaited()


@pytest.mark.asyncio
async def test_notice_policy_retry_routing_and_stream_metadata():
    from gateway.platforms.base import SendResult

    adapter = _DummyAdapter(Platform.TELEGRAM)
    event = _make_voice_event(Platform.TELEGRAM)
    key = build_session_key(event.source)
    metadata = {"thread_id": "topic-7", "profile": "other"}
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    with patch.object(adapter, "warning_notifications_enabled", return_value=False):
        await adapter._notify_auto_tts_failure(event, key, metadata)
    adapter.send.assert_not_awaited()
    with patch.object(adapter, "warning_notifications_enabled", return_value=True):
        adapter.send.side_effect = RuntimeError("transport failure")
        await adapter._notify_auto_tts_failure(event, key, metadata)
        adapter.send.side_effect = None
        await adapter._notify_auto_tts_failure(event, key, metadata)
        await adapter._notify_auto_tts_failure(event, key, metadata)
        assert adapter.send.await_count == 2
        await adapter._notify_auto_tts_failure(event, key + ":another-profile", metadata)
    assert adapter.send.await_count == 3
    assert adapter.send.call_args.kwargs["metadata"] == {**metadata, "_interim_send": True}
    assert adapter.send.call_args.kwargs["reply_to"] == event.message_id
    assert "_interim_send" not in metadata


@pytest.mark.asyncio
async def test_success_resets_outage_and_disabled_voice_has_no_notice(tmp_path):
    from gateway.platforms.base import SendResult

    adapter = _DummyAdapter(Platform.SLACK)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _: True
    adapter.play_tts = AsyncMock(return_value=SendResult(success=True))
    adapter.set_message_handler(lambda _: asyncio.sleep(0, result="reply text"))
    event = _make_voice_event(Platform.SLACK)
    key = build_session_key(event.source)

    def success(**_):
        path = tmp_path / "voice.mp3"
        path.write_bytes(b"fixture audio")
        return json.dumps({"success": True, "file_path": str(path)})

    with patch("tools.tts_tool.check_tts_requirements", return_value=True), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=success
    ):
        await adapter._process_message_background(event, key)
        assert len(adapter.sent) == 1
        assert adapter.play_tts.await_count == 1
        with patch("tools.tts_tool.check_tts_requirements", return_value=False):
            await adapter._process_message_background(event, key)
        await adapter._process_message_background(event, key)
        with patch("tools.tts_tool.check_tts_requirements", return_value=False):
            await adapter._process_message_background(event, key)
            adapter._should_auto_tts_for_chat = lambda _: False
            await adapter._process_message_background(event, key)
    assert [item["content"] for item in adapter.sent].count("reply text") == 5
    assert len(adapter.sent) == 7
