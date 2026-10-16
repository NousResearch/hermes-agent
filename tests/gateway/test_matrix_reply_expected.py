"""Matrix admission metadata drives the current gateway silence policy."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.session import SessionContext, build_session_context_prompt
from plugins.platforms.matrix.adapter import MatrixAdapter


def _adapter():
    adapter = MatrixAdapter(PlatformConfig(enabled=True, extra={
        "homeserver": "https://example.org", "user_id": "@hermes:example.org",
    }))
    adapter._require_mention = adapter._auto_thread = False
    adapter._matrix_session_scope = "room"
    adapter._startup_ts = time.time() - 10
    adapter._text_batch_delay_seconds = 0
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="Room", room_topic="", server_name="example.org"))
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    adapter._threads = MagicMock()
    adapter._threads.mark_async = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("case, expected", [
    pytest.param({}, False, id="room"),
    pytest.param({"session_scope": "thread"}, False, id="synthetic-thread"),
    pytest.param({"content": {"m.mentions": {"user_ids": ["@other:example.org"]}}},
                 False, id="other-mention"),
    pytest.param({"body": "@hermes:example.org help", "require_mention": True,
                  "content": {"m.mentions": {"user_ids": ["@hermes:example.org"]}}},
                 True, id="mention"),
    pytest.param({"body": "@hermes:example.org help", "require_mention": True},
                 True, id="legacy-mention"),
    pytest.param({"body": "!model"}, True, id="command"),
    pytest.param({"body": "> <@hermes:example.org> earlier\n\nany update?",
                  "content": {"m.mentions": {"user_ids": []},
                              "m.relates_to": {"m.in_reply_to": {"event_id": "$earlier"}}}},
                 True, id="bot-reply"),
    pytest.param({"body": "> <@bob:example.org> earlier\n\nany update?",
                  "content": {"m.mentions": {"user_ids": []},
                              "m.relates_to": {"m.in_reply_to": {"event_id": "$earlier"}}}},
                 None, id="human-reply"),
    pytest.param({"content": {"m.relates_to": {"rel_type": "m.thread", "event_id": "$root"}}},
                 None, id="thread-followup"),
    pytest.param({"dm": True}, True, id="dm"),
    pytest.param({"claimed_voice": True, "body": "voice.ogg", "content": {"msgtype": "m.audio"}},
                 True, id="claimed-voice"),
])
async def test_matrix_event_shapes_silence_only_when_unaddressed(case, expected, caplog):
    from gateway.run import GatewayRunner

    adapter = _adapter()
    adapter._matrix_session_scope = case.get("session_scope", "room")
    adapter._require_mention = case.get("require_mention", False)
    adapter._is_dm_room.return_value = case.get("dm", False)
    content = {"msgtype": "m.text", "body": case.get("body", "side chatter"),
               **case.get("content", {})}
    if case.get("claimed_voice"):
        await adapter._handle_media_message(
            "!room:example.org", "@alice:example.org", "$message", time.time(), content,
            content.get("m.relates_to", {}), "m.audio", mention_claimed=True)
    else:
        await adapter._on_room_message(SimpleNamespace(
            room_id="!room:example.org", sender="@alice:example.org", event_id="$message",
            timestamp=int(time.time() * 1000), content=content))
    event = adapter.handle_message.await_args.args[0]
    runner = object.__new__(GatewayRunner)
    response, silent, _messages = await runner._hmwa_shape_agent_response(
        {"final_response": "NO_REPLY", "messages": []}, event.source, [], None, "", "", 1,
        None, "matrix", time.time(), reply_expected=event.reply_expected)

    assert silent is (expected is False)
    assert event.reply_expected is expected
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "silence marker" in r.message]
    assert bool(warnings) is (expected is not False)
    if not silent:
        assert response and response != "NO_REPLY"


def test_shared_room_guidance_limits_silence_to_unaddressed_messages():
    from gateway.config import Platform
    from gateway.session import SessionSource

    context = SessionContext(
        source=SessionSource(platform=Platform.MATRIX, chat_id="!room:example.org", chat_type="group"),
        connected_platforms=[Platform.MATRIX], home_channels={}, session_key="test", session_id="test")
    prompt = build_session_context_prompt(context)

    assert "unaddressed" in prompt
    assert "NO_REPLY" in prompt
    assert "must answer" in prompt
