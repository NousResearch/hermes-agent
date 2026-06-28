"""Tests for the bundled NapCat QQ platform plugin."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.napcat.adapter import (
    NapCatAdapter,
    _chat_route,
    _sanitize_outgoing_text,
    _should_skip_send,
    _with_access_token,
    media_kind,
)
from tools.send_message_tool import _parse_target_ref


def test_napcat_media_kind_routes_images_voice_video_and_files():
    assert media_kind("/tmp/photo.png") == "image"
    assert media_kind("/tmp/audio.mp3") == "voice"
    assert media_kind("/tmp/audio.wav") == "voice"
    assert media_kind("/tmp/movie.mp4") == "video"
    assert media_kind("/tmp/report.pdf") == "file"
    assert media_kind("/tmp/report.pdf", is_voice=True) == "voice"


def test_napcat_chat_route_supports_group_and_private_prefixes():
    assert _chat_route("12345") == ("private", "12345")
    assert _chat_route("group:67890") == ("group", "67890")
    assert _chat_route("g:67890") == ("group", "67890")
    assert _chat_route("private:12345") == ("private", "12345")


def test_napcat_send_image_file_routes_group_media_segment(tmp_path):
    image_path = tmp_path / "result.png"
    image_path.write_bytes(b"fake png")
    adapter = _napcat_adapter()
    adapter.ws = SimpleNamespace(closed=False)
    adapter._send_action = AsyncMock(return_value={"status": "ok", "data": {"message_id": 10}})

    result = asyncio.run(adapter.send_image_file("group:610066383", str(image_path)))

    assert result.success is True
    payload = adapter._send_action.await_args.args[0]
    assert payload["action"] == "send_group_msg"
    assert payload["params"]["group_id"] == 610066383
    assert payload["params"]["message"][0]["type"] == "image"
    assert payload["params"]["message"][0]["data"]["file"].endswith("/result.png")


def test_napcat_send_video_routes_private_media_segment(tmp_path):
    video_path = tmp_path / "result.mp4"
    video_path.write_bytes(b"fake mp4")
    adapter = _napcat_adapter()
    adapter.ws = SimpleNamespace(closed=False)
    adapter._send_action = AsyncMock(return_value={"status": "ok", "data": {"message_id": 11}})

    result = asyncio.run(adapter.send_video("12345", str(video_path)))

    assert result.success is True
    payload = adapter._send_action.await_args.args[0]
    assert payload["action"] == "send_private_msg"
    assert payload["params"]["user_id"] == 12345
    assert payload["params"]["message"][0]["type"] == "video"
    assert payload["params"]["message"][0]["data"]["file"].endswith("/result.mp4")


def test_napcat_text_send_timeout_fails_and_reconnects_transport():
    adapter = _napcat_adapter()
    adapter.ws = SimpleNamespace(closed=False)
    adapter._send_action = AsyncMock(return_value=None)

    result = asyncio.run(adapter.send("12345", "hello"))

    assert result.success is False
    assert "timed out" in result.error
    assert adapter.is_connected is False
    assert adapter.fatal_error_code == "connection_lost"


def test_napcat_media_send_timeout_fails_and_reconnects_transport(tmp_path):
    image_path = tmp_path / "result.png"
    image_path.write_bytes(b"fake png")
    adapter = _napcat_adapter()
    adapter.ws = SimpleNamespace(closed=False)
    adapter._send_action = AsyncMock(return_value=None)

    result = asyncio.run(adapter.send_image_file("group:610066383", str(image_path)))

    assert result.success is False
    assert "timed out" in result.error
    assert adapter.is_connected is False
    assert adapter.fatal_error_code == "connection_lost"


class _FakeNapCatStatusWs:
    closed = False

    def __init__(self, response):
        self.response = response
        self.sent_payload = None

    async def send_json(self, payload):
        self.sent_payload = payload
        self.response["echo"] = payload["echo"]

    async def receive(self):
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(self.response))


def test_napcat_connect_status_probe_accepts_online_account():
    adapter = _napcat_adapter()
    adapter.ws = _FakeNapCatStatusWs(
        {"status": "ok", "retcode": 0, "data": {"online": True, "good": True}}
    )

    assert asyncio.run(adapter._verify_account_online_during_connect()) is True
    assert adapter.has_fatal_error is False


def test_napcat_connect_status_probe_rejects_offline_account():
    adapter = _napcat_adapter()
    adapter.ws = _FakeNapCatStatusWs(
        {"status": "ok", "retcode": 0, "data": {"online": False, "good": True}}
    )

    assert asyncio.run(adapter._verify_account_online_during_connect()) is False
    assert adapter.fatal_error_code == "account_offline"
    assert "offline" in adapter.fatal_error_message


def test_send_message_target_parser_accepts_napcat_prefixes():
    assert _parse_target_ref("napcat", "group:67890") == ("group:67890", None, True)
    assert _parse_target_ref("napcat", "direct:12345") == ("direct:12345", None, True)


def test_napcat_skip_marker_is_suppressed():
    assert _should_skip_send("[SKIP]")
    assert _should_skip_send(" skip ")
    assert _should_skip_send("[SILENT")
    assert _should_skip_send("[SILENT]")
    assert not _should_skip_send("skip this question")


def test_napcat_autonomy_leak_text_keeps_only_public_tail():
    leaked = """这是 NapCat bridge 发起的小星自主触发，不是爸爸发来的消息。
关键事实：不要发这些。
判断：需要问候。

爸，早上好，我醒来先来找你一下。"""

    assert _sanitize_outgoing_text(leaked) == "爸，早上好，我醒来先来找你一下。"


def test_napcat_autonomy_clean_short_text_is_allowed():
    text = "早上好呀，我醒来先来找你一下。"
    assert _sanitize_outgoing_text(text, {"xiaoxing_autonomy_trigger": True}) == text


def test_napcat_disables_streaming_edits_for_qq():
    assert NapCatAdapter.SUPPORTS_MESSAGE_EDITING is False


def test_napcat_access_token_append_preserves_existing_query():
    assert _with_access_token("ws://localhost:3005", "tok").endswith("?access_token=tok")
    assert _with_access_token("ws://localhost:3005/ws?foo=bar", "tok").endswith("&access_token=tok")
    assert _with_access_token("ws://localhost:3005?access_token=old", "tok").endswith("access_token=old")


def _clear_napcat_group_env(monkeypatch):
    for name in (
        "NAPCAT_GROUP_POLICY",
        "NAPCAT_GROUP_ALLOWED_CHATS",
        "NAPCAT_REQUIRE_MENTION",
        "NAPCAT_FREE_RESPONSE_CHATS",
        "NAPCAT_MENTION_PATTERNS",
        "NAPCAT_ALLOWED_USERS",
        "NAPCAT_GROUP_ALLOWED_USERS",
        "NAPCAT_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)


def _group_event(
    raw_message="hello",
    *,
    group_id="67890",
    user_id="111",
    self_id="12345",
    message=None,
    message_id=42,
    sender_card="Someone",
    sender_nickname=None,
):
    return {
        "post_type": "message",
        "message_type": "group",
        "self_id": self_id,
        "group_id": group_id,
        "user_id": user_id,
        "raw_message": raw_message,
        "message": [] if message is None else message,
        "sender": {"nickname": sender_nickname or sender_card, "card": sender_card},
        "message_id": message_id,
    }


def _napcat_adapter(extra=None):
    config = PlatformConfig(enabled=True, extra=extra or {})
    return NapCatAdapter(config)


def test_napcat_group_policy_defaults_to_disabled(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_allow_from": ["67890"]})

    assert adapter._should_process_group_message(_group_event("[CQ:at,qq=12345] hello")) is False


def test_napcat_allowlist_group_requires_mention_by_default(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "allowlist", "group_allow_from": ["67890"]})

    assert adapter._should_process_group_message(_group_event("hello")) is False
    assert adapter._should_process_group_message(_group_event("[CQ:at,qq=12345] hello")) is True


def test_napcat_ambient_group_processes_allowlisted_plain_messages(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})

    assert adapter.config.extra["group_sessions_per_user"] is False
    assert adapter._should_process_group_message(_group_event("hello")) is True
    assert adapter._should_process_group_message(_group_event("hello", group_id="99999")) is False


def test_napcat_ambient_group_keeps_messages_at_other_users_as_context(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})

    assert adapter._should_process_group_message(_group_event("[CQ:at,qq=99999] 你看看")) is True


def test_napcat_ambient_group_keeps_segment_at_other_users_as_context(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    event = _group_event(
        "",
        message=[
            {"type": "at", "data": {"qq": "99999"}},
            {"type": "text", "data": {"text": "你看看这个"}},
        ],
    )

    assert adapter._should_process_group_message(event) is True


def test_napcat_ambient_group_keeps_messages_at_bot(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})

    assert adapter._should_process_group_message(_group_event("[CQ:at,qq=12345] 小星你在吗")) is True


def test_napcat_ambient_group_overrides_gateway_session_default(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter(
        {
            "group_policy": "ambient",
            "group_allow_from": ["67890"],
            "group_sessions_per_user": True,
        }
    )

    assert adapter.config.extra["group_sessions_per_user"] is False


def test_napcat_ambient_group_event_reaches_handler_with_silent_prompt(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    asyncio.run(adapter._handle_event(_group_event("大家今天聊什么")))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "group:67890"
    assert event.source.chat_type == "group"
    assert event.source.user_id == "111"
    assert event.text == "大家今天聊什么"
    assert event.channel_prompt
    assert "[SILENT]" in event.channel_prompt


def test_napcat_group_event_marks_bot_at_in_channel_prompt(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    asyncio.run(adapter._handle_event(_group_event("[CQ:at,qq=12345] 小星你在吗")))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "[@你] 小星你在吗"
    assert "明确 @ 了你" in event.channel_prompt


def test_napcat_group_event_preserves_other_user_at_target(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    asyncio.run(adapter._handle_event(_group_event("[CQ:at,qq=99999] 你看看")))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "[@QQ:99999] 你看看"
    assert "本条消息 @ 的不是你" in event.channel_prompt
    assert "本轮最终只输出 [SILENT]" in event.channel_prompt


def test_napcat_group_reply_event_preserves_quoted_sender_context(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()
    adapter._send_action = AsyncMock(
        return_value={
            "status": "ok",
            "data": {
                "sender": {"card": "Alice", "nickname": "AliceNick", "user_id": "111"},
                "raw_message": "原来那句话",
            },
        }
    )

    asyncio.run(
        adapter._handle_event(
            _group_event("[CQ:reply,id=99]确实是这个意思", user_id="222")
        )
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == "222"
    assert event.text == "确实是这个意思"
    assert event.reply_to_message_id == "99"
    assert event.reply_to_text == "Alice|QQ:111: 原来那句话"
    payload = adapter._send_action.await_args.args[0]
    assert payload["action"] == "get_msg"
    assert payload["params"]["message_id"] == 99


def test_napcat_group_reply_event_prefers_recent_message_cache(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()
    adapter._send_action = AsyncMock()

    asyncio.run(
        adapter._handle_event(
            _group_event(
                "原来那句话",
                user_id="111",
                message_id=99,
                sender_card="Alice",
            )
        )
    )
    asyncio.run(
        adapter._handle_event(
            _group_event(
                "[CQ:reply,id=99]确实是这个意思",
                user_id="222",
                message_id=100,
                sender_card="Bob",
            )
        )
    )

    assert adapter.handle_message.await_count == 2
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == "222"
    assert event.text == "确实是这个意思"
    assert event.reply_to_message_id == "99"
    assert event.reply_to_text == "Alice|QQ:111: 原来那句话"
    adapter._send_action.assert_not_awaited()


def test_napcat_group_reply_to_recent_bot_message_marks_message_as_addressed(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.ws = SimpleNamespace(closed=False)
    adapter._send_action = AsyncMock(
        side_effect=[
            {"status": "ok", "data": {"message_id": 777}},
            None,
        ]
    )
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    send_result = asyncio.run(adapter.send("group:67890", "小星刚说的话"))
    asyncio.run(
        adapter._handle_event(
            _group_event(
                "[CQ:reply,id=777]那这个怎么办",
                user_id="222",
                message_id=778,
                sender_card="Dad",
            )
        )
    )

    assert send_result.success is True
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "那这个怎么办"
    assert event.reply_to_message_id == "777"
    assert event.reply_to_text == "小星: 小星刚说的话"
    assert adapter._send_action.await_count == 1


def test_napcat_live_group_reply_with_missing_cache_is_not_treated_as_plain_ambient(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"group_policy": "ambient", "group_allow_from": ["67890"]})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()

    async def run_live_receive_event():
        adapter._recv_task = asyncio.current_task()
        await adapter._handle_event(
            _group_event(
                "[CQ:reply,id=777]你还是没回我",
                user_id="222",
                message_id=778,
                sender_card="Dad",
            )
        )

    asyncio.run(run_live_receive_event())

    event = adapter.handle_message.await_args.args[0]
    assert event.text == "你还是没回我"
    assert event.reply_to_message_id == "777"
    assert event.reply_to_text
    assert "quoted text unavailable" in event.reply_to_text
    assert event.channel_prompt
    assert "QQ reply" in event.channel_prompt


def test_napcat_allow_all_users_bypasses_private_adapter_allowlist(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    adapter = _napcat_adapter({"allowed_users": "owner", "allow_all_users": True})
    adapter.handle_message = AsyncMock()
    adapter._message_handler = AsyncMock()
    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": "new-friend",
        "raw_message": "你好",
        "sender": {"nickname": "New Friend"},
        "message_id": 43,
    }

    asyncio.run(adapter._handle_event(event))

    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.source.user_id == "new-friend"
    assert delivered.text == "你好"


def test_gateway_authorizes_napcat_group_by_config_chat_allowlist(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform("napcat"): PlatformConfig(
                enabled=True,
                extra={"group_policy": "ambient", "group_allow_from": ["67890"]},
            )
        }
    )
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="group:67890",
        chat_type="group",
        user_id="unlisted-user",
    )

    assert runner._is_user_authorized(source) is True


def test_gateway_authorizes_napcat_private_by_config_allow_all(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform("napcat"): PlatformConfig(
                enabled=True,
                extra={"allowed_users": "owner", "allow_all_users": True},
            )
        }
    )
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="new-friend",
        chat_type="dm",
        user_id="new-friend",
    )

    assert runner._is_user_authorized(source) is True


def test_gateway_does_not_authorize_unlisted_napcat_group(monkeypatch):
    _clear_napcat_group_env(monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform("napcat"): PlatformConfig(
                enabled=True,
                extra={"group_policy": "ambient", "group_allow_from": ["67890"]},
            )
        }
    )
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_args: False)
    source = SessionSource(
        platform=Platform("napcat"),
        chat_id="group:99999",
        chat_type="group",
        user_id="unlisted-user",
    )

    assert runner._is_user_authorized(source) is False
