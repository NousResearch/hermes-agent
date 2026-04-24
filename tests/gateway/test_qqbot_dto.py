# -*- coding: utf-8 -*-
"""Tests for gateway.platforms.qqbot.dto — strongly-typed DTO models."""

from __future__ import annotations

import pytest

from gateway.platforms.qqbot.core.dto import (
    CloseAction,
    EventType,
    Intent,
    DEFAULT_INTENTS,
    MESSAGE_EVENT_TYPES,
    Member,
    Message,
    MessageAttachment,
    OPCode,
    QQMessageType,
    User,
    WSHelloData,
    WSPayload,
    WSReadyData,
    classify_close_code,
    parse_hello,
    parse_message,
    parse_ready,
    parse_ws_payload,
)


# ── OPCode ────────────────────────────────────────────────────────────

class TestOPCode:
    def test_values(self):
        assert OPCode.DISPATCH == 0
        assert OPCode.HEARTBEAT == 1
        assert OPCode.IDENTIFY == 2
        assert OPCode.RESUME == 6
        assert OPCode.HELLO == 10
        assert OPCode.HEARTBEAT_ACK == 11


# ── EventType ─────────────────────────────────────────────────────────

class TestEventType:
    def test_message_events_in_frozenset(self):
        assert EventType.C2C_MESSAGE_CREATE in MESSAGE_EVENT_TYPES
        assert EventType.GROUP_AT_MESSAGE_CREATE in MESSAGE_EVENT_TYPES
        assert EventType.READY not in MESSAGE_EVENT_TYPES

    def test_string_value(self):
        assert EventType.READY == "READY"
        assert EventType.C2C_MESSAGE_CREATE == "C2C_MESSAGE_CREATE"


# ── Intent ────────────────────────────────────────────────────────────

class TestIntent:
    def test_default_intents(self):
        assert Intent.GROUP_MESSAGES in DEFAULT_INTENTS
        assert Intent.DIRECT_MESSAGES in DEFAULT_INTENTS
        assert Intent.GUILD_AT_MESSAGE in DEFAULT_INTENTS

    def test_bitmask_values(self):
        assert Intent.GROUP_MESSAGES == 1 << 25
        assert Intent.GUILD_AT_MESSAGE == 1 << 30
        assert Intent.DIRECT_MESSAGES == 1 << 12


# ── User ──────────────────────────────────────────────────────────────

class TestUser:
    def test_default(self):
        u = User()
        assert u.id == ""
        assert u.username == ""
        assert u.user_openid == ""
        assert u.member_openid == ""
        assert u.bot is False

    def test_with_values(self):
        u = User(id="123", username="test", user_openid="oid-1")
        assert u.id == "123"
        assert u.username == "test"
        assert u.user_openid == "oid-1"


# ── MessageAttachment ─────────────────────────────────────────────────

class TestMessageAttachment:
    def test_resolved_url_normal(self):
        att = MessageAttachment(url="https://example.com/img.jpg")
        assert att.resolved_url == "https://example.com/img.jpg"

    def test_resolved_url_protocol_relative(self):
        att = MessageAttachment(url="//cdn.qq.com/voice.silk")
        assert att.resolved_url == "https://cdn.qq.com/voice.silk"

    def test_resolved_url_empty(self):
        att = MessageAttachment(url="")
        assert att.resolved_url == ""

    def test_resolved_url_with_spaces(self):
        att = MessageAttachment(url="  //cdn.qq.com/img.png  ")
        assert att.resolved_url == "https://cdn.qq.com/img.png"


# ── parse_message ─────────────────────────────────────────────────────

class TestParseMessage:
    def test_full_message(self):
        raw = {
            "id": "msg-001",
            "content": "Hello world",
            "timestamp": "2024-01-01T00:00:00+08:00",
            "channel_id": "ch-1",
            "guild_id": "guild-1",
            "group_openid": "group-1",
            "author": {
                "id": "author-1",
                "username": "testuser",
                "user_openid": "uoid-1",
                "member_openid": "moid-1",
            },
            "member": {
                "nick": "TestNick",
            },
            "attachments": [
                {
                    "url": "//cdn.qq.com/img.jpg",
                    "filename": "img.jpg",
                    "content_type": "image/jpeg",
                    "voice_wav_url": "",
                    "asr_refer_text": "",
                },
                {
                    "url": "https://cdn.qq.com/voice.silk",
                    "filename": "voice.silk",
                    "content_type": "voice",
                    "voice_wav_url": "//cdn.qq.com/voice.wav",
                    "asr_refer_text": "Hello voice",
                },
            ],
        }
        msg = parse_message(raw)
        assert msg.id == "msg-001"
        assert msg.content == "Hello world"
        assert msg.channel_id == "ch-1"
        assert msg.group_openid == "group-1"
        assert msg.author.id == "author-1"
        assert msg.author.user_openid == "uoid-1"
        assert msg.member is not None
        assert msg.member.nick == "TestNick"
        assert len(msg.attachments) == 2
        assert msg.attachments[0].filename == "img.jpg"
        assert msg.attachments[1].asr_refer_text == "Hello voice"
        assert msg.attachments[1].resolved_url == "https://cdn.qq.com/voice.silk"

    def test_empty_message(self):
        msg = parse_message({})
        assert msg.id == ""
        assert msg.content == ""
        assert msg.author.id == ""
        assert msg.attachments == []
        assert msg.member is None

    def test_missing_author(self):
        msg = parse_message({"id": "1", "author": None})
        assert msg.author.id == ""

    def test_non_dict_attachments_ignored(self):
        msg = parse_message({"attachments": ["not-a-dict", 123]})
        assert msg.attachments == []

    def test_parse_message_type_and_scene(self):
        """message_type and message_scene.ext should be parsed."""
        raw = {
            "id": "msg-quote",
            "content": "这是什么",
            "message_type": 103,
            "message_scene": {
                "ext": [
                    "",
                    "ref_msg_idx=REFIDX_abc",
                    "msg_idx=REFIDX_def",
                ],
            },
        }
        msg = parse_message(raw)
        assert msg.message_type == 103
        assert msg.message_scene is not None
        assert len(msg.message_scene.ext) == 3
        assert "ref_msg_idx=REFIDX_abc" in msg.message_scene.ext

    def test_parse_msg_elements(self):
        """msg_elements with content and attachments should be parsed."""
        raw = {
            "id": "msg-quote",
            "content": "reply text",
            "message_type": 103,
            "msg_elements": [
                {
                    "msg_idx": "REFIDX_orig",
                    "content": "Original content",
                    "attachments": [
                        {
                            "content_type": "image/jpeg",
                            "filename": "photo.png",
                            "url": "https://example.com/img",
                            "height": 100,
                            "width": 200,
                        },
                    ],
                },
            ],
        }
        msg = parse_message(raw)
        assert len(msg.msg_elements) == 1
        elem = msg.msg_elements[0]
        assert elem.msg_idx == "REFIDX_orig"
        assert elem.content == "Original content"
        assert len(elem.attachments) == 1
        assert elem.attachments[0].content_type == "image/jpeg"
        assert elem.attachments[0].filename == "photo.png"

    def test_parse_msg_elements_empty(self):
        """Message without msg_elements should have empty list."""
        msg = parse_message({"id": "1"})
        assert msg.msg_elements == []
        assert msg.message_scene is None
        assert msg.message_type == 0

    def test_parse_message_scene_missing_ext(self):
        """message_scene without ext should be None."""
        msg = parse_message({"id": "1", "message_scene": {}})
        assert msg.message_scene is None

    def test_parse_real_push_data(self):
        """Integration test with real QQ push data format."""
        import json
        raw = json.loads(
            '{"author":{"bot":false,"id":"FC6A2DE","user_openid":"FC6A2DE"},'
            '"content":"这是什么","id":"ROBOT1.0_xxx",'
            '"message_scene":{"ext":["","ref_msg_idx=REFIDX_abc","msg_idx=REFIDX_def"]},'
            '"message_type":103,'
            '"msg_elements":[{"attachments":[{"content":"","content_type":"image/jpeg",'
            '"filename":"photo.png","height":2622,"size":510336,'
            '"url":"https://multimedia.nt.qq.com.cn/download","width":1966}],'
            '"content":"","msg_idx":"REFIDX_abc"}],'
            '"timestamp":"2026-04-20T20:48:54+08:00"}'
        )
        msg = parse_message(raw)
        assert msg.message_type == 103
        assert msg.content == "这是什么"
        assert len(msg.msg_elements) == 1
        assert msg.msg_elements[0].msg_idx == "REFIDX_abc"
        assert msg.msg_elements[0].attachments[0].content_type == "image/jpeg"
        assert msg.message_scene is not None
        assert "ref_msg_idx=REFIDX_abc" in msg.message_scene.ext


# ── parse_ws_payload ──────────────────────────────────────────────────

class TestParseWSPayload:
    def test_basic(self):
        raw = {"op": 10, "s": 42, "t": "READY", "d": {"session_id": "s1"}}
        payload = parse_ws_payload(raw)
        assert payload.op == 10
        assert payload.s == 42
        assert payload.t == "READY"
        assert payload.d == {"session_id": "s1"}

    def test_minimal(self):
        payload = parse_ws_payload({"op": 11})
        assert payload.op == 11
        assert payload.s is None
        assert payload.t == ""


# ── parse_hello ───────────────────────────────────────────────────────

class TestParseHello:
    def test_normal(self):
        hello = parse_hello({"heartbeat_interval": 41250})
        assert hello.heartbeat_interval == 41250

    def test_default(self):
        hello = parse_hello(None)
        assert hello.heartbeat_interval == 30000


# ── parse_ready ───────────────────────────────────────────────────────

class TestParseReady:
    def test_with_session(self):
        ready = parse_ready({"session_id": "s-123", "version": 1})
        assert ready.session_id == "s-123"
        assert ready.version == 1

    def test_default(self):
        ready = parse_ready({})
        assert ready.session_id == ""


# ── classify_close_code ──────────────────────────────────────────────

class TestClassifyCloseCode:
    def test_fatal_4914(self):
        assert classify_close_code(4914) == CloseAction.STOP

    def test_fatal_4915(self):
        assert classify_close_code(4915) == CloseAction.STOP

    def test_rate_limit(self):
        assert classify_close_code(4008) == CloseAction.RATE_LIMIT

    def test_clear_token_is_now_fatal(self):
        # 4004 is not in the official close code table; treated as unknown → RECONNECT
        assert classify_close_code(4004) == CloseAction.RECONNECT

    def test_identify_only(self):
        assert classify_close_code(4006) == CloseAction.IDENTIFY_ONLY
        assert classify_close_code(4007) == CloseAction.IDENTIFY_ONLY
        assert classify_close_code(4900) == CloseAction.IDENTIFY_ONLY
        assert classify_close_code(4913) == CloseAction.IDENTIFY_ONLY

    def test_resume_ok(self):
        assert classify_close_code(4009) == CloseAction.RESUME_OK

    def test_unknown_reconnect(self):
        assert classify_close_code(4999) == CloseAction.RECONNECT
        assert classify_close_code(1000) == CloseAction.RECONNECT

    def test_none(self):
        assert classify_close_code(None) == CloseAction.RECONNECT


# ── QQMessageType ────────────────────────────────────────────────────

class TestQQMessageType:
    def test_values(self):
        assert QQMessageType.TEXT == 0
        assert QQMessageType.MARKDOWN == 2
        assert QQMessageType.RICH_MEDIA == 7
        assert QQMessageType.INPUT_NOTIFY == 6


# ── MessageToCreate.to_dict ──────────────────────────────────────────

class TestMessageToCreateToDict:
    def test_text_message(self):
        from gateway.platforms.qqbot.core.dto import MessageToCreate
        msg = MessageToCreate(content="hello", msg_type=0, msg_seq=1)
        d = msg.to_dict()
        assert d == {"msg_type": 0, "msg_seq": 1, "content": "hello"}

    def test_markdown_message(self):
        from gateway.platforms.qqbot.core.dto import MarkdownContent, MessageToCreate
        msg = MessageToCreate(
            msg_type=2,
            msg_seq=10,
            markdown=MarkdownContent(content="**bold**"),
        )
        d = msg.to_dict()
        assert d["msg_type"] == 2
        assert d["markdown"] == {"content": "**bold**"}
        assert "content" not in d

    def test_with_reference(self):
        from gateway.platforms.qqbot.core.dto import MessageReference, MessageToCreate
        msg = MessageToCreate(
            content="reply",
            msg_type=0,
            msg_id="m-1",
            message_reference=MessageReference(message_id="m-1"),
        )
        d = msg.to_dict()
        assert d["msg_id"] == "m-1"
        assert d["message_reference"] == {"message_id": "m-1"}

    def test_with_media(self):
        from gateway.platforms.qqbot.core.dto import MediaInfo, MessageToCreate
        msg = MessageToCreate(
            msg_type=7,
            media=MediaInfo(file_info="abc123"),
        )
        d = msg.to_dict()
        assert d["media"] == {"file_info": "abc123"}

    def test_with_input_notify(self):
        from gateway.platforms.qqbot.core.dto import InputNotify, MessageToCreate
        msg = MessageToCreate(
            msg_type=6,
            msg_id="orig-1",
            input_notify=InputNotify(input_type=1, input_second=60),
        )
        d = msg.to_dict()
        assert d["input_notify"] == {"input_type": 1, "input_second": 60}

    def test_omits_empty_fields(self):
        from gateway.platforms.qqbot.core.dto import MessageToCreate
        msg = MessageToCreate(msg_type=0)
        d = msg.to_dict()
        assert "content" not in d
        assert "msg_id" not in d
        assert "markdown" not in d
        assert "media" not in d


# ── RichMediaMessage.to_dict ─────────────────────────────────────────

class TestRichMediaMessageToDict:
    def test_url_upload(self):
        from gateway.platforms.qqbot.core.dto import RichMediaMessage
        msg = RichMediaMessage(file_type=1, url="https://example.com/img.jpg")
        d = msg.to_dict()
        assert d == {"file_type": 1, "srv_send_msg": False, "url": "https://example.com/img.jpg"}

    def test_file_data_upload(self):
        from gateway.platforms.qqbot.core.dto import RichMediaMessage
        msg = RichMediaMessage(file_type=4, file_data="base64data", file_name="doc.pdf")
        d = msg.to_dict()
        assert d["file_data"] == "base64data"
        assert d["file_name"] == "doc.pdf"
        assert "url" not in d

    def test_omits_empty_fields(self):
        from gateway.platforms.qqbot.core.dto import RichMediaMessage
        msg = RichMediaMessage(file_type=1, srv_send_msg=True)
        d = msg.to_dict()
        assert "url" not in d
        assert "file_data" not in d
        assert "file_name" not in d


# ── GuildMessageToCreate.to_dict ─────────────────────────────────────

class TestGuildMessageToCreateToDict:
    def test_basic(self):
        from gateway.platforms.qqbot.core.dto import GuildMessageToCreate
        msg = GuildMessageToCreate(content="hello", msg_id="m-1")
        d = msg.to_dict()
        assert d == {"content": "hello", "msg_id": "m-1"}

    def test_no_msg_id(self):
        from gateway.platforms.qqbot.core.dto import GuildMessageToCreate
        msg = GuildMessageToCreate(content="hello")
        d = msg.to_dict()
        assert d == {"content": "hello"}
