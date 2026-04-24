# -*- coding: utf-8 -*-
"""Unit tests for gateway.platforms.qqbot.core.event_parser.

Tests EventParser and InboundEvent — replaces the old test_qqbot_message_handler.py.
All imports reference core/ directly; no hermes symbols used.
"""

from __future__ import annotations

import pytest

from gateway.platforms.qqbot.core.dto import (
    EventType,
    Member,
    Message,
    MessageAttachment,
    MsgElement,
    User,
)
from gateway.platforms.qqbot.core.event_parser import (
    EventParser,
    InboundEvent,
    _strip_at_mention,
)


# ── _strip_at_mention ─────────────────────────────────────────────────

class TestStripAtMention:
    def test_removes_mention(self):
        assert _strip_at_mention("@BotName Hello world") == "Hello world"

    def test_no_mention(self):
        assert _strip_at_mention("Hello world") == "Hello world"

    def test_empty(self):
        assert _strip_at_mention("") == ""

    def test_only_mention(self):
        assert _strip_at_mention("@BotName") == ""

    def test_mention_with_extra_spaces(self):
        assert _strip_at_mention("  @Bot   Hello  ") == "Hello"


# ── EventParser.parse ─────────────────────────────────────────────────

class TestEventParser:
    def setup_method(self):
        self.parser = EventParser()

    # C2C
    def test_c2c_basic(self):
        event = self.parser.parse(EventType.C2C_MESSAGE_CREATE, {
            "id": "1", "content": "Hello", "timestamp": "2026-01-01T00:00:00",
            "author": {"user_openid": "uid-1"},
        })
        assert event is not None
        assert event.chat_id == "uid-1"
        assert event.user_id == "uid-1"
        assert event.chat_scope == "c2c"
        assert event.content == "Hello"

    def test_c2c_no_openid_returns_none(self):
        assert self.parser.parse(EventType.C2C_MESSAGE_CREATE, {
            "id": "1", "content": "Hello", "author": {},
        }) is None

    # GROUP
    def test_group_strips_at_mention(self):
        event = self.parser.parse(EventType.GROUP_AT_MESSAGE_CREATE, {
            "id": "2", "content": "@Bot hi", "timestamp": "",
            "group_openid": "grp-1", "author": {"member_openid": "mem-1"},
        })
        assert event is not None
        assert event.chat_scope == "group"
        assert event.content == "hi"
        assert event.chat_id == "grp-1"
        assert event.user_id == "mem-1"

    def test_group_no_openid_returns_none(self):
        assert self.parser.parse(EventType.GROUP_AT_MESSAGE_CREATE, {
            "id": "2", "content": "hi", "author": {},
        }) is None

    # GUILD
    def test_guild_uses_nick(self):
        event = self.parser.parse(EventType.GUILD_MESSAGE_CREATE, {
            "id": "3", "content": "Hello", "timestamp": "",
            "channel_id": "ch-1",
            "author": {"id": "a-1", "username": "User"},
            "member": {"nick": "NickName"},
        })
        assert event is not None
        assert event.chat_scope == "guild"
        assert event.user_name == "NickName"

    def test_guild_falls_back_to_username(self):
        event = self.parser.parse(EventType.GUILD_MESSAGE_CREATE, {
            "id": "3", "content": "Hello", "timestamp": "",
            "channel_id": "ch-1",
            "author": {"id": "a-1", "username": "Fallback"},
        })
        assert event.user_name == "Fallback"

    def test_guild_at_message(self):
        event = self.parser.parse(EventType.GUILD_AT_MESSAGE_CREATE, {
            "id": "4", "content": "Hi", "timestamp": "",
            "channel_id": "ch-2",
            "author": {"id": "a-2", "username": "U"},
        })
        assert event is not None
        assert event.chat_scope == "guild"

    def test_guild_no_channel_returns_none(self):
        assert self.parser.parse(EventType.GUILD_MESSAGE_CREATE, {
            "id": "3", "content": "Hello", "author": {},
        }) is None

    # DM
    def test_dm_basic(self):
        event = self.parser.parse(EventType.DIRECT_MESSAGE_CREATE, {
            "id": "5", "content": "Hi", "timestamp": "",
            "guild_id": "g-1", "author": {"id": "a-1"},
        })
        assert event is not None
        assert event.chat_scope == "dm"
        assert event.chat_id == "g-1"

    def test_dm_no_guild_returns_none(self):
        assert self.parser.parse(EventType.DIRECT_MESSAGE_CREATE, {
            "id": "5", "content": "Hi", "author": {},
        }) is None

    # Unknown
    def test_unknown_event_type(self):
        assert self.parser.parse("UNKNOWN_EVENT", {"id": "1"}) is None

    def test_non_dict_raw(self):
        assert self.parser.parse(EventType.C2C_MESSAGE_CREATE, None) is None

    # Attachments passthrough
    def test_attachments_passed_through(self):
        event = self.parser.parse(EventType.C2C_MESSAGE_CREATE, {
            "id": "1", "content": "", "timestamp": "",
            "author": {"user_openid": "u-1"},
            "attachments": [
                {"url": "https://x/img.jpg", "content_type": "image/jpeg", "filename": "img.jpg"},
            ],
        })
        assert len(event.attachments) == 1
        assert event.attachments[0].content_type == "image/jpeg"
