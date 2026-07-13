"""Tests for WhatsApp channel_prompt and auto_skill (channel_skill_bindings) resolution."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _make_adapter():
    """Create a WhatsAppAdapter instance bypassing __init__ (for resolver unit tests)."""
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = {}
    return adapter


def _make_real_adapter(extra: dict | None = None):
    """Create a real WhatsAppAdapter via PlatformConfig (for _build_message_event tests)."""
    base_extra = {"session_name": "test", "dm_policy": "allowlist", "allow_from": ["*"]}
    if extra:
        base_extra.update(extra)
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=base_extra))


class TestResolveChannelPrompt:
    def test_no_prompt_config_returns_none(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_prompt("12345@s.whatsapp.net") is None

    def test_match_by_chat_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_prompts": {"12345@s.whatsapp.net": "Support mode"}}
        assert adapter._resolve_channel_prompt("12345@s.whatsapp.net") == "Support mode"

    def test_non_matching_chat_id_returns_none(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_prompts": {"99999@s.whatsapp.net": "Other prompt"}}
        assert adapter._resolve_channel_prompt("12345@s.whatsapp.net") is None

    def test_blank_prompt_is_ignored(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_prompts": {"12345@s.whatsapp.net": "   "}}
        assert adapter._resolve_channel_prompt("12345@s.whatsapp.net") is None


class TestResolveChannelSkills:
    def test_no_skill_bindings_returns_none(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_skills("12345@s.whatsapp.net") is None

    def test_match_by_chat_id_returns_skills_list(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "12345@s.whatsapp.net", "skills": ["skill-a", "skill-b"]},
            ]
        }
        result = adapter._resolve_channel_skills("12345@s.whatsapp.net")
        assert result == ["skill-a", "skill-b"]

    def test_single_skill_string_accepted(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "12345@s.whatsapp.net", "skill": "solo-skill"},
            ]
        }
        result = adapter._resolve_channel_skills("12345@s.whatsapp.net")
        assert result == ["solo-skill"]

    def test_non_matching_chat_id_returns_none(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_skill_bindings": [
                {"id": "99999@s.whatsapp.net", "skills": ["skill-a"]},
            ]
        }
        assert adapter._resolve_channel_skills("12345@s.whatsapp.net") is None


@pytest.mark.asyncio
async def test_build_message_event_sets_channel_prompt_and_auto_skill():
    """_build_message_event should populate channel_prompt and auto_skill from config."""
    chat_id = "15551234567@s.whatsapp.net"
    adapter = _make_real_adapter(
        extra={
            "channel_prompts": {chat_id: "Support agent prompt"},
            "channel_skill_bindings": [
                {"id": chat_id, "skills": ["crm-lookup", "ticket-create"]},
            ],
        }
    )

    event = await adapter._build_message_event(
        {
            "body": "hello, I need help",
            "chatId": chat_id,
            "chatName": "Customer Chat",
            "senderId": chat_id,
            "senderName": "Customer",
            "isGroup": False,
        }
    )

    assert event is not None
    assert event.channel_prompt == "Support agent prompt"
    assert event.auto_skill == ["crm-lookup", "ticket-create"]


@pytest.mark.asyncio
async def test_build_message_event_no_config_returns_none_fields():
    """Without channel_prompts/skill_bindings config, fields should be None."""
    chat_id = "15551234567@s.whatsapp.net"
    adapter = _make_real_adapter()

    event = await adapter._build_message_event(
        {
            "body": "hello",
            "chatId": chat_id,
            "chatName": "Chat",
            "senderId": chat_id,
            "senderName": "User",
            "isGroup": False,
        }
    )

    assert event is not None
    assert event.channel_prompt is None
    assert event.auto_skill is None
