"""Teams require_mention gate: group/channel posts need an @mention (regression for #113577).

The flag was accepted on the config surface but never read, so the adapter answered every
group-chat message even with ``require_mention: true``. These tests pin the intended behavior:
1:1 chats always respond; group chats and channels are gated on an @mention unless the flag
is disabled (extra ``require_mention`` or ``TEAMS_REQUIRE_MENTION``).
"""

import os
import types
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from plugins.platforms.teams.adapter import TeamsAdapter


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TEAMS_REQUIRE_MENTION", raising=False)
    yield
    os.environ.pop("TEAMS_REQUIRE_MENTION", None)


def _adapter(extra=None) -> TeamsAdapter:
    adapter = object.__new__(TeamsAdapter)
    adapter.config = PlatformConfig(enabled=True, extra=dict(extra or {}))
    adapter.platform = Platform("teams")
    adapter.gateway_runner = None
    adapter._app = None
    adapter._dedup = MessageDeduplicator(max_size=1000)
    adapter._conv_refs = {}
    adapter._tenant_id = ""
    adapter.handle_message = AsyncMock()
    return adapter


def _activity(text, conv_type, msg_id="msg-1"):
    return types.SimpleNamespace(
        id=msg_id, text=text,
        from_=types.SimpleNamespace(id="user-1", name="U", aad_object_id=None),
        conversation=types.SimpleNamespace(
            id="conv-1", name="", conversation_type=conv_type, tenant_id="t"),
        attachments=[],
    )


async def _dispatch(adapter, text, conv_type, msg_id="msg-1"):
    ctx = types.SimpleNamespace(activity=_activity(text, conv_type, msg_id), conversation_ref=None)
    await adapter._on_message(ctx)
    return adapter.handle_message.await_count


class TestRequireMentionFlag:
    def test_default_true(self):
        assert _adapter()._teams_require_mention() is True

    def test_extra_false_disables(self):
        assert _adapter({"require_mention": False})._teams_require_mention() is False

    def test_env_false_disables(self, monkeypatch):
        monkeypatch.setenv("TEAMS_REQUIRE_MENTION", "false")
        assert _adapter()._teams_require_mention() is False

    def test_env_beats_extra(self, monkeypatch):
        monkeypatch.setenv("TEAMS_REQUIRE_MENTION", "true")
        assert _adapter({"require_mention": False})._teams_require_mention() is True


class TestRequireMentionGate:
    @pytest.mark.asyncio
    async def test_groupchat_without_mention_ignored_by_default(self):
        assert await _dispatch(_adapter(), "hello bot", "groupChat") == 0

    @pytest.mark.asyncio
    async def test_channel_without_mention_ignored_by_default(self):
        assert await _dispatch(_adapter(), "hello bot", "channel") == 0

    @pytest.mark.asyncio
    async def test_groupchat_with_mention_processed(self):
        assert await _dispatch(_adapter(), "<at>Bot</at> hello", "groupChat") == 1

    @pytest.mark.asyncio
    async def test_groupchat_without_mention_processed_when_disabled(self):
        assert await _dispatch(_adapter({"require_mention": False}), "hello bot", "groupChat") == 1

    @pytest.mark.asyncio
    async def test_personal_chat_always_processed(self):
        assert await _dispatch(_adapter(), "hello bot", "personal") == 1
