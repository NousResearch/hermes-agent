"""Tests for Mattermost adapter message reactions (👀 / ✅ / ❌).

The completion-side swap runs through the shared reaction-ack flow in
``BasePlatformAdapter.on_processing_complete`` — the adapter opts in via the
``_ACK_EMOJI`` / ``_OK_EMOJI`` / ``_FAIL_EMOJI`` class attributes and the
base-shaped ``_add_reaction`` / ``_remove_reaction`` primitives.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome


def _make_adapter(*, bot_user_id: str = "bot_xyz"):
    """Build a MattermostAdapter wired up enough to invoke reaction methods."""
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com"},
    )
    adapter = MattermostAdapter(config)
    adapter._bot_user_id = bot_user_id
    adapter._session = MagicMock()  # truthy session so guards pass
    adapter._api_post = AsyncMock(return_value={"id": "reaction_1"})
    adapter._api_delete = AsyncMock(return_value=True)
    return adapter


def _make_event(post_id: str = "post_abc", chat_id: str = "chan_1") -> MessageEvent:
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=SimpleNamespace(chat_id=chat_id),
        raw_message={"id": post_id} if post_id else None,
        message_id=post_id or None,
    )


class TestOptIn:
    def test_ack_emoji_class_attributes(self):
        """The adapter opts into the base reaction-ack flow via class attrs."""
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        assert MattermostAdapter._ACK_EMOJI == "eyes"
        assert MattermostAdapter._OK_EMOJI == "white_check_mark"
        assert MattermostAdapter._FAIL_EMOJI == "x"


class TestReactionsEnabled:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        assert adapter._reactions_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "FALSE", "No"])
    def test_disabled_values(self, monkeypatch, value):
        monkeypatch.setenv("MATTERMOST_REACTIONS", value)
        adapter = _make_adapter()
        assert adapter._reactions_enabled() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", ""])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("MATTERMOST_REACTIONS", value)
        adapter = _make_adapter()
        assert adapter._reactions_enabled() is True


class TestOnProcessingStart:
    @pytest.mark.asyncio
    async def test_adds_eyes_reaction(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        await adapter.on_processing_start(_make_event("post_abc"))
        adapter._api_post.assert_awaited_once_with(
            "reactions",
            {"user_id": "bot_xyz", "post_id": "post_abc", "emoji_name": "eyes"},
        )

    @pytest.mark.asyncio
    async def test_disabled_skips_call(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_REACTIONS", "false")
        adapter = _make_adapter()
        await adapter.on_processing_start(_make_event("post_abc"))
        adapter._api_post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_message_id_skips_call(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        await adapter.on_processing_start(_make_event(""))
        adapter._api_post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sourceless_event_skips_call(self, monkeypatch):
        """Events whose source carries no chat_id (e.g. synthetic events from
        other surfaces) must not trigger a 👀 the completion side could never
        remove."""
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        event = MessageEvent(
            text="hi",
            message_type=MessageType.TEXT,
            source="user_1",  # plain string: no chat_id attribute
            raw_message=None,
            message_id="post_abc",
        )
        await adapter.on_processing_start(event)
        adapter._api_post.assert_not_awaited()


class TestOnProcessingComplete:
    """Completion runs the inherited base reaction-ack flow end to end."""

    @pytest.mark.asyncio
    async def test_success_swaps_eyes_for_check(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        await adapter.on_processing_complete(
            _make_event("post_abc"), ProcessingOutcome.SUCCESS
        )
        adapter._api_delete.assert_awaited_once_with(
            "users/bot_xyz/posts/post_abc/reactions/eyes"
        )
        adapter._api_post.assert_awaited_once_with(
            "reactions",
            {
                "user_id": "bot_xyz",
                "post_id": "post_abc",
                "emoji_name": "white_check_mark",
            },
        )

    @pytest.mark.asyncio
    async def test_failure_swaps_eyes_for_x(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        await adapter.on_processing_complete(
            _make_event("post_abc"), ProcessingOutcome.FAILURE
        )
        adapter._api_delete.assert_awaited_once_with(
            "users/bot_xyz/posts/post_abc/reactions/eyes"
        )
        adapter._api_post.assert_awaited_once_with(
            "reactions",
            {"user_id": "bot_xyz", "post_id": "post_abc", "emoji_name": "x"},
        )

    @pytest.mark.asyncio
    async def test_cancelled_only_removes_eyes(self, monkeypatch):
        """CANCELLED should only clear the in-progress marker, no final stamp."""
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        await adapter.on_processing_complete(
            _make_event("post_abc"), ProcessingOutcome.CANCELLED
        )
        adapter._api_delete.assert_awaited_once_with(
            "users/bot_xyz/posts/post_abc/reactions/eyes"
        )
        adapter._api_post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_skips_all_calls(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_REACTIONS", "false")
        adapter = _make_adapter()
        await adapter.on_processing_complete(
            _make_event("post_abc"), ProcessingOutcome.SUCCESS
        )
        adapter._api_post.assert_not_awaited()
        adapter._api_delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_chat_id_skips_all_calls(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        adapter = _make_adapter()
        await adapter.on_processing_complete(
            _make_event("post_abc", chat_id=""), ProcessingOutcome.SUCCESS
        )
        adapter._api_post.assert_not_awaited()
        adapter._api_delete.assert_not_awaited()


class TestReactionHelpers:
    @pytest.mark.asyncio
    async def test_add_reaction_returns_false_without_bot_id(self):
        adapter = _make_adapter(bot_user_id="")
        assert await adapter._add_reaction("chan_1", "post_abc", "eyes") is False
        adapter._api_post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_remove_reaction_returns_false_without_bot_id(self):
        adapter = _make_adapter(bot_user_id="")
        assert await adapter._remove_reaction("chan_1", "post_abc") is False
        adapter._api_delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_reaction_returns_false_on_api_failure(self):
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={})  # _api_post returns {} on failure
        assert await adapter._add_reaction("chan_1", "post_abc", "eyes") is False


class TestApplyYamlConfigReactions:
    def test_reactions_yaml_to_env(self, monkeypatch):
        monkeypatch.delenv("MATTERMOST_REACTIONS", raising=False)
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        _apply_yaml_config({}, {"reactions": False})
        import os

        assert os.environ["MATTERMOST_REACTIONS"] == "false"

    def test_env_takes_precedence_over_yaml(self, monkeypatch):
        monkeypatch.setenv("MATTERMOST_REACTIONS", "true")
        from plugins.platforms.mattermost.adapter import _apply_yaml_config

        _apply_yaml_config({}, {"reactions": False})
        import os

        assert os.environ["MATTERMOST_REACTIONS"] == "true"
