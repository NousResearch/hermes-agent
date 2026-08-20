"""Tests for Mattermost adapter typing indicator (thread-scope aware)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter(*, bot_user_id: str = "bot_xyz"):
    """Build a MattermostAdapter wired up enough to invoke send_typing."""
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={"url": "https://mm.example.com"},
    )
    adapter = MattermostAdapter(config)
    adapter._bot_user_id = bot_user_id
    adapter._session = MagicMock()
    adapter._api_post = AsyncMock(return_value={})
    return adapter


class TestSendTyping:
    @pytest.mark.asyncio
    async def test_channel_level_typing_has_no_parent_id(self):
        """No metadata → bubble appears in the channel composer footer."""
        adapter = _make_adapter()
        await adapter.send_typing("channel_1")
        adapter._api_post.assert_awaited_once_with(
            "users/bot_xyz/typing",
            {"channel_id": "channel_1"},
        )

    @pytest.mark.asyncio
    async def test_thread_id_metadata_scopes_to_thread(self):
        """metadata.thread_id → parent_id keeps the bubble inside the thread."""
        adapter = _make_adapter()
        await adapter.send_typing("channel_1", metadata={"thread_id": "root_post_42"})
        adapter._api_post.assert_awaited_once_with(
            "users/bot_xyz/typing",
            {"channel_id": "channel_1", "parent_id": "root_post_42"},
        )

    @pytest.mark.asyncio
    async def test_root_id_metadata_also_scopes_to_thread(self):
        """Some callers spell it root_id — accept that too."""
        adapter = _make_adapter()
        await adapter.send_typing("channel_1", metadata={"root_id": "root_post_99"})
        adapter._api_post.assert_awaited_once_with(
            "users/bot_xyz/typing",
            {"channel_id": "channel_1", "parent_id": "root_post_99"},
        )

    @pytest.mark.asyncio
    async def test_thread_id_takes_precedence_over_root_id(self):
        """When both are present (unlikely) thread_id wins."""
        adapter = _make_adapter()
        await adapter.send_typing(
            "channel_1",
            metadata={"thread_id": "thread_winner", "root_id": "root_loser"},
        )
        adapter._api_post.assert_awaited_once_with(
            "users/bot_xyz/typing",
            {"channel_id": "channel_1", "parent_id": "thread_winner"},
        )

    @pytest.mark.asyncio
    async def test_empty_thread_id_omits_parent_id(self):
        """Falsy thread_id (None / '' / 0) shouldn't pollute the payload."""
        adapter = _make_adapter()
        await adapter.send_typing("channel_1", metadata={"thread_id": ""})
        adapter._api_post.assert_awaited_once_with(
            "users/bot_xyz/typing",
            {"channel_id": "channel_1"},
        )

    @pytest.mark.asyncio
    async def test_non_dict_metadata_is_ignored(self):
        """Defensive: unrelated metadata types shouldn't crash send_typing."""
        adapter = _make_adapter()
        await adapter.send_typing("channel_1", metadata="not-a-dict")  # type: ignore[arg-type]
        adapter._api_post.assert_awaited_once_with(
            "users/bot_xyz/typing",
            {"channel_id": "channel_1"},
        )
