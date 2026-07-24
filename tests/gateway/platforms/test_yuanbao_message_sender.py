"""
test_yuanbao_message_sender.py - Unit tests for MessageSender._build_msg_body_with_mentions

Tests the @-mention resolution logic, including automatic cache refresh
when the member cache is empty or stale.
"""

import sys
import os
import time
import json

# Ensure hermes-agent root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from unittest.mock import MagicMock, AsyncMock
from gateway.platforms.yuanbao import MessageSender


# Sample member list used in test scenarios
_SAMPLE_MEMBERS = [
    {"user_id": "user_abc123", "nickname": "元宝"},
    {"user_id": "user_def456", "nickname": "张三"},
    {"user_id": "user_ghi789", "nickname": "Alice"},
]


def _make_mock_adapter(member_cache: dict = None):
    """Create a mock YuanbaoAdapter with controllable _member_cache and _group_query."""
    adapter = MagicMock()
    adapter._member_cache = member_cache or {}
    adapter.MEMBER_CACHE_TTL_S = 300.0

    # _group_query.get_group_member_list_raw is an async method
    mock_group_query = MagicMock()
    mock_group_query.get_group_member_list_raw = AsyncMock()
    adapter._group_query = mock_group_query

    return adapter


def _make_sender(adapter):
    """Create a MessageSender with a given mock adapter."""
    return MessageSender(adapter)


class TestBuildMsgBodyWithMentions:
    """Tests for _build_msg_body_with_mentions cache + @-mention resolution."""

    @pytest.mark.asyncio
    async def test_cache_hit_resolves_mention(self):
        """Cache is fresh and populated → @-mention resolved to TIMCustomElem."""
        cache = {
            "group_001": (time.time(), _SAMPLE_MEMBERS),
        }
        adapter = _make_mock_adapter(cache)
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "@元宝 你好", "group_001"
        )

        # Should produce 2 elements: TIMCustomElem for @元宝 + TIMTextElem for " 你好"
        assert len(result) == 2
        assert result[0]["msg_type"] == "TIMCustomElem"
        assert result[0]["msg_content"]["data"]
        data = json.loads(result[0]["msg_content"]["data"])
        assert data["elem_type"] == 1002
        assert data["user_id"] == "user_abc123"
        assert data["text"] == "@元宝"
        assert result[1]["msg_type"] == "TIMTextElem"
        assert "你好" in result[1]["msg_content"]["text"]

        # Cache was fresh → no network call
        adapter._group_query.get_group_member_list_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_unknown_nickname_falls_back(self):
        """Cache hit but nickname not found → @mention sent as plain text."""
        cache = {
            "group_001": (time.time(), _SAMPLE_MEMBERS),
        }
        adapter = _make_mock_adapter(cache)
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "@不存在的昵称 测试", "group_001"
        )

        # Unknown nickname → split into TIMTextElem parts (no TIMCustomElem)
        assert len(result) == 2
        for elem in result:
            assert elem["msg_type"] == "TIMTextElem"
        assert "@不存在的昵称" in result[0]["msg_content"]["text"]
        assert "测试" in result[1]["msg_content"]["text"]

        adapter._group_query.get_group_member_list_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_empty_triggers_refresh_and_succeeds(self):
        """Cache empty → refresh from server → cache populated → @ resolved."""
        adapter = _make_mock_adapter({})  # empty cache
        # Mock the refresh to populate the cache
        async def _refresh_and_populate(group_code, offset=0, limit=200):
            adapter._member_cache[group_code] = (time.time(), _SAMPLE_MEMBERS)
            return {"members": _SAMPLE_MEMBERS}
        adapter._group_query.get_group_member_list_raw.side_effect = _refresh_and_populate
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "@元宝 你好", "group_001"
        )

        # Should have triggered a refresh
        adapter._group_query.get_group_member_list_raw.assert_awaited_once_with("group_001")

        # @-mention should be resolved
        assert len(result) == 2
        assert result[0]["msg_type"] == "TIMCustomElem"
        data = json.loads(result[0]["msg_content"]["data"])
        assert data["user_id"] == "user_abc123"

    @pytest.mark.asyncio
    async def test_cache_expired_triggers_refresh_and_succeeds(self):
        """Cache expired → refresh from server → cache updated → @ resolved."""
        old_time = time.time() - 600  # 10 minutes ago (past 300s TTL)
        cache = {
            "group_001": (old_time, _SAMPLE_MEMBERS),
        }
        adapter = _make_mock_adapter(cache)

        async def _refresh_and_update(group_code, offset=0, limit=200):
            fresh_members = [
                {"user_id": "user_abc123", "nickname": "元宝"},
            ]
            adapter._member_cache[group_code] = (time.time(), fresh_members)
            return {"members": fresh_members}
        adapter._group_query.get_group_member_list_raw.side_effect = _refresh_and_update
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "@元宝 你好", "group_001"
        )

        adapter._group_query.get_group_member_list_raw.assert_awaited_once_with("group_001")
        assert len(result) == 2
        assert result[0]["msg_type"] == "TIMCustomElem"
        data = json.loads(result[0]["msg_content"]["data"])
        assert data["user_id"] == "user_abc123"

    @pytest.mark.asyncio
    async def test_cache_empty_refresh_fails_falls_back(self):
        """Cache empty → refresh fails (returns None) → fall back to plain text."""
        adapter = _make_mock_adapter({})
        adapter._group_query.get_group_member_list_raw.return_value = None
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "@元宝 你好", "group_001"
        )

        adapter._group_query.get_group_member_list_raw.assert_awaited_once_with("group_001")
        # Fall back to plain text
        assert len(result) == 1
        assert result[0]["msg_type"] == "TIMTextElem"
        assert "@元宝 你好" in result[0]["msg_content"]["text"]

    @pytest.mark.asyncio
    async def test_cache_empty_refresh_returns_no_members_falls_back(self):
        """Cache empty → refresh returns empty member list → fall back to plain text."""
        adapter = _make_mock_adapter({})
        async def _refresh_empty(group_code, offset=0, limit=200):
            adapter._member_cache[group_code] = (time.time(), [])
            return {"members": []}
        adapter._group_query.get_group_member_list_raw.side_effect = _refresh_empty
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "@元宝 你好", "group_001"
        )

        adapter._group_query.get_group_member_list_raw.assert_awaited_once_with("group_001")
        assert len(result) == 1
        assert result[0]["msg_type"] == "TIMTextElem"

    @pytest.mark.asyncio
    async def test_no_at_symbol_skips_entire_logic(self):
        """No @ in text → entire method skipped, returns plain text."""
        adapter = _make_mock_adapter({})  # empty cache, but shouldn't matter
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "你好，世界", "group_001"
        )

        assert len(result) == 1
        assert result[0]["msg_type"] == "TIMTextElem"
        assert result[0]["msg_content"]["text"] == "你好，世界"
        # No network call
        adapter._group_query.get_group_member_list_raw.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_mentions_in_one_message(self):
        """Multiple @-mentions in one message all resolved from cache."""
        cache = {
            "group_001": (time.time(), _SAMPLE_MEMBERS),
        }
        adapter = _make_mock_adapter(cache)
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "@元宝 和 @张三 一起", "group_001"
        )

        # 元宝 → TIMCustomElem, " 和 " → TIMTextElem, 张三 → TIMCustomElem, " 一起" → TIMTextElem
        assert len(result) == 4
        assert result[0]["msg_type"] == "TIMCustomElem"
        assert json.loads(result[0]["msg_content"]["data"])["user_id"] == "user_abc123"
        assert result[1]["msg_type"] == "TIMTextElem"
        assert result[2]["msg_type"] == "TIMCustomElem"
        assert json.loads(result[2]["msg_content"]["data"])["user_id"] == "user_def456"
        assert result[3]["msg_type"] == "TIMTextElem"

    @pytest.mark.asyncio
    async def test_cold_cache_email_only_does_not_trigger_refresh(self):
        """Cold cache + literal '@' in an email address must NOT trigger a refresh.

        Regression for tek/sweeper review on PR #60324: the previous fast path
        used `if "@" not in text`, which matched any literal '@' — including
        emails like `user@example.com`. That forced an unnecessary
        `get_group_member_list_raw` await before delivery, even though no real
        mention could ever be produced. The fix gates refresh on the actual
        mention regex `_AT_USER_RE`.
        """
        adapter = _make_mock_adapter({})  # cold cache
        sender = _make_sender(adapter)

        result = await sender._build_msg_body_with_mentions(
            "请联系 user@example.com 获取资料", "group_001"
        )

        # Email '@' is not mention syntax → no refresh, no member lookup
        adapter._group_query.get_group_member_list_raw.assert_not_called()

        # Text returned as plain TIMTextElem, untouched
        assert len(result) == 1
        assert result[0]["msg_type"] == "TIMTextElem"
        assert result[0]["msg_content"]["text"] == "请联系 user@example.com 获取资料"

    @pytest.mark.asyncio
    async def test_send_group_message_propagates_async_builder(self):
        """send_group_message must await _build_msg_body_with_mentions correctly.

        Regression for tek/sweeper review on PR #60324: the sync→async builder
        boundary added by this PR requires a top-level test. We assert that
        send_group_message ends up calling send_group_msg_body with the builder's
        output, not a coroutine object.
        """
        cache = {
            "group_001": (time.time(), _SAMPLE_MEMBERS),
        }
        adapter = _make_mock_adapter(cache)
        sender = _make_sender(adapter)

        # Spy on send_group_msg_body, return a sentinel so we can assert it was called
        async def _spy_send_group_msg_body(group_code, msg_body, reply_to=None):
            return {"_spy_called_with_group": group_code, "_spy_msg_body": msg_body}

        sender.send_group_msg_body = _spy_send_group_msg_body

        result = await sender.send_group_message(
            "group_001", "@元宝 你好"
        )

        # The async builder's TIMCustomElem output should have been passed through
        assert result["_spy_called_with_group"] == "group_001"
        body = result["_spy_msg_body"]
        assert len(body) == 2
        assert body[0]["msg_type"] == "TIMCustomElem"
        assert json.loads(body[0]["msg_content"]["data"])["user_id"] == "user_abc123"
        assert body[1]["msg_type"] == "TIMTextElem"

        # Cache was fresh → no refresh side-effect
        adapter._group_query.get_group_member_list_raw.assert_not_called()
