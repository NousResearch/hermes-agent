"""Tests for WeChat iLink typing ticket refresh logic (issue #38085)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def weixin_adapter():
    """Create a minimal WeixinAdapter with mocked internals for typing tests."""
    from gateway.platforms.weixin import WeixinAdapter, TypingTicketCache

    config = MagicMock()
    config.extra = {"account_id": "test-account"}
    config.name = "weixin"

    with patch.object(WeixinAdapter, "__init__", lambda self, cfg: None):
        adapter = WeixinAdapter.__new__(WeixinAdapter)
        adapter._send_session = AsyncMock()
        adapter._token = "test-token"
        adapter._base_url = "https://ilinkai.weixin.qq.com"
        adapter._account_id = "test-account"
        adapter._typing_cache = TypingTicketCache(ttl_seconds=600.0)
        adapter._token_store = MagicMock()
        adapter._token_store.get.return_value = None  # no stored context_token
        adapter.platform = MagicMock()
        mock_value = MagicMock()
        mock_value.title.return_value = "Weixin"
        adapter.platform.value = mock_value

    return adapter


class TestEnsureTypingTicket:
    """Tests for _ensure_typing_ticket — the fix for stuck typing indicator."""

    @pytest.mark.asyncio
    async def test_returns_cached_ticket_when_fresh(self, weixin_adapter):
        """If the cached ticket is still valid, return it without refreshing."""
        weixin_adapter._typing_cache.set("user-123", "cached-ticket-abc")
        ticket = await weixin_adapter._ensure_typing_ticket("user-123")
        assert ticket == "cached-ticket-abc"

    @pytest.mark.asyncio
    async def test_refreshes_when_ticket_expired(self, weixin_adapter):
        """When the cached ticket has expired, fetch a new one via getConfig."""
        # Insert an expired ticket directly (bypass TTL check)
        weixin_adapter._typing_cache._cache["user-123"] = (
            "old-ticket",
            time.time() - 601,  # expired (TTL is 600s)
        )

        mock_response = {"typing_ticket": "fresh-ticket-xyz"}
        with patch("gateway.platforms.weixin._get_config", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            ticket = await weixin_adapter._ensure_typing_ticket("user-123")

        assert ticket == "fresh-ticket-xyz"
        mock_get.assert_called_once_with(
            weixin_adapter._send_session,
            base_url=weixin_adapter._base_url,
            token=weixin_adapter._token,
            user_id="user-123",
            context_token=None,
        )

    @pytest.mark.asyncio
    async def test_refreshes_when_no_cached_ticket(self, weixin_adapter):
        """When there is no cached ticket at all, fetch a new one."""
        mock_response = {"typing_ticket": "new-ticket"}
        with patch("gateway.platforms.weixin._get_config", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            ticket = await weixin_adapter._ensure_typing_ticket("user-456")

        assert ticket == "new-ticket"

    @pytest.mark.asyncio
    async def test_uses_stored_context_token_when_available(self, weixin_adapter):
        """Pass the stored context_token to getConfig when available."""
        weixin_adapter._token_store.get.return_value = "stored-ctx-token"

        mock_response = {"typing_ticket": "ticket-with-ctx"}
        with patch("gateway.platforms.weixin._get_config", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            ticket = await weixin_adapter._ensure_typing_ticket("user-789")

        assert ticket == "ticket-with-ctx"
        mock_get.assert_called_once_with(
            weixin_adapter._send_session,
            base_url=weixin_adapter._base_url,
            token=weixin_adapter._token,
            user_id="user-789",
            context_token="stored-ctx-token",
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_no_session(self, weixin_adapter):
        """Return None when there is no send session."""
        weixin_adapter._send_session = None
        ticket = await weixin_adapter._ensure_typing_ticket("user-123")
        assert ticket is None

    @pytest.mark.asyncio
    async def test_returns_none_when_getconfig_fails(self, weixin_adapter):
        """Return None when getConfig raises an exception."""
        with patch("gateway.platforms.weixin._get_config", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("network error")
            ticket = await weixin_adapter._ensure_typing_ticket("user-123")

        assert ticket is None

    @pytest.mark.asyncio
    async def test_returns_none_when_getconfig_returns_empty_ticket(self, weixin_adapter):
        """Return None when getConfig returns no typing_ticket."""
        with patch("gateway.platforms.weixin._get_config", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"typing_ticket": ""}
            ticket = await weixin_adapter._ensure_typing_ticket("user-123")

        assert ticket is None

    @pytest.mark.asyncio
    async def test_stop_typing_refreshes_ticket(self, weixin_adapter):
        """stop_typing should refresh the ticket when expired, not silently no-op."""
        # Expired ticket
        weixin_adapter._typing_cache._cache["user-123"] = (
            "old-ticket",
            time.time() - 601,
        )

        mock_response = {"typing_ticket": "refreshed-ticket"}
        with patch("gateway.platforms.weixin._get_config", new_callable=AsyncMock) as mock_get, \
             patch("gateway.platforms.weixin._send_typing", new_callable=AsyncMock) as mock_send:
            mock_get.return_value = mock_response
            await weixin_adapter.stop_typing("user-123")

        # _send_typing should have been called with TYPING_STOP=2
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert call_kwargs.kwargs["typing_ticket"] == "refreshed-ticket"
        assert call_kwargs.kwargs["status"] == 2  # TYPING_STOP

    @pytest.mark.asyncio
    async def test_send_typing_refreshes_ticket(self, weixin_adapter):
        """send_typing should refresh the ticket when expired."""
        # Expired ticket
        weixin_adapter._typing_cache._cache["user-123"] = (
            "old-ticket",
            time.time() - 601,
        )

        mock_response = {"typing_ticket": "refreshed-ticket"}
        with patch("gateway.platforms.weixin._get_config", new_callable=AsyncMock) as mock_get, \
             patch("gateway.platforms.weixin._send_typing", new_callable=AsyncMock) as mock_send:
            mock_get.return_value = mock_response
            await weixin_adapter.send_typing("user-123")

        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert call_kwargs.kwargs["typing_ticket"] == "refreshed-ticket"
        assert call_kwargs.kwargs["status"] == 1  # TYPING_START


class TestTypingTicketCache:
    """Tests for the TypingTicketCache TTL logic."""

    def test_returns_ticket_when_fresh(self):
        from gateway.platforms.weixin import TypingTicketCache
        cache = TypingTicketCache(ttl_seconds=600.0)
        cache.set("user-1", "ticket-1")
        assert cache.get("user-1") == "ticket-1"

    def test_returns_none_when_expired(self):
        from gateway.platforms.weixin import TypingTicketCache
        cache = TypingTicketCache(ttl_seconds=600.0)
        cache._cache["user-1"] = ("ticket-1", time.time() - 601)
        assert cache.get("user-1") is None

    def test_returns_none_when_missing(self):
        from gateway.platforms.weixin import TypingTicketCache
        cache = TypingTicketCache(ttl_seconds=600.0)
        assert cache.get("nonexistent") is None

    def test_expired_entry_is_removed_from_cache(self):
        from gateway.platforms.weixin import TypingTicketCache
        cache = TypingTicketCache(ttl_seconds=600.0)
        cache._cache["user-1"] = ("ticket-1", time.time() - 601)
        cache.get("user-1")
        assert "user-1" not in cache._cache
