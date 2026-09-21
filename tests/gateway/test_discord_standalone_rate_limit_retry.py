"""Tests for the inline 429 retry-after handling in Discord's standalone send.

Covers the fix for the 2026-07-08 incident where moss-fix-pipeline's delivery
was dropped after 3 tick-spaced retries (~2min apart) even though Discord's own
retry_after was under half a second each time. _standalone_send now retries
once inline, honouring Discord's retry_after (capped), before falling back to
the slower cron-level retry queue.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.platforms.discord import adapter as discord_adapter


def _make_pconfig():
    return SimpleNamespace(token="fake-token")


def _make_resp(status):
    resp = AsyncMock()
    resp.status = status
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_session(responses):
    session = AsyncMock()
    session.post = MagicMock(side_effect=responses)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


class TestStandaloneSendRateLimitRetry:
    @pytest.mark.asyncio
    async def test_retries_once_after_429_and_succeeds(self):
        session = _make_session([_make_resp(429), _make_resp(200)])

        with patch("aiohttp.ClientSession", return_value=session), \
             patch.object(
                 discord_adapter, "_standalone_read_text_limited",
                 AsyncMock(return_value='{"message": "You are being rate limited.", "retry_after": 0.5, "global": false}'),
             ), \
             patch.object(
                 discord_adapter, "_standalone_read_json_limited",
                 AsyncMock(return_value={"id": "msg123"}),
             ), \
             patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            result = await discord_adapter._standalone_send(
                _make_pconfig(), chat_id="111", message="hello", thread_id="222",
            )

        assert result.get("success") is True
        assert session.post.call_count == 2
        mock_sleep.assert_awaited_once()
        assert mock_sleep.call_args[0][0] == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.asyncio
    async def test_gives_up_after_second_429(self):
        session = _make_session([_make_resp(429), _make_resp(429)])

        with patch("aiohttp.ClientSession", return_value=session), \
             patch.object(
                 discord_adapter, "_standalone_read_text_limited",
                 AsyncMock(return_value='{"retry_after": 0.02}'),
             ), \
             patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            result = await discord_adapter._standalone_send(
                _make_pconfig(), chat_id="111", message="hello", thread_id="222",
            )

        assert "error" in result
        assert "429" in result["error"]
        assert session.post.call_count == 2
        mock_sleep.assert_awaited_once()
        # retry_after below the 0.1s floor is clamped up, not honoured verbatim
        assert mock_sleep.call_args[0][0] == pytest.approx(0.1, abs=1e-6)

    @pytest.mark.asyncio
    async def test_unparseable_429_body_falls_back_to_default_delay(self):
        session = _make_session([_make_resp(429), _make_resp(200)])

        with patch("aiohttp.ClientSession", return_value=session), \
             patch.object(
                 discord_adapter, "_standalone_read_text_limited",
                 AsyncMock(return_value="not json"),
             ), \
             patch.object(
                 discord_adapter, "_standalone_read_json_limited",
                 AsyncMock(return_value={"id": "msg123"}),
             ), \
             patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            result = await discord_adapter._standalone_send(
                _make_pconfig(), chat_id="111", message="hello", thread_id="222",
            )

        assert result.get("success") is True
        assert mock_sleep.call_args[0][0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_retry_after_is_capped(self):
        session = _make_session([_make_resp(429), _make_resp(200)])

        with patch("aiohttp.ClientSession", return_value=session), \
             patch.object(
                 discord_adapter, "_standalone_read_text_limited",
                 AsyncMock(return_value='{"retry_after": 999}'),
             ), \
             patch.object(
                 discord_adapter, "_standalone_read_json_limited",
                 AsyncMock(return_value={"id": "msg123"}),
             ), \
             patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            result = await discord_adapter._standalone_send(
                _make_pconfig(), chat_id="111", message="hello", thread_id="222",
            )

        assert result.get("success") is True
        assert mock_sleep.call_args[0][0] == pytest.approx(
            discord_adapter._DISCORD_STANDALONE_MAX_INLINE_RETRY_SECONDS, abs=1e-6
        )

    @pytest.mark.asyncio
    async def test_non_429_error_returns_immediately_without_retry(self):
        session = _make_session([_make_resp(500)])

        with patch("aiohttp.ClientSession", return_value=session), \
             patch.object(
                 discord_adapter, "_standalone_read_text_limited",
                 AsyncMock(return_value="internal server error"),
             ), \
             patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            result = await discord_adapter._standalone_send(
                _make_pconfig(), chat_id="111", message="hello", thread_id="222",
            )

        assert "error" in result
        assert session.post.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_forum_thread_creation_retries_once_after_429(self):
        """The no-media forum-thread-creation POST shares the same 429 retry
        as the direct-message send (2026-07-08 review finding: this path was
        initially missed since it isn't reached via thread_id)."""
        session = _make_session([_make_resp(429), _make_resp(200)])

        with patch("aiohttp.ClientSession", return_value=session), \
             patch.object(discord_adapter, "_probe_is_forum_cached", return_value=True), \
             patch.object(
                 discord_adapter, "_standalone_read_text_limited",
                 AsyncMock(return_value='{"retry_after": 0.3}'),
             ), \
             patch.object(
                 discord_adapter, "_standalone_read_json_limited",
                 AsyncMock(return_value={"id": "thread123", "message": {"id": "msg123"}}),
             ), \
             patch.object(
                 discord_adapter, "_blog_approval_components_for_message", return_value=None,
             ), \
             patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            result = await discord_adapter._standalone_send(
                _make_pconfig(), chat_id="111", message="hello", thread_id=None,
            )

        assert result.get("success") is True
        assert session.post.call_count == 2
        mock_sleep.assert_awaited_once()
        assert mock_sleep.call_args[0][0] == pytest.approx(0.3, abs=1e-6)
