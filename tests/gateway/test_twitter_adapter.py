"""
Tests for the Twitter/X platform adapter.

Covers:
1. Tweet text truncation to 280 chars
2. Thread chaining for long responses
3. Self-message filtering
4. Media upload (mock)
5. Rate limit backoff
6. OAuth token refresh
7. Stream reconnection logic
8. DM handling
9. build_source for Twitter messages
10. Sanitized logging (response bodies truncated)
11. Refresh token persistence
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Mock httpx before importing the adapter (httpx may not be installed)
# ---------------------------------------------------------------------------

def _ensure_httpx_mock():
    if "httpx" in sys.modules and hasattr(sys.modules["httpx"], "__file__"):
        return
    httpx_mod = MagicMock()
    # Create a proper mock for AsyncClient that supports is_closed attribute
    _mock_client_instance = MagicMock()
    _mock_client_instance.is_closed = False
    httpx_mod.AsyncClient = MagicMock(return_value=_mock_client_instance)
    httpx_mod.Timeout = MagicMock
    httpx_mod.HTTPError = Exception
    httpx_mod.HTTPStatusError = Exception
    sys.modules.setdefault("httpx", httpx_mod)

_ensure_httpx_mock()

# ---------------------------------------------------------------------------
# Imports after mocking
# ---------------------------------------------------------------------------

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.twitter import (
    TwitterAdapter,
    _truncate_for_log,
    _generate_code_verifier,
    _generate_code_challenge,
    _load_persisted_tokens,
    _save_persisted_tokens,
    RateLimitState,
    MAX_TWEET_LENGTH,
    MAX_DM_LENGTH,
    _LOG_BODY_LIMIT,
    check_twitter_requirements,
    TWITTER_API_BASE,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_adapter(**env_overrides) -> TwitterAdapter:
    """Create a TwitterAdapter with mock tokens for testing."""
    config = PlatformConfig(enabled=True, token="fake-token")
    with patch.dict(os.environ, {
        "TWITTER_ACCESS_TOKEN": "fake-access-token",
        "TWITTER_REFRESH_TOKEN": "fake-refresh-token",
        "TWITTER_CLIENT_ID": "fake-client-id",
        "TWITTER_CLIENT_SECRET": "fake-client-secret",
        **env_overrides,
    }, clear=False):
        adapter = TwitterAdapter(config)
    return adapter


def _mock_http_client(adapter: TwitterAdapter, responses: dict = None):
    """Replace adapter's HTTP client with a mock that returns canned responses.

    *responses* maps (method, url) tuples to (status_code, json_body) tuples.
    """
    if responses is None:
        responses = {}

    client = MagicMock()
    client.is_closed = False

    async def _mock_request(method, url, **kwargs):
        key = (method.upper(), url)
        status, body = responses.get(key, (200, {}))
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = body
        resp.text = json.dumps(body) if isinstance(body, dict) else str(body)
        resp.headers = {}
        return resp

    client.request = AsyncMock(side_effect=_mock_request)
    client.post = AsyncMock(side_effect=lambda url, **kw: _mock_request_response(responses, "POST", url, **kw))
    client.get = AsyncMock(side_effect=lambda url, **kw: _mock_request_response(responses, "GET", url, **kw))
    client.stream = MagicMock()
    client.aclose = AsyncMock()

    adapter._client = client
    return client


def _mock_request_response(responses, method, url, **kwargs):
    key = (method.upper(), url)
    status, body = responses.get(key, (200, {}))
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body) if isinstance(body, dict) else str(body)
    resp.headers = {}
    return resp


# ===========================================================================
# 1. Truncated logging (MEDIUM finding)
# ===========================================================================

class TestTruncatedLogging:
    """Verify that _truncate_for_log caps response bodies at 200 chars."""

    def test_short_string_unchanged(self):
        text = "short error message"
        assert _truncate_for_log(text) == text

    def test_long_string_truncated(self):
        text = "A" * 500
        result = _truncate_for_log(text)
        assert len(result) == _LOG_BODY_LIMIT + len("...[truncated]")
        assert result.endswith("...[truncated]")

    def test_exact_limit_unchanged(self):
        text = "B" * _LOG_BODY_LIMIT
        assert _truncate_for_log(text) == text

    def test_empty_string(self):
        assert _truncate_for_log("") == ""

    def test_none_input(self):
        assert _truncate_for_log(None) == ""

    def test_custom_limit(self):
        text = "X" * 100
        result = _truncate_for_log(text, limit=50)
        assert len(result) == 50 + len("...[truncated]")

    def test_non_string_input(self):
        result = _truncate_for_log(12345)
        assert isinstance(result, str)

    def test_preserves_sensitive_data_truncation(self):
        """Large API response bodies containing tokens/PII should be cut."""
        fake_body = json.dumps({
            "error": "rate limit exceeded",
            "detail": "sensitive_user_data_here" * 20,
            "access_token": "should_not_appear_in_full_in_logs",
        })
        result = _truncate_for_log(fake_body)
        assert len(result) <= _LOG_BODY_LIMIT + len("...[truncated]")
        # The full access token string should NOT appear
        assert "should_not_appear_in_full_in_logs" not in result


# ===========================================================================
# 2. Tweet text truncation to 280 chars
# ===========================================================================

class TestTweetTruncation:
    """Verify _split_for_tweets respects the 280-char limit."""

    def test_short_tweet_not_split(self):
        adapter = _make_adapter()
        text = "Hello, world!"
        chunks = adapter._split_for_tweets(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_limit_not_split(self):
        adapter = _make_adapter()
        text = "A" * MAX_TWEET_LENGTH
        chunks = adapter._split_for_tweets(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split_into_chunks(self):
        adapter = _make_adapter()
        text = "Word " * 200  # ~1000 chars
        chunks = adapter._split_for_tweets(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= MAX_TWEET_LENGTH

    def test_split_on_paragraph_boundary(self):
        adapter = _make_adapter()
        para = "Short sentence. " * 10  # ~180 chars
        text = para + "\n\n" + para + "\n\n" + para
        chunks = adapter._split_for_tweets(text)
        for chunk in chunks:
            assert len(chunk) <= MAX_TWEET_LENGTH

    def test_split_on_sentence_boundary(self):
        adapter = _make_adapter()
        sentence = "This is a sentence. "
        text = sentence * 30  # ~600 chars
        chunks = adapter._split_for_tweets(text)
        for chunk in chunks:
            assert len(chunk) <= MAX_TWEET_LENGTH

    def test_hard_cut_on_word_boundary(self):
        adapter = _make_adapter()
        # Long text without sentence/paragraph breaks
        text = "word " * 200
        chunks = adapter._split_for_tweets(text)
        for chunk in chunks:
            assert len(chunk) <= MAX_TWEET_LENGTH
        # All text should be preserved
        rejoined = "".join(chunks)
        assert len(rejoined) >= len(text.replace(" ", ""))

    def test_empty_text(self):
        adapter = _make_adapter()
        assert adapter._split_for_tweets("") == []

    def test_unicode_tweet(self):
        adapter = _make_adapter()
        # CJK characters — each is 1 char but may need careful handling
        text = "日本語のテキストです。" * 50
        chunks = adapter._split_for_tweets(text)
        for chunk in chunks:
            assert len(chunk) <= MAX_TWEET_LENGTH

    def test_all_chunks_within_limit(self):
        """Property: every chunk must be <= 280 chars."""
        adapter = _make_adapter()
        test_cases = [
            "A" * 300,
            "Hello! " * 100,
            "x" * 1000,
            "🎉 " * 200,
        ]
        for text in test_cases:
            chunks = adapter._split_for_tweets(text)
            for i, chunk in enumerate(chunks):
                assert len(chunk) <= MAX_TWEET_LENGTH, (
                    f"Chunk {i} exceeds limit: {len(chunk)} > {MAX_TWEET_LENGTH}"
                )


# ===========================================================================
# 3. Thread chaining
# ===========================================================================

class TestThreadChaining:
    """Verify that long responses produce threaded tweets."""

    @pytest.mark.asyncio
    async def test_send_creates_thread_for_long_response(self):
        adapter = _make_adapter()

        # Mock the API to return tweet IDs
        call_count = 0
        tweet_ids = ["tweet_1", "tweet_2", "tweet_3", "tweet_4", "tweet_5", "tweet_6"]

        async def mock_api_request(method, url, **kwargs):
            nonlocal call_count
            if method == "POST" and "/tweets" in url:
                body = kwargs.get("json_body", {})
                # Verify threading: reply field should reference previous tweet
                if call_count > 0:
                    reply_to = body.get("reply", {}).get("in_reply_to_tweet_id")
                    assert reply_to == tweet_ids[call_count - 1], (
                        f"Tweet {call_count} should reply to {tweet_ids[call_count - 1]}, "
                        f"got {reply_to}"
                    )
                idx = call_count
                call_count += 1
                tid = tweet_ids[idx] if idx < len(tweet_ids) else f"tweet_{idx+1}"
                return 200, {"data": {"id": tid}}
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api_request)

        # Long text that will be split into multiple tweets
        long_text = "This is a test sentence. " * 50  # ~1250 chars → ~5 tweets
        result = await adapter.send("ignored", long_text)

        assert result.success
        assert call_count > 1  # Multiple tweets were posted

    @pytest.mark.asyncio
    async def test_send_single_tweet_no_reply_field(self):
        adapter = _make_adapter()

        async def mock_api_request(method, url, **kwargs):
            if method == "POST" and "/tweets" in url:
                body = kwargs.get("json_body", {})
                assert "reply" not in body  # No reply field for single tweet
                return 200, {"data": {"id": "single_tweet"}}
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api_request)

        result = await adapter.send("ignored", "Short tweet")
        assert result.success

    @pytest.mark.asyncio
    async def test_send_respects_reply_to_message_id(self):
        adapter = _make_adapter()

        async def mock_api_request(method, url, **kwargs):
            if method == "POST" and "/tweets" in url:
                body = kwargs.get("json_body", {})
                reply = body.get("reply", {})
                assert reply.get("in_reply_to_tweet_id") == "original_tweet"
                return 200, {"data": {"id": "reply_tweet"}}
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api_request)

        result = await adapter.send(
            "ignored", "A reply", reply_to_message_id="original_tweet"
        )
        assert result.success


# ===========================================================================
# 4. Self-message filtering
# ===========================================================================

class TestSelfMessageFiltering:
    """Verify that messages from the bot itself are filtered out."""

    @pytest.mark.asyncio
    async def test_self_dm_ignored(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot_user_123"
        adapter._message_handler = AsyncMock()

        event = {
            "data": {
                "id": "msg_1",
                "event_type": "MessageCreate",
                "sender_id": "bot_user_123",  # Same as bot
                "text": "This should be ignored",
                "dm_conversation_id": "conv_1",
                "created_at": "2025-01-01T00:00:00Z",
            },
            "includes": {"users": [{"id": "bot_user_123", "username": "mybot"}]},
        }

        await adapter._process_dm_event(event)
        adapter._message_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_user_dm_processed(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot_user_123"
        received_events = []

        async def capture_handler(event):
            received_events.append(event)

        adapter._message_handler = capture_handler

        # Mock handle_message to call the handler directly (bypass base class internals)
        async def mock_handle_message(event):
            if adapter._message_handler:
                await adapter._message_handler(event)
        adapter.handle_message = mock_handle_message

        event = {
            "data": {
                "id": "msg_2",
                "event_type": "MessageCreate",
                "sender_id": "other_user_456",  # Different user
                "text": "Hello bot!",
                "dm_conversation_id": "conv_2",
                "created_at": "2025-01-01T00:00:00Z",
            },
            "includes": {"users": [{"id": "other_user_456", "username": "alice"}]},
        }

        await adapter._process_dm_event(event)
        assert len(received_events) == 1
        received = received_events[0]
        assert received.text == "Hello bot!"

    @pytest.mark.asyncio
    async def test_no_bot_id_allows_all(self):
        """When bot identity is unknown, no messages should be filtered."""
        adapter = _make_adapter()
        adapter._bot_user_id = None
        received_events = []

        async def capture_handler(event):
            received_events.append(event)

        adapter._message_handler = capture_handler

        async def mock_handle_message(event):
            if adapter._message_handler:
                await adapter._message_handler(event)
        adapter.handle_message = mock_handle_message

        event = {
            "data": {
                "id": "msg_3",
                "event_type": "MessageCreate",
                "sender_id": "anyone",
                "text": "Hello!",
                "dm_conversation_id": "conv_3",
                "created_at": "2025-01-01T00:00:00Z",
            },
            "includes": {"users": [{"id": "anyone", "username": "anyone"}]},
        }

        await adapter._process_dm_event(event)
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_duplicate_message_filtered(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot_1"
        received_events = []

        async def capture_handler(event):
            received_events.append(event)

        adapter._message_handler = capture_handler

        async def mock_handle_message(event):
            if adapter._message_handler:
                await adapter._message_handler(event)
        adapter.handle_message = mock_handle_message

        event = {
            "data": {
                "id": "dup_msg",
                "event_type": "MessageCreate",
                "sender_id": "user_1",
                "text": "Hello!",
                "dm_conversation_id": "conv_1",
                "created_at": "2025-01-01T00:00:00Z",
            },
            "includes": {"users": [{"id": "user_1", "username": "bob"}]},
        }

        await adapter._process_dm_event(event)
        assert len(received_events) == 1

        # Same event again — should be deduplicated
        await adapter._process_dm_event(event)
        assert len(received_events) == 1  # Still just 1


# ===========================================================================
# 5. Media upload (mock)
# ===========================================================================

class TestMediaUpload:
    """Verify media upload flow (INIT → APPEND → FINALIZE)."""

    @pytest.mark.asyncio
    async def test_upload_media_success(self):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        init_resp = MagicMock()
        init_resp.status_code = 200
        init_resp.json.return_value = {"media_id_string": "media_123"}
        init_resp.text = '{"media_id_string": "media_123"}'

        append_resp = MagicMock()
        append_resp.status_code = 200
        append_resp.text = "{}"

        finalize_resp = MagicMock()
        finalize_resp.status_code = 200
        finalize_resp.text = '{"processing_info": null}'

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(side_effect=[init_resp, append_resp, finalize_resp])
        adapter._client = client

        media_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # Fake JPEG
        result = await adapter.upload_media(media_bytes, "image/jpeg")

        assert result == "media_123"
        assert client.post.call_count == 3  # INIT + APPEND + FINALIZE

    @pytest.mark.asyncio
    async def test_upload_media_init_failure(self):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        init_resp = MagicMock()
        init_resp.status_code = 400
        init_resp.text = '{"error": "bad request"}'

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=init_resp)
        adapter._client = client

        result = await adapter.upload_media(b"data", "image/jpeg")
        assert result is None

    @pytest.mark.asyncio
    async def test_upload_media_chunked(self):
        """Large files should be split into 5MB chunks."""
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        init_resp = MagicMock()
        init_resp.status_code = 200
        init_resp.json.return_value = {"media_id_string": "big_media"}

        append_resp = MagicMock()
        append_resp.status_code = 200

        finalize_resp = MagicMock()
        finalize_resp.status_code = 200

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(side_effect=[init_resp, append_resp, append_resp, finalize_resp])
        adapter._client = client

        # 10 MB of data → 2 chunks + init + finalize = 4 calls
        large_data = b"\x00" * (10 * 1024 * 1024)
        result = await adapter.upload_media(large_data, "video/mp4")

        assert result == "big_media"
        assert client.post.call_count == 4


# ===========================================================================
# 6. Rate limit backoff
# ===========================================================================

class TestRateLimitBackoff:
    """Verify rate-limit backoff behavior."""

    def test_rate_limit_state_initial(self):
        state = RateLimitState()
        assert not state.should_backoff()
        assert state.attempts == 0

    def test_record_429_increments_attempts(self):
        state = RateLimitState()
        wait = state.record_429()
        assert state.attempts == 1
        assert wait > 0

    def test_exponential_backoff(self):
        state = RateLimitState()
        waits = []
        for _ in range(3):
            wait = state.record_429()
            waits.append(wait)
        # Each wait should be roughly double the previous
        assert waits[1] > waits[0]
        assert waits[2] > waits[1]

    def test_max_retries_gives_up(self):
        state = RateLimitState()
        for _ in range(5):
            state.record_429()
        result = state.record_429()  # 6th attempt
        assert result == -1  # Signal to give up

    def test_retry_after_respected(self):
        state = RateLimitState()
        wait = state.record_429(retry_after=120.0)
        assert wait == 120.0

    def test_reset_clears_state(self):
        state = RateLimitState()
        state.record_429()
        state.record_429()
        state.reset()
        assert state.attempts == 0
        assert not state.should_backoff()

    @pytest.mark.asyncio
    async def test_api_request_handles_429(self):
        adapter = _make_adapter()

        call_count = 0
        responses_429_then_200 = [
            (429, {"error": "rate limit"}),
            (429, {"error": "rate limit"}),
            (200, {"data": {"id": "ok"}}),
        ]

        async def mock_request(method, url, **kwargs):
            nonlocal call_count
            status, body = responses_429_then_200[min(call_count, len(responses_429_then_200) - 1)]
            call_count += 1
            resp = MagicMock()
            resp.status_code = status
            resp.json.return_value = body
            resp.text = json.dumps(body)
            resp.headers = {"Retry-After": "0.01"}  # Very short for tests
            return resp

        client = MagicMock()
        client.is_closed = False
        client.request = AsyncMock(side_effect=mock_request)
        adapter._client = client
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        # Patch sleep to avoid real waiting
        with patch("gateway.platforms.twitter.asyncio.sleep", new_callable=AsyncMock):
            code, data = await adapter._api_request("GET", f"{TWITTER_API_BASE}/2/test")

        assert code == 200
        assert data == {"data": {"id": "ok"}}
        assert call_count == 3


# ===========================================================================
# 7. OAuth token refresh
# ===========================================================================

class TestOAuthTokenRefresh:
    """Verify token refresh and persistence."""

    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        adapter = _make_adapter()
        adapter._access_token = None
        adapter._token_expiry = 0

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 7200,
        }
        resp.text = json.dumps(resp.json.return_value)

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        result = await adapter._refresh_access_token()

        assert result is True
        assert adapter._access_token == "new_access_token"
        assert adapter._refresh_token == "new_refresh_token"
        assert adapter._token_expiry > time.time()

    @pytest.mark.asyncio
    async def test_refresh_token_no_rotation(self):
        """If the server returns the same refresh token, don't re-persist."""
        adapter = _make_adapter()
        adapter._access_token = None
        original_refresh = adapter._refresh_token

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": "new_access",
            "refresh_token": original_refresh,  # Same token
            "expires_in": 7200,
        }
        resp.text = json.dumps(resp.json.return_value)

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        with patch("gateway.platforms.twitter._save_persisted_tokens") as mock_save:
            result = await adapter._refresh_access_token()

        assert result is True
        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_token_failure(self):
        adapter = _make_adapter()
        original_token = adapter._access_token

        resp = MagicMock()
        resp.status_code = 401
        resp.text = '{"error": "invalid_grant"}'

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        result = await adapter._refresh_access_token()
        assert result is False
        # Access token should remain unchanged on failure
        assert adapter._access_token == original_token

    @pytest.mark.asyncio
    async def test_ensure_valid_token_refreshes_when_expired(self):
        adapter = _make_adapter()
        adapter._access_token = "old_token"
        adapter._token_expiry = time.time() - 100  # Expired

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": "refreshed_token",
            "expires_in": 7200,
        }
        resp.text = json.dumps(resp.json.return_value)

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        result = await adapter._ensure_valid_token()
        assert result is True
        assert adapter._access_token == "refreshed_token"


# ===========================================================================
# 8. Refresh token persistence (LOW finding)
# ===========================================================================

class TestTokenPersistence:
    """Verify refresh tokens are persisted and loaded correctly."""

    def test_save_and_load_tokens(self, tmp_path):
        token_path = tmp_path / "twitter_tokens.json"
        with patch("gateway.platforms.twitter._TOKEN_PERSIST_PATH", token_path):
            _save_persisted_tokens({"refresh_token": "my_refresh"})
            loaded = _load_persisted_tokens()
        assert loaded["refresh_token"] == "my_refresh"

    def test_load_missing_file_returns_empty(self, tmp_path):
        token_path = tmp_path / "nonexistent.json"
        with patch("gateway.platforms.twitter._TOKEN_PERSIST_PATH", token_path):
            loaded = _load_persisted_tokens()
        assert loaded == {}

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        token_path = tmp_path / "twitter_tokens.json"
        token_path.write_text("NOT JSON {{{", encoding="utf-8")
        with patch("gateway.platforms.twitter._TOKEN_PERSIST_PATH", token_path):
            loaded = _load_persisted_tokens()
        assert loaded == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        token_path = tmp_path / "subdir" / "deep" / "twitter_tokens.json"
        with patch("gateway.platforms.twitter._TOKEN_PERSIST_PATH", token_path):
            _save_persisted_tokens({"refresh_token": "x"})
        assert token_path.is_file()

    @pytest.mark.asyncio
    async def test_token_rotation_persists(self):
        """When refresh token rotates, the new token should be persisted."""
        adapter = _make_adapter()

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "access_token": "new_access",
            "refresh_token": "rotated_refresh_token",
            "expires_in": 7200,
        }
        resp.text = json.dumps(resp.json.return_value)

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        with patch("gateway.platforms.twitter._save_persisted_tokens") as mock_save:
            await adapter._refresh_access_token()
            mock_save.assert_called_once_with({"refresh_token": "rotated_refresh_token"})

    def test_load_initial_tokens_prefers_env(self):
        """Env var takes priority over persisted file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"refresh_token": "persisted_token"}, f)
            persist_path = Path(f.name)
        try:
            with patch("gateway.platforms.twitter._TOKEN_PERSIST_PATH", persist_path):
                config = PlatformConfig(enabled=True)
                with patch.dict(os.environ, {
                    "TWITTER_REFRESH_TOKEN": "env_token",
                    "TWITTER_ACCESS_TOKEN": "",
                }, clear=False):
                    adapter = TwitterAdapter(config)
            assert adapter._refresh_token == "env_token"
        finally:
            persist_path.unlink(missing_ok=True)

    def test_load_initial_tokens_falls_back_to_file(self):
        """When env var is not set, fall back to persisted file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"refresh_token": "file_token"}, f)
            persist_path = Path(f.name)
        try:
            with patch("gateway.platforms.twitter._TOKEN_PERSIST_PATH", persist_path):
                config = PlatformConfig(enabled=True)
                # Remove env vars
                env = {k: v for k, v in os.environ.items()
                       if k not in ("TWITTER_REFRESH_TOKEN", "TWITTER_ACCESS_TOKEN")}
                with patch.dict(os.environ, env, clear=True):
                    adapter = TwitterAdapter(config)
            assert adapter._refresh_token == "file_token"
        finally:
            persist_path.unlink(missing_ok=True)


# ===========================================================================
# 9. DM handling
# ===========================================================================

class TestDMHandling:
    """Verify DM event processing and source building."""

    @pytest.mark.asyncio
    async def test_dm_event_builds_correct_source(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot_123"
        received_events = []

        async def capture_handler(event):
            received_events.append(event)

        adapter._message_handler = capture_handler

        async def mock_handle_message(event):
            if adapter._message_handler:
                await adapter._message_handler(event)
        adapter.handle_message = mock_handle_message

        event = {
            "data": {
                "id": "dm_msg_1",
                "event_type": "MessageCreate",
                "sender_id": "user_456",
                "text": "Hey, can you help me?",
                "dm_conversation_id": "conv_789",
                "created_at": "2025-06-15T10:30:00Z",
            },
            "includes": {
                "users": [{"id": "user_456", "username": "johndoe"}],
            },
        }

        await adapter._process_dm_event(event)

        assert len(received_events) == 1
        msg = received_events[0]
        assert msg.text == "Hey, can you help me?"
        assert msg.message_type == MessageType.TEXT
        assert msg.message_id == "dm_msg_1"
        assert msg.source.user_id == "user_456"
        assert msg.source.user_name == "johndoe"
        assert msg.source.chat_type == "dm"

    @pytest.mark.asyncio
    async def test_dm_event_non_message_create_ignored(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot_123"
        adapter._message_handler = AsyncMock()

        event = {
            "data": {
                "id": "evt_1",
                "event_type": "ParticipantsJoin",
                "sender_id": "user_456",
            },
        }

        await adapter._process_dm_event(event)
        adapter._message_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dm_event_empty_text_ignored(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot_123"
        adapter._message_handler = AsyncMock()

        event = {
            "data": {
                "id": "dm_empty",
                "event_type": "MessageCreate",
                "sender_id": "user_456",
                "text": "",
                "dm_conversation_id": "conv_1",
            },
            "includes": {"users": [{"id": "user_456", "username": "bob"}]},
        }

        await adapter._process_dm_event(event)
        adapter._message_handler.assert_not_called()


# ===========================================================================
# 10. build_source for Twitter messages
# ===========================================================================

class TestBuildSource:
    """Verify build_source produces correct SessionSource objects."""

    def test_dm_source(self):
        adapter = _make_adapter()
        source = adapter.build_source(
            chat_id="conv_123",
            chat_name="DM with @alice",
            chat_type="dm",
            user_id="user_456",
            user_name="alice",
        )
        assert source.platform == Platform.TWITTER
        assert source.chat_id == "conv_123"
        assert source.chat_type == "dm"
        assert source.user_id == "user_456"
        assert source.user_name == "alice"

    def test_source_description(self):
        adapter = _make_adapter()
        source = adapter.build_source(
            chat_id="conv_1",
            chat_type="dm",
            user_id="user_1",
            user_name="bob",
        )
        assert "bob" in source.description

    def test_source_to_dict_roundtrip(self):
        adapter = _make_adapter()
        source = adapter.build_source(
            chat_id="conv_99",
            chat_type="dm",
            user_id="uid_1",
            user_name="carol",
        )
        d = source.to_dict()
        assert d["platform"] == "twitter"
        assert d["chat_id"] == "conv_99"

        from gateway.session import SessionSource
        restored = SessionSource.from_dict(d)
        assert restored.platform == Platform.TWITTER
        assert restored.chat_id == "conv_99"


# ===========================================================================
# 11. Stream reconnection logic
# ===========================================================================

class TestStreamReconnection:
    """Verify exponential backoff on stream disconnects."""

    def test_stream_backoff_state(self):
        adapter = _make_adapter()
        # Initially no backoff
        assert not adapter._stream_backoff.should_backoff()

    def test_stream_backoff_increases(self):
        adapter = _make_adapter()
        w1 = adapter._stream_backoff.record_429()
        w2 = adapter._stream_backoff.record_429()
        assert w2 > w1

    @pytest.mark.asyncio
    async def test_stream_loop_respects_running_flag(self):
        """Stream loop should exit promptly when _running is set to False."""
        adapter = _make_adapter()
        adapter._running = False  # Already stopped

        # The stream loop should return immediately
        # We mock _listen_stream to avoid actual connection
        adapter._listen_stream = AsyncMock(side_effect=Exception("should not be called"))
        # Should not raise
        await adapter._stream_loop()


# ===========================================================================
# 12. Format message (strip markdown)
# ===========================================================================

class TestFormatMessage:
    """Verify format_message strips markdown for Twitter."""

    def test_strip_bold(self):
        adapter = _make_adapter()
        assert adapter.format_message("**bold** text") == "bold text"

    def test_strip_italic(self):
        adapter = _make_adapter()
        assert adapter.format_message("*italic* text") == "italic text"
        assert adapter.format_message("_italic_ text") == "italic text"

    def test_strip_inline_code(self):
        adapter = _make_adapter()
        assert adapter.format_message("use `code` here") == "use code here"

    def test_strip_code_block(self):
        adapter = _make_adapter()
        text = "```python\nprint('hello')\n```"
        result = adapter.format_message(text)
        assert "print('hello')" in result
        assert "```" not in result

    def test_collapse_blank_lines(self):
        adapter = _make_adapter()
        text = "line1\n\n\n\nline2"
        assert adapter.format_message(text) == "line1\n\nline2"

    def test_preserves_plain_text(self):
        adapter = _make_adapter()
        text = "Just a normal tweet about things."
        assert adapter.format_message(text) == text


# ===========================================================================
# 13. check_twitter_requirements
# ===========================================================================

class TestCheckRequirements:
    """Verify dependency checking."""

    def test_requirements_check(self):
        # httpx is mocked, so it should pass
        assert check_twitter_requirements() is True


# ===========================================================================
# 14. API error logging is sanitized
# ===========================================================================

class TestSanitizedErrorLogging:
    """Verify that API error responses are truncated in log messages."""

    @pytest.mark.asyncio
    async def test_api_error_log_truncated(self, caplog):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        large_error_body = json.dumps({
            "errors": [{"message": "X" * 1000}],
            "detail": "sensitive" * 200,
        })

        resp = MagicMock()
        resp.status_code = 500
        resp.text = large_error_body
        resp.json.return_value = {}

        client = MagicMock()
        client.is_closed = False
        client.request = AsyncMock(return_value=resp)
        adapter._client = client

        import logging
        with caplog.at_level(logging.ERROR):
            code, _ = await adapter._api_request("GET", f"{TWITTER_API_BASE}/2/test")

        assert code == 500
        # The log message should contain truncated body, not full 2000+ chars
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_logs) > 0
        for record in error_logs:
            msg = record.getMessage()
            # The full body should NOT appear in logs
            assert len(msg) < len(large_error_body) + 100  # Some overhead for prefix

    @pytest.mark.asyncio
    async def test_token_refresh_error_log_truncated(self, caplog):
        adapter = _make_adapter()

        large_error = "oauth_error: " + "X" * 500
        resp = MagicMock()
        resp.status_code = 400
        resp.text = large_error

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        import logging
        with caplog.at_level(logging.ERROR):
            result = await adapter._refresh_access_token()

        assert result is False
        # Error should mention the truncated body
        error_logs = [r for r in caplog.records if "token refresh failed" in r.getMessage()]
        assert len(error_logs) > 0


# ===========================================================================
# 15. Send result handling
# ===========================================================================

class TestSendResult:
    """Verify send returns proper SendResult objects."""

    @pytest.mark.asyncio
    async def test_send_success(self):
        adapter = _make_adapter()

        adapter._api_request = AsyncMock(return_value=(200, {"data": {"id": "tweet_123"}}))

        result = await adapter.send("ignored", "Hello!")
        assert result.success
        assert result.message_id == "tweet_123"

    @pytest.mark.asyncio
    async def test_send_api_failure(self):
        adapter = _make_adapter()

        adapter._api_request = AsyncMock(return_value=(403, {"error": "forbidden"}))

        result = await adapter.send("ignored", "Hello!")
        assert not result.success
        assert "403" in result.error

    @pytest.mark.asyncio
    async def test_send_empty_text(self):
        adapter = _make_adapter()
        adapter._split_for_tweets = MagicMock(return_value=[])
        result = await adapter.send("ignored", "")
        assert not result.success

    @pytest.mark.asyncio
    async def test_send_image_success(self):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        # Mock image download
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"\xff\xd8\xff" + b"\x00" * 100
        img_resp.raise_for_status = MagicMock()

        # Mock media upload
        adapter.upload_media = AsyncMock(return_value="media_id_1")

        # Mock tweet post
        adapter._api_request = AsyncMock(return_value=(201, {"data": {"id": "img_tweet"}}))

        client = MagicMock()
        client.is_closed = False
        client.get = AsyncMock(return_value=img_resp)
        adapter._client = client

        result = await adapter.send_image("ignored", "https://example.com/img.jpg", "caption")
        assert result.success
        assert result.message_id == "img_tweet"


# ===========================================================================
# 16. Connection lifecycle
# ===========================================================================

class TestConnectionLifecycle:
    """Verify connect/disconnect behavior."""

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self):
        config = PlatformConfig(enabled=True)
        with patch.dict(os.environ, {}, clear=True):
            adapter = TwitterAdapter(config)

        adapter._ensure_valid_token = AsyncMock(return_value=False)
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_cancels_stream(self):
        adapter = _make_adapter()
        adapter._running = True

        # Create a real task-like object that can be awaited and cancelled
        loop = asyncio.get_event_loop()
        cancelled = asyncio.Event()

        async def dummy_coro():
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = loop.create_task(dummy_coro())
        adapter._stream_task = task

        client = MagicMock()
        client.is_closed = False
        client.aclose = AsyncMock()
        adapter._client = client

        with patch.object(adapter, "_mark_disconnected"):
            await adapter.disconnect()

        assert not adapter._running
        client.aclose.assert_called_once()


# ===========================================================================
# 17. User profile enrichment (NEW)
# ===========================================================================

class TestUserEnrichment:
    """Verify user profile fetching with LRU cache."""

    @pytest.mark.asyncio
    async def test_get_user_context_success(self):
        adapter = _make_adapter()
        user_data = {
            "data": {
                "name": "Alice Smith",
                "username": "alice",
                "description": "Software developer",
                "verified": True,
                "public_metrics": {
                    "followers_count": 1500,
                    "following_count": 300,
                    "tweet_count": 5000,
                },
            }
        }
        adapter._api_request = AsyncMock(return_value=(200, user_data))

        profile = await adapter._get_user_context("user_123")
        assert profile["display_name"] == "Alice Smith"
        assert profile["username"] == "alice"
        assert profile["bio"] == "Software developer"
        assert profile["follower_count"] == 1500
        assert profile["verified"] is True

    @pytest.mark.asyncio
    async def test_get_user_context_caching(self):
        adapter = _make_adapter()
        user_data = {
            "data": {
                "name": "Bob",
                "username": "bob",
                "description": "",
                "public_metrics": {"followers_count": 100},
            }
        }
        adapter._api_request = AsyncMock(return_value=(200, user_data))

        # First call — fetches from API
        await adapter._get_user_context("user_456")
        assert adapter._api_request.call_count == 1

        # Second call — should use cache
        await adapter._get_user_context("user_456")
        assert adapter._api_request.call_count == 1  # No additional API call

    @pytest.mark.asyncio
    async def test_get_user_context_cache_expiry(self):
        adapter = _make_adapter()
        adapter._user_cache_ttl = 0.1  # 100ms TTL for test
        user_data = {"data": {"name": "Test", "username": "test", "public_metrics": {}}}
        adapter._api_request = AsyncMock(return_value=(200, user_data))

        await adapter._get_user_context("user_ttl")
        assert adapter._api_request.call_count == 1

        # Wait for cache to expire
        await asyncio.sleep(0.15)

        await adapter._get_user_context("user_ttl")
        assert adapter._api_request.call_count == 2  # Re-fetched

    @pytest.mark.asyncio
    async def test_get_user_context_api_failure(self):
        adapter = _make_adapter()
        adapter._api_request = AsyncMock(return_value=(404, {}))

        profile = await adapter._get_user_context("nonexistent")
        assert profile["display_name"] == ""
        assert profile["follower_count"] == 0

    def test_format_user_context(self):
        adapter = _make_adapter()
        profile = {
            "display_name": "Alice",
            "username": "alice",
            "bio": "Developer",
            "follower_count": 500,
            "verified": False,
        }
        result = adapter._format_user_context(profile)
        assert "Alice" in result
        assert "@alice" in result
        assert "Developer" in result
        assert "500" in result

    def test_format_user_context_empty(self):
        adapter = _make_adapter()
        result = adapter._format_user_context({})
        assert result == "No profile info"

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        adapter = _make_adapter()
        adapter._user_cache_max = 3
        user_counter = [0]

        async def mock_api(method, url, **kwargs):
            idx = user_counter[0]
            return 200, {
                "data": {
                    "name": f"User{idx}",
                    "username": f"user{idx}",
                    "public_metrics": {"followers_count": idx},
                }
            }

        adapter._api_request = AsyncMock(side_effect=mock_api)

        # Fill cache to capacity
        for i in range(3):
            user_counter[0] = i
            await adapter._get_user_context(f"uid_{i}")

        assert len(adapter._user_cache) == 3

        # Adding one more should evict the oldest
        user_counter[0] = 3
        await adapter._get_user_context("uid_3")

        assert len(adapter._user_cache) == 3  # Still at capacity
        assert "uid_0" not in adapter._user_cache  # Oldest evicted
        assert "uid_3" in adapter._user_cache  # Newest present


# ===========================================================================
# 18. Conversation context fetching (NEW)
# ===========================================================================

class TestConversationContext:
    """Verify conversation context fetching from referenced tweets."""

    @pytest.mark.asyncio
    async def test_get_conversation_context_reply(self):
        adapter = _make_adapter()
        tweet_data = {
            "referenced_tweets": [
                {"type": "replied_to", "id": "parent_1"}
            ]
        }

        async def mock_api(method, url, **kwargs):
            if "parent_1" in url:
                return 200, {
                    "data": {
                        "text": "Original tweet text",
                        "author_id": "author_1",
                    },
                    "includes": {
                        "users": [{"id": "author_1", "username": "bob"}]
                    },
                }
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api)

        context = await adapter._get_conversation_context(tweet_data)
        assert "In reply to" in context
        assert "@bob" in context
        assert "Original tweet text" in context

    @pytest.mark.asyncio
    async def test_get_conversation_context_no_refs(self):
        adapter = _make_adapter()
        tweet_data = {"referenced_tweets": []}
        context = await adapter._get_conversation_context(tweet_data)
        assert context == ""

    @pytest.mark.asyncio
    async def test_get_conversation_context_nested(self):
        adapter = _make_adapter()
        adapter._conversation_depth = 3
        tweet_data = {
            "referenced_tweets": [
                {"type": "replied_to", "id": "level1"}
            ]
        }

        async def mock_api(method, url, **kwargs):
            if "level1" in url:
                return 200, {
                    "data": {
                        "text": "Level 1 reply",
                        "author_id": "a1",
                        "referenced_tweets": [
                            {"type": "replied_to", "id": "level0"}
                        ],
                    },
                    "includes": {"users": [{"id": "a1", "username": "user1"}]},
                }
            if "level0" in url:
                return 200, {
                    "data": {
                        "text": "Level 0 original",
                        "author_id": "a0",
                    },
                    "includes": {"users": [{"id": "a0", "username": "user0"}]},
                }
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api)

        context = await adapter._get_conversation_context(tweet_data)
        assert "@user1" in context
        assert "Level 1 reply" in context

    @pytest.mark.asyncio
    async def test_get_conversation_context_max_depth(self):
        adapter = _make_adapter()
        adapter._conversation_depth = 1  # Only 1 level deep

        tweet_data = {
            "referenced_tweets": [{"type": "replied_to", "id": "t1"}]
        }

        async def mock_api(method, url, **kwargs):
            if "t1" in url:
                return 200, {
                    "data": {
                        "text": "T1",
                        "author_id": "a1",
                        "referenced_tweets": [
                            {"type": "replied_to", "id": "t0"}
                        ],
                    },
                    "includes": {"users": [{"id": "a1", "username": "u1"}]},
                }
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api)
        context = await adapter._get_conversation_context(tweet_data)
        # Should have T1 but not follow to t0 due to depth limit
        assert "T1" in context


# ===========================================================================
# 19. Conversation tree building (NEW)
# ===========================================================================

class TestConversationTree:
    """Verify conversation tree building."""

    @pytest.mark.asyncio
    async def test_build_conversation_tree(self):
        adapter = _make_adapter()

        async def mock_api(method, url, **kwargs):
            if "tweet_2" in url:
                return 200, {
                    "data": {
                        "text": "Reply tweet",
                        "author_id": "user_b",
                        "referenced_tweets": [
                            {"type": "replied_to", "id": "tweet_1"}
                        ],
                    },
                    "includes": {"users": [{"id": "user_b", "username": "bob"}]},
                }
            if "tweet_1" in url:
                return 200, {
                    "data": {
                        "text": "Original tweet",
                        "author_id": "user_a",
                    },
                    "includes": {"users": [{"id": "user_a", "username": "alice"}]},
                }
            return 404, {}

        adapter._api_request = AsyncMock(side_effect=mock_api)

        tree = await adapter._build_conversation_tree("tweet_2", max_depth=3)
        assert "@alice" in tree
        assert "@bob" in tree
        assert "Original tweet" in tree
        assert "Reply tweet" in tree

    @pytest.mark.asyncio
    async def test_build_conversation_tree_api_error(self):
        adapter = _make_adapter()
        adapter._api_request = AsyncMock(return_value=(404, {}))

        tree = await adapter._build_conversation_tree("nonexistent")
        assert tree == ""


# ===========================================================================
# 20. Tweet metrics (NEW)
# ===========================================================================

class TestTweetMetrics:
    """Verify tweet metrics fetching."""

    @pytest.mark.asyncio
    async def test_get_tweet_metrics_success(self):
        adapter = _make_adapter()
        adapter._api_request = AsyncMock(return_value=(200, {
            "data": {
                "public_metrics": {
                    "like_count": 42,
                    "retweet_count": 10,
                    "reply_count": 5,
                    "quote_count": 3,
                    "impression_count": 1000,
                    "bookmark_count": 8,
                }
            }
        }))

        metrics = await adapter._get_tweet_metrics("tweet_123")
        assert metrics["like_count"] == 42
        assert metrics["retweet_count"] == 10
        assert metrics["impression_count"] == 1000
        assert metrics["bookmark_count"] == 8

    @pytest.mark.asyncio
    async def test_get_tweet_metrics_api_error(self):
        adapter = _make_adapter()
        adapter._api_request = AsyncMock(return_value=(404, {}))

        metrics = await adapter._get_tweet_metrics("bad_id")
        assert metrics["like_count"] == 0
        assert metrics["impression_count"] == 0


# ===========================================================================
# 21. Thread builder (NEW)
# ===========================================================================

class TestThreadBuilder:
    """Verify the explicit thread builder method."""

    @pytest.mark.asyncio
    async def test_send_thread_creates_chain(self):
        adapter = _make_adapter()

        call_log = []

        async def mock_api(method, url, **kwargs):
            if method == "POST" and "/tweets" in url:
                body = kwargs.get("json_body", {})
                call_log.append(body)
                tid = f"t{len(call_log)}"
                return 201, {"data": {"id": tid}}
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api)

        ids = await adapter.send_thread(["First", "Second", "Third"])
        assert len(ids) == 3
        assert ids == ["t1", "t2", "t3"]

        # Verify chaining
        assert "reply" not in call_log[0]  # No parent for first tweet
        assert call_log[1]["reply"]["in_reply_to_tweet_id"] == "t1"
        assert call_log[2]["reply"]["in_reply_to_tweet_id"] == "t2"

    @pytest.mark.asyncio
    async def test_send_thread_with_reply_to(self):
        adapter = _make_adapter()

        call_log = []

        async def mock_api(method, url, **kwargs):
            if method == "POST" and "/tweets" in url:
                body = kwargs.get("json_body", {})
                call_log.append(body)
                return 201, {"data": {f"id": f"t{len(call_log)}"}}
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api)

        ids = await adapter.send_thread(["A", "B"], reply_to="parent_tweet")
        assert len(ids) == 2
        assert call_log[0]["reply"]["in_reply_to_tweet_id"] == "parent_tweet"
        assert call_log[1]["reply"]["in_reply_to_tweet_id"] == "t1"

    @pytest.mark.asyncio
    async def test_send_thread_stops_on_error(self):
        adapter = _make_adapter()

        async def mock_api(method, url, **kwargs):
            if method == "POST" and "/tweets" in url:
                return 403, {"error": "forbidden"}
            return 200, {}

        adapter._api_request = AsyncMock(side_effect=mock_api)

        ids = await adapter.send_thread(["First", "Second"])
        assert len(ids) == 0


# ===========================================================================
# 22. Image alt text (NEW)
# ===========================================================================

class TestImageAltText:
    """Verify image alt text support."""

    @pytest.mark.asyncio
    async def test_send_image_with_alt_text(self):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        adapter.upload_media = AsyncMock(return_value="media_123")
        adapter._set_media_alt_text = AsyncMock(return_value=True)
        adapter._api_request = AsyncMock(return_value=(201, {"data": {"id": "tweet_1"}}))

        client = MagicMock()
        client.is_closed = False
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"\xff\xd8\xff" + b"\x00" * 100
        img_resp.raise_for_status = MagicMock()
        client.get = AsyncMock(return_value=img_resp)
        adapter._client = client

        result = await adapter.send_image(
            "ignored", "https://example.com/img.jpg",
            caption="Look at this!", alt_text="A beautiful sunset"
        )
        assert result.success
        adapter._set_media_alt_text.assert_called_once_with("media_123", "A beautiful sunset")

    @pytest.mark.asyncio
    async def test_send_image_without_alt_text(self):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        adapter.upload_media = AsyncMock(return_value="media_456")
        adapter._set_media_alt_text = AsyncMock()
        adapter._api_request = AsyncMock(return_value=(201, {"data": {"id": "tweet_2"}}))

        client = MagicMock()
        client.is_closed = False
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = b"\xff\xd8\xff"
        img_resp.raise_for_status = MagicMock()
        client.get = AsyncMock(return_value=img_resp)
        adapter._client = client

        result = await adapter.send_image("ignored", "https://example.com/img.jpg")
        assert result.success
        adapter._set_media_alt_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_media_alt_text_success(self):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        result = await adapter._set_media_alt_text("media_1", "Alt text here")
        assert result is True

    @pytest.mark.asyncio
    async def test_set_media_alt_text_truncates(self):
        adapter = _make_adapter()
        adapter._access_token = "fake"
        adapter._token_expiry = time.time() + 3600

        resp = MagicMock()
        resp.status_code = 200
        resp.text = ""

        client = MagicMock()
        client.is_closed = False
        client.post = AsyncMock(return_value=resp)
        adapter._client = client

        long_text = "X" * 2000
        result = await adapter._set_media_alt_text("media_1", long_text)
        assert result is True
        # Verify the text was truncated in the call
        call_args = client.post.call_args
        payload = call_args.kwargs.get("json", {})
        assert len(payload["alt_text"]["text"]) <= 1000


# ===========================================================================
# 23. Rate limit queue (NEW)
# ===========================================================================

class TestRateLimitQueue:
    """Verify the rate-limit tweet queue."""

    @pytest.mark.asyncio
    async def test_enqueue_tweet(self):
        adapter = _make_adapter()
        adapter._running = True
        adapter._queue_enabled = True

        # Mock _api_request to succeed
        adapter._api_request = AsyncMock(return_value=(201, {"data": {"id": "queued_tweet"}}))

        # Start the queue processor
        task = asyncio.ensure_future(adapter._process_tweet_queue())

        # Enqueue a tweet
        code, data = await adapter._enqueue_tweet({"text": "queued"})
        assert code == 201
        assert data["data"]["id"] == "queued_tweet"

        adapter._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_queue_depth_logging(self):
        adapter = _make_adapter()
        adapter._running = True

        # Verify queue tracks depth
        await adapter._tweet_queue.put(({"text": "test"}, asyncio.Future()))
        assert adapter._tweet_queue.qsize() == 1

        adapter._running = False


# ===========================================================================
# 24. Bookmark sync (NEW)
# ===========================================================================

class TestBookmarkSync:
    """Verify bookmark sync functionality."""

    @pytest.mark.asyncio
    async def test_process_bookmarks_success(self):
        adapter = _make_adapter()
        adapter._bookmark_last_seen = None

        adapter._api_request = AsyncMock(return_value=(200, {
            "data": [
                {
                    "id": "bm_1",
                    "text": "Interesting bookmark",
                    "author_id": "author_1",
                    "created_at": "2025-01-01T00:00:00Z",
                }
            ],
            "includes": {
                "users": [{"id": "author_1", "username": "smartguy"}]
            },
        }))

        received = []
        async def mock_handle(event):
            received.append(event)
        adapter.handle_message = mock_handle

        await adapter._process_bookmarks()
        assert len(received) == 1
        assert "Interesting bookmark" in received[0].text
        assert adapter._bookmark_last_seen == "bm_1"

    @pytest.mark.asyncio
    async def test_process_bookmarks_empty(self):
        adapter = _make_adapter()
        adapter._api_request = AsyncMock(return_value=(200, {"data": []}))

        received = []
        async def mock_handle(event):
            received.append(event)
        adapter.handle_message = mock_handle

        await adapter._process_bookmarks()
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_process_bookmarks_api_error(self):
        adapter = _make_adapter()
        adapter._api_request = AsyncMock(return_value=(403, {"error": "forbidden"}))

        received = []
        async def mock_handle(event):
            received.append(event)
        adapter.handle_message = mock_handle

        await adapter._process_bookmarks()
        assert len(received) == 0  # No bookmarks processed on error
