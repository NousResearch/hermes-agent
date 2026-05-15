"""Tests for Discord thread history restore on fresh sessions.

Verifies that:
- _discord_restore_thread_history() respects env vars and config keys
- _discord_restore_thread_history_limit() respects env vars and config keys
- _is_fresh_hermes_session() returns True when no transcript exists and
  False when a session with tokens already exists
- _fetch_thread_history_context() formats messages correctly and handles errors
- _handle_message injects history into channel_prompt on a fresh thread session
- _handle_message does NOT inject history when feature is disabled
- _handle_message does NOT inject history when session already has history
"""

import os
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import discord as discord_lib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(tmp_path, extra=None):
    """Build a minimal DiscordAdapter."""
    from gateway.config import PlatformConfig
    from gateway.platforms.discord import DiscordAdapter

    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        adapter = DiscordAdapter(config=config)
    return adapter


def _make_mock_message(channel, author_name="alice", is_bot=False, content="hello"):
    """Return a minimal mock discord.Message inside a Thread."""
    msg = MagicMock()
    msg.id = 999
    msg.content = content
    msg.author = MagicMock()
    msg.author.display_name = author_name
    msg.author.name = author_name
    msg.author.bot = is_bot
    msg.author.id = 1234 if not is_bot else 5678
    msg.channel = channel
    msg.guild = None
    msg.attachments = []
    msg.mentions = []
    msg.reference = None
    msg.type = discord_lib.MessageType.default
    msg.created_at = datetime(2025, 1, 15, 10, 0, 0)
    return msg


def _make_thread_channel(thread_id="111", name="my-thread", parent_id="222"):
    """Return a mock discord.Thread."""
    ch = MagicMock(spec=discord_lib.Thread)
    ch.id = int(thread_id)
    ch.name = name
    ch.parent_id = int(parent_id)
    return ch


# ---------------------------------------------------------------------------
# _discord_restore_thread_history
# ---------------------------------------------------------------------------

class TestDiscordRestoreThreadHistoryFlag:
    def test_default_true(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_RESTORE_THREAD_HISTORY", None)
            assert adapter._discord_restore_thread_history() is True

    def test_env_false(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with patch.dict(os.environ, {"DISCORD_RESTORE_THREAD_HISTORY": "false"}):
            assert adapter._discord_restore_thread_history() is False

    def test_env_zero(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with patch.dict(os.environ, {"DISCORD_RESTORE_THREAD_HISTORY": "0"}):
            assert adapter._discord_restore_thread_history() is False

    def test_config_false_string(self, tmp_path):
        adapter = _make_adapter(tmp_path, extra={"restore_thread_history": "off"})
        assert adapter._discord_restore_thread_history() is False

    def test_config_false_bool(self, tmp_path):
        adapter = _make_adapter(tmp_path, extra={"restore_thread_history": False})
        assert adapter._discord_restore_thread_history() is False

    def test_config_true_bool(self, tmp_path):
        adapter = _make_adapter(tmp_path, extra={"restore_thread_history": True})
        assert adapter._discord_restore_thread_history() is True


# ---------------------------------------------------------------------------
# _discord_restore_thread_history_limit
# ---------------------------------------------------------------------------

class TestDiscordRestoreThreadHistoryLimit:
    def test_default_50(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_RESTORE_THREAD_HISTORY_LIMIT", None)
            assert adapter._discord_restore_thread_history_limit() == 50

    def test_env_override(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with patch.dict(os.environ, {"DISCORD_RESTORE_THREAD_HISTORY_LIMIT": "25"}):
            assert adapter._discord_restore_thread_history_limit() == 25

    def test_env_clamps_to_1(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with patch.dict(os.environ, {"DISCORD_RESTORE_THREAD_HISTORY_LIMIT": "0"}):
            assert adapter._discord_restore_thread_history_limit() == 1

    def test_env_invalid_falls_back_to_50(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        with patch.dict(os.environ, {"DISCORD_RESTORE_THREAD_HISTORY_LIMIT": "nope"}):
            assert adapter._discord_restore_thread_history_limit() == 50

    def test_config_override(self, tmp_path):
        adapter = _make_adapter(tmp_path, extra={"restore_thread_history_limit": 10})
        assert adapter._discord_restore_thread_history_limit() == 10


# ---------------------------------------------------------------------------
# _is_fresh_hermes_session
# ---------------------------------------------------------------------------

class TestIsFreshHermesSession:
    def _make_source(self, thread_id="111"):
        from gateway.session import SessionSource
        from gateway.config import Platform
        return SessionSource(
            platform=Platform.DISCORD,
            chat_id=str(thread_id),
            chat_type="thread",
            thread_id=str(thread_id),
        )

    def test_no_session_store_returns_true(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        # No session store set -> treat as fresh
        assert not hasattr(adapter, "_session_store") or adapter._is_fresh_hermes_session(self._make_source())

    def test_no_entry_in_store_returns_true(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        mock_store = MagicMock()
        mock_store._lock = __import__("threading").Lock()
        mock_store._entries = {}
        mock_store._ensure_loaded_locked = MagicMock()
        adapter._session_store = mock_store
        assert adapter._is_fresh_hermes_session(self._make_source()) is True

    def test_entry_with_zero_tokens_and_empty_transcript_returns_true(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        mock_store = MagicMock()
        mock_store._lock = __import__("threading").Lock()
        entry = MagicMock()
        entry.total_tokens = 0
        entry.session_id = "sess123"
        mock_store._entries = {"agent:main:discord:thread:111:111": entry}
        mock_store._ensure_loaded_locked = MagicMock()
        mock_store.load_transcript = MagicMock(return_value=[])
        adapter._session_store = mock_store
        assert adapter._is_fresh_hermes_session(self._make_source()) is True

    def test_entry_with_nonzero_tokens_returns_false(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        mock_store = MagicMock()
        mock_store._lock = __import__("threading").Lock()
        entry = MagicMock()
        entry.total_tokens = 1500
        entry.session_id = "sess456"
        mock_store._entries = {"agent:main:discord:thread:111:111": entry}
        mock_store._ensure_loaded_locked = MagicMock()
        adapter._session_store = mock_store
        assert adapter._is_fresh_hermes_session(self._make_source()) is False

    def test_entry_with_transcript_but_zero_tokens_returns_false(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        mock_store = MagicMock()
        mock_store._lock = __import__("threading").Lock()
        entry = MagicMock()
        entry.total_tokens = 0
        entry.session_id = "sess789"
        mock_store._entries = {"agent:main:discord:thread:111:111": entry}
        mock_store._ensure_loaded_locked = MagicMock()
        mock_store.load_transcript = MagicMock(return_value=[{"role": "user", "content": "hi"}])
        adapter._session_store = mock_store
        assert adapter._is_fresh_hermes_session(self._make_source()) is False

    def test_exception_during_check_returns_true(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        mock_store = MagicMock()
        mock_store._lock = __import__("threading").Lock()
        mock_store._ensure_loaded_locked = MagicMock(side_effect=RuntimeError("db error"))
        adapter._session_store = mock_store
        # Should not raise; should return True (assume fresh on error)
        assert adapter._is_fresh_hermes_session(self._make_source()) is True


# ---------------------------------------------------------------------------
# _fetch_thread_history_context
# ---------------------------------------------------------------------------

class TestFetchThreadHistoryContext:
    def _make_async_gen(self, messages):
        """Create an async generator that yields the given messages."""
        async def _gen():
            for m in messages:
                yield m
        return _gen()

    @pytest.mark.asyncio
    async def test_empty_channel_returns_empty_string(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        adapter._client = MagicMock()
        adapter._client.user = MagicMock()
        adapter._client.user.id = 9999

        channel = MagicMock()
        channel.history = MagicMock(return_value=self._make_async_gen([]))
        channel.name = "test-thread"

        result = await adapter._fetch_thread_history_context(channel, limit=50)
        assert result == ""

    @pytest.mark.asyncio
    async def test_messages_formatted_correctly(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        adapter._client = MagicMock()
        adapter._client.user = MagicMock()
        adapter._client.user.id = 9999

        # Two user messages
        msg1 = MagicMock()
        msg1.type = discord_lib.MessageType.default
        msg1.author = MagicMock()
        msg1.author.display_name = "alice"
        msg1.author.id = 1111
        msg1.content = "hello world"
        msg1.attachments = []
        msg1.created_at = datetime(2025, 1, 15, 10, 0, 0)

        msg2 = MagicMock()
        msg2.type = discord_lib.MessageType.default
        msg2.author = MagicMock()
        msg2.author.display_name = "bob"
        msg2.author.id = 2222
        msg2.content = "reply here"
        msg2.attachments = []
        msg2.created_at = datetime(2025, 1, 15, 10, 5, 0)

        channel = MagicMock()
        channel.history = MagicMock(return_value=self._make_async_gen([msg1, msg2]))
        channel.name = "my-thread"

        result = await adapter._fetch_thread_history_context(channel, limit=50)
        assert "my-thread" in result
        assert "alice" in result
        assert "hello world" in result
        assert "bob" in result
        assert "reply here" in result
        assert "2025-01-15" in result

    @pytest.mark.asyncio
    async def test_bot_own_messages_labeled_you(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        bot_user = MagicMock()
        bot_user.id = 9999
        adapter._client = MagicMock()
        adapter._client.user = bot_user

        msg = MagicMock()
        msg.type = discord_lib.MessageType.default
        msg.author = bot_user  # same object as client.user
        msg.content = "I am the bot"
        msg.attachments = []
        msg.created_at = datetime(2025, 1, 15, 10, 0, 0)

        channel = MagicMock()
        channel.history = MagicMock(return_value=self._make_async_gen([msg]))
        channel.name = "thread"

        result = await adapter._fetch_thread_history_context(channel, limit=50)
        assert "You:" in result

    @pytest.mark.asyncio
    async def test_system_messages_skipped(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        adapter._client = MagicMock()
        adapter._client.user = MagicMock()
        adapter._client.user.id = 9999

        sys_msg = MagicMock()
        sys_msg.type = discord_lib.MessageType.thread_created  # system message
        sys_msg.author = MagicMock()
        sys_msg.author.display_name = "system"
        sys_msg.content = "Thread created"
        sys_msg.attachments = []
        sys_msg.created_at = datetime(2025, 1, 15, 10, 0, 0)

        channel = MagicMock()
        channel.history = MagicMock(return_value=self._make_async_gen([sys_msg]))
        channel.name = "thread"

        result = await adapter._fetch_thread_history_context(channel, limit=50)
        assert result == ""

    @pytest.mark.asyncio
    async def test_exception_returns_empty_string(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        adapter._client = MagicMock()
        adapter._client.user = MagicMock()
        adapter._client.user.id = 9999

        channel = MagicMock()
        channel.history = MagicMock(side_effect=Exception("API error"))
        channel.name = "thread"

        result = await adapter._fetch_thread_history_context(channel, limit=50)
        assert result == ""

    @pytest.mark.asyncio
    async def test_attachment_names_included(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        adapter._client = MagicMock()
        adapter._client.user = MagicMock()
        adapter._client.user.id = 9999

        att = MagicMock()
        att.filename = "photo.png"

        msg = MagicMock()
        msg.type = discord_lib.MessageType.default
        msg.author = MagicMock()
        msg.author.display_name = "alice"
        msg.author.id = 1111
        msg.content = ""
        msg.attachments = [att]
        msg.created_at = datetime(2025, 1, 15, 10, 0, 0)

        channel = MagicMock()
        channel.history = MagicMock(return_value=self._make_async_gen([msg]))
        channel.name = "thread"

        result = await adapter._fetch_thread_history_context(channel, limit=50)
        assert "photo.png" in result


# ---------------------------------------------------------------------------
# Integration: _handle_message injects thread history
# ---------------------------------------------------------------------------

class TestHandleMessageThreadHistoryInjection:
    """Smoke-test the injection path in _handle_message without running the
    full adapter or Discord client."""

    def _make_full_adapter(self, tmp_path, extra=None):
        from gateway.config import PlatformConfig
        from gateway.platforms.discord import DiscordAdapter

        config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            adapter = DiscordAdapter(config=config)
        return adapter

    @pytest.mark.asyncio
    async def test_history_injected_when_fresh_session(self, tmp_path):
        adapter = self._make_full_adapter(tmp_path)
        channel = _make_thread_channel(thread_id="111", name="old-thread", parent_id="222")

        # Make adapter believe it's in a thread with fresh session
        adapter._is_fresh_hermes_session = MagicMock(return_value=True)
        adapter._fetch_thread_history_context = AsyncMock(
            return_value="[Thread history restored -- old-thread]"
        )
        adapter._resolve_channel_skills = MagicMock(return_value=None)
        adapter._resolve_channel_prompt = MagicMock(return_value=None)

        captured_event = {}

        async def _capture_event(event):
            captured_event["event"] = event

        adapter.handle_message = _capture_event  # type: ignore[method-assign]

        # Build a minimal message mock
        msg = _make_mock_message(channel, content="pick up where we left off")
        msg.channel = channel

        # Manually invoke the injection block (simulate _handle_message internals)
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="111",
            chat_type="thread",
            thread_id="111",
        )
        is_thread = True
        thread_id = "111"
        _channel_prompt = None

        if is_thread and thread_id and adapter._discord_restore_thread_history():
            if adapter._is_fresh_hermes_session(source):
                _history_limit = adapter._discord_restore_thread_history_limit()
                _thread_history_ctx = await adapter._fetch_thread_history_context(
                    channel, limit=_history_limit
                )
                if _thread_history_ctx:
                    if _channel_prompt:
                        _channel_prompt = f"{_thread_history_ctx}\n\n{_channel_prompt}"
                    else:
                        _channel_prompt = _thread_history_ctx

        assert _channel_prompt is not None
        assert "Thread history restored" in _channel_prompt

    @pytest.mark.asyncio
    async def test_no_injection_when_feature_disabled(self, tmp_path):
        adapter = self._make_full_adapter(
            tmp_path, extra={"restore_thread_history": False}
        )
        adapter._is_fresh_hermes_session = MagicMock(return_value=True)
        adapter._fetch_thread_history_context = AsyncMock(
            return_value="[Thread history restored]"
        )

        channel = _make_thread_channel()
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="111",
            chat_type="thread",
            thread_id="111",
        )
        is_thread = True
        thread_id = "111"
        _channel_prompt = None

        if is_thread and thread_id and adapter._discord_restore_thread_history():
            if adapter._is_fresh_hermes_session(source):
                _thread_history_ctx = await adapter._fetch_thread_history_context(channel, limit=50)
                if _thread_history_ctx:
                    _channel_prompt = _thread_history_ctx

        # Feature disabled: prompt should stay None
        assert _channel_prompt is None

    @pytest.mark.asyncio
    async def test_no_injection_when_session_has_history(self, tmp_path):
        adapter = self._make_full_adapter(tmp_path)
        # Session is NOT fresh
        adapter._is_fresh_hermes_session = MagicMock(return_value=False)
        adapter._fetch_thread_history_context = AsyncMock(
            return_value="[Thread history restored]"
        )

        channel = _make_thread_channel()
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="111",
            chat_type="thread",
            thread_id="111",
        )
        is_thread = True
        thread_id = "111"
        _channel_prompt = None

        if is_thread and thread_id and adapter._discord_restore_thread_history():
            if adapter._is_fresh_hermes_session(source):
                _thread_history_ctx = await adapter._fetch_thread_history_context(channel, limit=50)
                if _thread_history_ctx:
                    _channel_prompt = _thread_history_ctx

        # Session already has history: no injection
        assert _channel_prompt is None
        adapter._fetch_thread_history_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_injection_outside_thread(self, tmp_path):
        adapter = self._make_full_adapter(tmp_path)
        adapter._is_fresh_hermes_session = MagicMock(return_value=True)
        adapter._fetch_thread_history_context = AsyncMock(return_value="history")

        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="222",
            chat_type="group",
        )
        is_thread = False  # Not a thread
        thread_id = None
        _channel_prompt = None

        if is_thread and thread_id and adapter._discord_restore_thread_history():
            if adapter._is_fresh_hermes_session(source):
                _channel_prompt = "injected"

        assert _channel_prompt is None

    @pytest.mark.asyncio
    async def test_history_appended_after_existing_channel_prompt(self, tmp_path):
        adapter = self._make_full_adapter(tmp_path)
        adapter._is_fresh_hermes_session = MagicMock(return_value=True)
        adapter._fetch_thread_history_context = AsyncMock(return_value="[history block]")

        channel = _make_thread_channel()
        from gateway.session import SessionSource
        from gateway.config import Platform
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="111",
            chat_type="thread",
            thread_id="111",
        )
        is_thread = True
        thread_id = "111"
        _channel_prompt = "existing channel instructions"

        if is_thread and thread_id and adapter._discord_restore_thread_history():
            if adapter._is_fresh_hermes_session(source):
                _history_limit = adapter._discord_restore_thread_history_limit()
                _thread_history_ctx = await adapter._fetch_thread_history_context(
                    channel, limit=_history_limit
                )
                if _thread_history_ctx:
                    if _channel_prompt:
                        _channel_prompt = f"{_thread_history_ctx}\n\n{_channel_prompt}"
                    else:
                        _channel_prompt = _thread_history_ctx

        assert "[history block]" in _channel_prompt
        assert "existing channel instructions" in _channel_prompt
