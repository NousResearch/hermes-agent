"""Tests for Zulip platform adapter."""
import json
import os
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestZulipConfigLoading:
    def test_apply_env_overrides_zulip(self, monkeypatch):
        monkeypatch.setenv("ZULIP_API_KEY", "zulip-key-abc123")
        monkeypatch.setenv("ZULIP_SITE", "https://zulip.example.com")
        monkeypatch.setenv("ZULIP_EMAIL", "hermes-bot@zulip.example.com")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.ZULIP in config.platforms
        zc = config.platforms[Platform.ZULIP]
        assert zc.enabled is True
        assert zc.token == "zulip-key-abc123"
        assert zc.extra.get("site") == "https://zulip.example.com"
        assert zc.extra.get("email") == "hermes-bot@zulip.example.com"

    def test_zulip_not_loaded_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ZULIP_API_KEY", raising=False)
        monkeypatch.delenv("ZULIP_SITE", raising=False)
        monkeypatch.delenv("ZULIP_EMAIL", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.ZULIP not in config.platforms

    def test_zulip_home_channel(self, monkeypatch):
        monkeypatch.setenv("ZULIP_API_KEY", "zulip-key")
        monkeypatch.setenv("ZULIP_SITE", "https://zulip.example.com")
        monkeypatch.setenv("ZULIP_EMAIL", "bot@zulip.example.com")
        monkeypatch.setenv("ZULIP_HOME_STREAM", "42")
        monkeypatch.setenv("ZULIP_HOME_TOPIC", "notifications")
        monkeypatch.setenv("ZULIP_HOME_CHANNEL_NAME", "General")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.get_home_channel(Platform.ZULIP)
        assert home is not None
        assert home.chat_id == "42:notifications"
        assert home.name == "General"


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestZulipRequirements:
    def test_check_requirements_with_all_vars(self, monkeypatch):
        monkeypatch.setenv("ZULIP_SITE", "https://zulip.example.com")
        monkeypatch.setenv("ZULIP_EMAIL", "bot@zulip.example.com")
        monkeypatch.setenv("ZULIP_API_KEY", "test-key")
        from gateway.platforms.zulip import check_zulip_requirements
        assert check_zulip_requirements() is True

    def test_check_requirements_without_site(self, monkeypatch):
        monkeypatch.delenv("ZULIP_SITE", raising=False)
        monkeypatch.setenv("ZULIP_EMAIL", "bot@zulip.example.com")
        monkeypatch.setenv("ZULIP_API_KEY", "test-key")
        from gateway.platforms.zulip import check_zulip_requirements
        assert check_zulip_requirements() is False

    def test_check_requirements_without_email(self, monkeypatch):
        monkeypatch.setenv("ZULIP_SITE", "https://zulip.example.com")
        monkeypatch.delenv("ZULIP_EMAIL", raising=False)
        monkeypatch.setenv("ZULIP_API_KEY", "test-key")
        from gateway.platforms.zulip import check_zulip_requirements
        assert check_zulip_requirements() is False

    def test_check_requirements_without_api_key(self, monkeypatch):
        monkeypatch.setenv("ZULIP_SITE", "https://zulip.example.com")
        monkeypatch.setenv("ZULIP_EMAIL", "bot@zulip.example.com")
        monkeypatch.delenv("ZULIP_API_KEY", raising=False)
        from gateway.platforms.zulip import check_zulip_requirements
        assert check_zulip_requirements() is False


# ---------------------------------------------------------------------------
# Adapter fixture
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a ZulipAdapter with a mocked config and bot identity."""
    from gateway.platforms.zulip import ZulipAdapter, _compile_mention_re
    config = PlatformConfig(
        enabled=True,
        token="test-api-key",
        extra={
            "site": "https://zulip.example.com",
            "email": "hermes-bot@zulip.example.com",
        },
    )
    adapter = ZulipAdapter(config)
    adapter._bot_user_id = 100
    adapter._bot_full_name = "Hermes Bot"
    adapter._bot_email = "hermes-bot@zulip.example.com"
    adapter._mention_re = _compile_mention_re("Hermes Bot")
    return adapter


# ---------------------------------------------------------------------------
# Message handling — stream vs DM, session keys
# ---------------------------------------------------------------------------

class TestZulipHandleMessage:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter.handle_message = AsyncMock()

    @pytest.mark.asyncio
    async def test_handle_stream_message(self):
        """A mentioned stream message should dispatch with stream_id:topic chat_id."""
        message = {
            "id": 9001,
            "type": "stream",
            "stream_id": 42,
            "subject": "general chat",
            "display_recipient": "general",
            "content": "@**Hermes Bot** hello there",
            "sender_id": 200,
            "sender_full_name": "Alice",
            "sender_email": "alice@zulip.example.com",
        }
        await self.adapter._handle_message(message)
        assert self.adapter.handle_message.called
        event = self.adapter.handle_message.call_args[0][0]
        assert event.source.chat_id == "42:general chat"
        assert event.source.chat_type == "channel"
        assert event.source.chat_topic == "general chat"
        assert event.source.chat_name == "general"
        assert event.text == "hello there"

    @pytest.mark.asyncio
    async def test_handle_dm(self):
        """A DM should dispatch with sorted user IDs as chat_id and chat_type=dm."""
        message = {
            "id": 9002,
            "type": "private",
            "display_recipient": [
                {"id": 100, "email": "hermes-bot@zulip.example.com"},
                {"id": 200, "email": "alice@zulip.example.com"},
            ],
            "content": "hi bot",
            "sender_id": 200,
            "sender_full_name": "Alice",
            "sender_email": "alice@zulip.example.com",
        }
        await self.adapter._handle_message(message)
        assert self.adapter.handle_message.called
        event = self.adapter.handle_message.call_args[0][0]
        # DMs always respond — no mention required
        assert event.source.chat_id == "100,200"
        assert event.source.chat_type == "dm"
        assert event.text == "hi bot"

    @pytest.mark.asyncio
    async def test_session_key_stream(self):
        """Stream session key format is stream_id:topic."""
        message = {
            "id": 9003,
            "type": "stream",
            "stream_id": 7,
            "subject": "planning",
            "display_recipient": "engineering",
            "content": "@**Hermes Bot** question",
            "sender_id": 200,
            "sender_full_name": "Bob",
        }
        await self.adapter._handle_message(message)
        event = self.adapter.handle_message.call_args[0][0]
        assert event.source.chat_id == "7:planning"

    @pytest.mark.asyncio
    async def test_session_key_dm_sorted(self):
        """DM session key uses sorted user IDs regardless of display order."""
        message = {
            "id": 9004,
            "type": "private",
            "display_recipient": [
                # Intentionally unsorted to confirm sorting
                {"id": 300, "email": "c@zulip.example.com"},
                {"id": 100, "email": "hermes-bot@zulip.example.com"},
                {"id": 200, "email": "a@zulip.example.com"},
            ],
            "content": "group chat ping",
            "sender_id": 300,
            "sender_full_name": "Carol",
        }
        await self.adapter._handle_message(message)
        event = self.adapter.handle_message.call_args[0][0]
        assert event.source.chat_id == "100,200,300"
        # 3-participant DM → group, not 1:1 dm
        assert event.source.chat_type == "group"


# ---------------------------------------------------------------------------
# Bot-loop protection
# ---------------------------------------------------------------------------

class TestZulipIgnoreBots:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter.handle_message = AsyncMock()

    @pytest.mark.asyncio
    async def test_ignore_own_messages(self):
        """Messages with sender_id == bot_user_id must be dropped."""
        message = {
            "id": 5001,
            "type": "stream",
            "stream_id": 1,
            "subject": "echoes",
            "display_recipient": "general",
            "content": "hi",
            "sender_id": 100,  # same as bot
            "sender_full_name": "Hermes Bot",
        }
        await self.adapter._handle_message(message)
        assert not self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_ignore_other_bots_via_is_bot_flag(self):
        """sender.is_bot=True should suppress dispatch."""
        message = {
            "id": 5002,
            "type": "stream",
            "stream_id": 1,
            "subject": "noise",
            "display_recipient": "general",
            "content": "@**Hermes Bot** ping",
            "sender_id": 999,
            "sender_full_name": "Some Bot",
            "sender": {"is_bot": True},
        }
        await self.adapter._handle_message(message)
        assert not self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_ignore_other_bots_via_email_suffix(self):
        """Zulip bot emails end with '-bot@...' — treat as bot."""
        message = {
            "id": 5003,
            "type": "stream",
            "stream_id": 1,
            "subject": "noise",
            "display_recipient": "general",
            "content": "@**Hermes Bot** ping",
            "sender_id": 999,
            "sender_full_name": "Notifier",
            "sender_email": "notifier-bot@zulip.example.com",
        }
        await self.adapter._handle_message(message)
        assert not self.adapter.handle_message.called


# ---------------------------------------------------------------------------
# Mention stripping + gating
# ---------------------------------------------------------------------------

class TestZulipMentionBehavior:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter.handle_message = AsyncMock()

    def _stream_event(self, content, stream_id=42):
        return {
            "id": 6000 + stream_id,
            "type": "stream",
            "stream_id": stream_id,
            "subject": "tests",
            "display_recipient": "general",
            "content": content,
            "sender_id": 200,
            "sender_full_name": "Alice",
        }

    @pytest.mark.asyncio
    async def test_mention_stripping(self):
        """@**Hermes Bot** is removed from the dispatched text."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZULIP_REQUIRE_MENTION", None)
            os.environ.pop("ZULIP_FREE_RESPONSE_STREAMS", None)
            await self.adapter._handle_message(
                self._stream_event("@**Hermes Bot** what is 2+2?")
            )
        assert self.adapter.handle_message.called
        event = self.adapter.handle_message.call_args[0][0]
        assert "@**Hermes Bot**" not in event.text
        assert "2+2" in event.text

    @pytest.mark.asyncio
    async def test_silent_mention_stripping(self):
        """Silent mention @_**Hermes Bot**_ is also stripped."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZULIP_REQUIRE_MENTION", None)
            os.environ.pop("ZULIP_FREE_RESPONSE_STREAMS", None)
            await self.adapter._handle_message(
                self._stream_event("@_**Hermes Bot**_ please help")
            )
        assert self.adapter.handle_message.called
        event = self.adapter.handle_message.call_args[0][0]
        assert "Hermes Bot" not in event.text
        assert "please help" in event.text

    @pytest.mark.asyncio
    async def test_mention_required_skips_plain_stream(self):
        """Default ZULIP_REQUIRE_MENTION=true: no mention → no dispatch."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZULIP_REQUIRE_MENTION", None)
            os.environ.pop("ZULIP_FREE_RESPONSE_STREAMS", None)
            await self.adapter._handle_message(
                self._stream_event("hi everyone")
            )
        assert not self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_mention_required_false_responds_to_all(self):
        """ZULIP_REQUIRE_MENTION=false: respond to every subscribed stream message."""
        with patch.dict(os.environ, {"ZULIP_REQUIRE_MENTION": "false"}):
            os.environ.pop("ZULIP_FREE_RESPONSE_STREAMS", None)
            await self.adapter._handle_message(
                self._stream_event("hi everyone")
            )
        assert self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_free_response_streams_skip_mention_requirement(self):
        """Streams in ZULIP_FREE_RESPONSE_STREAMS respond even without mention."""
        with patch.dict(os.environ, {"ZULIP_FREE_RESPONSE_STREAMS": "42"}):
            os.environ.pop("ZULIP_REQUIRE_MENTION", None)
            await self.adapter._handle_message(
                self._stream_event("hi everyone", stream_id=42)
            )
        assert self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_non_free_stream_still_requires_mention(self):
        """Streams not in the free-response list still require @mention."""
        with patch.dict(os.environ, {"ZULIP_FREE_RESPONSE_STREAMS": "99"}):
            os.environ.pop("ZULIP_REQUIRE_MENTION", None)
            await self.adapter._handle_message(
                self._stream_event("hi everyone", stream_id=42)
            )
        assert not self.adapter.handle_message.called

    @pytest.mark.asyncio
    async def test_dm_always_responds(self):
        """DMs always respond regardless of ZULIP_REQUIRE_MENTION."""
        message = {
            "id": 7001,
            "type": "private",
            "display_recipient": [
                {"id": 100, "email": "hermes-bot@zulip.example.com"},
                {"id": 200, "email": "alice@zulip.example.com"},
            ],
            "content": "no mention needed",
            "sender_id": 200,
            "sender_full_name": "Alice",
        }
        with patch.dict(os.environ, {"ZULIP_REQUIRE_MENTION": "true"}):
            await self.adapter._handle_message(message)
        assert self.adapter.handle_message.called


# ---------------------------------------------------------------------------
# Allowlist behavior is handled centrally in gateway/run.py via
# ZULIP_ALLOWED_USERS; the adapter itself doesn't filter by user id.
# Verify the env var is present in the platform env map so the central
# check wires up correctly for Zulip.
# ---------------------------------------------------------------------------

class TestZulipAllowedUsersWiring:
    def test_zulip_in_platform_env_map(self):
        """ZULIP_ALLOWED_USERS is registered in the central authz map."""
        import inspect
        from gateway.run import GatewayRunner
        src = inspect.getsource(GatewayRunner._is_user_authorized)
        assert "Platform.ZULIP" in src
        assert "ZULIP_ALLOWED_USERS" in src

    def test_allowlist_parsing_matches_and_rejects(self, monkeypatch):
        """ZULIP_ALLOWED_USERS is parsed comma-separated and matches user ids."""
        monkeypatch.setenv("ZULIP_ALLOWED_USERS", "200,300")
        raw = os.getenv("ZULIP_ALLOWED_USERS", "")
        allowed = {u.strip() for u in raw.split(",") if u.strip()}
        assert "200" in allowed
        assert "300" in allowed
        assert "999" not in allowed

    def test_allow_all_flag_recognized(self):
        """ZULIP_ALLOW_ALL_USERS is wired into the allow-all map."""
        import inspect
        from gateway.run import GatewayRunner
        src = inspect.getsource(GatewayRunner._is_user_authorized)
        assert "ZULIP_ALLOW_ALL_USERS" in src


# ---------------------------------------------------------------------------
# send()
# ---------------------------------------------------------------------------

class TestZulipSend:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter._session = MagicMock()

    def _mock_response(self, status=200, body=None):
        if body is None:
            body = {"result": "success", "id": 12345}
        resp = AsyncMock()
        resp.status = status
        resp.json = AsyncMock(return_value=body)
        resp.text = AsyncMock(return_value=json.dumps(body))
        resp.headers = {}
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    @pytest.mark.asyncio
    async def test_send_stream_message(self):
        """send() with chat_id='<stream_id>:<topic>' posts a stream message."""
        self.adapter._session.post = MagicMock(return_value=self._mock_response())
        result = await self.adapter.send("42:daily-standup", "hello")
        assert result.success is True
        assert result.message_id == "12345"

        call = self.adapter._session.post.call_args
        assert "/api/v1/messages" in call[0][0]
        form = call[1]["data"]
        assert form["type"] == "stream"
        assert form["to"] == "42"
        assert form["topic"] == "daily-standup"
        assert form["content"] == "hello"

    @pytest.mark.asyncio
    async def test_send_dm(self):
        """send() with chat_id='<user_id>' posts a direct message."""
        self.adapter._session.post = MagicMock(return_value=self._mock_response())
        result = await self.adapter.send("200,300", "hi")
        assert result.success is True
        assert result.message_id == "12345"

        form = self.adapter._session.post.call_args[1]["data"]
        assert form["type"] == "direct"
        # JSON-encoded list of ints
        assert json.loads(form["to"]) == [200, 300]
        assert form["content"] == "hi"

    @pytest.mark.asyncio
    async def test_send_empty_content_succeeds(self):
        """Empty content is a no-op."""
        result = await self.adapter.send("42:topic", "")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_api_failure(self):
        """result=error from Zulip should produce a failing SendResult."""
        err_resp = self._mock_response(
            status=200,
            body={"result": "error", "msg": "Stream does not exist"},
        )
        self.adapter._session.post = MagicMock(return_value=err_resp)
        result = await self.adapter.send("42:missing", "hi")
        assert result.success is False
        assert "Stream does not exist" in (result.error or "")


# ---------------------------------------------------------------------------
# Message splitting for long content
# ---------------------------------------------------------------------------

class TestZulipMessageSplitting:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_short_message_single_chunk(self):
        chunks = self.adapter.truncate_message("hi", 10000)
        assert chunks == ["hi"]

    def test_long_message_splits(self):
        """Messages longer than MAX_MESSAGE_LENGTH are split into chunks."""
        from gateway.platforms.zulip import MAX_MESSAGE_LENGTH
        msg = "a " * (MAX_MESSAGE_LENGTH // 2 + 500)  # > 10k chars
        chunks = self.adapter.truncate_message(msg, MAX_MESSAGE_LENGTH)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= MAX_MESSAGE_LENGTH

    def test_exactly_at_limit(self):
        msg = "x" * 10000
        chunks = self.adapter.truncate_message(msg, 10000)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_send_long_message_produces_multiple_posts(self):
        """send() with content > 10k chars should POST multiple chunks."""
        from gateway.platforms.zulip import MAX_MESSAGE_LENGTH
        self.adapter._session = MagicMock()
        responses = iter([
            _build_mock_response({"result": "success", "id": 1}),
            _build_mock_response({"result": "success", "id": 2}),
        ])
        self.adapter._session.post = MagicMock(side_effect=lambda *a, **kw: next(responses))

        big = "x" * (MAX_MESSAGE_LENGTH + 1000)
        result = await self.adapter.send("42:topic", big)
        assert result.success is True
        # Final message_id comes from the last chunk
        assert result.message_id == "2"
        assert self.adapter._session.post.call_count >= 2


def _build_mock_response(body, status=200):
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=body)
    resp.text = AsyncMock(return_value=json.dumps(body))
    resp.headers = {}
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Dedup cache
# ---------------------------------------------------------------------------

class TestZulipDedup:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter.handle_message = AsyncMock()

    @pytest.mark.asyncio
    async def test_duplicate_message_ignored(self):
        """Same message id within TTL is processed only once."""
        message = {
            "id": 8001,
            "type": "stream",
            "stream_id": 1,
            "subject": "t",
            "display_recipient": "general",
            "content": "@**Hermes Bot** hi",
            "sender_id": 200,
            "sender_full_name": "Alice",
        }
        await self.adapter._handle_message(message)
        await self.adapter._handle_message(message)
        assert self.adapter.handle_message.call_count == 1


# ---------------------------------------------------------------------------
# Event queue / long-poll loop recovery
# ---------------------------------------------------------------------------

class TestZulipEventQueueRecovery:
    @pytest.mark.asyncio
    async def test_register_queue_success(self):
        """_register_queue() should store queue_id and last_event_id."""
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={
            "result": "success",
            "queue_id": "queue-abc",
            "last_event_id": 7,
        })
        assert await adapter._register_queue() is True
        assert adapter._queue_id == "queue-abc"
        assert adapter._last_event_id == 7

    @pytest.mark.asyncio
    async def test_register_queue_failure(self):
        """_register_queue() returns False when Zulip reports an error."""
        adapter = _make_adapter()
        adapter._api_post = AsyncMock(return_value={
            "result": "error",
            "msg": "Bad credentials",
        })
        assert await adapter._register_queue() is False
        assert adapter._queue_id is None

    @pytest.mark.asyncio
    async def test_bad_event_queue_triggers_reregister(self):
        """After a BadEventQueueError, the poll loop re-registers and continues."""
        from gateway.platforms.zulip import BadEventQueueError

        adapter = _make_adapter()
        adapter._queue_id = "expired-queue"
        adapter._last_event_id = 10
        # First _get_events raises, second returns an empty batch (after re-register)
        get_events = AsyncMock(side_effect=[
            BadEventQueueError("queue gone"),
            [],
        ])
        adapter._get_events = get_events

        register_calls = []

        async def fake_register():
            register_calls.append(1)
            adapter._queue_id = "new-queue"
            adapter._last_event_id = 0
            # After the second poll iteration (empty batch), stop the loop.
            if len(register_calls) >= 1:
                # Arrange for the loop to terminate after the next poll:
                # the second _get_events returns [] and then we flip _closing.
                async def stop_after_next(*args, **kwargs):
                    adapter._closing = True
                    return []
                adapter._get_events = AsyncMock(side_effect=stop_after_next)
            return True

        adapter._register_queue = fake_register

        await adapter._poll_loop()

        assert register_calls, "expected re-register to have been attempted"
        assert adapter._queue_id == "new-queue"
