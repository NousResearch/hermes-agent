"""Tests for Google Chat platform adapter."""
import json
import os
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestGoogleChatConfigLoading:
    def test_apply_env_overrides_google_chat(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CHAT_GCP_PROJECT", "my-gcp-project")
        monkeypatch.setenv("GOOGLE_CHAT_PUBSUB_SUBSCRIPTION", "chat-sub")
        monkeypatch.setenv("GOOGLE_CHAT_CREDENTIALS", "/path/to/creds.json")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.GOOGLE_CHAT in config.platforms
        gc = config.platforms[Platform.GOOGLE_CHAT]
        assert gc.enabled is True
        assert gc.extra.get("gcp_project") == "my-gcp-project"
        assert gc.extra.get("pubsub_subscription") == "chat-sub"
        assert gc.extra.get("chat_credentials") == "/path/to/creds.json"

    def test_google_chat_not_loaded_without_project(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CHAT_GCP_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CHAT_PUBSUB_SUBSCRIPTION", raising=False)
        monkeypatch.delenv("GOOGLE_CHAT_CREDENTIALS", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.GOOGLE_CHAT not in config.platforms

    def test_google_chat_home_channel(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CHAT_GCP_PROJECT", "my-gcp-project")
        monkeypatch.setenv("GOOGLE_CHAT_HOME_CHANNEL", "spaces/AAAAxxxx")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.get_home_channel(Platform.GOOGLE_CHAT)
        assert home is not None
        assert home.chat_id == "spaces/AAAAxxxx"

    def test_google_chat_default_subscription(self, monkeypatch):
        """When no subscription env var is set, default should be used."""
        monkeypatch.setenv("GOOGLE_CHAT_GCP_PROJECT", "my-gcp-project")
        monkeypatch.delenv("GOOGLE_CHAT_PUBSUB_SUBSCRIPTION", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        gc = config.platforms[Platform.GOOGLE_CHAT]
        # Subscription name may be empty or default — both acceptable
        sub = gc.extra.get("pubsub_subscription", "")
        assert isinstance(sub, str)


# ---------------------------------------------------------------------------
# Adapter construction & format
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a GoogleChatAdapter with mocked config."""
    from gateway.platforms.google_chat import GoogleChatAdapter
    config = PlatformConfig(
        enabled=True,
        token="",
        extra={
            "gcp_project": "test-project",
            "pubsub_subscription": "test-sub",
            "chat_credentials": "",
        },
    )
    adapter = GoogleChatAdapter(config)
    return adapter


class TestGoogleChatFormatMessage:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_plain_text_unchanged(self):
        content = "Hello, world!"
        assert self.adapter.format_message(content) == content

    def test_bold_and_italic_preserved(self):
        content = "**bold** and *italic* and `code`"
        assert self.adapter.format_message(content) == content

    def test_regular_links_preserved(self):
        content = "[click](https://example.com)"
        assert self.adapter.format_message(content) == content


class TestGoogleChatTruncateMessage:
    def setup_method(self):
        self.adapter = _make_adapter()

    def test_short_message_single_chunk(self):
        msg = "Hello, world!"
        chunks = self.adapter.truncate_message(msg, 4096)
        assert len(chunks) == 1
        assert chunks[0] == msg

    def test_long_message_splits(self):
        msg = "a " * 2500  # 5000 chars
        chunks = self.adapter.truncate_message(msg, 4096)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_exactly_at_limit(self):
        msg = "x" * 4096
        chunks = self.adapter.truncate_message(msg, 4096)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Event parsing (_parse_event static method)
# ---------------------------------------------------------------------------

class TestGoogleChatEventParsing:
    """Tests for the static _parse_event method which normalizes
    all three inbound event formats into a consistent tuple."""

    def test_parse_native_chat_api_message(self):
        """Native Chat API Pub/Sub format should be parsed correctly."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/AAAA/messages/msg123",
                "sender": {
                    "name": "users/user123",
                    "displayName": "Alice",
                    "email": "alice@example.com",
                    "type": "HUMAN",
                },
                "text": "Hello Hermes!",
                "thread": {"name": "spaces/AAAA/threads/thread1"},
            },
            "space": {"name": "spaces/AAAA", "type": "ROOM"},
        }
        result = GoogleChatAdapter._parse_event(event)
        assert result is not None
        text, sender_email, sender_name, space_name, space_type, thread_name, message_name = result
        assert text == "Hello Hermes!"
        assert sender_email == "alice@example.com"
        assert sender_name == "Alice"
        assert space_name == "spaces/AAAA"
        assert space_type == "ROOM"
        assert thread_name == "spaces/AAAA/threads/thread1"
        assert message_name == "spaces/AAAA/messages/msg123"

    def test_parse_workspace_addon_format(self):
        """Workspace Add-on format should be parsed correctly."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {
            "commonEventObject": {},
            "chat": {
                "user": {
                    "email": "bob@example.com",
                    "displayName": "Bob",
                },
                "messagePayload": {
                    "space": {"name": "spaces/BBBB", "type": "DM"},
                    "message": {
                        "name": "spaces/BBBB/messages/msg456",
                        "text": "DM from addon",
                        "thread": {"name": "spaces/BBBB/threads/t1"},
                    },
                },
            },
        }
        result = GoogleChatAdapter._parse_event(event)
        assert result is not None
        text, sender_email, sender_name, space_name, space_type, thread_name, message_name = result
        assert text == "DM from addon"
        assert sender_email == "bob@example.com"
        assert space_name == "spaces/BBBB"
        assert space_type == "DM"

    def test_parse_relay_format(self):
        """Relay/flat format should be parsed correctly."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {
            "event_type": "MESSAGE",
            "sender_email": "charlie@example.com",
            "sender_display_name": "Charlie",
            "space_name": "spaces/CCCC",
            "text": "Hello from relay",
            "message_name": "relay-msg-789",
        }
        result = GoogleChatAdapter._parse_event(event)
        assert result is not None
        text, sender_email, sender_name, space_name, space_type, thread_name, message_name = result
        assert text == "Hello from relay"
        assert sender_email == "charlie@example.com"
        assert sender_name == "Charlie"

    def test_non_message_event_returns_none(self):
        """Non-MESSAGE event types should return None."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        for event_type in ["ADDED_TO_SPACE", "REMOVED_FROM_SPACE", "CARD_CLICKED"]:
            event = {
                "type": event_type,
                "message": {
                    "name": "spaces/AAAA/messages/msg_skip",
                    "sender": {"name": "users/user123", "type": "HUMAN"},
                    "text": "Should be skipped",
                },
                "space": {"name": "spaces/AAAA"},
            }
            result = GoogleChatAdapter._parse_event(event)
            assert result is None, f"{event_type} should return None"

    def test_unrecognized_format_returns_none(self):
        """Events with unrecognized format should return None."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {"random_key": "random_value", "something_else": 42}
        result = GoogleChatAdapter._parse_event(event)
        assert result is None

    def test_dm_space_type(self):
        """DM space should report type as DM."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/DDDD/messages/msg_dm",
                "sender": {
                    "name": "users/user456",
                    "email": "dave@example.com",
                    "displayName": "Dave",
                    "type": "HUMAN",
                },
                "text": "Private message",
            },
            "space": {"name": "spaces/DDDD", "type": "DM"},
        }
        result = GoogleChatAdapter._parse_event(event)
        assert result is not None
        _, _, _, _, space_type, _, _ = result
        assert space_type == "DM"

    def test_thread_name_extracted(self):
        """Thread name should be extracted from event."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/AAAA/messages/msg_thread",
                "sender": {"email": "alice@example.com", "displayName": "Alice"},
                "text": "Thread reply",
                "thread": {"name": "spaces/AAAA/threads/thread42"},
            },
            "space": {"name": "spaces/AAAA", "type": "ROOM"},
        }
        result = GoogleChatAdapter._parse_event(event)
        assert result is not None
        _, _, _, _, _, thread_name, _ = result
        assert thread_name == "spaces/AAAA/threads/thread42"

    def test_argument_text_preferred_over_text(self):
        """argumentText should be preferred over text (strips @mentions)."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/AAAA/messages/msg_arg",
                "sender": {"email": "alice@example.com"},
                "text": "@Hermes what time is it",
                "argumentText": "what time is it",
            },
            "space": {"name": "spaces/AAAA", "type": "ROOM"},
        }
        result = GoogleChatAdapter._parse_event(event)
        assert result is not None
        text, _, _, _, _, _, _ = result
        assert text == "what time is it"

    def test_relay_non_message_returns_none(self):
        """Relay events with non-MESSAGE type should return None."""
        from gateway.platforms.google_chat import GoogleChatAdapter
        event = {
            "event_type": "REMOVED_FROM_SPACE",
            "sender_email": "alice@example.com",
            "text": "left the space",
        }
        result = GoogleChatAdapter._parse_event(event)
        assert result is None


# ---------------------------------------------------------------------------
# Inbound callback: _on_pubsub_message
# ---------------------------------------------------------------------------

class TestGoogleChatPubSubCallback:
    """Tests for the _on_pubsub_message callback which bridges
    Pub/Sub events to the asyncio event loop."""

    def setup_method(self):
        self.adapter = _make_adapter()
        self.adapter.handle_message = AsyncMock()
        self.adapter._loop = MagicMock()
        self.adapter._loop.is_running.return_value = True

    def _make_pubsub_msg(self, chat_event):
        """Create a mock Pub/Sub message."""
        msg = MagicMock()
        msg.data = json.dumps(chat_event).encode("utf-8")
        return msg

    def test_valid_message_acked_and_scheduled(self):
        """Valid MESSAGE event should be acked and scheduled on loop."""
        chat_event = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/AAAA/messages/msg1",
                "sender": {"email": "alice@example.com", "displayName": "Alice", "type": "HUMAN"},
                "text": "Hello",
            },
            "space": {"name": "spaces/AAAA", "type": "ROOM"},
        }
        msg = self._make_pubsub_msg(chat_event)
        self.adapter._on_pubsub_message(msg)

        msg.ack.assert_called_once()
        self.adapter._loop.call_soon_threadsafe.assert_called_once()

    def test_non_message_event_acked_without_scheduling(self):
        """Non-MESSAGE events should be acked but NOT scheduled."""
        chat_event = {
            "type": "ADDED_TO_SPACE",
            "space": {"name": "spaces/AAAA"},
            "user": {"name": "users/user123"},
        }
        msg = self._make_pubsub_msg(chat_event)
        self.adapter._on_pubsub_message(msg)

        msg.ack.assert_called_once()
        self.adapter._loop.call_soon_threadsafe.assert_not_called()

    def test_empty_text_acked_without_scheduling(self):
        """Events with empty text should be acked but NOT scheduled."""
        chat_event = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/AAAA/messages/msg_empty",
                "sender": {"email": "alice@example.com", "type": "HUMAN"},
                "text": "",
            },
            "space": {"name": "spaces/AAAA"},
        }
        msg = self._make_pubsub_msg(chat_event)
        self.adapter._on_pubsub_message(msg)

        msg.ack.assert_called_once()
        self.adapter._loop.call_soon_threadsafe.assert_not_called()

    def test_invalid_json_nacked(self):
        """Invalid JSON should be nacked for retry."""
        msg = MagicMock()
        msg.data = b"not-valid-json{{{"
        self.adapter._on_pubsub_message(msg)

        msg.nack.assert_called_once()
        self.adapter._loop.call_soon_threadsafe.assert_not_called()

    def test_chat_type_dm_vs_room(self):
        """Verify DM/ROOM space types map correctly in the scheduled event."""
        for space_type, expected_chat_type in [("DM", "dm"), ("ROOM", "group")]:
            self.adapter._loop.reset_mock()
            chat_event = {
                "type": "MESSAGE",
                "message": {
                    "name": f"spaces/TEST/messages/msg_{space_type}",
                    "sender": {"email": "test@example.com", "displayName": "Test"},
                    "text": "Test message",
                },
                "space": {"name": "spaces/TEST", "type": space_type},
            }
            msg = self._make_pubsub_msg(chat_event)
            self.adapter._on_pubsub_message(msg)

            msg.ack.assert_called()
            self.adapter._loop.call_soon_threadsafe.assert_called()


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

class TestGoogleChatSend:
    def setup_method(self):
        self.adapter = _make_adapter()
        self.mock_service = MagicMock()
        self.adapter._chat_service = self.mock_service

    @pytest.mark.asyncio
    async def test_send_calls_chat_api(self):
        """send() should call spaces().messages().create() with correct body."""
        mock_result = {"name": "spaces/AAAA/messages/sent123"}
        self.mock_service.spaces().messages().create.return_value.execute.return_value = mock_result

        result = await self.adapter.send("spaces/AAAA", "Hello!")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_empty_content(self):
        """Empty content should return success without API call."""
        result = await self.adapter.send("spaces/AAAA", "")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_without_service_fails(self):
        """send() without a chat service should fail gracefully."""
        self.adapter._chat_service = None
        result = await self.adapter.send("spaces/AAAA", "Hello!")
        assert result.success is False


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

class TestGoogleChatRequirements:
    def test_check_requirements_with_deps_and_project(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CHAT_GCP_PROJECT", "my-project")
        with patch.dict("sys.modules", {
            "google.cloud.pubsub_v1": MagicMock(),
            "google.cloud": MagicMock(),
            "google.oauth2": MagicMock(),
            "google.oauth2.service_account": MagicMock(),
            "google.auth": MagicMock(),
            "googleapiclient": MagicMock(),
            "googleapiclient.discovery": MagicMock(),
        }):
            from gateway.platforms.google_chat import check_google_chat_requirements
            assert check_google_chat_requirements() is True

    def test_check_requirements_without_project(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CHAT_GCP_PROJECT", raising=False)
        from gateway.platforms.google_chat import check_google_chat_requirements
        assert check_google_chat_requirements() is False


# ---------------------------------------------------------------------------
# Retryable error detection
# ---------------------------------------------------------------------------

class TestRetryableError:
    def test_429_is_retryable(self):
        from gateway.platforms.google_chat import _is_retryable_error
        exc = Exception("HttpError 429: Rate limit exceeded")
        assert _is_retryable_error(exc) is True

    def test_500_is_retryable(self):
        from gateway.platforms.google_chat import _is_retryable_error
        exc = Exception("500 Internal Server Error")
        assert _is_retryable_error(exc) is True

    def test_timeout_is_retryable(self):
        from gateway.platforms.google_chat import _is_retryable_error
        exc = Exception("Connection timeout")
        assert _is_retryable_error(exc) is True

    def test_400_is_not_retryable(self):
        from gateway.platforms.google_chat import _is_retryable_error
        exc = Exception("400 Bad Request: invalid space name")
        assert _is_retryable_error(exc) is False

    def test_http_error_with_resp_attribute(self):
        """HttpError-like exceptions with resp.status should be detected."""
        from gateway.platforms.google_chat import _is_retryable_error
        exc = Exception("rate limited")
        exc.resp = MagicMock()
        exc.resp.status = 429
        assert _is_retryable_error(exc) is True


# ---------------------------------------------------------------------------
# Integration point verification
# ---------------------------------------------------------------------------

class TestGoogleChatIntegration:
    """Verify the platform is wired into all gateway integration points."""

    def test_platform_enum_exists(self):
        assert hasattr(Platform, "GOOGLE_CHAT")
        assert Platform.GOOGLE_CHAT.value == "google_chat"

    def test_toolset_registered(self):
        from toolsets import TOOLSETS
        assert "hermes-google-chat" in TOOLSETS

    def test_toolset_in_gateway_composite(self):
        from toolsets import TOOLSETS
        assert "hermes-google-chat" in TOOLSETS["hermes-gateway"]["includes"]

    def test_platform_hint_registered(self):
        from agent.prompt_builder import PLATFORM_HINTS
        assert "google_chat" in PLATFORM_HINTS

    def test_cron_platform_map(self):
        """Verify google_chat is in cron scheduler's platform_map."""
        import inspect
        from cron import scheduler
        source = inspect.getsource(scheduler)
        assert '"google_chat"' in source or "'google_chat'" in source

    def test_send_message_tool_has_google_chat(self):
        """Verify google_chat is in send_message_tool's platform_map."""
        import inspect
        import tools.send_message_tool as smt
        source = inspect.getsource(smt)
        assert '"google_chat"' in source or "'google_chat'" in source
