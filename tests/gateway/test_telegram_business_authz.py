"""Tests for Telegram Business connection intake and authorization in Gateway.

Verifies end-to-end integration of telegram_business_connection_id from
Telegram Bot API Message objects through TelegramAdapter, BasePlatformAdapter,
SessionSource, and GatewayAuthorizationMixin (BDD-1 .. BDD-4).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


class MockGateway(GatewayAuthorizationMixin):
    def __init__(self):
        self._bot_loop_guard = MagicMock()
        self._bot_loop_guard.admit.return_value = (True, "admitted")
        self._bot_loop_guard.blocked.return_value = False
        self.adapters = {}

    def _bot_loop_guard_instance(self):
        return self._bot_loop_guard

    def _bot_loop_guard_conversation(self, source):
        return ("telegram", source.chat_id)

    def _adapter_profile_for_source(self, source):
        return None

    def _chat_scoped_grant(self, source, profile, is_group, allow_delegation):
        return False

    def _pairing_store_for(self, source):
        return None

    def _authorization_adapter(self, platform, profile=None):
        return None

    def _delivery_adapter_for(self, source):
        return None


def _make_adapter(allow_from=None, callback_auth=None, **extra_overrides):
    extra = {}
    if allow_from is not None:
        extra["allow_from"] = allow_from
    extra.update(extra_overrides)

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter.gateway_runner = None
    if callback_auth is not None:
        adapter._authorization_check = callback_auth
    return adapter


def _make_message(text="hello", *, from_user_id=111, chat_id=12345, chat_type="private", business_connection_id=None):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type=chat_type, title="Test Chat", full_name="Customer User", is_forum=False),
        from_user=SimpleNamespace(id=from_user_id, full_name="Customer User", first_name="Customer", is_bot=False),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
        business_connection_id=business_connection_id,
    )


def test_bdd1_telegram_business_intake_e2e_authorized():
    """BDD-1: Incoming message from business connection is authorized end-to-end without setattr."""
    gw = MockGateway()
    adapter = _make_adapter(
        allow_from=["111222333"],  # Owner allowlist; customer 999 is NOT in allow_from
        callback_auth=lambda uid, chat_type=None, chat_id=None, **extra: gw._is_user_authorized(
            SessionSource(platform=Platform.TELEGRAM, user_id=uid, chat_type=chat_type or "dm", chat_id=chat_id or "", **extra)
        ),
    )

    msg = _make_message(
        text="Hello, how much does the consultation cost?",
        from_user_id=999,
        chat_id=12345,
        chat_type="private",
        business_connection_id="biz_conn_prod_123",
    )

    # 1. Early auth source extraction
    source_auth = adapter._source_from_message_for_auth(msg)
    assert source_auth.telegram_business_connection_id == "biz_conn_prod_123"
    assert source_auth.platform == Platform.TELEGRAM
    assert source_auth.user_id == "999"

    # 2. Event building with build_source
    event = adapter._build_message_event(msg, MessageType.TEXT)
    assert event.source.telegram_business_connection_id == "biz_conn_prod_123"
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.user_id == "999"

    # 3. Gateway authorization verdict
    assert gw._is_user_authorized(source_auth) is True
    assert gw._is_user_authorized(event.source) is True

    # 4. Adapter prefilter allows message to proceed
    assert adapter._is_user_authorized_from_message(msg) is True


def test_bdd2_negative_control_direct_message_unauthorized():
    """BDD-2: Direct message without business_connection_id from unlisted user is rejected."""
    gw = MockGateway()
    adapter = _make_adapter(
        allow_from=["111222333"],
        callback_auth=lambda uid, chat_type=None, chat_id=None, **extra: gw._is_user_authorized(
            SessionSource(platform=Platform.TELEGRAM, user_id=uid, chat_type=chat_type or "dm", chat_id=chat_id or "", **extra)
        ),
    )

    msg = _make_message(
        text="Direct stranger message",
        from_user_id=777,
        chat_id=12345,
        chat_type="private",
        business_connection_id=None,
    )

    source_auth = adapter._source_from_message_for_auth(msg)
    assert source_auth.telegram_business_connection_id is None

    event = adapter._build_message_event(msg, MessageType.TEXT)
    assert event.source.telegram_business_connection_id is None

    assert gw._is_user_authorized(source_auth) is False
    assert gw._is_user_authorized(event.source) is False
    assert adapter._is_user_authorized_from_message(msg) is False


def test_bdd3_anti_spoofing_platform_isolation():
    """BDD-3: Non-Telegram platforms cannot activate telegram business auth."""
    gw = MockGateway()

    # Discord spoofing attempt
    discord_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="discord_chan_123",
        user_id="customer_999",
        telegram_business_connection_id="biz_conn_prod_123",
    )
    assert gw._is_user_authorized(discord_source) is False

    # Slack spoofing attempt
    slack_source = SessionSource(
        platform=Platform.SLACK,
        chat_id="slack_chan_123",
        user_id="customer_999",
        telegram_business_connection_id="biz_conn_prod_123",
    )
    assert gw._is_user_authorized(slack_source) is False


@pytest.mark.parametrize(
    "invalid_conn_id",
    [
        12345,  # int
        True,  # bool
        "",  # empty string
        "   ",  # whitespace string
        None,  # None
    ],
)
def test_bdd3_anti_spoofing_invalid_connection_id(invalid_conn_id):
    """BDD-3: Invalid, non-string, or empty connection IDs are rejected."""
    gw = MockGateway()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        user_id="customer_999",
        telegram_business_connection_id=invalid_conn_id,  # type: ignore[arg-type]
    )
    assert gw._is_user_authorized(source) is False


def test_bdd4_session_source_serialization_roundtrip():
    """BDD-4: SessionSource preserves telegram_business_connection_id across to_dict and from_dict."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        user_id="customer_999",
        telegram_business_connection_id="biz_conn_active_123",
    )

    data = source.to_dict()
    assert "telegram_business_connection_id" in data
    assert data["telegram_business_connection_id"] == "biz_conn_active_123"

    restored = SessionSource.from_dict(data)
    assert restored.telegram_business_connection_id == "biz_conn_active_123"
    assert restored == source

    # Backward compatibility with legacy stored sessions without the field
    legacy_data = {
        "platform": "telegram",
        "chat_id": "12345",
        "chat_name": "Test Chat",
        "chat_type": "dm",
        "user_id": "legacy_user",
        "user_name": "Legacy",
        "thread_id": None,
        "chat_topic": None,
    }
    restored_legacy = SessionSource.from_dict(legacy_data)
    assert restored_legacy.telegram_business_connection_id is None

    legacy_out = restored_legacy.to_dict()
    assert "telegram_business_connection_id" not in legacy_out


def test_effective_update_message_and_build_source():
    """Verify _effective_update_message supports business_message and build_source forwards the attribute."""
    adapter = _make_adapter()

    # 1. build_source forwarding
    built = adapter.build_source(
        chat_id="12345",
        user_id="cust_1",
        telegram_business_connection_id="biz_direct_789",
    )
    assert built.telegram_business_connection_id == "biz_direct_789"

    # 2. _effective_update_message resolution
    biz_msg = _make_message(business_connection_id="biz_msg_123")
    update_biz = SimpleNamespace(
        update_id=10,
        effective_message=None,
        business_message=biz_msg,
        message=None,
    )
    assert adapter._effective_update_message(update_biz) is biz_msg
