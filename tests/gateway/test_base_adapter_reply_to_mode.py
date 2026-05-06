"""Tests for BasePlatformAdapter._resolve_reply_to — gateway-level reply threading.

Covers the reply_to_mode config honouring at the base adapter layer,
which orchestrates gateway response sends for ALL platforms (not just
Discord, which has its own adapter-level implementation).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent


# ---------------------------------------------------------------------------
# Minimal concrete subclass — BasePlatformAdapter is ABC
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    """Trivially instantiable subclass for testing base-class helpers."""

    def __init__(self, config=None, platform=Platform.DISCORD):
        cfg = config or PlatformConfig(enabled=True, token="t")
        super().__init__(config=cfg, platform=platform)

    # Satisfy remaining abstract methods (not exercised in these tests)
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kw):
        pass

    async def _send_with_retry(self, *, chat_id, content, reply_to=None, metadata=None):
        return MagicMock(success=True, message_id="fake")

    async def get_chat_info(self, chat_id: str):
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(*, platform=Platform.DISCORD, message_id="123", thread_id=None,
                reply_to_message_id=None):
    """Build a minimal MessageEvent for _resolve_reply_to tests."""
    source = SimpleNamespace(platform=platform, chat_id="C1", thread_id=thread_id)
    return MessageEvent(
        source=source,
        text="hello",
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResolveReplyTo:
    """Tests for _resolve_reply_to respecting reply_to_mode."""

    def test_off_mode_returns_none(self):
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="off"))
        event = _make_event(message_id="100")
        assert adapter._resolve_reply_to(event) is None

    def test_off_mode_ignores_feishu_thread_reply(self):
        """Even in a Feishu thread with reply_to_message_id, 'off' wins."""
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="off"))
        event = _make_event(
            platform=Platform.FEISHU, message_id="100",
            thread_id="T1", reply_to_message_id="99",
        )
        assert adapter._resolve_reply_to(event) is None

    def test_first_mode_returns_message_id(self):
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="first"))
        event = _make_event(message_id="100")
        assert adapter._resolve_reply_to(event) == "100"

    def test_all_mode_returns_message_id(self):
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="all"))
        event = _make_event(message_id="100")
        assert adapter._resolve_reply_to(event) == "100"

    def test_default_mode_is_first(self):
        """No reply_to_mode set → defaults to 'first' → returns message_id."""
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"))
        event = _make_event(message_id="100")
        assert adapter._resolve_reply_to(event) == "100"

    def test_feishu_thread_returns_reply_to_message_id(self):
        """Feishu threads should use reply_to_message_id when available."""
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="first"))
        event = _make_event(
            platform=Platform.FEISHU, message_id="100",
            thread_id="T1", reply_to_message_id="99",
        )
        assert adapter._resolve_reply_to(event) == "99"

    def test_feishu_non_thread_returns_message_id(self):
        """Feishu without thread_id falls through to message_id."""
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="first"))
        event = _make_event(
            platform=Platform.FEISHU, message_id="100",
            thread_id=None, reply_to_message_id="99",
        )
        assert adapter._resolve_reply_to(event) == "100"

    def test_feishu_thread_no_reply_to_returns_message_id(self):
        """Feishu thread without reply_to_message_id falls back to message_id."""
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="first"))
        event = _make_event(
            platform=Platform.FEISHU, message_id="100",
            thread_id="T1", reply_to_message_id=None,
        )
        assert adapter._resolve_reply_to(event) == "100"

    def test_discord_ignores_reply_to_message_id(self):
        """Non-Feishu platforms always return message_id, ignoring reply_to_message_id."""
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="first"))
        event = _make_event(
            platform=Platform.DISCORD, message_id="100",
            reply_to_message_id="99",
        )
        assert adapter._resolve_reply_to(event) == "100"

    def test_telegram_ignores_reply_to_message_id(self):
        """Telegram (non-Feishu) returns message_id."""
        adapter = _StubAdapter(PlatformConfig(enabled=True, token="t", reply_to_mode="first"))
        event = _make_event(
            platform=Platform.TELEGRAM, message_id="100",
            reply_to_message_id="99",
        )
        assert adapter._resolve_reply_to(event) == "100"
