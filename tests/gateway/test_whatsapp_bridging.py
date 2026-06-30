"""Tests for WhatsApp self-send bridging.

Covers:
- WhatsAppAdapter.send_reaction(): POSTs to the bridge's /reaction endpoint
- GatewayRunner._whatsapp_bridge_target(): detects self-chat messages and
  resolves the WhatsApp/Telegram adapters + Telegram home channel chat_id
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_wa_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._bridge_port = 3000
    adapter._bridge_process = None
    adapter._running = True
    adapter._http_session = MagicMock()
    return adapter


def _self_chat_source(**overrides) -> SessionSource:
    base = dict(
        platform=Platform.WHATSAPP,
        chat_id="123456@s.whatsapp.net",
        chat_type="dm",
        user_id="123456@s.whatsapp.net",
        user_name="Ezra",
    )
    base.update(overrides)
    return SessionSource(**base)


def _make_runner(*, whatsapp_bridging=True, with_telegram_home=True, with_adapters=True):
    runner = object.__new__(GatewayRunner)
    platforms = {
        Platform.WHATSAPP: PlatformConfig(
            enabled=True,
            extra={"whatsapp_bridging": whatsapp_bridging},
        ),
    }
    if with_telegram_home:
        platforms[Platform.TELEGRAM] = PlatformConfig(
            enabled=True,
            home_channel=HomeChannel(
                platform=Platform.TELEGRAM, chat_id="7495977859", name="Home",
            ),
        )
    runner.config = GatewayConfig(platforms=platforms)
    runner.adapters = {}
    if with_adapters:
        runner.adapters[Platform.WHATSAPP] = MagicMock()
        if with_telegram_home:
            runner.adapters[Platform.TELEGRAM] = MagicMock()
    return runner


# ---------------------------------------------------------------------------
# WhatsAppAdapter.send_reaction
# ---------------------------------------------------------------------------

class TestSendReaction:
    @pytest.mark.asyncio
    async def test_posts_to_reaction_endpoint(self):
        adapter = _make_wa_adapter()
        resp = MagicMock(status=200)
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        result = await adapter.send_reaction("123@s.whatsapp.net", "ABCD1234", "\U0001F576")

        assert result is True
        adapter._http_session.post.assert_called_once()
        args, kwargs = adapter._http_session.post.call_args
        assert args[0] == "http://127.0.0.1:3000/reaction"
        assert kwargs["json"] == {
            "chatId": "123@s.whatsapp.net",
            "messageId": "ABCD1234",
            "reaction": "\U0001F576",
        }

    @pytest.mark.asyncio
    async def test_returns_false_on_error_status(self):
        adapter = _make_wa_adapter()
        resp = MagicMock(status=500)
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        result = await adapter.send_reaction("123@s.whatsapp.net", "ABCD1234", "\U0001F44D")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_message_id(self):
        adapter = _make_wa_adapter()
        adapter._http_session.post = MagicMock()

        result = await adapter.send_reaction("123@s.whatsapp.net", None, "\U0001F44D")

        assert result is False
        adapter._http_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_false_when_not_connected(self):
        adapter = _make_wa_adapter()
        adapter._running = False

        result = await adapter.send_reaction("123@s.whatsapp.net", "ABCD1234", "\U0001F44D")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        adapter = _make_wa_adapter()
        adapter._http_session.post = MagicMock(side_effect=RuntimeError("boom"))

        result = await adapter.send_reaction("123@s.whatsapp.net", "ABCD1234", "\U0001F44D")

        assert result is False


# ---------------------------------------------------------------------------
# GatewayRunner._whatsapp_bridge_target
# ---------------------------------------------------------------------------

class TestWhatsAppBridgeTarget:
    def test_self_chat_with_bridging_enabled_returns_targets(self):
        runner = _make_runner()
        source = _self_chat_source()

        result = runner._whatsapp_bridge_target(source)

        assert result is not None
        wa_adapter, tg_adapter, tg_chat_id = result
        assert wa_adapter is runner.adapters[Platform.WHATSAPP]
        assert tg_adapter is runner.adapters[Platform.TELEGRAM]
        assert tg_chat_id == "7495977859"

    def test_non_whatsapp_platform_returns_none(self):
        runner = _make_runner()
        source = _self_chat_source(platform=Platform.TELEGRAM)

        assert runner._whatsapp_bridge_target(source) is None

    def test_non_self_chat_returns_none(self):
        runner = _make_runner()
        source = _self_chat_source(chat_id="other@s.whatsapp.net")

        assert runner._whatsapp_bridge_target(source) is None

    def test_bridging_disabled_returns_none(self):
        runner = _make_runner(whatsapp_bridging=False)
        source = _self_chat_source()

        assert runner._whatsapp_bridge_target(source) is None

    def test_no_telegram_home_channel_returns_none(self):
        runner = _make_runner(with_telegram_home=False)
        source = _self_chat_source()

        assert runner._whatsapp_bridge_target(source) is None

    def test_missing_adapters_returns_none(self):
        runner = _make_runner(with_adapters=False)
        source = _self_chat_source()

        assert runner._whatsapp_bridge_target(source) is None


# ---------------------------------------------------------------------------
# /approve and /deny resolution for bridged approval prompts
# ---------------------------------------------------------------------------
#
# When a bridged WhatsApp session's approval prompt is redirected to the
# Telegram home channel, /approve and /deny typed there resolve to the
# *Telegram* session's key — which doesn't match the WhatsApp session_key
# the approval is queued under in tools.approval._gateway_queues.
# _wa_bridge_approval_redirects (populated by _run_agent) bridges that gap.

class TestBridgedApprovalRedirect:
    def setup_method(self):
        from tools import approval as mod
        mod._gateway_queues.clear()
        mod._gateway_notify_cbs.clear()

    def _telegram_home_source(self, runner) -> SessionSource:
        tg_home = runner.config.get_home_channel(Platform.TELEGRAM)
        return SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=tg_home.chat_id,
            chat_type="dm",
        )

    @pytest.mark.asyncio
    async def test_approve_in_telegram_resolves_bridged_whatsapp_approval(self):
        from gateway.platforms.base import MessageEvent
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        runner._pending_approvals = {}

        wa_session_key = runner._session_key_for_source(_self_chat_source())
        tg_source = self._telegram_home_source(runner)
        tg_session_key = runner._session_key_for_source(tg_source)
        runner._wa_bridge_approval_redirects = {tg_session_key: wa_session_key}

        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues[wa_session_key] = [entry]

        event = MessageEvent(text="/approve", source=tg_source, message_id="m1")
        result = await runner._handle_approve_command(event)

        assert "approved" in result.lower()
        assert entry.event.is_set()
        assert entry.result == "once"
        assert wa_session_key not in _gateway_queues

    @pytest.mark.asyncio
    async def test_deny_in_telegram_resolves_bridged_whatsapp_approval(self):
        from gateway.platforms.base import MessageEvent
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        runner._pending_approvals = {}

        wa_session_key = runner._session_key_for_source(_self_chat_source())
        tg_source = self._telegram_home_source(runner)
        tg_session_key = runner._session_key_for_source(tg_source)
        runner._wa_bridge_approval_redirects = {tg_session_key: wa_session_key}

        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues[wa_session_key] = [entry]

        event = MessageEvent(text="/deny", source=tg_source, message_id="m1")
        result = await runner._handle_deny_command(event)

        assert "denied" in result.lower()
        assert entry.event.is_set()
        assert entry.result == "deny"
        assert wa_session_key not in _gateway_queues

    @pytest.mark.asyncio
    async def test_approve_in_telegram_without_redirect_reports_no_pending(self):
        """No bridge redirect registered — falls back to the normal message."""
        from gateway.platforms.base import MessageEvent

        runner = _make_runner()
        runner._pending_approvals = {}
        runner._wa_bridge_approval_redirects = {}

        tg_source = self._telegram_home_source(runner)
        event = MessageEvent(text="/approve", source=tg_source, message_id="m1")
        result = await runner._handle_approve_command(event)

        assert "no pending" in result.lower()
