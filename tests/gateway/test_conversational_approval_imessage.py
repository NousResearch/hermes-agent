"""Conversational approval on button-less texting surfaces (iMessage).

Two contracts:

1. The prompt a texting user sees asks for a plain-language reply, not
   ``/approve``. Slash commands still work; they are not what we advertise.
2. "sure, go ahead" typed into the thread resolves the real pending approval
   through the canonical handler, and a refusal denies it — driven through
   ``_handle_active_session_busy_message`` (the production path), with a real
   ``_ApprovalEntry`` blocking, not a mocked matcher.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


class TestConversationalPromptWording:
    """A text-message approval prompt must read like a question to a person."""

    def _fallback(self, **kwargs):
        from gateway.run import _format_exec_approval_fallback

        return _format_exec_approval_fallback(
            "rm -rf /tmp/cache", "deletes a directory tree", "/", **kwargs
        )

    def test_conversational_prompt_asks_for_plain_reply(self):
        text = self._fallback(conversational=True)
        assert '"yes"' in text
        assert '"no"' in text
        # The whole point: no slash-command instructions in a text message.
        assert "/approve" not in text
        assert "/deny" not in text

    def test_conversational_prompt_still_shows_command_and_reason(self):
        text = self._fallback(conversational=True)
        assert "rm -rf /tmp/cache" in text
        assert "deletes a directory tree" in text

    def test_conversational_prompt_offers_the_scopes_it_has(self):
        text = self._fallback(conversational=True, allow_session=True, allow_permanent=True)
        assert "session" in text.lower()
        assert "always" in text.lower()

    def test_conversational_smart_deny_offers_only_one_operation(self):
        text = self._fallback(conversational=True, smart_denied=True, allow_permanent=False)
        assert '"yes"' in text
        assert "always" not in text.lower()
        assert "session" not in text.lower()

    def test_slash_platforms_keep_slash_wording(self):
        """Button/slash platforms must not regress into conversational wording."""
        text = self._fallback(conversational=False)
        assert "`/approve`" in text
        assert "/deny" in text


class TestImessageAdaptersOptIn:
    def test_photon_declares_conversational_approval(self):
        from gateway.config import PlatformConfig as PC
        from plugins.platforms.photon.adapter import PhotonAdapter

        adapter = PhotonAdapter(PC(enabled=True, token="", extra={}))
        assert adapter.conversational_approval is True

    def test_base_adapters_default_to_slash_wording(self):
        from gateway.platforms.base import BasePlatformAdapter

        assert BasePlatformAdapter.conversational_approval is False


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform("photon"), user_id="+15550001111", chat_id="space-1",
        user_name="tester", chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=_make_source(), message_id="m1",
    )


def _clear_approval_state():
    from tools import approval as mod

    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


def _make_runner():
    """Minimal GatewayRunner exercising the real busy-session handler."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform("photon"): PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="reply1")
    )
    adapter._unwrap_ephemeral = lambda r: (r, 0) if isinstance(r, str) else (None, 0)
    runner.adapters = {Platform("photon"): adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.session_store = None
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    return runner, adapter


def _register_blocking_approval(runner):
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry

    session_key = runner._session_key_for_source(_make_source())
    entry = _ApprovalEntry({"command": "rm -rf /tmp/cache"})
    _gateway_queues.setdefault(session_key, []).append(entry)
    return session_key, entry


class TestConversationalRepliesResolveRealApprovals:
    @pytest.mark.parametrize("reply", [
        "sure, go ahead", "yes please", "do it", "ok go ahead", "sounds good",
    ])
    def test_affirmative_approves_once(self, reply):
        _clear_approval_state()
        try:
            runner, adapter = _make_runner()
            session_key, entry = _register_blocking_approval(runner)

            handled = asyncio.run(
                runner._handle_active_session_busy_message(_make_event(reply), session_key)
            )

            assert handled is True
            assert entry.event.is_set()
            assert entry.result == "once"
            adapter._send_with_retry.assert_awaited()
        finally:
            _clear_approval_state()

    @pytest.mark.parametrize("reply", ["no, don't go ahead", "don't do it", "never mind"])
    def test_refusal_denies(self, reply):
        _clear_approval_state()
        try:
            runner, _adapter = _make_runner()
            session_key, entry = _register_blocking_approval(runner)

            handled = asyncio.run(
                runner._handle_active_session_busy_message(_make_event(reply), session_key)
            )

            assert handled is True
            assert entry.event.is_set()
            assert entry.result == "deny"
        finally:
            _clear_approval_state()

    def test_session_scope_reply_approves_for_session(self):
        _clear_approval_state()
        try:
            runner, _adapter = _make_runner()
            session_key, entry = _register_blocking_approval(runner)

            handled = asyncio.run(
                runner._handle_active_session_busy_message(
                    _make_event("for this session"), session_key
                )
            )

            assert handled is True
            assert entry.result == "session"
        finally:
            _clear_approval_state()

    @pytest.mark.parametrize("reply", [
        "are you sure?",
        "yes but not the second one",
        "go ahead and also delete the backups",
    ])
    def test_ambiguous_reply_leaves_approval_pending(self, reply):
        """Fail-safe: an unclear answer must not resolve a flagged command."""
        _clear_approval_state()
        try:
            runner, _adapter = _make_runner()
            session_key, entry = _register_blocking_approval(runner)

            asyncio.run(
                runner._handle_active_session_busy_message(_make_event(reply), session_key)
            )

            assert not entry.event.is_set()
            assert entry.result is None
        finally:
            _clear_approval_state()

    def test_conversational_yes_with_no_pending_approval_is_not_consumed(self):
        """The context gate: ordinary chat must never fire a command."""
        _clear_approval_state()
        try:
            runner, _adapter = _make_runner()
            session_key = runner._session_key_for_source(_make_source())

            asyncio.run(
                runner._handle_active_session_busy_message(
                    _make_event("sure, go ahead"), session_key
                )
            )

            from tools.approval import _gateway_queues

            assert session_key not in _gateway_queues
        finally:
            _clear_approval_state()
