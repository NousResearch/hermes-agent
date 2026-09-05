"""Bounded grants over the gateway: ``/approve for 30m``, the conversational
"yes for the next hour", and ``/grants`` list / revoke. Driven through the real
busy-session handler and the real slash handlers with a real blocking
``_ApprovalEntry``, so the wiring (not just the parser) is what is tested.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from tools import approval_grants as grants


@pytest.fixture(autouse=True)
def _fresh():
    grants.reset_for_tests()
    _clear_approval_state()
    yield
    _clear_approval_state()
    grants.reset_for_tests()


def _make_source(chat_id: str = "c1") -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id=chat_id,
                         user_name="tester", chat_type="dm")


def _make_event(text: str, chat_id: str = "c1") -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_make_source(chat_id), message_id="m1")


def _clear_approval_state():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._send_with_retry = AsyncMock(return_value=SimpleNamespace(success=True, message_id="r1"))
    adapter._unwrap_ephemeral = lambda r: (r, 0) if isinstance(r, str) else (None, 0)
    runner.adapters = {Platform.TELEGRAM: adapter}
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


def _register_blocking_approval(runner, chat_id: str = "c1", pattern_key: str = "dangerous:rm"):
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry

    session_key = runner._session_key_for_source(_make_source(chat_id))
    entry = _ApprovalEntry({"command": "rm -rf /tmp/x", "pattern_key": pattern_key,
                            "pattern_keys": [pattern_key], "description": "recursive delete"})
    _gateway_queues.setdefault(session_key, []).append(entry)
    return session_key, entry


class TestSlashApproveWithGrant:
    @pytest.mark.parametrize("args", ["for 30m", "for 2 hours", "3 times", "for today"])
    def test_approve_with_scope_resolves_as_grant(self, args):
        runner, adapter = _make_runner()
        session_key, entry = _register_blocking_approval(runner)

        reply = asyncio.run(runner._handle_approve_command(_make_event(f"/approve {args}")))

        assert entry.event.is_set()
        assert entry.result == f"grant:{args}"
        assert "approved" in reply.lower()

    def test_always_beats_grant_wording(self):
        runner, _ = _make_runner()
        _, entry = _register_blocking_approval(runner)
        asyncio.run(runner._handle_approve_command(_make_event("/approve always for 30m")))
        assert entry.result == "always"

    def test_plain_approve_is_still_once(self):
        runner, _ = _make_runner()
        _, entry = _register_blocking_approval(runner)
        asyncio.run(runner._handle_approve_command(_make_event("/approve")))
        assert entry.result == "once"


class TestConversationalGrant:
    @pytest.mark.parametrize("reply,expected_scope", [
        ("yes for the next hour", "for the next hour"),
        ("ok for 30m", "for 30m"),
        ("sure, 3 times", "3 times"),
        ("yes for today", "for today"),
    ])
    def test_affirmative_plus_scope_grants(self, reply, expected_scope):
        runner, _ = _make_runner()
        session_key, entry = _register_blocking_approval(runner)

        handled = asyncio.run(runner._handle_active_session_busy_message(_make_event(reply), session_key))

        assert handled is True
        assert entry.result == f"grant:{expected_scope}"

    @pytest.mark.parametrize("reply", [
        "yes for an hour and also wipe the disk",   # trailing instruction
        "no, not for an hour",                       # refusal mentioning a duration
        "for an hour?",                              # question
        "can you do it for an hour",                 # not a bare affirmative lead
    ])
    def test_unsafe_shapes_leave_approval_pending(self, reply):
        runner, _ = _make_runner()
        session_key, entry = _register_blocking_approval(runner)

        asyncio.run(runner._handle_active_session_busy_message(_make_event(reply), session_key))

        assert not entry.event.is_set(), reply
        assert entry.result is None


class TestGrantIsHonoredNextTime:
    def test_second_flagged_command_skips_the_prompt(self):
        """The whole point: after ``/approve for 30m`` the next identical pattern in this
        chat is approved by ``is_approved`` without a new prompt."""
        from tools.approval import _persist_choice, is_approved

        runner, _ = _make_runner()
        session_key, _entry = _register_blocking_approval(runner)
        # What the gate does after the human choice resolves:
        _persist_choice(session_key, "grant:for 30m", [("dangerous:rm", "recursive delete", False)])

        assert is_approved(session_key, "dangerous:rm") is True
        other_session = runner._session_key_for_source(_make_source("c2"))
        assert is_approved(other_session, "dangerous:rm") is False


class TestGrantsCommand:
    def test_list_empty(self):
        runner, _ = _make_runner()
        reply = asyncio.run(runner._handle_grants_command(_make_event("/grants")))
        assert "no active" in reply.lower()

    def test_list_shows_grant_and_does_not_consume(self):
        runner, _ = _make_runner()
        session_key = runner._session_key_for_source(_make_source())
        grants.create(session_key, "dangerous:rm", "recursive delete", grants.GrantSpec(max_uses=1))

        reply = asyncio.run(runner._handle_grants_command(_make_event("/grants")))

        assert "recursive delete" in reply
        assert "1 use" in reply
        assert grants.consume(session_key, "dangerous:rm") is True  # listing did not burn it

    def test_revoke_all_and_by_id(self):
        runner, _ = _make_runner()
        session_key = runner._session_key_for_source(_make_source())
        a = grants.create(session_key, "k1", "d1", grants.GrantSpec(seconds=3600))
        grants.create(session_key, "k2", "d2", grants.GrantSpec(seconds=3600))

        reply = asyncio.run(runner._handle_grants_command(_make_event(f"/grants revoke {a.id}")))
        assert "1" in reply
        assert [g.pattern_key for g in grants.list_active(session_key)] == ["k2"]

        reply = asyncio.run(runner._handle_grants_command(_make_event("/grants revoke all")))
        assert grants.list_active(session_key) == []
        assert "revoked" in reply.lower()


class TestPromptAdvertisesBoundedScope:
    def test_text_fallback_mentions_bounded_option(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback("rm -rf /tmp/x", "recursive delete", "/")
        assert "approve for 30m" in text
        assert "approve session" in text  # existing scopes still offered

    def test_smart_deny_does_not_offer_bounded_scope(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback("rm -rf /tmp/x", "d", "/", smart_denied=True, allow_permanent=False)
        assert "for 30m" not in text
