"""Tests for #46866: plain-text approval responses must resolve a blocking
dangerous-command approval instead of being steered/queued.

When the agent is blocked inside tools/approval.py waiting for a dangerous
command to be approved, a messaging user who replies "yes" / "approve" /
"deny" (without the leading slash) must have that response routed to the
approval handler.  Previously the bare-word reply fell through to the
steer/queue/interrupt logic in _handle_active_session_busy_message — the
approval never resolved, timed out, and auto-denied.

Slash forms (/approve, /deny) already bypass at the base-adapter guard;
this covers the bare-word forms Signal/SMS users naturally type.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m1",
    )


def _clear_approval_state():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


def _make_runner():
    """Minimal GatewayRunner that exercises the real busy-session handler."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="reply1")
    )
    # _unwrap_ephemeral is a real base-adapter method; emulate its contract.
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
    # _handle_active_session_busy_message uses these only on the
    # non-approval fall-through path; harmless to stub.
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    return runner, adapter


def _register_blocking_approval(runner):
    """Register a real blocking approval entry for the runner's session."""
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    entry = _ApprovalEntry({"command": "rm -rf /tmp/test"})
    _gateway_queues.setdefault(session_key, []).append(entry)
    return session_key, entry


@pytest.mark.parametrize("reply", ["yes", "approve", "ok", "y", "confirm"])
def test_plaintext_yes_resolves_approval(reply):
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(_make_event(reply), session_key)
    )

    assert handled is True
    assert entry.event.is_set()
    assert entry.result == "once"
    # The user gets a confirmation reply, not silence.
    adapter._send_with_retry.assert_awaited()
    _clear_approval_state()


def test_no_pending_approval_does_not_consume_conversational_yes():
    """A bare 'yes' with NO blocking approval must NOT be treated as an
    approval — it falls through to normal busy handling (design intent:
    'yes' in conversation must not execute a dangerous command)."""
    _clear_approval_state()
    runner, adapter = _make_runner()
    source = _make_source()
    session_key = runner._session_key_for_source(source)
    # No approval registered.

    handled = asyncio.run(
        runner._handle_active_session_busy_message(_make_event("yes"), session_key)
    )

    # No approval existed, so nothing was resolved — the "yes" is treated
    # as ordinary text, not as a dangerous-command approval (design intent).
    # (It still flows through normal busy handling, which may send a busy
    # ack; the contract here is only that no approval was consumed.)
    from tools.approval import _gateway_queues
    assert session_key not in _gateway_queues
    _clear_approval_state()




def _gated_runner(user_id: str):
    """Runner with slash gating on (``allow_admin_from``) and an event from ``user_id``."""
    from dataclasses import replace
    runner, adapter = _make_runner()
    runner.config.platforms[Platform.TELEGRAM].extra["allow_admin_from"] = ["admin1"]
    event = _make_event("always")
    event.source = replace(event.source, user_id=user_id)
    return runner, event


@pytest.mark.parametrize("user_id, resolves", [("admin1", True), ("member2", False)])
def test_plaintext_approval_obeys_the_slash_admin_gate(user_id, resolves):
    """A bare word IS /approve ('always' even allowlists the pattern permanently). With
    allow_admin_from set, a participant refused /approve must not approve by typing the word."""
    _clear_approval_state()
    runner, event = _gated_runner(user_id)
    session_key, entry = _register_blocking_approval(runner)

    asyncio.run(runner._handle_active_session_busy_message(event, session_key))

    assert entry.event.is_set() is resolves
    _clear_approval_state()


def _event_as(user_id: str, text: str) -> MessageEvent:
    """``text`` from ``user_id`` in a Telegram chat with a numeric id (the adapter sends to it)."""
    from dataclasses import replace
    event = _make_event(text)
    event.source = replace(event.source, chat_id="12345", user_id=user_id)
    return event


def _gated_confirm_prompt():
    """The admin's /reset confirm in a gated chat: registered by the runner's real
    ``_request_slash_confirm`` and rendered as native buttons by a real Telegram adapter."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    runner, _ = _make_runner()
    runner.config.platforms[Platform.TELEGRAM].extra["allow_admin_from"] = ["111"]
    telegram = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    telegram._bot = AsyncMock()
    telegram._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    runner.adapters = {Platform.TELEGRAM: telegram}
    ran = []

    async def _handler(choice):
        ran.append(choice)
        return "reset done"

    event = _event_as("111", "/reset")
    rendered = asyncio.run(runner._request_slash_confirm(
        event=event, command="reset", title="/reset", message="Reset?", handler=_handler))
    assert rendered is None  # buttons, not the text fallback
    return runner, telegram, runner._session_key_for_source(event.source), ran


def _click(telegram, user_id: str, confirm_id: str) -> None:
    query = AsyncMock()
    query.data = f"sc:once:{confirm_id}"
    query.message = MagicMock(chat_id=12345, message_thread_id=None)
    query.from_user = SimpleNamespace(id=int(user_id), first_name=f"user{user_id}")
    asyncio.run(telegram._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock()))


@pytest.mark.parametrize("answer", ["typed_reply", "telegram_button"])
@pytest.mark.parametrize("user_id, allowed", [("111", True), ("222", False)])
def test_slash_confirm_answers_obey_the_confirmed_commands_gate(monkeypatch, answer, user_id, allowed):
    """Answering a slash-confirm runs the confirmed command ('always' also persists the opt-out), so
    every answer path (a typed reply, any adapter's native button) must apply the slash policy for
    that command. A refused answer leaves the prompt live for the admin, and while gating is on an
    answer that names nobody (an adapter that does not identify the clicker) is refused."""
    from tools import slash_confirm

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")  # both users pass ordinary admission
    runner, telegram, key, ran = _gated_confirm_prompt()
    confirm_id = slash_confirm.get_pending(key)["confirm_id"]

    def _answer_as(uid):
        if answer == "typed_reply":
            asyncio.run(runner._hm_slash_confirm_reply(_event_as(uid, "/approve"), key))
        else:
            _click(telegram, uid, confirm_id)

    try:
        assert asyncio.run(slash_confirm.resolve(key, confirm_id, "once")) is None
        _answer_as(user_id)
        assert ran == (["once"] if allowed else []), ran
        _answer_as("111")
        assert ran == ["once"], ran
    finally:
        slash_confirm.clear(key)
