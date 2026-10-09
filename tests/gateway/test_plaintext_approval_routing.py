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
    adapter._pending_messages = {}
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


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("/approve", "once"),
        ("/approve all", "once"),
        ("/approve session", "session"),
        ("/approve always", "always"),
        pytest.param("> previous prompt\n/approve", "once", id="quoted-previous-prompt-/approve"),
        pytest.param("> quoted\n/approve all", "once", id="quoted-/approve-all"),
        pytest.param("> quoted\n/approve session", "session", id="quoted-/approve-session"),
        ("!approve", "once"),
        ("/yes", "once"),
        pytest.param("> quoted\n/yes", "once", id="quoted-/yes"),
    ],
)
def test_approval_routing_handles_slash_and_quoted_replies(reply, expected):
    """A quoted /approve (or last-line slash) must resolve, not be queued.

    Regression for #81026: the second (and subsequent) /approve in a session
    can be delivered as a quoted reply or with a display prefix. The busy
    handler previously only matched bare words and exact slash forms that
    bypassed the active-session guard, so the approval was queued/interrupted
    and the command timed out.
    """
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    event = _make_event(reply)
    handled = asyncio.run(runner._handle_active_session_busy_message(event, session_key))

    assert handled is True
    assert entry.event.is_set()
    assert entry.result == expected
    assert event.get_command() == "approve"
    adapter._send_with_retry.assert_awaited()
    _clear_approval_state()


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("always approve", "always"),
        ("approve always", "always"),
        ("session approve", "session"),
        ("approve session", "session"),
        ("always yes", "always"),
        ("session y", "session"),
    ],
)
def test_approval_scope_word_order(reply, expected):
    """Reversed scope shorthand ("always approve" etc.) resolves correctly.

    Regression from review on #81088: the helper previously
    synthesized "/approve approve" for "always approve", losing the scope.
    """
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(_make_event(reply), session_key)
    )

    assert handled is True
    assert entry.event.is_set()
    assert entry.result == expected
    adapter._send_with_retry.assert_awaited()
    _clear_approval_state()


@pytest.mark.parametrize("route", ["busy", "priority"])
def test_two_sequential_quoted_approvals_resolve_without_queueing(route):
    _clear_approval_state()
    runner, adapter = _make_runner()
    for index in range(2):
        session_key, entry = _register_blocking_approval(runner)
        event = _make_event("> pending command\n/approve")
        event.message_id = f"approval-{index}"
        if route == "busy":
            handled = asyncio.run(runner._handle_active_session_busy_message(event, session_key))
        else:
            handled, reply = asyncio.run(runner._hm_busy_slash_or_photo(event, event.source, session_key))
            assert reply
        assert handled is True
        assert entry.event.is_set() and entry.result == "once"
        assert event.text == "/approve"
        assert not adapter._pending_messages.get(session_key)
    _clear_approval_state()


@pytest.mark.parametrize("route", ["busy", "priority"])
@pytest.mark.parametrize("reply", [
    "yes please explain", "approve this paragraph", "always use dry-run",
    pytest.param("> quote\nyes", id="quoted-bare-yes"), "> /approve",
])
def test_pending_approval_does_not_consume_prose(route, reply):
    _clear_approval_state()
    runner, _ = _make_runner()
    session_key, entry = _register_blocking_approval(runner)
    event = _make_event(reply)
    if route == "busy":
        handled = asyncio.run(runner._route_plaintext_approval_while_busy(event, session_key))
    else:
        handled, result = asyncio.run(runner._hm_busy_slash_or_photo(event, event.source, session_key))
        assert result is None
    assert handled is False
    assert not entry.event.is_set()
    assert event.text == reply
    _clear_approval_state()


@pytest.mark.parametrize("route", ["busy", "priority"])
@pytest.mark.parametrize("reply", [pytest.param("> quote\n/approve", id="quoted-approve"), "!deny"])
def test_untrusted_control_text_cannot_resolve_approval(route, reply):
    _clear_approval_state()
    runner, _ = _make_runner()
    session_key, entry = _register_blocking_approval(runner)
    event = _make_event(reply)
    event.allow_gateway_control = False
    if route == "busy":
        handled = asyncio.run(runner._route_plaintext_approval_while_busy(event, session_key))
    else:
        handled, result = asyncio.run(runner._hm_busy_slash_or_photo(event, event.source, session_key))
        assert result is None
    assert handled is False
    assert not entry.event.is_set()
    assert event.text == reply
    _clear_approval_state()


@pytest.mark.parametrize("route", ["busy", "priority"])
@pytest.mark.parametrize("reply", [pytest.param("> quote\n/approve", id="quoted-approve"), "!deny", "yes"])
def test_normalized_approval_obeys_slash_access(route, reply):
    _clear_approval_state()
    runner, adapter = _make_runner()
    runner.config = GatewayConfig(platforms={
        Platform.TELEGRAM: PlatformConfig(
            enabled=True, token="***", extra={"allow_admin_from": ["admin"], "user_allowed_commands": []},
        ),
    })
    session_key, entry = _register_blocking_approval(runner)
    event = _make_event(reply)
    if route == "busy":
        handled = asyncio.run(runner._route_plaintext_approval_while_busy(event, session_key))
        response = adapter._send_with_retry.call_args.kwargs["content"]
    else:
        handled, response = asyncio.run(runner._hm_busy_slash_or_photo(event, event.source, session_key))
    assert handled is True
    assert "admin-only" in response
    assert not entry.event.is_set()
    assert event.text == reply
    _clear_approval_state()


@pytest.mark.parametrize("route", ["busy", "priority"])
@pytest.mark.parametrize("reply,expected", [("ja", "once"), ("sitzung", "session"), ("yes", "once"), ("nein", "deny")])
def test_localized_approval_words_remain_supported(route, reply, expected, monkeypatch):
    from agent import i18n

    monkeypatch.setenv("HERMES_LANGUAGE", "de")
    i18n.reset_language_cache()
    _clear_approval_state()
    try:
        runner, _ = _make_runner()
        session_key, entry = _register_blocking_approval(runner)
        event = _make_event(reply)
        if route == "busy":
            handled = asyncio.run(runner._handle_active_session_busy_message(event, session_key))
        else:
            handled, response = asyncio.run(runner._hm_busy_slash_or_photo(event, event.source, session_key))
            assert response
        assert handled is True
        assert entry.event.is_set() and entry.result == expected
        assert event.get_command() == ("deny" if expected == "deny" else "approve")
    finally:
        _clear_approval_state()
        i18n.reset_language_cache()


@pytest.mark.parametrize("route", ["busy", "priority"])
def test_quoted_denial_preserves_reason(route):
    _clear_approval_state()
    runner, _ = _make_runner()
    session_key, entry = _register_blocking_approval(runner)
    event = _make_event("> pending command\n!deny Use DRY-RUN instead")
    if route == "busy":
        handled = asyncio.run(runner._handle_active_session_busy_message(event, session_key))
    else:
        handled, response = asyncio.run(runner._hm_busy_slash_or_photo(event, event.source, session_key))
        assert response
    assert handled is True
    assert entry.event.is_set() and entry.result == "deny"
    assert event.text == "/deny Use DRY-RUN instead"
    assert entry.reason == "Use DRY-RUN instead"
    _clear_approval_state()


def test_no_pending_approval_does_not_consume_conversational_yes():
    """A bare 'yes' with NO blocking approval must NOT be treated as an
    approval — it falls through to normal busy handling (design intent:
    'yes' in conversation must not execute a dangerous command)."""
    _clear_approval_state()
    runner, _adapter = _make_runner()
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
