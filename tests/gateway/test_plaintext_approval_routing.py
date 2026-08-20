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
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _StubPlatformAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="reply1")

    async def get_chat_info(self, chat_id):
        return {}


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


def _make_base_adapter():
    adapter = _StubPlatformAdapter(
        PlatformConfig(enabled=True, token="***"),
        Platform.TELEGRAM,
    )
    message_handler = AsyncMock(return_value="should not be called")
    send_with_retry = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="reply1")
    )
    adapter._message_handler = message_handler
    adapter._send_with_retry = send_with_retry
    return adapter, message_handler, send_with_retry


def _base_session_key(source: SessionSource) -> str:
    return build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )


def _register_blocking_approval(runner):
    """Register a real blocking approval entry for the runner's session."""
    from tools.approval import _ApprovalEntry, _gateway_queues
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


@pytest.mark.parametrize("emoji", ["👍", "thumbsup", "+1", "THUMBSUP"])
def test_thumbs_up_reaction_resolves_approval(emoji):
    """A 👍 tapback on the approval prompt approves once.

    Reactions reach the gateway as the cross-platform synthetic message
    ``reaction:added:<emoji>`` (Slack/Feishu/Photon adapters).  Slack name
    forms and Feishu's uppercase emoji_type are accepted as fallbacks in
    case a reaction reaches this path before emoji translation.
    """
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(
            _make_event(f"reaction:added:{emoji}"), session_key
        )
    )

    assert handled is True
    assert entry.event.is_set()
    assert entry.result == "once"
    adapter._send_with_retry.assert_awaited()
    _clear_approval_state()


def test_reaction_approval_routes_through_base_adapter_busy_path():
    """Regression guard for the production handoff.

    ``BasePlatformAdapter.handle_message`` must route a synthetic reaction
    message through the registered busy-session handler while the session is
    active, instead of queuing it as an ordinary follow-up turn.
    """
    _clear_approval_state()
    runner, _ = _make_runner()
    adapter, message_handler, send_with_retry = _make_base_adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    session_key, entry = _register_blocking_approval(runner)
    assert session_key == _base_session_key(_make_source())
    adapter._active_sessions[session_key] = asyncio.Event()

    asyncio.run(adapter.handle_message(_make_event("reaction:added:👍")))

    assert entry.event.is_set()
    assert entry.result == "once"
    message_handler.assert_not_awaited()
    assert session_key not in adapter._pending_messages
    send_with_retry.assert_awaited()
    _clear_approval_state()


@pytest.mark.parametrize("emoji", ["👎", "thumbsdown", "-1", "THUMBSDOWN"])
def test_thumbs_down_reaction_denies_approval(emoji):
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(
            _make_event(f"reaction:added:{emoji}"), session_key
        )
    )

    assert handled is True
    assert entry.event.is_set()
    assert entry.result == "deny"
    adapter._send_with_retry.assert_awaited()
    _clear_approval_state()


@pytest.mark.parametrize("text", ["reaction:removed:👍", "reaction:removed:👎",
                                  "reaction:removed:thumbsup", "reaction:removed:-1"])
def test_reaction_removed_never_resolves_approval(text):
    """Un-reacting is not a decision — removals must not approve or deny."""
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    asyncio.run(
        runner._handle_active_session_busy_message(_make_event(text), session_key)
    )

    assert not entry.event.is_set()
    assert entry.result is None
    _clear_approval_state()


@pytest.mark.parametrize(
    "emoji", ["🚀", "✅", "👀", "eyes", "white_check_mark", "ok", "yes"]
)
def test_arbitrary_reaction_does_not_resolve_approval(emoji):
    """Only thumbs reactions count — no other emoji may approve a
    dangerous command, even while an approval is blocking."""
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    asyncio.run(
        runner._handle_active_session_busy_message(
            _make_event(f"reaction:added:{emoji}"), session_key
        )
    )

    assert not entry.event.is_set()
    assert entry.result is None
    _clear_approval_state()


def test_reaction_without_pending_approval_consumes_nothing():
    """A 👍 reaction with no blocking approval must not register or resolve
    an approval — the has_blocking_approval gate is the only door in."""
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key = runner._session_key_for_source(_make_source())

    asyncio.run(
        runner._handle_active_session_busy_message(
            _make_event("reaction:added:👍"), session_key
        )
    )

    from tools.approval import _gateway_queues
    assert session_key not in _gateway_queues
    _clear_approval_state()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("reaction:added:👍", "👍"),
        ("reaction:added:thumbsup", "👍"),
        ("reaction:added:+1", "👍"),
        ("reaction:added::thumbsup:", "👍"),
        ("reaction:added:👎", "👎"),
        ("reaction:added:thumbsdown", "👎"),
        ("reaction:added:-1", "👎"),
        ("reaction:removed:👍", None),
        ("reaction:removed:👎", None),
        ("reaction:added:🚀", None),
        ("reaction:added:", None),
        ("👍", None),
        ("yes", None),
    ],
)
def test_approval_reaction_word_classification(text, expected):
    """Unit-level contract for the reaction → approval-word mapping."""
    from gateway.run import _approval_reaction_word

    assert _approval_reaction_word(text) == expected


def test_unrelated_text_with_pending_approval_falls_through():
    """Text that is neither approve nor deny vocab must NOT resolve the
    approval — it falls through to normal busy handling."""
    _clear_approval_state()
    runner, adapter = _make_runner()
    session_key, entry = _register_blocking_approval(runner)

    handled = asyncio.run(
        runner._handle_active_session_busy_message(
            _make_event("what files are here?"), session_key
        )
    )

    # Approval still pending — not resolved by unrelated text.
    assert not entry.event.is_set()
    _clear_approval_state()
