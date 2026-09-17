"""Regression tests: /stop falls back to ANY running turn in the same chat when
the caller's exact session key (and thread-sibling keys) miss.

Two real shapes produce the miss (found via the Slack native stop button,
gateway-gateway#286 review):

- A turn triggered by a TOP-LEVEL channel message keys ``chat_type=channel``
  (with the relay's thread_id fallback), while a stop/`/stop` arriving from
  inside the reply thread normalizes to ``chat_type=thread`` — different key,
  same chat, and the run was invisible to /stop.
- Rolling-DM configs key the DM session WITHOUT a thread slot; a stop event
  carrying the thread keys a different session.

Semantics: "/stop" means "stop what's running in this chat". On an exact +
thread-sibling miss, an AUTHORIZED user's /stop interrupts the chat's running
turns; other chats are never touched.
"""

import pytest

from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import Platform
from gateway.platforms.event import MessageEvent, MessageType


class _FakeAgent:
    pass


class _StoreEntry:
    def __init__(self, session_key):
        self.session_key = session_key


class _FakeStore:
    def __init__(self, session_key):
        self._key = session_key

    def get_or_create_session(self, source):
        return _StoreEntry(self._key)


def _slack_source(chat_type, chat_id, thread_id=None, user_id="U-alice", scope_id="T1"):
    return SessionSource(
        platform=Platform.SLACK, chat_type=chat_type, chat_id=chat_id,
        thread_id=thread_id, user_id=user_id, scope_id=scope_id,
    )


def _runner_with_run(running_key, own_key, authorized=True):
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {running_key: _FakeAgent()}
    runner.session_store = _FakeStore(own_key)
    runner._is_user_authorized_for_source = lambda source, **kw: authorized
    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    return runner, interrupted


@pytest.mark.asyncio
async def test_stop_from_thread_reaches_top_level_channel_run():
    # Running turn: triggered by a top-level channel message (relay stamps the
    # message's own ts as thread_id; chat_type slot stays "channel").
    running_key = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    # The stop arrives from inside the reply thread → normalizes to "thread".
    stop_source = _slack_source("thread", "C9", thread_id="170.100")
    own_key = build_session_key(stop_source)
    assert own_key != running_key  # the miss under test

    runner, interrupted = _runner_with_run(running_key, own_key)
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)

    assert [k for k, _ in interrupted] == [running_key]
    assert "no active" not in str(getattr(result, "text", result)).lower()


@pytest.mark.asyncio
async def test_stop_with_thread_reaches_rolling_dm_run():
    # Rolling-DM config: the running session keys WITHOUT a thread slot.
    running_key = build_session_key(_slack_source("dm", "D1"))
    # The stop event carries the session thread → keys a different session.
    stop_source = _slack_source("dm", "D1", thread_id="170.100")
    own_key = build_session_key(stop_source)
    assert own_key != running_key

    runner, interrupted = _runner_with_run(running_key, own_key)
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)

    assert [k for k, _ in interrupted] == [running_key]
    assert "no active" not in str(getattr(result, "text", result)).lower()


@pytest.mark.asyncio
async def test_chat_scope_fallback_is_authorization_gated():
    running_key = build_session_key(_slack_source("channel", "C9", thread_id="170.100"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")
    own_key = build_session_key(stop_source)

    runner, interrupted = _runner_with_run(running_key, own_key, authorized=False)
    # The no-active tail touches adapters; keep it inert for this harness.
    runner.adapters = {}
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)

    assert interrupted == []
    assert "no active" in str(getattr(result, "text", result)).lower()


@pytest.mark.asyncio
async def test_chat_scope_fallback_never_crosses_chats():
    # A run in ANOTHER chat of the same workspace must stay invisible.
    running_key = build_session_key(_slack_source("channel", "C-other", thread_id="170.100"))
    stop_source = _slack_source("thread", "C9", thread_id="170.100")
    own_key = build_session_key(stop_source)

    runner, interrupted = _runner_with_run(running_key, own_key)
    runner.adapters = {}
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)

    assert interrupted == []
    assert "no active" in str(getattr(result, "text", result)).lower()
