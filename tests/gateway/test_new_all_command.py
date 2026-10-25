"""/new-all: bulk session rotation across every routed chat on the gateway.

Pins three things the command cannot regress into:

* every routed chat (not just the caller's) lands on a NEW session id, and the old row is closed
  with ``session_reset`` so the transcript stays retrievable;
* a chat with a turn in flight is LEFT ALONE (resetting it would kill a reply its user is waiting
  on) and the count is reported;
* ``/new`` still rotates exactly one conversation after sharing the rotation seam with /new-all.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import AsyncSessionStore, SessionSource, SessionStore, build_session_key


def _source(platform: Platform, user_id: str, chat_id: str) -> SessionSource:
    return SessionSource(platform=platform, user_id=user_id, chat_id=chat_id, chat_type="dm")


def _runner(store: SessionStore):
    """Bare GatewayRunner with a REAL SessionStore and mocked per-session side effects."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._running_agents = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    # Conversation-boundary side effects (agent teardown, cached-agent eviction, hooks) are other
    # suites' subject; here they are no-ops so the store transition is what's under test.
    for name in ("_invalidate_session_run_generation", "_release_running_agent_state",
                 "_evict_cached_agent", "_clear_conversation_scope"):
        setattr(runner, name, MagicMock())
    runner._cleanup_old_agent_for_reset = AsyncMock()
    runner._fire_session_reset_hooks = AsyncMock()
    return runner


@pytest.fixture()
def store(tmp_path):
    s = SessionStore(tmp_path / "sessions", GatewayConfig())
    yield s
    s._db.close()


def test_new_all_rotates_every_chat_and_closes_old_rows(store):
    import asyncio

    import gateway.slash_commands_session  # binds the mixin onto GatewayRunner

    a = store.get_or_create_session(_source(Platform.TELEGRAM, "u1", "c1"))
    b = store.get_or_create_session(_source(Platform.BLUEBUBBLES, "+15551234567", "+15551234567"))
    old_ids = {a.session_key: a.session_id, b.session_key: b.session_id}

    runner = _runner(store)
    event = MessageEvent(text="/new-all", source=_source(Platform.TELEGRAM, "u1", "c1"), message_id="m1")
    reply = asyncio.run(runner._handle_new_all_command(event))

    assert "2 chats" in str(reply)
    for key, old_id in old_ids.items():
        entry = store._entries[key]
        assert entry.session_id != old_id
        assert entry.is_fresh_reset is True
        assert store._db.get_session(old_id)["end_reason"] == "session_reset"
    # Hooks fire for every entry that still carries its delivery origin.
    assert runner._fire_session_reset_hooks.await_count == 2


def test_new_all_skips_a_chat_with_a_turn_in_flight(store):
    import asyncio

    a = store.get_or_create_session(_source(Platform.TELEGRAM, "u1", "c1"))
    b = store.get_or_create_session(_source(Platform.TELEGRAM, "u2", "c2"))
    a_id, b_id = a.session_id, b.session_id

    runner = _runner(store)
    runner._running_agents = {b.session_key: object()}
    event = MessageEvent(text="/new-all", source=_source(Platform.TELEGRAM, "u1", "c1"), message_id="m1")
    reply = asyncio.run(runner._handle_new_all_command(event))

    assert store._entries[a.session_key].session_id != a_id
    assert store._entries[b.session_key].session_id == b_id
    assert "Left 1 chat alone" in str(reply)


def test_single_new_rotates_only_the_callers_chat(store):
    import asyncio

    a = store.get_or_create_session(_source(Platform.TELEGRAM, "u1", "c1"))
    b = store.get_or_create_session(_source(Platform.TELEGRAM, "u2", "c2"))
    a_id, b_id = a.session_id, b.session_id

    runner = _runner(store)
    # /new's own header helpers: no session DB, no topic lane.
    runner._session_db = None
    runner._reset_notice_session_info = MagicMock(return_value="")
    runner._telegram_topic_new_header = MagicMock(return_value=None)
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    event = MessageEvent(text="/new", source=_source(Platform.TELEGRAM, "u1", "c1"), message_id="m1")
    asyncio.run(runner._handle_reset_command(event))

    assert store._entries[a.session_key].session_id != a_id
    assert store._entries[b.session_key].session_id == b_id


def test_new_all_is_registered_as_a_gateway_slash_command():
    from gateway.run_busy import GatewayBusySessionMixin
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

    assert resolve_command("/new-all").name == "new-all"
    assert "new-all" in GATEWAY_KNOWN_COMMANDS
    assert "new-all" in GatewayBusySessionMixin._IDLE_COMMANDS
    assert "new-all" not in GatewayBusySessionMixin._PLAIN_COMMANDS
