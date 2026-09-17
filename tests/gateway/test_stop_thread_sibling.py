"""Regression tests: /stop can interrupt a sibling participant's run in a
per-user thread.

When ``thread_sessions_per_user=True``, each participant in a thread gets an
isolated session key (``...:{thread_id}:{user_id}``).  A run another user
started lives under a different key, so the caller's own ``/stop`` used to find
nothing and reply "no active task to stop".  Authorized users should be able to
stop any run in the same thread.
"""

import pytest

from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import Platform
from gateway.platforms.event import MessageEvent, MessageType


class _FakeAgent:
    pass


def _thread_source(uid, thread_id="thr1", chat_id="chan1"):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_type="forum",
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=uid,
    )


def _per_user_key(uid, thread_id="thr1", chat_id="chan1"):
    return build_session_key(
        _thread_source(uid, thread_id, chat_id),
        thread_sessions_per_user=True,
    )


# ---------------------------------------------------------------------------
# _sibling_thread_run_keys
# ---------------------------------------------------------------------------


def test_sibling_returns_empty_for_non_thread_source():
    # Non-thread group/channel must NOT trigger the cross-user fallback.
    runner = object.__new__(GatewayRunner)
    nonthread = SessionSource(
        platform=Platform.DISCORD, chat_type="group", chat_id="chan1", user_id="userA"
    )
    grp_b = build_session_key(
        SessionSource(
            platform=Platform.DISCORD, chat_type="group", chat_id="chan1", user_id="userB"
        )
    )
    runner._running_agents = {grp_b: _FakeAgent()}
    assert runner._sibling_thread_run_keys(nonthread, "agent:main:discord:group:chan1:userA") == []


def test_sibling_matches_named_profile_runs():
    # Under multiplexing the sibling prefix must follow the source's profile namespace,
    # so /stop finds another participant's run under the SAME named profile...
    runner = object.__new__(GatewayRunner)
    source = _thread_source("userA")
    source.profile = "work"
    key_b = build_session_key(
        _thread_source("userB"), thread_sessions_per_user=True, profile="work"
    )
    runner._running_agents = {key_b: _FakeAgent()}
    assert runner._sibling_thread_run_keys(
        source, "agent:work:discord:forum:chan1:thr1:userA"
    ) == [key_b]


def test_sibling_does_not_cross_profiles():
    # ...and never reaches a DIFFERENT profile's run in the same chat/thread.
    runner = object.__new__(GatewayRunner)
    source = _thread_source("userA")
    source.profile = "work"
    main_key = build_session_key(_thread_source("userB"), thread_sessions_per_user=True)
    runner._running_agents = {main_key: _FakeAgent()}
    assert (
        runner._sibling_thread_run_keys(
            source, "agent:work:discord:forum:chan1:thr1:userA"
        )
        == []
    )


# ---------------------------------------------------------------------------
# _sibling_group_run_keys (#113846: plain group chats, per-sender keys)
# ---------------------------------------------------------------------------


def _group_source(uid, chat_id="chan1"):
    return SessionSource(
        platform=Platform.DISCORD, chat_type="group", chat_id=chat_id, user_id=uid
    )


def test_group_sibling_finds_bot_triggered_run():
    # The bot's turn runs under ...:group:chan1:bot; the human's /stop key is
    # ...:group:chan1:userA. The group fallback must bridge them.
    runner = object.__new__(GatewayRunner)
    bot_key = build_session_key(_group_source("bot"))
    runner._running_agents = {bot_key: _FakeAgent()}
    assert runner._sibling_group_run_keys(
        _group_source("userA"), "agent:main:discord:group:chan1:userA"
    ) == [bot_key]


def test_group_sibling_skips_threaded_sources():
    # Threaded stops keep thread scoping; the group fallback must not widen them.
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {"agent:main:discord:group:chan1:userB": _FakeAgent()}
    assert runner._sibling_group_run_keys(_thread_source("userA"), "x") == []


def test_group_sibling_does_not_cross_chats_or_profiles():
    runner = object.__new__(GatewayRunner)
    other_chat = build_session_key(_group_source("userB", chat_id="chan2"))
    source = _group_source("userA")
    source.profile = "work"
    main_key = build_session_key(_group_source("userB"))
    runner._running_agents = {other_chat: _FakeAgent(), main_key: _FakeAgent()}
    assert runner._sibling_group_run_keys(source, "agent:work:discord:group:chan1:userA") == []


@pytest.mark.asyncio
async def test_stop_interrupts_group_sibling_when_authorized(monkeypatch):
    runner = object.__new__(GatewayRunner)
    key_a = build_session_key(_group_source("userA"))
    key_bot = build_session_key(_group_source("bot"))
    runner._running_agents = {key_bot: _FakeAgent()}
    runner.session_store = _FakeStore(key_a)

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    runner._is_user_authorized_for_source = lambda source: True

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_group_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert interrupted == [(key_bot, "stop_command_group_sibling")]
    assert "no active" not in str(getattr(result, "text", result)).lower()


# ---------------------------------------------------------------------------
# _handle_stop_command fallback path
# ---------------------------------------------------------------------------


class _StoreEntry:
    def __init__(self, session_key):
        self.session_key = session_key


class _FakeStore:
    def __init__(self, session_key):
        self._key = session_key

    def get_or_create_session(self, source):
        return _StoreEntry(self._key)


@pytest.mark.asyncio
async def test_stop_does_not_interrupt_sibling_when_unauthorized(monkeypatch):
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _FakeAgent()}
    runner.session_store = _FakeStore(key_a)

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append(session_key)

    runner._interrupt_and_clear_session = _fake_interrupt
    runner._is_user_authorized = lambda source: False

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert interrupted == []
    assert "no active" in str(getattr(result, "text", result)).lower()


# ---------------------------------------------------------------------------
# /stop with no active agent still clears a stuck platform status (#32295)
# ---------------------------------------------------------------------------


class _FakeStatusAdapter:
    def __init__(self):
        self.cleared = []

    async def _stop_typing_with_metadata(self, chat_id, metadata=None):
        self.cleared.append((chat_id, metadata))


@pytest.mark.asyncio
async def test_stop_no_active_agent_survives_status_clear_failure():
    """A failing adapter clear must not break the /stop reply."""
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    key = _per_user_key("userA")
    runner.session_store = _FakeStore(key)
    runner._is_user_authorized = lambda source: True

    class _BoomAdapter:
        async def _stop_typing_with_metadata(self, chat_id, metadata=None):
            raise RuntimeError("boom")

    runner.adapters = {Platform.DISCORD: _BoomAdapter()}
    runner._thread_metadata_for_source = (
        lambda source, reply_to_message_id=None: None
    )
    runner._reply_anchor_for_event = lambda event: None

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert "no active" in str(getattr(result, "text", result)).lower()
