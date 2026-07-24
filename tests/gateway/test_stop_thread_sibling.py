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
from gateway.platforms.base import Platform, MessageEvent, MessageType


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


def _per_user_key(uid, thread_id="thr1", chat_id="chan1", profile=None):
    return build_session_key(
        _thread_source(uid, thread_id, chat_id),
        thread_sessions_per_user=True,
        profile=profile,
    )


def _dm_source(uid="@alice:ex", thread_id=None, chat_id="!dm:ex", profile=None):
    return SessionSource(
        platform=Platform.MATRIX,
        chat_type="dm",
        chat_id=chat_id,
        thread_id=thread_id,
        user_id=uid,
        profile=profile,
    )


def _dm_thread_key(thread_id, chat_id="!dm:ex", profile=None):
    return build_session_key(
        _dm_source(thread_id=thread_id, chat_id=chat_id, profile=profile), profile=profile
    )


def _set_dm_running_agent(runner, source, profile=None, agent=None):
    session_key = build_session_key(source, profile=profile)
    runner._running_agents = {session_key: agent or _FakeAgent()}
    runner._cache_session_source(session_key, source)
    return session_key


# ---------------------------------------------------------------------------
# _sibling_thread_run_keys
# ---------------------------------------------------------------------------


def test_sibling_finds_other_users_run_in_same_thread():
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _FakeAgent()}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == [key_b]


def test_sibling_excludes_callers_own_key():
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_a: _FakeAgent(), key_b: _FakeAgent()}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == [key_b]


def test_sibling_skips_pending_sentinel():
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _AGENT_PENDING_SENTINEL}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == []


def test_sibling_does_not_match_different_thread_same_chat():
    # thr1 caller must not match a run in thr11 (prefix-collision guard).
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA", thread_id="thr1")
    key_b_other = _per_user_key("userB", thread_id="thr11")
    runner._running_agents = {key_b_other: _FakeAgent()}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == []


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


def test_sibling_uses_callers_named_profile_namespace():
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA", profile="coder")
    key_b = _per_user_key("userB", profile="coder")
    runner._running_agents = {key_b: _FakeAgent()}

    assert runner._sibling_thread_run_keys(_thread_source("userA"), key_a) == [key_b]


# ---------------------------------------------------------------------------
# _sibling_dm_thread_run_keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [None, "coder"])
def test_dm_sibling_finds_threaded_run_in_callers_profile(profile):
    runner = object.__new__(GatewayRunner)
    own_key = build_session_key(_dm_source(profile=profile), profile=profile)
    run_source = _dm_source(thread_id="$root-event", profile=profile)
    run_key = _set_dm_running_agent(runner, run_source, profile)

    assert runner._sibling_dm_thread_run_keys(_dm_source(profile=profile), own_key) == [run_key]


def test_dm_sibling_rejects_same_room_run_from_another_profile():
    runner = object.__new__(GatewayRunner)
    own_key = build_session_key(_dm_source(profile="coder"), profile="coder")
    other_profile_key = _set_dm_running_agent(
        runner, _dm_source(thread_id="$root-event", profile="writer"), "writer"
    )

    assert runner._sibling_dm_thread_run_keys(_dm_source(profile="coder"), own_key) == []


def test_dm_sibling_rejects_room_id_prefix_collision():
    runner = object.__new__(GatewayRunner)
    stop_source = _dm_source(chat_id="!same:example.org")
    own_key = build_session_key(stop_source)
    _set_dm_running_agent(
        runner,
        _dm_source(thread_id="$root-event", chat_id="!same:example.org:8448"),
    )

    assert runner._sibling_dm_thread_run_keys(stop_source, own_key) == []


def test_dm_sibling_ignores_threaded_stop_source():
    runner = object.__new__(GatewayRunner)
    stop_source = _dm_source(thread_id="$idle-thread")
    own_key = build_session_key(stop_source)
    runner._running_agents = {_dm_thread_key("$root-event"): _FakeAgent()}

    assert runner._sibling_dm_thread_run_keys(stop_source, own_key) == []


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
async def test_stop_interrupts_sibling_thread_run_when_authorized(monkeypatch):
    runner = object.__new__(GatewayRunner)
    key_a = _per_user_key("userA")
    key_b = _per_user_key("userB")
    runner._running_agents = {key_b: _FakeAgent()}
    runner.session_store = _FakeStore(key_a)

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, interrupt_reason, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    runner._is_user_authorized = lambda source: True

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert interrupted == [(key_b, _INTERRUPT_REASON_STOP, "stop_command_thread_sibling")]
    # EphemeralReply or str — both carry the "stopped" message, not "no_active".
    assert "no active" not in str(getattr(result, "text", result)).lower()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [None, "coder"])
async def test_root_dm_stop_interrupts_auto_threaded_run_in_own_profile(profile):
    runner = object.__new__(GatewayRunner)
    stop_source = _dm_source(profile=profile)
    stop_key = build_session_key(stop_source, profile=profile)
    run_key = _set_dm_running_agent(
        runner, _dm_source(thread_id="$root-event", profile=profile), profile
    )
    runner.session_store = _FakeStore(stop_key)

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, interrupt_reason, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    runner._is_user_authorized = lambda source: False

    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)

    assert interrupted == [
        (run_key, _INTERRUPT_REASON_STOP, "stop_command_dm_thread_sibling")
    ]
    assert "no active" not in str(getattr(result, "text", result)).lower()


@pytest.mark.asyncio
async def test_root_dm_stop_clears_auto_threaded_pending_run():
    runner = object.__new__(GatewayRunner)
    stop_source = _dm_source()
    stop_key = build_session_key(stop_source)
    pending_key = _set_dm_running_agent(
        runner, _dm_source(thread_id="$root-event"), agent=_AGENT_PENDING_SENTINEL
    )
    runner.session_store = _FakeStore(stop_key)

    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason):
        interrupted.append((session_key, interrupt_reason, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    runner._is_user_authorized = lambda source: False

    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)

    assert interrupted == [
        (pending_key, _INTERRUPT_REASON_STOP, "stop_command_dm_thread_sibling")
    ]
    assert "no active" not in str(getattr(result, "text", result)).lower()


# ---------------------------------------------------------------------------
# /stop with no active agent still clears a stuck platform status (#32295)
# ---------------------------------------------------------------------------


class _FakeStatusAdapter:
    def __init__(self):
        self.cleared = []

    async def _stop_typing_with_metadata(self, chat_id, metadata=None):
        self.cleared.append((chat_id, metadata))


@pytest.mark.asyncio
async def test_stop_no_active_agent_clears_stuck_status():
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    key = _per_user_key("userA")
    runner.session_store = _FakeStore(key)
    runner._is_user_authorized = lambda source: True

    adapter = _FakeStatusAdapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._thread_metadata_for_source = (
        lambda source, reply_to_message_id=None: {"thread_id": source.thread_id}
    )
    runner._reply_anchor_for_event = lambda event: None

    event = MessageEvent(
        text="/stop", message_type=MessageType.TEXT, source=_thread_source("userA")
    )
    result = await runner._handle_stop_command(event)

    assert "no active" in str(getattr(result, "text", result)).lower()
    assert adapter.cleared == [("chan1", {"thread_id": "thr1"})]


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
