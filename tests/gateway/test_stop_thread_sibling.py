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


def test_sibling_matches_under_a_named_profile():
    """The prefix must follow the caller's namespace, not a hardcoded ``main``.

    ``_session_key_namespace`` puts a named profile in the namespace slot
    (``agent:coder:...``), so a hardcoded ``agent:main`` prefix matched nothing
    and ``/stop`` silently left the sibling's run alive under every profile
    except the default one.
    """
    runner = object.__new__(GatewayRunner)
    own = build_session_key(
        _thread_source("userA"), thread_sessions_per_user=True, profile="coder"
    )
    sibling = build_session_key(
        _thread_source("userB"), thread_sessions_per_user=True, profile="coder"
    )
    # Same chat+thread but a DIFFERENT profile must stay isolated.
    other_profile = build_session_key(
        _thread_source("userB"), thread_sessions_per_user=True, profile="writer"
    )
    # An unrelated thread whose id merely starts with this one.
    other_thread = build_session_key(
        _thread_source("userB", thread_id="thr11"),
        thread_sessions_per_user=True,
        profile="coder",
    )
    assert own.startswith("agent:coder:")
    runner._running_agents = {
        own: _FakeAgent(),
        sibling: _FakeAgent(),
        other_profile: _FakeAgent(),
        other_thread: _FakeAgent(),
    }
    assert runner._sibling_thread_run_keys(_thread_source("userA"), own) == [sibling]


def test_sibling_default_profile_prefix_is_unchanged():
    """The default profile keeps the byte-identical ``agent:main`` behaviour."""
    runner = object.__new__(GatewayRunner)
    own = _per_user_key("userA")
    sibling = _per_user_key("userB")
    assert own.startswith("agent:main:")
    runner._running_agents = {own: _FakeAgent(), sibling: _FakeAgent()}
    assert runner._sibling_thread_run_keys(_thread_source("userA"), own) == [sibling]


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
