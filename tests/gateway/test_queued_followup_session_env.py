"""A queued follow-up turn must re-bind HERMES_SESSION_* from its own sender.

The queued / interrupting follow-up runs inside the outer turn's task
(``_run_agent_queued_followup`` recursing into ``_run_agent``), where the outer
turn's session ContextVars — bound once in ``_hmwa_prepare_turn`` — are still in
effect. In a shared group/thread session the queued message can belong to a
different member, so tools authorizing by ``HERMES_SESSION_USER_ID`` (API writes,
audit logs) would decide for whoever's turn happened to be live when the message
was queued (#121977).

The follow-up now re-binds the session vars from ``pending_event``'s own source
for the recursion, like the cold path; when the recursion unwinds the binding is
cleared (empty, fail-closed) rather than left holding either member's identity.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session_context import (
    get_session_env,
    reset_session_vars,
    set_session_vars,
)

SESSION_KEY = "agent:main:telegram:group:-1001:topic:7"


@pytest.fixture(autouse=True)
def _clean_session_vars(monkeypatch):
    # The outer turn binds before the follow-up runs; get_session_env would otherwise
    # fall through to os.environ, so scrub the legacy names too.
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    reset_session_vars()
    yield
    reset_session_vars()


def _source(*, user_id, message_id=None):
    return SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="supergroup",
        chat_name="ops",
        thread_id="7",
        user_id=user_id,
        user_id_alt="",
        user_name=f"Member {user_id}",
        message_id=message_id,
        profile="",
    )


def _turn_ctx(source):
    return SimpleNamespace(
        source=source,
        session_id="sid",
        session_key=SESSION_KEY,
        run_generation=1,
        _interrupt_depth=0,
        history=[],
        _status_thread_metadata={"thread_id": "7"},
        context_prompt=None,
        result_holder=[None],
    )


def _runner(capture):
    from gateway.run import GatewayRunner

    async def _capture_run(**kwargs):
        capture["user_id"] = get_session_env("HERMES_SESSION_USER_ID")
        capture["session_key"] = get_session_env("HERMES_SESSION_KEY")
        return {"final_response": "done", "messages": []}

    runner = object.__new__(GatewayRunner)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(side_effect=_capture_run)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=SESSION_KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value="the follow-up"
    )
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    return runner


async def _drain(runner, turn_ctx, pending_event):
    from gateway.run import GatewayRunner

    return await GatewayRunner._run_agent_queued_followup(
        runner,
        turn_ctx,
        adapter=None,
        pending="hi again",
        pending_event=pending_event,
        response="resp",
        result={"interrupted": True, "messages": []},
        stream_task=None,
    )


@pytest.mark.asyncio
async def test_the_followup_turn_sees_its_own_senders_identity():
    """Member B's queued follow-up must not authorize as member A, whose turn was live."""
    # What _hmwa_prepare_turn left bound for A's still-running turn.
    set_session_vars(
        platform="telegram",
        chat_id="-1001",
        chat_type="supergroup",
        chat_name="ops",
        thread_id="7",
        user_id="user-A",
        user_name="Member A",
        session_key=SESSION_KEY,
    )
    capture = {}
    runner = _runner(capture)
    pending_event = SimpleNamespace(
        source=_source(user_id="user-B", message_id="6002"),
        message_id="6002",
        channel_prompt=None,
        message_type=None,
        internal=False,
        metadata={},
    )

    await _drain(runner, _turn_ctx(_source(user_id="user-A")), pending_event)

    runner._run_agent.assert_awaited_once()
    assert capture["user_id"] == "user-B"
    assert capture["session_key"] == SESSION_KEY


@pytest.mark.asyncio
async def test_after_the_followup_the_binding_is_cleared_not_left_foreign():
    """Once the recursion unwinds the vars are cleared (empty, fail-closed) — the outer
    turn's remaining cleanup must not keep authorizing as the queued member."""
    set_session_vars(
        platform="telegram",
        chat_id="-1001",
        chat_type="supergroup",
        thread_id="7",
        user_id="user-A",
        session_key=SESSION_KEY,
    )
    capture = {}
    runner = _runner(capture)
    pending_event = SimpleNamespace(
        source=_source(user_id="user-B", message_id="6002"),
        message_id="6002",
        channel_prompt=None,
        message_type=None,
        internal=False,
        metadata={},
    )

    await _drain(runner, _turn_ctx(_source(user_id="user-A")), pending_event)

    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_KEY") == ""


@pytest.mark.asyncio
async def test_a_followup_without_pending_event_keeps_the_running_turns_identity():
    """An interrupt-style drain with no pending event re-binds the same outer source —
    a no-op that must not crash or blank the identity the recursion runs under."""
    set_session_vars(
        platform="telegram",
        chat_id="-1001",
        chat_type="supergroup",
        thread_id="7",
        user_id="user-A",
        session_key=SESSION_KEY,
    )
    capture = {}
    runner = _runner(capture)

    await _drain(runner, _turn_ctx(_source(user_id="user-A")), None)

    runner._run_agent.assert_awaited_once()
    assert capture["user_id"] == "user-A"
