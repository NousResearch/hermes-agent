"""A queued / interrupting follow-up from sender B runs under B's session identity, not A's.

The session ContextVars (HERMES_SESSION_USER_ID etc.) are bound once per inbound message
(``_hmwa_prepare_turn``) and stay bound for the whole ``_run_agent`` chain. In a shared thread B's
message arriving while A's turn runs is queued under the same session key and executed by
``_run_agent_queued_followup`` as the next turn; that turn must re-bind B's identity so tool hooks
(``principal_hook_fields``), the approval requester and the Discord self-approval guard attribute
B's tools and exec prompts to B — while the outer (A's) turn keeps its own binding.
"""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars

SESSION_KEY = "discord:group:chan-1:thread:thread-1"


def _src(user_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD, user_id=user_id, user_name=f"name-{user_id}",
        chat_id="chan-1", chat_type="group", thread_id="thread-1",
    )


def _runner(seen: dict):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)

    async def _run_agent(message, context_prompt, history, source, session_id, **kw):
        seen["source_user_id"] = source.user_id
        seen["ctx_user_id"] = get_session_env("HERMES_SESSION_USER_ID", "")
        seen["ctx_user_name"] = get_session_env("HERMES_SESSION_USER_NAME", "")
        seen["ctx_session_key"] = get_session_env("HERMES_SESSION_KEY", "")
        return {"final_response": "ok", "messages": history, "history_offset": 0}

    async def _prepare(event=None, source=None, history=None, session_key=None, **kw):
        return event.text

    async def _refresh(session_key, session_id):
        return None

    runner._run_agent = _run_agent
    runner._prepare_profile_scoped_inbound_message_text = _prepare
    runner._refresh_agent_cache_message_count = _refresh
    runner._adapter_for_source = lambda source: None
    runner._session_key_for_source = lambda source: SESSION_KEY
    runner._reply_anchor_for_event = lambda event: None
    return runner


def _turn_ctx(source):
    return SimpleNamespace(
        source=source, session_id="sid", session_key=SESSION_KEY, run_generation=1, _interrupt_depth=0,
        history=[], _status_thread_metadata=None, result_holder=[None], context_prompt="",
    )


def _pending_event(source):
    return SimpleNamespace(source=source, text=f"hello from {source.user_id}", metadata={},
                           channel_prompt=None, message_type=None, internal=False)


@pytest.mark.asyncio
async def test_queued_followup_from_other_sender_runs_under_its_own_identity():
    seen = {}
    runner = _runner(seen)
    result = {"interrupted": True, "messages": [], "history_offset": 0}

    # What _hmwa_prepare_turn bound for A's inbound message; it stays bound for the whole chain.
    tokens = set_session_vars(platform="discord", chat_id="chan-1", chat_type="group", thread_id="thread-1",
                              user_id="A", user_name="name-A", session_key=SESSION_KEY)
    try:
        await runner._run_agent_queued_followup(
            _turn_ctx(_src("A")), None, "hello from B", _pending_event(_src("B")), "resp", result, None)
        # The outer turn (A) keeps its own binding after the follow-up returns.
        assert get_session_env("HERMES_SESSION_USER_ID") == "A"
    finally:
        clear_session_vars(tokens)

    assert seen["source_user_id"] == "B"
    assert seen["ctx_user_id"] == "B", seen
    assert seen["ctx_user_name"] == "name-B", seen
    assert seen["ctx_session_key"] == SESSION_KEY


@pytest.mark.asyncio
async def test_interrupt_followup_without_event_keeps_current_binding():
    """No pending_event (plain queued text for the same sender): nothing to re-bind."""
    seen = {}
    runner = _runner(seen)
    result = {"interrupted": True, "messages": [], "history_offset": 0}
    tokens = set_session_vars(platform="discord", chat_id="chan-1", user_id="A", user_name="name-A",
                              session_key=SESSION_KEY)
    try:
        await runner._run_agent_queued_followup(_turn_ctx(_src("A")), None, "more from A", None, "resp", result, None)
    finally:
        clear_session_vars(tokens)
    assert seen["source_user_id"] == "A"
    assert seen["ctx_user_id"] == "A"
