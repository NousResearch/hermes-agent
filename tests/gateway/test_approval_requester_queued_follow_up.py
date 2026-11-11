"""A message queued behind a running turn and drained as the next turn (the recursive ``_run_agent``)
is its own sender's turn: an approval it raises names that sender, not the member who opened the chain."""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_context
from tests.gateway.test_queued_followup_processing_hooks import HookRecordingAdapter, _install_fake_agent, _make_runner

KEY = "agent:main:telegram:group:-100"


class _ApprovingAgent:
    calls: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kw):
        from tools.approval_gateway_wait import _ApprovalEntry
        entry = _ApprovalEntry({"command": "rm -rf notes"})
        type(self).calls.append((message, entry.authority["requester_id"], entry.authority["requester_name"]))
        return {"final_response": "ok", "messages": [], "api_calls": 1}


def _event(text, uid, name, on_event):
    """Telegram observe mode carries the sender on the event; other platforms on the source."""
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group",
                        user_id=None if on_event else uid, user_name=None if on_event else name)
    who = {"user_id": uid, "user_name": name} if on_event else {}
    return MessageEvent(text=text, source=src, message_id=text, message_type=MessageType.TEXT, **who)


@pytest.mark.asyncio
@pytest.mark.parametrize("on_event", [True, False], ids=["observe-sender-on-event", "sender-on-source"])
@pytest.mark.parametrize(("follow_uid", "follow_name"), [("222", "Bob"), ("111", "Alice")],
                         ids=["other-member", "same-member"])
async def test_a_queued_follow_up_runs_under_its_own_sender(monkeypatch, tmp_path, on_event, follow_uid, follow_name):
    _ApprovingAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _ApprovingAgent)
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    alice = _event("alice-task", "111", "Alice", on_event)
    adapter._pending_messages[KEY] = _event("follow-task", follow_uid, follow_name, on_event)
    tokens = runner._set_session_env(build_session_context(alice.source, GatewayConfig(), event=alice))
    try:
        await runner._run_agent(message="alice-task", context_prompt="", history=[], source=alice.source,
                                session_id="s1", session_key=KEY)
        from gateway.session_context import get_session_env
        assert get_session_env("HERMES_SESSION_ACTOR_ID") == "111"  # the opener's binding is restored after
    finally:
        runner._clear_session_env(tokens)

    assert [(m.rsplit("] ", 1)[-1], uid, name) for m, uid, name in _ApprovingAgent.calls] == [
        ("alice-task", "111", "Alice"), ("follow-task", follow_uid, follow_name)]


@pytest.mark.asyncio
async def test_another_members_steer_into_a_queued_follow_up_fails_closed(monkeypatch, tmp_path):
    """Carol steers into Bob's drained follow-up: the fold clears the actor of the turn actually
    running (Bob's), so its approval names no one, not Bob alone."""
    _ApprovingAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _ApprovingAgent)
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    alice = _event("alice-task", "111", "Alice", True)
    adapter._pending_messages[KEY] = _event("bob-task", "222", "Bob", True)
    turn = runner._session_state(KEY).turn
    turn.event = alice  # the claim binds the slot to the opener
    carol = _event("carol-steer", "333", "Carol", True)
    original = _ApprovingAgent.run_conversation

    def steer_then_run(self, message, *args, **kwargs):
        if message.endswith("bob-task"):
            runner._fold_into_running_turn(turn.agent, KEY, carol)
        return original(self, message, *args, **kwargs)

    monkeypatch.setattr(_ApprovingAgent, "run_conversation", steer_then_run)
    tokens = runner._set_session_env(build_session_context(alice.source, GatewayConfig(), event=alice))
    try:
        await runner._run_agent(message="alice-task", context_prompt="", history=[], source=alice.source,
                                session_id="s1", session_key=KEY)
    finally:
        runner._clear_session_env(tokens)

    assert [(m.rsplit("] ", 1)[-1], uid) for m, uid, _ in _ApprovingAgent.calls] == [
        ("alice-task", "111"), ("bob-task", "")]


_CARRY_OVER_PATHS = {
    "compression-exhausted": lambda r, a, src: r._defer_past_exhaustion_reset(
        {"pending_steer": "carol-steer"}, a, src, KEY, _event("queued-task", "111", "Alice", False)),
    "depth-cap": lambda r, a, src: r._requeue_at_depth_cap(KEY, src, None, "carol-steer"),
    "behind-eventless-follow-up": lambda r, a, src: r._merge_leftover_steer(
        {"pending_steer": "carol-steer"}, a, src, KEY, None, "interrupt text"),
}


@pytest.mark.parametrize("path", list(_CARRY_OVER_PATHS))
def test_another_members_leftover_steer_never_runs_as_the_opener(path):
    """Carol's late steer into Alice's turn, re-queued as its own event, runs with no single
    actor: its approval binds to approvers and admins, never to Alice alone."""
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    alice = _event("alice-task", "111", "Alice", False)
    runner._session_state(KEY).turn.event = alice
    runner._fold_into_running_turn(None, KEY, _event("carol-steer", "333", "Carol", False))

    _CARRY_OVER_PATHS[path](runner, adapter, alice.source)

    queued = adapter._pending_messages[KEY]
    assert queued.text == "carol-steer"
    assert build_session_context(queued.source, GatewayConfig(), event=queued).actor_user_id == ""
