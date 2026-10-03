"""Slack slash-command turns carry the same channel prompt, skill binding and source names as messages.

A ``/hermes <question>`` turn is a human turn, so the gateway re-pins ``channel_pin`` and the
session-context key from it. Built without ``channel_prompt``, ``chat_name`` and ``user_name``, it
ran without the configured channel prompt, and where it shared a session with messages it flipped
both pins until the next message flipped them back. Without ``auto_skill``, a session opened by
``/hermes <question>`` never loaded the channel's bound skill, and neither did the turn a
``/goal <text>`` kickoff queues. These tests check what the adapter
hands the gateway (event fields, handoff order, lookups), not the agent or provider cache behind it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _adapter(channel_id: str) -> SlackAdapter:
    a = SlackAdapter(PlatformConfig(
        enabled=True, token="xoxb-fake", extra={
            "channel_prompts": {channel_id: "Answer in haiku."},
            "channel_skill_bindings": [{"id": channel_id, "skill": "triage"}]}))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={"user": {"profile": {"display_name": "Alice"}, "real_name": "Alice"}})
    a._app.client.conversations_info = AsyncMock(
        return_value={"ok": True, "channel": (
            {"id": channel_id, "is_im": True, "user": "U_ALICE"} if channel_id.startswith("D")
            else {"id": channel_id, "name": "ops"})})
    a._bot_user_id = "U_BOT"
    a._bot_display_name = "HermesBot"
    a._running = True
    a.handle_message = AsyncMock()
    return a


def _agent_turn(event):
    """The turn the agent runs for *event*: for ``/goal <text>`` that is the kickoff the gateway
    queues, not the command itself."""
    if event.get_command() != "goal":
        return event
    from gateway.slash_commands_goals import GatewayGoalCommandsMixin
    runner, queued = object.__new__(GatewayGoalCommandsMixin), []
    runner._adapter_and_key_for = lambda _event: (object(), "sk")
    runner._enqueue_fifo = lambda _key, turn, _adapter: queued.append(turn)
    runner._enqueue_goal_turn(event, event.get_command_args(), label="kickoff", kickoff=True)
    return queued[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id, channel_type, chat_name, texts, cancel_first", [
    ("C_OPS", "channel", "ops", ("what broke?", "and why?"), False),
    ("D_ALICE", "im", "Alice", ("what broke?", "and why?"), False),
    ("C_OPS", "channel", "ops", ("queue what broke?", "queue and why?"), False),
    ("C_OPS", "channel", "ops", ("queue what broke?", "queue and why?"), True),
    ("C_OPS", "channel", "ops", ("goal fix what broke", "queue and why?"), False),
], ids=["channel", "dm", "queue-command", "queue-first-cancelled", "goal-kickoff"])
async def test_slash_turns_reach_the_gateway_in_order_with_message_inputs(
        channel_id, channel_type, chat_name, texts, cancel_first):
    """Two slash turns of one session, the first held on a cold ``users.info``: both are handed
    over in arrival order (the gateway's /queue FIFO keeps the order it is given), each with the
    inputs a message turn carries. ``queue-command``: a registered command that starts a turn is a
    human turn like the free-form question. ``queue-first-cancelled``: a slash cancelled during its
    lookup does not strand the one behind it. ``goal-kickoff``: the turn ``/goal <text>`` queues
    carries them too."""
    adapter = _adapter(channel_id)
    held, release = asyncio.Event(), asyncio.Event()

    async def users_info(**_kwargs):
        if not held.is_set():
            held.set()
            await release.wait()
        return {"user": {"profile": {"display_name": "Alice"}, "real_name": "Alice"}}

    adapter._app.client.users_info = AsyncMock(side_effect=users_info)

    def slash(text):
        return adapter._handle_slash_command(
            {"command": "/hermes", "text": text, "user_id": "U_ALICE",
             "channel_id": channel_id, "team_id": "T1"})

    first = asyncio.create_task(slash(texts[0]))
    await asyncio.wait_for(held.wait(), 5)
    second = asyncio.create_task(slash(texts[1]))
    for _ in range(5):  # the second slash's own lookups are warm or immediate
        await asyncio.sleep(0)
    if cancel_first:
        first.cancel()
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 5)
    await adapter._handle_slack_message(
        {"text": "<@U_BOT> what broke?", "user": "U_ALICE", "channel": channel_id,
         "channel_type": channel_type, "ts": "1700.000100"},
        {"team_id": "T1"})
    *slashes, message = (c.args[0] for c in adapter.handle_message.await_args_list)
    handed = texts[1:] if cancel_first else texts
    assert [s.text for s in slashes] == [
        "/" + t if t.split()[0] in ("queue", "goal") else t for t in handed]
    assert message.channel_prompt and "Answer in haiku." in message.channel_prompt
    assert (message.source.chat_name, message.source.user_name) == (chat_name, "Alice")
    for turn in map(_agent_turn, slashes):
        assert turn.channel_prompt == message.channel_prompt
        assert turn.auto_skill == message.auto_skill == ["triage"]
        assert (turn.source.chat_name, turn.source.user_name) == (
            message.source.chat_name, message.source.user_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("hold", ["dispatch", "lookup-middle-cancelled"])
async def test_busy_session_queue_keeps_slash_arrival_order(hold):
    """``/queue`` on a busy session, through the real busy path into the runner's FIFO: the pending
    slot and overflow keep arrival order. ``dispatch``: the first is held inside the gateway after
    the adapter handed it over (where ``pre_gateway_dispatch`` awaits) while the second arrives.
    ``lookup-middle-cancelled``: the first is held on a cold ``users.info``; of the two behind it,
    the middle one is cancelled while it waits, and the third still waits for the first."""
    from gateway.platforms.base import MessageEvent
    from gateway.run_busy import GatewayBusySessionMixin
    from gateway.session_state import SessionState

    adapter = _adapter("C_OPS")
    base = {"command": "/queue", "user_id": "U_ALICE", "channel_id": "C_OPS", "team_id": "T1"}
    key = adapter._event_session_key(MessageEvent(text="x", source=adapter.build_source(
        chat_id="C_OPS", chat_type="group", user_id="U_ALICE",
        thread_id=adapter._slash_thread_id(base), scope_id="T1")))
    owner = asyncio.create_task(asyncio.Event().wait())
    adapter._active_sessions[key], adapter._session_tasks[key] = asyncio.Event(), owner

    class _Runner(GatewayBusySessionMixin):
        states: dict = {}
        def _session_state(self, k):
            return self.states.setdefault(k, SessionState())
        def _peek_session_state(self, k):
            return self.states.get(k)

    runner, held, release = _Runner(), asyncio.Event(), asyncio.Event()

    async def gateway(event):
        if hold == "dispatch" and event.text.endswith("first"):
            held.set()
            await release.wait()
        runner._enqueue_fifo(key, event, adapter)

    async def users_info(**_kwargs):
        if hold != "dispatch" and not held.is_set():
            held.set()
            await release.wait()
        return {"user": {"profile": {"display_name": "Alice"}, "real_name": "Alice"}}

    del adapter.handle_message  # the real base path, not the mock
    adapter._message_handler = gateway
    adapter._app.client.users_info = AsyncMock(side_effect=users_info)
    slash = lambda text: asyncio.create_task(adapter._handle_slash_command({**base, "text": text}))

    tasks = [slash("first")]
    await asyncio.wait_for(held.wait(), 5)
    tasks += [slash("middle"), slash("last")] if hold != "dispatch" else [slash("last")]
    for _ in range(10):
        await asyncio.sleep(0)
    if hold != "dispatch":
        tasks[1].cancel()
        for _ in range(10):
            await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    owner.cancel()
    queued = [adapter._pending_messages[key], *runner.states[key].conversation.queued_events]
    assert [e.text for e in queued] == ["/queue first", "/queue last"]


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id, authorized, command, reaches_runner", [
    ("C_OPS", False, "/hermes", 0), ("D_ALICE", False, "/hermes", 1),
    ("C_OPS", True, "/stop", 1), ("C_OPS", True, "/approve", 1), ("C_OPS", True, "/pause", 1),
], ids=["rejected-channel", "rejected-dm", "stop", "approve", "pause"])
async def test_slash_that_starts_no_turn_costs_no_slack_lookup(channel_id, authorized, command, reaches_runner):
    """The message path rejects an unauthorized sender before any Slack lookup, and the names the
    slash path now resolves must not cost one either. In a DM the runner still gets the event,
    without names: it answers an unauthorized DM per ``unauthorized_dm_behavior`` (pairing code
    or decline), and a slash command there is how an unpaired user gets that answer.
    ``stop`` / ``approve`` / ``pause``: a command that interrupts, unblocks or pauses work is
    handed over before any lookup. A cold ``users.info`` for a second operator must not hold
    ``/stop`` or the emergency ``/pause`` back while the worker keeps running."""
    adapter = _adapter(channel_id)
    adapter.set_authorization_check(lambda *_args, **_kwargs: authorized)
    await adapter._handle_slash_command(
        {"command": command, "text": "what broke?" if command == "/hermes" else "",
         "user_id": "U_MALLORY", "channel_id": channel_id, "team_id": "T1"})
    adapter._app.client.users_info.assert_not_awaited()
    adapter._app.client.conversations_info.assert_not_awaited()
    assert adapter.handle_message.await_count == reaches_runner
