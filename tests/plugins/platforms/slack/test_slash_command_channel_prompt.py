"""Slack slash-command turns carry the same channel prompt, skill binding and source names as messages.

A ``/hermes <question>`` turn is a human turn, so the gateway re-pins ``channel_pin`` and the
session-context key from it. Built without ``channel_prompt``, ``chat_name`` and ``user_name``,
it ran without the configured channel prompt and flipped both pins, and the next ordinary
message flipped them back: two agent rebuilds and two prompt-cache misses per slash turn. Without
``auto_skill``, a session opened by ``/hermes <question>`` never loaded the channel's bound skill,
and neither did the turn a ``/goal <text>`` kickoff queues.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_context
from gateway.slash_commands_goals import GatewayGoalCommandsMixin
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
    queues (real ``_enqueue_goal_turn``), not the command itself."""
    if event.get_command() != "goal":
        return event
    runner, queued = object.__new__(GatewayGoalCommandsMixin), []
    runner._adapter_and_key_for = lambda _event: (object(), "sk")
    runner._enqueue_fifo = lambda _key, turn, _adapter: queued.append(turn)
    runner._enqueue_goal_turn(event, event.get_command_args(), label="kickoff", kickoff=True)
    return queued[0]


def _distinct_pins(turns) -> int:
    """Distinct (session-context prompt, channel prompt) pairs the real gateway pins render."""
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, token="x")})
    pins = set()
    for turn in turns:
        channel_prompt, _ = runner._pinned_channel_inputs(
            "k", turn.channel_prompt, turn.source, internal=False)
        context = runner._pinned_session_context_prompt(
            build_session_context(turn.source, config), False, "k")
        pins.add((context, channel_prompt))
    return len(pins)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id, channel_type, chat_name, text", [
    ("C_OPS", "channel", "ops", "what broke?"), ("D_ALICE", "im", "Alice", "what broke?"),
    ("C_OPS", "channel", "ops", "queue what broke?"), ("C_OPS", "channel", "ops", "goal fix what broke"),
], ids=["channel", "dm", "queue-command", "goal-kickoff"])
async def test_slash_turn_matches_message_turn_prompt_and_names(channel_id, channel_type, chat_name, text):
    """message -> slash -> message in one thread session renders one pinned prompt pair.
    ``queue-command``: a registered command that starts a turn (``/queue <prompt>``) is a human
    turn like the free-form question. ``goal-kickoff``: the turn ``/goal <text>`` queues carries
    the same inputs too."""
    adapter = _adapter(channel_id)
    message = {"text": "<@U_BOT> what broke?", "user": "U_ALICE", "channel": channel_id,
               "channel_type": channel_type, "ts": "1700.000100", "thread_ts": "1700.000050"}
    await adapter._handle_slack_message(message, {"team_id": "T1"})
    await adapter._handle_slash_command(
        {"command": "/hermes", "text": text, "user_id": "U_ALICE",
         "channel_id": channel_id, "team_id": "T1", "thread_ts": "1700.000050"})
    await adapter._handle_slack_message(dict(message, ts="1700.000200"), {"team_id": "T1"})
    assert adapter.handle_message.await_count == 3
    first, slash, last = (c.args[0] for c in adapter.handle_message.await_args_list)
    slash = _agent_turn(slash)
    assert first.channel_prompt and "Answer in haiku." in first.channel_prompt
    assert slash.channel_prompt == first.channel_prompt
    assert slash.auto_skill == first.auto_skill == ["triage"]
    assert (first.source.chat_name, first.source.user_name) == (chat_name, "Alice")
    assert (slash.source.chat_name, slash.source.user_name) == (chat_name, "Alice")
    assert _distinct_pins((first, slash, last)) == 1


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
    ``stop`` / ``approve`` / ``pause``: a command that interrupts, unblocks or pauses work is handed over
    before any lookup. A cold ``users.info`` for a second operator must not hold ``/stop`` back
    while the worker keeps running."""
    adapter = _adapter(channel_id)
    adapter.set_authorization_check(lambda *_args, **_kwargs: authorized)
    await adapter._handle_slash_command(
        {"command": command, "text": "what broke?" if command == "/hermes" else "",
         "user_id": "U_MALLORY", "channel_id": channel_id, "team_id": "T1"})
    adapter._app.client.users_info.assert_not_awaited()
    adapter._app.client.conversations_info.assert_not_awaited()
    assert adapter.handle_message.await_count == reaches_runner


@pytest.mark.asyncio
async def test_thread_routed_sender_is_checked_against_the_routed_thread():
    """A thread-scoped ``profile_routes`` entry points this thread at a profile whose pairing store
    approved the caller; the default store did not. The gateway's auth callback resolves the store
    from ``thread_id``, so the early reject must pass it on both entry points, or the caller's
    ``/stop`` and messages in that thread are dropped before the runner sees them."""
    routed_thread = "1700.000050"
    adapter = _adapter("C_OPS")
    adapter.set_authorization_check(
        lambda _uid, _chat_type, _chat_id, *, thread_id=None, **_kw: thread_id == routed_thread)
    await adapter._handle_slash_command(
        {"command": "/stop", "text": "", "user_id": "U_ALICE", "channel_id": "C_OPS",
         "team_id": "T1", "thread_ts": routed_thread})
    await adapter._handle_slack_message(
        {"text": "<@U_BOT> what broke?", "user": "U_ALICE", "channel": "C_OPS",
         "channel_type": "channel", "ts": "1700.000100", "thread_ts": routed_thread}, {"team_id": "T1"})
    handed = [c.args[0] for c in adapter.handle_message.await_args_list]
    assert [(e.text, e.source.thread_id) for e in handed] == [
        ("/stop", routed_thread), ("what broke?", routed_thread)]


@pytest.mark.asyncio
async def test_concurrent_slashes_reach_the_gateway_in_arrival_order():
    """Two ``/hermes queue`` slashes through the Bolt listener: the first is held on a cold
    ``users.info`` while the second's lookup is warm. They still reach the gateway in arrival
    order, and a ``/stop`` sent meanwhile is handed over at once, not behind the held lookup."""
    adapter, entered, release = _adapter("C_OPS"), asyncio.Event(), asyncio.Event()

    async def users_info(**_kwargs):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return {"user": {"profile": {"display_name": "Alice"}, "real_name": "Alice"}}

    adapter._app.client.users_info = AsyncMock(side_effect=users_info)
    base = {"command": "/hermes", "user_id": "U_ALICE", "channel_id": "C_OPS", "team_id": "T1"}
    first = asyncio.create_task(adapter._handle_hermes_command(AsyncMock(), {**base, "text": "queue first"}))
    await asyncio.wait_for(entered.wait(), 5)
    second = asyncio.create_task(adapter._handle_hermes_command(AsyncMock(), {**base, "text": "queue second"}))
    await asyncio.wait_for(adapter._handle_hermes_command(AsyncMock(), {**base, "text": "stop"}), 5)
    handed = lambda: [c.args[0].text for c in adapter.handle_message.await_args_list]
    assert handed() == ["/stop"]
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), 5)
    assert handed() == ["/stop", "/queue first", "/queue second"]
    assert not adapter._slash_lanes.tails
