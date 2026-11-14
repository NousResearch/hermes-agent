"""Discord inbound that a planned gateway stop refused or dropped gets a turn after the restart.

Regression: ``drain_gateway_runtime`` withdraws every adapter's message handler, then the adapter
teardown flushes the Discord text-batch debounce into ``handle_message``, which found no handler
and dropped the event (logged once per adapter). Related: #146858.
"""

import asyncio
import datetime as dt
import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.Object = lambda *, id: SimpleNamespace(id=id)
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import discord
from plugins.platforms.discord.adapter import DiscordAdapter

BOT = SimpleNamespace(id=999, bot=True, display_name="Hermes", name="hermes")
STATE_FILE = ("gateway", "discord_stop_replay.json")


class FakeChannel:
    """A guild channel whose REST history and fetch_message serve a fixed message list."""

    def __init__(self, channel_id=123, messages=()):
        self.id = channel_id
        self.parent_id = None
        self.name = "inbox"
        self.guild = SimpleNamespace(id=777, name="guild")
        self.topic = None
        self.threads = []
        self.messages = list(messages)

    def history(self, **_kwargs):
        async def _gen():
            for message in sorted(self.messages, key=lambda m: m.id, reverse=True):
                yield message
        return _gen()

    async def fetch_message(self, message_id):
        for message in self.messages:
            if message.id == message_id:
                return message
        raise LookupError(message_id)


def make_message(message_id, channel, *, created_at=None, author_id=42, mention=True, content="please do it"):
    message = SimpleNamespace(
        id=message_id,
        content=f"<@{BOT.id}> {content}" if mention else content,
        reactions=[],
        author=SimpleNamespace(id=author_id, bot=False, display_name="User", name="user"),
        channel=channel,
        guild=channel.guild,
        created_at=created_at or dt.datetime.now(dt.UTC),
        attachments=[],
        mentions=[BOT] if mention else [],
        reference=None,
        type=discord.MessageType.default,
    )
    channel.messages.append(message)
    return message


def make_adapter(channel=None, extra=None):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token", extra=dict(extra or {})))
    channels = {channel.id: channel} if channel is not None else {}
    adapter._client = SimpleNamespace(user=BOT, get_channel=lambda cid: channels.get(cid), guilds=[])
    adapter._ready_event.set()
    adapter._running = True
    adapter._text_batch_delay_seconds = 0.05
    return adapter


def live_event(adapter, message):
    """The MessageEvent ``_handle_message`` builds for one live plain-text Discord message."""
    source = adapter.build_source(
        chat_id=str(message.channel.id), chat_type="group", user_id=str(message.author.id),
        message_id=str(message.id),
    )
    return MessageEvent(
        text=message.content, message_type=MessageType.TEXT, source=source, raw_message=message,
        message_id=str(message.id), timestamp=message.created_at,
    )


async def stop(adapter):
    """The planned-stop sequence: drain_gateway_runtime withdraws admission, then teardown."""
    adapter.set_message_handler(None)
    await adapter.cancel_background_tasks()


async def restart(channel, extra=None, *, allowed_user=None):
    """A fresh adapter on the same HERMES_HOME reaches on_ready and runs its reconnect recovery."""
    adapter = make_adapter(channel, extra)
    adapter._handle_message = AsyncMock(return_value=True)
    if allowed_user is not None:
        adapter._is_allowed_user = lambda user_id, *_a, **_kw: user_id == allowed_user
    await adapter._ensure_missed_message_backfill_task()
    return adapter


def dispatched_ids(adapter):
    return [call.args[0].id for call in adapter._handle_message.await_args_list]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    return tmp_path


@pytest.mark.asyncio
async def test_text_in_batch_debounce_at_stop_gets_one_turn_after_restart():
    channel = FakeChannel()
    stopping = make_adapter(channel)
    handler = AsyncMock(return_value=None)
    stopping.set_message_handler(handler)
    message = make_message(501, channel)
    stopping._enqueue_text_event(live_event(stopping, message))

    await stop(stopping)

    handler.assert_not_awaited()
    restarted = await restart(channel)
    restarted._handle_message.assert_awaited_once_with(message, role_authorized=False, recovered=True)


def state_path(home):
    return home.joinpath(*STATE_FILE)


def bot_reply_to(message):
    """A bot message replying to *message* (e.g. the drain-refusal notice)."""
    reply = SimpleNamespace(
        id=message.id + 1000, content="⏳ Gateway is restarting and is not accepting new work right now.",
        author=SimpleNamespace(id=BOT.id, bot=True), reference=SimpleNamespace(message_id=message.id),
        created_at=dt.datetime.now(dt.UTC), channel=message.channel, type=discord.MessageType.reply,
    )
    message.channel.messages.append(reply)
    return reply


@pytest.mark.asyncio
async def test_every_message_merged_into_a_flushed_batch_is_replayed_once(_env):
    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    first, second = make_message(511, channel), make_message(512, channel, mention=False)
    stopping._enqueue_text_event(live_event(stopping, first))
    stopping._enqueue_text_event(live_event(stopping, second))

    await stop(stopping)

    assert state_path(_env).stat().st_mode & 0o777 == 0o600
    restarted = await restart(channel)
    # 512 has no mention in a mention-gated channel: the live gate refuses it as it would live.
    assert dispatched_ids(restarted) == [511]
    assert not state_path(_env).exists()
    again = await restart(channel)
    assert dispatched_ids(again) == []


@pytest.mark.asyncio
async def test_window_replays_downtime_messages_but_not_those_from_before_the_stop(_env, monkeypatch):
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "123")
    channel = FakeChannel()
    now = dt.datetime.now(dt.UTC)
    # Steered into a running turn minutes before the stop: no bot reply references it.
    steered = make_message(601, channel, created_at=now - dt.timedelta(minutes=2), mention=False)
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    await stop(stopping)
    # Sent while the process was down: no live event ever reached the bot.
    downtime = make_message(602, channel, created_at=dt.datetime.now(dt.UTC), mention=False)

    restarted = await restart(channel)

    assert dispatched_ids(restarted) == [downtime.id]
    assert steered.id not in dispatched_ids(restarted)


@pytest.mark.asyncio
async def test_message_that_reached_a_session_just_before_withdrawal_is_not_replayed(_env, monkeypatch):
    monkeypatch.setenv("DISCORD_FREE_RESPONSE_CHANNELS", "123")
    channel = FakeChannel()
    stopping = make_adapter(channel)
    reached = asyncio.Event()

    async def handler(_event):
        reached.set()

    stopping.set_message_handler(handler)
    answered = make_message(611, channel, mention=False)  # inside the batch-window slice of the floor
    await stopping.handle_message(live_event(stopping, answered))
    await reached.wait()
    await stop(stopping)

    restarted = await restart(channel)

    assert dispatched_ids(restarted) == []


@pytest.mark.asyncio
async def test_drain_refused_message_gets_a_turn_although_the_bot_replied_with_the_notice(_env):
    from gateway.run_inbound_hooks import GatewayInboundHooksMixin

    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    refused = make_message(621, channel)
    event = live_event(stopping, refused)
    runner = SimpleNamespace(
        _draining=True, _status_action_gerund=lambda: "restarting",
        _intake_adapter_for=lambda source: stopping if source is event.source else None,
    )

    handled, notice, _ = await GatewayInboundHooksMixin._hm_dispatch_quick_and_plugin_commands(
        runner, event, event.source, None)
    bot_reply_to(refused)
    await stop(stopping)

    assert handled is True and notice
    restarted = await restart(channel)
    assert dispatched_ids(restarted) == [refused.id]


@pytest.mark.asyncio
async def test_window_older_than_the_bound_is_skipped(_env, caplog):
    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    stopping.defer_refused_inbound(live_event(stopping, make_message(631, channel)))
    await stop(stopping)
    state = json.loads(state_path(_env).read_text())
    state["opened_at"] -= 3600
    state["withdrawn_at"] -= 3600
    state_path(_env).write_text(json.dumps(state))

    with caplog.at_level(logging.INFO):
        restarted = await restart(channel)

    assert dispatched_ids(restarted) == []
    assert not state_path(_env).exists()
    assert "Stop-replay window skipped" in caplog.text


@pytest.mark.asyncio
async def test_replay_count_is_capped_per_start(_env):
    from plugins.platforms.discord.adapter_recovery import _MAX_REPLAYS

    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    for offset in range(_MAX_REPLAYS + 5):
        stopping.defer_refused_inbound(live_event(stopping, make_message(700 + offset, channel)))
    await stop(stopping)

    restarted = await restart(channel)

    assert len(dispatched_ids(restarted)) == _MAX_REPLAYS
    assert dispatched_ids(restarted) == sorted(dispatched_ids(restarted))


@pytest.mark.asyncio
async def test_replay_passes_the_live_authorization_gate(_env, monkeypatch):
    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    allowed, denied = make_message(641, channel), make_message(642, channel, author_id=41)
    for message in (allowed, denied):
        stopping.defer_refused_inbound(live_event(stopping, message))
    await stop(stopping)
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS")

    restarted = await restart(channel, allowed_user="42")

    assert dispatched_ids(restarted) == [allowed.id]


@pytest.mark.asyncio
async def test_a_live_event_for_a_replayed_message_does_not_run_twice(_env):
    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    replayed, live_first = make_message(651, channel), make_message(652, channel)
    for message in (replayed, live_first):
        stopping.defer_refused_inbound(live_event(stopping, message))
    await stop(stopping)

    restarted = make_adapter(channel)
    restarted._handle_message = AsyncMock(return_value=True)
    assert await restarted._dispatch_discord_message(live_first) is True  # arrives before the scan
    await restarted._ensure_missed_message_backfill_task()
    assert await restarted._dispatch_discord_message(replayed) is False  # RESUME after the replay

    assert dispatched_ids(restarted) == [live_first.id, replayed.id]


@pytest.mark.asyncio
async def test_corrupt_window_file_is_ignored_logged_and_removed(_env, caplog):
    path = state_path(_env)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    with caplog.at_level(logging.WARNING):
        restarted = await restart(FakeChannel())

    assert dispatched_ids(restarted) == []
    assert not path.exists()
    assert "Ignoring corrupt stop-replay window" in caplog.text


@pytest.mark.asyncio
async def test_switch_off_records_nothing_and_names_each_dropped_event(_env, caplog):
    channel = FakeChannel()
    stopping = make_adapter(channel, {"stop_replay": False})
    stopping.set_message_handler(AsyncMock(return_value=None))
    for message_id, author_id in ((661, 42), (662, 43)):  # two senders: two batches
        stopping._enqueue_text_event(live_event(stopping, make_message(message_id, channel, author_id=author_id)))

    with caplog.at_level(logging.WARNING):
        await stop(stopping)

    assert not state_path(_env).exists()
    dropped = [r.getMessage() for r in caplog.records if "Dropping inbound message" in r.getMessage()]
    assert len(dropped) == 2 and all("at shutdown" in line and "discord" in line for line in dropped)
    restarted = await restart(channel, {"stop_replay": False})
    assert dispatched_ids(restarted) == []


@pytest.mark.asyncio
async def test_switch_off_at_start_ignores_a_window_left_by_the_previous_process(_env):
    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    stopping.defer_refused_inbound(live_event(stopping, make_message(671, channel)))
    await stop(stopping)

    restarted = await restart(channel, {"stop_replay": "false"})

    assert dispatched_ids(restarted) == []
    assert not state_path(_env).exists()


@pytest.mark.asyncio
async def test_refusal_early_in_a_long_restart_drain_is_replayed(_env, monkeypatch):
    """The age bound is "down too long", not "the drain was long": a restart drain may wait up to
    30 min for a long turn, refusing new messages the whole time."""
    from plugins.platforms.discord import adapter_recovery

    clock = {"now": dt.datetime.now(dt.UTC).timestamp() - 31 * 60}
    monkeypatch.setattr(adapter_recovery, "time", SimpleNamespace(time=lambda: clock["now"]))
    channel = FakeChannel()
    stopping = make_adapter(channel)
    stopping.set_message_handler(AsyncMock(return_value=None))
    refused = make_message(681, channel)
    stopping.defer_refused_inbound(live_event(stopping, refused))  # the drain's first refusal
    clock["now"] += 29 * 60
    await stop(stopping)  # admission withdrawn 29 min later, inside the drain's 30 min cap
    clock["now"] += 2 * 60  # 31 min after the refusal, 2 min after the stop

    restarted = await restart(channel)

    assert dispatched_ids(restarted) == [refused.id]
