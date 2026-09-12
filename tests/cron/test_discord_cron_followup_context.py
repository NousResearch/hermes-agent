"""Offline reproduction: delivered cron context versus an active Discord turn.

Real scheduler, Discord send, SQLite session store, gateway busy entry points,
redirect buffer and redirect application. Only Discord's network boundary is
replaced; no model or Discord service is contacted. The live messages are a
snapshot taken before delivery, as in GatewayRunner._hmwa_prepare_turn.
"""

import asyncio
import copy
import threading
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from plugins.platforms.discord.adapter import DiscordAdapter
from run_agent import AIAgent

BRIEF = "Scheduled briefing: widget validation is broken; open an issue for it."
FOLLOWUP = "Aren't you opening an issue?"


@pytest.fixture(params=["dm", "thread"])
def conversation(tmp_path, monkeypatch, request):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "cron:\n  wrap_response: false\n"
        "platforms:\n  discord:\n    enabled: true\n    token: offline-test\n",
        encoding="utf-8",
    )
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True, token="offline-test")
    runner = GatewayRunner(config=config)
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="1001", chat_type=request.param, user_id="2001",
        thread_id="1001" if request.param == "thread" else None,
    )
    session = runner.session_store.get_or_create_session(source)
    history = [
        {"role": "user", "content": "Review the README pull request."},
        {"role": "assistant", "content": "The README pull request only needs a wording change."},
    ]
    for message in history:
        runner.session_store.append_to_transcript(session.session_id, message)

    # DiscordAdapter.send and create_handoff_thread remain real. DMs cannot
    # host a child thread; an explicitly targeted existing thread is retained.
    import discord

    channel = MagicMock(spec=discord.DMChannel if request.param == "dm" else discord.Thread)
    channel.id = int(source.chat_id)
    channel.send = AsyncMock(return_value=SimpleNamespace(id=3001))
    adapter = DiscordAdapter(config.platforms[Platform.DISCORD])
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel)
    adapter.set_session_store(runner.session_store)
    runner.adapters[Platform.DISCORD] = adapter
    assert get_hermes_home() == tmp_path
    return runner, source, session, adapter, channel, history


async def deliver(conversation, attach):
    runner, source, session, adapter, channel, history = conversation
    job = {"id": "offline-brief", "name": "Widget briefing", "deliver": "origin", "origin": asdict(source)}
    job["origin"]["platform"] = source.platform.value
    if attach is not None:
        job["attach_to_session"] = attach
    error = await asyncio.to_thread(
        _deliver_result, job, BRIEF, {Platform.DISCORD: adapter}, asyncio.get_running_loop(),
    )
    assert error is None
    channel.send.assert_awaited_once()
    assert channel.send.await_args.kwargs["content"] == BRIEF
    reply_session = runner.session_store.get_or_create_session(source)
    assert reply_session.session_id == session.session_id
    return runner.session_store.load_transcript(reply_session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("attach", [None, False, True])
async def test_idle_followup_observes_only_opted_in_delivery(conversation, attach):
    transcript = await deliver(conversation, attach)
    assert (BRIEF in str(transcript)) is (attach is True)
    assert [{key: message[key] for key in ("role", "content")} for message in transcript[:2]] == conversation[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["normal", "priority"])
async def test_active_followup_observes_opted_in_delivery(conversation, route):
    from agent.conversation_loop import _apply_active_turn_redirect

    runner, source, session, adapter, channel, history = conversation
    runner.session_store.append_to_transcript(
        session.session_id,
        {"role": "user", "content": "Continue reviewing that README pull request."},
    )
    live_messages = copy.deepcopy(runner.session_store.load_transcript(session.session_id))
    cached_prefix = copy.deepcopy(live_messages)
    transcript = await deliver(conversation, True)
    assert BRIEF in str(transcript), "The opted-in brief must first reach the exact reply-facing session"
    assert BRIEF not in str(live_messages), "Delivery must not mutate a running request"

    # Run the real AIAgent redirect implementation without constructing a
    # provider client. No model call is made; the request-active event is the
    # deterministic pause point for the in-flight snapshot.
    agent = AIAgent.__new__(AIAgent)
    agent._supports_active_turn_redirect = True
    agent._model_request_active = threading.Event()
    agent._model_request_active.set()
    agent._interrupt_requested = False
    agent._current_streamed_assistant_text = "The README wording can be improved."
    event = MessageEvent(text=FOLLOWUP, source=source, message_id="4001")
    assert event.reply_to_message_id is None and event.reply_to_text is None
    if route == "normal":
        outcome = await runner._resolve_busy_steer_or_redirect(event, session.session_key, "interrupt", agent)
        assert outcome.redirected
    else:
        await runner._hm_busy_interrupt(event, source, agent, session.session_key)
    correction = agent._drain_pending_redirect()
    assert correction and FOLLOWUP in correction
    _apply_active_turn_redirect(agent, live_messages, correction)
    assert live_messages[:len(cached_prefix)] == cached_prefix
    assert all(a["role"] != b["role"] for a, b in zip(live_messages, live_messages[1:]))
    assert BRIEF in str(live_messages), (
        "Successful opted-in Discord delivery is durable, but the accepted active follow-up "
        "only carries the README snapshot and the user's bare question, not the cron brief"
    )
