"""A Discord turn for an agent without the discord toolset carries the ids of a message jump URL.

Without the toolset the model never saw the guild, channel or message id, so it could not cite or
link the message that triggered the turn. The line rides the per-turn user message and is never
persisted as the user row's content.
"""
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_inbound import discord_triggering_note
from gateway.session import SessionSource

IDS = "[Message ids] guild=900 channel=700 message=5500 link=https://discord.com/channels/900/700/5500"


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    return runner


def _source(*, platform=Platform.DISCORD, chat_id="700", chat_type="group", guild_id="900", thread_id=None):
    return SessionSource(
        platform=platform, chat_id=chat_id, chat_type=chat_type, user_id="u1", user_name="Owen",
        guild_id=guild_id, thread_id=thread_id,
    )


async def _prepare(runner, event, source):
    return await runner._prepare_inbound_message_text(event=event, source=source, history=[])


@pytest.mark.asyncio
async def test_guild_turn_without_discord_tools_carries_the_id_line(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    runner, source = _runner(), _source()
    event = MessageEvent(text="what is this?", source=source, message_id="5500")

    model_text = await _prepare(runner, event, source)

    assert model_text == f"{IDS}\n\nwhat is this?"


@pytest.mark.asyncio
async def test_id_line_is_outermost_and_survives_channel_context_and_reply(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    runner, source = _runner(), _source()
    event = MessageEvent(
        text="yes", source=source, message_id="5500", channel_context="[Recent channel messages]\n[Bob] hi",
        reply_to_message_id="5499", reply_to_text="earlier",
    )

    model_text = await _prepare(runner, event, source)

    assert model_text.startswith(f'{IDS}\n\n[Replying to: "earlier"]\n\n')
    assert "[Recent channel messages]\n[Bob] hi\n\n[New message]\nyes" in model_text


@pytest.mark.asyncio
async def test_discord_tools_desk_keeps_the_existing_note_and_gets_no_id_line(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: True)
    runner, source = _runner(), _source()
    event = MessageEvent(text="hi", source=source, message_id="5500")

    model_text = await _prepare(runner, event, source)

    assert model_text == f"{discord_triggering_note('5500')}\n\nhi"
    assert "[Message ids]" not in model_text


@pytest.mark.asyncio
async def test_dm_uses_at_me_as_the_guild(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    runner = _runner()
    source = _source(chat_id="321", chat_type="dm", guild_id=None)
    event = MessageEvent(text="hi", source=source, message_id="5500")

    model_text = await _prepare(runner, event, source)

    assert model_text.startswith(
        "[Message ids] guild=@me channel=321 message=5500 link=https://discord.com/channels/@me/321/5500"
    )


@pytest.mark.asyncio
async def test_thread_turn_links_the_thread_channel(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    runner = _runner()
    source = _source(chat_id="7777", chat_type="thread", thread_id="7777")
    event = MessageEvent(text="hi", source=source, message_id="5500", raw_message=SimpleNamespace(channel=SimpleNamespace(id=7777)))

    model_text = await _prepare(runner, event, source)

    assert "channel=7777 message=5500 link=https://discord.com/channels/900/7777/5500" in model_text


@pytest.mark.asyncio
async def test_auto_created_thread_links_the_parent_channel_the_message_lives_in(monkeypatch):
    """chat_id is the NEW thread; the triggering message is still in the parent channel."""
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    runner = _runner()
    source = _source(chat_id="8888", chat_type="thread", thread_id="8888")
    event = MessageEvent(text="hi", source=source, message_id="5500", raw_message=SimpleNamespace(channel=SimpleNamespace(id=700)))

    model_text = await _prepare(runner, event, source)

    assert "channel=700 message=5500 link=https://discord.com/channels/900/700/5500" in model_text


@pytest.mark.asyncio
async def test_turn_without_a_message_id_and_other_platforms_are_unchanged(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    runner, source = _runner(), _source()
    no_id = MessageEvent(text="slash turn", source=source, message_id=None)
    assert await _prepare(runner, no_id, source) == "slash turn"

    tg_source = _source(platform=Platform.TELEGRAM, chat_id="5", chat_type="dm", guild_id=None)
    tg_event = MessageEvent(text="hello", source=tg_source, message_id="9")
    assert await _prepare(runner, tg_event, tg_source) == "hello"


@pytest.mark.asyncio
async def test_id_line_is_model_only_and_never_persisted(monkeypatch):
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: False)
    runner, source = _runner(), _source()
    event = MessageEvent(text="what is this?", source=source, message_id="5500")

    model_text = await _prepare(runner, event, source)
    message_text, persist_user_message, _ts = runner._hmwa_apply_message_timestamp(event, model_text)

    assert message_text.startswith(IDS)
    assert persist_user_message == "what is this?"
