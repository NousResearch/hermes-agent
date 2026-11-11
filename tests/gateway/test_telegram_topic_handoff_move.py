"""A handoff moves its conversation into a Telegram DM topic; a lost race never splits a topic.

Real ``SessionStore`` and its store: a topic's route and its persisted binding must agree after
every handoff, ``/resume`` or refusal, and no two topics may route to one session.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import HomeChannel, Platform
from gateway.session import SessionStore, build_session_key
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.test_telegram_topic_mode import _make_event, _make_runner, _make_source

CHAT = "208214988"


def _real_runner(tmp_path):
    runner = _make_runner()
    store = SessionStore(tmp_path / "sessions", runner.config)
    runner.session_store, runner._async_session_store = store, None
    db = store.persistence
    db.routing.enable_topic_mode(chat_id=CHAT, user_id=CHAT)
    runner.persistence = runner._session_db = db
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=CHAT, name="DM", thread_id=None)
    runner._handle_message = AsyncMock(return_value=None)
    runner._clear_conversation_scope = MagicMock()
    seed_session(db, "cli-a", "cli")
    seed_messages(db, "cli-a", {"role": "user", "content": "cli work"})
    return runner, store, db


def _topic(store, db, thread_id):
    """(route, binding) of one topic."""
    binding = db.routing.get_topic_binding(chat_id=CHAT, thread_id=thread_id)
    return store.peek_session_id(build_session_key(_make_source(thread_id=thread_id))), binding and binding.session_id


async def _handoff(runner):
    return (await asyncio.gather(runner._process_handoff(
        {"id": "cli-a", "title": "A", "handoff_platform": "telegram"}), return_exceptions=True))[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("bound_in", ["same_profile", "other_profile"])
async def test_repeat_handoff_moves_the_topic_and_a_refusal_opens_no_topic(tmp_path, bound_in):
    """The documented round trip (hand off into topic 701, /resume on the CLI, hand off again) lands
    the conversation in the new topic 702 and releases 701, so only one topic routes to it. A
    binding a handoff may not move (another profile's topic) is refused before any topic opens."""
    runner, store, db = _real_runner(tmp_path)
    create = runner.adapters[Platform.TELEGRAM].create_handoff_thread = AsyncMock(side_effect=["701", "702"])
    if bound_in == "same_profile":
        assert await _handoff(runner) is None
    else:
        db.routing.bind_topic(chat_id=CHAT, thread_id="900", user_id=CHAT, session_key="agent:work:x",
                              session_id="cli-a", profile_name="work")

    result = await _handoff(runner)

    if bound_in == "same_profile":
        assert result is None and runner._handle_message.await_count == 2
        assert _topic(store, db, "702") == ("cli-a", "cli-a")
        assert _topic(store, db, "701") == (None, None)
        # 701 starts a new conversation: its /model, /reasoning etc. overrides go with the old one.
        assert runner._clear_conversation_scope.call_args_list[-1].args == (
            build_session_key(_make_source(thread_id="701")),)
    else:
        assert "already linked" in str(result)
        create.assert_not_awaited()
        runner._handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["resume", "handoff"])
async def test_binding_race_after_the_check_keeps_route_and_binding_together(tmp_path, command):
    """The session gets bound to topic 999 between a command's check and its binding write. /resume
    puts the route back and answers with the refusal; a handoff moves the binding with it. Either
    way topic 800's route and binding name one session, and the user gets an answer."""
    runner, store, db = _real_runner(tmp_path)
    src = _make_source(thread_id="800")
    key = build_session_key(src)
    own = store.get_or_create_session(src).session_id
    db.routing.bind_topic(chat_id=CHAT, thread_id="800", user_id=CHAT, session_key=key, session_id=own)
    race_key = build_session_key(_make_source(thread_id="999"))

    def bind_elsewhere(session_id):
        db.routing.bind_topic(chat_id=CHAT, thread_id="999", user_id=CHAT, session_key=race_key,
                              session_id=session_id)

    if command == "resume":
        seed_session(db, "other", "telegram", title="Other Work", user_id=CHAT, chat_id=CHAT, chat_type="dm",
                     thread_id="800", session_key=key)
        check = runner._telegram_topic_switch_conflict

        async def check_then_race(source, session_id):
            verdict = await check(source, session_id)
            bind_elsewhere(session_id)
            return verdict

        runner._telegram_topic_switch_conflict = check_then_race
        reply = await runner._handle_resume_command(_make_event("/resume Other Work", thread_id="800"))
        assert "already linked" in reply
        assert _topic(store, db, "800") == (own, own)
        assert session_row(db, own)["ended_at"] is None
        return

    runner.config.platforms[Platform.TELEGRAM].home_channel.thread_id = "800"
    runner.adapters[Platform.TELEGRAM].create_handoff_thread = AsyncMock(return_value=None)
    repoint = runner._handoff_repoint

    async def repoint_then_race(*args):
        entry = await repoint(*args)
        bind_elsewhere("cli-a")
        return entry

    runner._handoff_repoint = repoint_then_race
    assert await _handoff(runner) is None
    runner._handle_message.assert_awaited_once()
    assert _topic(store, db, "800") == ("cli-a", "cli-a")
    assert db.routing.get_topic_binding(chat_id=CHAT, thread_id="999") is None
