"""Telegram DM topic lanes keep the session a command switched them to.

/branch --here and /resume repoint the topic's route; the persisted topic binding must follow,
or the next message's binding heal switches the topic back to the old session. Root /topic
activation enables topic mode without minting an untracked extra forum topic.
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import HomeChannel, Platform
from gateway.session import SessionEntry, build_session_key
from hermes_state import SessionDB
from tests.gateway.test_telegram_topic_mode import _make_event, _make_runner, _make_source
from tests.waiting import async_wait_until

CHAT = "208214988"
THREAD = "17585"


def _entry(key: str, session_id: str) -> SessionEntry:
    return SessionEntry(session_key=key, session_id=session_id, created_at=datetime.now(),
                        updated_at=datetime.now(), origin=_make_source(thread_id=THREAD))


def _bound_topic(tmp_path, *extra_sessions: str):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id=CHAT, user_id=CHAT)
    key = build_session_key(_make_source(thread_id=THREAD))
    for sid in ("parent", *extra_sessions):
        db.create_session(session_id=sid, source="telegram", user_id=CHAT, chat_id=CHAT,
                          chat_type="dm", thread_id=THREAD, session_key=key)
    db.append_message("parent", "user", "research ozon")
    db.append_message("parent", "assistant", "working")
    db.bind_telegram_topic(chat_id=CHAT, thread_id=THREAD, user_id=CHAT, session_key=key, session_id="parent")
    runner = _make_runner(session_db=db)
    routes = {key: "parent"}
    runner.session_store.peek_session_id.side_effect = routes.get

    def _switch(k, sid, **_kw):
        routes[k] = sid
        return _entry(k, sid)

    switch = AsyncMock(side_effect=_switch)
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=_entry(key, "parent")),
        load_transcript=AsyncMock(return_value=[{"role": "user", "content": "research ozon"},
                                                {"role": "assistant", "content": "working"}]),
        switch_session=switch, set_model_override=AsyncMock(),
    )
    runner._clear_conversation_scope = lambda *_a, **_kw: None
    return db, runner


def _bound_session(db) -> str:
    return db.get_telegram_topic_binding(chat_id=CHAT, thread_id=THREAD)["session_id"]


@pytest.mark.asyncio
async def test_branch_here_in_topic_rebinds_topic_and_renames_it(tmp_path):
    db, runner = _bound_topic(tmp_path)

    reply = await runner._handle_branch_command(_make_event("/branch --here ozon", thread_id=THREAD))

    branch_id = runner.async_session_store.switch_session.await_args.args[1]
    assert "ozon" in reply and branch_id != "parent"
    assert _bound_session(db) == branch_id
    runner.adapters[next(iter(runner.adapters))].rename_dm_topic.assert_awaited_once_with(
        chat_id=CHAT, thread_id=THREAD, name="ozon")


@pytest.mark.asyncio
async def test_resume_in_topic_rebinds_topic_to_resumed_session(tmp_path):
    db, runner = _bound_topic(tmp_path, "earlier")
    db.set_session_title("earlier", "Earlier Topic Work")

    reply = await runner._handle_resume_command(_make_event("/resume Earlier Topic Work", thread_id=THREAD))

    assert "Earlier Topic Work" in reply
    assert _bound_session(db) == "earlier"
    runner.adapters[next(iter(runner.adapters))].rename_dm_topic.assert_awaited_once_with(
        chat_id=CHAT, thread_id=THREAD, name="Earlier Topic Work")


@pytest.mark.asyncio
async def test_root_topic_activation_creates_no_extra_topic(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner(session_db=SessionDB(db_path=tmp_path / "state.db"))
    runner._run_agent = AsyncMock(side_effect=AssertionError("/topic must not enter the agent loop"))
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    reply = await runner._handle_message(_make_event("/topic"))

    assert "multi-session topics are enabled" in reply
    adapter = runner.adapters[next(iter(runner.adapters))]
    adapter._create_dm_topic.assert_not_called()
    adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_resume_rename_obeys_receiving_bot_kill_switch(tmp_path):
    """``disable_topic_auto_rename`` is the RECEIVING bot's setting: a secondary bot that opted out
    keeps its topic name on /resume even though the launch profile's Telegram config never set it."""
    from gateway.config import PlatformConfig

    db, runner = _bound_topic(tmp_path, "earlier")
    db.set_session_title("earlier", "Earlier Topic Work")
    adapter = runner.adapters[next(iter(runner.adapters))]
    adapter.config = PlatformConfig(enabled=True, token="***", extra={"disable_topic_auto_rename": True})

    await runner._handle_resume_command(_make_event("/resume Earlier Topic Work", thread_id=THREAD))

    assert _bound_session(db) == "earlier"
    adapter.rename_dm_topic.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_of_session_bound_to_another_topic_is_refused_before_switching(tmp_path):
    """A session links to one topic only: /resume of one bound elsewhere must refuse up front, not
    switch the route, fail the binding write and snap back to the old conversation next message."""
    db, runner = _bound_topic(tmp_path, "elsewhere")
    db.set_session_title("elsewhere", "Other Topic Work")
    db.bind_telegram_topic(chat_id=CHAT, thread_id="999", user_id=CHAT, session_key="other-key",
                           session_id="elsewhere")

    reply = await runner._handle_resume_command(_make_event("/resume Other Topic Work", thread_id=THREAD))

    assert "already linked" in reply
    runner.async_session_store.switch_session.assert_not_awaited()
    assert _bound_session(db) == "parent"


@pytest.mark.asyncio
async def test_older_resume_finishing_last_does_not_undo_newer_switch(tmp_path):
    """Commands dispatch concurrently: /resume A switches the route and is suspended, /resume B
    switches and binds, then A finishes. The topic must stay bound to B, the route's session."""
    db, runner = _bound_topic(tmp_path, "older", "newer")
    db.set_session_title("older", "Older Work")
    db.set_session_title("newer", "Newer Work")
    a_parked, release_a = asyncio.Event(), asyncio.Event()

    async def park_first_resume(_key, _value):
        if not a_parked.is_set():
            a_parked.set()
            await release_a.wait()

    runner.async_session_store.set_model_override = AsyncMock(side_effect=park_first_resume)
    older = asyncio.create_task(runner._handle_resume_command(_make_event("/resume Older Work", thread_id=THREAD)))
    await a_parked.wait()
    runner.async_session_store.get_or_create_session.return_value = _entry(
        build_session_key(_make_source(thread_id=THREAD)), "older")
    await runner._handle_resume_command(_make_event("/resume Newer Work", thread_id=THREAD))
    assert _bound_session(db) == "newer"
    release_a.set()
    await older

    assert _bound_session(db) == "newer"


@pytest.mark.asyncio
async def test_older_resume_parked_in_its_rename_does_not_land_its_title_last(tmp_path):
    """/resume A binds and is parked inside its topic rename while /resume B switches: B's rename
    must land after A's, so the topic ends named for B, the session it is bound to."""
    db, runner = _bound_topic(tmp_path, "older", "newer")
    db.set_session_title("older", "Older Work")
    db.set_session_title("newer", "Newer Work")
    adapter = runner.adapters[next(iter(runner.adapters))]
    a_renaming, release_a, landed = asyncio.Event(), asyncio.Event(), []

    async def rename(*, chat_id, thread_id, name):
        if not a_renaming.is_set():
            a_renaming.set()
            await release_a.wait()
        landed.append(name)

    adapter.rename_dm_topic = AsyncMock(side_effect=rename)
    older = asyncio.create_task(runner._handle_resume_command(_make_event("/resume Older Work", thread_id=THREAD)))
    await a_renaming.wait()
    runner.async_session_store.get_or_create_session.return_value = _entry(
        build_session_key(_make_source(thread_id=THREAD)), "older")
    newer = asyncio.create_task(runner._handle_resume_command(_make_event("/resume Newer Work", thread_id=THREAD)))
    topic = (CHAT, THREAD)
    # B has either renamed already (no per-topic ordering) or is queued behind A's switch.
    await async_wait_until(
        lambda: landed or runner.__dict__.get("_topic_switch_locks", {}).get(topic, [None, 0])[1] == 2,
        what="the newer /resume to rename or queue behind the older one")
    release_a.set()
    await asyncio.gather(older, newer)

    assert _bound_session(db) == "newer"
    assert landed[-1] == "Newer Work"


@pytest.mark.asyncio
async def test_message_while_newer_resume_waits_on_older_rename_keeps_newer_session(tmp_path):
    """/resume A is parked in its remote topic rename when /resume B switches the route; a normal
    message's binding heal in that window must not see B's route with A's binding and switch the
    topic back to A. Both commands reply success, so the topic ends on B: route, binding and title."""
    db, runner = _bound_topic(tmp_path, "older", "newer")
    db.set_session_title("older", "Older Work")
    db.set_session_title("newer", "Newer Work")
    adapter = runner.adapters[next(iter(runner.adapters))]
    a_renaming, release_a, landed = asyncio.Event(), asyncio.Event(), []

    async def rename(*, chat_id, thread_id, name):
        if name == "Older Work":
            a_renaming.set()
            await release_a.wait()
        landed.append(name)

    adapter.rename_dm_topic = AsyncMock(side_effect=rename)
    older = asyncio.create_task(runner._handle_resume_command(_make_event("/resume Older Work", thread_id=THREAD)))
    await asyncio.wait_for(a_renaming.wait(), 10)
    newer = asyncio.create_task(runner._handle_resume_command(_make_event("/resume Newer Work", thread_id=THREAD)))
    key = build_session_key(_make_source(thread_id=THREAD))
    await async_wait_until(lambda: runner.session_store.peek_session_id(key) == "newer"
                           and runner.__dict__.get("_topic_switch_locks", {}).get((CHAT, THREAD), [None, 0])[1] == 2,
                           what="the newer /resume to switch and queue behind the older rename")

    await runner._hmwa_heal_telegram_topic_binding(_make_source(thread_id=THREAD), _entry(key, "newer"), key)
    release_a.set()
    await asyncio.gather(older, newer)

    assert runner.session_store.peek_session_id(key) == "newer"
    assert _bound_session(db) == "newer"
    assert landed[-1] == "Newer Work"


@pytest.mark.asyncio
async def test_handoff_suspended_after_its_switch_does_not_undo_a_newer_resume(tmp_path):
    """A handoff repoints the topic at the CLI session and is suspended before its binding write;
    a /resume finishing meanwhile owns the topic. The handoff must neither rebind the topic to the
    CLI session nor run its synthetic turn in the resumed conversation: it fails instead."""
    db, runner = _bound_topic(tmp_path, "cli-session", "newer")
    db.set_session_title("newer", "Newer Work")
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=CHAT, name="Test DM", thread_id=THREAD)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.create_handoff_thread = AsyncMock(return_value=None)
    runner._handle_message = AsyncMock(return_value="handoff ok")
    repoint, parked, release = runner._handoff_repoint, asyncio.Event(), asyncio.Event()

    async def park_after_repoint(*args):
        entry = await repoint(*args)
        parked.set()
        await release.wait()
        return entry

    runner._handoff_repoint = park_after_repoint
    handoff = asyncio.create_task(runner._process_handoff(
        {"id": "cli-session", "title": "CLI work", "handoff_platform": "telegram"}))
    await asyncio.wait_for(parked.wait(), 10)
    await runner._handle_resume_command(_make_event("/resume Newer Work", thread_id=THREAD))
    release.set()

    with pytest.raises(RuntimeError, match="switched to another session"):
        await asyncio.wait_for(handoff, 10)
    key = build_session_key(_make_source(thread_id=THREAD))
    assert runner.session_store.peek_session_id(key) == _bound_session(db) == "newer"
    runner._handle_message.assert_not_awaited()
