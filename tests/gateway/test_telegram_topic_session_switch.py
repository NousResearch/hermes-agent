"""Telegram DM topic lanes keep the session a command switched them to.

/branch --here and /resume repoint the topic's route; the persisted topic binding must follow,
or the next message's binding heal switches the topic back to the old session. Root /topic
activation enables topic mode without minting an untracked extra forum topic.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session import SessionEntry, build_session_key
from hermes_state import SessionDB
from tests.gateway.test_telegram_topic_mode import _make_event, _make_runner, _make_source

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
    switch = AsyncMock(side_effect=lambda k, sid, **_kw: _entry(k, sid))
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
