"""Explicit session-title commands synchronize only real Discord threads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import utf16_len
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_state import AsyncSessionDB, SessionDB


def _event(text: str, chat_type: str = "thread") -> MessageEvent:
    thread_id = "777" if chat_type == "thread" else None
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.DISCORD,
            user_id="12345",
            chat_id=thread_id or "67890",
            chat_type=chat_type,
            thread_id=thread_id,
            parent_chat_id="67890" if thread_id else None,
        ),
    )


def _runner(db: SessionDB, adapter, history=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = AsyncSessionDB(db)
    store = MagicMock()
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-1", session_key="discord:777")
        ),
        load_transcript=AsyncMock(return_value=history or []),
    )
    runner._run_in_executor_with_context = AsyncMock(
        side_effect=lambda function, *args: function(*args)
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._profile_adapters = {}
    runner._voice_mode = {}
    return runner


@pytest.mark.asyncio
async def test_title_renames_only_discord_threads_with_safe_name_and_keeps_failed_title(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-1", "discord")
    adapter = SimpleNamespace(rename_thread=AsyncMock(side_effect=[True, False]))
    runner = _runner(db, adapter)

    long_title = "😀" * 50 + " Discord thread work"
    success = await runner._handle_title_command(_event(f"/title {long_title}"))
    renamed_title = adapter.rename_thread.await_args_list[0].args[1]

    assert db.get_session_title("session-1") == long_title
    assert utf16_len(renamed_title) <= 80
    assert "could not be renamed" not in success

    failed_title = "Persist this despite missing Manage Threads"
    failure = await runner._handle_title_command(_event(f"/title {failed_title}"))

    assert db.get_session_title("session-1") == failed_title
    assert "Discord thread could not be renamed" in failure

    for chat_type in ("group", "dm"):
        reply = await runner._handle_title_command(_event(f"/title Keep {chat_type}", chat_type))
        assert "Discord thread could not be renamed" not in reply
    assert adapter.rename_thread.await_count == 2
    db.close()


@pytest.mark.asyncio
async def test_retitle_persists_before_reporting_discord_thread_rename_failure(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session-1", "discord")
    db.set_session_title("session-1", "Old title")
    adapter = SimpleNamespace(rename_thread=AsyncMock(return_value=False))
    runner = _runner(
        db,
        adapter,
        history=[{"role": "user", "content": "Diagnose Discord thread permissions"}],
    )

    with patch(
        "agent.session_retitle.generate_retitle",
        return_value="Persisted retitle despite Discord refusal",
    ):
        reply = await runner._handle_retitle_command(_event("/retitle"))

    assert db.get_session_title("session-1") == "Persisted retitle despite Discord refusal"
    adapter.rename_thread.assert_awaited_once_with(
        "777", "Persisted retitle despite Discord refusal"
    )
    assert "Session retitled to: Persisted retitle despite Discord refusal" in reply
    assert "Discord thread could not be renamed" in reply
    db.close()
