import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.run import _build_thread_title_callback, _maybe_auto_title_after_response
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_build_thread_title_callback_schedules_discord_thread_rename():
    adapter = MagicMock()
    adapter.rename_thread_title = AsyncMock(return_value=True)
    adapter.smart_thread_titles_enabled = MagicMock(return_value=True)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="thread",
        thread_id="456",
    )

    callback = _build_thread_title_callback(adapter, source, asyncio.get_running_loop())
    assert callback is not None

    callback("Smart Thread Title")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    adapter.rename_thread_title.assert_awaited_once_with("456", "Smart Thread Title")


@pytest.mark.asyncio
async def test_build_thread_title_callback_ignores_non_discord_threads():
    adapter = MagicMock()
    adapter.rename_thread_title = AsyncMock(return_value=True)
    adapter.smart_thread_titles_enabled = MagicMock(return_value=True)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="group",
        thread_id="456",
    )

    callback = _build_thread_title_callback(adapter, source, asyncio.get_running_loop())
    assert callback is None


@pytest.mark.asyncio
async def test_build_thread_title_callback_respects_disabled_setting():
    adapter = MagicMock()
    adapter.rename_thread_title = AsyncMock(return_value=True)
    adapter.smart_thread_titles_enabled = MagicMock(return_value=False)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="thread",
        thread_id="456",
    )

    callback = _build_thread_title_callback(adapter, source, asyncio.get_running_loop())
    assert callback is None


@pytest.mark.asyncio
async def test_maybe_auto_title_after_response_uses_captured_loop_from_worker_thread():
    adapter = MagicMock()
    adapter.rename_thread_title = AsyncMock(return_value=True)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="thread",
        thread_id="456",
    )
    result_holder = [{"messages": [{"role": "user", "content": "hello"}]}]
    session_db = MagicMock()
    loop = asyncio.get_running_loop()

    observed = {}

    def fake_maybe_auto_title(db, session_id, user_message, assistant_response, all_msgs, on_title=None):
        observed["db"] = db
        observed["session_id"] = session_id
        observed["user_message"] = user_message
        observed["assistant_response"] = assistant_response
        observed["all_msgs"] = all_msgs
        observed["has_callback"] = on_title is not None
        if on_title:
            on_title("Smart Thread Title")

    with patch("agent.title_generator.maybe_auto_title", side_effect=fake_maybe_auto_title):
        worker = threading.Thread(
            target=_maybe_auto_title_after_response,
            args=(session_db, adapter, source, "sess-1", "hello", "hi there", result_holder, loop),
        )
        worker.start()
        worker.join()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert observed == {
        "db": session_db,
        "session_id": "sess-1",
        "user_message": "hello",
        "assistant_response": "hi there",
        "all_msgs": [{"role": "user", "content": "hello"}],
        "has_callback": True,
    }
    adapter.rename_thread_title.assert_awaited_once_with("456", "Smart Thread Title")
