"""Recovery is an interruption-scoped wake, not a new human instruction."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import build_resume_recovery_note
from gateway.session import SessionEntry
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def setup_pending():
    runner, adapter = make_restart_runner()
    # Disable the restart-loop breaker: it is unrelated to supersession, but
    # _resume_pending_candidates records a boot on EVERY call that has candidates, and these
    # tests schedule more than once. Left on, the breaker trips on the tests' own bookkeeping
    # (default threshold 3) and starves the scheduler of candidates, so a genuinely new
    # interruption looks like a suppressed duplicate. ``max_restarts <= 0`` is the documented
    # disable — see GatewayRunner._restart_loop_guard_config.
    runner._restart_loop_guard_config = lambda: (0, 3600, 300)
    source = make_restart_source()
    key = runner._session_key_for_source(source)
    entry = SessionEntry(session_key=key, session_id="sid", created_at=datetime.now(),
                         updated_at=datetime.now(), origin=source, resume_pending=True,
                         resume_reason="restart_timeout", last_resume_marked_at=datetime.now())
    runner.session_store._entries = {key: entry}
    adapter.handle_message = AsyncMock()
    return runner, adapter, entry


@pytest.mark.asyncio
async def test_reconnect_does_not_repeat_same_interruption_after_failed_turn():
    runner, adapter, entry = setup_pending()
    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*list(runner._background_tasks))
    assert entry.resume_pending  # retain real-human recovery safety
    assert runner._schedule_resume_pending_sessions() == 0
    assert adapter.handle_message.await_count == 1
    entry.last_resume_marked_at += timedelta(seconds=1)
    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*list(runner._background_tasks))
    assert adapter.handle_message.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cleared", "suspended", "new_session", "human"])
async def test_scheduled_recovery_revalidates_before_dispatch(change):
    runner, adapter, entry = setup_pending()
    assert runner._schedule_resume_pending_sessions() == 1
    if change == "cleared":
        entry.resume_pending = False
    elif change == "suspended":
        entry.suspended = True
    elif change == "new_session":
        entry.session_id = "replacement"
    else:
        human = MessageEvent(text="/stop", message_type=MessageType.COMMAND, source=entry.origin)
        runner._queue_startup_restore_event(human)
    await asyncio.gather(*list(runner._background_tasks))
    adapter.handle_message.assert_not_awaited()
    assert not runner._is_session_running(entry.session_key)
    if change == "human":
        assert await runner._drain_startup_restore_queue() == 1
        assert adapter.handle_message.await_args.args[0] is human


def test_new_instruction_can_explicitly_resume_unfinished_work():
    text = "Continue the unfinished task, checking existing outputs first."
    note = build_resume_recovery_note("restart_timeout", text)
    assert text in note
    assert "skip any unfinished work" not in note
    assert "Do NOT re-execute" in note
