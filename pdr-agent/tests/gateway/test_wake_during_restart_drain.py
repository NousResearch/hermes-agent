"""A wake that arrives while the gateway drains for a restart must be refused, not lost.

A draining gateway answers new work with a drain notice and discards queued follow-ups, so a
wake it admitted would never run, while the Kanban notifier had already advanced its cursor.
"""

import asyncio
import threading

import pytest

from gateway.config import Platform
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.platforms.event import MessageEvent
from gateway.wake import WakeNotAccepted, deliver_wake
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


async def drain(adapter):
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
        await asyncio.sleep(0)


def wake_route():
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner
    adapter.set_message_handler(runner._handle_message)
    runner._delivery_adapter_for = lambda source: adapter
    runner._kanban_dispatcher_lock_handle = object()
    turns = []

    async def agent_turn(event, source, quick_key, run_generation):
        turns.append(event.text)

    runner._handle_message_with_agent = agent_turn
    return runner, adapter, make_restart_source(), turns


@pytest.mark.asyncio
async def test_kanban_wake_during_restart_drain_runs_after_restart(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    runner, adapter, source, turns = wake_route()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="long task", assignee="worker")
        kbn.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=source.chat_id,
            user_id=source.user_id,
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="PR merged")

    async def notifier_tick():
        deliveries = await asyncio.to_thread(
            _notifier_collect,
            runner,
            kb,
            notifier_profile=None,
            gc_due=False,
            gc_retention_days=30,
        )
        for delivery in deliveries:
            await _KanbanNotification(
                runner, delivery, platform_cls=Platform, sub_fail_counts={}
            ).deliver()
        await drain(adapter)

    def undelivered():
        with kbc.connect() as conn:
            return kbn.unseen_events_for_sub(
                conn,
                task_id=tid,
                platform="telegram",
                chat_id=source.chat_id,
                kinds=["completed"],
            )[1]

    runner._draining = True  # /restart is waiting for other sessions to finish
    await notifier_tick()
    assert turns == []
    assert undelivered(), "the completion must stay claimable for the restarted gateway"
    pings = list(adapter.sent)
    assert len(pings) == 1 and tid in pings[0]  # the passive ping still lands once

    runner._draining = False  # the restarted gateway's notifier retries the wake
    await notifier_tick()
    assert len(turns) == 1 and tid in turns[0]
    assert not undelivered()
    assert adapter.sent == pings  # no repeated ping, no drain notice


@pytest.mark.asyncio
async def test_wake_for_a_busy_session_is_refused_while_draining():
    runner, adapter, source, _turns = wake_route()
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    adapter._active_sessions[key] = (
        asyncio.Event()
    )  # a turn is still running in this chat
    runner._draining = True
    with pytest.raises(WakeNotAccepted) as refused:
        await deliver_wake(adapter, text="[kanban] Task t_1 completed.", source=source)
    assert refused.value.reason == "runtime_draining" and not refused.value.permanent
    assert key not in adapter._pending_messages  # would be discarded when the turn ends


@pytest.mark.asyncio
async def test_wake_is_refused_when_drain_starts_during_admission():
    runner, adapter, source, turns = wake_route()
    entered, release = threading.Event(), threading.Event()

    def slow_topic_recovery(
        _source,
    ):  # Telegram DM admission awaits this before it claims
        entered.set()
        release.wait(5)

    adapter._topic_recovery_fn = slow_topic_recovery
    admission = asyncio.create_task(
        deliver_wake(adapter, text="[kanban] Task t_1 completed.", source=source)
    )
    assert await asyncio.to_thread(entered.wait, 5)
    runner._draining = True  # /restart begins while admission is parked
    release.set()
    with pytest.raises(WakeNotAccepted) as refused:
        await admission
    assert refused.value.reason == "runtime_draining"
    await drain(adapter)
    assert turns == []


@pytest.mark.asyncio
async def test_wake_is_refused_when_drain_starts_after_the_busy_handler_queued_it():
    """Guard for a busy handler that awaits after it stores the wake (no in-tree handler does today):
    a drain that begins there dooms the queued follow-up, so the wake must be refused and retried,
    not receipted and discarded."""
    runner, adapter, source, _turns = wake_route()
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    adapter._active_sessions[key] = asyncio.Event()  # a turn is still running in this chat
    store = adapter._busy_session_handler

    async def store_then_drain(event, session_key):
        handled = await store(event, session_key)  # the wake is now the pending follow-up
        runner._draining = True  # /restart begins after the store, before the handler returns
        return handled

    adapter._busy_session_handler = store_then_drain
    with pytest.raises(WakeNotAccepted) as refused:
        await deliver_wake(adapter, text="[kanban] Task t_1 completed.", source=source)
    assert "during admission" in str(refused.value)  # the post-admission re-check fired
    event, text = await runner._run_agent_drain_pending(
        {"final_response": "done"}, adapter, source, key
    )
    assert (event, text) == (None, None)  # discarded by the drain, so it runs only on the retry
