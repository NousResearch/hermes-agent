"""Concurrent reconnect paths share a synchronously reserved resume slot."""

import asyncio
import logging

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform
from gateway.run_reconnect_recovery import (
    recover_reconnected_sessions,
    schedule_recovery,
)
from tests.gateway.test_recovery_continuation import (
    add_session,
    continuation_runner,
    settle,
)
from tests.gateway.test_relay_ledger_recovery import record


@pytest.mark.asyncio
@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("profile", ["default", "secondary"])
async def test_inline_and_worker_recovery_reserve_exactly_one_resume_before_task_start(
    tmp_path, monkeypatch, relay, profile
):
    runner, adapter, _, resumed = continuation_runner(
        tmp_path, monkeypatch, relay=relay, profile=profile
    )
    entry = add_session(runner, "unanswered", owner=profile)
    entered, release = asyncio.Event(), asyncio.Event()
    dispatch = runner._run_startup_resume_event

    async def slow_dispatch(live, event, key):
        assert runner._is_session_running(key)
        entered.set()
        await release.wait()
        await dispatch(live, event, key)

    runner._run_startup_resume_event = slow_dispatch
    platform = Platform.RELAY if relay else Platform.SLACK
    inline = asyncio.create_task(
        recover_reconnected_sessions(runner, platform, profile)
    )
    schedule_recovery(runner, platform, profile=profile)
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        await inline
        for task, _ in list(runner._reconnect_recovery_tasks.values()):
            await task
        await recover_reconnected_sessions(runner, platform, profile)
        resumes = [task for task in runner._background_tasks if not task.done()]
        assert len(resumes) == 1
        assert runner._is_session_running(entry.session_key)
        assert not resumed
    finally:
        release.set()
        await inline
        await settle(runner)
    assert resumed == [(adapter, entry.session_key)]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["dead_chat", "expired", "exhausted"])
async def test_nonretryable_saved_answer_is_not_regenerated_and_reports_its_hold(
    tmp_path, monkeypatch, caplog, case
):
    runner, _, _, resumed = continuation_runner(tmp_path, monkeypatch)
    entry = add_session(runner, "chat")
    record()
    ledger.mark_failed("owed", "Forbidden: bot was blocked by the user")
    if case != "dead_chat":
        with ledger._transaction() as db:
            if case == "expired":
                db.execute("UPDATE delivery_obligations SET created_at=0")
            else:
                db.execute(
                    "UPDATE delivery_obligations SET attempts=?", (ledger.MAX_ATTEMPTS,)
                )
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        assert runner._schedule_resume_pending_sessions(platform=Platform.SLACK) == 0
    await settle(runner)
    assert not resumed
    assert entry.session_key in caplog.text
    assert "a saved reply still owns delivery" in caplog.text
