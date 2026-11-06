"""A restart-recovered process completion is delivered once, then never replayed again.

Startup re-arms a watcher from a completion receipt whose notification never settled
(``ProcessRegistry.restore_pending_notifications``); the watcher must settle the receipt once
it is done, or every later restart would inject the same completion again.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_restored_completion_is_delivered_once_across_restarts(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    import tools.process_registry as pr_module
    from tools.process_registry import ProcessRegistry, ProcessSession

    before_restart = ProcessRegistry()
    session = ProcessSession(
        id="proc_replay0000g1", command="echo review", task_id="task-1", owner_task_id="task-1",
        session_key="agent:main:telegram:dm:123", parent_session_id="sess-owner",
        started_at=time.time(), notify_on_complete=True, watcher_interval=5,
        watcher_platform="telegram", watcher_chat_id="123", output_buffer="REVIEW DONE\n",
    )
    with before_restart._lock:
        before_restart._running[session.id] = session
    session.mark_exited(0)
    before_restart._move_to_finished(session)  # the gateway stops before its watcher delivers

    restarted = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", restarted)
    assert restarted.restore_pending_notifications() == 1

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    (tmp_path / "config.yaml").write_text("display:\n  background_process_notifications: concise\n",
                                          encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[Platform.TELEGRAM] = SimpleNamespace(send=AsyncMock(), _active_sessions={})
    runner._enqueue_process_completion_notification = AsyncMock(return_value=True)

    [watcher] = restarted.pending_watchers
    await runner._run_process_watcher(watcher)

    runner._enqueue_process_completion_notification.assert_awaited_once()
    synth_text, event = runner._enqueue_process_completion_notification.await_args.args
    assert "REVIEW DONE" in synth_text
    assert (event["session_id"], event["chat_id"], event["parent_session_id"]) == (
        "proc_replay0000g1", "123", "sess-owner")
    assert ProcessRegistry().restore_pending_notifications() == 0
