"""Saved side work participates in existing drain accounting and owner listing."""
import asyncio
import threading

import pytest

from gateway.side_runs import SideRunService
from hermes_cli.plugin_side_runs import SideRunConfig
from hermes_cli.session_listing import query_session_listing
from tests.gateway.test_plugin_side_run_failures import fixture_runner


@pytest.mark.asyncio
async def test_child_is_listable_but_execution_approval_key_is_independent(tmp_path, monkeypatch):
    runner, db, _, event = fixture_runner(tmp_path)
    approval_keys = []

    class Agent:
        def __init__(self, **kwargs):
            self.provider, self.model = kwargs["provider"], kwargs["model"]
            approval_keys.append(kwargs["gateway_session_key"])
        def run_conversation(self, **kwargs):
            return {"final_response": "answer"}
        def close(self):
            pass

    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {"provider": "openai"})
    service = SideRunService(runner)
    sid = service.start(event, "fixture", "prompt", SideRunConfig.from_mapping({"provider": "openai", "model": "fixture"}))
    try:
        await service.wait()
        rows = query_session_listing(db, source="telegram", session_key="parent", include_unnamed=True)
        assert sid in {row["id"] for row in rows}
        assert approval_keys == ["plugin-side:" + sid]
    finally:
        runner._shutdown_executor()
        db.close()


@pytest.mark.asyncio
async def test_shutdown_accounts_for_child_and_does_not_wait_forever_on_supervisor(tmp_path, monkeypatch):
    runner, db, _, event = fixture_runner(tmp_path)
    runner._running_agents = {}
    runner._running_agent_count = lambda: 0
    runner._snapshot_running_agents = lambda: {}
    runner._active_cron_job_count = lambda: 0
    runner._active_api_run_count = lambda: 0
    runner._update_runtime_status = lambda *args: None
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Agent:
        def __init__(self, **kwargs):
            self.provider, self.model = kwargs["provider"], kwargs["model"]
        def run_conversation(self, **kwargs):
            entered.set()
            assert release.wait(10)
            return {"final_response": "answer"}
        def interrupt(self, *args, **kwargs):
            pass
        def close(self):
            closed.set()

    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {"provider": "openai"})
    service = SideRunService(runner)
    sid = service.start(event, "fixture", "prompt", SideRunConfig.from_mapping({"provider": "openai", "model": "fixture"}))
    run = service.runs[sid]
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert runner._active_work_count() == 1
        _, timed_out = await runner._drain_active_agents(0)
        assert timed_out
        assert runner._interrupt_deferred_agent_workers("shutdown") == 1
        assert run.cancelled.is_set()
        service.shutdown()
        run.task.cancel()
        done, _ = await asyncio.wait([run.task], timeout=2)
        assert run.task in done, "shutdown supervisor blocked on an uninterruptible worker"
        assert run.task.cancelled()
        assert not closed.is_set(), "the worker should still be unwinding independently"
    finally:
        release.set()
        await asyncio.to_thread(closed.wait, 5)
        await asyncio.gather(run.task, return_exceptions=True)
        runner._shutdown_executor()
        db.close()
