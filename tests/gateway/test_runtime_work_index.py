"""Active-work queries scale with live drains, including post-execution cleanup."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_runtime_workers import executing_sessions, uncounted_runtime_work
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
@pytest.mark.parametrize('cancelled', [False, True])
async def test_work_queries_skip_idle_history_but_keep_post_execution_work(tmp_path, monkeypatch, cancelled):
    from gateway import session_finite, session_ingress, session_settlement_recovery

    db, authority = _authority(tmp_path, monkeypatch)
    runner = authority.runner
    runner.session_authority, runner._running_agents = authority, {}
    started, returned, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    reads = []

    class IdleSession:
        task = None

        @property
        def event_stream(self):
            reads.append(True)
            return SimpleNamespace(execution={})

    async def execute(*_):
        started.set()
        if cancelled:
            await asyncio.Event().wait()
        return 'done'

    settle_with_retry = session_settlement_recovery.settle_with_retry

    async def settlement(*args):
        returned.set()
        await asyncio.wait_for(release.wait(), 5)
        return await settle_with_retry(*args)

    async def cleanup(*_):
        returned.set()
        await asyncio.wait_for(release.wait(), 5)

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(session_settlement_recovery, 'settle_with_retry', settlement)
    if cancelled:
        monkeypatch.setattr(session_ingress, 'fire_post_delivery', cleanup)
    task = None
    with db:
        try:
            await _submit(authority, 'work-query')
            await asyncio.wait_for(started.wait(), 5)
            live = authority.sessions['s']
            task = live.task
            if cancelled:
                task.cancel()
            await asyncio.wait_for(returned.wait(), 5)
            assert 's' not in authority.executing
            authority.sessions.update({f'idle-{i}': IdleSession() for i in range(1000)})
            active = list(executing_sessions(authority))
            assert [(sid, session) for sid, session, _ in active] == [('s', live)]
            assert uncounted_runtime_work(runner) >= 1
            assert not reads, 'work accounting inspected inactive historical sessions'
        finally:
            release.set()
            if task is not None:
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
        assert list(executing_sessions(authority)) == []
        assert uncounted_runtime_work(runner) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('eager', [False, pytest.param(True, marks=pytest.mark.skipif(
    not hasattr(asyncio, 'eager_task_factory'), reason='eager task factories require Python 3.12'))])
async def test_completed_drain_callback_cannot_lose_its_successor(tmp_path, monkeypatch, eager):
    db, authority = _authority(tmp_path, monkeypatch)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    release = asyncio.Event()
    successor_running = asyncio.Event()
    tasks = []
    starts = []

    async def drain(ref):
        starts.append(ref)
        if len(starts) == 1:
            return
        successor_running.set()
        await release.wait()

    monkeypatch.setattr(authority, '_drain', drain)
    live = authority.sessions['s']
    live.event_stream.execution = {'execution_generation': 7}
    def factory(loop, coroutine, **kwargs):
        if eager:
            task = asyncio.eager_task_factory(loop, coroutine, **kwargs)
        else:
            task = asyncio.Task(coroutine, loop=loop, **kwargs)
        # Start a successor before the authority's completion callback runs.
        task.add_done_callback(lambda _: authority._schedule(REF))
        return task

    loop.set_task_factory(factory)
    with db:
        try:
            authority._schedule(REF)
            first = live.task
            tasks.append(first)
            loop.set_task_factory(previous_factory)
            await asyncio.wait_for(successor_running.wait(), 5)
            successor = live.task
            assert successor is not first
            assert [(sid, session) for sid, session, _ in executing_sessions(authority)] == [('s', live)]
            # A restored/replaced live object must not inherit the old task's work.
            authority.sessions['s'] = SimpleNamespace(task=None)
            assert list(executing_sessions(authority)) == []
            authority.sessions['s'] = live
        finally:
            release.set()
            loop.set_task_factory(previous_factory)
            await asyncio.wait_for(asyncio.gather(*tasks, live.task, return_exceptions=True), 5)
        assert list(executing_sessions(authority)) == []
        assert not authority._drain_tasks
