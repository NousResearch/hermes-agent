"""Capacity is a physical-completion contract, not an asyncio cancellation count."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from gateway.session_parallel import conversation_scope
from hermes_state_parallel import active_work
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_parallel_native_ingress import admissions, event, setup_owner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_terminal_admission_still_occupies_capacity_until_thread_exits(monkeypatch):
    from agent.physical_work import bind_work, submit_work
    from gateway.session_parallel import physical_claim
    gate = asyncio.Event()
    started, release = threading.Event(), threading.Event()

    async def produce(*args):
        await gate.wait()
        return 'DONE'

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        original = event(adapter, 'm1', 'work')
        await adapter.handle_message(original)
        await async_wait_until(lambda: len(admissions(authority)) == 1, what='admission')

        def tool():
            started.set()
            release.wait(15)

        with bind_work(lambda: physical_claim(runner, original.source)):
            future = submit_work(pool, tool)
        await async_wait_until(started.is_set, what='physical tool entered')
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='parent settled')
        scope = conversation_scope(original.source)
        assert len(active_work(authority.db, scope)) == 1
        # Cancellation of the caller cannot release a thread which is already running.
        assert not future.cancel()
        binding = original.source.parallel_work
        from hermes_state_parallel import commit_work
        from dataclasses import replace
        replacement = replace(binding, work_id='f' * 64, route=binding.route + '-new')
        with pytest.raises(RuntimeStoreError, match='queue_full'):
            authority.db._execute_write(lambda conn: commit_work(
                conn, replacement, admissions(authority)[0]['target_session_id'], 'new', limit=1))
        release.set()
        await asyncio.to_thread(future.result, 10)
        assert active_work(authority.db, scope) == []
    finally:
        release.set()
        gate.set()
        await asyncio.to_thread(pool.shutdown, wait=True)
        await async_wait_until(lambda: not adapter._background_tasks, what='cleanup')
        authority.db.close()
