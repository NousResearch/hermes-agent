"""Adversarial frozen-diff probes; no external services."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_parallel_native_ingress import setup_owner, event, admissions
from tests.waiting import async_wait_until
from gateway.session_parallel import execute_turn, conversation_scope
from hermes_state_parallel import active_work


@pytest.mark.asyncio
@pytest.mark.parametrize('descendant_kind', ['tool', 'delegation'])
async def test_retirement_must_wait_for_real_abandoned_tool(monkeypatch, descendant_kind):
    from agent import tool_executor as te
    from gateway.run_runtime import _retire_profile_authority
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    agent = SimpleNamespace(_tool_worker_threads=set(), _tool_worker_threads_lock=threading.Lock(),
                            _interrupt_requested=False, _touch_activity=lambda *a: None)

    def tool_middleware(*args, **kwargs):
        started.set()
        try:
            assert release.wait(20)
            return te._ManagedToolResult(result='late', args={}, middleware_trace=[], blocked=False, dispatched=True)
        finally:
            finished.set()

    monkeypatch.setattr(te, '_run_agent_tool_execution_middleware', tool_middleware)
    monkeypatch.setattr(te, '_resolve_sequential_tool_timeout', lambda: 0.1)
    monkeypatch.setattr(te, '_emit_terminal_post_tool_call', lambda *a, **k: None)
    runner = None
    original = None

    async def produce(authority, ref, row):
        def turn():
            if descendant_kind == 'delegation':
                from tools.delegate_tool_dispatch import _run_children_parallel
                from tests.tools.test_delegate_parallel_interrupt_join import _batch
                parent = SimpleNamespace(_interrupt_requested=False, _delegate_spinner=None, quiet_mode=True)
                batch = _batch([SimpleNamespace(_delegate_role='leaf')], parent)
                def child(*args):
                    started.set()
                    parent._interrupt_requested = True
                    try:
                        assert release.wait(20)
                        return {'task_index': 0, 'status': 'completed'}
                    finally:
                        finished.set()
                batch.run_child = child
                results = []
                _run_children_parallel(batch, results, honor_parent_interrupt=True)
                return SimpleNamespace(result=results)
            return te._run_sequential_tool_execution_middleware(
                agent, function_name='web_search', function_args={}, effective_task_id=ref.session_id,
                tool_call_id='review-tool', execute=lambda args: 'unused')
        result = await execute_turn(runner, original.source, turn)
        return str(result.result)

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    original = event(adapter, 'drain-m1', 'work')
    try:
        await adapter.handle_message(original)
        await async_wait_until(started.is_set, what='real executor tool entered')
        await async_wait_until(lambda: not adapter._background_tasks, what='parent admission settled')
        scope = conversation_scope(original.source)
        active = len(active_work(authority.db, scope))
        with authority.db._read_ctx() as conn:
            leases = conn.execute('SELECT COUNT(*) FROM parallel_work_physical').fetchone()[0]
        counts = runner._drain_work_counts()
        retired = await _retire_profile_authority(authority, timeout=0)
        print(f'active_work={active}; physical_leases={leases}; drain_counts={counts}; tool_still_running={not finished.is_set()}; retire_result={retired}; authority_retired={authority.retired}', flush=True)
        assert not retired, 'retirement released authority while real descendant tool still executing'
    finally:
        release.set()
        await async_wait_until(finished.is_set, what='tool returned')
        await async_wait_until(lambda: not active_work(authority.db, conversation_scope(original.source)), what='physical lease released')
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['cancel_before_start', 'submission_rejected'])
async def test_cancel_before_executor_submission_releases_reservation(monkeypatch, failure):
    from gateway.session_runtime_workers import start_turn_worker
    entered = threading.Event()
    async def produce(*args):
        return 'DONE'
    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    original = event(adapter, 'cancel-m1', 'work')
    try:
        await adapter.handle_message(original)
        await async_wait_until(lambda: not adapter._background_tasks, what='original settled')
        worker = SimpleNamespace(worker_done=threading.Event(), task_id=admissions(authority)[0]['target_session_id'])
        if failure == 'submission_rejected':
            runner._executor_closing = True
        start_turn_worker(runner, worker, [None], entered.set, source=original.source)
        if failure == 'cancel_before_start':
            worker.executor_task.cancel()
        results = await asyncio.gather(worker.executor_task, return_exceptions=True)
        with authority.db._read_ctx() as conn:
            leases = conn.execute('SELECT COUNT(*) FROM parallel_work_physical').fetchone()[0]
        print(f'target_entered={entered.is_set()}; executor_task_cancelled={worker.executor_task.cancelled()}; physical_leases={leases}; active_work={len(active_work(authority.db, conversation_scope(original.source)))}; executor_result={results!r}', flush=True)
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner
        from gateway.session_authority import initialize_session_authority
        authority.db.close()
        cold_runner = GatewayRunner(GatewayConfig())
        cold = await initialize_session_authority(cold_runner, profile_id='default', instance_id='cold-review')
        try:
            cold_active = len(active_work(cold.db, conversation_scope(original.source)))
            print(f'cold_authority_active_work={cold_active}; target_entered={entered.is_set()}', flush=True)
        finally:
            cold.db.close()
        assert leases == 0, 'cancelled-before-submission turn leaked durable physical lease'
    finally:
        authority.db.close()
