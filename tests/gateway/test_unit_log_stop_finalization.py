"""Native Stop's human FIFO sees the final original unit log before resuming."""
import asyncio
import json
import shlex
import sys
import threading
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _submit
from tests.waiting import async_wait_until

pytestmark = pytest.mark.platforms('linux')
pytest_plugins = ['tests.tools.test_unit_log_final_tail']


@pytest.fixture(autouse=True)
def native_parent():
    from gateway.session_context import set_session_vars, clear_session_vars
    tokens = set_session_vars(session_id='s', session_key='route')
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize('reload_owner', [False, True], ids=['live', 'cold'])
@pytest.mark.parametrize('hold_publication', [False, True], ids=['ordinary-reader', 'held-reader'])
async def test_unit_stop_collects_final_payload_before_native_human_fifo(
        unit_log_job, monkeypatch, reload_owner, hold_publication):
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_context import set_session_vars, clear_session_vars
    from hermes_state import SessionDB
    from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, recover_session_inputs, list_session_admissions
    from tools.process_registry_job import job_quiescent
    module, registry, own, home, reap_child = unit_log_job
    assert own.parent_session_id == 's' and own.session_key == 'route'
    monkeypatch.setattr(module, 'process_registry', registry)
    launch = registry._scope_argv
    monkeypatch.setattr(registry, '_scope_argv', module.ProcessRegistry._scope_argv.__get__(registry))
    tokens = set_session_vars(session_id='foreign-parent', session_key='route')
    try:
        neighbor = registry.spawn_local(shlex.join([sys.executable, '-c',
            'import threading; threading.Event().wait(30)']), cwd=str(home), task_id='foreign', session_key='route')
    finally:
        clear_session_vars(tokens)
        monkeypatch.setattr(registry, '_scope_argv', launch)
    db = SessionDB(db_path=home / 'state.db')
    db.create_session('s', source='test')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='current'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    stop_empty, ready, executed = module._stop_systemd_unit, False, []
    unit = own.systemd_unit
    def verified_unit_stop(value):
        assert value == unit
        return stop_empty(value) if ready else False
    monkeypatch.setattr(module, '_stop_systemd_unit', verified_unit_stop)
    async def execute(owner, ref, row):
        current = module.process_registry.get(own.id)
        retained = json.loads((home / 'logs' / 'process-results' / (own.id + '.json')).read_text())
        assert job_quiescent(current) and current._result_complete and retained['result_complete']
        assert current.output_log_state['final'] and retained['output_log_state']['final']
        for value in (current.output_buffer, retained['output']):
            assert value.count('PRIMARY ORIGINAL OUTPUT') == value.count('LATE ORIGINAL OUTPUT') == 1
        assert current.exit_code == retained['exit_code'] == 7 and neighbor.process.poll() is None
        executed.append(row['request_id'])
        return 'continued once'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    release_reader = threading.Event()
    current = registry
    try:
        with pytest.raises(RuntimeStoreError, match='process_stop_unknown'):
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        await _submit(authority, 'only-human')
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert not executed and process_stop_pending(db, 's')
        if reload_owner:
            epoch = begin_runtime_epoch(db, instance_id='cold')
            recover_session_inputs(db, epoch=epoch)
            old = authority
            authority = SessionAuthority(runner, profile_id='owned', instance_id='cold', db=db, epoch=epoch)
            authority.sessions['s'] = LiveSession(old.sessions['s'].source, 'route')
            current = module.ProcessRegistry()
            monkeypatch.setattr(module, 'process_registry', current)
            current.recover_from_checkpoint()
            authority._schedule(REF)
            await asyncio.wait_for(authority.sessions['s'].task, 10)
            assert not executed and process_stop_pending(db, 's')
        await asyncio.to_thread(reap_child)
        ready = True
        entered, hold_next = threading.Event(), threading.Event()
        refusals = []
        move, refuse = module.ProcessRegistry._move_to_finished, module.ProcessRegistry._refuse_backend_stop
        def publish(owner, process):
            if process.id == own.id and hold_next.is_set():
                hold_next.clear()
                entered.set()
                assert release_reader.wait(5)
            return move(owner, process)
        def refused(owner, process, consume_output):
            if process.id == own.id:
                refusals.append((process, process._completion_event.is_set(),
                                 process._result_persisted_generation))
            return refuse(owner, process, consume_output)
        monkeypatch.setattr(module.ProcessRegistry, '_refuse_backend_stop', refused)
        if hold_publication:
            hold_next.set()
            monkeypatch.setattr(module.ProcessRegistry, '_move_to_finished', publish)
        try:
            await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
            assert not hold_publication, 'A held original publisher cannot acknowledge cleanup'
        except RuntimeStoreError as exc:
            assert exc.reason == 'process_stop_unknown'
            # Recovery may start a fresh original reader after the child was reaped.
            # Only that genuine unfinished publication permits the bounded refusal.
            assert refusals and not refusals[-1][1]
            process, _completed, persisted = refusals[-1]
            assert process._reader_thread is not None and persisted == -1
            assert not executed and process_stop_pending(db, 's') and neighbor.process.poll() is None
            if hold_publication:
                assert entered.is_set()
            release_reader.set()
            assert await asyncio.to_thread(process._completion_event.wait, 5)
        finally:
            release_reader.set()
        await async_wait_until(lambda: executed == ['only-human'] and authority.sessions['s'].task.done(), timeout=10)
        assert process_stop_pending(db, 's') is None and neighbor.process.poll() is None
        rows = [r for r in list_session_admissions(db, session_id='s', pending_only=False)
                if r['principal_id'] != 'native-process-stop']
        assert [(r['request_id'], r['status'], r['outcome']) for r in rows] == [('only-human', 'terminal', 'completed')]
    finally:
        release_reader.set()
        await asyncio.to_thread(registry.kill_process, neighbor.id)
        assert await asyncio.to_thread(neighbor._completion_event.wait, 5)
        if current is not registry:
            copied = current.get(neighbor.id)
            if copied is not None:
                assert await asyncio.to_thread(copied._completion_event.wait, 5)
        db.close()
