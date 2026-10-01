"""Authority-owned preparation completes beyond one batch and retains damaged evidence."""
import asyncio
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.session_authority import initialize_session_authority
from gateway.session_cron import unbind_owner
from gateway.session_logical_preparation import stop_logical_preparation
from hermes_state import SessionDB
import hermes_state_logical_attempts as index
from hermes_state_runtime import begin_runtime_epoch, RuntimeStoreError
from tests.hermes_state.test_logical_attempt_index import accept, lookup


def _legacy_inventory(db, monkeypatch, size):
    epoch = begin_runtime_epoch(db, instance_id='old-owner')
    with monkeypatch.context() as old:
        old.setattr(index, 'project_admission', lambda *args, **kwargs: None)
        return [accept(db, epoch, sid=f'member-{n}', request=f'run-{n}', task=f'task-{n}')
                for n in range(size)]


async def _authority(db):
    runner = SimpleNamespace(_draining=False, adapters={},
                             session_store=SimpleNamespace(), config=SimpleNamespace(multiplex_profiles=False))
    return await initialize_session_authority(runner, profile_id=str(Path(db.db_path).parent),
                                              instance_id='new-owner', db=db)


def _status(db):
    with db._read_ctx() as conn:
        return index.logical_preparation_status(conn)


async def _stop(authority):
    await stop_logical_preparation(authority)
    unbind_owner(authority)


@pytest.mark.asyncio
async def test_authority_drains_more_than_one_batch_without_blocking_other_loop_work(tmp_path, monkeypatch):
    with SessionDB(tmp_path / 'state.db') as db:
        rows = _legacy_inventory(db, monkeypatch, 270)
        authority = await _authority(db)
        ticks = 0
        async def other_work():
            nonlocal ticks
            while not authority._logical_preparation_task.done():
                ticks += 1
                await asyncio.sleep(0)
        pulse = asyncio.create_task(other_work())
        try:
            with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
                lookup(db, sid='new-session')
            await asyncio.wait_for(authority._logical_preparation_task, 20)
            await pulse
            assert ticks > 1
            assert _status(db) == dict(state='ready', phase='covered', prepared=270, pending=0, held=0)
            assert lookup(db, sid='member-269', task='task-269')['admission_id'] == rows[-1]['admission_id']
        finally:
            await _stop(authority)


@pytest.mark.asyncio
async def test_owner_crash_mid_drain_resumes_the_committed_cursor(tmp_path, monkeypatch):
    path, ready = tmp_path / 'state.db', tmp_path / 'first-batch'
    with SessionDB(path) as db:
        rows = _legacy_inventory(db, monkeypatch, 270)
    root = Path(__file__).resolve().parents[2]
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ', 'SYSTEMROOT') if key in os.environ}
    env.update(HERMES_HOME=str(tmp_path), HOME=str(tmp_path / 'home'), PYTHONPATH=str(root))
    child = subprocess.Popen([sys.executable, str(Path(__file__).parent / 'fixtures/logical_preparation_crash.py'),
                              str(path), str(ready)], cwd=root, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            if child.poll() is not None or time.monotonic() > deadline:
                pytest.fail('preparation child did not commit its first batch')
            await asyncio.sleep(.05)
        child.kill()
        await asyncio.to_thread(child.wait, 10)
    finally:
        if child.poll() is None:
            child.kill()
            await asyncio.to_thread(child.wait, 10)
        if child.stderr:
            child.stderr.close()
    with SessionDB(path) as db:
        before = _status(db)
        assert before['prepared'] == 128 and before['pending'] == 142
        authority = await _authority(db)
        try:
            before_resume = _status(db)
            with pytest.raises(RuntimeStoreError, match='stale_epoch'):
                index.prepare_logical_attempt_index(db, epoch=authority.epoch - 1)
            assert _status(db) == before_resume
            await asyncio.wait_for(authority._logical_preparation_task, 20)
            assert _status(db)['state'] == 'ready'
            assert lookup(db, sid='member-269', task='task-269')['admission_id'] == rows[-1]['admission_id']
        finally:
            await _stop(authority)


@pytest.mark.asyncio
@pytest.mark.parametrize('shutdown_during_backoff', [False, True], ids=['retry', 'shutdown'])
async def test_transient_writer_contention_retries_or_stops_without_another_batch(
        tmp_path, monkeypatch, shutdown_during_backoff):
    import gateway.session_logical_preparation as preparation
    original = preparation.prepare_logical_attempt_index
    failed = threading.Event()
    calls = 0
    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            failed.set()
            raise sqlite3.OperationalError('database is locked')
        return original(*args, **kwargs)
    monkeypatch.setattr(preparation, 'prepare_logical_attempt_index', fail_once)
    with SessionDB(tmp_path / 'state.db') as db:
        _legacy_inventory(db, monkeypatch, 2)
        authority = await _authority(db)
        try:
            assert await asyncio.to_thread(failed.wait, 5)
            async with asyncio.timeout(5):
                while authority._logical_preparation_state != 'retrying':
                    await asyncio.sleep(0)
            if shutdown_during_backoff:
                await _stop(authority)
                assert calls == 1 and authority._logical_preparation_state == 'stopped'
                assert _status(db)['state'] == 'waiting'
            else:
                await asyncio.wait_for(authority._logical_preparation_task, 10)
                assert calls == 2 and _status(db)['state'] == 'ready'
        finally:
            await _stop(authority)


@pytest.mark.asyncio
async def test_unclassifiable_terminal_hold_backs_off_without_blocking_later_good_inventory(tmp_path, monkeypatch):
    import gateway.session_logical_preparation as preparation
    from hermes_state_terminal import ADMISSION_PREFIX
    original = preparation.prepare_logical_attempt_index
    covered_calls = 0
    def counted(*args, **kwargs):
        nonlocal covered_calls
        step = original(*args, **kwargs)
        if step['phase'] == 'covered':
            covered_calls += 1
        return step
    monkeypatch.setattr(preparation, 'prepare_logical_attempt_index', counted)
    with SessionDB(tmp_path / 'state.db') as db:
        _legacy_inventory(db, monkeypatch, 270)
        key = ADMISSION_PREFIX + 'unclassifiable'
        db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                                                    (key, '{corrupt')))
        authority = await _authority(db)
        try:
            async with asyncio.timeout(20):
                while _status(db)['phase'] != 'covered':
                    await asyncio.sleep(.05)
            first = covered_calls
            await asyncio.sleep(2.1)
            assert covered_calls - first <= 3  # held inventory does not transact every 50 ms
            assert _status(db)['held'] == 1
            with db._read_ctx() as conn:
                assert conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0] == '{corrupt'
                assert conn.execute('SELECT 1 FROM logical_attempts WHERE task_id=?', ('task-269',)).fetchone()
            with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
                lookup(db, sid='member-269', task='task-269')  # unknown scope still fences absence globally
        finally:
            await _stop(authority)


@pytest.mark.asyncio
async def test_corrupt_record_stays_held_while_later_batches_finish_and_doctor_reports_it(tmp_path, monkeypatch, capsys):
    from hermes_cli.doctor_state import _logical_attempt_preparation
    with SessionDB(tmp_path / 'state.db') as db:
        rows = _legacy_inventory(db, monkeypatch, 270)
        aid = rows[0]['admission_id']
        db._execute_write(lambda conn: conn.execute('UPDATE session_admissions SET payload_json=? '
                                                    'WHERE admission_id=?', ('{corrupt', aid)))
        authority = await _authority(db)
        try:
            async with asyncio.timeout(20):
                while _status(db)['phase'] != 'covered':
                    await asyncio.sleep(.05)
            status = _status(db)
            assert (status['prepared'], status['pending'], status['held']) == (269, 0, 1)
            assert lookup(db, sid='member-269', task='task-269')['admission_id'] == rows[-1]['admission_id']
            with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
                lookup(db, sid='member-0', task='task-0')
            with db._read_ctx() as conn:
                assert conn.execute('SELECT payload_json FROM session_admissions WHERE admission_id=?',
                                    (aid,)).fetchone()[0] == '{corrupt'
            _logical_attempt_preparation(Path(db.db_path))
            assert '269 prepared, 0 pending, 1 held' in capsys.readouterr().out
        finally:
            await _stop(authority)
