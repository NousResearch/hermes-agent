"""Regressions for independent review: shared lock, generation proof and durability."""
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

import pytest

import offline_rebuild as worker
import service_runner as service
import storage_safety as safety
from bounded_verification import UIDBudget
import test_wrapper

pytestmark = pytest.mark.platforms('linux')


@pytest.fixture
def fx():
    fixture = test_wrapper.WrapperIntegrationTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def state(pid=123, started=55):
    stamp = datetime.now(UTC).isoformat()
    return {'kind': 'hermes-gateway', 'pid': pid, 'start_time': started, 'updated_at': stamp,
            'gateway_state': 'running', 'session_store': {'status': 'ok'}, 'served_profiles': [],
            'platforms': {'synthetic': {'state': 'connected', 'writer_pid': pid,
                                     'writer_start_time': started, 'updated_at': stamp}}}


def arguments(fx, monkeypatch):
    budget = UIDBudget(0, 2000, 2, 2000, 2)
    path = fx.home / 'budget.json'
    path.write_text(json.dumps(asdict(budget)), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['service_runner', '--execute-offline', '--service', 'synthetic.service',
                        '--uid-budget', str(path), '--search', 'alpha', '--search', 'beta',
                        '--expect-platform', 'synthetic'])
    return budget


def test_shared_lock_blocks_both_entry_points_before_db_or_service_actions(fx, monkeypatch):
    budget = arguments(fx, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('must not open a database or control a service')
    with safety.maintenance_lock(fx.db) as token:
        with safety.maintenance_lock(fx.db, token) as inherited:
            assert inherited is token
        monkeypatch.setattr(service, 'command', forbidden)
        monkeypatch.setattr(worker.sqlite3, 'connect', forbidden)
        with pytest.raises(BlockingIOError):
            worker.main(execute_offline=True, budget=budget)
        with pytest.raises(BlockingIOError):
            service.main()
    with pytest.raises(RuntimeError, match='Invalid inherited'):
        with safety.maintenance_lock(fx.db, token):
            pass


def test_supervisor_retains_same_lock_through_verified_restart_and_readiness(fx, monkeypatch):
    arguments(fx, monkeypatch)
    status = fx.home / 'gateway_state.json'
    status.write_text(json.dumps(state()), encoding='utf-8')
    calls = []
    current = {'pid': 123, 'started': 55}
    def assert_locked():
        with pytest.raises(BlockingIOError):
            with safety.maintenance_lock(fx.db):
                pass
    def command(args):
        if 'stop' in args:
            calls.append('stop'); assert_locked(); return ''
        if 'start' in args:
            calls.append('start'); assert_locked()
            current.update(pid=124, started=56)
            status.write_text(json.dumps(state(124, 56)), encoding='utf-8')
            return ''
        if 'MainPID' in args:
            return str(current['pid'])
        if 'is-active' in args:
            assert_locked()
        return 'active'
    def maintain(*args, lock, **kwargs):
        lock.validate(fx.db); assert_locked(); calls.append('maintain'); return 0
    monkeypatch.setattr(service, 'command', command)
    monkeypatch.setattr(service, 'process_start', lambda pid: current['started'])
    monkeypatch.setattr(worker, 'holders', lambda path: [current['pid']])
    monkeypatch.setattr(worker, 'main', maintain)
    monkeypatch.setattr(service, 'verify_receipt', lambda *args: True)
    assert service.main() == 0
    assert calls == ['stop', 'maintain', 'start']
    with safety.maintenance_lock(fx.db):
        pass


@pytest.mark.parametrize('mode', ['multiplex', 'named-profile-no-local-receipt', 'stale-standalone'])
def test_unsupported_or_stale_topology_refuses_before_stop(fx, monkeypatch, mode):
    arguments(fx, monkeypatch)
    record = state()
    if mode == 'multiplex': record['served_profiles'] = ['default', 'synthetic']
    if mode == 'stale-standalone': record['start_time'] = 1
    if mode != 'named-profile-no-local-receipt':
        (fx.home / 'gateway_state.json').write_text(json.dumps(record), encoding='utf-8')
    controls = []
    def command(args):
        if 'stop' in args or 'start' in args: controls.append(args)
        return '123' if 'MainPID' in args else 'active'
    monkeypatch.setattr(service, 'command', command)
    monkeypatch.setattr(service, 'process_start', lambda pid: 55)
    monkeypatch.setattr(worker, 'holders', lambda path: [123])
    assert service.main() == 1
    assert controls == []


@pytest.mark.parametrize('mutation', ['missing', 'empty', 'other-pid', 'old-generation', 'old-time', 'pid-reuse'])
def test_readiness_rejects_missing_or_stale_platform_proof(fx, monkeypatch, mutation):
    record = state()
    cutoff = datetime.now(UTC).timestamp() - 1
    mutations = {
        'missing': lambda: record.pop('platforms'),
        'empty': lambda: record.update(platforms={}),
        'other-pid': lambda: record['platforms']['synthetic'].update(writer_pid=999),
        'old-generation': lambda: record['platforms']['synthetic'].update(writer_start_time=1),
        'old-time': lambda: record['platforms']['synthetic'].update(updated_at='2000-01-01T00:00:00+00:00'),
        'pid-reuse': lambda: record.update(start_time=1),
    }
    mutations[mutation]()
    (fx.home / 'gateway_state.json').write_text(json.dumps(record), encoding='utf-8')
    monkeypatch.setattr(service, 'command', lambda args: '123' if 'MainPID' in args else 'active')
    monkeypatch.setattr(service, 'process_start', lambda pid: 55)
    assert not service.ready_once(fx.home, 'synthetic.service', cutoff, {'synthetic'})


def test_platformless_requires_explicit_expected_empty_roster(fx, monkeypatch):
    record = state(); record['platforms'] = {}
    (fx.home / 'gateway_state.json').write_text(json.dumps(record), encoding='utf-8')
    monkeypatch.setattr(service, 'command', lambda args: '123' if 'MainPID' in args else 'active')
    monkeypatch.setattr(service, 'process_start', lambda pid: 55)
    assert service.ready_once(fx.home, 'synthetic.service', 0, set())
    assert not service.ready_once(fx.home, 'synthetic.service', 0, {'synthetic'})


def test_directory_entries_fsync_their_parents_in_creation_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(safety, 'sync_directory', calls.append)
    safety.ensure_directory(tmp_path / 'parent' / 'child')
    assert calls == [tmp_path, tmp_path / 'parent']


def test_directory_durability_failure_prevents_database_access(fx, monkeypatch):
    def fail(path): raise OSError('synthetic fsync failure')
    monkeypatch.setattr(safety, 'sync_directory', fail)
    with pytest.raises(OSError, match='fsync failure'):
        fx.run_worker()
    assert fx.effects == []


def test_backup_fsync_failure_blocks_optimizer_and_completion(fx, monkeypatch):
    original = worker.sync_file
    def fail(path):
        if Path(path).name == 'state-before.db.zst': raise OSError('synthetic backup fsync failure')
        return original(path)
    monkeypatch.setattr(worker, 'sync_file', fail)
    with pytest.raises(OSError, match='backup fsync'):
        fx.run_worker()
    assert 'synthetic-uid-backfill' not in fx.effects
    assert fx.receipt()['status'] == 'blocked'


def test_backup_and_baseline_are_synced_before_mutation(fx, monkeypatch):
    calls = []; original_sync = worker.sync_file; original_mutate = fx.fake_optimize
    def sync(path): calls.append(Path(path).name); return original_sync(path)
    def mutate(*args, **kwargs): calls.append('MUTATION'); return original_mutate(*args, **kwargs)
    monkeypatch.setattr(worker, 'sync_file', sync)
    monkeypatch.setattr(fx, 'fake_optimize', mutate)
    assert fx.run_worker() == 0
    for name in ['state-before.db.zst', 'state-before.db', 'before.json']:
        assert calls.index(name) < calls.index('MUTATION')
    for name in ['state-after.db', 'after.json', 'optimize.log']:
        assert calls.index(name) > calls.index('MUTATION')


def test_integrity_timeout_blocks_completion(fx, monkeypatch):
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired(['synthetic-check'], 1)
    monkeypatch.setattr(worker, 'check_integrity', timeout)
    with pytest.raises(subprocess.TimeoutExpired): fx.run_worker()
    assert fx.receipt()['status'] == 'blocked'
