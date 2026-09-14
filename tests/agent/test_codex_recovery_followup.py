"""R review counterexamples: real owner stores, retained consumers and pipes."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from tests.agent.test_codex_conditional_recovery import fleet, disk, fresh, token, P, worker, finish
from agent import credential_pool as cp
from agent import codex_pool_recovery as recovery
from hermes_cli import auth, auth_codex


def mutate(owner, fn):
    with auth._auth_store_lock(target_path=owner):
        store = auth._load_auth_store(owner)
        fn(store['credential_pool'][P])
        auth._save_auth_store(store, target_path=owner)


RETAINED = r'''
import json, sys
from agent import credential_pool as cp
from hermes_cli import auth
pool = cp.load_pool('openai-codex')
original = pool.entries()[0]
print('READY', flush=True)
for command in sys.stdin:
    command = command.strip()
    if command == 'FAIL':
        pool._mark_exhausted(original, 429, {'reason': 'usage_limit_reached', 'reset_at': 4102444800})
        print('FAILED', flush=True)
    elif command == 'SELECT':
        auth._probe_codex_quota_restored = lambda t, **kw: t == original.access_token
        selected = pool.select()
        print(json.dumps(None if selected is None else [selected.id, selected.last_status]), flush=True)
    elif command.startswith('PENDING:'):
        timestamp = float(command.split(':')[1])
        cp.time.time = lambda: timestamp
        pool._mark_exhausted(original, 429, {'reason': 'usage_limit_reached'}, persist=False)
        print('PENDING', flush=True)
    elif command == 'SAVE':
        pool._persist()
        print('SAVED', flush=True)
    elif command == 'END':
        break
'''


def start_retained(owner, local):
    profile = local if owner == local / 'auth.json' else local.parent / 'beta'
    profile.mkdir(exist_ok=True)
    p = subprocess.Popen([sys.executable, '-c', RETAINED], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=Path(__file__).resolve().parents[2], env=dict(os.environ, HERMES_HOME=str(profile)))
    assert p.stdout.readline().strip() == 'READY'
    return p


def command(p, cmd):
    p.stdin.write(cmd + '\n'); p.stdin.flush()
    return p.stdout.readline().strip()


@pytest.mark.parametrize('inflight_failure', [False, True])
def test_retained_process_reconciles_epoch(fleet, monkeypatch, inflight_failure):
    owner, local, _ = fleet
    p = start_retained(owner, local)
    try:
        monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda t, **kw: t == token('a'))
        assert cp.load_pool(P).select().id == 'a'
        epoch = disk(owner)[0][recovery.EPOCH]
        if inflight_failure:
            assert command(p, 'FAIL') == 'FAILED'
            assert disk(owner)[0]['last_status'] == 'exhausted'
            assert disk(owner)[0][recovery.EPOCH] == epoch
        assert json.loads(command(p, 'SELECT')) == ['a', 'ok']
        assert fresh(local)['a'] == 'ok'
    finally:
        p.communicate('END\n', timeout=20)
        assert p.returncode == 0


@pytest.mark.parametrize('pending_first', [False, True])
@pytest.mark.parametrize('delta', [-10, 0, 10])
def test_terminal_precedence_over_inflight_429(fleet, monkeypatch, pending_first, delta):
    owner, local, _ = fleet
    a, b = cp.load_pool(P), cp.load_pool(P)
    now = time.time()
    def pending():
        monkeypatch.setattr(cp.time, 'time', lambda: now + delta)
        b._mark_exhausted(b.entries()[0], 429, {'reason': 'usage_limit_reached'}, persist=False)
    if pending_first:
        pending()
    monkeypatch.setattr(cp.time, 'time', lambda: now)
    a._mark_exhausted(a.entries()[0], 401, {'reason': 'invalid_grant'})
    dead = disk(owner)[0]
    if not pending_first:
        pending()
    b._persist()
    assert disk(owner)[0] == dead
    assert b.select() is None
    assert b.entries()[0].last_status == 'dead'
    assert fresh(local)['a'] == 'dead'


@pytest.mark.parametrize('delta', [-10, 0, 10])
def test_pending_failures_preserve_newer_event(fleet, monkeypatch, delta):
    owner, local, _ = fleet
    a, b = cp.load_pool(P), cp.load_pool(P)
    now = time.time()
    monkeypatch.setattr(cp.time, 'time', lambda: now + delta)
    b._mark_exhausted(b.entries()[0], 429, {'reason': 'pending-quota'}, persist=False)
    pending_id = b.entries()[0].extra[recovery.FAILURE]
    monkeypatch.setattr(cp.time, 'time', lambda: now)
    a._mark_exhausted(a.entries()[0], 429, {'reason': 'committed-quota'})
    committed_id = disk(owner)[0][recovery.FAILURE]
    b._persist()
    assert disk(owner)[0][recovery.FAILURE] == (pending_id if delta > 0 else committed_id)


@pytest.mark.parametrize('stamp', ['iso', 'numeric'])
@pytest.mark.parametrize('defaults', [False, True])
def test_legacy_semantic_observation_and_raw_cas(fleet, monkeypatch, stamp, defaults):
    owner, local, _ = fleet
    def legacy(rows):
        if stamp == 'iso':
            rows[0]['last_status_at'] = datetime.fromtimestamp(rows[0]['last_status_at'], timezone.utc).isoformat()
        if defaults:
            rows[0].pop('auth_type'); rows[0].pop('source')
    mutate(owner, legacy)
    pool = cp.load_pool(P)
    before = disk(owner)[0]
    observed = recovery.observe(pool.entries()[0])
    assert observed is not None
    assert observed.row == before
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda t, **kw: t == token('a'))
    assert pool.select().id == 'a'
    assert disk(owner)[0]['access_token'] == before['access_token']
    assert disk(owner)[0]['refresh_token'] == before['refresh_token']
    assert fresh(local)['a'] == 'ok'
    # Normalization is recognition only, never permission to weaken raw CAS.
    assert recovery.recover(observed, proof='quota') is None


@pytest.mark.parametrize('duplicate', [False, True])
def test_resolver_binds_probe_to_current_row_endpoint(fleet, monkeypatch, duplicate):
    owner, local, _ = fleet
    mutate(owner, lambda rows: rows[0].update(base_url='https://old.invalid/backend-api/codex'))
    lookup = auth_codex._codex_pool_rate_limit_status
    def racing_lookup():
        snapshot = lookup()
        def change(rows):
            rows[0]['base_url'] = 'https://current.invalid/backend-api/codex'
            if duplicate:
                rows[1]['access_token'] = rows[0]['access_token']
                rows[1]['base_url'] = 'https://sister.invalid/backend-api/codex'
                rows.reverse()
        mutate(owner, change)
        return snapshot
    monkeypatch.setattr(auth_codex, '_codex_pool_rate_limit_status', racing_lookup)
    urls = []
    @contextmanager
    def client(**kw):
        def get(url, headers):
            urls.append(url)
            assert headers['Authorization'] == 'Bearer ' + token('a')
            return SimpleNamespace(status_code=200, json=lambda: {'rate_limit': {'primary_window': {'used_percent': 1}}})
        yield SimpleNamespace(get=get)
    monkeypatch.setattr(auth_codex, '_codex_http_client', client)
    monkeypatch.setattr(auth, '_codex_quota_probe_cache', {})
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', auth_codex._probe_codex_quota_restored)
    auth_codex.resolve_codex_runtime_credentials(refresh_if_expiring=False)
    assert urls == ['https://current.invalid/backend-api/wham/usage']
    rows = {r['id']: r for r in disk(owner)}
    assert rows['a']['last_status'] == 'ok'
    assert rows['b']['last_status'] == 'exhausted'


def test_failure_save_retains_intent_until_retry(fleet, monkeypatch):
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    original = pool.entries()[0]
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda t, **kw: t == token('a'))
    assert cp.load_pool(P).select().id == 'a'
    def broken(*args, **kw):
        raise OSError('invented save failure')
    with monkeypatch.context() as m:
        m.setattr(auth, '_save_auth_store', broken)
        m.setattr(cp, '_save_auth_store', broken)
        with pytest.raises(OSError):
            pool._mark_exhausted(original, 429, {'reason': 'usage_limit_reached', 'reset_at': 4102444800})
        assert pool._pending_codex_failures
        assert pool.select() is None
    pool._persist()
    assert not pool._pending_codex_failures
    assert disk(owner)[0]['last_status'] == 'exhausted'
    assert pool.select().id == 'a'
    assert fresh(local)['a'] == 'ok'


@pytest.mark.parametrize('pending_first', [False, True])
@pytest.mark.parametrize('delta', [-10, 0, 10])
def test_process_barriers_preserve_terminal_event(fleet, monkeypatch, pending_first, delta):
    owner, local, _ = fleet
    p = start_retained(owner, local)
    now = time.time()
    try:
        if pending_first:
            assert command(p, 'PENDING:' + str(now + delta)) == 'PENDING'
        monkeypatch.setattr(cp.time, 'time', lambda: now)
        pool = cp.load_pool(P)
        pool._mark_exhausted(pool.entries()[0], 401, {'reason': 'invalid_grant'})
        dead = disk(owner)[0]
        if not pending_first:
            assert command(p, 'PENDING:' + str(now + delta)) == 'PENDING'
        assert command(p, 'SAVE') == 'SAVED'
        assert disk(owner)[0] == dead
        assert json.loads(command(p, 'SELECT')) is None
        assert fresh(local)['a'] == 'dead'
    finally:
        p.communicate('END\n', timeout=20)
        assert p.returncode == 0


@pytest.mark.parametrize('action', ['remove', 'dead', 'owner', 'access_token', 'refresh_token'])
def test_reconciliation_requires_new_authority_observation(fleet, monkeypatch, action):
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    stale = pool.entries()[0]
    finish(worker(owner, local, action))
    before = owner.read_bytes()
    assert recovery.observe(stale) is None or action == 'owner'
    assert pool.select() is None
    assert pool.select() is None
    assert owner.read_bytes() == before
    if action in ('access_token', 'refresh_token'):
        assert getattr(pool.entries()[0], action) == 'invented-concurrent-material'
    if action == 'dead':
        assert pool.entries()[0].last_status == 'dead'
    if action == 'remove':
        assert fresh(local) == {'b': 'exhausted'}
        assert [e.id for e in pool.entries()] == ['b']
        pool._persist()
        assert fresh(local) == {'b': 'exhausted'}


def test_reconciliation_does_not_reset_selection_counter(fleet, monkeypatch):
    owner, local, _ = fleet
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda t, **kw: t == token('a'))
    pool = cp.load_pool(P)
    first = pool.select()
    second = pool.select()
    assert second.request_count == first.request_count + 1


def test_legacy_iso_recovers_in_new_process(fleet):
    owner, local, _ = fleet
    mutate(owner, lambda rows: rows[0].update(last_status_at='2026-01-01T00:00:00+00:00'))
    before = disk(owner)[0]
    p = start_retained(owner, local)
    try:
        assert json.loads(command(p, 'SELECT')) == ['a', 'ok']
        assert disk(owner)[0]['access_token'] == before['access_token']
        assert disk(owner)[0]['refresh_token'] == before['refresh_token']
        assert fresh(local)['a'] == 'ok'
    finally:
        p.communicate('END\n', timeout=20)
        assert p.returncode == 0
