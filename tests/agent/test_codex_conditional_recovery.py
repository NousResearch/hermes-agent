"""Real stores/locks/processes; only quota transport is synthetic."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import agent.credential_pool as cp
import hermes_cli.auth as auth

P = 'openai-codex'

def token(tag):
    enc = lambda x: base64.urlsafe_b64encode(json.dumps(x).encode()).decode().rstrip('=')
    return enc({'alg': 'none'}) + '.' + enc({'exp': 4102444800, 'synthetic': tag}) + '.invented'

@pytest.fixture(params=['local', 'global'])
def fleet(tmp_path, monkeypatch, request):
    root = tmp_path / 'hermes-root'
    local = root / 'profiles' / 'alpha'
    home = tmp_path / 'fakehome'
    local.mkdir(parents=True)
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('HERMES_HOME', str(local))
    assert os.environ.get('PYTEST_CURRENT_TEST')
    assert auth._global_auth_file_path() == root / 'auth.json'
    assert cp._guarded_global_root(root / 'auth.json') == root / 'auth.json'
    owner = (local if request.param == 'local' else root) / 'auth.json'
    rows = []
    for name in ['a', 'b']:
        rows.append(dict(id=name, provider=P, source='device_code', auth_type='oauth',
            priority=len(rows), access_token=token(name), refresh_token='invented-'+name,
            last_status='exhausted', last_status_at=time.time()-120,
            last_error_code=429, last_error_reason='usage_limit_reached',
            last_error_reset_at=time.time()+86400, failure_reason='rate_limit'))
    owner.write_text(json.dumps(dict(version=1, providers={}, credential_pool={P: rows})))
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda *a, **k: False)
    return owner, local, rows

def disk(owner):
    return json.loads(owner.read_text())['credential_pool'][P]

def fresh(local):
    code = "import json; from agent.credential_pool import load_pool; print(json.dumps([(e.id,e.last_status) for e in load_pool('openai-codex').entries()]))"
    env = dict(os.environ, HERMES_HOME=str(local))
    p = subprocess.run([sys.executable, '-c', code], env=env, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=True)
    return dict(json.loads(p.stdout))

@pytest.mark.parametrize('proof', ['positive', 'negative', 'unknown', 'expired'])
def test_recovery_durable(fleet, monkeypatch, proof):
    owner, local, rows = fleet
    if proof == 'expired':
        store = json.loads(owner.read_text())
        store['credential_pool'][P][0]['last_error_reset_at'] = time.time()-60
        owner.write_text(json.dumps(store))
    calls = []
    def probe(t, **kw):
        calls.append(t)
        return {'positive': True, 'negative': False, 'unknown': None, 'expired': False}[proof] if t == token('a') else False
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', probe)
    pool = cp.load_pool(P)
    assert len(pool.entries()) == 2
    before_clear = json.loads(owner.read_text())
    selected = pool.select()
    assert json.loads(owner.read_text())['providers'] == before_clear['providers']
    assert disk(owner)[1] == before_clear['credential_pool'][P][1]
    for key in ('access_token', 'refresh_token', 'source', 'priority', 'id'):
        assert disk(owner)[0][key] == before_clear['credential_pool'][P][0][key]
    restored = proof in ('positive', 'expired')
    assert (selected is not None) == restored
    assert disk(owner)[0]['last_status'] == ('ok' if restored else 'exhausted')
    assert fresh(local)['a'] == ('ok' if restored else 'exhausted')
    assert disk(owner)[1]['last_status'] == 'exhausted'
    if restored:
        assert not disk(owner)[0].get('failure_reason')
        assert selected is not None and selected.last_status == 'ok'
    if proof == 'expired':
        assert token('a') not in calls
    if owner != local / 'auth.json':
        assert not (local / 'auth.json').exists() or not json.loads((local / 'auth.json').read_text()).get('credential_pool', {}).get(P)

def test_protected_root_remains_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes' / 'profiles' / 'alpha'))
    target = tmp_path / '.hermes' / 'auth.json'
    target.parent.mkdir()
    target.write_text(json.dumps({'credential_pool': {P: [{'id': 'synthetic'}]}}))
    assert auth._load_global_auth_store() == {}
    assert cp._guarded_global_root(target) is None


WORKER = r'''
import json, os, sys
from pathlib import Path
from agent import credential_pool as cp
from hermes_cli import auth
assert os.environ.get('PYTEST_CURRENT_TEST')
P = 'openai-codex'
owner = Path(sys.argv[1])
action = sys.argv[2]
pool = cp.load_pool(P)
print('READY', flush=True)
assert sys.stdin.readline().strip() == 'GO'
if action == 'stale':
    pool._persist()
elif action in ('429', 'dead'):
    entry = pool.entries()[0]
    cp.time.time = lambda: entry.last_status_at or 1
    pool._mark_exhausted(entry, 401 if action == 'dead' else 429,
        {'reason': 'invalid_grant' if action == 'dead' else 'usage_limit_reached',
         'reset_at': entry.last_error_reset_at}, failure_reason='auth' if action == 'dead' else 'rate_limit')
else:
    with auth._auth_store_lock(target_path=owner):
        store = auth._load_auth_store(owner)
        row = store['credential_pool'][P][0]
        if action in ('access_token', 'refresh_token'):
            row[action] = 'invented-concurrent-material'
        elif action in ('remove', 'reinsert'):
            store['credential_pool'][P].remove(row)
            auth._save_auth_store(store, target_path=owner)
            if action == 'reinsert':
                store['credential_pool'][P].insert(0, row)
        elif action == 'label':
            row['label'] = 'synthetic-concurrent-label'
        elif action == 'owner':
            auth._save_auth_store(store)
        auth._save_auth_store(store, target_path=owner)
print('DONE', flush=True)
'''


def worker(owner, local, action):
    profile = local
    if owner != local / 'auth.json' and action != 'owner':
        profile = local.parent / 'beta'
        profile.mkdir(exist_ok=True)
    p = subprocess.Popen([sys.executable, '-c', WORKER, str(owner), action],
        env=dict(os.environ, HERMES_HOME=str(profile)),
        cwd=Path(__file__).resolve().parents[2], text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Pipe barriers, not wall-clock sleeps; pytest's file timeout is a backstop.
    assert p.stdout is not None
    assert p.stdout.readline().strip() == 'READY'
    return p


def finish(p):
    out, err = p.communicate('GO\n', timeout=20)
    assert p.returncode == 0, err
    assert out.strip() == 'DONE', (out, err)


@pytest.mark.parametrize('action', ['429', 'dead', 'access_token', 'refresh_token', 'remove', 'reinsert', 'label', 'owner'])
def test_concurrent_change_invalidates_proof(fleet, monkeypatch, action):
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    p = worker(owner, local, action)
    def probe(t, **kw):
        if t != token('a'):
            return False
        finish(p)
        return True
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', probe)
    assert pool.select() is None
    current = disk(owner)
    if action == 'remove':
        assert [r['id'] for r in current] == ['b']
    else:
        assert current[0]['last_status'] == ('dead' if action == 'dead' else 'exhausted')
    if action in ('access_token', 'refresh_token'):
        assert current[0][action] == 'invented-concurrent-material'
    if action == 'label':
        assert current[0]['label'] == 'synthetic-concurrent-label'
    if action not in ('remove', 'access_token', 'refresh_token'):
        assert fresh(local)['a'] == ('dead' if action == 'dead' else 'exhausted')


def test_stale_writer_after_clear_and_new_failure(fleet, monkeypatch):
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    p = worker(owner, local, 'stale')
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda t, **kw: t == token('a'))
    assert pool.select() is not None
    after_clear = owner.read_bytes()
    finish(p)
    assert disk(owner)[0]['last_status'] == 'ok'
    assert fresh(local)['a'] == 'ok'
    assert disk(owner)[1]['last_status'] == 'exhausted'
    # A genuinely later failure from a reader of the new epoch still wins.
    p = worker(owner, local, '429')
    finish(p)
    assert disk(owner)[0]['last_status'] == 'exhausted'
    assert fresh(local)['a'] == 'exhausted'


@pytest.mark.parametrize('failure', ['save', 'probe'])
def test_failed_recovery_is_not_available(fleet, monkeypatch, failure):
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    before = owner.read_bytes()
    def broken(*a, **k):
        raise OSError('synthetic failure')
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', broken if failure == 'probe' else lambda *a, **k: True)
    if failure == 'save':
        monkeypatch.setattr(auth, '_save_auth_store', broken)
    assert pool.select() is None
    assert owner.read_bytes() == before
    assert all(e.last_status == 'exhausted' for e in pool.entries())
    assert fresh(local)['a'] == 'exhausted'


@pytest.mark.parametrize('field', ['access_token', 'refresh_token'])
@pytest.mark.parametrize('status', ['exhausted', 'dead'])
def test_material_sync_is_not_quota_proof(fleet, monkeypatch, field, status):
    owner, local, _ = fleet
    store = json.loads(owner.read_text())
    store['credential_pool'][P][0]['last_status'] = status
    owner.write_text(json.dumps(store))
    pool = cp.load_pool(P)
    store['providers'][P] = {'tokens': {'access_token': token('a'), 'refresh_token': 'invented-a'}}
    store['providers'][P]['tokens'][field] = token('new') if field == 'access_token' else 'invented-new-refresh'
    owner.write_text(json.dumps(store))
    assert pool.select() is None
    assert pool.entries()[0].last_status == status
    assert disk(owner)[0]['last_status'] == status


def test_explicit_reset_still_supported(fleet):
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    entry = pool.entries()[0]
    pool._adopt(entry, persist=False, **cp._MARK_OK)
    pool._persist(status_cleared_ids=['a'])
    assert disk(owner)[0]['last_status'] == 'ok'


def test_aggregate_probe_does_not_clear_model_scope(fleet, monkeypatch):
    owner, local, _ = fleet
    store = json.loads(owner.read_text())
    store['credential_pool'][P][0]['quota_scope'] = 'model:synthetic-a'
    owner.write_text(json.dumps(store))
    pool = cp.load_pool(P)
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda t, **kw: t == token('a'))
    assert pool.select() is None
    assert disk(owner)[0]['last_status'] == 'exhausted'


@pytest.mark.parametrize('action', ['429', 'dead'])
def test_new_failure_from_inflight_reader_after_clear(fleet, monkeypatch, action):
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    p = worker(owner, local, action)
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', lambda t, **kw: t == token('a'))
    assert pool.select() is not None
    finish(p)
    expected = 'dead' if action == 'dead' else 'exhausted'
    assert disk(owner)[0]['last_status'] == expected
    assert fresh(local)['a'] == expected


def test_positive_cache_cannot_clear_a_new_failure(fleet, monkeypatch):
    from hermes_cli import auth_codex
    from contextlib import contextmanager
    from types import SimpleNamespace
    owner, local, _ = fleet
    calls = []
    @contextmanager
    def client(**kw):
        def get(url, headers):
            calls.append(url)
            return SimpleNamespace(status_code=200, json=lambda: {'rate_limit': {'primary_window': {'used_percent': 1}}})
        yield SimpleNamespace(get=get)
    monkeypatch.setattr(auth_codex, '_codex_http_client', client)
    monkeypatch.setattr(auth, '_codex_quota_probe_cache', {})
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', auth_codex._probe_codex_quota_restored)
    pool = cp.load_pool(P)
    # Both accounts are reported positive by this explicitly synthetic server.
    assert pool.select() is not None
    a = next(e for e in pool.entries() if e.id == 'a')
    pool._mark_exhausted(a, 429, {'reason': 'usage_limit_reached', 'reset_at': time.time()+86400})
    before = len(calls)
    available, _ = pool._available_entries(clear_expired=True)
    assert 'a' not in [e.id for e in available]
    assert len(calls) == before
    assert disk(owner)[0]['last_status'] == 'exhausted'


@pytest.mark.parametrize('concurrent', [False, True])
def test_pool_only_resolver_scopes_conditional_recovery(fleet, monkeypatch, concurrent):
    from hermes_cli.auth_codex import resolve_codex_runtime_credentials
    owner, local, _ = fleet
    p = worker(owner, local, '429') if concurrent else None
    def probe(t, **kw):
        assert t == token('a')
        if p is not None:
            finish(p)
        return True
    monkeypatch.setattr(auth, '_probe_codex_quota_restored', probe)
    if concurrent:
        with pytest.raises(auth.AuthError):
            resolve_codex_runtime_credentials(refresh_if_expiring=False)
    else:
        result = resolve_codex_runtime_credentials(refresh_if_expiring=False)
        assert result['api_key'] == token('a')
    assert disk(owner)[0]['last_status'] == ('exhausted' if concurrent else 'ok')
    assert disk(owner)[1]['last_status'] == 'exhausted'
    assert fresh(local)['a'] == ('exhausted' if concurrent else 'ok')


def test_stale_writer_cannot_replace_same_timestamp_failure(fleet):
    owner, local, _ = fleet
    p = worker(owner, local, 'stale')
    finish(worker(owner, local, '429'))
    failure = disk(owner)[0].get('_codex_failure_id')
    finish(p)
    assert failure
    assert disk(owner)[0].get('_codex_failure_id') == failure
    assert fresh(local)['a'] == 'exhausted'


def test_success_accounting_does_not_clear(fleet):
    from agent.turn_usage import record_response_usage
    from types import SimpleNamespace as NS
    owner, local, _ = fleet
    pool = cp.load_pool(P)
    before = owner.read_bytes()
    agent = NS(context_compressor=NS(), session_api_calls=0, model='synthetic-model', provider=P, credential_pool=pool)
    response = NS(content='synthetic success', usage=None)
    record_response_usage(agent, response, messages=[], api_call_count=1, api_duration=0.01,
                          compression_attempts=0, max_compression_attempts=2)
    assert agent.session_api_calls == 1
    assert owner.read_bytes() == before
    assert all(e.last_status == 'exhausted' for e in pool.entries())
