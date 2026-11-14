"""Gateway-owned tick -> admission -> FIFO -> run_job, with real scripts and a real agent.

Only the provider is a loopback fake. Suppressed fires retain ENDED resumable receipts and
terminal admissions, not open rows; no_agent bypasses admission. Exact retries and owner
restart must not repeat gates, inference or the firing scheduler's bookkeeping tail.
"""
import asyncio
from datetime import timedelta
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_state_local import POLICY_PREFIX
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tests.waiting import async_wait_until


def _counts(home, db):
    with db._read_ctx() as conn:
        counts = {
            'sessions': conn.execute("SELECT COUNT(*) FROM sessions WHERE source='cron'").fetchone()[0],
            'open_sessions': conn.execute("SELECT COUNT(*) FROM sessions WHERE source='cron' AND ended_at IS NULL").fetchone()[0],
            'receipts': conn.execute('SELECT COUNT(*) FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)).fetchone()[0],
            'admissions': conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0],
            'routing': conn.execute('SELECT COUNT(*) FROM gateway_routing').fetchone()[0],
        }
    mirror = home / 'sessions' / 'sessions.json'
    data = json.loads(mirror.read_text()) if mirror.exists() else {}
    counts['sessions_json'] = sum(1 for key in data if not key.startswith('_'))
    counts['journals'] = len(list((home / 'cron' / 'admissions').glob('*.json')))
    return counts


_NOTHING = dict(sessions=0, open_sessions=0, receipts=0, admissions=0, routing=0,
                sessions_json=0, journals=0)
_ENDED = dict(_NOTHING, sessions=1, receipts=1, admissions=1, routing=1)


@pytest.fixture
def owner(tmp_path, monkeypatch):
    from cron import jobs
    import cron.scheduler as sched
    from cron import executions
    from hermes_cli import runtime_provider

    home = tmp_path / 'home'
    for sub in ('cron', 'scripts', 'sessions'):
        (home / sub).mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('INVOCATION_ID', raising=False)
    monkeypatch.setattr(executions, 'EXECUTIONS_FILE', home / 'cron' / 'executions.db')
    monkeypatch.setattr(sched, '_hermes_home', home)
    monkeypatch.setattr(sched, '_sweep_mcp_orphans', lambda: None)
    monkeypatch.setattr(sched, '_resolve_cron_enabled_toolsets', lambda *a: [])
    sched._running_job_ids.clear()
    effects = {}
    for name in ('save_job_output', 'mark_job_run', '_deliver_result'):
        spy = Mock(wraps=getattr(sched, name))
        monkeypatch.setattr(sched, name, spy)
        effects[name] = spy
    with FakeLLMServer(lambda _: Text('agent done'), aux=lambda _: Text('agent done')) as server:
        write_hermes_home(home, server.base_url, extra_config='cron:\n  preflight: false\n')
        monkeypatch.setattr(runtime_provider, 'resolve_runtime_provider', lambda **kw: {
            'provider': 'custom', 'api_key': 'sk-fake-e2e', 'base_url': server.base_url,
            'api_mode': 'chat_completions'})
        with jobs.use_cron_store(home):
            yield SimpleNamespace(home=home, server=server, effects=effects)
    sched._shutdown_parallel_pool()


def _make_due(job_id):
    from cron import jobs
    stored = jobs.load_jobs()
    # Preserve subsecond precision: truncation aliases two fast test fires to the same occurrence.
    slot = (jobs._hermes_now() - timedelta(minutes=1)).isoformat()
    next(row for row in stored if row['id'] == job_id)['next_run_at'] = slot
    jobs.save_jobs(stored)


def _script(home, name, output):
    (home / 'scripts' / name).write_text(
        'from pathlib import Path\n'
        'p = Path(__file__).with_suffix(".count")\n'
        'p.write_text(str(int(p.read_text()) + 1) if p.exists() else "1")\n'
        + output + '\n', encoding='utf-8')
    return name


def _jobs(home):
    from cron import jobs
    (home / 'monitor-value').write_text('steady')
    _script(home, 'monitor.py', 'print((Path(__file__).parent.parent / "monitor-value").read_text())')
    _script(home, 'gate.py', 'print(\'{"wakeAgent": false}\')')
    _script(home, 'wake.py', 'print(\'{"wakeAgent": true, "data": "report"}\')')
    _script(home, 'report.py', 'print("no-agent report")')
    common = dict(schedule='every 1h', deliver='local', model='fake-model')
    return {
        'monitor': jobs.create_job(prompt='summarise the change', name='monitor', monitor_script='monitor.py', **common),
        'gate': jobs.create_job(prompt='only when woken', name='gate', script='gate.py', **common),
        'wake': jobs.create_job(prompt='report when woken', name='wake', script='wake.py', **common),
        'real': jobs.create_job(prompt='write the report', name='real', **common),
        'no_agent': jobs.create_job(prompt=None, name='plain script', script='report.py', no_agent=True, **common),
    }


async def _serve(home):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from hermes_state import AsyncSessionDB
    from hermes_state_registry import acquire
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / 'sessions'))
    db = acquire(home / 'state.db')
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='gated-fire', db=db)
    return runner, authority, db


def _stop(authority, db):
    from gateway.session_cron import unbind_owner
    from hermes_state_registry import release_or_close
    unbind_owner(authority)
    release_or_close(db)


async def _settled(authority):
    await async_wait_until(lambda: not any(
        live.task is not None and not live.task.done() for live in authority.sessions.values()), timeout=5)


def _effects(owner):
    return {name: spy.call_count for name, spy in owner.effects.items()}


def _admissions(db):
    with db._read_ctx() as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM session_admissions')]


async def _fire(owner, authority, db, job, *, script=None, agent=False, suppressed=False):
    from cron import jobs, scheduler
    before = _counts(owner.home, db)
    prior = {r['admission_id'] for r in _admissions(db)}
    requests = len(owner.server.requests)
    effects = _effects(owner)
    deliveries = effects['_deliver_result']
    counter = owner.home / 'scripts' / f'{script}.count' if script else None
    count = int(counter.read_text()) if counter and counter.exists() else 0
    _make_due(job['id'])
    assert await asyncio.to_thread(scheduler.tick, verbose=False) == 1
    await _settled(authority)
    after = _counts(owner.home, db)
    delta = {key: after[key] - before[key] for key in after}
    assert delta == (_NOTHING if job.get('no_agent') else _ENDED), delta
    assert jobs.get_job(job['id'])['last_status'] == 'ok'
    for name in ('save_job_output', 'mark_job_run'):
        assert owner.effects[name].call_count == effects[name] + 1
    if counter:
        assert int(counter.read_text()) == count + 1, 'gate/source executed more than once'
    if agent:
        assert len(owner.server.requests) > requests
    else:
        assert len(owner.server.requests) == requests
    if suppressed:
        assert owner.effects['_deliver_result'].call_count == deliveries
    new = [r for r in _admissions(db) if r['admission_id'] not in prior]
    assert len(new) == (0 if job.get('no_agent') else 1)
    if not new:
        return None
    row = new[0]
    assert row['status'] == 'terminal'
    session = db.get_session(row['target_session_id'])
    assert session['ended_at'] is not None
    messages = db.get_messages(row['target_session_id'])
    if suppressed:
        assert session['end_reason'] == 'cron_suppressed'
        assert messages == []
    else:
        assert any(m['role'] == 'assistant' and m['content'] == 'agent done' for m in messages)
    print('CRON_LIFECYCLE', job['name'], json.dumps(delta, sort_keys=True))
    return row


async def _retry(owner, authority, db, job, row):
    from gateway import session_cron
    from cron.scheduler_authority import run_canonical_job, reconcile_pending
    params = dict(job_id=job['id'], request_id=row['request_id'].rsplit(':', 1)[-1], extra_prompt=None)
    receipt = dict(session_id=row['target_session_id'], admission_id=row['admission_id'])
    expected = await session_cron.operation(authority, 'status', receipt)
    before = (_counts(owner.home, db), len(owner.server.requests), _effects(owner),
              {p.name: p.read_text() for p in (owner.home / 'scripts').glob('*.count')})
    assert await session_cron.operation(authority, 'submit', params) == receipt
    recovered = await session_cron.operation(authority, 'recover', params)
    assert recovered['status'] == 'terminal' and recovered['result'] == expected['result']
    result = await asyncio.to_thread(run_canonical_job, job, execution_id=params['request_id'])
    assert list(result) == expected['result']
    # This retry leaves a journal but the finished executions ledger suppresses tail replay.
    await asyncio.to_thread(reconcile_pending)
    await _settled(authority)
    assert (_counts(owner.home, db), len(owner.server.requests), _effects(owner),
            {p.name: p.read_text() for p in (owner.home / 'scripts').glob('*.count')}) == before


def test_gated_ticks_end_rows_retain_receipts_and_never_replay(owner):
    jobs = _jobs(owner.home)

    async def probe():
        _, authority, db = await _serve(owner.home)
        saved = []
        try:
            for changed in ('steady', 'steady', 'changed', 'changed'):
                (owner.home / 'monitor-value').write_text(changed)
                active = len(saved) in (0, 2)
                row = await _fire(owner, authority, db, jobs['monitor'], script='monitor', agent=active, suppressed=not active)
                saved.append((jobs['monitor'], row))
            for name, script, active, suppressed in (
                ('gate', 'gate', False, True), ('gate', 'gate', False, True),
                ('wake', 'wake', True, False), ('real', None, True, False),
                ('no_agent', 'report', False, False),
            ):
                row = await _fire(owner, authority, db, jobs[name], script=script, agent=active, suppressed=suppressed)
                if row:
                    saved.append((jobs[name], row))
            for job, row in saved:
                await _retry(owner, authority, db, job, row)
            counts = _counts(owner.home, db)
        finally:
            _stop(authority, db)
        # Fresh authority/epoch and DB handle, with exactly the persisted receipts.
        _, authority, db = await _serve(owner.home)
        try:
            await _settled(authority)
            assert _counts(owner.home, db) == counts
            for job, row in saved:
                await _retry(owner, authority, db, job, row)
            await _fire(owner, authority, db, jobs['gate'], script='gate', suppressed=True)
            await _fire(owner, authority, db, jobs['no_agent'], script='report')
        finally:
            _stop(authority, db)

    asyncio.run(probe())


def test_stalled_suppressed_end_releases_dispatch_guard(owner, monkeypatch):
    import threading
    import time
    from cron import scheduler
    jobs = _jobs(owner.home)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    async def probe():
        _, authority, db = await _serve(owner.home)
        end = db.end_session

        def stalled(session_id, reason):
            calls.append(session_id)
            entered.set()
            release.wait(5)
            try:
                return end(session_id, reason)
            finally:
                finished.set()

        monkeypatch.setattr(db, 'end_session', stalled)
        monkeypatch.setattr(scheduler, '_cron_cleanup_timeout_seconds', lambda: 0.02)
        try:
            _make_due(jobs['gate']['id'])
            start = time.monotonic()
            assert await asyncio.to_thread(scheduler.tick, verbose=False) == 1
            assert time.monotonic() - start < 2
            await _settled(authority)
            assert entered.is_set() and len(calls) == 1
            assert jobs['gate']['id'] not in scheduler.get_running_job_ids()
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
            assert _counts(owner.home, db)['open_sessions'] == 0
            await _fire(owner, authority, db, jobs['gate'], script='gate', suppressed=True)
            assert len(calls) == 2
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 2)
            _stop(authority, db)

    asyncio.run(probe())


def test_settled_suppressed_fire_can_be_pruned_with_exact_retry_tombstone(owner):
    from gateway.session_results import admission_result
    from hermes_state_runtime import admit_session_input, get_session_admission
    jobs = _jobs(owner.home)

    async def probe():
        _, authority, db = await _serve(owner.home)
        try:
            row = await _fire(owner, authority, db, jobs['gate'], script='gate', suppressed=True)
            row = get_session_admission(db, admission_id=row['admission_id'])
            sid = row['target_session_id']
            result = admission_result(db, row['admission_id'])
            assert db._execute_write(lambda conn: conn.execute('PRAGMA foreign_keys').fetchone()[0]) == 1
            assert await asyncio.to_thread(db.prune_sessions, older_than_days=None) == 1
            assert db.get_session(sid) is None
            assert _counts(owner.home, db) == _NOTHING
            with db._read_ctx() as conn:
                assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
            replay = admit_session_input(db, epoch=authority.epoch, principal_id=row['principal_id'],
                                         session_id=sid, request_id=row['request_id'], payload=row['payload'])
            assert replay['status'] == 'terminal' and replay['admission_id'] == row['admission_id']
            retained = admission_result(db, row['admission_id'])
            assert retained['result']['cron_result'] == result['result']['cron_result']
            assert retained['result']['messages'] == []
            assert retained['usage'] == result['usage']
            assert (owner.home / 'scripts' / 'gate.count').read_text() == '1'
            assert owner.server.requests == []
        finally:
            _stop(authority, db)

    asyncio.run(probe())
