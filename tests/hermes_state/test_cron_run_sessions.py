"""A cron job's run history includes the runs the unified gateway runtime creates (#138812).

Since the unified runtime, an agent fire is created through the session authority
(``gateway/session_cron.py``) under a runtime-minted ``local-*`` id; only runs from before it
carry ``cron_{job_id}_{timestamp}``. These drive that real producer over SQLite (admission only,
no agent turn) and read back through ``SessionDB.list_cron_job_runs``.
"""
import sqlite3
from types import SimpleNamespace

import pytest

from cron import executions, jobs
from gateway import run, session_cron
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


@pytest.fixture()
def owner(tmp_path, monkeypatch):
    """A real SessionAuthority over ``tmp_path/state.db`` that admits cron fires without running them."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('model: test-model\n', encoding='utf-8')
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    db = SessionDB(tmp_path / 'state.db')
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()),
                             adapters={})
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    runner.session_authority = authority
    yield authority
    db.close()


async def _fire(authority, job_id):
    """One scheduler fire through the real producer: its ledger row, then the owner's submit."""
    execution_id = executions.create_execution(job_id, source='test')['id']
    receipt = await session_cron.operation(
        authority, 'submit', {'job_id': job_id, 'request_id': execution_id, 'extra_prompt': None})
    authority.db.append_message(receipt['session_id'], role='user', content=f'run {job_id}')
    return receipt['session_id'], execution_id


@pytest.mark.asyncio
async def test_unified_runs_are_listed_with_legacy_runs_and_only_for_their_job(owner):
    db = owner.db
    alpha = jobs.create_job(prompt='alpha report', schedule='every 1h', deliver='local')['id']
    beta = jobs.create_job(prompt='beta report', schedule='every 1h', deliver='local')['id']
    legacy = f'cron_{alpha}_20261011_120000'
    db.create_session(legacy, source='cron')  # a run from before the unified runtime
    db._conn.execute('UPDATE sessions SET started_at = started_at - 86400 WHERE id = ?', (legacy,))
    db._conn.commit()
    alpha_run, alpha_fire = await _fire(owner, alpha)
    beta_run, beta_fire = await _fire(owner, beta)
    assert alpha_run.startswith('local-') and beta_run.startswith('local-')

    alpha_runs = db.list_cron_job_runs(alpha)
    beta_runs = db.list_cron_job_runs(beta)

    assert [r['id'] for r in alpha_runs] == [alpha_run, legacy]  # newest first
    assert [r['id'] for r in beta_runs] == [beta_run]
    assert [(r.get('cron_job_id'), r.get('cron_execution_id')) for r in alpha_runs] == [
        (alpha, alpha_fire), (alpha, None)]
    assert beta_runs[0].get('cron_execution_id') == beta_fire
    assert alpha_runs[0]['preview'] == f'run {alpha}'
    assert [r['id'] for r in db.list_cron_job_runs(alpha, limit=1, offset=1)] == [legacy]


@pytest.mark.asyncio
async def test_an_execution_id_with_a_colon_is_refused_before_it_can_misfile_a_run(owner):
    """``cron:<job>:<execution>`` splits at its last ``:``, so a fire ``x:y`` of job ``a`` would read
    as job ``a:x``: missing from ``a``'s history while its detail names another job."""
    job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')['id']
    for operation in ('submit', 'recover'):
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            await session_cron.operation(
                owner, operation, {'job_id': job, 'request_id': 'fire:1', 'extra_prompt': None})
    assert owner.db.list_cron_job_runs(job) == []
    assert owner.db.list_cron_job_runs(f'{job}:fire') == []


def test_legacy_runs_of_a_job_whose_id_extends_another_stay_with_their_own_job(tmp_path):
    """``cron_daily_`` is a prefix of job ``daily_digest``'s run ids too; only an exact
    ``YYYYmmdd_HHMMSS`` suffix makes a legacy run ``daily``'s."""
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session('cron_daily_20261011_120000', source='cron')
        db.create_session('cron_daily_digest_20261011_130000', source='cron')

        daily = db.list_cron_job_runs('daily')
        digest = db.list_cron_job_runs('daily_digest')

    assert [(r['id'], r['cron_job_id']) for r in daily] == [('cron_daily_20261011_120000', 'daily')]
    assert [(r['id'], r['cron_job_id']) for r in digest] == [('cron_daily_digest_20261011_130000', 'daily_digest')]


def test_run_lookup_is_index_served_on_new_and_upgraded_stores(tmp_path, monkeypatch):
    """A job's unified runs come from an index range bounded by that job's runs, not from a scan
    that grows with every session. An older store gains the index on its next open."""
    path = tmp_path / 'state.db'
    with SessionDB(path):
        pass
    raw = sqlite3.connect(path)
    raw.execute('DROP INDEX IF EXISTS idx_session_admissions_cron_request')  # an older store
    raw.commit()
    raw.close()
    with SessionDB(path):  # reopening migrates it; doing it twice must be a no-op
        pass
    with SessionDB(path) as db:
        queries = []
        read_rows = db._read_rows
        monkeypatch.setattr(db, '_read_rows', lambda sql, params=(): queries.append((sql, params)) or read_rows(sql, params))
        assert db.list_cron_job_runs('alpha') == []
        [(sql, params)] = queries
        with db._read_ctx() as conn:
            plan = ' '.join(str(r[-1]) for r in conn.execute('EXPLAIN QUERY PLAN ' + sql, params))
    assert 'idx_session_admissions_cron_request' in plan, plan
