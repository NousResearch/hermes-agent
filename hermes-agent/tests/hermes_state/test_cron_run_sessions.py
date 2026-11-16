"""A cron job's run history includes the runs the unified gateway runtime creates (#138812).

Since the unified runtime, an agent fire is created through the session authority
(``gateway/session_cron.py``) under a runtime-minted ``local-*`` id; only runs from before it
carry ``cron_{job_id}_{timestamp}``. These drive that real producer over SQLite (admission only,
no agent turn) and read back through ``SessionDB.list_cron_job_runs``.
"""
import sqlite3

import pytest

from cron import jobs
from gateway import session_cron
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tests.fakes.cron_owner import admit_fire, cron_owner


@pytest.fixture()
def owner(tmp_path, monkeypatch):
    """A real SessionAuthority over ``tmp_path/state.db`` that admits cron fires without running them."""
    authority = cron_owner(tmp_path, monkeypatch)
    yield authority
    authority.db.close()


@pytest.mark.asyncio
async def test_unified_runs_are_listed_with_legacy_runs_and_only_for_their_job(owner):
    db = owner.db
    alpha = jobs.create_job(prompt='alpha report', schedule='every 1h', deliver='local')['id']
    beta = jobs.create_job(prompt='beta report', schedule='every 1h', deliver='local')['id']
    legacy = f'cron_{alpha}_20261011_120000'
    db.create_session(legacy, source='cron')  # a run from before the unified runtime
    db._conn.execute('UPDATE sessions SET started_at = started_at - 86400 WHERE id = ?', (legacy,))
    db._conn.commit()
    alpha_run, alpha_fire = await admit_fire(owner, alpha)
    beta_run, beta_fire = await admit_fire(owner, beta)
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
async def test_a_fresh_execution_id_with_a_colon_is_refused_before_it_can_misfile_a_run(owner):
    """``cron:<job>:<execution>`` splits at its last ``:``, so a fire ``x:y`` of job ``a`` would read
    as job ``a:x``: missing from ``a``'s history while its detail names another job."""
    job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')['id']
    params = {'job_id': job, 'request_id': 'fire:1', 'extra_prompt': None}
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await session_cron.operation(owner, 'submit', params)
    assert await session_cron.operation(owner, 'recover', params) == {'status': 'missing', 'result': None}
    assert owner.db.list_cron_job_runs(job) == []
    assert owner.db.list_cron_job_runs(f'{job}:fire') == []


class _PreUpgradeRequestId(str):
    """An execution id as the producer saw it before ``:`` was refused: the guard finds none."""

    def __contains__(self, item):
        return item != ':' and super().__contains__(item)


@pytest.mark.asyncio
async def test_an_execution_id_admitted_with_a_colon_before_the_upgrade_keeps_its_admission(owner):
    """A durable ``cron:<job>:fire:1`` admission from before the guard is still recovered and
    retried under its original identity: same admission back, a changed payload refused, nothing
    replayed, and history files it under its job by the stored policy, not the last ``:``."""
    job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')['id']
    params = {'job_id': job, 'request_id': 'fire:1', 'extra_prompt': 'extra'}
    receipt = await session_cron.operation(
        owner, 'submit', dict(params, request_id=_PreUpgradeRequestId('fire:1')))

    def admissions():
        return owner.db._conn.execute('SELECT admission_id, request_id FROM session_admissions').fetchall()
    before = [tuple(r) for r in admissions()]

    assert await session_cron.operation(owner, 'submit', params) == receipt
    recovered = await session_cron.operation(owner, 'recover', params)
    assert (recovered['session_id'], recovered['admission_id']) == (receipt['session_id'], receipt['admission_id'])
    for operation in ('submit', 'recover'):
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await session_cron.operation(owner, operation, dict(params, extra_prompt='changed'))
    assert [tuple(r) for r in admissions()] == before == [(receipt['admission_id'], f'cron:{job}:fire:1')]

    sid = receipt['session_id']
    assert [(r['id'], r['cron_job_id'], r['cron_execution_id']) for r in owner.db.list_cron_job_runs(job)] == [
        (sid, job, 'fire:1')]
    assert owner.db.list_cron_job_runs(f'{job}:fire') == []
    assert owner.db.cron_run_identity(sid) == {'cron_job_id': job, 'cron_execution_id': 'fire:1'}


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
