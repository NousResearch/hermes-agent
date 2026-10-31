"""Dashboard/Desktop run history for agent cron runs the unified gateway runtime creates (#138812).

A unified-runtime fire is a healthy ``source='cron'`` session under a runtime-minted ``local-*``
id. Run history used to find it only by the legacy ``cron_{job_id}_{timestamp}`` id, so the run
degraded to an inert ``cron_output`` doc row (no transcript, no usage, double-represented) and
session detail could not say whether the scheduler still owned it. These drive the real producer
(``gateway/session_cron.py``) over SQLite, the real execution ledger and output docs, and read
through the real routes in a temp default profile.
"""
from datetime import timedelta
from zoneinfo import ZoneInfo

import pytest

import hermes_cli.web_routers.cron as cron_routes
import hermes_cli.web_routers.sessions as session_routes
import hermes_time
from cron import executions, jobs
from hermes_state_runtime import claim_session_input, settle_session_input
from tests.fakes.cron_owner import admit_fire, cron_owner


@pytest.fixture()
def owner(tmp_path, monkeypatch):
    """The default profile's gateway authority: admits cron fires through the real producer
    (without running an agent turn) into the state.db the dashboard routes read."""
    from hermes_cli import profiles

    home = tmp_path / '.hermes'
    (home / 'cron').mkdir(parents=True)
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: home)
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: home / 'profiles')
    authority = cron_owner(home, monkeypatch)
    # Pin the configured zone so the output writer and the run-history reader resolve the same
    # one, whatever the process's C-level zone (conftest sets TZ=UTC in the env without tzset).
    utc = ZoneInfo('UTC')
    monkeypatch.setattr(hermes_time, 'get_timezone', lambda: utc)
    monkeypatch.setattr(cron_routes, '_get_timezone', lambda: utc)
    yield authority
    authority.db.close()


@pytest.fixture()
def clock(monkeypatch):
    """Seconds added to the ledger's and the output writer's wall clock."""
    offset = {'seconds': 0.0}

    def shifted():
        return hermes_time.now() + timedelta(seconds=offset['seconds'])

    monkeypatch.setattr(executions, '_hermes_now', shifted)
    monkeypatch.setattr(jobs, '_hermes_now', shifted)
    return offset


async def _start_fire(authority, job_id):
    """A scheduler fire as it starts: its running ledger row, then the owner admits and claims it."""
    session_id, execution_id = await admit_fire(authority, job_id, running=True)
    claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
    return session_id, execution_id, claim


def _close_run(authority, session_id, claim):
    """The owner's run finishes: the answer is written, the admission settled, the session ended."""
    authority.db.append_message(session_id, role='assistant', content='report done')
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=claim['admission_id'],
                         generation=claim['generation'], outcome='completed',
                         result={'result': {'cron_result': [True, 'doc', 'report done', None]}, 'usage': {}})
    authority.db.end_session(session_id, 'cron_complete')


def _runs(job_id):
    return cron_routes._list_cron_job_runs_sync(job_id, profile='default', limit=20)['runs']


async def _detail(session_id):
    return await session_routes.get_session_detail(session_id, profile='default')


@pytest.mark.asyncio
async def test_unified_run_is_one_owned_session_row_and_detail_agrees(owner, clock):
    job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')['id']
    other = jobs.create_job(prompt='other report', schedule='every 1h', deliver='local')['id']
    session_id, execution_id, claim = await _start_fire(owner, job)
    await _start_fire(owner, other)

    # While it runs: the transcript is a row of this job only, owned by the scheduler.
    [running] = _runs(job)
    assert (running['id'], running['source'], running['scheduler_owned']) == (session_id, 'cron', True)
    detail = await _detail(session_id)
    assert detail.get('scheduler_owned') is True
    assert running.get('cron_execution_id') == execution_id
    assert (detail.get('cron_job_id'), detail.get('cron_execution_id')) == (job, execution_id)

    # Done: the agent fire's output doc is the same execution as its session, not a second row.
    _close_run(owner, session_id, claim)
    clock['seconds'] = 5
    jobs.save_job_output(job, '# Cron Job: report\n\nreport done')
    executions.finish_execution(execution_id, success=True)

    [finished] = _runs(job)
    assert (finished['id'], finished['source'], finished['message_count']) == (session_id, 'cron', 2)
    assert finished['scheduler_owned'] is False
    assert (await _detail(session_id))['scheduler_owned'] is False


@pytest.mark.asyncio
async def test_detail_still_recognises_legacy_runs_and_leaves_other_sessions_alone(owner):
    job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')['id']
    legacy = f'cron_{job}_20261011_120000'
    owner.db.create_session(legacy, source='cron')
    owner.db.end_session(legacy, 'cron_complete')
    owner.db.create_session('20261015_120000_abc123', source='cli')

    assert (await _detail(legacy))['scheduler_owned'] is False
    assert 'scheduler_owned' not in await _detail('20261015_120000_abc123')
    assert [r['id'] for r in _runs(job)] == [legacy]


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_fire_wrote_doc', [False, True])
async def test_a_fire_that_failed_beside_a_unified_run_keeps_its_own_row(owner, clock, failed_fire_wrote_doc):
    """Output rows pair with sessions per execution: the next fire failing a minute later, before
    it created a session, is its own run, not swallowed by its neighbour's time window."""
    job = jobs.create_job(prompt='report', schedule='every 1m', deliver='local')['id']
    session_id, first, claim = await _start_fire(owner, job)
    _close_run(owner, session_id, claim)
    clock['seconds'] = 20
    jobs.save_job_output(job, '# Cron Job: report\n\nreport done')
    executions.finish_execution(first, success=True)
    clock['seconds'] = 60
    second = executions.create_execution(job, source='test')['id']
    executions.mark_execution_running(second)
    clock['seconds'] = 70
    if failed_fire_wrote_doc:
        # Its doc also falls in the first run's grace window: it must stay the failed fire's row.
        jobs.save_job_output(job, '# Cron Job: report\n\nscript failed')
    executions.finish_execution(second, success=False, error='owner_unavailable')

    runs = _runs(job)

    assert [r['source'] for r in runs] == ['cron_output', 'cron'], runs
    assert runs[0]['title'].startswith('FAILED')
    if not failed_fire_wrote_doc:
        assert runs[0]['preview'] == 'owner_unavailable'
    assert (runs[1]['id'], runs[1]['cron_execution_id']) == (session_id, first)


@pytest.mark.asyncio
@pytest.mark.parametrize('neighbour_doc_survives', [True, False])
async def test_a_same_second_output_doc_is_not_pinned_on_the_earlier_execution(
        owner, monkeypatch, neighbour_doc_survives):
    """Output docs never record their execution; the filename keeps whole seconds only. An agent
    fire claimed at ``:00.500`` writing at ``:00.750`` files its doc as ``:00``, before its claim,
    while an earlier script-only fire that failed within the grace also spans that second. The
    guess must not decide the doc is that neighbour's: one row per execution, and the neighbour's
    failure keeps its own row."""
    start = hermes_time.now().replace(microsecond=0) - timedelta(hours=1)
    wall = {'at': start}
    monkeypatch.setattr(executions, '_hermes_now', lambda: wall['at'])
    monkeypatch.setattr(jobs, '_hermes_now', lambda: wall['at'])

    def at(seconds):
        wall['at'] = start + timedelta(seconds=seconds)
        return wall['at'].timestamp()

    job = jobs.create_job(prompt='report', schedule='every 1m', deliver='local')['id']
    at(0)
    neighbour = executions.create_execution(job, source='test')['id']  # a script-only fire
    executions.mark_execution_running(neighbour)
    at(1)
    neighbour_doc = jobs.save_job_output(job, '# Cron Job: report\n\nscript failed')
    executions.finish_execution(neighbour, success=False, error='script_failed')
    if not neighbour_doc_survives:
        neighbour_doc.unlink()

    at(60.5)  # the job switched to agent mode; its next fire is claimed mid-second
    session_id, fire, claim = await _start_fire(owner, job)
    _close_run(owner, session_id, claim)
    at(60.75)
    jobs.save_job_output(job, '# Cron Job: report\n\nreport done')  # filed as second 60
    executions.finish_execution(fire, success=True)
    owner.db._conn.execute('UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?',
                           (start.timestamp() + 60.6, start.timestamp() + 60.9, session_id))
    owner.db._conn.commit()

    runs = _runs(job)

    assert [r['source'] for r in runs] == ['cron', 'cron_output'], runs
    assert (runs[0]['id'], runs[0]['cron_execution_id']) == (session_id, fire)
    assert runs[1]['title'].startswith('FAILED')
    assert ('script failed' if neighbour_doc_survives else 'script_failed') in runs[1]['preview']


@pytest.mark.asyncio
@pytest.mark.parametrize('overlap_finishes_after_doc', [False, True])
async def test_an_overlapping_attempt_claimed_before_a_runs_doc_keeps_its_own_row(
        owner, clock, overlap_finishes_after_doc):
    """A manual ``hermes cron run`` overlapping a tick: attempt B is claimed while run A is still
    going, before A writes its doc, and fails without creating a session. B is the doc's newest
    candidate, but the doc is A's: it folds into A's session, and B keeps its own failed row."""
    job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')['id']
    session_id, fire, claim = await _start_fire(owner, job)
    clock['seconds'] = 10
    overlap = executions.create_execution(job, source='test')['id']
    executions.mark_execution_running(overlap)
    if not overlap_finishes_after_doc:
        clock['seconds'] = 12
        executions.finish_execution(overlap, success=False, error='owner_unavailable')
    _close_run(owner, session_id, claim)
    clock['seconds'] = 20
    jobs.save_job_output(job, '# Cron Job: report\n\nreport done')
    executions.finish_execution(fire, success=True)
    if overlap_finishes_after_doc:
        clock['seconds'] = 25
        executions.finish_execution(overlap, success=False, error='owner_unavailable')

    runs = _runs(job)

    assert [r['source'] for r in runs] == ['cron_output', 'cron'], runs
    assert runs[0]['title'].startswith('FAILED') and runs[0]['preview'] == 'owner_unavailable', runs
    assert (runs[1]['id'], runs[1]['cron_execution_id']) == (session_id, fire)
