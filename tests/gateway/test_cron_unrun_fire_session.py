"""A cron fire whose agent never runs still ends the session row the owner pre-created for it.

``gateway/session_cron.py`` commits a fire's ``local-*`` row before ``run_job``; only the agent path
(``_finalize_cron_session``) ended it, so every suppressed monitor tick left an empty row open until
a restart or the stale-open sweep, listed as an active cron session.
"""
import pytest

from cron import jobs
from cron.monitor import check_monitor
from gateway import session_cron
from gateway.config import Platform
from gateway.session_contract import SessionRef
from gateway.session_local_recovery import local_adapter_map
from hermes_state_runtime import claim_session_input
from tests.fakes.cron_owner import cron_owner


@pytest.fixture()
def owner(tmp_path, monkeypatch):
    authority = cron_owner(tmp_path, monkeypatch)
    yield authority
    authority.db.close()


@pytest.mark.asyncio
async def test_suppressed_monitor_fire_ends_its_owner_session_row(owner, tmp_path):
    from cron.executions import create_execution

    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'mon.sh').write_text("echo 'state A'\n", encoding='utf-8')
    job = jobs.create_job(prompt='Summarize what changed', model='test-model', schedule='every 5m',
                          monitor_script='mon.sh', deliver='local')
    # The previous tick already saw this exact output, so this fire's monitor gate suppresses the agent.
    assert check_monitor(job).changed

    execution_id = create_execution(job['id'], source='test')['id']
    receipt = await session_cron.operation(
        owner, 'submit', {'job_id': job['id'], 'request_id': execution_id, 'extra_prompt': None})
    sid = receipt['session_id']
    assert owner.db.get_session(sid)['ended_at'] is None  # pre-created, open, before the run

    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
    policy = local_adapter_map(owner)[Platform.LOCAL].policies[sid]
    result = await session_cron.execute(owner, SessionRef(owner.profile_id, sid), row, policy)

    assert result == '[SILENT]'
    assert 'no_change (agent run suppressed)' in owner.pending_results[row['admission_id']]['result']['cron_result'][1]
    session = owner.db.get_session(sid)
    assert session['ended_at'] is not None
    assert session['end_reason'] == 'cron_agent_skipped'


@pytest.mark.asyncio
async def test_fire_blocked_before_the_agent_ends_its_owner_session_row(owner):
    """The sibling bail inside run_job's try: no provider credential blocks setup, no agent runs."""
    from cron.executions import create_execution

    job = jobs.create_job(prompt='report', model='test-model', schedule='every 5m', deliver='local')
    execution_id = create_execution(job['id'], source='test')['id']
    receipt = await session_cron.operation(
        owner, 'submit', {'job_id': job['id'], 'request_id': execution_id, 'extra_prompt': None})
    sid = receipt['session_id']
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
    policy = local_adapter_map(owner)[Platform.LOCAL].policies[sid]
    with pytest.raises(RuntimeError, match='blocked_config'):
        await session_cron.execute(owner, SessionRef(owner.profile_id, sid), row, policy)

    session = owner.db.get_session(sid)
    assert session['ended_at'] is not None
    assert session['end_reason'] == 'cron_agent_skipped'


@pytest.mark.asyncio
async def test_fire_whose_prompt_preparation_raises_ends_its_owner_session_row(owner, tmp_path, monkeypatch):
    """Preparation runs before run_job's main try/finally: a monitor source that raises must not strand the row."""
    import cron.monitor
    from cron.executions import create_execution

    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'mon.sh').write_text("echo 'state A'\n", encoding='utf-8')
    job = jobs.create_job(prompt='Summarize what changed', model='test-model', schedule='every 5m',
                          monitor_script='mon.sh', deliver='local')

    def unreadable(job):
        raise OSError('monitor snapshot unreadable')
    monkeypatch.setattr(cron.monitor, 'check_monitor', unreadable)
    execution_id = create_execution(job['id'], source='test')['id']
    receipt = await session_cron.operation(
        owner, 'submit', {'job_id': job['id'], 'request_id': execution_id, 'extra_prompt': None})
    sid = receipt['session_id']
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
    policy = local_adapter_map(owner)[Platform.LOCAL].policies[sid]
    with pytest.raises(OSError, match='monitor snapshot unreadable'):
        await session_cron.execute(owner, SessionRef(owner.profile_id, sid), row, policy)

    session = owner.db.get_session(sid)
    assert session['ended_at'] is not None
    assert session['end_reason'] == 'cron_agent_skipped'
