"""A gated cron fire leaves no open, empty session in the default session listing.

``submit`` commits the fire's session before ``run_job`` runs its gates (wake gate, empty script
output, monitor gate, injection block). A fire that returned before the agent ran used to leave
that row open and untitled, so every client listing sessions showed one empty chat per gated tick.
"""
import pytest

from cron import executions, jobs, scheduler
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


async def _fire(owner, monkeypatch, run_job):
    job = jobs.create_job(prompt='check inbox', schedule='every 1h', deliver='local')['id']
    execution_id = executions.create_execution(job, source='test')['id']
    receipt = await session_cron.operation(
        owner, 'submit', {'job_id': job, 'request_id': execution_id, 'extra_prompt': None})
    sid = receipt['session_id']
    assert sid in {s['id'] for s in owner.db.list_sessions_rich()}
    monkeypatch.setattr(scheduler, 'run_job', run_job)
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
    policy = local_adapter_map(owner)[Platform.LOCAL].policies[sid]
    await session_cron.execute(owner, SessionRef(owner.profile_id, sid), row, policy)
    return job, sid


@pytest.mark.asyncio
async def test_a_gated_fire_leaves_no_visible_empty_session(owner, monkeypatch):
    def gated(job, **kwargs):
        return True, '', scheduler.SILENT_MARKER, None

    job, sid = await _fire(owner, monkeypatch, gated)

    assert sid not in {s['id'] for s in owner.db.list_sessions_rich()}
    session = owner.db.get_session(sid)
    assert session['ended_at'] is not None
    assert session['end_reason'] == session_cron.CRON_SKIPPED_END_REASON
    assert [r['id'] for r in owner.db.list_cron_job_runs(job)] == [sid]


@pytest.mark.asyncio
async def test_a_full_run_keeps_its_own_booking_and_stays_listed(owner, monkeypatch):
    def full(job, **kwargs):
        sid = session_cron.current_execution()[1]
        owner.db.append_message(sid, role='user', content='check inbox')
        owner.db.append_message(sid, role='assistant', content='done')
        owner.db.end_session(sid, 'cron_complete')
        return True, 'doc', 'done', None

    _, sid = await _fire(owner, monkeypatch, full)

    assert sid in {s['id'] for s in owner.db.list_sessions_rich()}
    assert owner.db.get_session(sid)['end_reason'] == 'cron_complete'
