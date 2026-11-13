"""A finished delegation or background process resumes its unviewed local parent as a turn.

Reported case: a Desktop/webapp session dispatched four subagents and ended its turn; the laptop went
to sleep, the socket dropped, and every result was written as a note with no model call. When the
client reconnected nothing ran, and the session sat idle until the user typed "status update" over half
an hour later. A viewer going away is not the user ending the work: the result runs as the parent's
turn. Only a chat the user closed (exit, /new and the other viewer closes) keeps its results as notes,
and so do a manual cron run's finished text and the early per-task failure notice.
"""
import pytest

from tests.gateway.test_completion_admission import pending
from tests.gateway.test_session_end_delegations import _admissions, _local
from tests.gateway.test_unviewed_completion_notes import _process_event, _target, _turns
from tests.gateway.test_unviewed_delegation_failure_turn import _batch, _fail, _notes
from hermes_agent.tools import async_delegation as delegation


@pytest.fixture(autouse=True)
def _no_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


@pytest.mark.asyncio
async def test_finished_delegation_resumes_the_parent_with_no_viewer(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        assert not authority.sessions[ref.session_id].subscribers, 'no viewer is watching'
        evt = dict(pending(route, 'deleg_done'), parent_session_id=target)
        assert await runner._deliver_async_delegation_group([evt]) is True
        seen = _turns(runner, authority, target, 'deleg_done')
        await authority._drain(ref)
        assert len(seen) == 1 and '[ASYNC DELEGATION COMPLETE — deleg_done]' in seen[0][0]
        assert _notes(authority, target, 'deleg_done') == []
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_finished_process_resumes_the_parent_with_no_viewer(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        evt = _process_event(authority, ref, target, 'proc_done')
        assert await runner._deliver_completion_notification('[finished] RESULT_proc_done', evt) is True
        seen = _turns(runner, authority, target, 'proc_done')
        await authority._drain(ref)
        assert len(seen) == 1 and seen[0][0].startswith('[finished] RESULT_proc_done')
        assert _notes(authority, target, 'proc_done') == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['ws_orphan_reap', 'startup_orphan_reap', 'agent_close'])
async def test_row_ended_without_the_user_reopens_for_the_result(tmp_path, monkeypatch, reason):
    """The orphan reaper ends the row of a viewer that vanished (a sleeping laptop): that is not the
    user's close, so the result reopens the row and runs as the turn."""
    from hermes_agent.hermes_state_local import end_idle_local_session
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                      target_id=target, reason=reason)
        evt = dict(pending(route, 'deleg_reaped'), parent_session_id=target)
        assert await runner._deliver_async_delegation_group([evt]) is True
        seen = _turns(runner, authority, target, 'deleg_reaped')
        await authority._drain(ref)
        assert len(seen) == 1
        assert authority.db.get_session(target)['ended_at'] is None, 'the turn reopened the row'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_result_in_a_chat_the_user_closed_stays_a_note(tmp_path, monkeypatch):
    """Exit or /new refuses the session's dispatches; a viewer attaching lifts it and results resume."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        token = (authority.profile_id, ref.session_id)
        delegation.refuse_for_session(token, session_key=route, parent_session_ids=(ref.session_id, target))
        seen = _turns(runner, authority, target, 'deleg_closed')
        evt = dict(pending(route, 'deleg_closed0'), parent_session_id=target)
        assert await runner._deliver_async_delegation_group([evt]) is True
        await authority._drain(ref)
        assert seen == [] and len(_notes(authority, target, 'deleg_closed0')) == 1
        delegation.allow_for_session(token)
        evt = dict(pending(route, 'deleg_closed1'), parent_session_id=target)
        assert await runner._deliver_async_delegation_group([evt]) is True
        await authority._drain(ref)
        assert len(seen) == 1 and 'deleg_closed1' in seen[0][0]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_successful_manual_cron_run_stays_a_note(tmp_path, monkeypatch):
    """``cronjob(action='run')``'s outcome is finished text, as a scheduled run's is."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        evt = dict(pending(route, 'deleg_cron_ok'), role='cron_run')
        assert await runner._deliver_async_delegation_group([dict(evt, parent_session_id=target)]) is True
        seen = _turns(runner, authority, target, 'deleg_cron_ok')
        await authority._drain(ref)
        assert seen == [] and len(_notes(authority, target, 'deleg_cron_ok')) == 1
    finally:
        authority.db.close()


def test_resume_predicate_covers_results_not_diagnostics():
    from hermes_agent.tools.process_registry_notifications import completion_resumes_parent
    assert completion_resumes_parent({'type': 'async_delegation', 'status': 'completed'})
    assert completion_resumes_parent({'type': 'completion', 'exit_code': 0})
    assert not completion_resumes_parent({'type': 'async_delegation', 'task_failure_notice': True})
    assert not completion_resumes_parent({'type': 'async_delegation', 'role': 'cron_run'})
    assert completion_resumes_parent({'type': 'async_delegation', 'role': 'cron_run', '_sibling_resumes': True})
    for kind in ('watch_match', 'watch_disabled', 'heartbeat', 'cron'):
        assert not completion_resumes_parent({'type': kind})


@pytest.mark.asyncio
async def test_successes_do_not_spend_the_failure_wake_budget(tmp_path, monkeypatch):
    """A long unattended pipeline (dispatch, finish, dispatch again) keeps going; only failed results in
    a row are bounded by ``UNATTENDED_WAKE_LIMIT``."""
    from hermes_agent.gateway.session_completion_notes import UNATTENDED_WAKE_LIMIT
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        seen = _turns(runner, authority, target, 'deleg_step')
        for i in range(UNATTENDED_WAKE_LIMIT + 2):
            evt = dict(pending(route, f'deleg_step{i}'), parent_session_id=target)
            assert await runner._deliver_async_delegation_group([evt]) is True
            await authority._drain(ref)
        assert len(seen) == UNATTENDED_WAKE_LIMIT + 2, 'every finished step resumed the parent'
        for i in range(UNATTENDED_WAKE_LIMIT + 1):
            await _fail(runner, authority, ref, route, target, f'deleg_stepfail{i}')
        assert len(seen) == 2 * UNATTENDED_WAKE_LIMIT + 2, 'failures are still bounded'
        assert len(_notes(authority, target, 'deleg_stepfail')) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason, resumes', [('cli_close', False), ('new_session', False), ('tui_close', False),
                                             ('ws_orphan_reap', True), ('startup_orphan_reap', True)])
async def test_close_pending_across_a_restart_still_decides(tmp_path, monkeypatch, reason, resumes):
    """The user exits while the result is queued (the close waits for the idle FIFO), then the gateway
    restarts: recovery re-arms the stored close, but the in-memory dispatch refusal is gone."""
    from hermes_agent.gateway.config import GatewayConfig
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session_authority import initialize_session_authority
    from hermes_agent.gateway.session_local_recovery import recover_local_sessions
    from hermes_agent.gateway.session_viewer_close import _PendingClose, _store_and_settle
    from hermes_agent.hermes_state import AsyncSessionDB, SessionDB
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    target = _target(runner, authority, ref)
    evt = _process_event(authority, ref, target, 'proc_pending_close')
    assert await runner._deliver_completion_notification('[finished] RESULT_proc_pending_close', evt) is True
    authority.pending_closes[ref.session_id] = close = _PendingClose(reason)
    await _store_and_settle(authority, ref.session_id, authority.sessions[ref.session_id], close)
    assert authority.db.get_session(target)['ended_at'] is None, 'the queued result holds the close'
    authority.db.close()
    delegation._reset_for_tests()  # the next process starts with no refusals
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    runner.session_store._db = db = SessionDB(tmp_path / 'state.db')
    runner._session_db = AsyncSessionDB(db)
    restarted = await initialize_session_authority(runner, profile_id='default', instance_id='lc7b', db=db)
    try:
        recover_local_sessions(restarted, schedule=True)
        seen = _turns(runner, restarted, target, 'proc_pending_close')
        await restarted._drain(ref)
        assert len(seen) == int(resumes) and len(_notes(restarted, target, 'proc_pending_close')) == 1 - int(resumes)
        assert restarted.db.get_session(target)['end_reason'] == reason, 'the close settled after it'
    finally:
        db.close()
