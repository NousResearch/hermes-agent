"""Work results and supervision follow the original conversation across route changes."""
import pytest

from tests.gateway.test_process_work_results import enrolled, frame
from tests.gateway.test_detached_conversation_lifecycle import _moved_on
from tests.gateway.test_authority_lane_moves_on import _app_turn, _chat
from hermes_state_runtime import list_session_admissions
from tools import process_work_results as work


@pytest.mark.asyncio
@pytest.mark.parametrize('stop_during_validation', [False, True])
async def test_enrollment_fences_original_owner_after_resume(monkeypatch, stop_during_validation):
    from hermes_state import SessionDB

    _runner, adapter, authority, source, sent, _ran, _gates, lane, ref = await _moved_on(monkeypatch)
    sid = ref.session_id
    assert await _app_turn(authority, ref, 'start work') == ('terminal', 'completed')
    detached_route = authority.sessions[sid].route
    await _chat(adapter, source, sent, '/resume First chat')
    original_get = SessionDB.get_session
    crossed = False

    def validate(db, session_id, *args, **kwargs):
        nonlocal crossed
        if session_id == sid and not crossed:
            crossed = True
            if stop_during_validation:
                authority.fence_route(lane)
        return original_get(db, session_id, *args, **kwargs)

    monkeypatch.setattr(SessionDB, 'get_session', validate)
    if stop_during_validation:
        with pytest.raises(ValueError, match='crossed a user boundary'):
            enrolled(detached_route, sid)
        assert work.records(authority.db) == []
    else:
        enrolled(detached_route, sid)
        assert work.records(authority.db)[0]['disposition'] == 'enrolled'
    assert crossed


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['result', 'never_started', 'supervision_failed'])
@pytest.mark.parametrize('transition', ['resume', 'detach', 'stop'])
async def test_obligation_follows_original_routed_owner(monkeypatch, kind, transition):
    runner, adapter, authority, source, sent, _ran, _gates, lane, ref = await _moved_on(monkeypatch)
    sid = ref.session_id
    assert await _app_turn(authority, ref, 'start work') == ('terminal', 'completed')
    route = lane if transition == 'detach' else authority.sessions[sid].route
    registry, process = enrolled(route, sid)
    if transition != 'detach':
        await _chat(adapter, source, sent, '/resume First chat')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    if transition == 'stop':
        authority.fence_route(lane)
    if kind == 'result':
        registry._ingest_output(process, frame())
    elif kind == 'never_started':
        work.mark_never_started(process)
    else:
        work.finish_without_result(process)
    before = work.records(authority.db)[0]
    original = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    replacement = runner.session_store.peek_session_id(lane)
    replacement_rows = list_session_admissions(authority.db, session_id=replacement, pending_only=False)
    for _ in range(2):
        await runner._drain_watch_notifications(registry.completion_queue)
    result = work.records(authority.db)[0]
    expected = 'admitted' if kind == 'result' else kind + '_admitted'
    assert result['disposition'] == ('dropped' if transition == 'stop' else expected)
    assert (result['parent'], result['route'], result['payload'], result['error']) == (
        sid, route, before['payload'], before['error'])
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == len(original) + (transition != 'stop')
    if transition != 'stop':
        assert rows[-1]['payload']['text']
    if replacement != sid:
        assert list_session_admissions(authority.db, session_id=replacement, pending_only=False) == replacement_rows
