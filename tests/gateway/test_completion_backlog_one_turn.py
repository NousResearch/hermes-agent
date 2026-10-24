"""A backlog of queued completions runs as one turn, not one turn per result.

Results that finish while a turn runs are each committed as their own admission, so a watched local
session used to replay the backlog as back-to-back turns once that turn ended (mostly "nothing new").
Every completion of a run but the last is now a note (the LC-7 note path, no model call) and the
last runs the one turn with the others already in its history. A human input, or any other
non-completion admission, ends the run: completions ahead of it still get their own turn.
"""
import pytest

from tests.gateway.test_session_end_delegations import _admissions, _deliver, _local
from tests.gateway.test_unviewed_completion_notes import _human, _target, _texts, _turns
from tools import async_delegation as delegation


@pytest.fixture(autouse=True)
def _no_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


def _proc(name):
    return ('completion', name, 1700000000.0)


def _watch(authority, ref, actor):
    authority.sessions[ref.session_id].subscribers['watching'] = actor


@pytest.mark.asyncio
async def test_completions_queued_behind_a_turn_run_as_one_turn(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        _watch(authority, ref, actor)
        _human(authority, ref, 'human', 'FIRST')
        for name in ('proc_a', 'proc_b', 'proc_c'):
            await _deliver(runner, authority, ref, _proc(name))
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('FIRST', 0), ('[finished] RESULT_proc_c', 2)], 'one turn, the earlier two in its history'
        assert [m['content'] for m in _texts(authority, target, 'RESULT_')] == [
            '[finished] RESULT_proc_a', '[finished] RESULT_proc_b']
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed'] * 3
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_human_input_ends_the_run(tmp_path, monkeypatch):
    """A completion ahead of a human input keeps its turn; the run never folds across the human."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        _watch(authority, ref, actor)
        await _deliver(runner, authority, ref, _proc('proc_a'))
        _human(authority, ref, 'human', 'MIDDLE')
        await _deliver(runner, authority, ref, _proc('proc_b'))
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert [text for text, _ in seen] == ['[finished] RESULT_proc_a', 'MIDDLE', '[finished] RESULT_proc_b']
        assert _texts(authority, target, 'RESULT_') == [], 'no note: every completion ran as its own turn'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_delegation_result_folds_into_the_next_process_turn(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        _watch(authority, ref, actor)
        await _deliver(runner, authority, ref, ('async_delegation', 'deleg_a', ''))
        await _deliver(runner, authority, ref, _proc('proc_b'))
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('[finished] RESULT_proc_b', 1)]
        assert [m['content'] for m in _texts(authority, target, 'RESULT_')] == ['[finished] RESULT_deleg_a']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_lone_watched_completion_is_still_a_turn(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        _watch(authority, ref, actor)
        await _deliver(runner, authority, ref, _proc('proc_a'))
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('[finished] RESULT_proc_a', 0)]
        assert _texts(authority, target, 'RESULT_') == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_another_automation_wake_ends_the_run(tmp_path, monkeypatch):
    """Only a completion carries the run's turn: a /loop tick or Kanban wake behind it keeps the
    completion's own turn, and the wake runs as its own."""
    from gateway.session_automation import admit_route_wake
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        _watch(authority, ref, actor)
        live = authority.sessions[ref.session_id]
        await _deliver(runner, authority, ref, _proc('proc_a'))
        assert await admit_route_wake(runner, runner._adapter_for_source(live.source), live.source,
                                      'TICK', 'loop-tick')
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert [text for text, _ in seen] == ['[finished] RESULT_proc_a', 'TICK']
        assert _texts(authority, target, 'RESULT_') == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_successor_its_claim_refuses_never_absorbs_the_head(tmp_path, monkeypatch):
    """A successor the preclaim check refuses pauses the FIFO and would never answer a note: the head
    runs its own turn instead of being folded onto it."""
    import gateway.session_automation as automation
    from hermes_state_runtime import RuntimeStoreError
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    stock = automation.check_local_automation

    def refuse_b(authority, ref, row, **kwargs):
        if 'proc_b' in row['request_id']:
            raise RuntimeStoreError('admission_conflict')
        return stock(authority, ref, row, **kwargs)
    monkeypatch.setattr(automation, 'check_local_automation', refuse_b)
    try:
        target = _target(runner, authority, ref)
        _watch(authority, ref, actor)
        await _deliver(runner, authority, ref, _proc('proc_a'))
        await _deliver(runner, authority, ref, _proc('proc_b'))
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert [text for text, _ in seen] == ['[finished] RESULT_proc_a']
        assert _texts(authority, target, 'RESULT_') == [], 'the head was not folded onto a refused successor'
        assert [r['status'] for r in _admissions(authority, ref)] == ['terminal', 'queued']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_watched_fold_keeps_the_cached_agent(tmp_path, monkeypatch):
    """The fold re-baselines the cached agent like an unviewed note, so the run's turn reuses it."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target, route = _target(runner, authority, ref), authority.sessions[ref.session_id].route
        _watch(authority, ref, actor)
        agent, before = object(), authority.db.get_session(target)['message_count']
        runner._agent_cache[route] = (agent, 'sig', before, target)
        await _deliver(runner, authority, ref, _proc('proc_a'))
        await _deliver(runner, authority, ref, _proc('proc_b'))
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert seen == [('[finished] RESULT_proc_b', 1)]
        after = authority.db.get_session(target)['message_count']
        assert after == before + 1
        assert runner._agent_cache[route] == (agent, 'sig', after, target), 'the run turn would rebuild the agent'
    finally:
        authority.db.close()
