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
