"""A completion queued behind a finite turn (``chat -q``, ``--oneshot``) is a note, never a second turn.

Regression for #156255 (#143547 CLI-56). A background process the one-shot turn started finishes while
that turn still runs; its completion waits in the FIFO behind the finite admission and is claimed the
moment the reply is sent, while the one-shot client is still attached for a few hundred milliseconds.
The viewer check saw that client and ran the completion as a new model turn nobody would read, with a
fresh budget, still changing files after the command had exited. The finite client's contract ends at
its reply, so the completion is written into the session as a note, as it is once the client detaches.
"""
import pytest

from persistence_assembly import close_stores_under
from tests.gateway.test_session_end_delegations import _admissions, _local, _notes
from tests.gateway.test_unviewed_completion_notes import _human, _process_event, _target, _turns
from tools import async_delegation as delegation
from gateway.session_authority import admission_row


@pytest.fixture(autouse=True)
def _no_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


def _input(authority, ref, request_id, text, finite):
    return admission_row(authority.persistence.admissions.admit(
        epoch=authority.epoch, principal_id='test-owner', session_id=ref.session_id, request_id=request_id,
        payload={'text': text, 'finite': finite})[0])


async def _run(tmp_path, monkeypatch, inputs):
    """Watch the session, admit *inputs* ``(text, finite)``, queue a finished process behind them and
    drain: ``(turns run, completion notes written)``."""
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        target = _target(runner, authority, ref)
        authority.sessions[ref.session_id].subscribers['oneshot'] = actor  # the client is still attached
        for i, (text, finite) in enumerate(inputs):
            _input(authority, ref, f'in{i}', text, finite)
        evt = _process_event(authority, ref, target, 'proc_finite')
        assert await runner._deliver_completion_notification('[finished] RESULT_proc_finite', evt) is True
        seen = _turns(runner, authority, target)
        await authority._drain(ref)
        assert all(r['status'] == 'terminal' for r in _admissions(authority, ref))
        return [text for text, _notes_seen in seen], _notes(authority, target)
    finally:
        close_stores_under(authority.persistence.scope.profile_dir)


@pytest.mark.asyncio
async def test_completion_behind_a_finite_turn_is_a_note(tmp_path, monkeypatch):
    turns, notes = await _run(tmp_path, monkeypatch, [('BUILD', True)])
    assert turns == ['BUILD'], 'no second, unattended turn after the one-shot reply'
    assert len(notes) == 1 and 'RESULT_proc_finite' in str(notes[0])


@pytest.mark.asyncio
async def test_completion_behind_an_interactive_turn_still_runs(tmp_path, monkeypatch):
    turns, notes = await _run(tmp_path, monkeypatch, [('ASK', False)])
    assert len(turns) == 2 and turns[1].startswith('[finished] RESULT_proc_finite')
    assert notes == []


@pytest.mark.asyncio
async def test_interactive_input_after_a_finite_one_restores_the_turn(tmp_path, monkeypatch):
    """A one-shot into a chat someone then keeps using: the latest input decides."""
    turns, notes = await _run(tmp_path, monkeypatch, [('BUILD', True), ('NEXT', False)])
    assert turns[:2] == ['BUILD', 'NEXT'] and len(turns) == 3
    assert notes == []
