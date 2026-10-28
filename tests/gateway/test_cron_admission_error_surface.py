"""A message typed into a cron session fails with a renderable read-only descriptor (#141348)."""
import json
import threading
from types import SimpleNamespace

import pytest

# Import-time update recovery stats the checkout's git dir; collection runs before the home I/O guard.
import gateway.session_managed_worker
from gateway.session_authority import SessionAuthority


_SURFACE = {'layer': 'gateway', 'code': 'cron_session_read_only', 'retryable': False}


async def _typed_turn_in_cron_session(owner, monkeypatch):
    """Run a person's message through a cron session's real executor: ``(row, response, outcome, captured)``."""
    from cron import scheduler
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_contract import SessionRef
    from gateway.session_local import LocalSessionAdapter
    from hermes_state_runtime import admit_session_input, claim_session_input

    session_id = 'cron-session'
    owner.db.create_session(session_id, source='local')
    # The job's last fire: the history a reopened chat shows above the refusal.
    owner.db.append_message(session_id, 'user', 'summarize the inbox')
    owner.db.append_message(session_id, 'assistant', 'Inbox summary: nothing urgent.')
    source = SessionSource(platform=Platform.LOCAL, chat_id=session_id, user_id='cron-user', profile='default')
    owner.sessions[session_id] = SimpleNamespace(
        source=source, review=None, event_stream=SimpleNamespace(lock=threading.RLock()), subscribers=())
    adapter = LocalSessionAdapter(owner)
    adapter.register_source(source)
    adapter.policies[session_id] = SimpleNamespace(source='cron', request_json=json.dumps(
        {'cron_job': {'id': 'job-1'}, 'extra_prompt': '', 'request_id': 'cron:job-1:fire-1'}))
    owner.runner._adapter_for_source = lambda _source: adapter
    monkeypatch.setattr(scheduler, 'run_job', lambda *a, **k: pytest.fail('a typed turn must not run the job'))
    # The person's message, not the fire's own admission.
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='desktop-user', session_id=session_id,
                        request_id='typed-1', payload={'text': 'run this again now'})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=session_id)

    response, outcome = await SessionAuthority._execute_claimed(
        owner, SessionRef('default', session_id), owner.sessions[session_id], row)
    return row, response, outcome, owner.pending_results[row['admission_id']]


@pytest.mark.asyncio
async def test_typed_turn_in_cron_session_settles_with_read_only_surface(owner, monkeypatch):
    from gateway.session_results import prepare_result
    from gateway.session_settlement_recovery import completion_payload

    row, response, outcome, captured = await _typed_turn_in_cron_session(owner, monkeypatch)
    settled_outcome, _stored = prepare_result(row, response, outcome, captured)
    assert settled_outcome == 'failed'
    complete = completion_payload(row, {'outcome': settled_outcome}, response, captured)
    assert complete['status'] == 'error'
    assert complete['error_surface'] == _SURFACE
    assert 'scheduled job' in complete['text'] and 'Start a new chat' in complete['text']


@pytest.mark.asyncio
async def test_cron_refusal_is_in_the_reopened_transcript(owner, monkeypatch, tmp_path):
    """Settlement commits the refusal as a failed-turn boundary carrying its error card, in the same
    write: a reopened chat redraws the card from history, not from the admission result."""
    from gateway.session_results import finish_result
    from hermes_state import SessionDB

    row, response, outcome, captured = await _typed_turn_in_cron_session(owner, monkeypatch)
    settled, _response = finish_result(owner.db, epoch=owner.epoch, row=row, response=response,
                                       outcome=outcome, result=captured)
    assert settled['outcome'] == 'failed'

    reopened = SessionDB(tmp_path / 'state.db')
    try:
        history = reopened.get_messages(row['target_session_id'])
    finally:
        reopened.close()
    assert len(history) == 3, 'the refusal is the turn after the job\'s own exchange'
    tail = history[-1]
    assert (tail['role'], tail['display_kind']) == ('assistant', 'failed_turn')
    assert 'scheduled job' in tail['content']
    metadata = tail['display_metadata']
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    assert metadata == {'error': tail['content'], 'error_surface': _SURFACE}
