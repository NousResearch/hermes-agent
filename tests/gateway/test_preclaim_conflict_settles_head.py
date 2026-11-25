"""A preclaim ``admission_conflict`` settles that head instead of pausing its FIFO for good."""
import asyncio

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.core.errors import StoreContention
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
async def test_conflicted_head_settles_with_its_refusal_and_the_successor_runs(tmp_path, monkeypatch):
    from gateway import session_api_turn, session_finite
    from gateway.config import Platform

    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    executed = []

    def check_api_turn(authority, ref, payload):
        if payload['text'] == 'stale':
            raise AuthorityRefusal('admission_conflict')  # its route moved under it, on every drain
    monkeypatch.setattr(session_api_turn, 'check_api_turn', check_api_turn)

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        authority._schedule = lambda ref: None
        head = await _submit(authority, 'stale')
        await _submit(authority, 'follower')
        waiter = authority.waiters.setdefault(head.admission_id, asyncio.get_running_loop().create_future())
        del authority._schedule
        authority._schedule(REF)
        await asyncio.wait_for(authority.sessions['s'].task, 5)

        assert executed == ['follower']
        statuses = {r.request_id: (r.status, r.outcome)
                    for r in authority.persistence.admissions.list(session_id='s', pending_only=False)}
        assert statuses == {'stale': ('terminal', 'cancelled'), 'follower': ('terminal', 'completed')}
        assert waiter.exception().reason == 'admission_conflict'


@pytest.mark.asyncio
async def test_refusal_write_storage_error_retries_and_still_settles_that_head(tmp_path, monkeypatch):
    from gateway import session_api_turn, session_authority, session_finite
    from gateway.config import Platform

    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    monkeypatch.setattr(session_authority, '_CLAIM_RETRY_MIN_S', 0.0)
    executed, cancels = [], []

    def check_api_turn(authority, ref, payload):
        if payload['text'] == 'stale':
            raise AuthorityRefusal('admission_conflict')
    monkeypatch.setattr(session_api_turn, 'check_api_turn', check_api_turn)

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    cancel = authority.persistence.admissions.cancel

    def locked_once(**kwargs):
        cancels.append(kwargs['admission_id'])
        if len(cancels) == 1:
            raise StoreContention('database is locked')
        return cancel(**kwargs)
    monkeypatch.setattr(authority.persistence.admissions, 'cancel', locked_once)

    with db:
        authority._schedule = lambda ref: None
        head = await _submit(authority, 'stale')
        await _submit(authority, 'follower')
        waiter = authority.waiters.setdefault(head.admission_id, asyncio.get_running_loop().create_future())
        del authority._schedule
        authority._schedule(REF)
        await asyncio.wait_for(authority.sessions['s'].task, 5)

        assert cancels == [head.admission_id, head.admission_id]
        assert executed == ['follower']
        statuses = {r.request_id: (r.status, r.outcome)
                    for r in authority.persistence.admissions.list(session_id='s', pending_only=False)}
        assert statuses == {'stale': ('terminal', 'cancelled'), 'follower': ('terminal', 'completed')}
        assert waiter.exception().reason == 'admission_conflict'


@pytest.mark.asyncio
async def test_storage_error_after_the_refusal_commits_still_publishes_its_completion_once(tmp_path, monkeypatch):
    from gateway import session_api_turn, session_authority, session_finite, session_queued_controls
    from gateway.config import Platform

    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    monkeypatch.setattr(session_authority, '_CLAIM_RETRY_MIN_S', 0.0)
    executed, frames, committed = [], [], []

    def check_api_turn(authority, ref, payload):
        if payload['text'] == 'stale':
            raise AuthorityRefusal('admission_conflict')
    monkeypatch.setattr(session_api_turn, 'check_api_turn', check_api_turn)

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    cancel, rows = authority.persistence.admissions.cancel, session_queued_controls.pending_rows

    def cancel_and_note(**kwargs):
        row = cancel(**kwargs)
        committed.append(row.status)
        return row

    def locked_once_after_the_commit(*args, **kwargs):
        if committed == ['terminal']:
            committed.append('locked')
            raise StoreContention('database is locked')  # inside the settlement's pending read
        return rows(*args, **kwargs)
    monkeypatch.setattr(authority.persistence.admissions, 'cancel', cancel_and_note)
    monkeypatch.setattr(session_queued_controls, 'pending_rows', locked_once_after_the_commit)

    with db:
        authority._schedule = lambda ref: None
        head = await _submit(authority, 'stale')
        await _submit(authority, 'follower')
        authority.sessions['s'].event_stream.observers.add(frames.append)
        del authority._schedule
        authority._schedule(REF)
        await asyncio.wait_for(authority.sessions['s'].task, 5)

        assert 'locked' in committed
        assert executed == ['follower']
        completions = [f['params'] for f in frames if f['params']['type'] == 'message.complete'
                       and f['params']['admission_id'] == head.admission_id]
        assert len(completions) == 1, frames
        assert completions[0]['payload']['outcome'] == 'cancelled'
        assert head.admission_id not in authority.cancel_obligations
