"""An empty preflight must not claim a row that lands before its own transaction.

The drain that read an empty FIFO validated nothing: a row admitted after that read but
before the claim's write commits has had no preclaim checks of its own, so that pass must
claim nothing and let the drain re-read it — the new head earns its own authorization.
"""
import asyncio
from types import SimpleNamespace
from contextlib import closing

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt
import gateway.session_authority as sa
from gateway.session_contract import SessionRef
from gateway.config import Platform


@pytest.mark.asyncio
async def test_a_head_landing_before_the_claim_transaction_is_not_claimed_by_it(tmp_path, monkeypatch):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', 'cli')
        epoch = rt.begin_runtime_epoch(db, instance_id='one')
        authority = sa.SessionAuthority(SimpleNamespace(_draining=False, _persist_active_agents=None),
                                        profile_id='p', instance_id='one', db=db, epoch=epoch)
        live = SimpleNamespace(claim_gate=asyncio.Lock(), mutation_lock=asyncio.Lock(), rescan=False,
                               source=SimpleNamespace(platform=Platform.API_SERVER), route='r', detached=False)
        authority._stamp_claim = lambda ref, live, row: row
        authority._classic_turn_holds = lambda route: False
        original = rt.claim_session_input

        def racing(db, **kw):
            # The row commits while the claim parks for the writer, BEFORE its transaction reads the FIFO.
            rt.admit_session_input(db, epoch=epoch, principal_id='u', session_id='s',
                                   request_id='late', payload={'text': 'late'})
            return original(db, **kw)

        monkeypatch.setattr(rt, 'claim_session_input', racing)
        result, first = await authority._claim_next(SessionRef('p', 's'), live)
        assert first is None
        assert result is False, 'a new head must earn its own preclaim checks'
        assert rt.list_session_admissions(db, session_id='s')[0]['status'] == 'queued'


@pytest.mark.asyncio
async def test_a_late_revoked_head_earns_its_own_preclaim_authorization(tmp_path, monkeypatch):
    from tests.gateway.test_session_authority_cancel_settlement import _authority, REF
    from gateway import session_api_turn, session_finite
    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    original = rt.claim_session_input
    inserted = []

    def racing(db, **kw):
        if not inserted:
            inserted.append(True)
            rt.admit_session_input(db, epoch=authority.epoch, principal_id='u', session_id='s',
                                   request_id='late', payload={'text': 'revoked'})
        return original(db, **kw)

    monkeypatch.setattr(rt, 'claim_session_input', racing)
    checked, executed = [], []

    def deny(authority, ref, payload):
        checked.append(payload['text'])
        raise rt.RuntimeStoreError('permission_denied')

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'

    monkeypatch.setattr(session_api_turn, 'check_api_turn', deny)
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await authority._drain(REF)
        assert executed == [], 'revoked input bypassed its preclaim authorization'
        assert checked == ['revoked']
