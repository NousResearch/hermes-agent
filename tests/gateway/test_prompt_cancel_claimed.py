"""prompt.cancel on a row the drain claims between the client's queued read and the cancel commit
stops that claim (as API and ACP do) instead of refusing while the withdrawn prompt runs (N113)."""
import pytest

from tests.gateway.test_canonical_stop_hard_cancel import Agent
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


@pytest.mark.asyncio
async def test_prompt_cancel_racing_the_claim_stops_the_claimed_turn(tmp_path, monkeypatch):
    from gateway import session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_state_runtime import claim_session_input

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    agent = Agent()
    authority.runner._cached_agent_for = {'route': agent}.get
    real_cancel = session_authority.cancel_session_input

    def claimed_first(db, **kwargs):
        # The drain's claim commits between the client's queued read and the cancel transaction.
        claim_session_input(db, epoch=authority.epoch, session_id='s')
        return real_cancel(db, **kwargs)
    monkeypatch.setattr(session_authority, 'cancel_session_input', claimed_first)
    with db:
        receipt = await _submit(authority, 'withdrawn')
        conn = AuthorityConnection(authority, object(), {'user_id': 'human'})
        reply = await conn.dispatch({'id': 1, 'method': 'prompt.cancel', 'params': {
            'session_id': REF.session_id, 'admission_id': receipt.admission_id}})
    assert 'error' not in reply, f'prompt.cancel refused a live claim: {reply}'
    assert reply['result']['status'] == 'started'
    assert agent.stops == ['hard'], 'the withdrawn prompt kept running'
