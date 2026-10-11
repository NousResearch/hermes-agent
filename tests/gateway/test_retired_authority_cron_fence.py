"""A retired profile authority refuses cron mutations from a stale connection: its ownership is
released, but until a successor advances the runtime epoch its epoch would still pass the store's
stale-epoch fence, so cron.cancel could cancel an admission the next owner is about to run."""
import asyncio

import pytest

from gateway import run_runtime, session_cron
from gateway.session_contract import Principal, Submission
from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_profile_retire_stops_turns import REF, _Agent, _authority

CRON = Principal('cron-owner', 'owned', frozenset({'session:submit', 'session:control'}), 'cron-ticker')


@pytest.mark.asyncio
async def test_retired_authority_refuses_cron_cancel_before_successor_epoch(tmp_path, monkeypatch):
    db, authority, _execute, _returned = _authority(tmp_path, _Agent())
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda authority: None)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)  # the row stays queued for the next owner
    try:
        with db:
            receipt = await authority.submit(CRON, Submission(request_id='cron:j:1', ref=REF, payload={'text': ''}, intent='queue'))
            assert await run_runtime._retire_profile_authority(authority, timeout=1.0) is True
            with pytest.raises(RuntimeStoreError, match='runtime_retired'):
                await session_cron.operation(authority, 'cancel', {'session_id': 's', 'admission_id': receipt.admission_id})
            [row] = list_session_admissions(db, session_id='s', pending_only=False)
            assert row['status'] == 'queued'
    finally:
        await asyncio.sleep(.05)
        db.close()
