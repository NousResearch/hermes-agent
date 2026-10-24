"""A session's durable FIFO holds at most main's busy-queue cap of queued inputs (N62), paused or not (B2-07)."""
import pytest

from hermes_state_runtime import ADMISSION_QUEUE_MAX, RuntimeStoreError, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit


@pytest.mark.asyncio
async def test_queued_inputs_past_the_cap_are_refused_queue_full_and_retries_replay(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)  # one turn runs forever: nothing drains
    with db:
        for n in range(ADMISSION_QUEUE_MAX):
            await _submit(authority, f'r{n}')
        with pytest.raises(RuntimeStoreError) as refused:
            await _submit(authority, 'overflow')
        assert refused.value.reason == 'queue_full'
        assert len(list_session_admissions(db, session_id='s')) == ADMISSION_QUEUE_MAX
        # An exact retry of accepted input is a replay, never a refusal.
        assert (await _submit(authority, 'r0')).status == 'queued'

        # Paused behind an unknown turn (B2-07): main's cap still holds, so the backlog that runs after
        # Discard stays bounded. The paused head is the in-flight turn (main never counted it either).
        db._execute_write(lambda conn: conn.execute(
            "UPDATE session_admissions SET status='unknown' WHERE request_id='r0'"))
        assert (await _submit(authority, 'refill')).status == 'queued'
        for request_id in ('paused-overflow', 'paused-overflow-2'):
            with pytest.raises(RuntimeStoreError) as paused:
                await _submit(authority, request_id)
            assert paused.value.reason == 'queue_full'
        assert len(list_session_admissions(db, session_id='s')) == ADMISSION_QUEUE_MAX + 1
