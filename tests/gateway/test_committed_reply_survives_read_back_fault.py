"""A transient storage fault on the recovery read-back must not drop a committed reply.

Regression for #148993: ``settle_and_deliver``'s recovery branch re-reads the committed
result with a single unguarded ``admission_result`` call. A transient
``sqlite3.OperationalError`` (e.g. "database is locked") on that read left ``settled`` as
None, so the ``finally`` block dropped the producer turn and the exception left before
``deliver_settled`` / ``_resolve_settled_waiter`` — a terminal reply was lost and a native
waiter stayed registered with no result.

The fix reuses ``recover_failed_settlement``'s retry pattern (``while True: try: ...
except (OSError, sqlite3.Error): await asyncio.sleep(1)``) for the read-back, and moves
the read out of ``event_stream.lock`` (whose contract says it never spans storage I/O).
"""
import asyncio
import sqlite3

import pytest

from gateway.session_contract import Submission
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


@pytest.mark.asyncio
async def test_committed_reply_survives_a_transient_read_back_fault(tmp_path, monkeypatch):
    """One ``sqlite3.OperationalError`` on the recovery read-back is retried; the committed
    outcome stays terminal and is published, not dropped."""
    from gateway import session_results
    from gateway import session_settlement_recovery as recovery

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0.01,), raising=False)
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {
            'result': {'final_response': 'done', 'completed': True}, 'usage': {}}
        return 'done'

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)

    # Settlement: commit the result each call, then report failure (do NOT restore finish_result,
    # so settle_with_retry exhausts its delays and falls into the recovery branch where the
    # read-back lives; _mark_unknown then sees the committed terminal status and returns it).
    original_finish = session_results.finish_result

    def fail_after_commit(*args, **kwargs):
        original_finish(*args, **kwargs)
        raise OSError('commit response lost')

    monkeypatch.setattr(session_results, 'finish_result', fail_after_commit)

    # Recovery read-back: fail once with a transient locked-read error, then succeed.
    # _read_committed_result imports admission_result from gateway.session_results, so
    # patching the module attribute makes the retry see the flaky version.
    original_admission_result = session_results.admission_result
    read_attempts = []

    def flaky_admission_result(db, admission_id):
        read_attempts.append(admission_id)
        if len(read_attempts) == 1:
            raise sqlite3.OperationalError('database is locked')
        return original_admission_result(db, admission_id)

    monkeypatch.setattr(session_results, 'admission_result', flaky_admission_result)

    with db:
        receipt = await authority.submit(ACTOR, Submission('committed', REF, {'text': 'work'}, 'queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 10)

    row = get_session_admission(db, admission_id=receipt.admission_id)
    assert (row['status'], row['outcome']) == ('terminal', 'completed'), (
        'the committed outcome stayed terminal; the read-back fault did not turn it unknown')
    assert len(read_attempts) >= 2, 'the read-back retried after the transient fault'
    complete = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
    assert complete, 'the committed reply was published despite the read-back fault'
    assert complete[0]['text'] == 'done'
    assert receipt.admission_id not in authority.pending_results
