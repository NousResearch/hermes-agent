"""Settlement's write runs off the owner loop; completion, idle info and waiters keep their order."""
import asyncio
import threading

import pytest

from gateway.session_contract import Submission
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


@pytest.mark.asyncio
async def test_settlement_write_leaves_the_loop_free_and_publishes_in_order(tmp_path, monkeypatch):
    from gateway import session_results
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    loop_ran, events, served_during_write = threading.Event(), [], []
    authority.sessions['s'].event_stream.observers.add(
        lambda frame: events.append((frame['params']['type'], frame['params']['payload'].get('running'))))

    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'done'}, 'usage': {}}
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    settle = session_results.settle_session_input

    def slow_settle(*args, **kwargs):
        # A contended writer: the owner loop must keep serving other work meanwhile.
        served_during_write.append(loop_ran.wait(2))
        return settle(*args, **kwargs)
    monkeypatch.setattr(session_results, 'settle_session_input', slow_settle)

    async def other_session_work():
        await asyncio.sleep(0.01)
        loop_ran.set()

    with db:
        receipt = await authority.submit(ACTOR, Submission('work', REF, {'text': 'work'}, 'queue'))
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())

        def resolved(_):
            events.append(('waiter', get_session_admission(db, admission_id=receipt.admission_id)['status']))
        waiter.add_done_callback(resolved)
        events.clear()
        other = asyncio.create_task(other_session_work())
        await asyncio.wait_for(authority._drain(REF), 5)
        await other
        await asyncio.sleep(0)
    assert served_during_write == [True], 'the settlement write blocked the owner loop'
    assert events[-3:] == [('message.complete', None), ('session.info', False), ('waiter', 'terminal')], events
