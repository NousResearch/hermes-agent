"""Canonical independent-discussion admission behind exact retained Output."""
import asyncio

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_retry import tick, pending, clock
from tests.tui_gateway.test_hosted_room_peer_backoff_progress import LocalRPC
from gateway import hosted_room_driver as tasks
from gateway.run import _profile_runtime_scope
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.asyncio
@pytest.mark.parametrize('hold', ['transient', 'permanent', 'same-thread', 'unknown'])
async def test_real_planner_admits_only_independent_known_settled_output_followers(files_target, monkeypatch, hold):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        original = tasks.get_task(c.db.db_path, c.task['identity'])
        def unavailable(self, *args):
            raise PeerRunsHTTPError('held import', retryable=hold != 'permanent',
                                    status_code=403 if hold == 'permanent' else 503)
        monkeypatch.setattr(PeerOutputCustody, 'read', unavailable)
        await tick(c)
        obligation, = pending(c)
        local = LocalRPC()
        c.service.member_rpcs[('room-one', 'reader', 'default', 'alice', str(c.home))] = local
        c.service.runtime._leases['room-one'] = c.attempt.lease
        if hold == 'unknown':
            # Explicit negative: a queue row is never settlement/admission authority.
            c.db._execute_write(lambda conn: conn.execute("UPDATE hosted_room_driver_tasks SET status='indeterminate' WHERE task_id=?", (c.task['identity'].task_id,)))
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            await asyncio.to_thread(c.service.send, room_id='room-one', event_id='independent-request',
                payload=dict(thread_id='thread-one' if hold == 'same-thread' else 'independent-thread',
                             text='@reader Give an independent text reply.'))
            # No isolated planner, manually inserted sibling, or manual settlement.
            # Unknown's existing recovery path is deliberately not exercised here.
            if hold != 'unknown':
                await asyncio.to_thread(c.service.runtime._process_room, c.binding)
                await asyncio.to_thread(c.service.prepare_room, c.binding)
        submits = [x for x in local.calls if x[0] == 'submit']
        healthy = [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
        if hold in {'transient', 'permanent'}:
            ordinary = c.service.policy_checkpoint.snapshot(room_id='room-one',
                latest_seq=c.service._room('room-one')['latest_seq'])
            assert any(e['event_id'] == 'request-one' for e in ordinary.events), 'generic FIFO must still select oldest'
            assert len(submits) == 1, 'normal oldest-discussion selection starved independent work'
            assert len(healthy) == 1 and healthy[0]['payload']['member_id'] == 'reader'
            selected = [t for t in tasks.list_tasks(c.db.db_path, room_id='room-one') if t['identity'] != c.task['identity']]
            assert len(selected) == 1 and selected[0]['status'] == 'settled'
        else:
            assert not submits and not healthy, 'same-thread dependency/unknown work must not gain NEW admission'
            assert not tasks.list_tasks(c.db.db_path, room_id='room-one', status='queued')
        retained = tasks.get_task(c.db.db_path, c.task['identity'])
        assert retained['payload'] == original['payload'] and retained['result'] == original['result']
        assert retained['execution_generation'] == original['execution_generation']
        assert pending(c)[0]['task_id'] == obligation['task_id']
        assert pending(c)[0]['attempts'] == obligation['attempts']
        assert len(c.executions) == len(c.launched) == 1
