"""Authenticated Route-to-Output recovery and healthy sibling publication."""
import asyncio
from dataclasses import replace
import time

import pytest
from tests.gateway.test_canonical_peer_target_setup import target, invite, invitation  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_retry import tick, pending, clock
from gateway.run import _profile_runtime_scope
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


async def renew(c, room_id='room-one', member_id='writer'):
    from gateway.session_group_renewal import CanonicalPeerRenewal
    with _profile_runtime_scope(c.home, hydrate_secrets=False):
        route = c.service.peer_routes[(room_id, member_id)]
        client = c.service.peer_clients[(room_id, member_id)]
        renewal = CanonicalPeerRenewal(c.service, room_id, member_id, route, client, route.grant)
        result = await asyncio.to_thread(client.refresh_grant, grant=route.grant)
        grant = result['grant']
        probe = await asyncio.to_thread(client.probe, grant=grant)
        catalog = renewal.verify(route.grant, grant, probe)
        return await asyncio.to_thread(renewal.publish, grant, catalog)


@pytest.mark.asyncio
async def test_authenticated_renewal_unblocks_only_exact_retained_member_work(files_target, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        original = PeerOutputCustody.read
        with monkeypatch.context() as denial:
            def refuse(self, *args):
                raise PeerRunsHTTPError('inert denied', status_code=403)
            denial.setattr(PeerOutputCustody, 'read', refuse)
            await tick(c)
        assert pending(c)[0]['blocked'] == 1
        # Unknown/newer/foreign records are negative controls, never recovered as
        # inferred work. A real authenticated renewal must leave each untouched.
        row = pending(c)[0]
        def foreign_rows(conn):
            for room, task_id, generation, member in [
                ('other-room', row['task_id'], row['execution_generation'], 'writer'),
                ('room-one', 'unknown-task', row['execution_generation'], 'writer'),
                ('room-one', row['task_id'], row['execution_generation'] + 1, 'writer'),
                ('room-one', 'other-member-task', row['execution_generation'], 'reader')]:
                copy = dict(row, room_id=room, task_id=task_id, execution_generation=generation, member_id=member)
                conn.execute('INSERT INTO hosted_room_artifact_retries VALUES (' + ','.join('?' for _ in copy) + ')', tuple(copy.values()))
        c.db._execute_write(foreign_rows)
        before, after = await renew(c)
        assert before.grant != after.grant
        rows = pending(c)
        real = next(r for r in rows if r['room_id'] == 'room-one' and r['task_id'] == c.task['identity'].task_id and r['execution_generation'] == c.attempt.execution_generation)
        assert real['blocked'] == 1, 'Route notification is consumed by the next canonical tick, not a foreign writer'
        assert all(r['blocked'] == 1 for r in rows if r != real)
        await tick(c)
        assert len(pending(c)) == 4 and all(r['blocked'] == 1 for r in pending(c))
        assert len([e for e in c.service._events('room-one') if e['kind'] == 'message.member']) == 1
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
async def test_healthy_sibling_publishes_during_output_backoff_without_policy_lock_io(files_target, monkeypatch):
    from gateway import hosted_room_driver as tasks
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        locks = []
        original = PeerOutputCustody.read
        def unavailable(self, *args):
            # A policy update can proceed while a peer call is in flight.
            from concurrent.futures import ThreadPoolExecutor
            def inspect_lock():
                got = c.service._policy_lock.acquire(blocking=False)
                if got:
                    c.service._policy_lock.release()
                return got
            with ThreadPoolExecutor(max_workers=1) as reader:
                locks.append(reader.submit(inspect_lock).result(timeout=2))
            raise PeerRunsHTTPError('inert unavailable', retryable=True, status_code=503)
        monkeypatch.setattr(PeerOutputCustody, 'read', unavailable)
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            await asyncio.to_thread(c.service.send, room_id='room-one', event_id='sibling-request',
                payload=dict(thread_id='healthy-thread', text='@reader Give a text reply.'))
            # Materialize a second already-eligible terminal through the real
            # planner/driver, independently of the first thread's publication.
            from gateway import hosted_room_discussion as discussion
            event = next(e for e in c.service._events('room-one') if e['event_id'] == 'sibling-request')
            decision = discussion.plan_next_task(c.service._room('room-one'), [event],
                local_profiles=c.service.local_profiles(), freeze_input_context=True)
            assert decision.task is not None
            healthy = tasks.admit_task(c.db.db_path, decision.task.identity,
                payload=decision.task.payload, clock=time.time)
            assert healthy['payload']['target_member_id'] == 'reader'
            attempt = tasks.start_task(c.db.db_path, healthy['identity'], c.attempt.lease,
                                      expected_cancel_generation=0, clock=time.time)
            tasks.settle_task(c.db.db_path, attempt, settlement_id='inert-healthy-terminal', status='settled',
                              result={'text': 'Healthy reply.'}, clock=time.time)
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before
        replies = [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
        assert len(replies) == 1 and replies[0]['payload']['member_id'] == 'reader'
        assert locks and all(locks)
        monkeypatch.setattr(PeerOutputCustody, 'read', original)
        now[0] = pending(c)[0]['next_attempt_at']
        await tick(c)
        replies = [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
        assert {e['payload']['member_id'] for e in replies} == {'writer', 'reader'}
        assert len(replies) == 2
        await tick(c)
        assert [e for e in c.service._events('room-one') if e['kind'] == 'message.member'] == replies


@pytest.mark.asyncio
async def test_recreated_canonical_service_reuses_due_and_completion_not_network(files_target, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        c.wire.faults.lost_ack = True
        await tick(c)
        due = pending(c)[0]['next_attempt_at']
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            c.service = CanonicalHostedRoomService(c.authority, asyncio.get_running_loop())
            c.authority.hosted_room_service = c.service
            c.service._artifact_clock = lambda: now[0]
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before
        now[0] = due
        await tick(c)
        assert pending(c) == []
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            c.service = CanonicalHostedRoomService(c.authority, asyncio.get_running_loop())
            c.authority.hosted_room_service = c.service
            c.service._artifact_clock = lambda: now[0]
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
async def test_route_recovery_survives_output_writer_failure(files_target, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        with monkeypatch.context() as denied:
            def fail(self, *args):
                raise PeerRunsHTTPError('inert authorization failure', status_code=403)
            denied.setattr(PeerOutputCustody, 'read', fail)
            await tick(c)
        c.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_unblock BEFORE UPDATE ON hosted_room_artifact_retries BEGIN SELECT RAISE(ABORT,'inert output writer failure'); END"))
        # Successful Route publication must not be mistaken for failed renewal
        # merely because the independently owned Output queue cannot yet write.
        before, after = await renew(c)
        assert before.grant != after.grant
        assert pending(c)[0]['blocked'] == 1
        c.db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_unblock'))
        await tick(c)
        assert pending(c) == []
        assert len(c.executions) == len(c.launched) == 1
