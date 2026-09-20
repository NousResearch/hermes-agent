"""Durable output-only work through the canonical service's manual ticks.

Real Home/target stores, target signature/receipt owners and registered HTTP;
only the socket/model launch boundary is inert. No coordinator or NEW replay.
"""
import asyncio
import time

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from gateway.run import _profile_runtime_scope
from gateway.hosted_room_artifacts import RoomArtifactScope
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


async def tick(c):
    with _profile_runtime_scope(c.home, hydrate_secrets=False):
        await asyncio.to_thread(c.service.prepare_room, c.binding)


def pending(c):
    return [dict(r) for r in c.db._conn.execute('SELECT * FROM hosted_room_artifact_retries')]


def clock(c):
    now = [time.time()]
    c.service._artifact_clock = lambda: now[0]
    return now


async def supersede(c):
    from gateway.session_hosted_attachments import append_user_event
    from gateway import hosted_room_discussion as discussion
    with _profile_runtime_scope(c.home, hydrate_secrets=False):
        room = c.service._room('room-one')
        payload = discussion.validate_user_payload(dict(thread_id='thread-one', text='@writer superseding request'),
            member_ids=[m['member_id'] for m in room['members']])
        append_user_event(c.service, room_id='room-one', event_id='newer-request', payload=payload,
            gateway_id=room['authority_gateway_id'], epoch=room['authority_epoch'])


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['read', 'ack', 'discard'])
async def test_canonical_tick_persists_due_and_exact_completion(files_target, monkeypatch, operation):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        if operation == 'discard':
            await supersede(c)
            c.wire.faults.lost_discard = True
        elif operation == 'ack':
            c.wire.faults.lost_ack = True
        else:
            original = PeerOutputCustody.read
            failures = [True]
            def read(self, *args):
                if failures.pop() if failures else False:
                    raise PeerRunsHTTPError('inert unavailable', retryable=True, status_code=503)
                return original(self, *args)
            monkeypatch.setattr(PeerOutputCustody, 'read', read)
        await tick(c)
        retry, = pending(c)
        assert retry['attempts'] == 1 and retry['blocked'] == 0
        assert retry['next_attempt_at'] == now[0] + 1
        assert retry['task_id'] == c.task['identity'].task_id
        assert retry['execution_generation'] == c.attempt.execution_generation
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before, 'an incidental prepare must not bypass due gating'
        now[0] = retry['next_attempt_at']
        await tick(c)
        assert pending(c) == []
        done, = c.db._conn.execute('SELECT * FROM hosted_room_artifact_completions')
        assert done['task_id'] == retry['task_id']
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before, 'exact completed work must not retransmit'
        events = c.service._events('room-one')
        terminal = 'dterminal:' + c.task['identity'].task_id.removeprefix('dtask:')
        assert len([e for e in events if e['event_id'] == terminal]) == 1
        assert len([e for e in events if e['kind'] == 'message.member']) == (0 if operation == 'discard' else 1)
        scope = RoomArtifactScope.from_mapping(c.stored['result']['artifact_scope'])
        assert c.target.adapter._peer_output_outbox.retirement_complete(scope)
        assert len(c.launched) == len(c.executions) == 1


@pytest.mark.asyncio
async def test_permanent_denial_is_durable_and_never_retries_incidentally(files_target, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        attempts = []
        def denied(self, *args):
            attempts.append(1)
            raise PeerRunsHTTPError('inert grant denial', status_code=403, error_code='room_grant_invalid')
        monkeypatch.setattr(PeerOutputCustody, 'read', denied)
        await tick(c)
        retry, = pending(c)
        assert retry['blocked'] == 1
        now[0] += 1000
        await tick(c)
        assert attempts == [1]
        assert pending(c)[0]['attempts'] == 1
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            state = c.service.status('room-one')
        action, = [a for a in state['pending_actions'] if a['kind'] == 'output_retry']
        assert action['blocked'] is True and action['task_id'] == c.task['identity'].task_id
        assert not [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
        assert len(c.launched) == len(c.executions) == 1
