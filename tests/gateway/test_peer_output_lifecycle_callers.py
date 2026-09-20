"""Native peer Stop/disband caller evidence at the real in-process HTTP seam."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from gateway import hosted_rooms, hosted_room_driver as tasks
from gateway.run import _profile_runtime_scope
from gateway.session_peer_output_custody import PeerOutputCustody
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_canonical_output_lifecycle import connection, dispatch
from tests.gateway.test_peer_output_retry import tick, pending, clock
from tests.tui_gateway.test_hosted_room_peer_backoff_progress import LocalRPC
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.asyncio
@pytest.mark.parametrize('published', [False, True])
async def test_native_stop_uses_retained_peer_disposition(files_target, monkeypatch, published):
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        if published:
            with monkeypatch.context() as m:
                def unavailable(*args, **kwargs):
                    raise PeerRunsHTTPError('inert ACK unavailable', retryable=True, status_code=503)
                m.setattr(PeerOutputCustody, 'acknowledge', unavailable)
                await tick(c)
            now[0] = pending(c)[0]['next_attempt_at']
        client = connection(c.authority, c.service)
        before = len(c.wire.calls)
        try:
            result = await dispatch(client, 'stop', room_id='room-one', cancel_id='peer-stop')
            assert 'result' in result, result
        finally:
            c.service.runtime._thread = None
        operation = '/artifacts/ack' if published else '/artifacts/discard'
        assert any(path.endswith(operation) for method, path in c.wire.calls[before:])
        assert not pending(c)
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            messages = [x for x in c.service._events('room-one') if x['kind'] == 'message.member']
        assert bool(messages) is published
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('missing_service', [False, True])
async def test_pending_peer_cleanup_blocks_before_revoke(files_target, monkeypatch, missing_service):
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        now[0] = c.claims['expires_at'] + 1
        client = connection(c.authority, c.service)
        called = []
        monkeypatch.setattr(c.service, 'revoke_room_routes', lambda *a: called.append(a))
        try:
            stopped = await dispatch(client, 'stop', room_id='room-one', cancel_id='expired-stop')
            assert 'result' in stopped, stopped
            assert pending(c)
            if missing_service:
                c.service.runtime._thread = None
            result = await dispatch(client, 'disband', room_id='room-one')
            assert 'error' in result, result
            assert not called
            with _profile_runtime_scope(c.home, hydrate_secrets=False):
                assert hosted_rooms.room_state(c.db.db_path, room_id='room-one').get('disbanded_at') is None
        finally:
            c.service.runtime._thread = None


@pytest.mark.asyncio
async def test_native_stop_drains_completed_peer_without_fixture_settlement(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch, settle_target=False) as c:
        owner, ref, row = c.launched[0]
        with _profile_runtime_scope(c.target.home, hydrate_secrets=False):
            await owner._drain(ref)
        c.service.runtime._leases['room-one'] = c.attempt.lease
        client = connection(c.authority, c.service)
        try:
            stopped = await dispatch(client, 'stop', room_id='room-one', cancel_id='completion-race')
            assert 'result' in stopped, stopped
            with _profile_runtime_scope(c.home, hydrate_secrets=False):
                current = tasks.get_task(c.db.db_path, c.task['identity'])
                assert current['status'] == 'settled'
                assert not [a for a in c.service.status('room-one')['pending_actions'] if a['kind'] == 'output_cleanup']
            assert any(path.endswith('/artifacts/discard') for method, path in c.wire.calls)
            assert len(c.executions) == 1
        finally:
            c.service.runtime._thread = None


@pytest.mark.asyncio
async def test_pending_peer_cleanup_is_not_age_success(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        def unavailable(*args, **kwargs):
            raise PeerRunsHTTPError('inert ACK unavailable', retryable=True, status_code=503)
        monkeypatch.setattr(PeerOutputCustody, 'acknowledge', unavailable)
        await tick(c)
        assert pending(c)
        now[0] += tasks.ARTIFACT_RETRY_RETENTION_SECONDS + 1
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            assert tasks.prune_published_terminal_tasks(c.db.db_path, room_id='room-one', clock=lambda: now[0], retain=0) == 0
            c.service._prune_output_retry_metadata('room-one')
        assert pending(c) and tasks.get_task(c.db.db_path, c.task['identity'])['status'] == 'settled'


@pytest.mark.asyncio
@pytest.mark.parametrize('held_operation', ['stop', 'status'])
async def test_stop_control_io_does_not_hold_global_policy_lock(files_target, monkeypatch, held_operation):
    from tui_gateway import hosted_room_peer_http
    async with peer_case(files_target, monkeypatch, settle_target=False) as c:
        client = connection(c.authority, c.service)
        original = hosted_room_peer_http._open_roomlink_url
        entered, release = threading.Event(), threading.Event()
        def held(wire, **kwargs):
            if (wire.full_url.endswith('/stop') if held_operation == 'stop' else
                    wire.get_method() == 'GET' and '/v1/runs/' in wire.full_url):
                entered.set()
                assert release.wait(12)
            return original(wire, **kwargs)
        monkeypatch.setattr(hosted_room_peer_http, '_open_roomlink_url', held)
        local = LocalRPC()
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            c.service.authorize_room('alice', 'room-two', create=True)
            c.service.create_room(room_id='room-two', name='Independent', members=[
                dict(member_id='reader', profile='default', handle='reader'),
                {**c.service._room('room-one')['members'][0], 'member_id': 'observer', 'handle': 'observer'}])
            c.service.member_rpcs[('room-two', 'reader', 'default', 'alice', str(c.home))] = local
            binding = next(b for b in c.service.bindings() if b.room_id == 'room-two')
        stopping = asyncio.create_task(dispatch(client, 'stop', room_id='room-one', cancel_id='held-stop'))
        repeated = None
        try:
            assert await asyncio.to_thread(entered.wait, 8), 'actual peer control I/O must be exercised'
            repeated = asyncio.create_task(dispatch(client, 'stop', room_id='room-one', cancel_id='held-stop'))
            def available():
                got = c.service._policy_lock.acquire(blocking=False)
                if got:
                    c.service._policy_lock.release()
                return got
            assert await asyncio.to_thread(available), 'Stop transport retained the global policy lock'
            sent = await asyncio.wait_for(dispatch(client, 'send', room_id='room-two', event_id='healthy',
                payload=dict(thread_id='healthy-thread', text='@reader reply')), 4)
            assert 'result' in sent, sent
            with _profile_runtime_scope(c.home, hydrate_secrets=False):
                await asyncio.to_thread(c.service.runtime._process_room, binding)
            assert len([x for x in local.calls if x[0] == 'submit']) == 1
        finally:
            release.set()
            response = await asyncio.wait_for(stopping, 15)
            if repeated is not None:
                replay = await asyncio.wait_for(repeated, 15)
                assert 'result' in replay, replay
            c.service.runtime._thread = None
        assert 'result' in response, response
        assert tasks.get_task(c.db.db_path, c.task['identity'])['cancel_generation'] == 1
