"""Real retained-output regressions; finite private stores and inert transport only."""
import asyncio
import json
import sqlite3

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_retry import tick, pending, clock
from tests.gateway.test_peer_output_retry_recovery import renew
from gateway.run import _profile_runtime_scope
from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactScope
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['missing-message', 'changed-terminal', 'missing-artifacts'])
async def test_pending_disposition_never_becomes_discard_or_text(files_target, monkeypatch, drift):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        with monkeypatch.context() as failure:
            def fail(self, *args, **kwargs):
                raise PeerRunsHTTPError('before target commitment', retryable=True, status_code=503)
            failure.setattr(PeerOutputCustody, 'read' if drift == 'missing-artifacts' else 'acknowledge', fail)
            await tick(c)  # includes the real policy synchronization
        retry, = pending(c)
        assert retry['operation'] == ('publish' if drift == 'missing-artifacts' else 'ack')
        if drift == 'missing-artifacts':
            result = dict(c.stored['result'])
            result.pop('artifacts')
            c.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET result_json=?', (json.dumps(result),)))
        elif drift == 'missing-message':
            c.db._execute_write(lambda conn: conn.execute("DELETE FROM hosted_room_events WHERE kind='message.member'"))
        else:
            def change(conn):
                event = conn.execute("SELECT event_id,payload_json FROM hosted_room_events WHERE kind='turn.settled'").fetchone()
                payload = json.loads(event['payload_json'])
                payload['message_event_id'] = 'foreign-message'
                conn.execute('UPDATE hosted_room_events SET payload_json=? WHERE event_id=?', (json.dumps(payload), event['event_id']))
            c.db._execute_write(change)
        now[0] = retry['next_attempt_at']
        before = list(c.wire.calls)
        try:
            await tick(c)
        except RoomArtifactError:
            pass
        assert c.wire.calls == before, 'missing Home evidence must not authorize ACK or destructive discard'
        row, = pending(c)
        assert row['operation'] == retry['operation'] and row['blocked'] == 1
        assert not c.db._conn.execute('SELECT 1 FROM hosted_room_artifact_completions').fetchall()
        scope = RoomArtifactScope.from_mapping(c.stored['result']['artifact_scope'])
        assert not c.target.adapter._peer_output_outbox.retirement_complete(scope)
        from tests.gateway.test_peer_output_fences import unretired
        unretired(c)  # exact producer bytes and unacknowledged rows, not absence-as-proof
        if drift == 'missing-artifacts':
            assert not [e for e in c.service._events('room-one') if e['kind'] in {'message.member', 'turn.settled'}]
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
async def test_failure_backoff_starts_at_transport_completion(files_target, monkeypatch):
    from tui_gateway import hosted_room_peer_artifacts
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        with monkeypatch.context() as failure:
            def slow(request, **kwargs):
                assert request.get_method() == 'GET' and '/artifacts/' in request.full_url
                assert kwargs['reject_redirects'] is True
                now[0] += 10
                raise TimeoutError('slow artifact socket boundary')
            failure.setattr(hosted_room_peer_artifacts, '_open_roomlink_url', slow)
            await tick(c)
        retry, = pending(c)
        assert retry['next_attempt_at'] == now[0] + 1
        before = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == before
        now[0] = retry['next_attempt_at']
        await tick(c)
        assert pending(c) == []
        due_calls = c.wire.calls[len(before):]
        assert len([x for x in due_calls if x[0] == 'GET' and '/artifacts/' in x[1]]) == 1
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('queue_fault', [False, True])
async def test_authenticated_renewal_survives_tracked_health_changes(files_target, monkeypatch, queue_fault):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        with monkeypatch.context() as failure:
            def denied(self, *args):
                raise PeerRunsHTTPError('denied', status_code=403)
            failure.setattr(PeerOutputCustody, 'read', denied)
            await tick(c)
        before, after = await renew(c)
        assert before.grant != after.grant
        if queue_fault:
            c.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_unblock BEFORE UPDATE ON hosted_room_artifact_retries BEGIN SELECT RAISE(ABORT,'inert queue fault'); END"))
            with pytest.raises(sqlite3.Error):
                await tick(c)
            c.db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_unblock'))
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            tracked = c.service._tracked_peer_client('room-one', 'writer', c.client)
            params = dict(room_id='room-one', profile='default', session_id=c.rpc._session_id, grant=after.grant)
            with monkeypatch.context() as unavailable:
                def fail_status(**kwargs):
                    raise PeerRunsHTTPError('tracked unavailable', status_code=503, retryable=True, not_admitted=True)
                unavailable.setattr(c.client, 'status', fail_status)
                with pytest.raises(PeerRunsHTTPError):
                    await asyncio.to_thread(tracked.status, **params)
            assert c.service._peer_route_status[('room-one', 'writer')] == 'unavailable'
            await tick(c)  # health is not positive recovery authority
            assert pending(c)[0]['blocked'] == 1
            await asyncio.to_thread(tracked.status, **params)
            assert c.service._peer_route_status[('room-one', 'writer')] == 'ready'
        await tick(c)
        assert pending(c) == []
        calls = list(c.wire.calls)
        await tick(c)
        assert c.wire.calls == calls
        assert len([e for e in c.service._events('room-one') if e['kind'] == 'message.member']) == 1
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['grant', 'endpoint', 'policy', 'member', 'cancel', 'legacy-marker'])
async def test_authenticated_notification_cannot_unblock_foreign_transition(files_target, monkeypatch, drift):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        with monkeypatch.context() as failure:
            def denied(self, *args):
                raise PeerRunsHTTPError('denied', status_code=403)
            failure.setattr(PeerOutputCustody, 'read', denied)
            await tick(c)
        await renew(c)
        if drift == 'legacy-marker':
            sql, params = "UPDATE state_meta SET value=? WHERE key LIKE 'gateway.hosted.route.recovered.v1:%'", ('f' * 64,)
        elif drift == 'grant':
            sql, params = 'UPDATE hosted_room_links SET grant=?', (c.issued['grant'],)
        elif drift == 'endpoint':
            sql, params = "UPDATE hosted_room_links SET target_url='https://foreign.invalid'", ()
        elif drift == 'policy':
            sql, params = "UPDATE hosted_room_links SET catalog_json='{}'", ()
        elif drift == 'member':
            sql, params = "UPDATE hosted_room_links SET member_id='reader'", ()
        else:
            sql, params = 'UPDATE hosted_room_driver_tasks SET cancel_generation=cancel_generation+1', ()
        c.db._execute_write(lambda conn: conn.execute(sql, params))
        calls = list(c.wire.calls)
        try:
            await tick(c)
        except (RoomArtifactError, ValueError):
            if drift == 'legacy-marker':
                raise  # old marker must remain inert, not poison ordinary prepare
        assert c.wire.calls == calls
        assert pending(c)[0]['blocked'] == 1
        assert not c.db._conn.execute('SELECT 1 FROM hosted_room_artifact_completions').fetchall()
        assert len(c.executions) == len(c.launched) == 1
