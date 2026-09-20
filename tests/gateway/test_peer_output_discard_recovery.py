"""Durable fences survive SQL, disk and lost-reply failures; no listener."""
import asyncio

import os


import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_discard import custody, discard
from tests.gateway.test_peer_output_fences import read, unretired
from gateway.run import _profile_runtime_scope
from gateway.session_results import admission_result
from gateway.hosted_room_artifacts import RoomArtifactError
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.asyncio
async def test_lost_discard_requires_replay_and_never_reopens_admission(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch) as c:
        source = custody(c)
        c.wire.faults.lost_discard = True
        with pytest.raises(PeerRunsHTTPError) as error:
            await discard(c, source)
        assert error.value.retryable
        assert c.target.adapter._peer_output_outbox.retirement_complete(source[0])
        assert await discard(c) == 1
        assert len(c.launched) == len(c.executions) == 1
        assert c.target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1


@pytest.mark.asyncio
async def test_sql_retirement_rollback_then_disk_failure_keeps_cleanup_intent(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch) as c:
        scope = custody(c)[0]
        c.target.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_retirement BEFORE UPDATE ON hosted_room_output_artifacts BEGIN SELECT RAISE(ABORT, 'inert SQL fault'); END"))
        with pytest.raises(PeerRunsHTTPError) as initial_error:
            await discard(c)
        assert initial_error.value.status_code == 503 and initial_error.value.retryable
        unretired(c)
        assert 'peer_output_discard' not in admission_result(c.target.db, c.row['admission_id'])
        fence = c.target.db._conn.execute('SELECT retired_generation FROM hosted_room_output_generation_fences WHERE lineage_identity=?', (scope.lineage_json,)).fetchone()
        assert fence[0] < scope.execution_generation
        c.target.db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_retirement'))
        original = os.unlink
        def fail(path, *args, **kwargs):
            if str(path).startswith('blob_') and kwargs.get('dir_fd') is not None:
                raise OSError('inert disk failure')
            return original(path, *args, **kwargs)
        with monkeypatch.context() as fault:
            fault.setattr(os, 'unlink', fail)
            with pytest.raises(PeerRunsHTTPError) as error:
                await discard(c)
            assert error.value.status_code == 503 and error.value.retryable
        row = c.target.db._conn.execute('SELECT cleanup_required_at, acknowledged_at FROM hosted_room_output_artifacts').fetchone()
        assert all(x is not None for x in row)
        assert not c.target.adapter._peer_output_outbox.retirement_complete(scope)
        assert admission_result(c.target.db, c.row['admission_id'])['peer_output_discard']['receipt'] == dict(discarded=True, removed=1)
        with pytest.raises(PeerRunsHTTPError):
            await read(c)
        with pytest.raises(RoomArtifactError):
            c.target.adapter._peer_output_outbox.put_bytes(scope=scope, source_name='late.txt', data=b'late')
        # Physical unlink succeeds but SQL cleanup commit fails: fence and exact
        # obligation remain, and another replay can safely see missing bytes.
        c.target.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_cleanup BEFORE DELETE ON hosted_room_output_artifacts BEGIN SELECT RAISE(ABORT, 'inert cleanup SQL fault'); END"))
        with pytest.raises(PeerRunsHTTPError) as error:
            await discard(c)
        assert error.value.status_code == 503
        c.target.db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_cleanup'))
        assert await discard(c) == 1
        assert c.target.adapter._peer_output_outbox.retirement_complete(scope)


@pytest.mark.asyncio
async def test_real_silent_publication_waits_for_discard_and_replays_lost_reply(files_target, monkeypatch):
    from gateway.session_hosted_attachments import append_user_event
    from gateway import hosted_room_discussion as discussion
    async with peer_case(files_target, monkeypatch) as c:
        scope = custody(c)[0]
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            room = c.service._room('room-one')
            payload = discussion.validate_user_payload(dict(thread_id='thread-one', text='@writer superseding request'),
                member_ids=[m['member_id'] for m in room['members']])
            append_user_event(c.service, room_id='room-one', event_id='newer-request', payload=payload,
                gateway_id=room['authority_gateway_id'], epoch=room['authority_epoch'])
            c.wire.faults.lost_discard = True
            with pytest.raises(PeerRunsHTTPError):
                await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
            assert not [e for e in c.service._events('room-one') if e['event_id'] == 'dterminal:' + scope.task_id.removeprefix('dtask:')]
            due = c.db._conn.execute('SELECT next_attempt_at FROM hosted_room_artifact_retries').fetchone()[0]
            c.service._artifact_clock = lambda: due
            assert await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
            events = [e for e in c.service._events('room-one') if e['event_id'] == 'dterminal:' + scope.task_id.removeprefix('dtask:')]
            assert len(events) == 1 and events[0]['kind'] == 'turn.cancelled'
            assert not [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
        assert c.target.adapter._peer_output_outbox.retirement_complete(scope)
        assert len(c.executions) == len(c.launched) == 1
