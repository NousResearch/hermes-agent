"""Adversarial real peer-output boundaries; all Runs come from the canonical drain."""
import asyncio
import json
from pathlib import Path

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from gateway.run import _profile_runtime_scope
from gateway.hosted_room_artifacts import RoomArtifactScope, RoomArtifactOutbox, RoomArtifactError
from gateway.session_results import _RESULT_PREFIX
from tui_gateway.hosted_room_peer_artifacts import read_artifact, acknowledge_artifacts
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


def commitment(case):
    manifest = case.stored['result']['artifacts']
    return dict(artifact_ids=[x['artifact_id'] for x in manifest['items']],
                manifest_digest=manifest['manifest_digest'],
                message_event_id='dmessage:' + case.task['identity'].task_id.removeprefix('dtask:'))


async def ack(case, *, grant=None, run_id=None, body=None):
    return await asyncio.to_thread(acknowledge_artifacts, case.client,
        run_id=run_id or case.accepted['run_id'], grant=grant or case.issued['grant'], **(body or commitment(case)))


async def read(case, *, grant=None, run_id=None, artifact_id=None):
    return await asyncio.to_thread(read_artifact, case.client,
        run_id=run_id or case.accepted['run_id'], grant=grant or case.issued['grant'],
        artifact_id=artifact_id or commitment(case)['artifact_ids'][0])


def unretired(case):
    rows = list(case.target.db._conn.execute('SELECT acknowledged_at,blob_reclaimed_at FROM hosted_room_output_artifacts'))
    assert rows and all(tuple(row) == (None, None) for row in rows)
    assert [p.read_bytes() for p in case.target.adapter._peer_output_outbox.blob_root.iterdir()] == [case.output.read_bytes()]


@pytest.mark.asyncio
async def test_lost_ack_then_expired_tombstone_replays_same_event_and_bytes(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch) as c:
        scope = RoomArtifactScope.from_mapping(c.stored['result']['artifact_scope'])
        c.wire.faults.lost_ack = True
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            with pytest.raises(PeerRunsHTTPError):
                await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
            events = [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
            assert len(events) == 1
            await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
            assert [e for e in c.service._events('room-one') if e['kind'] == 'message.member'] == events
        assert c.target.adapter._peer_output_outbox.retirement_complete(scope)
        expiry = c.target.db._conn.execute('SELECT receipt_expires_at FROM hosted_room_output_artifacts').fetchone()[0]
        assert c.target.adapter._peer_output_outbox.prune_acknowledged_receipts(now=expiry + 1) == 1
        assert c.target.db._conn.execute('SELECT count(*) FROM hosted_room_output_artifacts').fetchone()[0] == 0
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
            assert [e for e in c.service._events('room-one') if e['kind'] == 'message.member'] == events
        assert len(c.executions) == len(c.outputs) == len(c.launched) == 1
        assert [Path(x['path']).read_bytes() for x in c.row['payload']['api_turn_v1']['settings']['room_input_media']['media']] == c.raw
        # Positive retirement is NOT absence: erase only the exact durable ACK
        # commitment as a negative corruption witness. No successful ACK follows.
        key = _RESULT_PREFIX + c.row['admission_id']
        saved = json.loads(c.target.db._conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
        saved.pop('peer_output_ack')
        c.target.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(saved), key)))
        with pytest.raises(PeerRunsHTTPError):
            await ack(c)


@pytest.mark.asyncio
@pytest.mark.parametrize('rights', [('approve', 'dispatch', 'status', 'stop'),
    ('approve', 'dispatch', 'status', 'stop', 'attachment.stage'), ('status',), ('attachment.stage',), ('artifact.read',)])
async def test_old_partial_and_status_grants_never_retire_output(files_target, monkeypatch, rights):
    from gateway.hosted_room_peer import issue_room_grant
    async with peer_case(files_target, monkeypatch) as c:
        assert await read(c) == c.output.read_bytes()
        claims = c.claims
        signer = {k: claims[k] for k in ('grant_id', 'room_id', 'home_install_id', 'authority_gateway_id',
            'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'issued_at')}
        token = issue_room_grant(c.target.adapter._room_grant_secret(), **signer, permissions=rights,
            ttl_seconds=claims['expires_at'] - claims['issued_at'], status_expires_at=claims['status_expires_at'])
        if 'artifact.read' not in rights:
            with pytest.raises(PeerRunsHTTPError):
                await read(c, grant=token)
        with pytest.raises(PeerRunsHTTPError):
            await ack(c, grant=token)
        unretired(c)


@pytest.mark.asyncio
async def test_artifact_expiry_denies_even_while_status_is_live(files_target, monkeypatch):
    from gateway.hosted_room_peer import decode_room_grant
    async with peer_case(files_target, monkeypatch) as c:
        assert await read(c) == c.output.read_bytes()
        with monkeypatch.context() as clock:
            clock.setattr('gateway.hosted_room_peer.time.time', lambda: c.claims['expires_at'] + 1)
            decode_room_grant(c.target.adapter._room_grant_secret(), c.issued['grant'], permission='status')
            with pytest.raises(PeerRunsHTTPError):
                await read(c)
            with pytest.raises(PeerRunsHTTPError):
                await ack(c)
        unretired(c)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['ids', 'digest', 'event', 'run', 'artifact', 'nonterminal', 'generation', 'failed', 'result'])
async def test_wrong_exact_commitment_or_run_refuses(files_target, monkeypatch, change):
    async with peer_case(files_target, monkeypatch) as c:
        assert await read(c) == c.output.read_bytes()
        body = commitment(c)
        run_id = c.accepted['run_id']
        if change == 'ids':
            body['artifact_ids'] = ['rart_' + '0' * 32]
        elif change == 'digest':
            body['manifest_digest'] = '0' * 64
        elif change == 'event':
            body['message_event_id'] += '-foreign'
        elif change == 'run':
            run_id = 'run_foreign'
        elif change == 'artifact':
            with pytest.raises(PeerRunsHTTPError):
                await read(c, artifact_id='rart_' + '0' * 32)
            unretired(c)
            return
        elif change in ('nonterminal', 'generation'):
            sql = "UPDATE session_admissions SET status='started'" if change == 'nonterminal' else 'UPDATE session_admissions SET generation=generation+1'
            c.target.db._execute_write(lambda conn: conn.execute(sql))
        else:
            key = _RESULT_PREFIX + c.row['admission_id']
            saved = json.loads(c.target.db._conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
            saved['result']['failed' if change == 'failed' else 'final_response'] = True if change == 'failed' else 'changed result'
            c.target.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(saved), key)))
        with pytest.raises(PeerRunsHTTPError):
            await ack(c, body=body, run_id=run_id)
        unretired(c)


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['body', 'write'])
@pytest.mark.parametrize('change', ['grant', 'epoch', 'owner'])
async def test_current_authority_rechecked_after_body_and_at_ack_write(files_target, monkeypatch, boundary, change):
    from gateway import hosted_rooms
    async with peer_case(files_target, monkeypatch) as c:
        assert await read(c) == c.output.read_bytes()
        def mutate():
            if change == 'grant':
                hosted_rooms.revoke_room_grant_id(c.target.db.db_path, claims=c.claims, expires_at=c.claims['status_expires_at'])
            elif change == 'epoch':
                c.target.db._execute_write(lambda conn: conn.execute('UPDATE runtime_epoch SET epoch=epoch+1'))
            else:
                c.target.runner.session_authority = None
        if boundary == 'body':
            c.wire.faults.after_body = mutate
        else:
            original = RoomArtifactOutbox.acknowledge
            def before(self, *args, **kwargs):
                mutate()
                return original(self, *args, **kwargs)
            monkeypatch.setattr(RoomArtifactOutbox, 'acknowledge', before)
        with pytest.raises(PeerRunsHTTPError):
            await ack(c)
        unretired(c)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['recipient', 'result', 'peer_receipt', 'owner', 'link'])
async def test_home_rechecks_actual_frozen_custody_after_byte_io(files_target, monkeypatch, change):
    async with peer_case(files_target, monkeypatch) as c:
        original = RoomArtifactOutbox.read
        def mutate_after_read(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            if change == 'recipient':
                payload = dict(c.task['payload'], recipient_member_ids=['reader'])
                c.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET payload_json=?', (json.dumps(payload),)))
            elif change == 'result':
                result = dict(c.stored['result'], peer_result_digest='0' * 64)
                c.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET result_json=?', (json.dumps(result),)))
            elif change == 'peer_receipt':
                c.db._execute_write(lambda conn: conn.execute("UPDATE hosted_room_remote_runs SET run_id='foreign'"))
            elif change == 'link':
                c.db._execute_write(lambda conn: conn.execute("UPDATE hosted_room_links SET target_url='https://foreign.example.test'"))
            else:
                c.authority.runner.session_authority = None
            return value
        monkeypatch.setattr(RoomArtifactOutbox, 'read', mutate_after_read)
        with _profile_runtime_scope(c.home, hydrate_secrets=False), pytest.raises(RoomArtifactError):
            await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
        assert not [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
        assert not [call for call in c.wire.calls if call[1].endswith('/artifacts/ack')]
        unretired(c)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['grant', 'provider', 'owner', 'generation'])
async def test_target_rechecks_authority_after_reading_body_bytes(files_target, monkeypatch, change):
    from gateway import hosted_rooms
    async with peer_case(files_target, monkeypatch) as c:
        original = RoomArtifactOutbox.read
        def after(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            if change == 'grant':
                hosted_rooms.revoke_room_grant_id(c.target.db.db_path, claims=c.claims, expires_at=c.claims['status_expires_at'])
            elif change == 'provider':
                c.target.adapter._room_output_invitation_permissions = None
            elif change == 'owner':
                c.target.runner.session_authority = None
            else:
                c.target.db._execute_write(lambda conn: conn.execute('UPDATE session_admissions SET generation=generation+1'))
            return value
        monkeypatch.setattr(RoomArtifactOutbox, 'read', after)
        with pytest.raises(PeerRunsHTTPError):
            await read(c)
        unretired(c)


@pytest.mark.asyncio
async def test_peer_append_transaction_rechecks_actual_custody(files_target, monkeypatch):
    from gateway import hosted_rooms
    async with peer_case(files_target, monkeypatch) as c:
        original = hosted_rooms.append_event
        def before(*args, **kwargs):
            if kwargs.get('expected_output') is not None:
                c.db._execute_write(lambda conn: conn.execute("UPDATE hosted_room_remote_runs SET run_id='foreign'"))
            return original(*args, **kwargs)
        monkeypatch.setattr(hosted_rooms, 'append_event', before)
        with _profile_runtime_scope(c.home, hydrate_secrets=False), pytest.raises(RoomArtifactError):
            await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
        assert not [e for e in c.service._events('room-one') if e['kind'] == 'message.member']
        unretired(c)


@pytest.mark.asyncio
async def test_actual_ack_and_retirement_replay_hold_both_grant_stores(files_target, monkeypatch):
    from gateway.platforms import api_server_room_artifacts
    async with peer_case(files_target, monkeypatch) as c:
        checked, commits = [], []
        require = api_server_room_artifacts.require_current_grant
        def inspect(conn, claims):
            assert conn.in_transaction
            checked.append(Path(conn.execute('PRAGMA database_list').fetchone()[2]).name)
            return require(conn, claims)
        monkeypatch.setattr(api_server_room_artifacts, 'require_current_grant', inspect)
        original = RoomArtifactOutbox.acknowledge
        def ack_writer(self, *args, **kwargs):
            current = self.authorize_write
            def fence(conn, scope):
                assert conn.in_transaction
                commits.append(scope.as_mapping())
                return current(conn, scope)
            self.authorize_write = fence
            return original(self, *args, **kwargs)
        monkeypatch.setattr(RoomArtifactOutbox, 'acknowledge', ack_writer)
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
            initial = len(checked)
            await asyncio.to_thread(c.service._publish_terminal_tasks, c.service._room('room-one'))
        assert commits == [c.stored['result']['artifact_scope']]
        assert len(checked) == initial, 'completed Home work no longer retransmits'
        # Explicit exact target replay still holds the original two-store checks.
        await ack(c)
        assert len(checked) > initial
        assert all(checked[i:i+2] == ['shared-state.db', 'state.db'] for i in range(0, len(checked), 2))


@pytest.mark.asyncio
async def test_signed_wrong_policy_does_not_authorize_existing_run(files_target, monkeypatch):
    from gateway.hosted_room_peer import issue_room_grant
    async with peer_case(files_target, monkeypatch) as c:
        assert await read(c) == c.output.read_bytes()
        signer = {k: c.claims[k] for k in ('grant_id', 'room_id', 'home_install_id', 'authority_gateway_id',
            'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'issued_at')}
        signer['execution_policy_digest'] = '0' * 64
        token = issue_room_grant(c.target.adapter._room_grant_secret(), **signer, permissions=c.claims['permissions'],
            ttl_seconds=c.claims['expires_at'] - c.claims['issued_at'], status_expires_at=c.claims['status_expires_at'])
        with pytest.raises(PeerRunsHTTPError):
            await read(c, grant=token)
        with pytest.raises(PeerRunsHTTPError):
            await ack(c, grant=token)
        unretired(c)
