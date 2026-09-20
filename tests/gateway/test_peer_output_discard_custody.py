"""Home custody cannot promote drifted or unknown work into disposal/NEW."""

import json

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_discard import custody, discard, wire_discard
from tests.gateway.test_peer_output_fences import unretired

from gateway.hosted_room_artifacts import RoomArtifactError
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['recipient', 'result', 'receipt', 'link', 'unknown'])
async def test_home_drift_after_status_io_never_dispatches_discard(files_target, monkeypatch, change):
    async with peer_case(files_target, monkeypatch) as c:
        source = custody(c)
        original = source[2]._verify_remote
        def after(*args):
            original(*args)
            if change == 'recipient':
                payload = dict(c.task['payload'], recipient_member_ids=['writer'])
                c.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET payload_json=?', (json.dumps(payload),)))
            elif change == 'result':
                result = dict(c.stored['result'], peer_result_digest='0'*64)
                c.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_driver_tasks SET result_json=?', (json.dumps(result),)))
            else:
                sql = {'receipt': "UPDATE hosted_room_remote_runs SET run_id='foreign'",
                       'link': "UPDATE hosted_room_links SET target_url='https://foreign.example.test'",
                       'unknown': "UPDATE hosted_room_driver_tasks SET status='indeterminate'"}[change]
                c.db._execute_write(lambda conn: conn.execute(sql))
        monkeypatch.setattr(source[2], '_verify_remote', after)
        with pytest.raises(RoomArtifactError):
            await discard(c, source)
        assert not [x for x in c.wire.calls if x[1].endswith('/artifacts/discard')]
        unretired(c)
        assert len(c.launched) == len(c.executions) == 1


@pytest.mark.asyncio
async def test_home_drift_after_retirement_reply_is_not_confirmation(files_target, monkeypatch):
    async with peer_case(files_target, monkeypatch) as c:
        source = custody(c)
        def drift():
            c.db._execute_write(lambda conn: conn.execute("UPDATE hosted_room_remote_runs SET run_id='foreign'"))
        c.wire.faults.after_discard = drift
        with pytest.raises(RoomArtifactError):
            await discard(c, source)
        assert c.target.adapter._peer_output_outbox.retirement_complete(source[0])
        assert len(c.launched) == len(c.executions) == 1


@pytest.mark.asyncio
async def test_exact_replay_reauthorizes_grant_and_retains_status_horizon(files_target, monkeypatch):
    from gateway.hosted_room_peer import issue_room_grant, decode_room_grant
    from gateway import hosted_rooms
    async with peer_case(files_target, monkeypatch) as c:
        # A legitimately target-signed renewal is not a foreign grant merely
        # because its grant_id differs. Existing current scope/policy rules own it.
        signer = {k: c.claims[k] for k in ('grant_id', 'room_id', 'home_install_id', 'authority_gateway_id',
            'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'issued_at')}
        signer['grant_id'] = 'grant-refresh-test'
        renewed = issue_room_grant(c.target.adapter._room_grant_secret(), **signer,
            permissions=c.claims['permissions'], ttl_seconds=c.claims['expires_at']-c.claims['issued_at'],
            status_expires_at=c.claims['status_expires_at'])
        assert await wire_discard(c, grant=renewed) == dict(discarded=True, removed=1)
        with monkeypatch.context() as clock:
            clock.setattr('gateway.hosted_room_peer.time.time', lambda: c.claims['expires_at']+1)
            decode_room_grant(c.target.adapter._room_grant_secret(), renewed, permission='status')
            with pytest.raises(PeerRunsHTTPError):
                await wire_discard(c, grant=renewed)
        hosted_rooms.revoke_room_grant_id(c.target.db.db_path, claims=c.claims, expires_at=c.claims['status_expires_at'])
        with pytest.raises(PeerRunsHTTPError):
            await discard(c)
