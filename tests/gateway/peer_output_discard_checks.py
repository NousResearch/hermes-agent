"""Canonical Output retirement; real HTTP/signing/Run/producer, inert agent only."""

import json


import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_fences import unretired


from gateway.session_results import _RESULT_PREFIX, admission_result
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


from tests.gateway.test_peer_output_discard import wire_discard

async def assert_discard_denied(files_target, monkeypatch, change):
    from gateway.hosted_room_peer import issue_room_grant, decode_room_grant
    from gateway import hosted_rooms
    async with peer_case(files_target, monkeypatch) as c:
        body = dict(reason='verification_failed', result_digest=c.stored['result']['peer_result_digest'])
        token, run_id = c.issued['grant'], c.accepted['run_id']
        if change in ('digest', 'reason', 'extra'):
            body[{'digest': 'result_digest', 'reason': 'reason', 'extra': 'extra'}[change]] = 'changed'
        elif change == 'run':
            run_id = 'run_foreign'
        elif change in ('read_only', 'policy', 'grant', 'profile'):
            fields = {k: c.claims[k] for k in ('grant_id', 'room_id', 'home_install_id', 'authority_gateway_id',
                'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'issued_at')}
            if change != 'read_only':
                fields[{'policy': 'execution_policy_digest', 'grant': 'member_id', 'profile': 'target_profile'}[change]] = '0' * 64 if change == 'policy' else 'foreign'
            token = issue_room_grant(c.target.adapter._room_grant_secret(), **fields,
                permissions=('artifact.read',) if change == 'read_only' else c.claims['permissions'],
                ttl_seconds=c.claims['expires_at']-c.claims['issued_at'], status_expires_at=c.claims['status_expires_at'])
        elif change == 'owner':
            c.target.runner.session_authority = None
        elif change in ('request_digest', 'unknown', 'generation'):
            sql = {'request_digest': "UPDATE session_admissions SET payload_digest='changed'",
                   'unknown': "UPDATE session_admissions SET status='unknown'",
                   'generation': 'UPDATE session_admissions SET generation=generation+1'}[change]
            c.target.db._execute_write(lambda conn: conn.execute(sql))
        elif change == 'result':
            saved = admission_result(c.target.db, c.row['admission_id'])
            saved['result']['final_response'] = 'changed'
            c.target.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                (json.dumps(saved), _RESULT_PREFIX + c.row['admission_id'])))
        elif change == 'horizon':
            monkeypatch.setattr('gateway.hosted_room_peer.time.time', lambda: c.claims['expires_at']+1)
            decode_room_grant(c.target.adapter._room_grant_secret(), token, permission='status')
        else:
            hosted_rooms.revoke_room_grant_id(c.target.db.db_path, claims=c.claims, expires_at=c.claims['status_expires_at'])
        with pytest.raises((PeerRunsHTTPError, ValueError)):
            await wire_discard(c, body=body, grant=token, run_id=run_id)
        unretired(c)
        retained = json.loads(c.target.db._conn.execute('SELECT value FROM state_meta WHERE key=?',
            (_RESULT_PREFIX + c.row['admission_id'],)).fetchone()[0])
        assert 'peer_output_discard' not in retained
        if change == 'unknown':
            assert c.target.db._conn.execute('SELECT status FROM session_admissions').fetchone()[0] == 'unknown'
        assert len(c.launched) == 1
