import base64
import hashlib
import hmac
import json

import pytest

from tests.gateway.test_canonical_peer_target_setup import invite, target

__all__ = ['target']


@pytest.mark.asyncio
@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('expired', [False, True])
async def test_native_legacy_horizon_preserves_signed_identity_and_denial(target, monkeypatch, exact, expired):
    from gateway import hosted_rooms
    from gateway.hosted_room_peer import decode_room_grant
    issued = await invite(target)
    secret = target.adapter._room_grant_secret()
    claims = decode_room_grant(secret, issued['grant'], permission='status')
    payload = {k: v for k, v in claims.items() if k not in {'_token_sha256', 'status_expires_at'}}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('ascii')

    def encode(data):
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')

    token = encode(encoded) + '.' + encode(hmac.new(secret, encoded, hashlib.sha256).digest())
    legacy = decode_room_grant(secret, token, permission='status')
    assert 'status_expires_at' not in legacy
    assert legacy['_token_sha256'] == hashlib.sha256(token.encode('ascii')).hexdigest()
    if expired:
        monkeypatch.setattr('gateway.hosted_room_peer.time.time', lambda: legacy['expires_at'] + 1)
    target.runner.adapters.clear()
    result = await target.connection.dispatch(dict(id=2,
        method='groups.peer.revoke_exact' if exact else 'groups.peer.revoke', params={'grant': token}))
    assert result.get('result') == {'revoked': True, 'exact': exact}, result
    for path in (target.home / 'shared-state.db', target.home / 'state.db'):
        assert hosted_rooms.room_grant_is_revoked(path, claims=legacy) is not expired
        if exact:
            assert not hosted_rooms.room_grant_is_revoked(path, claims=claims)
        with hosted_rooms._transaction(path) as conn:
            table = 'hosted_room_revoked_grant_tokens' if exact else 'hosted_room_revoked_grants'
            rows = conn.execute(f'SELECT * FROM {table}').fetchall()
            if expired:
                assert not rows
            else:
                assert len(rows) == 1
                assert rows[0]['expires_at'] == legacy['expires_at']
                if exact:
                    assert rows[0]['token_sha256'] == legacy['_token_sha256']
