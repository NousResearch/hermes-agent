"""Real target bindings distinguish unavailable gateways from standalone Serve."""
import json

import pytest

from gateway import hosted_rooms as rooms
from gateway.hosted_room_peer import decode_room_grant
from gateway.platforms import api_server_room_grants as grants
from gateway.session_authorities import SessionAuthorities
from tests.gateway.test_canonical_peer_target_setup import request, target  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize('binding', ['detached', 'missing', 'standalone', 'bound'])
async def test_target_invitation_respects_real_owner_binding(target, monkeypatch, binding):
    adapter = target.adapter
    if binding == 'detached':
        target.runner.adapters.clear()
    elif binding == 'missing':
        target.runner.session_authorities = SessionAuthorities(target.home)
        target.runner.session_authority = None
    elif binding == 'standalone':
        adapter.gateway_runner = None
    body = dict(room_id='room', home_install_id='home', authority_gateway_id='home', authority_epoch=1, member_id='remote')
    secret_reads = []
    original_secret = adapter._room_grant_secret
    def secret():
        secret_reads.append(True)
        return original_secret()
    monkeypatch.setattr(adapter, '_room_grant_secret', secret)
    installation_id = rooms.local_authority_gateway_id()
    _, catalog = grants._local_room_catalog(adapter, 'default', installation_id)
    response = await adapter._handle_room_member_invitation(request(body))
    value = json.loads(response.text)
    available = binding in {'standalone', 'bound'}
    assert catalog['text'] is available
    assert catalog['attachments'] is False
    if available:
        assert response.status == 201, value
        assert secret_reads
        claims = decode_room_grant(original_secret(), value['grant'], permission='dispatch')
        assert claims['target_install_id'] == installation_id
        assert claims['room_id'] == body['room_id']
    else:
        assert response.status == 409, value
        assert value['error']['code'] == 'canonical_room_peer_unsupported'
        assert not secret_reads
        assert 'grant' not in value
