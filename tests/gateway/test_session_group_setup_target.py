"""Canonical target refusal does not remove legacy invitation availability."""
import contextlib
import contextvars
import json
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms as rooms
from gateway.hosted_room_peer import decode_room_grant
from gateway.platforms import api_server_room_grants as grants


@pytest.mark.asyncio
@pytest.mark.parametrize('owner', ['canonical', 'missing', 'legacy'])
async def test_target_catalog_and_new_invitation_refuse_canonical_owner(tmp_path, monkeypatch, owner):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(rooms, 'local_authority_gateway_id', lambda: 'target')
    registry = SimpleNamespace(active=lambda: None)
    runner = {'canonical': SimpleNamespace(session_authority=object()),
        'missing': SimpleNamespace(session_authorities=registry), 'legacy': SimpleNamespace()}[owner]
    body = dict(room_id='room', home_install_id='home', authority_gateway_id='home', authority_epoch=1, member_id='remote')
    async def read(_):
        return body, None
    secret_reads = []
    def secret():
        secret_reads.append(True)
        return b'z' * 32
    adapter = SimpleNamespace(gateway_runner=runner, _profile_scope=lambda p: contextlib.nullcontext(),
        _check_auth=lambda req: None, _read_json_body=read, _room_grant_secret=secret)
    error = lambda message, **kw: {'error': {'message': message, **kw}}
    profile = contextvars.ContextVar('target_profile', default='default')
    _, catalog = grants._local_room_catalog(adapter, 'default', 'target')
    response = await grants._handle_room_member_invitation(adapter, object(), _openai_error=error, _api_request_profile=profile)
    value = json.loads(response.text)
    assert catalog['text'] is (owner == 'legacy')
    assert catalog['attachments'] is False
    if owner == 'legacy':
        assert response.status == 201, value
        claims = decode_room_grant(b'z' * 32, value['grant'], permission='dispatch')
        assert claims['target_install_id'] == 'target'
        assert claims['room_id'] == body['room_id']
    else:
        assert response.status == 409, value
        assert value['error']['code'] == 'canonical_room_peer_unsupported'
        assert not secret_reads
        assert 'grant' not in value
