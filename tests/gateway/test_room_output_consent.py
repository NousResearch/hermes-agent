"""Route export-consent metadata only, never positive Output readiness."""
import asyncio
from dataclasses import replace
import hashlib
import hmac
import json
import threading

import pytest

from gateway import hosted_room_peer as peer
from tests.gateway.test_hosted_room_attachment_wire import SECRET, signed, TEXT_RIGHTS, STAGE_RIGHTS
from tests.gateway.test_hosted_room_peer import _dispatch
from tests.gateway.test_canonical_peer_target_setup import target as base_target, invite, invitation, request  # noqa: F401
from tests.gateway.test_canonical_peer_invitation_permissions import receipts


OUTPUT_RIGHTS = ('artifact.ack', 'artifact.read')
PROVIDER = '_room_output_invitation_permissions'


@pytest.fixture
def target(base_target, monkeypatch):
    # Codec helper imports load gateway.run before the temporary root exists.
    # Bind its process config home to the actual root, never fake the policy.
    from gateway import run
    monkeypatch.setattr(run, '_hermes_home', base_target.home)
    return base_target


@pytest.mark.parametrize('right', OUTPUT_RIGHTS)
def test_explicit_export_is_member_bound_and_short_lived_without_widening_defaults(right):
    token = signed(permissions=('status', right))
    assert peer.verify_room_grant(SECRET, token, _dispatch(), permission=right, now=110)
    with pytest.raises(peer.HostedRoomGrantError, match='scope'):
        peer.verify_room_grant(SECRET, token, replace(_dispatch(), member_id='other'), permission=right, now=110)
    for denied in ('attachment.stage', *[r for r in OUTPUT_RIGHTS if r != right]):
        with pytest.raises(peer.HostedRoomGrantError, match='allow'):
            peer.decode_room_grant(SECRET, token, permission=denied, now=110)
    with pytest.raises(peer.HostedRoomGrantError, match='expired'):
        peer.decode_room_grant(SECRET, token, permission=right, now=160)
    assert peer.decode_room_grant(SECRET, token, permission='status', now=160)
    for rights in (None, TEXT_RIGHTS, STAGE_RIGHTS, ('status',), ('dispatch',)):
        old = signed(**({} if rights is None else {'permissions': rights}))
        with pytest.raises(peer.HostedRoomGrantError, match='allow'):
            peer.decode_room_grant(SECRET, old, permission=right, now=110)


@pytest.mark.parametrize('rights', [('status', 'status'), ('unknown', 'status'), (True, 'status'), True])
def test_permission_codec_rejects_malformed_selection_in_signer_and_signed_payload(rights):
    with pytest.raises(peer.HostedRoomGrantError):
        signed(permissions=rights)
    # Malformed independently signed input must not pass just because it has status.
    payload = json.loads(peer._b64decode(signed().split('.')[0]))
    payload['permissions'] = list(rights) if isinstance(rights, tuple) else rights
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('ascii')
    token = peer._b64encode(encoded) + '.' + peer._b64encode(hmac.new(SECRET, encoded, hashlib.sha256).digest())
    with pytest.raises(peer.HostedRoomGrantError):
        peer.decode_room_grant(SECRET, token, permission='status', now=110)


@pytest.mark.parametrize('attachments', [False, True])
@pytest.mark.parametrize('extra', [(), OUTPUT_RIGHTS, ('artifact.read',)])
def test_trusted_selector_keeps_input_and_output_independent(target, monkeypatch, attachments, extra):
    from gateway.platforms.api_server_room_grants import _invitation_permissions
    catalog = peer.catalog_mapping(installation_id='metadata-only', persistent_process=True, attachments=attachments)
    base = STAGE_RIGHTS if attachments else TEXT_RIGHTS
    assert _invitation_permissions(target.adapter, 'default', catalog) == base
    calls = []
    def provider(*, profile, catalog, connection):
        calls.append((profile, catalog, connection))
        return extra
    monkeypatch.setattr(target.adapter, PROVIDER, provider, raising=False)
    # Synthetic catalog/provider tests the selector contract, not actual output.
    assert _invitation_permissions(target.adapter, 'default', catalog) == tuple(sorted((*base, *extra)))
    assert calls == [('default', catalog, None)]
    assert peer.invitation_permissions(catalog) == base


@pytest.mark.parametrize('bad', [None, True, False, 'artifact.read', ['artifact.read'],
    ('artifact.read', 'artifact.read'), ('unknown',), ('attachment.stage',), (True,), ({},)])
@pytest.mark.asyncio
async def test_malformed_provider_cannot_sign_or_reserve(target, monkeypatch, bad):
    def provider(**kwargs):
        return bad
    monkeypatch.setattr(target.adapter, PROVIDER, provider, raising=False)
    def no_sign(*args, **kwargs):
        pytest.fail('invalid provider reached signer')
    monkeypatch.setattr(peer, 'issue_room_grant', no_sign)
    native = await target.connection.dispatch(dict(id=1, method='groups.peer.invite', params=invitation()))
    assert 'error' in native
    params = invitation()
    params.pop('request_id')
    response = await target.adapter._handle_room_member_invitation(request(params))
    assert response.status == 400
    assert receipts(target) == []


@pytest.mark.asyncio
async def test_metadata_provider_is_fenced_and_recorded_by_both_issuers(target, monkeypatch):
    calls = []
    def provider(*, profile, catalog, connection):
        if connection is not None:
            assert connection.in_transaction
            assert connection.execute('PRAGMA database_list').fetchone()[2] == str(target.home / 'state.db')
        calls.append(connection is not None)
        assert profile == 'default' and catalog['attachments'] is False
        return OUTPUT_RIGHTS
    monkeypatch.setattr(target.adapter, PROVIDER, provider, raising=False)
    native = await invite(target)
    params = invitation()
    params.pop('request_id')
    http = await target.adapter._handle_room_member_invitation(request(params))
    assert http.status == 201, http.text
    rights = sorted((*TEXT_RIGHTS, *OUTPUT_RIGHTS))
    for item in (native, json.loads(http.text)):
        claims = peer.decode_room_grant(target.adapter._room_grant_secret(), item['grant'], permission='dispatch')
        assert claims['permissions'] == rights
    saved = receipts(target)[0]
    assert saved['intent']['permissions'] == saved['issue']['permissions'] == rights
    assert calls == [False, True, True, False, True]
    # No Output producer, outbox, reader, ACK or runtime was installed by this stub.


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['prepare', 'complete'])
@pytest.mark.parametrize('drift', ['appear', 'disappear', 'change'])
async def test_native_output_selection_drift_refuses_at_owned_write(target, monkeypatch, boundary, drift):
    initial = () if drift == 'appear' else OUTPUT_RIGHTS
    monkeypatch.setattr(target.adapter, PROVIDER, lambda **kw: initial, raising=False)
    entered, resume = threading.Event(), threading.Event()
    original = target.db._execute_write
    def write(fn, *args, **kwargs):
        if fn.__name__ == boundary:
            entered.set()
            assert resume.wait(10)
        return original(fn, *args, **kwargs)
    monkeypatch.setattr(target.db, '_execute_write', write)
    pending = asyncio.create_task(target.connection.dispatch(dict(id=1, method='groups.peer.invite', params=invitation())))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if drift == 'disappear':
            monkeypatch.delattr(target.adapter, PROVIDER)
        else:
            monkeypatch.setattr(target.adapter, PROVIDER, lambda **kw: OUTPUT_RIGHTS if drift == 'appear' else ('artifact.read',))
    finally:
        resume.set()
    result = await asyncio.wait_for(pending, 10)
    assert result.get('error', {}).get('message') == 'room_capability_catalog_changed', result
    saved = receipts(target)
    if boundary == 'prepare':
        assert not saved
    else:
        assert saved[0]['status'] == 'pending'
        # Restore the metadata selection: pending must still not remint.
        monkeypatch.setattr(target.adapter, PROVIDER, lambda **kw: initial, raising=False)
        def no_sign(*args, **kwargs):
            pytest.fail('pending invitation reminted')
        monkeypatch.setattr(peer, 'issue_room_grant', no_sign)
        retry = await target.connection.dispatch(dict(id=2, method='groups.peer.invite', params=invitation()))
        assert retry['error']['message'] == 'room_invitation_pending'


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['disappear', 'change'])
async def test_http_output_selection_drift_refuses_after_reservation(target, monkeypatch, drift):
    monkeypatch.setattr(target.adapter, PROVIDER, lambda **kw: OUTPUT_RIGHTS, raising=False)
    original = target.db._execute_write
    def write(fn, *args, **kwargs):
        if fn.__name__ == 'confirm':
            if drift == 'disappear':
                monkeypatch.delattr(target.adapter, PROVIDER)
            else:
                monkeypatch.setattr(target.adapter, PROVIDER, lambda **kw: ('artifact.read',))
        return original(fn, *args, **kwargs)
    monkeypatch.setattr(target.db, '_execute_write', write)
    params = invitation()
    params.pop('request_id')
    response = await target.adapter._handle_room_member_invitation(request(params))
    assert response.status == 400, response.text
    assert 'catalog changed' in json.loads(response.text)['error']['message']


@pytest.mark.asyncio
@pytest.mark.parametrize('change', [{'permissions': list(OUTPUT_RIGHTS)}, {'output': True},
    {'_room_output_invitation_permissions': True}, {'artifact.read': True}, {'raw_room_artifact_publication': True}])
async def test_client_cannot_select_export_rights(target, change):
    params = invitation()
    native = await target.connection.dispatch(dict(id=1, method='groups.peer.invite', params=params | change))
    assert 'error' in native
    params.pop('request_id')
    http = await target.adapter._handle_room_member_invitation(request(params | change))
    assert http.status == 400
    assert receipts(target) == []


@pytest.mark.asyncio
async def test_http_cannot_skip_confirmation_when_adapter_unbound_after_reservation(target, monkeypatch):
    from gateway import hosted_room_grant_state
    monkeypatch.setattr(target.adapter, PROVIDER, lambda **kw: OUTPUT_RIGHTS, raising=False)
    original = hosted_room_grant_state.reserve_grant_state
    def reserve(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(target.adapter, 'gateway_runner', None)
        return result
    monkeypatch.setattr(hosted_room_grant_state, 'reserve_grant_state', reserve)
    params = invitation()
    params.pop('request_id')
    response = await target.adapter._handle_room_member_invitation(request(params))
    assert response.status == 400, response.text
