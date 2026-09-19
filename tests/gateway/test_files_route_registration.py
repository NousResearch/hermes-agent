"""Real outbound registration authenticates signed rights, not idle readiness."""
import copy
from types import SimpleNamespace

import pytest

from tests.gateway.test_canonical_peer_target_setup import target, invite, invitation  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target, inprocess_http  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['none', 'installation', 'profile', 'policy', 'protocol', 'link', 'persistent', 'text', 'endpoint', 'rights', 'signature', 'revoked'])
async def test_registration_tolerates_only_files_bit_with_authenticated_rights(files_target, monkeypatch, change):
    from gateway import hosted_rooms, hosted_room_links
    from gateway.hosted_room_peer import _catalog_digest, decode_room_grant, issue_room_grant
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from tui_gateway import hosted_room_service
    t = files_target
    monkeypatch.setattr(hosted_room_service, 'HostedRoomRuntime', lambda **kwargs: SimpleNamespace(
        status=lambda: {'running': True, 'stopping': False}, wakeup=lambda: None))
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'https://target.example.test')
    service = CanonicalHostedRoomService(t.authority, None)
    t.authority.hosted_room_service = service
    installation = hosted_rooms.local_authority_gateway_id()
    issued = await invite(t, invitation() | dict(home_install_id=installation, authority_gateway_id=installation))
    assert issued['catalog']['attachments'] is True
    catalog = copy.deepcopy(issued['catalog'])
    token = issued['grant']
    claims = decode_room_grant(t.adapter._room_grant_secret(), token, permission='attachment.stage')
    if change in ('installation', 'protocol', 'link', 'persistent', 'text', 'endpoint'):
        field, value = {
            'installation': ('installation_id', 'foreign-install'),
            'protocol': ('protocol_versions', [1, 2]), 'link': ('link_modes', ['direct', 'pull']),
            'persistent': ('persistent_process', False), 'text': ('text', False),
            'endpoint': ('endpoint', {'available': True, 'url': 'https://foreign.example.test', 'transport_security': 'tls'})}[change]
        catalog[field] = value
    elif change in ('profile', 'policy'):
        from gateway.hosted_room_execution_policy import _policy_digest
        policy = dict(catalog['execution_policy'])
        policy['target_profile' if change == 'profile' else 'max_iterations'] = 'foreign' if change == 'profile' else 99
        # A canonical policy with changed stable content, not an invalid digest.
        policy['policy_digest'] = _policy_digest({k: v for k, v in policy.items() if k != 'policy_digest'})
        catalog['execution_policy'] = policy
    elif change == 'rights':
        signer = {k: claims[k] for k in ('grant_id', 'room_id', 'home_install_id', 'authority_gateway_id',
            'authority_epoch', 'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest', 'issued_at')}
        token = issue_room_grant(t.adapter._room_grant_secret(), **signer,
            permissions=('approve', 'dispatch', 'status', 'stop'),
            ttl_seconds=claims['expires_at']-claims['issued_at'], status_expires_at=claims['status_expires_at'])
    elif change == 'signature':
        encoded, signature = token.split('.')
        token = encoded + '.' + ('A' if signature[0] != 'A' else 'B') + signature[1:]
    elif change == 'revoked':
        hosted_rooms.revoke_room_grant_id(t.home / 'state.db', claims=claims, expires_at=claims['status_expires_at'])
    catalog['catalog_digest'] = _catalog_digest(catalog)
    from gateway.hosted_room_peer import GatewayRoomCatalog
    assert GatewayRoomCatalog.from_mapping(catalog).as_mapping() == catalog
    service.authorize_room(t.connection.actor.subject, 'room-one', create=True)
    hosted_rooms.create_room(t.db.db_path, room_id='room-one', name='Room', authority_gateway_id=installation,
        members=[{'member_id': 'member-one', 'profile': 'default', 'handle': 'member-one', 'target': {
            'kind': 'peer', 'peer_id': catalog['installation_id'], 'installation_id': catalog['installation_id'],
            'profile': 'default', 'capability_digest': catalog['catalog_digest']}}])
    responses = []
    inprocess_http(t, monkeypatch, responses=responses)
    result = await t.connection.dispatch(dict(id=2, method='groups.peer.register', params=dict(
        request_id='register-files', room_id='room-one', member_id='member-one', target_url=issued['endpoint']['url'],
        target_profile='default', grant=token, catalog=catalog)))
    stored = hosted_room_links.load_room_link(t.db.db_path, room_id='room-one', member_id='member-one')
    if change == 'none':
        assert result.get('result', {}).get('registered') is True, result
        assert stored.grant == token and stored.catalog.as_mapping() == issued['catalog']
        assert service.peer_routes[('room-one', 'member-one')].attachments is True
        passive = [body['catalog'] for _, path, status, body in responses if path.endswith('/capabilities') and status == 200]
        assert passive and all(c['attachments'] is False for c in passive)
        assert all(_catalog_digest(dict(c, attachments=True)) == catalog['catalog_digest'] for c in passive)
    else:
        assert 'error' in result, result
        assert stored is None and ('room-one', 'member-one') not in service.peer_routes
