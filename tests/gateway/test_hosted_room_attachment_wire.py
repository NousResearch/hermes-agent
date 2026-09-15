"""Metadata-only I wire compatibility; these tests do not activate Files."""
import hashlib
import json

import pytest

from gateway import hosted_room_peer as peer
from tests.gateway.test_hosted_room_peer import _dispatch


TEXT_RIGHTS = ('approve', 'dispatch', 'status', 'stop')
STAGE_RIGHTS = ('approve', 'attachment.stage', 'dispatch', 'status', 'stop')
SECRET = b'metadata-only-test-signing-key-32bytes'


def entry(**changes):
    return dict(attachment_id='attachment-1', kind='file', name='note.txt',
                size=3, mime='text/plain', sha256=hashlib.sha256(b'abc').hexdigest()) | changes


def signed(**changes):
    dispatch = _dispatch()
    fields = {name: getattr(dispatch, name) for name in (
        'room_id', 'home_install_id', 'authority_gateway_id', 'authority_epoch',
        'member_id', 'target_install_id', 'target_profile', 'execution_policy_digest')}
    return peer.issue_room_grant(SECRET, **(fields | dict(grant_id='grant-1',
        issued_at=100, ttl_seconds=60, status_ttl_seconds=120) | changes))


def test_optional_digest_preserves_text_wire_bytes_and_binds_manifest():
    original = _dispatch().as_mapping()
    assert 'attachment_manifest_digest' not in original
    encoded = json.dumps(original, sort_keys=True, separators=(',', ':')).encode()
    text = peer.HostedMemberDispatch.from_mapping(original)
    assert text.attachment_manifest_digest is None
    assert json.dumps(text.as_mapping(), sort_keys=True, separators=(',', ':')).encode() == encoded
    digest = hashlib.sha256(b'manifest').hexdigest()
    wire = original | {'attachment_manifest_digest': digest}
    attached = peer.HostedMemberDispatch.from_mapping(wire)
    assert attached.as_mapping() == wire
    assert attached.prompt_digest == text.prompt_digest
    assert hashlib.sha256(json.dumps(wire, sort_keys=True, separators=(',', ':')).encode()).digest() != hashlib.sha256(encoded).digest()


@pytest.mark.parametrize('value', [None, True, 1, '', 'a' * 63, 'A' * 64, 'g' * 64, [], {}])
def test_dispatch_rejects_present_invalid_manifest_digest(value):
    with pytest.raises(peer.HostedRoomPeerError):
        _dispatch(attachment_manifest_digest=value)


def test_manifest_normalization_order_digest_and_exact_bounds():
    manifest = [entry(name=' résumé.txt ', mime=' TEXT/PLAIN '), entry(attachment_id='second', kind='pdf')]
    expected = [entry(name='résumé.txt'), entry(attachment_id='second', kind='pdf')]
    assert peer.canonical_attachment_manifest(manifest) == expected
    assert manifest[0]['name'] == ' résumé.txt '
    encoded = json.dumps(expected, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    digest = peer.attachment_manifest_digest(manifest)
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert peer.attachment_manifest_digest(expected[::-1]) != digest
    count = peer.MAX_ROOM_LINK_ATTACHMENTS
    assert len(peer.canonical_attachment_manifest([entry(attachment_id=f'a-{i}') for i in range(count)])) == count
    sizes = [15_000_000, 15_000_000, 15_000_000, 5_000_000]
    batch = [entry(attachment_id=f'a-{i}', size=size) for i, size in enumerate(sizes)]
    assert sum(item['size'] for item in peer.canonical_attachment_manifest(batch)) == peer.MAX_ROOM_LINK_ATTACHMENT_BATCH_BYTES
    batch[-1]['size'] += 1
    with pytest.raises(peer.HostedRoomPeerError):
        peer.canonical_attachment_manifest(batch)


@pytest.mark.parametrize('value', [None, {}, (), [], [None], [True], [entry(), entry()],
    [entry(attachment_id=f'a-{i}') for i in range(17)], [entry(path='/private/file')],
    [{key: value for key, value in entry().items() if key != 'sha256'}],
    *[[entry(**{key: value})] for key, values in {
        'attachment_id': ['', True, 1, 'a' * 257],
        'kind': [True, 1, [], {}, 'video'],
        'name': ['', ' ', '.', '..', 'a/b', 'a\\b', 'a\x00b', 'a\nb', 'a\rb', 'a' * 256, True],
        'size': [True, False, 0, -1, 1.5, '3', 15_000_001],
        'mime': ['', 'plain', True, 'a/' + 'b' * 126],
        'sha256': [None, True, 'A' * 64, 'a' * 63],
    }.items() for value in values]])
def test_manifest_rejects_malformed_or_unbounded_metadata(value):
    with pytest.raises(peer.HostedRoomPeerError):
        peer.canonical_attachment_manifest(value)
    with pytest.raises(peer.HostedRoomPeerError):
        peer.attachment_manifest_digest(value)


def test_generic_signer_default_stays_text_only_but_explicit_stage_has_short_horizon():
    old = signed()
    assert tuple(peer.decode_room_grant(SECRET, old, permission='dispatch', now=110)['permissions']) == TEXT_RIGHTS
    for rights in [TEXT_RIGHTS, ('dispatch',), ('status',)]:
        with pytest.raises(peer.HostedRoomGrantError, match='allow'):
            peer.decode_room_grant(SECRET, signed(permissions=rights), permission='attachment.stage', now=110)
    explicit = signed(permissions=STAGE_RIGHTS)
    assert tuple(peer.decode_room_grant(SECRET, explicit, permission='attachment.stage', now=110)['permissions']) == STAGE_RIGHTS
    with pytest.raises(peer.HostedRoomGrantError, match='expired'):
        peer.decode_room_grant(SECRET, explicit, permission='attachment.stage', now=160)
    assert peer.decode_room_grant(SECRET, explicit, permission='status', now=160)['status_expires_at'] == 220


@pytest.mark.parametrize('attachments,expected', [(False, TEXT_RIGHTS), (True, STAGE_RIGHTS)])
def test_invitation_permissions_from_validated_catalog_metadata_only(attachments, expected):
    catalog = peer.catalog_mapping(installation_id='test', persistent_process=True, attachments=attachments)
    assert peer.invitation_permissions(catalog) == expected
    for malformed in [None, 0, 1, 'true', [], {}]:
        with pytest.raises(peer.HostedRoomPeerError):
            peer.invitation_permissions(catalog | {'attachments': malformed})
    with pytest.raises(peer.HostedRoomPeerError):
        peer.invitation_permissions(catalog | {'attachments': not attachments})
