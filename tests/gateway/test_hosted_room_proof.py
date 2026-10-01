"""Both directions bind raw ciphertext without disclosing the grant or private bodies."""
import json

import pytest

from gateway import hosted_room_proof as proof
from gateway.hosted_room_peer import _b64decode, _b64encode, issue_room_grant


def grant():
    return issue_room_grant(b'x' * 32, grant_id='proof', room_id='room', home_install_id='home',
        authority_gateway_id='home', authority_epoch=1, member_id='member', target_install_id='target',
        target_profile='default', execution_policy_digest='a' * 64, issued_at=100, ttl_seconds=3600)


def request(**kwargs):
    return proof.request_proof(grant(), **(dict(installation_id='target', method='POST',
        path='/v1/runs', body=b'{"input":"private hello"}', now=120) | kwargs))


def verify(header, wire, **kwargs):
    return proof.verify_request(header, **(dict(secret=b'x' * 32, installation_id='target', method='POST',
        path='/v1/runs', body=wire, now=120) | kwargs))


def test_proof_encrypts_private_bodies_and_never_transmits_reusable_signature():
    header, key, mac, wire = request()
    token, derived, verified, envelope, plain = verify(header, wire)
    assert token == grant() and key == derived and mac == verified
    assert plain == b'{"input":"private hello"}' and b'private hello' not in wire
    assert grant() not in header
    assert grant().split('.')[1] not in json.dumps(json.loads(_b64decode(header[len(proof.SCHEME):])))
    cipher, nonce, response_mac = proof.seal_response(key, mac, 202, b'{"output":"private reply"}')
    assert b'private reply' not in cipher
    assert proof.verify_response(key, mac, 202, cipher, response_mac, nonce) == b'{"output":"private reply"}'


@pytest.mark.parametrize('changed', [
    {'method': 'GET'}, {'path': '/v1/room-members/grants/revoke'}, {'body': b'{"input":"other"}'},
    {'headers': {'Idempotency-Key': 'forged'}}, {'headers': {'X-Hermes-Session-Key': 'other'}},
    {'installation_id': 'replacement'}, {'secret': b'y' * 32}, {'now': 181}, {'now': 59}])
def test_request_tampering_wrong_installation_and_staleness_are_refused(changed):
    header, _, _, wire = request()
    with pytest.raises(ValueError):
        verify(header, wire, **changed)


@pytest.mark.parametrize('field,value', [('request_id', 'another-request-id'), ('issued_at', 121),
                                       ('v', 3), ('body_nonce', 'AAAAAAAAAAAAAAAA')])
def test_signed_request_identity_and_nonce_cannot_be_changed(field, value):
    header, _, _, wire = request()
    envelope = json.loads(_b64decode(header[len(proof.SCHEME):]))
    envelope[field] = value
    with pytest.raises(ValueError):
        verify(proof.SCHEME + _b64encode(json.dumps(envelope).encode()), wire)


@pytest.mark.parametrize('change', ['status', 'body', 'request', 'nonce'])
def test_response_cannot_change_or_move_to_another_request(change):
    _, key, mac, _ = request()
    cipher, nonce, signed = proof.seal_response(key, mac, 202, b'private reply')
    status = 200 if change == 'status' else 202
    cipher = bytes([cipher[0] ^ 1]) + cipher[1:] if change == 'body' else cipher
    with pytest.raises(ValueError):
        proof.verify_response(key, 'other' if change == 'request' else mac, status, cipher, signed,
                              'AAAAAAAAAAAAAAAA' if change == 'nonce' else nonce)


def test_fresh_wire_nonces_preserve_plaintext_issuance_identity_and_empty_get_semantics():
    first = request(path='/v1/room-members/grants/refresh', body=b'{"ttl_seconds":3600}')
    second = request(path='/v1/room-members/grants/refresh', body=b'{"ttl_seconds":3600}')
    a = verify(first[0], first[3], path='/v1/room-members/grants/refresh')
    b = verify(second[0], second[3], path='/v1/room-members/grants/refresh')
    assert first[3] != second[3] and a[3]['body_nonce'] != b[3]['body_nonce']
    assert a[3]['request_id'] == b[3]['request_id'] and a[4] == b[4]
    header, _, _, wire = request(method='GET', body=b'')
    assert wire == b'' and verify(header, wire, method='GET')[4] == b''
