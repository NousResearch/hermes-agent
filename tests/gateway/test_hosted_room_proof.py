"""Proofs bind both directions without disclosing the room grant bearer."""
import json

import pytest

from gateway import hosted_room_proof as proof
from gateway.hosted_room_peer import _b64decode, _b64encode, issue_room_grant


def grant():
    return issue_room_grant(b'x' * 32, grant_id='proof', room_id='room', home_install_id='home',
        authority_gateway_id='home', authority_epoch=1, member_id='member', target_install_id='target',
        target_profile='default', execution_policy_digest='a' * 64, issued_at=100, ttl_seconds=3600)


def request(token=None, **kwargs):
    return proof.request_proof(token or grant(), installation_id='target', method='POST',
                              path='/v1/runs', body=b'{"input":"hello"}', now=120, **kwargs)


def verify(header, **kwargs):
    return proof.verify_request(header, **(dict(secret=b'x' * 32, installation_id='target', method='POST',
        path='/v1/runs', body=b'{"input":"hello"}', now=120) | kwargs))


def test_proof_uses_grant_without_transmitting_reusable_signature():
    header, key, mac = request()
    token, derived, verified, envelope = verify(header)
    assert token == grant() and key == derived and mac == verified
    assert grant() not in header
    assert grant().split('.')[1] not in json.dumps(json.loads(_b64decode(header[len(proof.SCHEME):])))
    response = proof.response_proof(key, mac, 202, b'{"run_id":"accepted"}')
    proof.verify_response(key, mac, 202, b'{"run_id":"accepted"}', response)


@pytest.mark.parametrize('changed', [
    {'method': 'GET'}, {'path': '/v1/room-members/grants/revoke'}, {'body': b'{"input":"other"}'},
    {'headers': {'Idempotency-Key': 'forged'}}, {'headers': {'X-Hermes-Session-Key': 'other'}},
    {'installation_id': 'replacement'}, {'secret': b'y' * 32}, {'now': 181}, {'now': 59}])
def test_request_tampering_wrong_installation_and_staleness_are_refused(changed):
    with pytest.raises(ValueError):
        verify(request()[0], **changed)


@pytest.mark.parametrize('field,value', [('request_id', 'another-request-id'), ('issued_at', 121), ('v', 2)])
def test_signed_request_identity_cannot_be_changed(field, value):
    header = request()[0]
    envelope = json.loads(_b64decode(header[len(proof.SCHEME):]))
    envelope[field] = value
    with pytest.raises(ValueError):
        verify(proof.SCHEME + _b64encode(json.dumps(envelope).encode()))


@pytest.mark.parametrize('status,body,request_mac', [(200, b'ok', None), (202, b'changed', None), (202, b'ok', 'other')])
def test_response_cannot_change_or_move_to_another_request(status, body, request_mac):
    _, key, mac = request()
    signed = proof.response_proof(key, mac, 202, b'ok')
    with pytest.raises(ValueError):
        proof.verify_response(key, request_mac or mac, status, body, signed)
