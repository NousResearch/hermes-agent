"""Room proof v1: use a grant without disclosing its bearer signature to the endpoint.

The target derives the shared proof key with its installation signing secret. Every
request and response is bound end to end, including through a reverse proxy. The
unsigned grant payload is public scope metadata; the signature never crosses the wire.
"""
import hashlib
import hmac
import json
import time
import uuid

from gateway.hosted_room_peer import _b64decode, _b64encode, _split_token

SCHEME = 'HermesRoomProof '
RESPONSE_HEADER = 'Hermes-Room-Proof'
FRESHNESS_SECONDS = 60


def _mac(key, domain, fields):
    encoded = json.dumps(fields, separators=(',', ':'), ensure_ascii=True).encode()
    return hmac.new(key, domain + b'\0' + encoded, hashlib.sha256).hexdigest()


_BOUND_HEADERS = ('idempotency-key', 'x-hermes-session-key', 'last-event-id', 'content-type')


def _headers(headers):
    normalized, counts = {}, {}
    for key, value in (headers or {}).items():
        key = key.lower()
        if key in (*_BOUND_HEADERS, 'authorization'):
            counts[key] = counts.get(key, 0) + 1
            if counts[key] != 1:
                raise ValueError('duplicate room proof header')
        normalized[key] = value
    return {key: normalized.get(key, '') for key in _BOUND_HEADERS}


def issuance_request_id(grant, body):
    payload, _ = _split_token(grant)
    return hashlib.sha256(payload + b'\0refresh\0' + body).hexdigest()


def request_proof(grant, *, installation_id, method, path, body, headers=None, now=None):
    payload, key = _split_token(grant)
    fingerprint = hashlib.sha256(body).hexdigest()
    # Retrying issuance for this old grant and frozen parameters retrieves one successor.
    request_id = (issuance_request_id(grant, body)
                  if method == 'POST' and path == '/v1/room-members/grants/refresh'
                  else uuid.uuid4().hex)
    envelope = dict(v=1, payload=_b64encode(payload), target_install_id=installation_id,
                    request_id=request_id, issued_at=time.time() if now is None else now)
    envelope['mac'] = _mac(key, b'hermes-room-request-v1', [
        envelope, method, path, fingerprint, _headers(headers)])
    return SCHEME + _b64encode(json.dumps(envelope, separators=(',', ':')).encode()), key, envelope['mac']


def verify_request(header, *, secret, installation_id, method, path, body, headers=None, now=None):
    if not header.startswith(SCHEME) or len(header) > 24576:
        raise ValueError('invalid room proof')
    envelope = json.loads(_b64decode(header[len(SCHEME):]))
    if set(envelope) != {'v', 'payload', 'target_install_id', 'request_id', 'issued_at', 'mac'}:
        raise ValueError('invalid room proof fields')
    supplied = envelope.pop('mac')
    now = time.time() if now is None else now
    if (envelope['v'] != 1 or envelope['target_install_id'] != installation_id
            or type(envelope['issued_at']) not in (int, float)
            or not abs(now - envelope['issued_at']) <= FRESHNESS_SECONDS
            or not isinstance(envelope['request_id'], str)
            or not 16 <= len(envelope['request_id']) <= 64):
        raise ValueError('invalid or stale room proof')
    payload = _b64decode(envelope['payload'])
    key = hmac.new(secret, payload, hashlib.sha256).digest()
    expected = _mac(key, b'hermes-room-request-v1', [
        envelope, method, path, hashlib.sha256(body).hexdigest(), _headers(headers)])
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise ValueError('invalid room proof signature')
    token = _b64encode(payload) + '.' + _b64encode(key)
    return token, key, supplied, envelope


def response_proof(key, request_mac, status, body):
    return _mac(key, b'hermes-room-response-v1', [request_mac, status, hashlib.sha256(body).hexdigest()])


def verify_response(key, request_mac, status, body, supplied):
    if not isinstance(supplied, str) or not hmac.compare_digest(
            supplied, response_proof(key, request_mac, status, body)):
        raise ValueError('peer response did not prove the pinned installation')
