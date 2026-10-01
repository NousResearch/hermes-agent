"""Room proof v2: authenticate the pinned installation and keep bodies confidential.

Only public grant scope crosses in the header. The bearer signature derives separate
request/response AEAD keys; it is never transmitted. Random wire ciphertext never
changes the plaintext task or issuance identity used by the owning runtime.
"""
import hashlib
import hmac
import json
import secrets
import time
import uuid

from gateway.hosted_room_peer import _b64decode, _b64encode, _split_token

SCHEME = 'HermesRoomProof '
RESPONSE_HEADER = 'Hermes-Room-Proof'
RESPONSE_NONCE_HEADER = 'Hermes-Room-Nonce'
FRESHNESS_SECONDS = 60
WIRE_OVERHEAD = 16  # AES-GCM tag; the 12-byte nonce is carried in the bounded header
_BOUND_HEADERS = ('idempotency-key', 'x-hermes-session-key', 'last-event-id', 'content-type')


def _encoded(fields):
    return json.dumps(fields, separators=(',', ':'), ensure_ascii=True, sort_keys=True).encode()


def _mac(key, domain, fields):
    return hmac.new(key, domain + b'\0' + _encoded(fields), hashlib.sha256).hexdigest()


def _body_key(key, direction):
    return hmac.new(key, b'hermes-room-body-key-v2\0' + direction, hashlib.sha256).digest()


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


def _decrypt(key, nonce, body, aad):
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    try:
        decoded = _b64decode(nonce)
        if len(decoded) != 12:
            raise ValueError('invalid room body nonce')
        return AESGCM(key).decrypt(decoded, body, aad)
    except InvalidTag as exc:
        raise ValueError('room body authentication failed') from exc


def issuance_request_id(grant, body):
    payload, _ = _split_token(grant)
    return hashlib.sha256(payload + b'\0refresh\0' + body).hexdigest()


def request_proof(grant, *, installation_id, method, path, body, headers=None, now=None):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    payload, key = _split_token(grant)
    request_id = (issuance_request_id(grant, body)
                  if method == 'POST' and path == '/v1/room-members/grants/refresh'
                  else uuid.uuid4().hex)
    nonce = secrets.token_bytes(12) if body else b''
    envelope = dict(v=2, payload=_b64encode(payload), target_install_id=installation_id,
                    request_id=request_id, issued_at=time.time() if now is None else now,
                    body_nonce=_b64encode(nonce))
    binding = [envelope, method, path, _headers(headers)]
    wire = AESGCM(_body_key(key, b'request')).encrypt(
        nonce, body, b'hermes-room-request-body-v2\0' + _encoded(binding)) if body else b''
    envelope['mac'] = _mac(key, b'hermes-room-request-v2', [
        *binding, hashlib.sha256(wire).hexdigest()])
    return SCHEME + _b64encode(_encoded(envelope)), key, envelope['mac'], wire


def verify_request(header, *, secret, installation_id, method, path, body, headers=None, now=None):
    if not header.startswith(SCHEME) or len(header) > 24576:
        raise ValueError('invalid room proof')
    envelope = json.loads(_b64decode(header[len(SCHEME):]))
    if set(envelope) != {'v', 'payload', 'target_install_id', 'request_id', 'issued_at', 'body_nonce', 'mac'}:
        raise ValueError('invalid room proof fields')
    supplied = envelope.pop('mac')
    now = time.time() if now is None else now
    if (envelope['v'] != 2 or envelope['target_install_id'] != installation_id
            or type(envelope['issued_at']) not in (int, float)
            or not abs(now - envelope['issued_at']) <= FRESHNESS_SECONDS
            or not isinstance(envelope['request_id'], str)
            or not 16 <= len(envelope['request_id']) <= 64
            or not isinstance(envelope['body_nonce'], str)):
        raise ValueError('invalid or stale room proof')
    payload = _b64decode(envelope['payload'])
    key = hmac.new(secret, payload, hashlib.sha256).digest()
    binding = [envelope, method, path, _headers(headers)]
    expected = _mac(key, b'hermes-room-request-v2', [*binding, hashlib.sha256(body).hexdigest()])
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        raise ValueError('invalid room proof signature')
    if envelope['body_nonce']:
        plaintext = _decrypt(_body_key(key, b'request'), envelope['body_nonce'], body,
                             b'hermes-room-request-body-v2\0' + _encoded(binding))
    elif body:
        raise ValueError('unencrypted room proof body')
    else:
        plaintext = b''
    token = _b64encode(payload) + '.' + _b64encode(key)
    return token, key, supplied, envelope, plaintext


def response_proof(key, request_mac, status, body, nonce):
    return _mac(key, b'hermes-room-response-v2', [request_mac, status, nonce, hashlib.sha256(body).hexdigest()])


def seal_response(key, request_mac, status, body):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = secrets.token_bytes(12)
    aad = b'hermes-room-response-body-v2\0' + _encoded([request_mac, status])
    wire = AESGCM(_body_key(key, b'response')).encrypt(nonce, body, aad)
    encoded_nonce = _b64encode(nonce)
    return wire, encoded_nonce, response_proof(key, request_mac, status, wire, encoded_nonce)


def verify_response(key, request_mac, status, body, supplied, nonce):
    if not isinstance(nonce, str) or not isinstance(supplied, str) or not hmac.compare_digest(
            supplied, response_proof(key, request_mac, status, body, nonce)):
        raise ValueError('peer response did not prove the pinned installation')
    return _decrypt(_body_key(key, b'response'), nonce, body,
                    b'hermes-room-response-body-v2\0' + _encoded([request_mac, status]))
