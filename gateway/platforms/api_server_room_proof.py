"""Canonical RoomLink proof handling and durable, exact grant-refresh receipts."""
import hashlib
import json
import time
from collections.abc import MutableMapping

from aiohttp import web

from gateway import hosted_room_proof as proof, hosted_rooms

_REFRESH = '/v1/room-members/grants/refresh'
_MAX_ISSUANCE_RECEIPTS = 4096


class PlaintextRequest(MutableMapping):
    """Forward request identity/state while exposing only authenticated body bytes.

    The underlying aiohttp request is left untouched. Handler mapping writes remain
    on that request; transport, route, profile and session metadata are delegated.
    """
    def __init__(self, request, body, limit):
        self._request, self._body = request, body
        self.client_max_size = limit

    def __getattr__(self, name):
        return getattr(self._request, name)

    def __getitem__(self, key):
        return self._request[key]

    def __setitem__(self, key, value):
        self._request[key] = value

    def __delitem__(self, key):
        del self._request[key]

    def __iter__(self):
        return iter(self._request)

    def __len__(self):
        return len(self._request)

    @property
    def content_length(self):
        return len(self._body)

    @property
    def body_exists(self):
        return bool(self._body)

    @property
    def can_read_body(self):
        return bool(self._body)

    async def read(self):
        return self._body

    async def text(self):
        return self._body.decode('utf-8')

    async def json(self, *, loads=json.loads):
        return loads(await self.text())


def _seal(response, key, request_mac):
    wire, nonce, signature = proof.seal_response(key, request_mac, response.status, response.body or b'')
    response.body = wire
    response.headers.pop('Content-Length', None)
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers[proof.RESPONSE_HEADER] = signature
    response.headers[proof.RESPONSE_NONCE_HEADER] = nonce
    return response


def wrap(adapter, handler, *, max_bytes=None):
    async def handle(request):
        if len(request.headers.getall('Authorization', [])) > 1:
            return web.json_response({'error': {'code': 'invalid_room_proof'}}, status=401)
        header = request.headers.get('Authorization', '')
        if not header.startswith(proof.SCHEME):
            return await handler(request)
        plaintext_limit = request.client_max_size if max_bytes is None else max_bytes
        body_reader = request.clone(client_max_size=plaintext_limit + proof.WIRE_OVERHEAD)
        wire_body = await body_reader.read()
        try:
            token, key, request_mac, envelope, body = proof.verify_request(
                header, secret=adapter._room_grant_secret(),
                installation_id=hosted_rooms.local_authority_gateway_id(),
                method=request.method, path=request.raw_path, body=wire_body, headers=request.headers)
        except Exception:
            return web.json_response({'error': {'code': 'invalid_room_proof'}}, status=401)
        if len(body) > plaintext_limit:
            return _seal(web.json_response({'error': {'code': 'body_too_large'}}, status=413), key, request_mac)
        request = PlaintextRequest(request, body, plaintext_limit)
        request['verified_room_grant'] = token
        if request.path.endswith('/events'):
            return _seal(web.json_response({'error': {'code': 'room_proof_streaming_unsupported'}}, status=400),
                         key, request_mac)
        cache_key = None
        try:
            if request.method == 'POST' and request.path == _REFRESH:
                # A retired/expired grant cannot redeem a cached successor.
                claims = adapter._room_grant_claims(request, permission='dispatch')
                from gateway.platforms.api_server_room_grants import _grant_db
                db_path = _grant_db(adapter)
                cache_key = hashlib.sha256((token + '\0' + envelope['request_id']).encode()).hexdigest()
                fingerprint = hashlib.sha256(request.method.encode() + b'\0' + request.raw_path.encode()
                                             + b'\0' + body).hexdigest()
                with hosted_rooms._transaction(db_path, immediate=True) as conn:
                    from gateway.platforms.api_server_room_grants import _room_grant_claims
                    _room_grant_claims(adapter, request, permission='dispatch', conn=conn)
                    conn.execute('DELETE FROM hosted_room_grant_refresh_receipts WHERE expires_at<=?', (time.time(),))
                    row = conn.execute('SELECT * FROM hosted_room_grant_refresh_receipts WHERE request_key=?',
                                       (cache_key,)).fetchone()
                    if row is not None:
                        if row['retired_at'] is not None:
                            raise ValueError('room refresh issuance was retired')
                        if row['fingerprint'] != fingerprint:
                            raise ValueError('room proof replay changed its request')
                        request['room_proof_issued_at'] = row['issued_at']
                        if row['body'] is not None:
                            response = web.Response(body=bytes(row['body']), status=row['status'],
                                                    content_type='application/json')
                            return _seal(response, key, request_mac)
                    else:
                        count = conn.execute('SELECT COUNT(*) FROM hosted_room_grant_refresh_receipts').fetchone()[0]
                        if count >= _MAX_ISSUANCE_RECEIPTS:
                            raise ValueError('room refresh receipt capacity exhausted')
                        request['room_proof_issued_at'] = time.time()
                        conn.execute('''INSERT INTO hosted_room_grant_refresh_receipts(
                            request_key,fingerprint,request_body,issued_at,expires_at) VALUES (?,?,?,?,?)''',
                            (cache_key, fingerprint, body.decode('utf-8'), request['room_proof_issued_at'], claims['status_expires_at']))
                request['room_proof_request_id'] = envelope['request_id']
            response = await handler(request)
            if not isinstance(response, web.Response):
                raise ValueError('room proof requires a bounded response')
            if cache_key is not None and response.status < 300:
                with hosted_rooms._transaction(db_path, immediate=True) as conn:
                    updated = conn.execute('UPDATE hosted_room_grant_refresh_receipts SET status=?,body=? WHERE request_key=? AND retired_at IS NULL',
                                           (response.status, response.body, cache_key))
                    if updated.rowcount != 1:
                        raise ValueError('room refresh issuance was retired during the request')
        except Exception as exc:
            from gateway.platforms.api_server_room_grants import RoomGrantReauthorizationRequired
            if isinstance(exc, RoomGrantReauthorizationRequired):
                response = web.json_response({'error': {'code': 'room_reauthorization_required'}}, status=403)
            else:
                # An exception may occur after accepting work or losing its durable reply.
                # It is uncertainty, never proof that the target did not admit the request.
                response = web.json_response({'error': {'code': 'room_proof_outcome_unknown'}}, status=503)
        return _seal(response, key, request_mac)
    return handle


async def cleanup_issuance(adapter, request):
    """Status-only cleanup of one issuance; never redeem a successor after dispatch expiry."""
    from gateway.hosted_room_peer import decode_room_grant
    from gateway.platforms.api_server_room_grants import _grant_db, _hard_expiry, _local_target
    from gateway.platforms.api_server import _api_request_profile
    body = await request.json()
    if (set(body) != {'request_id'} or not isinstance(body['request_id'], str)
            or len(body['request_id']) != 64 or any(c not in '0123456789abcdef' for c in body['request_id'])):
        return web.json_response({'error': {'code': 'invalid_issuance_cleanup'}}, status=400)
    token = adapter._room_grant_token(request)
    try:
        claims = decode_room_grant(adapter._room_grant_secret(), token, permission='status',
                                   allow_expired_for_revocation=True)
        _local_target(claims, _api_request_profile)
    except ValueError:
        return web.json_response({'error': {'code': 'invalid_room_grant'}}, status=401)
    if _hard_expiry(claims) <= time.time():
        return web.json_response({'revoked': True, 'expired': True})
    # Exact old-grant revocation after successful custody prevents killing its installed successor.
    adapter._room_grant_claims(request, permission='status')
    key = hashlib.sha256((token + '\0' + body['request_id']).encode()).hexdigest()
    db_path = _grant_db(adapter)
    with hosted_rooms._transaction(db_path, immediate=True) as conn:
        from gateway.platforms.api_server_room_grants import _room_grant_claims
        _room_grant_claims(adapter, request, permission='status', conn=conn)
        row = conn.execute('SELECT * FROM hosted_room_grant_refresh_receipts WHERE request_key=?', (key,)).fetchone()
        if row is None:
            # Fence an issuance whose delayed original request has not arrived yet.
            conn.execute('DELETE FROM hosted_room_grant_refresh_receipts WHERE expires_at<=?', (time.time(),))
            if conn.execute('SELECT COUNT(*) FROM hosted_room_grant_refresh_receipts').fetchone()[0] >= _MAX_ISSUANCE_RECEIPTS:
                return web.json_response({'error': {'code': 'issuance_cleanup_capacity'}}, status=503)
            conn.execute("""INSERT INTO hosted_room_grant_refresh_receipts(
                request_key,fingerprint,issued_at,expires_at,retired_at) VALUES (?,'',?,?,?)""",
                (key, time.time(), _hard_expiry(claims), time.time()))
        else:
            conn.execute('UPDATE hosted_room_grant_refresh_receipts SET retired_at=? WHERE request_key=?',
                         (time.time(), key))
        # Minting has no external effect before the receipt commits. A pending receipt uses the
        # same deterministic ID/time as its retry, so its exact successor is reconstructible.
        if row is not None and (row['body'] is not None or row['request_body'] is not None):
            from gateway.hosted_room_peer import issue_room_grant, room_grant_token_digest
            if row['body'] is not None:
                successor = json.loads(row['body'])['grant']
            else:
                from gateway.platforms.api_server_room_grants import _room_identity
                from gateway.hosted_room_peer import MAX_DISPATCH_GRANT_TTL_SECONDS
                request_body = json.loads(row['request_body'])
                issued_at = row['issued_at']
                successor = issue_room_grant(adapter._room_grant_secret(),
                    grant_id='grant-refresh-' + body['request_id'], **_room_identity(claims),
                    target_install_id=claims['target_install_id'], target_profile=claims['target_profile'],
                    execution_policy_digest=claims['execution_policy_digest'], permissions=claims['permissions'],
                    issued_at=issued_at, ttl_seconds=min(float(request_body.get('ttl_seconds', MAX_DISPATCH_GRANT_TTL_SECONDS)),
                        MAX_DISPATCH_GRANT_TTL_SECONDS, _hard_expiry(claims) - issued_at),
                    status_expires_at=_hard_expiry(claims))
            successor_claims = decode_room_grant(adapter._room_grant_secret(), successor, permission='status',
                                                 allow_expired_for_revocation=True)
            hosted_rooms.revoke_room_grant_token(db_path, claims=successor_claims,
                token_sha256=room_grant_token_digest(successor), expires_at=_hard_expiry(successor_claims), conn=conn)
    return web.json_response({'revoked': True})
