"""Canonical RoomLink proof handling and durable, exact grant-refresh receipts."""
import hashlib
import json
import time

from aiohttp import web

from gateway import hosted_room_proof as proof, hosted_rooms

_REFRESH = '/v1/room-members/grants/refresh'
_MAX_ISSUANCE_RECEIPTS = 4096


def wrap(adapter, handler):
    async def handle(request):
        header = request.headers.get('Authorization', '')
        if not header.startswith(proof.SCHEME):
            return await handler(request)
        # aiohttp's application body limit is enforced by read before proof decoding.
        body = await request.read()
        try:
            token, key, request_mac, envelope = proof.verify_request(
                header, secret=adapter._room_grant_secret(),
                installation_id=hosted_rooms.local_authority_gateway_id(),
                method=request.method, path=request.raw_path, body=body, headers=request.headers)
        except Exception:
            return web.json_response({'error': {'code': 'invalid_room_proof'}}, status=401)
        request['verified_room_grant'] = token
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
                    conn.execute('DELETE FROM hosted_room_grant_refresh_receipts WHERE expires_at<=?', (time.time(),))
                    row = conn.execute('SELECT * FROM hosted_room_grant_refresh_receipts WHERE request_key=?',
                                       (cache_key,)).fetchone()
                    if row is not None:
                        if row['fingerprint'] != fingerprint:
                            raise ValueError('room proof replay changed its request')
                        request['room_proof_issued_at'] = row['issued_at']
                        if row['body'] is not None:
                            response = web.Response(body=bytes(row['body']), status=row['status'],
                                                    content_type='application/json')
                            response.headers[proof.RESPONSE_HEADER] = proof.response_proof(
                                key, request_mac, response.status, response.body)
                            return response
                    else:
                        count = conn.execute('SELECT COUNT(*) FROM hosted_room_grant_refresh_receipts').fetchone()[0]
                        if count >= _MAX_ISSUANCE_RECEIPTS:
                            raise ValueError('room refresh receipt capacity exhausted')
                        request['room_proof_issued_at'] = time.time()
                        conn.execute('''INSERT INTO hosted_room_grant_refresh_receipts(
                            request_key,fingerprint,issued_at,expires_at) VALUES (?,?,?,?)''',
                            (cache_key, fingerprint, request['room_proof_issued_at'], claims['status_expires_at']))
                request['room_proof_request_id'] = envelope['request_id']
            response = await handler(request)
            if not isinstance(response, web.Response):
                raise ValueError('room proof requires a bounded response')
            if cache_key is not None and response.status < 300:
                with hosted_rooms._transaction(db_path, immediate=True) as conn:
                    conn.execute('UPDATE hosted_room_grant_refresh_receipts SET status=?,body=? WHERE request_key=?',
                                 (response.status, response.body, cache_key))
        except Exception as exc:
            from gateway.platforms.api_server_room_grants import RoomGrantReauthorizationRequired
            if isinstance(exc, RoomGrantReauthorizationRequired):
                response = web.json_response({'error': {'code': 'room_reauthorization_required'}}, status=403)
            else:
                # An exception may occur after accepting work or losing its durable reply.
                # It is uncertainty, never proof that the target did not admit the request.
                response = web.json_response({'error': {'code': 'room_proof_outcome_unknown'}}, status=503)
        response.headers[proof.RESPONSE_HEADER] = proof.response_proof(
            key, request_mac, response.status, response.body or b'')
        return response
    return handle
