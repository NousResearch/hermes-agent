"""``POST /v1/group-replicas/retire``: a home's one-purpose capability retires this gateway's copy.

An Ed25519 signature authorizes exactly the copy and generation enrolled by this gateway's
operator. The target stores only its public verifier. Outer Room proof protects transport and
authenticates the installation even after the original member grant expires or is revoked;
it grants no execution permission. Served at the installation endpoint, never a profile prefix.
"""

from __future__ import annotations

import asyncio

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway import hosted_rooms as rooms
from gateway.platforms.api_server_room_grants import _grant_db
from gateway import hosted_room_replica_retirement as retirement

MAX_RETIREMENT_REQUEST_BYTES = 16 * 1024


def http_routes(adapter):
    def failure(message, code, status):
        from gateway.platforms.api_server import _openai_error
        return web.json_response(_openai_error(message, code=code), status=status)

    async def retire(request):
        from gateway.platforms.api_server import _api_request_profile
        if request.match_info.get("profile") or _api_request_profile.get() not in {None, "default"}:
            return failure("Use this gateway's installation endpoint.", "installation_endpoint_required", 400)
        proof_grant = request.get('verified_room_grant')
        if not isinstance(proof_grant, str):
            return failure("Copy retirement requires pinned installation proof.", "replica_retirement_not_authorized", 403)
        body, denied = await adapter._read_json_body(request)
        if denied is not None:
            return denied
        try:
            from gateway.hosted_room_peer import unverified_room_grant_claims
            if set(body) != {'notice', 'signature'} or not isinstance(body['notice'], dict):
                raise retirement.RetirementError('invalid signed retirement request')
            payload, value = body['notice'], body['signature']
            # The wrapper proved this payload belongs to this installation's secret. Its
            # expired/revoked grant is not redeemed: only the inner enrollment signature
            # authorizes retiring this exact destination copy.
            claims = unverified_room_grant_claims(proof_grant)
            if (claims.get('target_install_id') != rooms.local_authority_gateway_id()
                    or any(claims.get(k) != payload.get(k) for k in
                           ('room_id', 'authority_gateway_id', 'authority_epoch', 'target_install_id'))):
                raise retirement.RetirementAuthorizationError('retirement installation proof scope differs')
            result = await asyncio.to_thread(
                retirement.retire_copy, _grant_db(adapter), payload=payload, value=value,
                local_gateway_id=rooms.local_authority_gateway_id())
        except retirement.RetirementAuthorizationError:
            return failure("Copy retirement is not authorized.", "replica_retirement_not_authorized", 403)
        except retirement.RetirementConflictError:
            return failure("The Group Chat copy or its enrollment has changed.", "replica_retirement_conflict", 409)
        except (rooms.HostedRoomError, ValueError, TypeError):
            return failure("Invalid copy retirement request.", "invalid_replica_retirement", 400)
        return web.json_response(result)

    return [("POST", "/v1/group-replicas/retire", retire)]
