"""``POST /v1/room-members/replica``: opt-in passive history, authenticated by a room grant."""

from __future__ import annotations

import asyncio

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway import hosted_rooms as rooms
from gateway.platforms.api_server_room_grants import _grant_db
from gateway import hosted_room_replicas as replicas
from gateway.hosted_room_peer import HostedRoomGrantError
from gateway.hosted_room_replica_ingress import ingest_granted_page

# Pages are bounded UTF-8 JSON; keep framing and roster headroom.
MAX_REPLICA_HTTP_BYTES = 2 * (rooms.MAX_LOG_PAGE_BYTES + rooms.MAX_MEMBERS_JSON_BYTES) + 512 * 1024


def http_routes(adapter):
    def failure(message, code, status):
        from gateway.platforms.api_server import _openai_error
        return web.json_response(_openai_error(message, code=code), status=status)

    async def receive(request):
        from gateway.platforms import api_server
        from gateway.platforms.api_server_room_grants import _local_target, _room_grant_error_response

        try:
            claims = adapter._room_grant_claims(request, permission="replicate")
            profile, installation_id = _local_target(claims, api_server._api_request_profile)
        except Exception as exc:
            return _room_grant_error_response(exc, _openai_error=api_server._openai_error)
        body, error = await adapter._read_json_body((request if request.get("verified_room_grant") else request.clone(client_max_size=MAX_REPLICA_HTTP_BYTES)))
        if error is not None:
            return error
        if not {"room_id", "room_name", "members", "page"} <= set(body) <= {
                "room_id", "room_name", "members", "page", "custody"}:
            return failure("Invalid Group Chat history page.", "invalid_room_replica", 400)
        try:
            result = await asyncio.to_thread(
                ingest_granted_page, _grant_db(adapter),
                token=adapter._room_grant_token(request), secret=adapter._room_grant_secret(),
                target_install_id=installation_id, target_profile=profile, **body)
        except HostedRoomGrantError as exc:
            return _room_grant_error_response(exc, _openai_error=api_server._openai_error)
        except replicas.ReplicaCapacityError:
            return failure("Group Chat history storage is full on this gateway.", "room_replica_storage_full", 507)
        except replicas.ReplicaError as exc:
            code = "room_replica_gap" if isinstance(exc, replicas.ReplicaGapError) else "invalid_room_replica"
            return failure("Group Chat history could not be accepted.", code, 409)
        except (rooms.HostedRoomError, ValueError, TypeError):
            return failure("Invalid Group Chat history page.", "invalid_room_replica", 400)
        return web.json_response({"object": "hermes.room_member.replica", **result})

    return [("POST", "/v1/room-members/replica", receive)]
