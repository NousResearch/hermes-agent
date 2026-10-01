"""``POST /v1/room-members/work-records``: opt-in passive task evidence, never admission or recovery."""

import asyncio

try:
    from aiohttp import web
except ImportError:
    web = None

from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from gateway.platforms.api_server_room_grants import _grant_db
from gateway.hosted_room_peer import HostedRoomGrantError


def http_routes(adapter):
    def failure(message, code, status):
        from gateway.platforms.api_server import _openai_error
        return web.json_response(_openai_error(message, code=code), status=status)

    async def receive(request):
        from gateway.platforms import api_server
        from gateway.platforms.api_server_room_grants import _local_target, _room_grant_error_response
        try:
            claims = adapter._room_grant_claims(request, permission=records.PERMISSION)
            profile, installation_id = _local_target(claims, api_server._api_request_profile)
        except Exception as exc:
            return _room_grant_error_response(exc, _openai_error=api_server._openai_error)
        body, error = await adapter._read_json_body((request if request.get("verified_room_grant") else request.clone(client_max_size=records.MAX_BYTES + 1024)))
        if error is not None:
            return error
        try:
            if set(body) != {"record"}:
                raise records.WorkRecordError("work record request fields are invalid")
            result = await asyncio.to_thread(
                records.ingest, _grant_db(adapter), record=body["record"],
                token=adapter._room_grant_token(request), secret=adapter._room_grant_secret(),
                target_install_id=installation_id, target_profile=profile)
        except HostedRoomGrantError as exc:
            return _room_grant_error_response(exc, _openai_error=api_server._openai_error)
        except records.WorkRecordCapacityError:
            return failure("Passive work-record storage is full.", "work_records_storage_full", 507)
        except records.WorkRecordPrefixError:
            return failure("Matching history is required.", "work_records_prefix", 409)
        except (records.WorkRecordError, rooms.HostedRoomError, ValueError, TypeError):
            return failure("Passive work records were rejected.", "invalid_work_records", 409)
        return web.json_response({"object": "hermes.room_member.work_records", **result})

    return [("POST", "/v1/room-members/work-records", receive)]
