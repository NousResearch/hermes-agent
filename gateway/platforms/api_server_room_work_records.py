"""Explicitly authorized passive work evidence, not admission or recovery."""

import asyncio

try:
    from aiohttp import web
except ImportError:
    web = None

from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from gateway.hosted_room_peer import HostedRoomGrantError


def http_routes(adapter):
    async def receive(request):
        from gateway.platforms import api_server
        from gateway.platforms.api_server_room_grants import _local_target, _room_grant_error_response
        try:
            claims = adapter._room_grant_claims(request, permission=records.PERMISSION)
            profile, installation_id = _local_target(claims, api_server._api_request_profile)
        except Exception as exc:
            return _room_grant_error_response(exc, _openai_error=api_server._openai_error)
        body, error = await adapter._read_json_body(request.clone(client_max_size=records.MAX_BYTES + 1024))
        if error is not None:
            return error
        try:
            if set(body) != {"record"}:
                raise records.WorkRecordError("work record request fields are invalid")
            result = await asyncio.to_thread(
                records.ingest, rooms.default_db_path(), record=body["record"],
                token=adapter._room_grant_token(request), secret=adapter._room_grant_secret(),
                target_install_id=installation_id, target_profile=profile)
        except HostedRoomGrantError as exc:
            return _room_grant_error_response(exc, _openai_error=api_server._openai_error)
        except records.WorkRecordCapacityError:
            return web.json_response(api_server._openai_error(
                "Passive work-record storage is full.", code="work_records_storage_full"), status=507)
        except records.WorkRecordPrefixError:
            return web.json_response(api_server._openai_error(
                "Matching history is required.", code="work_records_prefix"), status=409)
        except (records.WorkRecordError, rooms.HostedRoomError, ValueError, TypeError):
            return web.json_response(api_server._openai_error(
                "Passive work records were rejected.", code="invalid_work_records"), status=409)
        return web.json_response({"object": "hermes.room_member.work_records", **result})
    return [("POST", "/v1/room-members/work-records", receive)]
