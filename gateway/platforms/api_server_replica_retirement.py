"""Installation-scoped setup and one-purpose passive-copy retirement."""

from __future__ import annotations

import asyncio

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway import hosted_rooms as rooms
from gateway import hosted_room_replica_retirement as retirement

MAX_RETIREMENT_REQUEST_BYTES = 16 * 1024


def http_routes(adapter):
    def error(message, code, status):
        from gateway.platforms.api_server import _openai_error

        return web.json_response(_openai_error(message, code=code), status=status)

    async def handle(request, operation):
        from gateway.platforms.api_server import _api_request_profile

        if request.match_info.get("profile") or _api_request_profile.get() not in {
            None,
            "default",
        }:
            return error(
                "Use this gateway's installation endpoint.",
                "installation_endpoint_required",
                400,
            )
        if operation != "retire":
            if not adapter._expected_api_key():
                return adapter._auth_failed_response()
            denied = adapter._check_auth(request)
            if denied is not None:
                return denied
        body, denied = await adapter._read_json_body(
            request.clone(client_max_size=MAX_RETIREMENT_REQUEST_BYTES)
        )
        if denied is not None:
            return denied
        db_path = rooms.default_db_path()
        try:
            if operation == "enroll":
                if (
                    set(body)
                    - {"enrollment", "expected_enrollment_id", "expected_state", "authority_history"}
                    or "enrollment" not in body
                ):
                    raise retirement.RetirementError("invalid enrollment request")
                result = await asyncio.to_thread(
                    retirement.enroll_target,
                    db_path,
                    enrollment=body["enrollment"],
                    target_install_id=rooms.local_authority_gateway_id(),
                    expected_enrollment_id=body.get("expected_enrollment_id"),
                    expected_state=body.get("expected_state", "active"),
                    authority_history=body.get("authority_history"),
                )
            elif operation == "revoke":
                if set(body) != {"room_id", "enrollment_id"}:
                    raise retirement.RetirementError(
                        "invalid enrollment revocation request"
                    )
                result = await asyncio.to_thread(
                    retirement.revoke_target_enrollment, db_path, **body
                )
            else:
                scheme, separator, value = str(
                    request.headers.get("Authorization") or ""
                ).partition(" ")
                if not separator or scheme.lower() != "hermesreplicaretirement":
                    raise retirement.RetirementAuthorizationError(
                        "retirement capability is required"
                    )
                result = await asyncio.to_thread(
                    retirement.retire_copy,
                    db_path,
                    payload=body,
                    value=value.strip(),
                    local_gateway_id=rooms.local_authority_gateway_id(),
                )
        except retirement.RetirementAuthorizationError:
            return error(
                "Copy retirement is not authorized.",
                "replica_retirement_not_authorized",
                403,
            )
        except retirement.RetirementConflictError:
            return error(
                "The Group Chat copy or its enrollment has changed.",
                "replica_retirement_conflict",
                409,
            )
        except retirement.RetirementCapacityError:
            return error(
                "Copy retirement enrollment capacity is full.",
                "replica_retirement_capacity",
                507,
            )
        except (rooms.HostedRoomError, ValueError, TypeError):
            return error(
                "Invalid copy retirement request.", "invalid_replica_retirement", 400
            )
        return web.json_response(result)

    async def enroll(request):
        return await handle(request, "enroll")

    async def revoke(request):
        return await handle(request, "revoke")

    async def retire(request):
        return await handle(request, "retire")

    return [
        ("POST", "/v1/group-replicas/enroll", enroll),
        ("POST", "/v1/group-replicas/revoke-enrollment", revoke),
        ("POST", "/v1/group-replicas/retire", retire),
    ]
