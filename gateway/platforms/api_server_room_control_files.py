"""Reciprocal control reads restricted to published, recipient-eligible files."""

from __future__ import annotations

import asyncio
import base64
import json

from aiohttp import web

from gateway.hosted_room_file_contract import (
    FILE_TIMEOUT_SECONDS,
    FileAccessError,
    MAX_CATALOG_BYTES,
    MAX_FILE_BYTES,
    MAX_QUERY_BYTES,
    MAX_RECEIPT_CHARS,
    catalog_options,
    identifier,
    scope,
)


def _http_routes(adapter):
    async def catalog(request):
        return await _handle(request, binary=False)

    async def read(request):
        return await _handle(request, binary=True)

    async def shared_message(request):
        return await _handle(request, binary=False, shared_message=True)

    async def resolve(request):
        return await _handle(request, binary=False, lookup="file")

    async def latest(request):
        return await _handle(request, binary=False, lookup="reply")

    return [
        ("GET", "/v1/room-controls/{room_id}/files", catalog),
        ("GET", "/v1/room-controls/{room_id}/files/resolve", resolve),
        ("GET", "/v1/room-controls/{room_id}/files/{attachment_id}", read),
        ("GET", "/v1/room-controls/{room_id}/latest-reply", latest),
        ("GET", "/v1/room-controls/{room_id}/messages/{event_id}", shared_message),
    ]


def _authorize(request):
    from gateway import hosted_rooms
    from gateway.hosted_room_file_access import _require_viewer_room
    from gateway.platforms.api_server_room_controls import _authorize as authorize

    room, member_id = authorize(request)
    if room["authority_gateway_id"] != hosted_rooms.local_authority_gateway_id():
        raise FileAccessError("file_access_denied")
    try:
        _require_viewer_room(hosted_rooms.default_db_path(), room)
    except Exception:
        raise FileAccessError("file_access_denied") from None
    member = next(
        item for item in room["members"] if item.get("member_id") == member_id
    )
    target = member.get("target") or {}
    profile = (
        target.get("target_profile") or target.get("profile") or member.get("profile")
    )
    if target.get("kind") != "peer":
        raise FileAccessError("file_access_denied")
    expected = scope(room, member_id, profile)
    if any(
        str(request.headers.get(header) or "") != str(expected[field])
        for header, field in (
            ("X-Hermes-Room-Profile", "target_profile"),
            ("X-Hermes-Room-Authority", "authority_gateway_id"),
            ("X-Hermes-Room-Epoch", "authority_epoch"),
        )
    ):
        raise FileAccessError("file_access_denied")
    return room, expected


def _options(request, binary):
    if request.can_read_body or len(request.raw_path.encode("utf-8")) > MAX_QUERY_BYTES:
        raise FileAccessError("file_invalid_request")
    allowed = (
        {"event_id", "max_bytes"}
        if binary
        else {"cursor", "limit", "query", "producer_member_id"}
    )
    if set(request.query) - allowed or any(
        len(request.query.getall(key)) != 1 for key in request.query
    ):
        raise FileAccessError("file_invalid_request")
    options = dict(request.query)
    integer = "max_bytes" if binary else "limit"
    if integer in options:
        try:
            options[integer] = int(options[integer])
        except ValueError:
            raise FileAccessError("file_invalid_request") from None
    if binary:
        identifier(options.get("event_id"))
        options.setdefault("max_bytes", MAX_FILE_BYTES)
        return options
    return catalog_options(options)


async def _operation(request, binary, shared_message=False):
    from gateway.hosted_room_file_access import list_local_files, read_local_file
    from gateway.platforms.api_server_room_controls import _backend

    room, expected = await asyncio.to_thread(_authorize, request)
    if shared_message:
        from gateway.hosted_room_shared_message_access import (
            MAX_REPLY_RESPONSE_BYTES,
            read_local_shared_message,
        )

        if request.query or request.can_read_body:
            raise FileAccessError("file_invalid_request")
        backend = await asyncio.to_thread(_backend)
        reply = await asyncio.to_thread(
            read_local_shared_message,
            backend,
            room=room,
            member_id=expected["member_id"],
            event_id=request.match_info["event_id"],
        )
        _fresh_room, fresh = await asyncio.to_thread(_authorize, request)
        if fresh != expected:
            raise FileAccessError("file_access_denied")
        payload = json.dumps(
            {"scope": expected, "reply": reply}, ensure_ascii=False
        ).encode("utf-8")
        if len(payload) > MAX_REPLY_RESPONSE_BYTES:
            raise FileAccessError("file_too_large")
        return web.Response(
            body=payload,
            content_type="application/json",
            headers={"Cache-Control": "no-store"},
        )
    options = _options(request, binary)
    backend = await asyncio.to_thread(_backend)
    if binary:
        stored = await asyncio.to_thread(
            read_local_file,
            backend,
            room=room,
            member_id=expected["member_id"],
            selected_id=request.match_info["attachment_id"],
            **options,
        )
        fresh_room, fresh = await asyncio.to_thread(_authorize, request)
        if fresh != expected:
            raise FileAccessError("file_access_denied")
        receipt = base64.urlsafe_b64encode(
            json.dumps(
                {"scope": expected, "attachment": stored.attachment},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).decode("ascii")
        if len(receipt) > MAX_RECEIPT_CHARS:
            raise FileAccessError("file_invalid_response")
        return web.Response(
            body=stored.data,
            content_type=stored.attachment["mime"],
            headers={
                "X-Hermes-Room-File": receipt,
                "Content-Disposition": "attachment",
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
    page = await asyncio.to_thread(
        list_local_files,
        backend,
        room=room,
        member_id=expected["member_id"],
        **options,
    )
    _fresh_room, fresh = await asyncio.to_thread(_authorize, request)
    if fresh != expected:
        raise FileAccessError("file_access_denied")
    payload = json.dumps({**page, "scope": expected}, ensure_ascii=False).encode(
        "utf-8"
    )
    if len(payload) > MAX_CATALOG_BYTES:
        raise FileAccessError("file_too_large")
    return web.Response(
        body=payload,
        content_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


async def _lookup_operation(request, kind):
    from gateway.hosted_room_file_contract import FileCodeAmbiguous
    from gateway.hosted_room_file_lookup import (
        latest_local_reply,
        resolve_local_file,
        validate_code,
    )
    from gateway.hosted_room_shared_message_access import MAX_REPLY_RESPONSE_BYTES
    from gateway.platforms.api_server_room_controls import _backend

    room, expected = await asyncio.to_thread(_authorize, request)
    if request.can_read_body or len(request.raw_path.encode("utf-8")) > MAX_QUERY_BYTES:
        raise FileAccessError("file_invalid_request")
    backend = await asyncio.to_thread(_backend)
    if kind == "file":
        if set(request.query) != {"code"} or len(request.query.getall("code")) != 1:
            raise FileAccessError("file_invalid_request")
        field, maximum = "selection", MAX_CATALOG_BYTES
        try:
            result = await asyncio.to_thread(
                resolve_local_file,
                backend,
                room=room,
                member_id=expected["member_id"],
                code=validate_code(request.query["code"]),
            )
        except FileCodeAmbiguous as exc:
            result, field = list(exc.matches), "matches"
    else:
        if request.query:
            raise FileAccessError("file_invalid_request")
        result = await asyncio.to_thread(
            latest_local_reply, backend, room=room, member_id=expected["member_id"]
        )
        field, maximum = "reply", MAX_REPLY_RESPONSE_BYTES
    _room, fresh = await asyncio.to_thread(_authorize, request)
    if fresh != expected:
        raise FileAccessError("file_access_denied")
    payload = json.dumps({"scope": expected, field: result}, ensure_ascii=False).encode(
        "utf-8"
    )
    if len(payload) > maximum:
        raise FileAccessError("file_too_large")
    return web.Response(
        body=payload,
        content_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


async def _handle(request, *, binary, shared_message=False, lookup=None):
    try:
        return await asyncio.wait_for(
            _lookup_operation(request, lookup)
            if lookup
            else _operation(request, binary, shared_message),
            FILE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        error = FileAccessError("file_timeout", retryable=True)
    except FileAccessError as exc:
        error = exc
    except ImportError:
        error = FileAccessError("file_access_unsupported")
    except PermissionError:
        error = FileAccessError("file_access_denied")
    except Exception:
        error = FileAccessError("file_access_denied")
    status = {
        "file_access_denied": 403,
        "file_access_unsupported": 501,
        "file_invalid_request": 400,
        "file_unavailable": 404,
        "file_too_large": 413,
        "file_timeout": 504,
    }.get(error.code, 409)
    return web.json_response(
        {
            "error": {
                "code": error.code,
                "message": str(error),
                "retryable": error.retryable,
            }
        },
        status=status,
        headers={"Cache-Control": "no-store"},
    )
