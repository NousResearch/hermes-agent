"""Private, room-scoped control API for participating RoomLink gateways."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway import hosted_room_controls, hosted_rooms
from gateway.hosted_room_messaging import MessagingRoomBackend


MAX_CONTROL_TEXT_CHARS = 64 * 1024
MAX_CONTROL_EVENTS = 5
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def _http_routes(self) -> list[tuple[str, str, Any]]:
    from gateway.platforms import api_server_room_control_files

    async def read(request):
        from gateway.platforms import api_server

        return await _handle_room_control_read(
            self,
            request,
            _openai_error=api_server._openai_error,
        )

    async def mutate(request):
        from gateway.platforms import api_server

        return await _handle_room_control_mutate(
            self,
            request,
            _openai_error=api_server._openai_error,
        )

    async def revoke(request):
        from gateway.platforms import api_server

        return await _handle_room_control_revoke(
            self,
            request,
            _openai_error=api_server._openai_error,
        )

    return [
        ("GET", "/v1/room-controls/{room_id}", read),
        ("POST", "/v1/room-controls/{room_id}", mutate),
        ("DELETE", "/v1/room-controls/{room_id}", revoke),
    ] + api_server_room_control_files._http_routes(self)


def _control_token(request: "web.Request") -> str:
    authorization = str(request.headers.get("Authorization") or "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "hermesroomcontrol":
        return ""
    return token.strip()


def _identifier(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"invalid {label}")
    return normalized


def _authorize(request: "web.Request") -> tuple[dict[str, Any], str]:
    try:
        room_id = _identifier(request.match_info.get("room_id"), label="room_id")
        member_id = _identifier(
            request.headers.get("X-Hermes-Room-Member"),
            label="room member",
        )
    except ValueError as exc:
        raise PermissionError("room control scope is missing") from exc
    token = _control_token(request)
    if not token:
        raise PermissionError("room control credential is missing")
    try:
        room = hosted_rooms.room_state(
            hosted_rooms.default_db_path(),
            room_id=room_id,
        )
    except hosted_rooms.RoomNotFoundError as exc:
        raise PermissionError("room control credential is invalid") from exc
    if not hosted_room_controls.verify_home_control_token(
        hosted_rooms.default_db_path(),
        room_id=room_id,
        member_id=member_id,
        authority_gateway_id=str(room["authority_gateway_id"]),
        authority_epoch=int(room["authority_epoch"]),
        control_token=token,
    ):
        raise PermissionError("room control credential is invalid")
    return room, member_id


def _backend() -> MessagingRoomBackend:
    from tui_gateway.methods_groups import get_hosted_room_service

    service = get_hosted_room_service()
    return MessagingRoomBackend(
        db_path=(
            service.db_path if service is not None else hosted_rooms.default_db_path()
        ),
        service=service,
    )


def _visible_events(room_id: str) -> list[dict[str, Any]]:
    state = hosted_rooms.room_state(hosted_rooms.default_db_path(), room_id=room_id)
    delta = hosted_rooms.read_events(
        hosted_rooms.default_db_path(),
        room_id=room_id,
        since_seq=max(0, int(state.get("latest_seq") or 0) - 80),
        limit=80,
    )
    visible: list[dict[str, Any]] = []
    for event in delta.get("events", []):
        if not isinstance(event, dict) or event.get("kind") not in {
            "message.member",
            "message.user",
        }:
            continue
        raw_actor = event.get("actor")
        actor = raw_actor if isinstance(raw_actor, Mapping) else {}
        raw_payload = event.get("payload")
        payload = raw_payload if isinstance(raw_payload, Mapping) else {}
        visible.append({
            "kind": event["kind"],
            "actor": {
                "id": str(actor.get("id") or "")[:256],
                "display_name": str(actor.get("display_name") or "")[:128],
            },
            "payload": {
                "member_id": str(payload.get("member_id") or "")[:256],
                "text": str(payload.get("text") or "")[:MAX_CONTROL_TEXT_CHARS],
            },
        })
    return visible[-MAX_CONTROL_EVENTS:]


def _summary(room: Mapping[str, Any], backend: MessagingRoomBackend) -> dict[str, Any]:
    room_id = str(room["room_id"])
    raw_status = backend.status(room_id)
    raw_counts = raw_status.get("counts")
    counts = raw_counts if isinstance(raw_counts, Mapping) else {}
    members = []
    for raw_member in list(room.get("members") or []):
        if not isinstance(raw_member, Mapping):
            continue
        members.append({
            "member_id": str(raw_member.get("member_id") or "")[:256],
            "handle": str(raw_member.get("handle") or "")[:128],
            "display_name": str(raw_member.get("display_name") or "")[:128],
        })
    return {
        "room": {
            "room_id": room_id,
            "name": str(room.get("name") or room_id),
            "members": members,
            "authority_gateway_id": str(room["authority_gateway_id"]),
            "authority_epoch": int(room["authority_epoch"]),
            "latest_seq": int(room.get("latest_seq") or 0),
        },
        "status": {
            "working": raw_status.get("working") is True,
            "blocked": raw_status.get("blocked") is True,
            "counts": {
                status: int(counts.get(status) or 0)
                for status in (
                    "queued",
                    "running",
                    "stopping",
                    "deferred",
                    "indeterminate",
                    "settled",
                    "failed",
                    "cancelled",
                )
                if int(counts.get(status) or 0) > 0
            },
        },
        "events": _visible_events(room_id),
    }


def _error_response(_openai_error, message: str, *, status: int, code: str):
    return web.json_response(
        _openai_error(message, code=code),
        status=status,
    )


async def _handle_room_control_read(
    self,
    request: "web.Request",
    *,
    _openai_error,
) -> "web.Response":
    try:
        room, _member_id = _authorize(request)
        result = _summary(room, _backend())
    except PermissionError:
        return _error_response(
            _openai_error,
            "Room control is unavailable or expired.",
            status=401,
            code="invalid_room_control",
        )
    except hosted_rooms.RoomNotFoundError:
        return _error_response(
            _openai_error,
            "Group Chat not found.",
            status=404,
            code="room_not_found",
        )
    except Exception:
        return _error_response(
            _openai_error,
            "Group Chat status could not be loaded.",
            status=409,
            code="room_control_unavailable",
        )
    return web.json_response(result)


async def _handle_room_control_mutate(
    self,
    request: "web.Request",
    *,
    _openai_error,
) -> "web.Response":
    from gateway.platforms.api_server import _reserve_pending_api_work

    try:
        _authorize(request)
        with _reserve_pending_api_work(self):
            body, body_error = await self._read_json_body(request)
            if body_error:
                return body_error
            # Body reads can yield while authorization or drain state changes.
            room, member_id = _authorize(request)
            if not isinstance(body, Mapping):
                raise ValueError("room control body must be an object")
            allowed = {"action", "actor_display_name", "command_id", "text"}
            if set(body) - allowed or not {"action", "command_id"} <= set(body):
                raise ValueError("room control fields are invalid")
            action = str(body.get("action") or "").casefold()
            if action in {"send", "retry"}:
                draining = self._draining_response()
                if draining is not None:
                    return draining
            command_id = _identifier(body.get("command_id"), label="command_id")
            room_id = str(room["room_id"])
            backend = _backend()
            display_name = str(body.get("actor_display_name") or "Messaging").strip()
            display_name = re.sub(r"\s+", " ", display_name)[:128] or "Messaging"

            if action == "send":
                text = str(body.get("text") or "").strip()
                if not text or len(text) > MAX_CONTROL_TEXT_CHARS:
                    raise ValueError("Group Chat message is empty or too large")
                event = backend.send(
                    room_id=room_id,
                    event_id=command_id,
                    payload={"text": text, "thread_id": command_id},
                    actor={
                        "kind": "user",
                        "id": f"peer:{member_id}",
                        "display_name": display_name,
                    },
                )
                result = {"action": "send", "event": event}
            elif action == "stop":
                cancelled = backend.stop_room(room_id, cancel_id=f"control:{command_id}")
                result = {"action": "stop", "cancelled": int(cancelled)}
            elif action == "retry":
                pending = [
                    str(item.get("task_id") or "")
                    for item in backend.status(room_id).get("pending_actions", [])
                    if isinstance(item, Mapping)
                    and item.get("kind") == "retry"
                    and str(item.get("task_id") or "")
                ][:8]
                plan = hosted_room_controls.begin_control_retry(
                    hosted_rooms.default_db_path(),
                    command_id=command_id,
                    room_id=room_id,
                    member_id=member_id,
                    task_ids=pending,
                )
                if plan.result is not None:
                    result = plan.result
                else:
                    result = {
                        "action": "retry",
                        "queued": True,
                        "retried": len(plan.task_ids),
                    }
            else:
                raise ValueError("room control action must be send, retry, or stop")
            return web.json_response({**result, "summary": _summary(room, backend)})
    except PermissionError:
        return _error_response(
            _openai_error,
            "Room control is unavailable or expired.",
            status=401,
            code="invalid_room_control",
        )
    except hosted_rooms.RoomNotFoundError:
        return _error_response(
            _openai_error,
            "Group Chat not found.",
            status=404,
            code="room_not_found",
        )
    except (ValueError, hosted_room_controls.HostedRoomControlError) as exc:
        return _error_response(
            _openai_error,
            str(exc),
            status=400,
            code="invalid_room_control",
        )
    except Exception:
        return _error_response(
            _openai_error,
            "Group Chat control could not be applied.",
            status=409,
            code="room_control_unavailable",
        )


async def _handle_room_control_revoke(
    self,
    request: "web.Request",
    *,
    _openai_error,
) -> "web.Response":
    try:
        room_id = _identifier(request.match_info.get("room_id"), label="room_id")
        member_id = _identifier(
            request.headers.get("X-Hermes-Room-Member"),
            label="room member",
        )
        token = _control_token(request)
        if not token:
            raise PermissionError("room control credential is missing")
        revoked = hosted_room_controls.revoke_home_control_token_value(
            hosted_rooms.default_db_path(),
            room_id=room_id,
            member_id=member_id,
            control_token=token,
        )
    except (PermissionError, ValueError, hosted_room_controls.HostedRoomControlError):
        return _error_response(
            _openai_error,
            "Room control is unavailable or expired.",
            status=401,
            code="invalid_room_control",
        )
    except Exception:
        return _error_response(
            _openai_error,
            "Group Chat control could not be revoked.",
            status=409,
            code="room_control_unavailable",
        )
    return web.json_response({"revoked": int(revoked)})
