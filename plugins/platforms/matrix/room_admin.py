"""Matrix administration with current session, actor and client checks."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote

from gateway.config import Platform
from gateway.session_context import get_session_env
from gateway.session_identity import RoutingIdentity
from plugins.platforms.matrix.admin_selection import MatrixAdminSelection
from plugins.platforms.matrix.client_events import Method, raw_event
from plugins.platforms.matrix.room_inspection import (
    _InspectionRejected, _InspectionRoomScope, _is_missing_state, _permission_levels,
    _RoomCreate, _state_path, change_matrix_pin,
)
from tools.matrix_tool_runtime import MatrixOwner

_ACTIONS = frozenset({"create", "invite", "leave", "forget", "redact"})

UNKNOWN_OUTCOME_STEPS = {
    "create": "Ask the user whether the room was created before retrying, so that no duplicate room is created",
    "invite": "Ask the user whether the invite arrived before retrying",
    "leave": "Ask the user whether the bot is still in the room before retrying",
    "forget": "Forget changes only the bot's account, so retrying it is harmless",
    "redact": "Read the event with matrix_read kind=event before retrying",
}


@dataclass(frozen=True)
class _AdminContext:
    owner: MatrixOwner
    interrupted: Callable[[], bool]
    operation: str

    @classmethod
    def capture(cls, adapter: Any, interrupted: Callable[[], bool], operation: str) -> _AdminContext:
        owner = MatrixOwner.capture()
        if owner.adapter is not adapter:
            raise ValueError("Matrix session or client ownership changed")
        return cls(owner, interrupted, operation)

    @property
    def adapter(self) -> Any:
        return self.owner.adapter

    @property
    def client(self) -> Any:
        return self.owner.client

    @property
    def crypto(self) -> Any:
        return self.owner.crypto

    @property
    def bot(self) -> str:
        return self.owner.bot or ""

    @property
    def identity(self) -> RoutingIdentity | None:
        return self.owner.identity

    @property
    def room(self) -> str:
        return self.owner.session[1]

    @property
    def actor(self) -> str:
        return self.owner.session[2]

    def check(self, chat_type: str, *, joined: bool = True) -> RoutingIdentity:
        if self.owner.session[0] != "matrix" or not self.room or not self.actor:
            raise ValueError("Matrix administration requires a live Matrix session")
        identity = self.identity
        if identity is None or identity.adapter() is not self.adapter:
            raise ValueError("Matrix administration requires live routing provenance")
        if self.interrupted():
            raise ValueError(f"{self.operation} interrupted")
        runner = getattr(self.adapter, "gateway_runner", None)
        if (not self.owner.matches() or getattr(self.adapter, "_closing", False)
                or self.owner.home != identity.authorization_home
                or runner is not None
                and runner._adapters_for_profile(identity.transport_profile).get(Platform.MATRIX) is not self.adapter):
            raise ValueError("Matrix session or client ownership changed")
        if joined and self.room not in self.adapter._joined_rooms:
            raise ValueError("Matrix room is not joined")
        allowed = self.adapter._allowed_room_ids
        if allowed and self.room not in allowed and chat_type != "dm":
            raise ValueError("Matrix room is not allowed")
        if self.adapter._is_sender_authorized(self.actor, chat_type=chat_type, chat_id=self.room) is not True:
            raise ValueError("Matrix requester is not authorized for this room")
        return identity

    async def require_selection(self, chat_type: str, *, joined: bool = True) -> MatrixAdminSelection:
        identity = self.check(chat_type, joined=joined)
        homes = tuple(dict.fromkeys((identity.runtime_home, identity.authorization_home)))
        selection = await asyncio.to_thread(MatrixAdminSelection.capture, homes)
        self.check(chat_type, joined=joined)
        return selection

    async def access(self, chat_type: str | None = None, *, joined: bool = True) -> str:
        expected = chat_type or get_session_env("HERMES_SESSION_CHAT_TYPE") or "group"
        self.check(expected, joined=joined)
        allowed = await self.adapter._is_allowed_matrix_room_event(self.room)
        self.check(expected, joined=joined)
        if not allowed:
            raise ValueError("Matrix room is not allowed")
        current = "dm" if await self.adapter._is_dm_room(self.room) else "group"
        self.check(current, joined=joined)
        if chat_type is not None and current != chat_type:
            raise ValueError("Matrix room changed between a direct chat and a group room")
        return current


async def _get(context: _AdminContext, path: str, query: dict[str, str] | None = None) -> Any:
    return await asyncio.wait_for(context.client.api.request(Method.GET, path, query_params=query), timeout=10.0)


async def _state(context: _AdminContext, event_type: str, state_key: str = "") -> dict[str, Any] | None:
    try:
        value = await _get(context, _state_path(context.room, event_type, state_key))
    except Exception as exc:
        if _is_missing_state(exc):
            return None
        raise
    return value if isinstance(value, dict) else {}


async def _member(context: _AdminContext, user: str) -> str | None:
    return (await _state(context, "m.room.member", user) or {}).get("membership")


async def _permissions(
    context: _AdminContext, chat_type: str, *, joined: bool,
) -> dict[str, Any]:
    encryption = await _state(context, "m.room.encryption") or {}
    create = _RoomCreate.parse(await _get(context, _state_path(context.room, "m.room.create"), {"format": "event"}))
    access_started = time.monotonic()
    await context.access(chat_type, joined=joined)
    room_scope = _InspectionRoomScope.capture(
        context.adapter, context.room, refreshed_since=access_started,
    )
    power = await _state(context, "m.room.power_levels")
    context.check(chat_type, joined=joined)
    try:
        room_scope.check(context.adapter, context.room)
    except _InspectionRejected as exc:
        raise ValueError("Matrix room is not allowed") from exc
    return _permission_levels(context.actor, context.bot, power, encryption, create)


def _validate(args: dict[str, Any]) -> None:
    if args.get("action") not in _ACTIONS:
        raise ValueError("action must be create, invite, leave, forget, or redact")
    for key in ("name", "topic", "reason"):
        if key in args and not isinstance(args[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("encrypted", "is_direct"):
        if key in args and not isinstance(args[key], bool):
            raise ValueError(f"{key} must be a boolean")
    invitees = args.get("invite", [])
    if (not isinstance(invitees, list) or len(invitees) > 50
            or any(not isinstance(user, str) or re.fullmatch(r"@[^\s:]+:[^\s]+", user) is None for user in invitees)):
        raise ValueError("invite must contain at most 50 full Matrix user IDs")
    if args["action"] == "invite" and (
        not isinstance(args.get("user_id"), str)
        or re.fullmatch(r"@[^\s:]+:[^\s]+", args["user_id"]) is None
    ):
        raise ValueError("user_id must be a full Matrix user ID")
    if args["action"] == "redact" and (
        not isinstance(args.get("event_id"), str) or not args["event_id"].startswith("$")
        or len(args["event_id"]) < 2
    ):
        raise ValueError("event_id must be a Matrix event ID")
    if args.get("is_direct") and len(invitees) != 1:
        raise ValueError("is_direct requires exactly one invitee")
    if args.get("preset", "private_chat") != "private_chat":
        raise ValueError("Only private rooms can be created")


_REQUIREMENTS = {
    "invite": ("invite users", ("invite",)),
    "remove": ("remove the bot from this room", ("kick",)),
    "redact_own": ("redact this event", ("send_redaction",)),
    "redact_other": ("redact this event", ("send_redaction", "redact_other")),
}


async def _refusal(
    context: _AdminContext, requirement: str, chat_type: str, *, joined: bool,
) -> dict[str, Any] | None:
    permissions = await _permissions(context, chat_type, joined=joined)
    context.check(chat_type, joined=joined)
    requester = permissions["requester"]
    purpose, keys = _REQUIREMENTS[requirement]
    required = max(permissions["required"][key] for key in keys)
    if requester["creator_override"] or requester["level"] is not None and requester["level"] >= required:
        return None
    return {"error": f"Matrix requester lacks permission to {purpose}", "required": required,
            "level": requester["level"]}


async def _requester_is_only_other_member(context: _AdminContext, chat_type: str, *, joined: bool) -> bool:
    members = await context.adapter._get_room_members(context.room)
    context.check(chat_type, joined=joined)
    return members is not None and set(members) - {context.bot} == {context.actor}


async def administer_matrix_room(
    adapter: Any, args: dict[str, Any], *, interrupt_check: Callable[[], bool], before_write: Callable[[], None],
) -> dict[str, Any]:
    sent = False
    selection: MatrixAdminSelection | None = None

    def send() -> None:
        nonlocal sent
        context.check(chat_type, joined=action != "forget")
        assert selection is not None
        selection.require_current()
        before_write()
        sent = True

    try:
        context = _AdminContext.capture(adapter, interrupt_check, "Matrix administration")
        _validate(args)
        action = args["action"]
        if context.client is None:
            raise ValueError("Matrix client is disconnected")
        await context.require_selection(get_session_env("HERMES_SESSION_CHAT_TYPE") or "group",
                                        joined=action != "forget")
        chat_type = await context.access(joined=action != "forget")
        if await _member(context, context.actor) != "join":
            raise ValueError("Matrix requester is not a joined room member")
        context.check(chat_type, joined=action != "forget")
        bot_membership = await _member(context, context.bot)
        context.check(chat_type, joined=action != "forget")
        if bot_membership != ("leave" if action == "forget" else "join"):
            raise ValueError("Matrix bot must have left before forget" if action == "forget" else "Matrix bot is not joined")
        if action == "create" and args.get("encrypted") and context.crypto is None:
            raise ValueError("Encrypted room creation requires the owning client's active crypto store")
        if action == "redact":
            event = raw_event(await _get(
                context, f"/_matrix/client/v3/rooms/{quote(context.room, safe='')}/event/{quote(args['event_id'], safe='')}",
            ))
            context.check(chat_type)
            if event.get("event_id") != args["event_id"] or event.get("room_id") != context.room:
                raise ValueError("Matrix event does not belong to the current room")
        requirement = {"invite": "invite", "leave": "remove", "forget": "remove"}.get(action)
        if action == "redact":
            requirement = "redact_own" if event.get("sender") == context.actor else "redact_other"
        if action in {"leave", "forget"} and await _requester_is_only_other_member(
            context, chat_type, joined=action != "forget",
        ):
            requirement = None
        await context.access(chat_type, joined=action != "forget")
        if action == "create" and (unauthorised := [
            user for user in args.get("invite", []) if user != context.actor and not adapter._is_authorized_user(user)
        ]):
            return {"error": "Matrix room creation can invite only the requester and authorised users",
                    "unauthorised": unauthorised}
        if requirement is not None and (refusal := await _refusal(
            context, requirement, chat_type, joined=action != "forget",
        )) is not None:
            return refusal
        await context.access(chat_type, joined=action != "forget")
        for _attempt in range(2):
            selection = await context.require_selection(chat_type, joined=action != "forget")
            await context.access(chat_type, joined=action != "forget")
            if requirement is not None and (refusal := await _refusal(
                context, requirement, chat_type, joined=action != "forget",
            )) is not None:
                return refusal
            if selection.current():
                break
        selection.require_current()
        result = await _MUTATIONS[action](context, args, send)
        try:
            context.check(chat_type, joined=action not in {"leave", "forget"})
        except ValueError as exc:
            return {**result, "warning": str(exc)}
        return result
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        errcode = getattr(exc, "errcode", None)
        if sent and errcode is None:
            return {"error": "Matrix administration failed after the change was sent to the homeserver",
                    "outcome": "unknown", "next_step": UNKNOWN_OUTCOME_STEPS[args["action"]]}
        result = {"error": f"Matrix administration failed: {type(exc).__name__}"}
        if errcode:
            result.update(errcode=str(errcode), message=str(exc))
        return result


async def _create(
    context: _AdminContext, args: dict[str, Any], before_write: Callable[[], None],
) -> dict[str, Any]:
    from mautrix.types import RoomCreatePreset, UserID

    initial_state = [{"type": "m.room.encryption", "state_key": "",
                      "content": {"algorithm": "m.megolm.v1.aes-sha2"}}] if args.get("encrypted") else []
    before_write()
    created = await context.client.create_room(
        name=args.get("name") or None, topic=args.get("topic") or None,
        invitees=[UserID(user) for user in args.get("invite", [])],
        is_direct=args.get("is_direct", False), preset=RoomCreatePreset.PRIVATE,
        initial_state=initial_state,
    )
    if args.get("is_direct"):
        await context.adapter._record_dm_room(str(created), args["invite"][0])
    return {"action": "create", "room_id": str(created), "encrypted": args.get("encrypted", False)}


async def _invite(
    context: _AdminContext, args: dict[str, Any], before_write: Callable[[], None],
) -> dict[str, Any]:
    from mautrix.types import RoomID, UserID

    before_write()
    await context.client.invite_user(RoomID(context.room), UserID(args["user_id"]), reason=args.get("reason") or None)
    return {"action": "invite", "room_id": context.room, "user_id": args["user_id"]}


async def _redact(
    context: _AdminContext, args: dict[str, Any], before_write: Callable[[], None],
) -> dict[str, Any]:
    from mautrix.types import EventID, RoomID

    before_write()
    redaction = await context.client.redact(
        RoomID(context.room), EventID(args["event_id"]), reason=args.get("reason") or None,
    )
    return {"action": "redact", "room_id": context.room, "event_id": args["event_id"],
            "redaction_event_id": str(redaction)}


async def _leave(
    context: _AdminContext, args: dict[str, Any], before_write: Callable[[], None],
) -> dict[str, Any]:
    from mautrix.types import RoomID

    before_write()
    try:
        await context.client.leave_room(RoomID(context.room), reason=args.get("reason") or None, raise_not_in_room=True)
    except BaseException as exc:
        # Only a homeserver refusal proves that the bot is still joined. Sync adds a room
        # back to the cache when a leave that may have been sent did not happen.
        if getattr(exc, "errcode", None) is None:
            context.adapter._joined_rooms.discard(context.room)
        raise
    context.adapter._joined_rooms.discard(context.room)
    return {"action": "leave", "room_id": context.room}


async def _forget(
    context: _AdminContext, args: dict[str, Any], before_write: Callable[[], None],
) -> dict[str, Any]:
    from mautrix.types import RoomID

    before_write()
    await context.client.forget_room(RoomID(context.room))
    return {"action": "forget", "room_id": context.room}


_MUTATIONS = {"create": _create, "invite": _invite, "redact": _redact, "leave": _leave, "forget": _forget}


async def administer_matrix_pin(
    adapter: Any, action: str, room_id: str, event_id: str, *, requester: str,
    interrupt_check: Callable[[], bool], before_write: Callable[[], None],
) -> dict[str, Any]:
    try:
        context = _AdminContext.capture(adapter, interrupt_check, "Matrix pin update")
    except ValueError as exc:
        return {"error": str(exc)}

    async def recheck_before_write() -> MatrixAdminSelection:
        await context.access(chat_type)
        if await _member(context, context.actor) != "join":
            raise ValueError("Matrix requester is not a joined room member")
        context.check(chat_type)
        await context.access(chat_type)
        return await context.require_selection(chat_type)

    try:
        if room_id != context.room or requester != context.actor:
            raise ValueError("Matrix pin actions are limited to the current actor and room")
        await context.require_selection(get_session_env("HERMES_SESSION_CHAT_TYPE") or "group")
        chat_type = await context.access()

        def interrupted() -> bool:
            context.check(chat_type)
            return context.interrupted()

        return await change_matrix_pin(
            adapter, action, room_id, event_id, requester=requester,
            interrupt_check=interrupted, before_write=before_write,
            recheck_before_write=recheck_before_write, expected_client=context.client,
        )
    except ValueError as exc:
        return {"error": str(exc)}
