"""Native group compose consumer of the existing authenticated Send operation."""

import asyncio
import time
from dataclasses import dataclass

from gateway.native_reply_input import text


@dataclass(frozen=True)
class ComposeTarget:
    runner: object
    backend: object
    adapter: object
    profile: str
    source_key: str
    reference: str
    room_key: tuple


async def begin(menu):
    from gateway.hosted_room_messaging_files import label, _room_key

    room = await menu.fresh_room()
    target = ComposeTarget(menu.runner, menu.backend, menu.adapter, menu.profile,
                           menu.source_key, menu.reference, _room_key(room))
    previous = getattr(menu, "compose_request", None)
    if previous is not None:
        await asyncio.to_thread(previous.cancel)
    try:
        request, sent = await menu.adapter.send_reply_input(
            menu.event,
            text("title", group=label(room.get("name") or menu.reference)),
            lambda event, request: submit(target, event, request),
        )
    except Exception:
        return text("unavailable")
    if not sent.success:
        return text("unavailable")
    try:
        if _room_key(await menu.fresh_room()) != target.room_key:
            raise PermissionError("denied")
    except Exception:
        await asyncio.to_thread(request.cancel)
        return text("closed")
    menu.compose_request = request
    return await menu.room_page()


async def submit(target, event, request):
    from gateway.hosted_rooms import AuthorityConflictError
    from gateway.group_home_consent import _disclosure_stamp
    from gateway.group_chat_work import GroupChatMaintenanceError, run_group_command_work
    from gateway.hosted_room_messaging import (
        RoomControlError, list_messaging_rooms, messaging_event_id, resolve_room, send_to_room,
    )
    from gateway.hosted_room_messaging_files import _authorized, _room_key
    from gateway.platforms.base import MessageType

    runner = target.runner
    try:
        _authorized(runner, event, target.source_key, target.adapter)
        if (
            event.message_type != MessageType.TEXT
            or not event.text.strip()
            or len(event.text) > 4096
            or getattr(event.source, "message_had_attachments", False)
            or event.media_urls or event.media_types
        ):
            return text("text_only")
        stamp = _disclosure_stamp(runner, event)
        if stamp is None:
            return text("closed")
        event_id = messaging_event_id(event)

        def mutate():
            _authorized(runner, event, target.source_key, target.adapter)
            rooms = list_messaging_rooms(target.backend, profile=target.profile)
            room = resolve_room(rooms, target.reference)
            if _room_key(room) != target.room_key or stamp != _disclosure_stamp(runner, event):
                return text("closed")
            if room.get("_room_mode") == "desktop" and str(room.get("room_id", "")).startswith("name:"):
                return text("closed")
            if request.deadline <= time.monotonic():
                return text("closed")
            _authorized(runner, event, target.source_key, target.adapter)
            previous = request.claim(event_id)
            if previous is not None:
                return previous
            denial = runner._group_chat_rate_limit_denial(event, action="send")
            if denial:
                request.finish(event_id, denial)
                return denial
            try:
                _authorized(runner, event, target.source_key, target.adapter)
                if stamp != _disclosure_stamp(runner, event):
                    raise PermissionError("denied")
                result = send_to_room(
                    target.backend, room, event, event.text,
                    expected_authority=(target.room_key[1], target.room_key[2]),
                )
            except (PermissionError, AuthorityConflictError):
                result = text("closed")
            except Exception:
                result = text("unknown")
            request.finish(event_id, result)
            return result

        result = await run_group_command_work(runner, "send", mutate)
        _authorized(runner, event, target.source_key, target.adapter)
        return result
    except GroupChatMaintenanceError as exc:
        return str(exc)
    except (PermissionError, RoomControlError):
        return text("closed")
