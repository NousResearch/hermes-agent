"""Opt-in automatic routing and Continue for hosted Group Chats (pure: no I/O, no model knowledge).

A room whose settings say ``routing: auto`` sends an unaddressed human message through one cheap, tool-less
model call (``tui_gateway.hosted_room_router_call``) that picks the member(s) it is for. The deterministic
rules still come first and never reach the router: an @mention or @all/@everyone, a scheduled post, and a
message inside an existing thread (a hover Reply prefixes the member's handle, so it is a mention). The
answer is written to the room log as ``room.routed`` BEFORE any turn is planned, so the planner stays a pure
function of the log and a restart re-plans identically without asking again. A router that fails or answers
something invalid records a fallback to everyone (the ``everyone`` room behaviour).

``room.continued`` is the user's Continue on a discussion that hit its cap (``room.activity`` ``bounded``):
each one grants the same discussion one more budget of rounds and messages and reopens it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.hosted_room_discussion import DiscussionMember, DiscussionRoom, _ValidatedEvent

ROUTED_EVENT_KIND = "room.routed"
CONTINUED_EVENT_KIND = "room.continued"
ROUTED_FIELDS = frozenset({"discussion_event_id", "thread_id", "member_ids", "everyone", "fallback"})
CONTINUED_FIELDS = frozenset({"discussion_event_id", "thread_id", "grant"})
#: Continues one discussion may take; each one is another full budget of rounds and messages.
MAX_DISCUSSION_CONTINUES = 5
#: Recent room messages the router sees, and the characters kept from each.
ROUTER_RECENT_MESSAGES = 5
ROUTER_MESSAGE_CHARS = 600
ROUTE_PENDING = object()


def routed_event_id(discussion_event_id: str) -> str:
    return f"droute:{discussion_event_id}"


def routed_event(events: Sequence[_ValidatedEvent], discussion_event_id: str) -> _ValidatedEvent | None:
    return next((event for event in events if event.kind == ROUTED_EVENT_KIND
                 and event.payload.get("discussion_event_id") == discussion_event_id), None)


def routed_responders(
    room: DiscussionRoom, discussion: _ValidatedEvent, events: Sequence[_ValidatedEvent], context: Mapping[str, str],
) -> tuple[DiscussionMember, ...] | object | None:
    """Round-0 responders the router picked; ``ROUTE_PENDING`` while the message still waits for its
    ``room.routed``; None when this message is not routed (not an auto room, or a deterministic rule applies)."""
    from gateway.hosted_room_lead import routes_through_lead

    if context.get("routing") != "auto" or not routes_through_lead(discussion, room, events):
        return None
    if (event := routed_event(events, discussion.event_id)) is None:
        return ROUTE_PENDING
    if event.payload["everyone"]:
        return tuple(room.members)
    chosen = set(event.payload["member_ids"])
    return tuple(member for member in room.members if member.member_id in chosen)


def continue_count(events: Sequence[_ValidatedEvent], discussion_event_id: str) -> int:
    return sum(1 for event in events if event.kind == CONTINUED_EVENT_KIND
               and event.payload.get("discussion_event_id") == discussion_event_id)


def resume_round(events: Sequence[_ValidatedEvent], discussion_event_id: str,
                 member_messages: Sequence[_ValidatedEvent]) -> int | None:
    """The round after the last reply before the latest Continue (None without one): the room resumes there."""
    latest = max((event.seq for event in events if event.kind == CONTINUED_EVENT_KIND
                  and event.payload.get("discussion_event_id") == discussion_event_id), default=None)
    if latest is None:
        return None
    return 1 + max((int(event.payload["round_index"]) for event in member_messages if event.seq < latest), default=-1)


def continued_through(events: Sequence[_ValidatedEvent]) -> dict[str, int]:
    """discussion_event_id -> seq of its latest ``room.continued``: activity before it no longer completes it."""
    latest: dict[str, int] = {}
    for event in events:
        if event.kind == CONTINUED_EVENT_KIND:
            latest[str(event.payload["discussion_event_id"])] = event.seq
    return latest


def validate_routed_event(kind: str, payload: Mapping[str, Any], actor: Mapping[str, Any],
                          room: DiscussionRoom) -> Mapping[str, Any]:
    from gateway.hosted_room_discussion import (
        DiscussionValidationError, _exact_fields, _identifier, _require_gateway_actor)

    _exact_fields(payload, label=f"{kind} payload", required=ROUTED_FIELDS)
    _require_gateway_actor(actor, room, f"{kind} requires the room gateway")
    for field in ("discussion_event_id", "thread_id"):
        _identifier(payload.get(field), label=field)
    member_ids, everyone = payload.get("member_ids"), payload.get("everyone")
    roster = {member.member_id for member in room.members}
    if (not isinstance(everyone, bool) or not isinstance(payload.get("fallback"), bool)
            or not isinstance(member_ids, list) or len(set(member_ids)) != len(member_ids)
            or not set(member_ids) <= roster or (everyone and member_ids) or (not everyone and not member_ids)):
        raise DiscussionValidationError(f"{kind} must name seated members or everyone")
    return payload


def validate_continued_event(kind: str, payload: Mapping[str, Any], actor: Mapping[str, Any],
                             room: DiscussionRoom) -> Mapping[str, Any]:
    from gateway.hosted_room_discussion import (
        _bounded_int, _exact_fields, _identifier, _require_gateway_actor)

    _exact_fields(payload, label=f"{kind} payload", required=CONTINUED_FIELDS)
    _require_gateway_actor(actor, room, f"{kind} requires the room gateway")
    for field in ("discussion_event_id", "thread_id"):
        _identifier(payload.get(field), label=field)
    _bounded_int(payload.get("grant"), message=f"{kind} grant must be a positive integer", low=1,
                 high=MAX_DISCUSSION_CONTINUES)
    return payload


def paused_discussion(events: Sequence[Mapping[str, Any]], discussion_event_id: str) -> dict[str, Any] | None:
    """``{"thread_id", "grant"}`` when the discussion's latest outcome is the cap (``room.activity``
    ``bounded`` after its last Continue) and neither a Stop nor a newer message in its thread followed;
    else None. ``grant`` is the number the next Continue takes (1-based)."""
    bounded = None
    grants = 0
    for event in events:
        payload = event.get("payload") or {}
        if payload.get("discussion_event_id") == discussion_event_id:
            if event.get("kind") == CONTINUED_EVENT_KIND:
                grants, bounded = grants + 1, None
            elif event.get("kind") == "room.activity":
                bounded = event if payload.get("status") == "bounded" else None
    if bounded is None:
        return None
    thread_id = bounded["payload"].get("thread_id")
    if any(int(event["seq"]) > int(bounded["seq"]) and (
            event.get("kind") == "room.stop_requested"
            or (event.get("kind") == "message.user" and (event.get("payload") or {}).get("thread_id") == thread_id))
           for event in events):
        return None
    return {"thread_id": thread_id, "grant": grants + 1}


def routed_payload(*, discussion_event_id: str, thread_id: str, member_ids: Sequence[str] | None) -> dict[str, Any]:
    """The ``room.routed`` payload for one router answer: None (failed or invalid) falls back to everyone."""
    return {"discussion_event_id": discussion_event_id, "thread_id": thread_id,
            "member_ids": list(member_ids or ()), "everyone": not member_ids, "fallback": member_ids is None}


# -- the router's question and its strict answer --------------------------------------------------

ROUTER_SYSTEM_PROMPT = (
    "You route messages in a group chat between a user and several AI assistants. Decide which members "
    "the user's new message is for: pick each member whose role or area the request touches, and only "
    "those. Pick everyone only when the message is clearly for the whole group (a greeting, a question to "
    "all, a request every member has a part in). Never pick a member only to acknowledge or repeat another. "
    'Reply with only a JSON object and nothing else: {"members": ["handle", ...]} or {"members": "everyone"}.')


def router_prompt(*, room_name: str, purpose: str, notes: str, roster: Sequence[Mapping[str, str]],
                  recent: Sequence[Mapping[str, str]], message: str, attachments: Sequence[str] = ()) -> str:
    """The router's user message: the room, its members (handle, name, role) and the last few messages.
    Everything quoted is data, never instructions."""
    lines = [f"Room: {room_name}"]
    if purpose:
        lines.append(f"Purpose: {purpose}")
    if notes:
        lines.append(f"Notes: {notes[:2000]}")
    lines += ["", "Members:"]
    for member in roster:
        role = " ".join(str(member.get("description") or "").split())[:300]
        name = member.get("name") or ""
        lines.append(f"- @{member['handle']}" + (f" ({name})" if name and name != member["handle"] else "")
                     + (f": {role}" if role else ""))
    if recent:
        lines += ["", "Recent messages (oldest first):"]
        lines += [f"  {item['author']}: {_one_line(item['text'])}" for item in recent]
    lines += ["", "The user's new message:", f"  {message[:20000]}"]
    if attachments:
        lines.append("Attached files: " + ", ".join(attachments))
    return "\n".join(lines)


def _one_line(text: str) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= ROUTER_MESSAGE_CHARS else flat[:ROUTER_MESSAGE_CHARS] + "..."


def parse_router_answer(text: Any, handles: Mapping[str, str]) -> tuple[str, ...] | None:
    """Member ids for a strict ``{"members": [...]}`` answer (``()`` for ``"everyone"``); None when the
    answer is not exactly that object, or names anyone outside the roster. ``handles`` maps casefolded
    handle -> member id."""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        answer = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(answer, dict) or set(answer) != {"members"}:
        return None
    picked = answer["members"]
    if picked == "everyone":
        return ()
    if not isinstance(picked, list) or not picked or not all(isinstance(item, str) for item in picked):
        return None
    keys = [item.strip().lstrip("@").casefold() for item in picked]
    if len(set(keys)) != len(keys) or any(key not in handles for key in keys):
        return None
    return tuple(handles[key] for key in keys)
