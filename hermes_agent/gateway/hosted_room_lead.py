"""Opt-in lead routing for hosted Group Chats.

A room whose settings say ``routing: lead`` sends an unaddressed human message to its lead first. The lead
runs one routing turn on its own member session and answers with the handles that should take the message;
the gateway records that as a ``room.assignment`` event (the Desktop renders "Ada gave this to Linus") and
only the chosen members take the normal round. Anything the lead answers that names no roster member wakes
the lead alone: a routing answer can never broadcast.

Everything else keeps the room's normal resolution: an @mention or @all/@everyone, a scheduled post, a
reply inside an existing thread, and every message in an ``everyone`` room (the default).

The routing turn is an ordinary driver task whose payload carries ``lead_routing: true``. Only the planner's
lead round sets it; the driver accepts it only as exactly true, and it is hashed into the task id (the prompt
is not: it names the room, and a rename must not orphan the receipt the planner looks up by that id), so a
persisted task without it (every task admitted before lead routing existed, and every member turn, including
batch-bounded ones whose ``seen_through_seq`` sits below their source message) is an ordinary member turn.
Its ``seen_through_seq`` is the message's seq minus one, so its terminal receipt never advances the lead's
watermark past the message the lead still has to answer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gateway.hosted_room_discussion import (
        DiscussionMember, DiscussionRoom, DiscussionTaskPlan, EventPlan, _ValidatedEvent)

ASSIGNMENT_EVENT_KIND = "room.assignment"
ASSIGNMENT_FIELDS = frozenset(
    {"discussion_event_id", "thread_id", "task_id", "lead_member_id", "member_ids", "fallback"})
_ROUTING_INSTRUCTIONS = (
    "You are the lead of this room. A message arrived that names nobody, so you decide who takes it: pick the "
    "member(s) whose role the request touches (yourself included when it is yours). Do not answer the message "
    "itself here; whoever you pick answers it next.",
    'Reply with only a JSON object and nothing else, for example {"assign": ["handle"]}.')


def lead_member(room: DiscussionRoom, context: Mapping[str, str]) -> DiscussionMember | None:
    """The room's lead when lead routing is on and the lead is still seated, else None (everyone mode)."""
    if context.get("routing") != "lead":
        return None
    return next((member for member in room.members if member.member_id == context.get("lead_member_id")), None)


def routes_through_lead(discussion: _ValidatedEvent, room: DiscussionRoom, events: Sequence[_ValidatedEvent]) -> bool:
    """A human message that opens a new thread and addresses nobody goes to the lead first."""
    from gateway.hosted_room_context import scheduled_post_kind
    from gateway.hosted_room_discussion import _MENTION_RE

    if scheduled_post_kind(discussion.actor) is not None:
        return False
    handles = {member.handle.casefold() for member in room.members} | {"all", "everyone"}
    if any(match.group(1).casefold() in handles for match in _MENTION_RE.finditer(str(discussion.payload["text"]))):
        return False
    thread_id = discussion.payload["thread_id"]
    return not any(
        event.seq < discussion.seq and event.kind in {"message.user", "message.member"}
        and event.payload.get("thread_id") == thread_id for event in events)


def is_routing_task(task: DiscussionTaskPlan) -> bool:
    return task.payload.get("lead_routing") is True


def routing_prompt(
    *, room: DiscussionRoom, lead: DiscussionMember, discussion: _ValidatedEvent, context_lines: Sequence[str]
) -> str:
    roster = [f"- @{member.handle}" + (f" ({member.display_name})" if member.display_name else "")
              + (" (you)" if member.member_id == lead.member_id else "") for member in room.members]
    return "\n".join([
        f'[Discussion: "{room.name}"] You are @{lead.handle}, the lead of this room.', "",
        *context_lines, "Members:", *roster, "",
        "The message (from the user):", f"  {discussion.payload['text']}", *_attachment_lines(discussion), "",
        *_ROUTING_INSTRUCTIONS])


def _attachment_lines(discussion: _ValidatedEvent) -> list[str]:
    """Name and kind of each attached file: the lead routes on them (an attachment-only message has no text);
    the files themselves reach only the member who takes the message."""
    files = discussion.payload.get("attachments") or ()
    return ["Attached files:", *(f"  - {' '.join(str(item.get('name') or 'file').split())} ({item.get('kind')}"
                                 + (f", {item['mime']}" if item.get("mime") else "") + ")" for item in files)] if files else []


def _first_assign_object(text: str) -> Mapping[str, Any] | None:
    """The first balanced JSON object in ``text`` carrying an ``assign`` key. A lead often wraps its answer
    in prose that has braces of its own (``see {notes}``), so a greedy outer match is not the answer."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text, start)
        except ValueError:
            value = None
        if isinstance(value, Mapping) and "assign" in value:
            return value
        start = text.find("{", start + 1)
    return None


def parse_assignment(text: Any, room: DiscussionRoom) -> tuple[str, ...]:
    """Member ids the lead's answer names, in roster order; empty when it names none (or is not JSON)."""
    answer = _first_assign_object(str(text or ""))
    picked = answer.get("assign") if answer is not None else None
    if not isinstance(picked, list):
        return ()
    handles = {str(item).strip().lstrip("@").casefold() for item in picked if isinstance(item, str)}
    return tuple(member.member_id for member in room.members if member.handle.casefold() in handles)


def assignment(events: Sequence[_ValidatedEvent], discussion_event_id: str) -> _ValidatedEvent | None:
    return next((event for event in events if event.kind == ASSIGNMENT_EVENT_KIND
                 and event.payload.get("discussion_event_id") == discussion_event_id), None)


def first_round_members(
    room: DiscussionRoom, lead: DiscussionMember, events: Sequence[_ValidatedEvent], discussion_event_id: str,
    routing_task_id: str,
) -> tuple[DiscussionMember, ...] | None:
    """Who the lead gave the message to; ``(lead,)`` when its routing turn reached a terminal receipt
    without an assignment (cancelled: discarded or superseded); None while the routing turn has not
    finished (``turn.deferred`` is not terminal: it awaits Retry or Discard)."""
    if (event := assignment(events, discussion_event_id)) is not None:
        chosen = set(event.payload["member_ids"])
        return tuple(member for member in room.members if member.member_id in chosen)
    if any(event.payload.get("task_id") == routing_task_id for event in events if event.kind.startswith("turn.")
           and event.kind != "turn.deferred"):
        return (lead,)
    return None


def validate_assignment_event(kind: str, payload: Mapping[str, Any], actor: Mapping[str, Any],
                              room: DiscussionRoom) -> Mapping[str, Any]:
    from gateway.hosted_room_discussion import DiscussionValidationError, _exact_fields, _require_gateway_actor

    _exact_fields(payload, label=f"{kind} payload", required=ASSIGNMENT_FIELDS)
    _require_gateway_actor(actor, room, f"{kind} requires the room gateway")
    roster = {member.member_id for member in room.members}
    member_ids = payload.get("member_ids")
    if (not isinstance(member_ids, list) or not member_ids or not set(member_ids) <= roster
            or payload.get("lead_member_id") not in roster or not isinstance(payload.get("fallback"), bool)):
        raise DiscussionValidationError(f"{kind} must name seated members and a seated lead")
    return payload


def assignment_effect(result: Any, *, task: DiscussionTaskPlan, room: DiscussionRoom, status: str) -> list[EventPlan]:
    """The ``room.assignment`` a finished routing turn publishes ahead of its terminal receipt. A settled
    answer naming no member, or a failed routing turn, records a fallback to the lead alone. A cancelled
    (discarded or superseded) or deferred routing turn records no assignment; once it is terminal,
    ``first_round_members`` still gives the message to the lead, because Discard gives up one attempt, never
    the user's message (a discarded member turn in an everyone room leaves the rest of the round running).
    Stop ends the whole round through the stop fence instead."""
    from gateway.hosted_room_discussion import EventPlan, _terminal_text

    if status not in {"settled", "failed"}:
        return []
    chosen = parse_assignment(_terminal_text(result, field="text", fallback=""), room) if status == "settled" else ()
    digest = task.identity.task_id.removeprefix("dtask:")
    return [EventPlan(
        event_id=f"dassign:{digest}", kind=ASSIGNMENT_EVENT_KIND, actor={"kind": "gateway", "id": room.gateway_id},
        payload={"discussion_event_id": task.discussion_event_id, "thread_id": task.identity.thread_id,
                 "task_id": task.identity.task_id, "lead_member_id": task.member.member_id,
                 "member_ids": list(chosen or (task.member.member_id,)), "fallback": not chosen},
        authority_gateway_id=room.gateway_id, authority_epoch=room.authority_epoch)]
