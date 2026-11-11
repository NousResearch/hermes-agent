"""Room purpose, shared notes and scheduled-job posts for hosted Group Chats.

Both live in the append-only room log, never in a side file: the Discussion planner must stay a pure
function of durable room state, because a member turn's task id hashes its prompt. Purpose and notes are
read as of the message that opened a discussion, so editing them mid-discussion never re-plans a turn that
is already admitted; the edit applies from the next message. The same event carries the room's routing:
``everyone`` (every member gets a turn on an unaddressed message, the default) or ``lead`` (the lead picks
who takes it; ``gateway/hosted_room_lead.py``). The settings rules are ``session_rules.hosted_rooms``'s, which
the store applies inside its settings write.

Scheduled posts are ordinary ``message.user`` events whose server-owned actor id names the cron job. Only
this module mints that actor (``groups.send`` hard-codes ``desktop``), so a client cannot impersonate a job.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from gateway import hosted_rooms
from persistence.gateway.hosted_rooms import HostedPlacement
from session_rules.hosted_rooms import (
    CONTEXT_EVENT_KIND, MAX_NOTES_BYTES, MAX_PURPOSE_CHARS, ROUTING_MODES, context_from_payload, validate_context)

if TYPE_CHECKING:
    from persistence_assembly import Persistence

# Actor id prefixes for posts the scheduler makes. A result post can start a round when it @mentions a
# member; a failure notice never does (its text is machine-written, not addressed to anyone).
SCHEDULED_POST_PREFIX = "cron:"
SCHEDULED_NOTICE_PREFIX = "cron-notice:"


def _local_authority(store: Persistence, placement: HostedPlacement, room_id: Any) -> dict[str, Any]:
    state = hosted_rooms.room_state(store, placement=placement, room_id=room_id)
    if state["authority_gateway_id"] != hosted_rooms.local_authority_gateway_id():
        raise hosted_rooms.AuthorityConflictError("This Group Chat is managed by another gateway.")
    return state


def room_context(
    store: Persistence, *, placement: HostedPlacement, room_id: Any, before_seq: int | None = None,
) -> dict[str, str]:
    """The latest purpose/notes, or the ones in force before ``before_seq`` when given."""
    room_id = hosted_rooms._room_id(room_id)
    return context_from_payload(store.hosted_rooms.latest_event_payload(
        placement=placement, room_id=room_id, kind=CONTEXT_EVENT_KIND, before_seq=before_seq))


def set_room_context(
    store: Persistence, *, placement: HostedPlacement, room_id: Any, event_id: Any, purpose: Any, notes: Any,
    routing: Any = None, lead_member_id: Any = None, now: float | None = None,
) -> dict[str, Any]:
    """Record new room settings as one replayable room event. Omitted routing fields keep their current
    values, so a client that only edits purpose/notes never switches a lead room back to everyone. The store
    reads those values inside the settings write: a concurrent routing change cannot be written back stale.
    A retried save (same ``event_id``) fills them from its own first write instead, so it returns that event
    even after another client changed the routing meanwhile."""
    state = _local_authority(store, placement, room_id)
    seated = {str(member.get("member_id")) for member in state.get("members") or ()}
    return dict(store.hosted_rooms.save_room_settings(
        placement=placement, room_id=state["room_id"], event_id=hosted_rooms._event_id(event_id), purpose=purpose,
        notes=notes, routing=routing, lead_member_id=lead_member_id, seated_member_ids=seated,
        authority_gateway_id=state["authority_gateway_id"], authority_epoch=state["authority_epoch"],
        now=hosted_rooms._now(now)))


def scheduled_post_kind(actor: Mapping[str, Any] | None) -> str | None:
    """``"post"``/``"notice"`` for a scheduler-authored ``message.user`` actor, else None (the human)."""
    if not isinstance(actor, Mapping) or actor.get("kind") != "user":
        return None
    actor_id = str(actor.get("id") or "")
    if actor_id.startswith(SCHEDULED_NOTICE_PREFIX):
        return "notice"
    return "post" if actor_id.startswith(SCHEDULED_POST_PREFIX) else None


def post_scheduled_message(
    store: Persistence, *, placement: HostedPlacement, room_id: Any, job_id: str, job_name: str, run_key: str,
    text: str, notice: bool = False, now: float | None = None,
) -> dict[str, Any]:
    """Append one cron output to a room. Idempotent per ``run_key``: a retried delivery of the same run
    returns the original event instead of posting twice."""
    from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES, validate_user_payload

    state = _local_authority(store, placement, room_id)
    digest = hashlib.sha256("\0".join((str(state["room_id"]), job_id, run_key, str(notice))).encode()).hexdigest()
    suffix = "\n\n[Output truncated to fit the room.]"
    if len(text.encode("utf-8")) > MAX_USER_TEXT_BYTES:
        text = text.encode("utf-8")[:MAX_USER_TEXT_BYTES - len(suffix.encode())].decode("utf-8", "ignore") + suffix
    payload = validate_user_payload({"text": text, "thread_id": f"cron-{digest[:24]}"})
    prefix = SCHEDULED_NOTICE_PREFIX if notice else SCHEDULED_POST_PREFIX
    job_label = re.sub(r"[^A-Za-z0-9._:-]", "-", job_id)[: hosted_rooms.MAX_ACTOR_ID_CHARS - len(prefix)] or "job"
    actor = {"kind": "user", "id": f"{prefix}{job_label}",
             "display_name": (job_name or job_id)[: hosted_rooms.MAX_ACTOR_LABEL_CHARS]}
    event_id = f"cron-post:{digest[:40]}"
    try:
        return hosted_rooms.append_event(
            store, placement=placement, room_id=room_id, event_id=event_id, kind="message.user", actor=actor,
            payload=payload, authority_gateway_id=state["authority_gateway_id"],
            authority_epoch=state["authority_epoch"], now=now)
    except hosted_rooms.EventConflictError:
        # A retry of the same run re-rendered different text (a re-run agent, a new timestamp): the first
        # post already landed and the run is delivered, so report that post rather than a failure.
        event = store.hosted_rooms.event_by_id(placement=placement, room_id=state["room_id"], event_id=event_id)
        if event is None:
            raise
        return {**event, "idempotent": True}
