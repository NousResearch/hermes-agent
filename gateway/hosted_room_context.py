"""Room purpose, shared notes and scheduled-job posts for hosted Group Chats.

Both live in the append-only room log, never in a side file: the Discussion planner must stay a pure
function of durable room state, because a member turn's task id hashes its prompt. Purpose and notes are
read as of the message that opened a discussion, so editing them mid-discussion never re-plans a turn that
is already admitted; the edit applies from the next message. The same event carries the room's routing:
``everyone`` (every member gets a turn on an unaddressed message, the default) or ``lead`` (the lead picks
who takes it; ``gateway/hosted_room_lead.py``).

Scheduled posts are ordinary ``message.user`` events whose server-owned actor id names the cron job. Only
this module mints that actor (``groups.send`` hard-codes ``desktop``), so a client cannot impersonate a job.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from contextlib import closing
from typing import Any

from gateway import hosted_rooms
from gateway.hosted_rooms_common import DbPath

MAX_PURPOSE_CHARS = 500
MAX_NOTES_BYTES = 4 * 1024
CONTEXT_EVENT_KIND = "room.context_set"
_CONTEXT_ACTOR = {"kind": "system", "id": "room-control"}
# Actor id prefixes for posts the scheduler makes. A result post can start a round when it @mentions a
# member; a failure notice never does (its text is machine-written, not addressed to anyone).
SCHEDULED_POST_PREFIX = "cron:"
SCHEDULED_NOTICE_PREFIX = "cron-notice:"
ROUTING_MODES = ("everyone", "lead")
_EMPTY_CONTEXT = {"purpose": "", "notes": "", "routing": "everyone", "lead_member_id": ""}


def validate_context(purpose: Any, notes: Any, routing: Any = "everyone", lead_member_id: Any = "") -> dict[str, str]:
    """Normalize one room settings record; purpose and notes may be empty (clears them). ``lead`` routing
    needs a lead; ``everyone`` routing never keeps one."""
    if not all(isinstance(value, str) for value in (purpose, notes, routing, lead_member_id)):
        raise hosted_rooms.HostedRoomError("purpose, notes, routing and lead_member_id must be strings")
    purpose = " ".join(purpose.split())
    notes = notes.strip()
    if len(purpose) > MAX_PURPOSE_CHARS:
        raise hosted_rooms.HostedRoomError(f"purpose must be at most {MAX_PURPOSE_CHARS} characters")
    if len(notes.encode("utf-8")) > MAX_NOTES_BYTES:
        raise hosted_rooms.HostedRoomError(f"notes must be at most {MAX_NOTES_BYTES} bytes")
    if routing not in ROUTING_MODES:
        raise hosted_rooms.HostedRoomError("routing must be 'everyone' or 'lead'")
    if routing == "lead" and not lead_member_id.strip():
        raise hosted_rooms.HostedRoomError("lead routing needs a lead_member_id")
    return {"purpose": purpose, "notes": notes, "routing": routing,
            "lead_member_id": lead_member_id.strip() if routing == "lead" else ""}


def _local_authority(db_path: DbPath, room_id: Any) -> dict[str, Any]:
    state = hosted_rooms.room_state(db_path, room_id=room_id)
    if state["authority_gateway_id"] != hosted_rooms.local_authority_gateway_id():
        raise hosted_rooms.AuthorityConflictError("This Group Chat is managed by another gateway.")
    return state


def room_context(db_path: DbPath, *, room_id: Any, before_seq: int | None = None) -> dict[str, str]:
    """The latest purpose/notes, or the ones in force before ``before_seq`` when given."""
    room_id = hosted_rooms._room_id(room_id)
    sql = "SELECT payload_json FROM hosted_room_events WHERE room_id=? AND kind=?"
    params: tuple[Any, ...] = (room_id, CONTEXT_EVENT_KIND)
    if before_seq is not None:
        sql, params = sql + " AND seq<?", (*params, int(before_seq))
    with closing(hosted_rooms._read_connection(db_path)) as conn:
        row = conn.execute(sql + " ORDER BY seq DESC LIMIT 1", params).fetchone()
    if row is None:
        return dict(_EMPTY_CONTEXT)
    payload = json.loads(row["payload_json"])
    return {key: str(payload.get(key) or default) for key, default in _EMPTY_CONTEXT.items()}


def set_room_context(
    db_path: DbPath, *, room_id: Any, event_id: Any, purpose: Any, notes: Any, routing: Any = None,
    lead_member_id: Any = None, now: float | None = None,
) -> dict[str, Any]:
    """Record new room settings as one replayable room event. Omitted routing fields keep their current
    values, so a client that only edits purpose/notes never switches a lead room back to everyone."""
    current = room_context(db_path, room_id=room_id)
    context = validate_context(
        purpose, notes, current["routing"] if routing is None else routing,
        current["lead_member_id"] if lead_member_id is None else lead_member_id)
    state = _local_authority(db_path, room_id)
    if context["routing"] == "lead" and context["lead_member_id"] not in {
            str(member.get("member_id")) for member in state.get("members") or ()}:
        raise hosted_rooms.HostedRoomError("the lead must be a member of this Group Chat")
    event = hosted_rooms.append_event(
        db_path, room_id=room_id, event_id=event_id, kind=CONTEXT_EVENT_KIND, actor=dict(_CONTEXT_ACTOR),
        payload=context, authority_gateway_id=state["authority_gateway_id"],
        authority_epoch=state["authority_epoch"], now=now)
    return {"context": context, "event": event}


def scheduled_post_kind(actor: Mapping[str, Any] | None) -> str | None:
    """``"post"``/``"notice"`` for a scheduler-authored ``message.user`` actor, else None (the human)."""
    if not isinstance(actor, Mapping) or actor.get("kind") != "user":
        return None
    actor_id = str(actor.get("id") or "")
    if actor_id.startswith(SCHEDULED_NOTICE_PREFIX):
        return "notice"
    return "post" if actor_id.startswith(SCHEDULED_POST_PREFIX) else None


def post_scheduled_message(
    db_path: DbPath, *, room_id: Any, job_id: str, job_name: str, run_key: str, text: str, notice: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Append one cron output to a room. Idempotent per ``run_key``: a retried delivery of the same run
    returns the original event instead of posting twice."""
    from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES, validate_user_payload

    state = _local_authority(db_path, room_id)
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
            db_path, room_id=room_id, event_id=event_id, kind="message.user", actor=actor,
            payload=payload, authority_gateway_id=state["authority_gateway_id"],
            authority_epoch=state["authority_epoch"], now=now)
    except hosted_rooms.EventConflictError:
        # A retry of the same run re-rendered different text (a re-run agent, a new timestamp): the first
        # post already landed and the run is delivered, so report that post rather than a failure.
        with closing(hosted_rooms._read_connection(db_path)) as conn:
            row = hosted_rooms._load_event(conn, state["room_id"], event_id)
        if row is None:
            raise
        return hosted_rooms._event_from_row(row, idempotent=True)
