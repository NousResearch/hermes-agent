"""Deterministic policy for same-gateway hosted-room Discussions.

Pure (no I/O, transport or model knowledge): a frozen local roster plus the complete typed room log yields
at most one next driver task. Discussion coordinates live in deterministic ``TaskIdentity`` values and typed
terminal events rather than a widened driver payload, so a restart reconstructs tasks from durable state.
Callers must reconcile terminal driver rows into publication plans before asking for the next task.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Literal

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway import hosted_rooms_common as common
from gateway.hosted_rooms_common import compact_json


MAX_DISCUSSION_MEMBERS = 6
MIN_DISCUSSION_MEMBERS = 2
MAX_DISCUSSION_ROUNDS = 3
MAX_DISCUSSION_MESSAGES = 10
MAX_DISCUSSION_DELTA_LINES = 24
MAX_USER_TEXT_BYTES = 64 * 1024
MAX_MEMBER_TEXT_BYTES = 64 * 1024
_TRUNCATED_REPLY_NOTICE = "\n\n[Reply truncated. Ask the Bot to share the full result as a file.]"

Payload = Mapping[str, Any]
DecisionStatus = Literal["idle", "task", "settled", "bounded"]
TerminalKind = Literal["settled", "failed", "cancelled", "deferred"]

_MENTION_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9._:-]*)", re.IGNORECASE)
_TURN_ID_RE = re.compile(
    r"^d(?P<source>[1-9][0-9]*)\.r(?P<round>[0-2])\."
    r"p(?P<position>[0-5])\.s(?P<seen>[1-9][0-9]*)\."
    r"m(?P<member>[0-9a-f]{24})$")

_TARGET_FIELDS = {
    "local": frozenset({"kind", "profile"}),
    "peer": frozenset({"kind", "peer_id", "installation_id", "profile", "capability_digest"})}
_REMOTE_MEMBER_FIELDS = frozenset({
    "connectionId", "connectionKind", "connectionLabel", "connection_id", "connection_kind", "connection_label",
    "remoteSource", "route", "sourceMissing", "sourceReachable", "sourceScoped", "targetProfile", "target_profile"})
_USER_PAYLOAD_FIELDS = frozenset({"text", "thread_id"})
_TURN_COORDINATE_FIELDS = frozenset(
    {"discussion_event_id", "member_id", "member_index", "round_index", "task_id", "thread_id", "turn_id", })
_MEMBER_MESSAGE_FIELDS = _TURN_COORDINATE_FIELDS | {"text"}
_TERMINAL_FIELDS = {  # kind -> exact payload fields (coordinates + seen_through_seq + per-kind extras)
    kind: _TURN_COORDINATE_FIELDS | {"seen_through_seq", *extra} for kind, extra in (
        ("turn.settled", ("message_event_id", "passed")), ("turn.failed", ("error",)),
        ("turn.cancelled", ("reason",)), ("turn.deferred", ("execution_generation", "reason")))}
_TERMINAL_OPTIONAL_FIELDS = {"turn.failed": frozenset({"reason_code"})}
_TERMINAL_EVENT_KINDS = frozenset(_TERMINAL_FIELDS)
# Gateway-authored control events: kind -> (exact payload fields, identifier fields).
_GATEWAY_EVENT_FIELDS = {
    "room.activity": (
        frozenset({"status", "reason_code", "thread_id", "discussion_event_id"}),
        ("reason_code", "thread_id", "discussion_event_id")),
    "room.stop_requested": (frozenset({"cancel_id"}), ("cancel_id",))}
_EPOCH_STAMPED_KINDS = _TERMINAL_EVENT_KINDS | {"message.member", *_GATEWAY_EVENT_FIELDS}


class DiscussionPolicyError(ValueError): """Base class for invalid policy input or unreconstructable state."""

class DiscussionValidationError(DiscussionPolicyError):
    """Raised when a room, roster, payload, or typed event is malformed."""

class DiscussionReconstructionError(DiscussionPolicyError):
    """Raised when a persisted task cannot be reproduced from durable state."""


@dataclass(frozen=True)
class DiscussionMember:
    """One immutable local or peer member of the hosted room."""
    member_id: str
    profile: str
    handle: str
    display_name: str = ""
    target: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DiscussionRoom:
    """Validated policy projection of one active hosted room."""
    room_id: str
    name: str
    members: tuple[DiscussionMember, ...]
    gateway_id: str
    authority_epoch: int


@dataclass(frozen=True)
class DiscussionTaskPlan:
    """One deterministic member turn compatible with the driver schema."""
    identity: driver.TaskIdentity
    payload: Mapping[str, Any]
    discussion_event_id: str
    member: DiscussionMember
    member_index: int
    round_index: int
    seen_through_seq: int


@dataclass(frozen=True)
class DiscussionDecision:
    """Current result of replaying one room's Discussion policy."""
    status: DecisionStatus
    reason: str
    discussion_event_id: str | None = None
    source_event_seq: int | None = None
    thread_id: str | None = None
    task: DiscussionTaskPlan | None = None


@dataclass(frozen=True)
class EventPlan:
    """One idempotent append for :func:`gateway.hosted_rooms.append_event`."""
    event_id: str
    kind: str
    actor: Mapping[str, str]
    payload: Mapping[str, Any]
    authority_gateway_id: str
    authority_epoch: int

    def append_kwargs(self, room_id: str) -> dict[str, Any]:
        """Return keyword arguments accepted by ``append_event`` (field order matches the dataclass)."""
        return {"room_id": room_id, **asdict(self)}


@dataclass(frozen=True)
class PublicationPlan:
    """Ordered visible and terminal effects for one driver task."""
    task_id: str
    terminal_kind: str
    events: tuple[EventPlan, ...]


@dataclass(frozen=True)
class _ValidatedEvent:
    raw: Payload
    seq: int
    event_id: str
    kind: str
    actor: Payload
    payload: Payload


_identifier = partial(common.identifier, error=DiscussionValidationError, max_chars=driver.MAX_IDENTIFIER_CHARS)
_exact_fields = partial(common.exact_fields, error=DiscussionValidationError)
_bounded_int = partial(common.bounded_int, error=DiscussionValidationError)
_text = partial(common.text, error=DiscussionValidationError)


def _positive_int(value: Any, *, label: str) -> int:
    return _bounded_int(value, message=f"{label} must be a positive integer", low=1)


def _zero_based_int(value: Any, *, label: str, maximum: int) -> int:
    return _bounded_int(value, message=f"{label} must be an integer between 0 and {maximum}", high=maximum)


def _peer_id(member: DiscussionMember) -> str | None:
    target = member.target
    return target.get("peer_id") if target and target.get("kind") == "peer" else None


def _member_actor(member: DiscussionMember, *, display_name: bool = True) -> dict[str, Any]:
    """Member actor for the room log; ``connection_id`` (and ``display_name``) only when present."""
    actor = {"kind": "member", "id": member.member_id, "profile": member.profile}
    if (peer_id := _peer_id(member)) is not None:
        actor["connection_id"] = peer_id
    if display_name and member.display_name:
        actor["display_name"] = member.display_name
    return actor


def _all_failure_reasons() -> frozenset[str]:
    from tools.bot_failure_reasons import ALL_REASONS
    return ALL_REASONS


def validate_user_payload(value: Any) -> dict[str, Any]:
    """Validate and normalize the exact ``message.user`` Discussion payload."""
    payload = _exact_fields(value, label="user payload", required=_USER_PAYLOAD_FIELDS)
    return {
        "text": _text(payload["text"], label="user payload text", max_bytes=MAX_USER_TEXT_BYTES),
        "thread_id": _identifier(payload["thread_id"], label="thread_id")}


def _validate_member_target(value: Any, *, profile: str, known_profiles: set[str], index: int) -> dict[str, Any]:
    if value is None:
        if profile not in known_profiles:
            raise DiscussionValidationError(f"member {index} profile '{profile}' is not local to this gateway")
        return {"kind": "local", "profile": profile}
    if not isinstance(value, Mapping):
        raise DiscussionValidationError(f"member {index} target must be an object")
    kind = value.get("kind")
    if kind not in _TARGET_FIELDS:
        raise DiscussionValidationError(f"member {index} target kind must be local or peer")
    target = _exact_fields(value, label=f"member {index} {kind} target", required=_TARGET_FIELDS[kind])
    target_profile = _identifier(target["profile"], label=f"member {index} target profile")
    if kind == "local":
        if target_profile != profile or profile not in known_profiles:
            raise DiscussionValidationError(f"member {index} local target does not match a local profile")
        return {"kind": "local", "profile": profile}
    if target_profile != profile:
        raise DiscussionValidationError(f"member {index} peer target profile does not match member profile")
    digest = target["capability_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DiscussionValidationError(f"member {index} capability_digest must be a sha256 digest")
    return {
        "kind": "peer", "peer_id": _identifier(target["peer_id"], label=f"member {index} peer_id"),
        "installation_id": _identifier(target["installation_id"], label=f"member {index} installation_id"),
        "profile": target_profile, "capability_digest": digest}


def _validate_member(raw: Any, index: int, known_profiles: set[str]) -> DiscussionMember:
    if not isinstance(raw, Mapping):
        raise DiscussionValidationError(f"member {index} must be an object")
    if remote_fields := frozenset(raw) & _REMOTE_MEMBER_FIELDS:
        raise DiscussionValidationError(
            f"member {index} contains cross-gateway fields: {', '.join(sorted(remote_fields))}")
    member = _exact_fields(
        raw, label=f"member {index}", required=frozenset({"member_id", "profile", "handle"}),
        optional=frozenset({"display_name", "target"}))
    member_id, profile, handle = (
        _identifier(member[field], label=f"member {index} {label}")
        for field, label in (("member_id", "id"), ("profile", "profile"), ("handle", "handle")))
    target = _validate_member_target(member.get("target"), profile=profile, known_profiles=known_profiles, index=index)
    if not isinstance(display_name := member.get("display_name", ""), str):
        raise DiscussionValidationError(f"member {index} display_name must be a string")
    if len(display_name := display_name.strip()) > hosted_rooms.MAX_ACTOR_LABEL_CHARS:
        raise DiscussionValidationError(f"member {index} display_name is too long")
    return DiscussionMember(member_id, profile, handle, display_name, target)


def validate_roster(value: Any, *, local_profiles: Iterable[str]) -> tuple[DiscussionMember, ...]:
    """Validate a frozen 2-6 member roster of profiles on this gateway."""
    if not isinstance(value, list):
        raise DiscussionValidationError("members must be a list")
    if not MIN_DISCUSSION_MEMBERS <= len(value) <= MAX_DISCUSSION_MEMBERS:
        raise DiscussionValidationError(
            f"members must contain between {MIN_DISCUSSION_MEMBERS} and {MAX_DISCUSSION_MEMBERS} entries")
    known_profiles = {_identifier(profile, label="local profile") for profile in local_profiles}
    members: list[DiscussionMember] = []
    targets: set[str] = set()
    handles: set[str] = {"all", "everyone"}  # reserved mention handles
    member_ids: set[str] = set()
    for index, raw in enumerate(value):
        member = _validate_member(raw, index, known_profiles)
        unique = "profiles" if member.target.get("kind") == "local" else "targets"
        for key, seen, message in (
            (compact_json(member.target, ensure_ascii=False).casefold(), targets, f"member {unique} must be unique"),
            (member.handle.casefold(), handles, "member handles must be unique and cannot reserve @all or @everyone"),
            (member.member_id.casefold(), member_ids, "member ids must be unique")):
            if key in seen:
                raise DiscussionValidationError(message)
            seen.add(key)
        members.append(member)
    return tuple(members)


def validate_room(value: Any, *, local_profiles: Iterable[str]) -> DiscussionRoom:
    """Project a hosted-room row into the strict same-gateway policy shape."""
    if not isinstance(value, Mapping):
        raise DiscussionValidationError("room must be an object")
    if value.get("disbanded_at") is not None:
        raise DiscussionValidationError("room is disbanded")
    room_id = _identifier(value.get("room_id"), label="room_id")
    if not isinstance(name := value.get("name"), str) or not name.strip():
        raise DiscussionValidationError("room name must be a non-empty string")
    if len(name := name.strip()) > hosted_rooms.MAX_ROOM_NAME_CHARS:
        raise DiscussionValidationError("room name is too long")
    gateway_id = _identifier(value.get("authority_gateway_id"), label="authority_gateway_id")
    authority_epoch = _positive_int(value.get("authority_epoch"), label="authority_epoch")
    members = validate_roster(value.get("members"), local_profiles=local_profiles)
    return DiscussionRoom(room_id, name, members, gateway_id, authority_epoch)


def is_pass_text(value: Any) -> bool:
    """Return whether a settled member result is Discussion silence."""
    text = str(value or "").strip()
    return not text or re.fullmatch(r"\(?\s*pass\s*\)?\.?", text, re.IGNORECASE) is not None


def resolve_mentions(
    texts: Iterable[str], members: Sequence[DiscussionMember], *, default_all: bool = True
) -> tuple[DiscussionMember, ...]:
    """Resolve member handles deterministically against the frozen roster."""
    by_handle = {member.handle.casefold(): member for member in members}
    mentioned: set[str] = set()
    everyone = False
    for text in texts:
        for match in _MENTION_RE.finditer(str(text or "")):
            handle = match.group(1).casefold()
            if handle in {"all", "everyone"}:
                everyone = True
            elif handle in by_handle:
                mentioned.add(handle)
    if everyone or (default_all and not mentioned):
        return tuple(members)
    return tuple(member for member in members if member.handle.casefold() in mentioned)


def _unaddressed_member_mentions(
    messages: Sequence[_ValidatedEvent], room: DiscussionRoom) -> tuple[DiscussionMember, ...]:
    """Return peers explicitly cited by a Bot and not heard from afterward."""
    cited_at: dict[str, int] = {}
    last_post_at: dict[str, int] = {}
    for event in messages:
        if event.kind != "message.member":
            continue
        speaker_id = str(event.payload["member_id"])
        last_post_at[speaker_id] = event.seq
        for member in resolve_mentions((str(event.payload["text"]),), room.members, default_all=False):
            if member.member_id != speaker_id:
                cited_at[member.member_id] = event.seq
    return tuple(
        member for member in room.members
        if member.member_id in cited_at and last_post_at.get(member.member_id, 0) <= cited_at[member.member_id])


def _require_gateway_actor(actor: Mapping[str, Any], room: DiscussionRoom, message: str) -> None:
    if actor.get("kind") != "gateway" or actor.get("id") != room.gateway_id:
        raise DiscussionValidationError(message)


def _member_by_id(room: DiscussionRoom, member_id: Any) -> DiscussionMember:
    normalized = _identifier(member_id, label="member_id")
    if (member := next((m for m in room.members if m.member_id == normalized), None)) is None:
        raise DiscussionValidationError(f"unknown Discussion member '{normalized}'")
    return member


def _validate_turn_coordinates(payload: Mapping[str, Any], room: DiscussionRoom) -> None:
    _member_by_id(room, payload.get("member_id"))
    _zero_based_int(payload.get("member_index"), label="member_index", maximum=MAX_DISCUSSION_MEMBERS - 1)
    _zero_based_int(payload.get("round_index"), label="round_index", maximum=MAX_DISCUSSION_ROUNDS - 1)
    for field in ("thread_id", "task_id", "turn_id", "discussion_event_id"):
        _identifier(payload.get(field), label=field)


# -- per-kind event payload validators (dispatched by _validate_event) ---------
# Each takes (kind, payload, actor, room) and returns the payload to record.
def _validate_user_event(kind: str, payload: Payload, actor: Payload, room: DiscussionRoom) -> Payload:
    payload = validate_user_payload(payload)
    if actor.get("kind") != "user":
        raise DiscussionValidationError("message.user requires a user actor")
    return payload


def _validate_member_message(kind: str, payload: Payload, actor: Payload, room: DiscussionRoom) -> Payload:
    _exact_fields(payload, label="message.member payload", required=_MEMBER_MESSAGE_FIELDS)
    _validate_turn_coordinates(payload, room)
    if not isinstance(text := payload.get("text"), str) or not text.strip() or is_pass_text(text):
        raise DiscussionValidationError("message.member text must be a non-pass string")
    member = _member_by_id(room, payload.get("member_id"))
    expected = {**_member_actor(member, display_name=False), "connection_id": _peer_id(member)}
    if any(actor.get(key) != value for key, value in expected.items()):
        raise DiscussionValidationError("message.member actor does not match roster")
    return payload


def _validate_terminal_event(kind: str, payload: Payload, actor: Payload, room: DiscussionRoom) -> Payload:
    _exact_fields(
        payload, label=f"{kind} payload", required=_TERMINAL_FIELDS[kind],
        optional=_TERMINAL_OPTIONAL_FIELDS.get(kind, frozenset()))
    _validate_turn_coordinates(payload, room)
    _positive_int(payload.get("seen_through_seq"), label="seen_through_seq")
    if actor.get("connection_id") is not None:
        raise DiscussionValidationError(f"{kind} requires a gateway actor")
    _require_gateway_actor(actor, room, f"{kind} requires a gateway actor")
    if kind == "turn.settled":
        if not isinstance(payload.get("passed"), bool):
            raise DiscussionValidationError("turn.settled passed must be a boolean")
        message_event_id = payload.get("message_event_id")
        if not payload["passed"]:
            _identifier(message_event_id, label="message_event_id")
        elif message_event_id is not None:
            raise DiscussionValidationError("a passed turn cannot reference a member message")
        return payload
    field = "error" if kind == "turn.failed" else "reason"
    if not isinstance(payload.get(field), str) or not payload[field].strip():
        raise DiscussionValidationError(f"{kind} {field} must be non-empty")
    if kind == "turn.deferred":
        _positive_int(payload.get("execution_generation"), label="execution_generation")
    if kind == "turn.failed" and "reason_code" in payload and payload["reason_code"] not in _all_failure_reasons():
        raise DiscussionValidationError("turn.failed reason_code must use the shared failure vocabulary")
    return payload


def _validate_gateway_event(kind: str, payload: Payload, actor: Payload, room: DiscussionRoom) -> Payload:
    fields, identifier_fields = _GATEWAY_EVENT_FIELDS[kind]
    _exact_fields(payload, label=f"{kind} payload", required=fields)
    if kind == "room.activity" and payload.get("status") not in {"settled", "bounded"}:
        raise DiscussionValidationError("invalid room.activity status")
    for field in identifier_fields:
        _identifier(payload.get(field), label=field)
    _require_gateway_actor(actor, room, f"{kind} requires the room gateway")
    return payload


_EVENT_VALIDATORS = {
    "message.user": _validate_user_event, "message.member": _validate_member_message,
    **dict.fromkeys(_TERMINAL_EVENT_KINDS, _validate_terminal_event),
    **dict.fromkeys(_GATEWAY_EVENT_FIELDS, _validate_gateway_event)}


def _validate_event(raw: Any, *, room: DiscussionRoom, previous_seq: int) -> _ValidatedEvent:
    if not isinstance(raw, Mapping):
        raise DiscussionValidationError("room event must be an object")
    if raw.get("room_id") != room.room_id:
        raise DiscussionValidationError("room event belongs to a different room")
    if (seq := _positive_int(raw.get("seq"), label="event seq")) <= previous_seq:
        raise DiscussionValidationError("room events must be in strict sequence order")
    event_id = _identifier(raw.get("event_id"), label="event_id")
    kind, actor, payload = raw.get("kind"), raw.get("actor"), raw.get("payload")
    for value, expected, message in (
        (kind, str, "event kind must be a string"), (actor, Mapping, "event actor must be an object"),
        (payload, Mapping, "event payload must be an object")):
        if not isinstance(value, expected):
            raise DiscussionValidationError(message)
    if kind in _EPOCH_STAMPED_KINDS and raw.get("authority_epoch") != room.authority_epoch:
        raise DiscussionValidationError(f"{kind} authority epoch does not match the room")
    if (validator := _EVENT_VALIDATORS.get(kind)) is not None:
        payload = validator(kind, payload, actor, room)
    return _ValidatedEvent(raw=raw, seq=seq, event_id=event_id, kind=kind, actor=actor, payload=payload)


def _validated_events(events: Sequence[Mapping[str, Any]], *, room: DiscussionRoom) -> tuple[_ValidatedEvent, ...]:
    validated: list[_ValidatedEvent] = []
    event_ids: set[str] = set()
    for raw in events:
        event = _validate_event(raw, room=room, previous_seq=validated[-1].seq if validated else 0)
        if event.event_id in event_ids:
            raise DiscussionValidationError("room event ids must be unique")
        validated.append(event)
        event_ids.add(event.event_id)
    return tuple(validated)


def derive_member_watermarks(
    room_value: Any, events: Sequence[Mapping[str, Any]], *, local_profiles: Iterable[str]
) -> dict[tuple[str, str], int]:
    """Derive ``(thread_id, member_id)`` watermarks from terminal events."""
    return _derive_member_watermarks(
        _validated_events(events, room=validate_room(room_value, local_profiles=local_profiles)))


def _derive_member_watermarks(events: Sequence[_ValidatedEvent]) -> dict[tuple[str, str], int]:
    messages_by_id = {event.event_id: event for event in events if event.kind == "message.member"}
    terminal_by_task: dict[str, _ValidatedEvent] = {}
    watermarks: dict[tuple[str, str], int] = {}
    for event in events:
        if event.kind not in _TERMINAL_EVENT_KINDS:
            continue
        task_id = str(event.payload["task_id"])
        if (previous := terminal_by_task.get(task_id)) is not None:
            if previous.kind != "turn.deferred":
                raise DiscussionValidationError(f"task '{task_id}' has more than one terminal room event")
            if event.kind == "turn.deferred" and int(
                event.payload["execution_generation"]) <= int(previous.payload["execution_generation"]):
                raise DiscussionValidationError(f"task '{task_id}' deferral generation did not advance")
        terminal_by_task[task_id] = event
        key = (str(event.payload["thread_id"]), str(event.payload["member_id"]))
        watermark = int(event.payload["seen_through_seq"])
        if event.kind == "turn.settled" and not event.payload["passed"]:
            message = messages_by_id.get(str(event.payload["message_event_id"]))
            if message is None or any(
                message.payload.get(f) != event.payload.get(f) for f in ("task_id", "member_id", "thread_id")):
                raise DiscussionValidationError("turn.settled references no matching member message")
            watermark = max(watermark, message.seq)
        watermarks[key] = max(watermarks.get(key, 0), watermark)
    return watermarks


def _member_digest(member: DiscussionMember) -> str:
    target = compact_json(member.target or {"kind": "local", "profile": member.profile}, ensure_ascii=False)
    return hashlib.sha256(f"{member.member_id}\0{member.profile}\0{member.handle}\0{target}".encode()).hexdigest()[:24]


def _rotate(members: Sequence[DiscussionMember], round_index: int) -> tuple[DiscussionMember, ...]:
    shift = round_index % len(members) if members else 0
    return tuple((*members[shift:], *members[:shift]))


def _format_message(event: _ValidatedEvent, room: DiscussionRoom) -> str:
    if event.kind == "message.user":
        return f"User (user): {event.payload['text']}"
    return f"@{_member_by_id(room, event.payload['member_id']).handle}: {event.payload['text']}"


def _truncate_utf8_text(value: Any, *, max_bytes: int, suffix: str = "") -> str:
    text = str(value or "")
    if len(encoded := text.encode("utf-8")) <= max_bytes:
        return text
    # ``ignore`` drops only the incomplete trailing sequence of a cut valid-UTF-8 prefix.
    prefix = encoded[: max(0, max_bytes - len(suffix.encode("utf-8")))].decode("utf-8", "ignore")
    return prefix + suffix if prefix else suffix.strip()


def _build_prompt(
    *, room: DiscussionRoom, member: DiscussionMember, messages: Sequence[_ValidatedEvent], watermark: int,
    seen_through_seq: int) -> str:
    delta = [event for event in messages if watermark < event.seq <= seen_through_seq][- MAX_DISCUSSION_DELTA_LINES:]
    peers = ", ".join(f"@{candidate.handle}" for candidate in room.members if candidate.member_id != member.member_id)
    opening = [
        f'[Discussion: "{room.name}"] You are @{member.handle}, one participant '
        f"with {peers or 'no other members'} and the user.", "",
        "New messages in this thread since your last turn (oldest first):"]
    rules = [
        "", "Rules for this Discussion:",
        "- Reply with one conversational message only when you have something new worth adding.",
        '- If you have nothing new to add, reply with exactly "(pass)".',
        "- Mention a teammate by handle to pull them into the next round; do not repeat points already made.",
        "- Never reveal content from private conversations. Your reply is published verbatim."]
    fixed_bytes = len("\n".join([*opening, *rules]).encode("utf-8"))
    available = max(0, driver.MAX_PROMPT_BYTES - fixed_bytes - 1)
    selected: list[str] = []
    for event in reversed(delta):
        line = f"  {_format_message(event, room)}"
        if (line_bytes := len(line.encode("utf-8")) + 1) > available:
            if not selected and available > 32:
                selected.append(_truncate_utf8_text(line, max_bytes=available))
            selected.append("  [Earlier content omitted to fit this turn.]")
            break
        selected.append(line)
        available -= line_bytes
    selected.reverse()
    if len((prompt := "\n".join([*opening, *selected, *rules])).encode("utf-8")) > driver.MAX_PROMPT_BYTES:
        raise DiscussionValidationError("Discussion prompt exceeds the driver limit")
    return prompt


def _make_task_plan(
    *, room: DiscussionRoom, discussion_event: _ValidatedEvent, member: DiscussionMember, member_index: int,
    round_index: int, seen_through_seq: int, prompt: str) -> DiscussionTaskPlan:
    turn_id = f"d{discussion_event.seq}.r{round_index}.p{member_index}.s{seen_through_seq}.m{_member_digest(member)}"
    seed = compact_json({
        "discussion_event_id": discussion_event.event_id, "member_id": member.member_id, "member_index": member_index,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "room_id": room.room_id,
        "round_index": round_index, "seen_through_seq": seen_through_seq, "source_event_seq": discussion_event.seq,
        "thread_id": discussion_event.payload["thread_id"]})
    identity = driver.TaskIdentity(
        room_id=room.room_id, task_id=f"dtask:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:48]}",
        thread_id=str(discussion_event.payload["thread_id"]), turn_id=turn_id)
    payload = {
        "target_member_id": member.member_id, "target_profile": member.profile, "prompt": prompt,
        "source_event_seq": discussion_event.seq}
    return DiscussionTaskPlan(
        identity, payload, discussion_event.event_id, member, member_index, round_index, seen_through_seq)


def _pending_discussion(validated: Sequence[_ValidatedEvent]) -> _ValidatedEvent | None:
    """Oldest latest-per-thread user message not stopped and not yet completed."""
    stopped_through_seq = max((event.seq for event in validated if event.kind == "room.stop_requested"), default=0)
    completed_discussion_ids = {
        str(event.payload["discussion_event_id"]) for event in validated
        if event.kind == "room.activity" and event.payload.get("status") in {"settled", "bounded"}}
    latest_by_thread = {
        str(event.payload["thread_id"]): event for event in validated if event.kind == "message.user"}
    return next((
        event for event in sorted(latest_by_thread.values(), key=lambda item: item.seq)
        if event.seq > stopped_through_seq and event.event_id not in completed_discussion_ids), None)


def _thread_messages(
    validated: Sequence[_ValidatedEvent], discussion: _ValidatedEvent
) -> tuple[tuple[_ValidatedEvent, ...], tuple[_ValidatedEvent, ...], tuple[_ValidatedEvent, ...]]:
    """Return (thread messages, messages since the discussion, this discussion's member messages)."""
    thread_id = str(discussion.payload["thread_id"])
    committed_member_message_ids = {
        str(event.payload["message_event_id"]) for event in validated
        if event.kind == "turn.settled" and event.payload.get("message_event_id") is not None}
    # Publication writes the visible member message before the terminal event.
    # A crash in that gap leaves the message in the log, but it is not committed
    # policy input yet: ignoring it reproduces the original task coordinates so
    # the caller can inspect the terminal driver row and finish publication.
    thread_messages = tuple(
        event for event in validated
        if event.payload.get("thread_id") == thread_id and (
            event.kind == "message.user"
            or (event.kind == "message.member" and event.event_id in committed_member_message_ids)))
    return thread_messages, tuple(event for event in thread_messages if event.seq >= discussion.seq), tuple(
        event for event in thread_messages
        if event.kind == "message.member" and event.payload.get("discussion_event_id") == discussion.event_id)


def _effective_watermarks(
    validated: Sequence[_ValidatedEvent], initial_watermarks: Mapping[tuple[str, str], int] | None
) -> dict[tuple[str, str], int]:
    watermarks = {
        (str(thread_id), str(member_id)): int(value)
        for (thread_id, member_id), value in (initial_watermarks or {}).items() if int(value) >= 0}
    for key, value in _derive_member_watermarks(validated).items():
        watermarks[key] = max(watermarks.get(key, 0), value)
    return watermarks


def plan_next_task(
    room_value: Any, events: Sequence[Mapping[str, Any]], *, local_profiles: Iterable[str],
    initial_watermarks: Mapping[tuple[str, str], int] | None = None) -> DiscussionDecision:
    """Replay the complete room log and return at most one next member task."""
    room = validate_room(room_value, local_profiles=local_profiles)
    validated = _validated_events(events, room=room)
    if (discussion := _pending_discussion(validated)) is None:
        return DiscussionDecision(status="idle", reason="no_pending_user_event")
    thread_id = str(discussion.payload["thread_id"])
    decide = partial(
        DiscussionDecision, discussion_event_id=discussion.event_id, source_event_seq=discussion.seq,
        thread_id=thread_id)
    thread_messages, discussion_messages, member_messages = _thread_messages(validated, discussion)
    if len(member_messages) >= MAX_DISCUSSION_MESSAGES:
        return decide("bounded", "max_messages")
    terminals = {
        (int(event.payload["round_index"]), str(event.payload["member_id"])) for event in validated
        if event.kind in _TERMINAL_EVENT_KINDS and event.payload.get("discussion_event_id") == discussion.event_id}
    watermarks = _effective_watermarks(validated, initial_watermarks)
    seen_through_seq = max(event.seq for event in thread_messages)
    for round_index in range(MAX_DISCUSSION_ROUNDS):
        # The user's message selects the first round, with no mention meaning
        # everyone. Later rounds are opt-in: only a peer explicitly cited by a
        # Bot and not heard from afterward gets another turn. Every member's
        # watermark remains intact, so a peer cited later still receives the
        # complete bounded transcript delta without consuming turns meanwhile.
        responders = (
            resolve_mentions((str(discussion.payload["text"]),), room.members) if round_index == 0
            else _unaddressed_member_mentions(discussion_messages, room))
        for member_index, member in enumerate(_rotate(responders, round_index)):
            if (round_index, member.member_id) in terminals:
                continue
            watermark = watermarks.get((thread_id, member.member_id), 0)
            if not any(watermark < event.seq <= seen_through_seq for event in thread_messages):
                continue
            prompt = _build_prompt(
                room=room, member=member, messages=thread_messages, watermark=watermark,
                seen_through_seq=seen_through_seq)
            return decide("task", "member_turn", task=_make_task_plan(
                room=room, discussion_event=discussion, member=member, member_index=member_index,
                round_index=round_index, seen_through_seq=seen_through_seq, prompt=prompt))
        if not any(int(event.payload["round_index"]) == round_index for event in member_messages):
            return decide("settled", "silent_round")
        if round_index == MAX_DISCUSSION_ROUNDS - 1:
            return decide("bounded", "max_rounds")
    raise AssertionError("bounded Discussion loop exhausted unexpectedly")


def reconstruct_task_plan(
    room_value: Any, events: Sequence[Mapping[str, Any]], task: Mapping[str, Any], *, local_profiles: Iterable[str]
) -> DiscussionTaskPlan:
    """Reconstruct and verify one persisted driver task after a restart."""
    room = validate_room(room_value, local_profiles=local_profiles)
    validated = _validated_events(events, room=room)
    identity, payload = task.get("identity"), task.get("payload")
    if not isinstance(identity, driver.TaskIdentity) or not isinstance(payload, Mapping):
        raise DiscussionReconstructionError("driver task has no valid identity or payload")
    fields = frozenset(payload)
    if not driver._TASK_PAYLOAD_REQUIRED_FIELDS <= fields <= (
        driver._TASK_PAYLOAD_REQUIRED_FIELDS | driver._TASK_PAYLOAD_OPTIONAL_FIELDS):
        raise DiscussionReconstructionError("driver task payload shape changed")
    if (match := _TURN_ID_RE.fullmatch(identity.turn_id)) is None:
        raise DiscussionReconstructionError("turn_id is not a Discussion coordinate")
    source_event_seq = int(match.group("source"))
    if payload.get("source_event_seq") != source_event_seq:
        raise DiscussionReconstructionError("task source event does not match turn_id")
    discussion = next((e for e in validated if e.seq == source_event_seq and e.kind == "message.user"), None)
    if discussion is None:
        raise DiscussionReconstructionError("task source user event is missing")
    if identity.room_id != room.room_id or identity.thread_id != discussion.payload["thread_id"]:
        raise DiscussionReconstructionError("task identity does not match its room thread")
    profile, target_member_id = payload.get("target_profile"), payload.get("target_member_id")
    member = next((
        m for m in room.members
        if m.profile == profile and (target_member_id is None or m.member_id == target_member_id)), None)
    if member is None or _member_digest(member) != match.group("member"):
        raise DiscussionReconstructionError("task target member does not match turn_id")
    if not isinstance(prompt := payload.get("prompt"), str) or not prompt.strip():
        raise DiscussionReconstructionError("task prompt is missing")
    if len(prompt.encode("utf-8")) > driver.MAX_PROMPT_BYTES:
        raise DiscussionReconstructionError("task prompt exceeds the driver limit")
    reconstructed = _make_task_plan(
        room=room, discussion_event=discussion, member=member, member_index=int(match.group("position")),
        round_index=int(match.group("round")), seen_through_seq=int(match.group("seen")), prompt=prompt)
    if reconstructed.identity != identity or dict(reconstructed.payload) != dict(payload):
        raise DiscussionReconstructionError("driver task failed deterministic reconstruction")
    return reconstructed


def _turn_coordinates(task: DiscussionTaskPlan) -> dict[str, Any]:
    """The coordinate fields every member message and terminal event carries."""
    return {
        "discussion_event_id": task.discussion_event_id, "member_id": task.member.member_id,
        "member_index": task.member_index, "round_index": task.round_index, "task_id": task.identity.task_id,
        "thread_id": task.identity.thread_id, "turn_id": task.identity.turn_id}


def _terminal_text(result: Any, *, field: str, fallback: str) -> str:
    if isinstance(result, Mapping):
        result = result.get(field) if field != "error" or result.get("error") is not None else result.get("text")
    return str(result or "").strip() or fallback


# -- per-status terminal payload builders (dispatched by plan_publication) -----
# Each returns (extra terminal payload fields, visible effects to publish first).
Effects = tuple[dict[str, Any], list[EventPlan]]


def _settled_effects(
    result: Any, *, task: DiscussionTaskPlan, room: DiscussionRoom, message_event_id: str, **_: Any) -> Effects:
    text = _truncate_utf8_text(
        _terminal_text(result, field="text", fallback=""), max_bytes=MAX_MEMBER_TEXT_BYTES,
        suffix=_TRUNCATED_REPLY_NOTICE)
    if is_pass_text(text):
        return {"message_event_id": None, "passed": True}, []
    return {"message_event_id": message_event_id, "passed": False}, [EventPlan(
        event_id=message_event_id, kind="message.member", actor=_member_actor(task.member),
        payload={**_turn_coordinates(task), "text": text}, authority_gateway_id=room.gateway_id,
        authority_epoch=room.authority_epoch)]


def _failed_effects(result: Any, **_: Any) -> Effects:
    error_text = _terminal_text(result, field="error", fallback="member turn failed")
    from tools.bot_failure_reasons import classify_agent_error
    supplied_reason = (
        str(result.get("reason_code") or result.get("reason") or "").strip() if isinstance(result, Mapping) else "")
    reason_code = supplied_reason if supplied_reason in _all_failure_reasons() else classify_agent_error(error_text)
    return {"error": error_text, "reason_code": reason_code}, []


def _cancelled_effects(result: Any, *, newer_same_thread: bool, **_: Any) -> Effects:
    if newer_same_thread:
        return {"reason": "superseded_by_newer_user_event"}, []
    return {"reason": _terminal_text(result, field="reason", fallback="member turn cancelled")}, []


def _deferred_effects(result: Any, *, execution_generation: int | None, **_: Any) -> Effects:
    reason = _terminal_text(result, field="reason", fallback="member_unavailable")
    return {"execution_generation": execution_generation, "reason": reason}, []


_TERMINAL_EFFECTS = {
    "settled": _settled_effects, "failed": _failed_effects, "cancelled": _cancelled_effects,
    "deferred": _deferred_effects}


def plan_publication(
    room_value: Any, events: Sequence[Mapping[str, Any]], task: DiscussionTaskPlan, *, status: TerminalKind,
    result: Any = None, execution_generation: int | None = None, local_profiles: Iterable[str]) -> PublicationPlan:
    """Plan idempotent room effects for one terminal driver task.

    A newer user event in the same thread supersedes a late result: the task stays terminal in driver state,
    but only a deterministic cancellation is published so stale prose and its watermark cannot hide it.
    """
    room = validate_room(room_value, local_profiles=local_profiles)
    validated = _validated_events(events, room=room)
    for failed, message in (
        (task.identity.room_id != room.room_id, "task belongs to a different room"),
        (task.member not in room.members, "task member is not in the frozen roster"),
        (status not in _TERMINAL_EFFECTS, "invalid terminal publication status")):
        if failed:
            raise DiscussionValidationError(message)
    if status == "deferred":
        _bounded_int(execution_generation, message="deferred publication requires an execution generation", low=1)
    newer_same_thread = any(
        event.kind == "message.user" and event.seq > task.seen_through_seq
        and event.payload.get("thread_id") == task.identity.thread_id for event in validated)
    effective_status: TerminalKind = ("cancelled" if newer_same_thread and status != "deferred" else status)
    digest = task.identity.task_id.removeprefix("dtask:")
    terminal_event_id = (
        f"ddeferred:{digest}:g{execution_generation}" if effective_status == "deferred" else f"dterminal:{digest}")
    extra, effects = _TERMINAL_EFFECTS[effective_status](
        result, task=task, room=room, message_event_id=f"dmessage:{digest}", newer_same_thread=newer_same_thread,
        execution_generation=execution_generation)
    terminal_kind = f"turn.{effective_status}"
    effects.append(EventPlan(
        event_id=terminal_event_id, kind=terminal_kind, actor={"kind": "gateway", "id": room.gateway_id},
        payload={**_turn_coordinates(task), "seen_through_seq": task.seen_through_seq, **extra},
        authority_gateway_id=room.gateway_id, authority_epoch=room.authority_epoch))
    return PublicationPlan(task_id=task.identity.task_id, terminal_kind=terminal_kind, events=tuple(effects))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
