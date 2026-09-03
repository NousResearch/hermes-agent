"""Deterministic policy for same-gateway hosted-room Discussions.

This module translates a frozen local member roster and the complete typed room
log into one next driver task.  It performs no I/O, starts no workers, and knows
nothing about transports or model runtimes.  Callers persist the returned task
with :mod:`gateway.hosted_room_driver` and append publication plans with
:mod:`gateway.hosted_rooms`.

The unpublished driver payload intentionally remains unchanged.  Discussion
coordinates live in deterministic ``TaskIdentity`` values and typed terminal
events; a restart can therefore reconstruct a task without widening the driver
schema.  Callers must reconcile terminal driver rows into publication plans
before asking for the next task.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Literal

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms


MAX_DISCUSSION_MEMBERS = 6
MIN_DISCUSSION_MEMBERS = 2
MAX_DISCUSSION_ROUNDS = 3
MAX_DISCUSSION_MESSAGES = 10
MAX_DISCUSSION_DELTA_LINES = 24
MAX_USER_TEXT_BYTES = 64 * 1024
MAX_MEMBER_TEXT_BYTES = 64 * 1024
_TRUNCATED_REPLY_NOTICE = (
    "\n\n[Reply truncated. Ask the Bot to share the full result as a file.]"
)

DecisionStatus = Literal["idle", "task", "settled", "bounded"]
TerminalKind = Literal["settled", "failed", "cancelled", "deferred"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_MENTION_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9._:-]*)", re.IGNORECASE)
_FENCE_START_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BARE_URI_START_RE = re.compile(r"(?i)(?:https?://|ftp://|mailto:|www\.)")
_TURN_ID_RE = re.compile(
    r"^d(?P<source>[1-9][0-9]*)\.r(?P<round>[0-2])\."
    r"p(?P<position>[0-5])\.s(?P<seen>[1-9][0-9]*)\."
    r"m(?P<member>[0-9a-f]{24})$"
)

_MEMBER_FIELDS = frozenset(
    {"member_id", "profile", "handle", "display_name", "target"}
)
_LOCAL_TARGET_FIELDS = frozenset({"kind", "profile"})
_PEER_TARGET_FIELDS = frozenset(
    {"kind", "peer_id", "installation_id", "profile", "capability_digest"}
)
_REMOTE_MEMBER_FIELDS = frozenset({
    "connectionId",
    "connectionKind",
    "connectionLabel",
    "connection_id",
    "connection_kind",
    "connection_label",
    "remoteSource",
    "route",
    "sourceMissing",
    "sourceReachable",
    "sourceScoped",
    "targetProfile",
    "target_profile",
})
_USER_PAYLOAD_FIELDS = frozenset({"text", "thread_id"})
_MEMBER_MESSAGE_FIELDS = frozenset({
    "discussion_event_id",
    "member_id",
    "member_index",
    "round_index",
    "task_id",
    "text",
    "thread_id",
    "turn_id",
})
_TERMINAL_COMMON_FIELDS = frozenset({
    "discussion_event_id",
    "member_id",
    "member_index",
    "round_index",
    "seen_through_seq",
    "task_id",
    "thread_id",
    "turn_id",
})
_TERMINAL_EXTRA_FIELDS = {
    "turn.settled": frozenset({"message_event_id", "passed"}),
    "turn.failed": frozenset({"error"}),
    "turn.cancelled": frozenset({"reason"}),
    "turn.deferred": frozenset({"execution_generation", "reason"}),
}
_TERMINAL_OPTIONAL_FIELDS = {
    "turn.failed": frozenset({"reason_code"}),
}
_TERMINAL_EVENT_KINDS = frozenset(_TERMINAL_EXTRA_FIELDS)
_ROOM_ACTIVITY_FIELDS = frozenset({
    "status",
    "reason_code",
    "thread_id",
    "discussion_event_id",
})
_ROOM_STOP_FIELDS = frozenset({"cancel_id"})


class DiscussionPolicyError(ValueError):
    """Base class for invalid policy input or unreconstructable state."""


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
        """Return keyword arguments accepted by ``append_event``."""

        return {
            "room_id": room_id,
            "event_id": self.event_id,
            "kind": self.kind,
            "actor": dict(self.actor),
            "payload": dict(self.payload),
            "authority_gateway_id": self.authority_gateway_id,
            "authority_epoch": self.authority_epoch,
        }


@dataclass(frozen=True)
class PublicationPlan:
    """Ordered visible and terminal effects for one driver task."""

    task_id: str
    terminal_kind: str
    events: tuple[EventPlan, ...]


@dataclass(frozen=True)
class _ValidatedEvent:
    raw: Mapping[str, Any]
    seq: int
    event_id: str
    kind: str
    actor: Mapping[str, Any]
    payload: Mapping[str, Any]


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise DiscussionValidationError(f"{label} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > driver.MAX_IDENTIFIER_CHARS
        or not _IDENTIFIER_RE.fullmatch(normalized)
    ):
        raise DiscussionValidationError(f"invalid {label}")
    return normalized


def _positive_int(value: Any, *, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DiscussionValidationError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise DiscussionValidationError(f"{label} must be at most {maximum}")
    return value


def _zero_based_int(value: Any, *, label: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise DiscussionValidationError(
            f"{label} must be an integer between 0 and {maximum}"
        )
    return value


def _exact_fields(
    value: Any,
    *,
    label: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiscussionValidationError(f"{label} must be an object")
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise DiscussionValidationError(
            f"{label} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise DiscussionValidationError(
            f"{label} has unknown fields: {', '.join(sorted(unknown))}"
        )
    return value


def validate_user_payload(value: Any) -> dict[str, Any]:
    """Validate and normalize the exact ``message.user`` Discussion payload."""

    payload = _exact_fields(
        value,
        label="user payload",
        required=_USER_PAYLOAD_FIELDS,
    )
    text = payload["text"]
    if not isinstance(text, str):
        raise DiscussionValidationError("user payload text must be a string")
    text = text.strip()
    if not text:
        raise DiscussionValidationError("user payload text must not be empty")
    if len(text.encode("utf-8")) > MAX_USER_TEXT_BYTES:
        raise DiscussionValidationError("user payload text is too large")
    thread_id = _identifier(payload["thread_id"], label="thread_id")
    return {"text": text, "thread_id": thread_id}


def _validate_member_target(
    value: Any,
    *,
    profile: str,
    known_profiles: set[str],
    index: int,
) -> dict[str, Any]:
    if value is None:
        if profile not in known_profiles:
            raise DiscussionValidationError(
                f"member {index} profile '{profile}' is not local to this gateway"
            )
        return {"kind": "local", "profile": profile}
    if not isinstance(value, Mapping):
        raise DiscussionValidationError(f"member {index} target must be an object")
    kind = value.get("kind")
    if kind == "local":
        target = _exact_fields(
            value,
            label=f"member {index} local target",
            required=_LOCAL_TARGET_FIELDS,
        )
        target_profile = _identifier(
            target["profile"], label=f"member {index} target profile"
        )
        if target_profile != profile or profile not in known_profiles:
            raise DiscussionValidationError(
                f"member {index} local target does not match a local profile"
            )
        return {"kind": "local", "profile": profile}
    if kind == "peer":
        target = _exact_fields(
            value,
            label=f"member {index} peer target",
            required=_PEER_TARGET_FIELDS,
        )
        target_profile = _identifier(
            target["profile"], label=f"member {index} target profile"
        )
        if target_profile != profile:
            raise DiscussionValidationError(
                f"member {index} peer target profile does not match member profile"
            )
        capability_digest = target["capability_digest"]
        if (
            not isinstance(capability_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", capability_digest)
        ):
            raise DiscussionValidationError(
                f"member {index} capability_digest must be a sha256 digest"
            )
        return {
            "kind": "peer",
            "peer_id": _identifier(
                target["peer_id"], label=f"member {index} peer_id"
            ),
            "installation_id": _identifier(
                target["installation_id"],
                label=f"member {index} installation_id",
            ),
            "profile": target_profile,
            "capability_digest": capability_digest,
        }
    raise DiscussionValidationError(
        f"member {index} target kind must be local or peer"
    )


def validate_roster(
    value: Any,
    *,
    local_profiles: Iterable[str],
) -> tuple[DiscussionMember, ...]:
    """Validate a frozen 2-6 member roster of profiles on this gateway."""

    if not isinstance(value, list):
        raise DiscussionValidationError("members must be a list")
    if not MIN_DISCUSSION_MEMBERS <= len(value) <= MAX_DISCUSSION_MEMBERS:
        raise DiscussionValidationError(
            f"members must contain between {MIN_DISCUSSION_MEMBERS} and "
            f"{MAX_DISCUSSION_MEMBERS} entries"
        )

    known_profiles = {
        _identifier(profile, label="local profile") for profile in local_profiles
    }
    members: list[DiscussionMember] = []
    targets: set[str] = set()
    handles: set[str] = set()
    member_ids: set[str] = set()

    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise DiscussionValidationError(f"member {index} must be an object")
        remote_fields = frozenset(raw) & _REMOTE_MEMBER_FIELDS
        if remote_fields:
            raise DiscussionValidationError(
                f"member {index} contains cross-gateway fields: "
                f"{', '.join(sorted(remote_fields))}"
            )
        member = _exact_fields(
            raw,
            label=f"member {index}",
            required=frozenset({"member_id", "profile", "handle"}),
            optional=frozenset({"display_name", "target"}),
        )
        member_id = _identifier(member["member_id"], label=f"member {index} id")
        profile = _identifier(member["profile"], label=f"member {index} profile")
        handle = _identifier(member["handle"], label=f"member {index} handle")
        target = _validate_member_target(
            member.get("target"),
            profile=profile,
            known_profiles=known_profiles,
            index=index,
        )
        display_name = member.get("display_name", "")
        if not isinstance(display_name, str):
            raise DiscussionValidationError(
                f"member {index} display_name must be a string"
            )
        display_name = display_name.strip()
        if len(display_name) > hosted_rooms.MAX_ACTOR_LABEL_CHARS:
            raise DiscussionValidationError(f"member {index} display_name is too long")

        target_key = json.dumps(
            target,
            sort_keys=True,
            separators=(",", ":"),
        ).casefold()
        handle_key = handle.casefold()
        member_key = member_id.casefold()
        if target_key in targets:
            if target.get("kind") == "local":
                raise DiscussionValidationError("member profiles must be unique")
            raise DiscussionValidationError("member targets must be unique")
        if handle_key in handles or handle_key in {"all", "everyone"}:
            raise DiscussionValidationError(
                "member handles must be unique and cannot reserve @all or @everyone"
            )
        if member_key in member_ids:
            raise DiscussionValidationError("member ids must be unique")
        targets.add(target_key)
        handles.add(handle_key)
        member_ids.add(member_key)
        members.append(
            DiscussionMember(
                member_id=member_id,
                profile=profile,
                handle=handle,
                display_name=display_name,
                target=target,
            )
        )
    return tuple(members)


def validate_room(
    value: Any,
    *,
    local_profiles: Iterable[str],
) -> DiscussionRoom:
    """Project a hosted-room row into the strict same-gateway policy shape."""

    if not isinstance(value, Mapping):
        raise DiscussionValidationError("room must be an object")
    if value.get("disbanded_at") is not None:
        raise DiscussionValidationError("room is disbanded")
    room_id = _identifier(value.get("room_id"), label="room_id")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise DiscussionValidationError("room name must be a non-empty string")
    name = name.strip()
    if len(name) > hosted_rooms.MAX_ROOM_NAME_CHARS:
        raise DiscussionValidationError("room name is too long")
    gateway_id = _identifier(
        value.get("authority_gateway_id"),
        label="authority_gateway_id",
    )
    authority_epoch = _positive_int(
        value.get("authority_epoch"),
        label="authority_epoch",
    )
    members = validate_roster(value.get("members"), local_profiles=local_profiles)
    return DiscussionRoom(
        room_id=room_id,
        name=name,
        members=members,
        gateway_id=gateway_id,
        authority_epoch=authority_epoch,
    )


def is_pass_text(value: Any) -> bool:
    """Return whether a settled member result is Discussion silence."""

    text = str(value or "").strip()
    return (
        not text
        or re.fullmatch(r"\(?\s*pass\s*\)?\.?", text, re.IGNORECASE) is not None
    )


def resolve_mentions(
    texts: Iterable[str],
    members: Sequence[DiscussionMember],
    *,
    default_all: bool = True,
) -> tuple[DiscussionMember, ...]:
    """Resolve member handles deterministically against the frozen roster."""

    return _mention_resolution(texts, members, default_all=default_all)[0]


def _masked_markdown_code(value: str) -> str:
    """Replace Markdown code with spaces while preserving mention boundaries."""

    chars = list(value)

    def mask(start: int, end: int) -> None:
        for index in range(start, end):
            if chars[index] not in "\r\n":
                chars[index] = " "

    offset = 0
    fence: tuple[str, int, int] | None = None
    for line in value.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        match = _FENCE_START_RE.match(body)
        if fence is None and match is not None:
            marker = match.group(1)
            fence = (marker[0], len(marker), offset)
        elif fence is not None and match is not None:
            marker = match.group(1)
            trailing = body[match.end():]
            if (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not trailing.strip(" \t")
            ):
                mask(fence[2], offset + len(line))
                fence = None
        offset += len(line)
    if fence is not None:
        mask(fence[2], len(value))

    visible = "".join(chars)
    chars = list(visible)
    escaped = _escaped_positions(visible)
    index = 0
    while index < len(visible):
        if visible[index] != "`":
            index += 1
            continue
        if escaped[index]:
            index += 1
            continue
        run_end = index + 1
        while run_end < len(visible) and visible[run_end] == "`":
            run_end += 1
        run_length = run_end - index
        cursor = run_end
        closing_end = None
        while cursor < len(visible):
            if visible[cursor] != "`":
                cursor += 1
                continue
            candidate_end = cursor + 1
            while candidate_end < len(visible) and visible[candidate_end] == "`":
                candidate_end += 1
            if candidate_end - cursor == run_length:
                closing_end = candidate_end
                break
            cursor = candidate_end
        mask(index, closing_end if closing_end is not None else len(visible))
        index = closing_end if closing_end is not None else len(visible)
    return "".join(chars)


def _escaped_positions(value: str) -> tuple[bool, ...]:
    """Return whether each character has an odd preceding backslash run."""

    result: list[bool] = []
    backslashes = 0
    for char in value:
        result.append(backslashes % 2 == 1)
        backslashes = backslashes + 1 if char == "\\" else 0
    return tuple(result)


def _masked_markdown_destinations(value: str) -> str:
    """Hide inline link destinations while retaining their visible labels."""

    chars = list(value)
    escaped = _escaped_positions(value)
    brackets: list[int] = []
    index = 0
    while index < len(value):
        if escaped[index]:
            index += 1
            continue
        if value[index] == "[":
            brackets.append(index)
            index += 1
            continue
        if value[index] != "]" or not brackets:
            index += 1
            continue
        brackets.pop()
        if index + 1 >= len(value) or value[index + 1] != "(" or escaped[index + 1]:
            index += 1
            continue
        depth = 1
        cursor = index + 2
        angle = False
        quote = ""
        title_position = False
        while cursor < len(value):
            if escaped[cursor]:
                cursor += 1
                continue
            char = value[cursor]
            if angle:
                if char == ">":
                    angle = False
            elif quote:
                if char == quote:
                    quote = ""
            elif char == "<":
                angle = True
            elif char in {'"', "'"} and title_position:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    for masked in range(index + 1, cursor + 1):
                        if chars[masked] not in "\r\n":
                            chars[masked] = " "
                    index = cursor
                    break
            if not angle and not quote:
                title_position = depth == 1 and char in " \t\r\n"
            cursor += 1
        if depth:
            for masked in range(index + 1, len(value)):
                if chars[masked] not in "\r\n":
                    chars[masked] = " "
            break
        index += 1
    return "".join(chars)


def _masked_bare_uris(value: str) -> str:
    chars = list(value)
    cursor = 0
    while match := _BARE_URI_START_RE.search(value, cursor):
        end = match.end()
        parentheses = 0
        brackets = 0
        while end < len(value) and value[end] not in "\r\n\t <>{}":
            char = value[end]
            if char == "(":
                parentheses += 1
            elif char == ")":
                if end + 1 < len(value) and value[end + 1] == "@" and parentheses == 0:
                    break
                parentheses = max(0, parentheses - 1)
            elif char == "[":
                brackets += 1
            elif char == "]":
                if end + 1 < len(value) and value[end + 1] == "@" and brackets == 0:
                    break
                brackets = max(0, brackets - 1)
            elif (
                char in {",", "!"}
                and end + 1 < len(value)
                and value[end + 1] == "@"
            ):
                break
            end += 1
        for index in range(match.start(), end):
            if chars[index] not in "\r\n":
                chars[index] = " "
        cursor = max(end, match.end())
    return "".join(chars)


class _VisibleHTMLParser(HTMLParser):
    """Collect only rendered text from Markdown's embedded HTML."""

    _HIDDEN = frozenset({"script", "style", "template"})
    _BREAKS = frozenset({
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "caption",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "legend",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._HIDDEN:
            self.hidden.append(tag)
        elif not self.hidden and tag in self._BREAKS:
            self.parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._HIDDEN:
            self.hidden.append(tag)
            if tag in {"script", "style"}:
                self.set_cdata_mode(tag)
        elif not self.hidden and tag in self._BREAKS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            if self.hidden[-1] == tag:
                self.hidden.pop()
            return
        if tag in self._BREAKS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _visible_html_text(value: str) -> str:
    parser = _VisibleHTMLParser()
    parser.feed(value)
    parser.close()
    return "".join(parser.parts)


def _has_mention_boundary(value: str, index: int) -> bool:
    if index == 0:
        return True
    previous = value[index - 1]
    category = unicodedata.category(previous)
    return previous not in "._%+\\-/:?#=&" and category[0] not in {"L", "M", "N"}


def _mention_resolution(
    texts: Iterable[str],
    members: Sequence[DiscussionMember],
    *,
    default_all: bool,
) -> tuple[tuple[DiscussionMember, ...], bool]:
    """Return resolved members and whether an explicit token was unresolved."""

    by_handle = {member.handle.casefold(): member for member in members}
    candidates = tuple(sorted(
        (*by_handle, "all", "everyone"),
        key=len,
        reverse=True,
    ))
    mentioned: set[str] = set()
    everyone = False
    unresolved = False
    for text in texts:
        visible = _masked_bare_uris(
            _masked_markdown_destinations(
                _visible_html_text(
                    _masked_markdown_code(str(text or ""))
                )
            )
        )
        for match in _MENTION_RE.finditer(visible):
            if not _has_mention_boundary(visible, match.start()):
                continue
            token = match.group(1).casefold()
            handle = next(
                (
                    candidate
                    for candidate in candidates
                    if token.startswith(candidate)
                    and set(token[len(candidate):]) <= {".", ":"}
                ),
                "",
            )
            if handle in by_handle:
                mentioned.add(handle)
            elif handle in {"all", "everyone"}:
                everyone = True
            else:
                unresolved = True
    if unresolved:
        return (), True
    if everyone or (default_all and not mentioned):
        return tuple(members), False
    resolved = tuple(
        member for member in members if member.handle.casefold() in mentioned
    )
    return resolved, False


def _unaddressed_member_mentions(
    messages: Sequence[_ValidatedEvent],
    room: DiscussionRoom,
) -> tuple[tuple[DiscussionMember, ...], bool]:
    """Return peers explicitly cited by a Bot and not heard from afterward."""

    cited_at: dict[str, int] = {}
    last_post_at: dict[str, int] = {}
    unresolved = False
    for event in messages:
        if event.kind != "message.member":
            continue
        speaker_id = str(event.payload["member_id"])
        last_post_at[speaker_id] = event.seq
        cited, message_unresolved = _mention_resolution(
            (str(event.payload["text"]),),
            room.members,
            default_all=False,
        )
        unresolved = unresolved or message_unresolved
        for member in cited:
            if member.member_id != speaker_id:
                cited_at[member.member_id] = event.seq
    return (
        tuple(
            member
            for member in room.members
            if member.member_id in cited_at
            and last_post_at.get(member.member_id, 0) <= cited_at[member.member_id]
        ),
        unresolved,
    )


def _validate_event(
    raw: Any,
    *,
    room: DiscussionRoom,
    previous_seq: int,
) -> _ValidatedEvent:
    if not isinstance(raw, Mapping):
        raise DiscussionValidationError("room event must be an object")
    if raw.get("room_id") != room.room_id:
        raise DiscussionValidationError("room event belongs to a different room")
    seq = _positive_int(raw.get("seq"), label="event seq")
    if seq <= previous_seq:
        raise DiscussionValidationError("room events must be in strict sequence order")
    event_id = _identifier(raw.get("event_id"), label="event_id")
    kind = raw.get("kind")
    if not isinstance(kind, str):
        raise DiscussionValidationError("event kind must be a string")
    actor = raw.get("actor")
    if not isinstance(actor, Mapping):
        raise DiscussionValidationError("event actor must be an object")
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        raise DiscussionValidationError("event payload must be an object")

    if kind == "message.user":
        payload = validate_user_payload(payload)
        if actor.get("kind") != "user":
            raise DiscussionValidationError("message.user requires a user actor")
    elif kind == "message.member":
        if raw.get("authority_epoch") != room.authority_epoch:
            raise DiscussionValidationError(
                "message.member authority epoch does not match the room"
            )
        _validate_member_message(payload, actor=actor, room=room)
    elif kind in _TERMINAL_EVENT_KINDS:
        if raw.get("authority_epoch") != room.authority_epoch:
            raise DiscussionValidationError(
                f"{kind} authority epoch does not match the room"
            )
        _validate_terminal_event(kind, payload, actor=actor, room=room)
    elif kind == "room.activity":
        if raw.get("authority_epoch") != room.authority_epoch:
            raise DiscussionValidationError(
                "room.activity authority epoch does not match the room"
            )
        _exact_fields(
            payload,
            label="room.activity payload",
            required=_ROOM_ACTIVITY_FIELDS,
        )
        if payload.get("status") not in {"settled", "bounded"}:
            raise DiscussionValidationError("invalid room.activity status")
        _identifier(payload.get("reason_code"), label="reason_code")
        _identifier(payload.get("thread_id"), label="thread_id")
        _identifier(payload.get("discussion_event_id"), label="discussion_event_id")
        if actor.get("kind") != "gateway" or actor.get("id") != room.gateway_id:
            raise DiscussionValidationError("room.activity requires the room gateway")
    elif kind == "room.stop_requested":
        if raw.get("authority_epoch") != room.authority_epoch:
            raise DiscussionValidationError(
                "room.stop_requested authority epoch does not match the room"
            )
        _exact_fields(
            payload,
            label="room.stop_requested payload",
            required=_ROOM_STOP_FIELDS,
        )
        _identifier(payload.get("cancel_id"), label="cancel_id")
        if actor.get("kind") != "gateway" or actor.get("id") != room.gateway_id:
            raise DiscussionValidationError(
                "room.stop_requested requires the room gateway"
            )

    return _ValidatedEvent(
        raw=raw,
        seq=seq,
        event_id=event_id,
        kind=kind,
        actor=actor,
        payload=payload,
    )


def _member_by_id(room: DiscussionRoom, member_id: Any) -> DiscussionMember:
    normalized = _identifier(member_id, label="member_id")
    for member in room.members:
        if member.member_id == normalized:
            return member
    raise DiscussionValidationError(f"unknown Discussion member '{normalized}'")


def _validate_turn_coordinates(
    payload: Mapping[str, Any], room: DiscussionRoom
) -> None:
    _member_by_id(room, payload.get("member_id"))
    member_index = _zero_based_int(
        payload.get("member_index"),
        label="member_index",
        maximum=MAX_DISCUSSION_MEMBERS - 1,
    )
    round_index = _zero_based_int(
        payload.get("round_index"),
        label="round_index",
        maximum=MAX_DISCUSSION_ROUNDS - 1,
    )
    thread_id = _identifier(payload.get("thread_id"), label="thread_id")
    task_id = _identifier(payload.get("task_id"), label="task_id")
    turn_id = _identifier(payload.get("turn_id"), label="turn_id")
    discussion_event_id = _identifier(
        payload.get("discussion_event_id"),
        label="discussion_event_id",
    )
    del member_index, round_index, thread_id, task_id, turn_id, discussion_event_id


def _validate_member_message(
    payload: Mapping[str, Any],
    *,
    actor: Mapping[str, Any],
    room: DiscussionRoom,
) -> None:
    _exact_fields(
        payload,
        label="message.member payload",
        required=_MEMBER_MESSAGE_FIELDS,
    )
    _validate_turn_coordinates(payload, room)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or is_pass_text(text):
        raise DiscussionValidationError("message.member text must be a non-pass string")
    member = _member_by_id(room, payload.get("member_id"))
    expected_connection = None
    if member.target and member.target.get("kind") == "peer":
        expected_connection = member.target.get("peer_id")
    if (
        actor.get("kind") != "member"
        or actor.get("id") != member.member_id
        or actor.get("profile") != member.profile
        or actor.get("connection_id") != expected_connection
    ):
        raise DiscussionValidationError("message.member actor does not match roster")


def _validate_terminal_event(
    kind: str,
    payload: Mapping[str, Any],
    *,
    actor: Mapping[str, Any],
    room: DiscussionRoom,
) -> None:
    required = _TERMINAL_COMMON_FIELDS | _TERMINAL_EXTRA_FIELDS[kind]
    _exact_fields(
        payload,
        label=f"{kind} payload",
        required=required,
        optional=_TERMINAL_OPTIONAL_FIELDS.get(kind, frozenset()),
    )
    _validate_turn_coordinates(payload, room)
    _positive_int(payload.get("seen_through_seq"), label="seen_through_seq")
    if (
        actor.get("kind") != "gateway"
        or actor.get("id") != room.gateway_id
        or actor.get("connection_id") is not None
    ):
        raise DiscussionValidationError(f"{kind} requires a gateway actor")
    if kind == "turn.settled":
        if not isinstance(payload.get("passed"), bool):
            raise DiscussionValidationError("turn.settled passed must be a boolean")
        message_event_id = payload.get("message_event_id")
        if payload["passed"]:
            if message_event_id is not None:
                raise DiscussionValidationError(
                    "a passed turn cannot reference a member message"
                )
        else:
            _identifier(message_event_id, label="message_event_id")
    else:
        field = "error" if kind == "turn.failed" else "reason"
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise DiscussionValidationError(f"{kind} {field} must be non-empty")
        if kind == "turn.deferred":
            _positive_int(
                payload.get("execution_generation"),
                label="execution_generation",
            )
        if kind == "turn.failed" and "reason_code" in payload:
            from tools.bot_failure_reasons import ALL_REASONS

            if payload["reason_code"] not in ALL_REASONS:
                raise DiscussionValidationError(
                    "turn.failed reason_code must use the shared failure vocabulary"
                )


def _validated_events(
    events: Sequence[Mapping[str, Any]],
    *,
    room: DiscussionRoom,
) -> tuple[_ValidatedEvent, ...]:
    validated: list[_ValidatedEvent] = []
    previous_seq = 0
    event_ids: set[str] = set()
    for raw in events:
        event = _validate_event(raw, room=room, previous_seq=previous_seq)
        if event.event_id in event_ids:
            raise DiscussionValidationError("room event ids must be unique")
        validated.append(event)
        previous_seq = event.seq
        event_ids.add(event.event_id)
    return tuple(validated)


def _discussion_user_events(
    events: Sequence[_ValidatedEvent],
) -> tuple[_ValidatedEvent, ...]:
    return tuple(event for event in events if event.kind == "message.user")


def _message_events(
    events: Sequence[_ValidatedEvent],
    *,
    thread_id: str | None = None,
    maximum_seq: int | None = None,
) -> tuple[_ValidatedEvent, ...]:
    result = []
    for event in events:
        if event.kind not in {"message.user", "message.member"}:
            continue
        if thread_id is not None and event.payload.get("thread_id") != thread_id:
            continue
        if maximum_seq is not None and event.seq > maximum_seq:
            continue
        result.append(event)
    return tuple(result)


def derive_member_watermarks(
    room_value: Any,
    events: Sequence[Mapping[str, Any]],
    *,
    local_profiles: Iterable[str],
) -> dict[tuple[str, str], int]:
    """Derive ``(thread_id, member_id)`` watermarks from terminal events."""

    room = validate_room(room_value, local_profiles=local_profiles)
    validated = _validated_events(events, room=room)
    return _derive_member_watermarks(validated)


def _derive_member_watermarks(
    events: Sequence[_ValidatedEvent],
) -> dict[tuple[str, str], int]:
    messages_by_id = {
        event.event_id: event for event in events if event.kind == "message.member"
    }
    terminal_by_task: dict[str, _ValidatedEvent] = {}
    watermarks: dict[tuple[str, str], int] = {}
    for event in events:
        if event.kind not in _TERMINAL_EVENT_KINDS:
            continue
        task_id = str(event.payload["task_id"])
        previous = terminal_by_task.get(task_id)
        if previous is not None:
            if previous.kind != "turn.deferred":
                raise DiscussionValidationError(
                    f"task '{task_id}' has more than one terminal room event"
                )
            if event.kind == "turn.deferred" and int(
                event.payload["execution_generation"]
            ) <= int(previous.payload["execution_generation"]):
                raise DiscussionValidationError(
                    f"task '{task_id}' deferral generation did not advance"
                )
        terminal_by_task[task_id] = event
        key = (str(event.payload["thread_id"]), str(event.payload["member_id"]))
        watermark = int(event.payload["seen_through_seq"])
        if event.kind == "turn.settled" and not event.payload["passed"]:
            message_id = str(event.payload["message_event_id"])
            message = messages_by_id.get(message_id)
            if (
                message is None
                or message.payload.get("task_id") != task_id
                or message.payload.get("member_id") != event.payload.get("member_id")
                or message.payload.get("thread_id") != event.payload.get("thread_id")
            ):
                raise DiscussionValidationError(
                    "turn.settled references no matching member message"
                )
            watermark = max(watermark, message.seq)
        watermarks[key] = max(watermarks.get(key, 0), watermark)
    return watermarks


def _member_digest(member: DiscussionMember) -> str:
    target = json.dumps(
        member.target or {"kind": "local", "profile": member.profile},
        sort_keys=True,
        separators=(",", ":"),
    )
    seed = (
        f"{member.member_id}\0{member.profile}\0{member.handle}\0{target}"
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def _rotate(
    members: Sequence[DiscussionMember], round_index: int
) -> tuple[DiscussionMember, ...]:
    if len(members) < 2:
        return tuple(members)
    shift = round_index % len(members)
    return tuple((*members[shift:], *members[:shift]))


def _format_message(event: _ValidatedEvent, room: DiscussionRoom) -> str:
    text = str(event.payload["text"])
    if event.kind == "message.user":
        return f"User (user): {text}"
    member = _member_by_id(room, event.payload["member_id"])
    return f"@{member.handle}: {text}"


def _truncate_utf8_text(value: Any, *, max_bytes: int, suffix: str = "") -> str:
    text = str(value or "")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    suffix_bytes = suffix.encode("utf-8")
    prefix = encoded[: max(0, max_bytes - len(suffix_bytes))]
    while prefix:
        try:
            return prefix.decode("utf-8") + suffix
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return suffix.strip()


def _build_prompt(
    *,
    room: DiscussionRoom,
    member: DiscussionMember,
    messages: Sequence[_ValidatedEvent],
    watermark: int,
    seen_through_seq: int,
) -> str:
    delta = [event for event in messages if watermark < event.seq <= seen_through_seq][
        -MAX_DISCUSSION_DELTA_LINES:
    ]
    peers = ", ".join(
        f"@{candidate.handle}"
        for candidate in room.members
        if candidate.member_id != member.member_id
    )
    opening = [
        f'[Discussion: "{room.name}"] You are @{member.handle}, one participant '
        f"with {peers or 'no other members'} and the user.",
        "",
        "New messages in this thread since your last turn (oldest first):",
    ]
    rules = [
        "",
        "Rules for this Discussion:",
        "- Reply with one conversational message only when you have something new worth adding.",
        '- If you have nothing new to add, reply with exactly "(pass)".',
        "- Mention a teammate by handle to pull them into the next round; do not repeat points already made.",
        "- Never reveal content from private conversations. Your reply is published verbatim.",
    ]
    fixed_bytes = len("\n".join([*opening, *rules]).encode("utf-8"))
    available = max(0, driver.MAX_PROMPT_BYTES - fixed_bytes - 1)
    selected: list[str] = []
    omitted = False
    for event in reversed(delta):
        line = f"  {_format_message(event, room)}"
        line_bytes = len(line.encode("utf-8")) + 1
        if line_bytes <= available:
            selected.append(line)
            available -= line_bytes
            continue
        if not selected and available > 32:
            selected.append(_truncate_utf8_text(line, max_bytes=available))
        omitted = True
        break
    selected.reverse()
    if omitted:
        selected.insert(0, "  [Earlier content omitted to fit this turn.]")
    prompt = "\n".join([*opening, *selected, *rules])
    if len(prompt.encode("utf-8")) > driver.MAX_PROMPT_BYTES:
        raise DiscussionValidationError("Discussion prompt exceeds the driver limit")
    return prompt


def _turn_id(
    *,
    source_event_seq: int,
    round_index: int,
    member_index: int,
    seen_through_seq: int,
    member: DiscussionMember,
) -> str:
    return (
        f"d{source_event_seq}.r{round_index}.p{member_index}."
        f"s{seen_through_seq}.m{_member_digest(member)}"
    )


def _task_id(
    *,
    room: DiscussionRoom,
    discussion_event: _ValidatedEvent,
    member: DiscussionMember,
    member_index: int,
    round_index: int,
    seen_through_seq: int,
    prompt: str,
) -> str:
    seed = json.dumps(
        {
            "discussion_event_id": discussion_event.event_id,
            "member_id": member.member_id,
            "member_index": member_index,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "room_id": room.room_id,
            "round_index": round_index,
            "seen_through_seq": seen_through_seq,
            "source_event_seq": discussion_event.seq,
            "thread_id": discussion_event.payload["thread_id"],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"dtask:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:48]}"


def _make_task_plan(
    *,
    room: DiscussionRoom,
    discussion_event: _ValidatedEvent,
    member: DiscussionMember,
    member_index: int,
    round_index: int,
    seen_through_seq: int,
    prompt: str,
) -> DiscussionTaskPlan:
    turn_id = _turn_id(
        source_event_seq=discussion_event.seq,
        round_index=round_index,
        member_index=member_index,
        seen_through_seq=seen_through_seq,
        member=member,
    )
    task_id = _task_id(
        room=room,
        discussion_event=discussion_event,
        member=member,
        member_index=member_index,
        round_index=round_index,
        seen_through_seq=seen_through_seq,
        prompt=prompt,
    )
    identity = driver.TaskIdentity(
        room_id=room.room_id,
        task_id=task_id,
        thread_id=str(discussion_event.payload["thread_id"]),
        turn_id=turn_id,
    )
    payload = {
        "target_member_id": member.member_id,
        "target_profile": member.profile,
        "prompt": prompt,
        "source_event_seq": discussion_event.seq,
    }
    return DiscussionTaskPlan(
        identity=identity,
        payload=payload,
        discussion_event_id=discussion_event.event_id,
        member=member,
        member_index=member_index,
        round_index=round_index,
        seen_through_seq=seen_through_seq,
    )


def plan_next_task(
    room_value: Any,
    events: Sequence[Mapping[str, Any]],
    *,
    local_profiles: Iterable[str],
    initial_watermarks: Mapping[tuple[str, str], int] | None = None,
) -> DiscussionDecision:
    """Replay the complete room log and return at most one next member task."""

    room = validate_room(room_value, local_profiles=local_profiles)
    validated = _validated_events(events, room=room)
    user_events = _discussion_user_events(validated)
    stopped_through_seq = max(
        (event.seq for event in validated if event.kind == "room.stop_requested"),
        default=0,
    )
    completed_discussion_ids = {
        str(event.payload["discussion_event_id"])
        for event in validated
        if event.kind == "room.activity"
        and event.payload.get("status") in {"settled", "bounded"}
    }
    latest_by_thread: dict[str, _ValidatedEvent] = {}
    for event in user_events:
        latest_by_thread[str(event.payload["thread_id"])] = event
    pending_user_events = tuple(
        event
        for event in sorted(latest_by_thread.values(), key=lambda item: item.seq)
        if event.seq > stopped_through_seq
        and event.event_id not in completed_discussion_ids
    )
    if not pending_user_events:
        return DiscussionDecision(status="idle", reason="no_pending_user_event")

    discussion = pending_user_events[0]
    thread_id = str(discussion.payload["thread_id"])
    initial_responders, unresolved_mention = _mention_resolution(
        (str(discussion.payload["text"]),),
        room.members,
        default_all=True,
    )
    if unresolved_mention:
        return DiscussionDecision(
            status="settled",
            reason="unresolved_mention",
            discussion_event_id=discussion.event_id,
            source_event_seq=discussion.seq,
            thread_id=thread_id,
        )
    committed_member_message_ids = {
        str(event.payload["message_event_id"])
        for event in validated
        if event.kind == "turn.settled"
        and event.payload.get("message_event_id") is not None
    }
    # Publication writes the visible member message before the terminal event.
    # A crash in that gap leaves the message in the log, but it is not committed
    # policy input yet: ignoring it reproduces the original task coordinates so
    # the caller can inspect the terminal driver row and finish publication.
    thread_messages = tuple(
        event
        for event in _message_events(validated, thread_id=thread_id)
        if event.kind == "message.user"
        or event.event_id in committed_member_message_ids
    )
    discussion_messages = tuple(
        event for event in thread_messages if event.seq >= discussion.seq
    )
    member_messages = tuple(
        event
        for event in thread_messages
        if event.kind == "message.member"
        and event.payload.get("discussion_event_id") == discussion.event_id
    )
    if len(member_messages) >= MAX_DISCUSSION_MESSAGES:
        return DiscussionDecision(
            status="bounded",
            reason="max_messages",
            discussion_event_id=discussion.event_id,
            source_event_seq=discussion.seq,
            thread_id=thread_id,
        )

    terminals = {
        (int(event.payload["round_index"]), str(event.payload["member_id"])): event
        for event in validated
        if event.kind in _TERMINAL_EVENT_KINDS
        and event.payload.get("discussion_event_id") == discussion.event_id
    }
    watermarks = {
        (str(thread_id), str(member_id)): int(value)
        for (thread_id, member_id), value in (initial_watermarks or {}).items()
        if int(value) >= 0
    }
    for key, value in _derive_member_watermarks(validated).items():
        watermarks[key] = max(watermarks.get(key, 0), value)
    seen_through_seq = max(event.seq for event in thread_messages)

    for round_index in range(MAX_DISCUSSION_ROUNDS):
        # The user's message selects the first round, with no mention meaning
        # everyone. Later rounds are opt-in: only a peer explicitly cited by a
        # Bot and not heard from afterward gets another turn. Every member's
        # watermark remains intact, so a peer cited later still receives the
        # complete bounded transcript delta without consuming turns meanwhile.
        if round_index == 0:
            responders = initial_responders
            unresolved_handoff = False
        else:
            responders, unresolved_handoff = _unaddressed_member_mentions(
                discussion_messages,
                room,
            )
        if unresolved_handoff:
            return DiscussionDecision(
                status="settled",
                reason="unresolved_mention",
                discussion_event_id=discussion.event_id,
                source_event_seq=discussion.seq,
                thread_id=thread_id,
            )
        ordered = _rotate(responders, round_index)
        for member_index, member in enumerate(ordered):
            if (round_index, member.member_id) in terminals:
                continue
            watermark = watermarks.get((thread_id, member.member_id), 0)
            delta = [
                event
                for event in thread_messages
                if watermark < event.seq <= seen_through_seq
            ]
            if not delta:
                continue
            prompt = _build_prompt(
                room=room,
                member=member,
                messages=thread_messages,
                watermark=watermark,
                seen_through_seq=seen_through_seq,
            )
            task = _make_task_plan(
                room=room,
                discussion_event=discussion,
                member=member,
                member_index=member_index,
                round_index=round_index,
                seen_through_seq=seen_through_seq,
                prompt=prompt,
            )
            return DiscussionDecision(
                status="task",
                reason="member_turn",
                discussion_event_id=discussion.event_id,
                source_event_seq=discussion.seq,
                thread_id=thread_id,
                task=task,
            )

        spoke = any(
            int(event.payload["round_index"]) == round_index
            for event in member_messages
        )
        if not spoke:
            return DiscussionDecision(
                status="settled",
                reason="silent_round",
                discussion_event_id=discussion.event_id,
                source_event_seq=discussion.seq,
                thread_id=thread_id,
            )
        if round_index == MAX_DISCUSSION_ROUNDS - 1:
            return DiscussionDecision(
                status="bounded",
                reason="max_rounds",
                discussion_event_id=discussion.event_id,
                source_event_seq=discussion.seq,
                thread_id=thread_id,
            )

    raise AssertionError("bounded Discussion loop exhausted unexpectedly")


def reconstruct_task_plan(
    room_value: Any,
    events: Sequence[Mapping[str, Any]],
    task: Mapping[str, Any],
    *,
    local_profiles: Iterable[str],
) -> DiscussionTaskPlan:
    """Reconstruct and verify one persisted driver task after a restart."""

    room = validate_room(room_value, local_profiles=local_profiles)
    validated = _validated_events(events, room=room)
    identity = task.get("identity")
    payload = task.get("payload")
    if not isinstance(identity, driver.TaskIdentity) or not isinstance(
        payload, Mapping
    ):
        raise DiscussionReconstructionError(
            "driver task has no valid identity or payload"
        )
    required_payload = frozenset({
        "target_profile",
        "prompt",
        "source_event_seq",
    })
    if not required_payload <= frozenset(payload) or (
        frozenset(payload) - required_payload - {"target_member_id"}
    ):
        raise DiscussionReconstructionError("driver task payload shape changed")
    match = _TURN_ID_RE.fullmatch(identity.turn_id)
    if match is None:
        raise DiscussionReconstructionError("turn_id is not a Discussion coordinate")
    source_event_seq = int(match.group("source"))
    round_index = int(match.group("round"))
    member_index = int(match.group("position"))
    seen_through_seq = int(match.group("seen"))
    if payload.get("source_event_seq") != source_event_seq:
        raise DiscussionReconstructionError("task source event does not match turn_id")
    discussion = next(
        (
            event
            for event in validated
            if event.seq == source_event_seq and event.kind == "message.user"
        ),
        None,
    )
    if discussion is None:
        raise DiscussionReconstructionError("task source user event is missing")
    if (
        identity.room_id != room.room_id
        or identity.thread_id != discussion.payload["thread_id"]
    ):
        raise DiscussionReconstructionError(
            "task identity does not match its room thread"
        )
    profile = payload.get("target_profile")
    target_member_id = payload.get("target_member_id")
    member = next(
        (
            candidate
            for candidate in room.members
            if (
                candidate.member_id == target_member_id
                if target_member_id is not None
                else candidate.profile == profile
            )
        ),
        None,
    )
    if member is not None and member.profile != profile:
        member = None
    if member is None or _member_digest(member) != match.group("member"):
        raise DiscussionReconstructionError("task target member does not match turn_id")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise DiscussionReconstructionError("task prompt is missing")
    if len(prompt.encode("utf-8")) > driver.MAX_PROMPT_BYTES:
        raise DiscussionReconstructionError("task prompt exceeds the driver limit")
    reconstructed = _make_task_plan(
        room=room,
        discussion_event=discussion,
        member=member,
        member_index=member_index,
        round_index=round_index,
        seen_through_seq=seen_through_seq,
        prompt=prompt,
    )
    if reconstructed.identity != identity or dict(reconstructed.payload) != dict(
        payload
    ):
        raise DiscussionReconstructionError(
            "driver task failed deterministic reconstruction"
        )
    return reconstructed


def _terminal_text(result: Any, *, field: str, fallback: str) -> str:
    if isinstance(result, Mapping):
        value = result.get(field)
        if value is None and field == "error":
            value = result.get("text")
    else:
        value = result
    text = str(value or "").strip()
    return text or fallback


def plan_publication(
    room_value: Any,
    events: Sequence[Mapping[str, Any]],
    task: DiscussionTaskPlan,
    *,
    status: TerminalKind,
    result: Any = None,
    execution_generation: int | None = None,
    local_profiles: Iterable[str],
) -> PublicationPlan:
    """Plan idempotent room effects for one terminal driver task.

    A newer user event in the same thread supersedes a late result.  The task
    remains terminal in driver state, but only a deterministic cancellation is
    published, preventing stale prose and its watermark from hiding the newer
    user message.
    """

    room = validate_room(room_value, local_profiles=local_profiles)
    validated = _validated_events(events, room=room)
    if task.identity.room_id != room.room_id:
        raise DiscussionValidationError("task belongs to a different room")
    if task.member not in room.members:
        raise DiscussionValidationError("task member is not in the frozen roster")
    if status not in {"settled", "failed", "cancelled", "deferred"}:
        raise DiscussionValidationError("invalid terminal publication status")
    if status == "deferred" and (
        isinstance(execution_generation, bool)
        or not isinstance(execution_generation, int)
        or execution_generation < 1
    ):
        raise DiscussionValidationError(
            "deferred publication requires an execution generation"
        )

    newer_same_thread = any(
        event.kind == "message.user"
        and event.seq > task.seen_through_seq
        and event.payload.get("thread_id") == task.identity.thread_id
        for event in validated
    )
    effective_status: TerminalKind = (
        "cancelled" if newer_same_thread and status != "deferred" else status
    )
    digest = task.identity.task_id.removeprefix("dtask:")
    message_event_id = f"dmessage:{digest}"
    terminal_event_id = (
        f"ddeferred:{digest}:g{execution_generation}"
        if effective_status == "deferred"
        else f"dterminal:{digest}"
    )
    common = {
        "discussion_event_id": task.discussion_event_id,
        "member_id": task.member.member_id,
        "member_index": task.member_index,
        "round_index": task.round_index,
        "seen_through_seq": task.seen_through_seq,
        "task_id": task.identity.task_id,
        "thread_id": task.identity.thread_id,
        "turn_id": task.identity.turn_id,
    }
    effects: list[EventPlan] = []

    if effective_status == "settled":
        text = _truncate_utf8_text(
            _terminal_text(result, field="text", fallback=""),
            max_bytes=MAX_MEMBER_TEXT_BYTES,
            suffix=_TRUNCATED_REPLY_NOTICE,
        )
        passed = is_pass_text(text)
        if not passed:
            member_actor = {
                "kind": "member",
                "id": task.member.member_id,
                "profile": task.member.profile,
            }
            if task.member.target and task.member.target.get("kind") == "peer":
                member_actor["connection_id"] = task.member.target["peer_id"]
            if task.member.display_name:
                member_actor["display_name"] = task.member.display_name
            effects.append(
                EventPlan(
                    event_id=message_event_id,
                    kind="message.member",
                    actor=member_actor,
                    payload={
                        "discussion_event_id": task.discussion_event_id,
                        "member_id": task.member.member_id,
                        "member_index": task.member_index,
                        "round_index": task.round_index,
                        "task_id": task.identity.task_id,
                        "text": text,
                        "thread_id": task.identity.thread_id,
                        "turn_id": task.identity.turn_id,
                    },
                    authority_gateway_id=room.gateway_id,
                    authority_epoch=room.authority_epoch,
                )
            )
        terminal_payload = {
            **common,
            "message_event_id": None if passed else message_event_id,
            "passed": passed,
        }
        terminal_kind = "turn.settled"
    elif effective_status == "failed":
        error_text = _terminal_text(
            result,
            field="error",
            fallback="member turn failed",
        )
        from tools.bot_failure_reasons import ALL_REASONS, classify_agent_error

        supplied_reason = (
            str(result.get("reason_code") or result.get("reason") or "").strip()
            if isinstance(result, Mapping)
            else ""
        )
        reason_code = (
            supplied_reason
            if supplied_reason in ALL_REASONS
            else classify_agent_error(error_text)
        )
        terminal_payload = {
            **common,
            "error": error_text,
            "reason_code": reason_code,
        }
        terminal_kind = "turn.failed"
    elif effective_status == "cancelled":
        terminal_payload = {
            **common,
            "reason": (
                "superseded_by_newer_user_event"
                if newer_same_thread
                else _terminal_text(
                    result,
                    field="reason",
                    fallback="member turn cancelled",
                )
            ),
        }
        terminal_kind = "turn.cancelled"
    else:
        terminal_payload = {
            **common,
            "execution_generation": execution_generation,
            "reason": _terminal_text(
                result,
                field="reason",
                fallback="member_unavailable",
            ),
        }
        terminal_kind = "turn.deferred"

    effects.append(
        EventPlan(
            event_id=terminal_event_id,
            kind=terminal_kind,
            actor={"kind": "gateway", "id": room.gateway_id},
            payload=terminal_payload,
            authority_gateway_id=room.gateway_id,
            authority_epoch=room.authority_epoch,
        )
    )
    return PublicationPlan(
        task_id=task.identity.task_id,
        terminal_kind=terminal_kind,
        events=tuple(effects),
    )
