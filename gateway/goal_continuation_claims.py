"""Crash-consistent recovery records for claimed goal continuations.

A claim record is a private, versioned FIFO owned by the gateway process.  It is
published atomically before an adapter task removes the synthetic continuation
from ordinary in-memory queue state.  The record remains authoritative until
its events are completed or an authoritative lifecycle transition retires it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

CLAIM_VERSION = 1
CLAIM_DIR_NAME = "goal_continuation_claims_v1"
MAX_CLAIM_BYTES = 1024 * 1024
MAX_CLAIMS = 512
MAX_EVENTS = 256
MAX_CONSUMED_EVENT_IDS = MAX_EVENTS * 4
MAX_TEXT_BYTES = 256 * 1024
MAX_COMPLETED_RESULT_BYTES = 256 * 1024

_CLAIM_ID_ATTR = "_hermes_goal_claim_id"
_EVENT_ID_ATTR = "_hermes_goal_claim_event_id"
_STORE_LOCK = threading.RLock()


def _serialized(function):
    """Serialize claim read-modify-write transitions within one gateway."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with _STORE_LOCK:
            return function(*args, **kwargs)

    return wrapper


_EVENT_KEYS = {
    "event_id",
    "text",
    "message_type",
    "user_id",
    "user_name",
    "source",
    "raw_message",
    "message_id",
    "platform_update_id",
    "media_urls",
    "media_types",
    "reply_to_message_id",
    "reply_to_text",
    "reply_to_author_id",
    "reply_to_author_name",
    "reply_to_is_own_message",
    "prompt_response",
    "auto_skill",
    "channel_prompt",
    "channel_context",
    "internal",
    "metadata",
    "timestamp",
    "allow_gateway_control",
    "goal_continuation",
}
_SOURCE_KEYS = {
    "platform",
    "chat_id",
    "chat_name",
    "chat_type",
    "user_id",
    "user_name",
    "thread_id",
    "chat_topic",
    "user_id_alt",
    "chat_id_alt",
    "scope_id",
    "guild_id",
    "parent_chat_id",
    "message_id",
    "profile",
    "auto_thread_created",
    "auto_thread_initial_name",
    "prospective_thread_id",
    "is_bot",
    "role_authorized",
}


class GoalContinuationClaimError(RuntimeError):
    """A durable claim could not be published, decoded, bound, or updated."""


@dataclass(frozen=True)
class RecoveredGoalContinuationClaim:
    claim_id: str
    session_key: str
    session_id: str
    profile: str | None
    synthetic_head_pending: bool
    session_rebind_allowed: bool
    consumed_event_ids: tuple[str, ...]
    completed_results: dict[str, str]
    completed_delivery_texts: dict[str, str]
    completed_turn_tokens: dict[str, str]
    events: tuple[MessageEvent, ...]
    path: Path


def claim_directory(home: Path | str | None = None) -> Path:
    """Return the private directory that contains version-one claim records."""
    if home is None:
        from hermes_constants import get_hermes_home

        root = get_hermes_home()
    else:
        root = Path(home)
    directory = Path(root) / CLAIM_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        os.chmod(directory, 0o700)
    return directory


def claim_path(session_key: str, home: Path | str | None = None) -> Path:
    """Return the non-reversible, traversal-safe path for one session claim."""
    _validate_identifier("session_key", session_key, 4096)
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return claim_directory(home) / f"claim-{digest}.json"


def _successor_path(
    session_key: str,
    ordinal: int,
    event_id: str,
    *,
    home: Path | str | None = None,
) -> Path:
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return claim_directory(home) / (
        f"successor-{digest}-{ordinal:06d}-{event_id}.json"
    )


def _successor_paths(
    session_key: str, *, home: Path | str | None = None
) -> list[Path]:
    digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return sorted(claim_directory(home).glob(f"successor-{digest}-*.json"))


def event_claim_identity(event: MessageEvent) -> tuple[str, str] | None:
    claim_id = getattr(event, _CLAIM_ID_ATTR, None)
    event_id = getattr(event, _EVENT_ID_ATTR, None)
    if isinstance(claim_id, str) and isinstance(event_id, str):
        return claim_id, event_id
    return None


def clear_event_claim_identity(event: MessageEvent) -> None:
    for name in (_CLAIM_ID_ATTR, _EVENT_ID_ATTR):
        try:
            delattr(event, name)
        except AttributeError:
            pass


@_serialized
def publish_claim(
    session_key: str,
    session_id: str,
    events: Iterable[MessageEvent],
    *,
    home: Path | str | None = None,
) -> RecoveredGoalContinuationClaim:
    """Atomically publish a new exclusive FIFO claim for one session."""
    _validate_identifier("session_key", session_key, 4096)
    _validate_identifier("session_id", session_id, 512)
    ordered = _deduplicate_events(events)
    if not ordered or len(ordered) > MAX_EVENTS:
        raise GoalContinuationClaimError("invalid durable continuation event count")
    if not _is_typed_continuation(ordered[0]):
        raise GoalContinuationClaimError("durable claim head is not a typed continuation")

    path = claim_path(session_key, home)
    if path.exists():
        raise GoalContinuationClaimError("durable continuation claim already exists")
    if len(list(path.parent.glob("claim-*.json"))) >= MAX_CLAIMS:
        raise GoalContinuationClaimError("durable continuation claim file limit exceeded")

    claim_id = uuid.uuid4().hex
    encoded_events = [
        _encode_event(event, claim_id=claim_id, event_id=uuid.uuid4().hex)
        for event in ordered
    ]
    profile = ordered[0].source.profile if ordered[0].source is not None else None
    if any(
        event.source is None or event.source.profile != profile
        for event in ordered
    ):
        raise GoalContinuationClaimError("durable continuation claim crosses profiles")
    payload = {
        "version": CLAIM_VERSION,
        "claim_id": claim_id,
        "session_key": session_key,
        "session_id": session_id,
        "profile": profile,
        "synthetic_head_pending": True,
        "session_rebind_allowed": False,
        "consumed_event_ids": [],
        "completed_results": {},
        "completed_delivery_texts": {},
        "completed_turn_tokens": {},
        "events": encoded_events,
    }
    _write_payload(path, payload, must_not_exist=True)
    decoded = _decode_payload(path, payload)
    for source_event, recovered_event in zip(ordered, decoded.events):
        setattr(source_event, _CLAIM_ID_ATTR, claim_id)
        setattr(source_event, _EVENT_ID_ATTR, getattr(recovered_event, _EVENT_ID_ATTR))
    return RecoveredGoalContinuationClaim(
        claim_id=claim_id,
        session_key=session_key,
        session_id=session_id,
        profile=profile,
        synthetic_head_pending=True,
        session_rebind_allowed=False,
        consumed_event_ids=(),
        completed_results={},
        completed_delivery_texts={},
        completed_turn_tokens={},
        events=tuple(ordered),
        path=path,
    )


@_serialized
def append_claim_event(
    session_key: str,
    claim_id: str,
    event: MessageEvent,
    *,
    home: Path | str | None = None,
) -> str:
    """Durably append one distinct successor before in-memory admission."""
    path = claim_path(session_key, home)
    payload = _read_payload(path)
    base_claim = _decode_payload(path, payload)
    claim, sidecars = _combine_claim_with_successors(base_claim, payload, home=home)
    if claim.claim_id != claim_id:
        raise GoalContinuationClaimError("durable continuation claim identity mismatch")
    if len(claim.events) >= MAX_EVENTS:
        raise GoalContinuationClaimError("durable continuation claim event limit exceeded")
    if event.source is None or event.source.profile != claim.profile:
        raise GoalContinuationClaimError("durable continuation successor profile mismatch")

    identity = event_claim_identity(event)
    if identity is not None:
        if identity[0] != claim_id:
            raise GoalContinuationClaimError("event belongs to another durable claim")
        if any(
            getattr(item, _EVENT_ID_ATTR, None) == identity[1]
            for item in claim.events
        ):
            return identity[1]
    event_id = uuid.uuid4().hex
    ordinal = sidecars[-1][0] + 1 if sidecars else 0
    sidecar_payload = {
        "version": CLAIM_VERSION,
        "claim_id": claim_id,
        "session_key": claim.session_key,
        "session_id": claim.session_id,
        "profile": claim.profile,
        "ordinal": ordinal,
        "event": _encode_event(event, claim_id=claim_id, event_id=event_id),
    }
    _write_payload(
        _successor_path(session_key, ordinal, event_id, home=home),
        sidecar_payload,
        must_not_exist=True,
    )
    setattr(event, _CLAIM_ID_ATTR, claim_id)
    setattr(event, _EVENT_ID_ATTR, event_id)
    return event_id


@_serialized
def stage_completed_result(
    session_key: str,
    claim_id: str,
    event_id: str,
    content: str,
    *,
    delivery_text: str | None = None,
    active_turn_token: str | None = None,
    home: Path | str | None = None,
) -> None:
    """Checkpoint completed output inside its input claim before ledger transfer."""
    if not isinstance(content, str) or len(
        content.encode("utf-8", "replace")
    ) > MAX_COMPLETED_RESULT_BYTES:
        raise GoalContinuationClaimError(
            "invalid durable continuation completed result"
        )
    if active_turn_token is not None and (
        not isinstance(active_turn_token, str)
        or not active_turn_token
        or len(active_turn_token) > 256
    ):
        raise GoalContinuationClaimError(
            "invalid durable continuation active-turn token"
        )
    if delivery_text is None:
        delivery_text = content
    if not isinstance(delivery_text, str) or len(
        delivery_text.encode("utf-8", "replace")
    ) > MAX_COMPLETED_RESULT_BYTES:
        raise GoalContinuationClaimError(
            "invalid durable continuation delivery text"
        )
    path = claim_path(session_key, home)
    payload = _read_payload(path)
    claim = _decode_payload(path, payload)
    if claim.claim_id != claim_id:
        raise GoalContinuationClaimError(
            "durable continuation claim identity mismatch"
        )
    head_id = getattr(claim.events[0], _EVENT_ID_ATTR, None)
    if head_id != event_id:
        raise GoalContinuationClaimError(
            "durable continuation result is out of order"
        )
    completed = dict(claim.completed_results)
    completed_delivery_texts = dict(claim.completed_delivery_texts)
    completed_turn_tokens = dict(claim.completed_turn_tokens)
    existing = completed.get(event_id)
    if existing is not None:
        if (
            existing != content
            or completed_delivery_texts.get(event_id) != delivery_text
            or completed_turn_tokens.get(event_id) != active_turn_token
        ):
            raise GoalContinuationClaimError(
                "conflicting durable continuation completed result"
            )
        return
    completed[event_id] = content
    payload["completed_results"] = completed
    completed_delivery_texts[event_id] = delivery_text
    payload["completed_delivery_texts"] = completed_delivery_texts
    if active_turn_token is not None:
        completed_turn_tokens[event_id] = active_turn_token
    payload["completed_turn_tokens"] = completed_turn_tokens
    _write_payload(path, payload)


@_serialized
def complete_claim_event(
    session_key: str,
    claim_id: str,
    event_id: str,
    *,
    home: Path | str | None = None,
) -> bool:
    """Acknowledge exactly the current durable FIFO head."""
    path = claim_path(session_key, home)
    if not path.exists():
        return False
    payload = _read_payload(path)
    base_claim = _decode_payload(path, payload)
    claim, sidecars = _combine_claim_with_successors(base_claim, payload, home=home)
    if claim.claim_id != claim_id:
        raise GoalContinuationClaimError("durable continuation claim identity mismatch")
    head_id = getattr(claim.events[0], _EVENT_ID_ATTR, None)
    if head_id != event_id:
        raise GoalContinuationClaimError("durable continuation completion is out of order")
    consumed_event_ids = list(payload["consumed_event_ids"])
    if event_id not in consumed_event_ids:
        if len(consumed_event_ids) >= MAX_CONSUMED_EVENT_IDS:
            raise GoalContinuationClaimError(
                "durable continuation consumed-event limit exceeded"
            )
        consumed_event_ids.append(event_id)
    payload["consumed_event_ids"] = consumed_event_ids
    completed_results = dict(payload.get("completed_results", {}))
    completed_results.pop(event_id, None)
    payload["completed_results"] = completed_results
    completed_delivery_texts = dict(payload.get("completed_delivery_texts", {}))
    completed_delivery_texts.pop(event_id, None)
    payload["completed_delivery_texts"] = completed_delivery_texts
    completed_turn_tokens = dict(payload.get("completed_turn_tokens", {}))
    completed_turn_tokens.pop(event_id, None)
    payload["completed_turn_tokens"] = completed_turn_tokens
    remaining = payload["events"][1:]
    if payload["synthetic_head_pending"]:
        payload["synthetic_head_pending"] = False
    if remaining:
        payload["events"] = remaining
        _write_payload(path, payload)
    elif sidecars:
        promoted = [
            sidecar["event"]
            for _ordinal, _sidecar_path, sidecar, sidecar_event in sidecars
            if getattr(sidecar_event, _EVENT_ID_ATTR)
            not in set(consumed_event_ids)
        ]
        if promoted:
            payload["events"] = promoted
            _write_payload(path, payload)
        else:
            _unlink_payload(path)
        _discard_successor_records(sidecars)
    else:
        _unlink_payload(path)
    return True


@_serialized
def retire_claim(
    session_key: str,
    claim_id: str,
    *,
    home: Path | str | None = None,
) -> bool:
    """Retire synthetic members while keeping genuine successors recoverable."""
    path = claim_path(session_key, home)
    if not path.exists():
        return False
    payload = _read_payload(path)
    base_claim = _decode_payload(path, payload)
    claim, sidecars = _combine_claim_with_successors(base_claim, payload, home=home)
    if claim.claim_id != claim_id:
        raise GoalContinuationClaimError("durable continuation claim identity mismatch")
    if claim.completed_results:
        raise GoalContinuationClaimError(
            "completed continuation result still owns publication"
        )
    encoded_by_id = {
        getattr(event, _EVENT_ID_ATTR): encoded
        for event, encoded in zip(base_claim.events, payload["events"])
    }
    for _ordinal, _sidecar_path, sidecar, event in sidecars:
        encoded_by_id.setdefault(getattr(event, _EVENT_ID_ATTR), sidecar["event"])
    remaining = [
        encoded_by_id[getattr(event, _EVENT_ID_ATTR)]
        for event in claim.events
        if not event.goal_continuation
    ]
    consumed_event_ids = list(payload["consumed_event_ids"])
    for event in claim.events:
        event_id = getattr(event, _EVENT_ID_ATTR)
        if event.goal_continuation and event_id not in consumed_event_ids:
            if len(consumed_event_ids) >= MAX_CONSUMED_EVENT_IDS:
                raise GoalContinuationClaimError(
                    "durable continuation consumed-event limit exceeded"
                )
            consumed_event_ids.append(event_id)
    payload["consumed_event_ids"] = consumed_event_ids
    if remaining:
        payload["synthetic_head_pending"] = False
        payload["session_rebind_allowed"] = True
        payload["events"] = remaining
        _write_payload(path, payload)
        _discard_successor_records(sidecars)
    else:
        _discard_successor_records(sidecars)
        _unlink_payload(path)
    return True


@_serialized
def load_claims(
    *, home: Path | str | None = None
) -> list[RecoveredGoalContinuationClaim]:
    """Strictly decode every claim, preserving every invalid file on failure."""
    directory = claim_directory(home)
    paths = sorted(directory.glob("claim-*.json"))
    successor_paths = set(directory.glob("successor-*.json"))
    if len(paths) > MAX_CLAIMS or len(successor_paths) > MAX_CLAIMS * MAX_EVENTS:
        raise GoalContinuationClaimError("durable continuation claim file limit exceeded")
    claims: list[RecoveredGoalContinuationClaim] = []
    consumed_successors: set[Path] = set()
    seen_sessions: set[str] = set()
    for path in paths:
        payload = _read_payload(path)
        base_claim = _decode_payload(path, payload)
        claim, sidecars = _combine_claim_with_successors(
            base_claim, payload, home=home
        )
        expected = claim_path(claim.session_key, home)
        if expected.name != path.name:
            raise GoalContinuationClaimError("durable continuation claim filename mismatch")
        if claim.session_key in seen_sessions:
            raise GoalContinuationClaimError("duplicate durable continuation session claim")
        seen_sessions.add(claim.session_key)
        consumed_successors.update(item[1] for item in sidecars)
        claims.append(claim)
    if successor_paths != consumed_successors:
        raise GoalContinuationClaimError("orphan durable continuation successor record")
    return claims


def _validate_identifier(name: str, value: Any, limit: int) -> None:
    if not isinstance(value, str) or not value or len(value) > limit or "\x00" in value:
        raise GoalContinuationClaimError(f"invalid durable continuation {name}")


def _deduplicate_events(events: Iterable[MessageEvent]) -> list[MessageEvent]:
    ordered: list[MessageEvent] = []
    seen: set[int] = set()
    for event in events:
        if not isinstance(event, MessageEvent):
            raise GoalContinuationClaimError("invalid durable continuation event")
        identity = id(event)
        if identity not in seen:
            seen.add(identity)
            ordered.append(event)
    return ordered


def _is_typed_continuation(event: MessageEvent) -> bool:
    if not (
        event.goal_continuation
        and not event.internal
        and not event.allow_gateway_control
    ):
        return False
    text = event.text or ""
    return text.startswith("[Continuing toward your standing goal]\nGoal:") or text.startswith(
        "[Continuing toward your standing goal — a quality gate failed]\nGoal:"
    )


def _encode_event(event: MessageEvent, *, claim_id: str, event_id: str) -> dict[str, Any]:
    if event.source is None:
        raise GoalContinuationClaimError("durable continuation event has no source")
    if len((event.text or "").encode("utf-8")) > MAX_TEXT_BYTES:
        raise GoalContinuationClaimError("durable continuation event text is too large")
    source = event.source.to_dict()
    source["is_bot"] = bool(event.source.is_bot)
    # Role membership is transient authorization evidence.  A restart must
    # re-evaluate current policy instead of replaying an earlier role grant.
    source["role_authorized"] = False
    payload = {
        "event_id": event_id,
        "text": event.text,
        "message_type": event.message_type.value,
        "user_id": event.user_id,
        "user_name": event.user_name,
        "source": source,
        "raw_message": event.raw_message,
        "message_id": event.message_id,
        "platform_update_id": event.platform_update_id,
        "media_urls": list(event.media_urls),
        "media_types": list(event.media_types),
        "reply_to_message_id": event.reply_to_message_id,
        "reply_to_text": event.reply_to_text,
        "reply_to_author_id": event.reply_to_author_id,
        "reply_to_author_name": event.reply_to_author_name,
        "reply_to_is_own_message": bool(event.reply_to_is_own_message),
        "prompt_response": event.prompt_response,
        "auto_skill": event.auto_skill,
        "channel_prompt": event.channel_prompt,
        "channel_context": event.channel_context,
        "internal": bool(event.internal),
        "metadata": event.metadata,
        "timestamp": event.timestamp.isoformat(),
        "allow_gateway_control": bool(event.allow_gateway_control),
        "goal_continuation": bool(event.goal_continuation),
    }
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GoalContinuationClaimError(
            "durable continuation event is not JSON serializable"
        ) from exc
    return payload


def _decode_payload(path: Path, payload: Any) -> RecoveredGoalContinuationClaim:
    required_keys = {
        "version",
        "claim_id",
        "session_key",
        "session_id",
        "profile",
        "synthetic_head_pending",
        "session_rebind_allowed",
        "consumed_event_ids",
        "events",
    }
    allowed_keys = required_keys | {
        "completed_results",
        "completed_delivery_texts",
        "completed_turn_tokens",
    }
    if (
        not isinstance(payload, dict)
        or not required_keys.issubset(payload)
        or not set(payload).issubset(allowed_keys)
    ):
        raise GoalContinuationClaimError("invalid durable continuation claim schema")
    if payload["version"] != CLAIM_VERSION:
        raise GoalContinuationClaimError("unsupported durable continuation claim version")
    _validate_identifier("claim_id", payload["claim_id"], 128)
    _validate_identifier("session_key", payload["session_key"], 4096)
    _validate_identifier("session_id", payload["session_id"], 512)
    profile = payload["profile"]
    if profile is not None and (not isinstance(profile, str) or not profile or len(profile) > 128):
        raise GoalContinuationClaimError("invalid durable continuation claim profile")
    synthetic_head_pending = payload["synthetic_head_pending"]
    if not isinstance(synthetic_head_pending, bool):
        raise GoalContinuationClaimError(
            "invalid durable continuation semantic head state"
        )
    session_rebind_allowed = payload["session_rebind_allowed"]
    if not isinstance(session_rebind_allowed, bool):
        raise GoalContinuationClaimError(
            "invalid durable continuation session rebind state"
        )
    consumed_event_ids = payload["consumed_event_ids"]
    if (
        not isinstance(consumed_event_ids, list)
        or len(consumed_event_ids) > MAX_CONSUMED_EVENT_IDS
        or any(
            not isinstance(event_id, str)
            or not event_id
            or len(event_id) > 128
            for event_id in consumed_event_ids
        )
        or len(set(consumed_event_ids)) != len(consumed_event_ids)
    ):
        raise GoalContinuationClaimError(
            "invalid durable continuation consumed-event ledger"
        )
    encoded_events = payload["events"]
    if not isinstance(encoded_events, list) or not 1 <= len(encoded_events) <= MAX_EVENTS:
        raise GoalContinuationClaimError("invalid durable continuation event count")
    events = tuple(
        _decode_event(item, claim_id=payload["claim_id"])
        for item in encoded_events
    )
    event_ids = [getattr(event, _EVENT_ID_ATTR) for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise GoalContinuationClaimError("duplicate durable continuation event identity")
    if set(event_ids).intersection(consumed_event_ids):
        raise GoalContinuationClaimError(
            "canonical durable continuation event is already consumed"
        )
    completed_results = payload.get("completed_results", {})
    if (
        not isinstance(completed_results, dict)
        or len(completed_results) > 1
        or any(
            not isinstance(event_id, str)
            or event_id not in event_ids
            or not isinstance(content, str)
            or len(content.encode("utf-8", "replace"))
            > MAX_COMPLETED_RESULT_BYTES
            for event_id, content in completed_results.items()
        )
        or (
            completed_results
            and set(completed_results) != {event_ids[0]}
        )
    ):
        raise GoalContinuationClaimError(
            "invalid durable continuation completed-result state"
        )
    completed_turn_tokens = payload.get("completed_turn_tokens", {})
    completed_delivery_texts = payload.get("completed_delivery_texts", {})
    if (
        not isinstance(completed_delivery_texts, dict)
        or set(completed_delivery_texts) != set(completed_results)
        or any(
            not isinstance(text, str)
            or len(text.encode("utf-8", "replace"))
            > MAX_COMPLETED_RESULT_BYTES
            for text in completed_delivery_texts.values()
        )
    ):
        raise GoalContinuationClaimError(
            "invalid durable continuation delivery-text state"
        )
    if (
        not isinstance(completed_turn_tokens, dict)
        or set(completed_turn_tokens) - set(completed_results)
        or any(
            not isinstance(event_id, str)
            or event_id not in event_ids
            or not isinstance(token, str)
            or not token
            or len(token) > 256
            for event_id, token in completed_turn_tokens.items()
        )
    ):
        raise GoalContinuationClaimError(
            "invalid durable continuation completed-turn state"
        )
    if synthetic_head_pending and not _is_typed_continuation(events[0]):
        raise GoalContinuationClaimError("invalid durable continuation semantic head")
    if session_rebind_allowed and any(event.goal_continuation for event in events):
        raise GoalContinuationClaimError(
            "session-rebindable durable continuation queue contains synthetic event"
        )
    if any(event.source is None or event.source.profile != profile for event in events):
        raise GoalContinuationClaimError("durable continuation claim profile mismatch")
    return RecoveredGoalContinuationClaim(
        claim_id=payload["claim_id"],
        session_key=payload["session_key"],
        session_id=payload["session_id"],
        profile=profile,
        synthetic_head_pending=synthetic_head_pending,
        session_rebind_allowed=session_rebind_allowed,
        consumed_event_ids=tuple(consumed_event_ids),
        completed_results=dict(completed_results),
        completed_delivery_texts=dict(completed_delivery_texts),
        completed_turn_tokens=dict(completed_turn_tokens),
        events=events,
        path=path,
    )


def _load_successor_records(
    claim: RecoveredGoalContinuationClaim,
    *,
    home: Path | str | None = None,
) -> list[tuple[int, Path, dict[str, Any], MessageEvent]]:
    records: list[tuple[int, Path, dict[str, Any], MessageEvent]] = []
    seen_ordinals: set[int] = set()
    for path in _successor_paths(claim.session_key, home=home):
        payload = _read_payload(path)
        if not isinstance(payload, dict) or set(payload) != {
            "version",
            "claim_id",
            "session_key",
            "session_id",
            "profile",
            "ordinal",
            "event",
        }:
            raise GoalContinuationClaimError(
                "invalid durable continuation successor schema"
            )
        ordinal = payload["ordinal"]
        if (
            payload["version"] != CLAIM_VERSION
            or payload["claim_id"] != claim.claim_id
            or payload["session_key"] != claim.session_key
            or payload["session_id"] != claim.session_id
            or payload["profile"] != claim.profile
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or ordinal < 0
            or ordinal >= MAX_EVENTS
            or ordinal in seen_ordinals
        ):
            raise GoalContinuationClaimError(
                "invalid durable continuation successor binding"
            )
        event = _decode_event(payload["event"], claim_id=claim.claim_id)
        event_id = getattr(event, _EVENT_ID_ATTR)
        if path != _successor_path(
            claim.session_key, ordinal, event_id, home=home
        ):
            raise GoalContinuationClaimError(
                "durable continuation successor filename mismatch"
            )
        if event.source is None or event.source.profile != claim.profile:
            raise GoalContinuationClaimError(
                "durable continuation successor profile mismatch"
            )
        seen_ordinals.add(ordinal)
        records.append((ordinal, path, payload, event))
    records.sort(key=lambda item: item[0])
    return records


def _combine_claim_with_successors(
    claim: RecoveredGoalContinuationClaim,
    payload: dict[str, Any],
    *,
    home: Path | str | None = None,
) -> tuple[
    RecoveredGoalContinuationClaim,
    list[tuple[int, Path, dict[str, Any], MessageEvent]],
]:
    records = _load_successor_records(claim, home=home)
    events = list(claim.events)
    encoded_by_id = {
        getattr(event, _EVENT_ID_ATTR): encoded
        for event, encoded in zip(claim.events, payload["events"])
    }
    consumed_event_ids = set(claim.consumed_event_ids)
    for _ordinal, _path, sidecar, event in records:
        event_id = getattr(event, _EVENT_ID_ATTR)
        if event_id in consumed_event_ids:
            continue
        existing = encoded_by_id.get(event_id)
        if existing is not None:
            if existing != sidecar["event"]:
                raise GoalContinuationClaimError(
                    "conflicting durable continuation successor identity"
                )
            continue
        encoded_by_id[event_id] = sidecar["event"]
        events.append(event)
    if len(events) > MAX_EVENTS:
        raise GoalContinuationClaimError(
            "durable continuation claim event limit exceeded"
        )
    if claim.session_rebind_allowed and any(
        event.goal_continuation for event in events
    ):
        raise GoalContinuationClaimError(
            "session-rebindable durable continuation queue contains synthetic event"
        )
    return (
        RecoveredGoalContinuationClaim(
            claim_id=claim.claim_id,
            session_key=claim.session_key,
            session_id=claim.session_id,
            profile=claim.profile,
            synthetic_head_pending=claim.synthetic_head_pending,
            session_rebind_allowed=claim.session_rebind_allowed,
            consumed_event_ids=claim.consumed_event_ids,
            completed_results=dict(claim.completed_results),
            completed_delivery_texts=dict(claim.completed_delivery_texts),
            completed_turn_tokens=dict(claim.completed_turn_tokens),
            events=tuple(events),
            path=claim.path,
        ),
        records,
    )


def _discard_successor_records(
    records: list[tuple[int, Path, dict[str, Any], MessageEvent]],
) -> None:
    # The canonical record is already complete before sidecars are removed.
    # A leftover duplicate is harmless and is deduplicated by event identity
    # on restart, so cleanup failure must not corrupt the durable transition.
    for _ordinal, path, _payload, _event in records:
        try:
            path.unlink()
        except OSError:
            pass


def _decode_event(payload: Any, *, claim_id: str) -> MessageEvent:
    if not isinstance(payload, dict) or set(payload) != _EVENT_KEYS:
        raise GoalContinuationClaimError("invalid durable continuation event schema")
    _validate_identifier("event_id", payload["event_id"], 128)
    if not isinstance(payload["text"], str):
        raise GoalContinuationClaimError("invalid durable continuation event text")
    if len(payload["text"].encode("utf-8")) > MAX_TEXT_BYTES:
        raise GoalContinuationClaimError("durable continuation event text is too large")
    try:
        message_type = MessageType(payload["message_type"])
    except (TypeError, ValueError) as exc:
        raise GoalContinuationClaimError(
            "invalid durable continuation message type"
        ) from exc
    for name in (
        "user_id",
        "user_name",
        "message_id",
        "reply_to_message_id",
        "reply_to_text",
        "reply_to_author_id",
        "reply_to_author_name",
        "channel_prompt",
        "channel_context",
    ):
        value = payload[name]
        if value is not None and not isinstance(value, str):
            raise GoalContinuationClaimError(
                "invalid durable continuation optional text field"
            )
    platform_update_id = payload["platform_update_id"]
    if platform_update_id is not None and (
        not isinstance(platform_update_id, int)
        or isinstance(platform_update_id, bool)
    ):
        raise GoalContinuationClaimError(
            "invalid durable continuation platform update identity"
        )
    prompt_response = payload["prompt_response"]
    if prompt_response is not None and not isinstance(prompt_response, dict):
        raise GoalContinuationClaimError(
            "invalid durable continuation prompt response"
        )
    auto_skill = payload["auto_skill"]
    if auto_skill is not None and not (
        isinstance(auto_skill, str)
        or (
            isinstance(auto_skill, list)
            and all(isinstance(item, str) for item in auto_skill)
        )
    ):
        raise GoalContinuationClaimError("invalid durable continuation auto skill")
    source_payload = payload["source"]
    if not isinstance(source_payload, dict) or not set(source_payload).issubset(_SOURCE_KEYS):
        raise GoalContinuationClaimError("invalid durable continuation source schema")
    if "platform" not in source_payload or "chat_id" not in source_payload:
        raise GoalContinuationClaimError("incomplete durable continuation source")
    source_data = dict(source_payload)
    is_bot = source_data.pop("is_bot", False)
    role_authorized = source_data.pop("role_authorized", False)
    if not isinstance(is_bot, bool) or not isinstance(role_authorized, bool):
        raise GoalContinuationClaimError("invalid durable continuation source trust fields")
    if role_authorized:
        raise GoalContinuationClaimError(
            "durable continuation source contains stale role authorization"
        )
    try:
        source = SessionSource.from_dict(source_data)
    except (KeyError, TypeError, ValueError) as exc:
        raise GoalContinuationClaimError("invalid durable continuation source") from exc
    source.is_bot = is_bot
    source.role_authorized = False
    if not isinstance(source.chat_id, str):
        raise GoalContinuationClaimError("invalid durable continuation source chat id")
    for name in (
        "chat_name",
        "chat_type",
        "user_id",
        "user_name",
        "thread_id",
        "chat_topic",
        "user_id_alt",
        "chat_id_alt",
        "scope_id",
        "guild_id",
        "parent_chat_id",
        "message_id",
        "profile",
        "auto_thread_initial_name",
        "prospective_thread_id",
    ):
        value = getattr(source, name)
        if value is not None and not isinstance(value, str):
            raise GoalContinuationClaimError(
                "invalid durable continuation source text field"
            )
    if not isinstance(source.auto_thread_created, bool):
        raise GoalContinuationClaimError(
            "invalid durable continuation source boolean field"
        )
    try:
        timestamp = datetime.fromisoformat(payload["timestamp"])
    except (TypeError, ValueError) as exc:
        raise GoalContinuationClaimError("invalid durable continuation timestamp") from exc
    for name in ("media_urls", "media_types"):
        value = payload[name]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise GoalContinuationClaimError("invalid durable continuation media list")
    for name in (
        "internal", "allow_gateway_control", "goal_continuation",
        "reply_to_is_own_message",
    ):
        if not isinstance(payload[name], bool):
            raise GoalContinuationClaimError("invalid durable continuation boolean field")
    if not isinstance(payload["metadata"], dict):
        raise GoalContinuationClaimError("invalid durable continuation metadata")
    event = MessageEvent(
        text=payload["text"],
        message_type=message_type,
        user_id=payload["user_id"],
        user_name=payload["user_name"],
        source=source,
        raw_message=payload["raw_message"],
        message_id=payload["message_id"],
        platform_update_id=payload["platform_update_id"],
        media_urls=list(payload["media_urls"]),
        media_types=list(payload["media_types"]),
        reply_to_message_id=payload["reply_to_message_id"],
        reply_to_text=payload["reply_to_text"],
        reply_to_author_id=payload["reply_to_author_id"],
        reply_to_author_name=payload["reply_to_author_name"],
        reply_to_is_own_message=payload["reply_to_is_own_message"],
        prompt_response=payload["prompt_response"],
        auto_skill=payload["auto_skill"],
        channel_prompt=payload["channel_prompt"],
        channel_context=payload["channel_context"],
        internal=payload["internal"],
        metadata=dict(payload["metadata"]),
        timestamp=timestamp,
        allow_gateway_control=payload["allow_gateway_control"],
        goal_continuation=payload["goal_continuation"],
    )
    if event.goal_continuation and not _is_typed_continuation(event):
        raise GoalContinuationClaimError("invalid durable continuation provenance")
    setattr(event, _CLAIM_ID_ATTR, claim_id)
    setattr(event, _EVENT_ID_ATTR, payload["event_id"])
    return event


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_CLAIM_BYTES:
            raise GoalContinuationClaimError("invalid durable continuation claim size")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except GoalContinuationClaimError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GoalContinuationClaimError(
            "durable continuation claim is unreadable"
        ) from exc
    return payload


def _write_payload(path: Path, payload: dict[str, Any], *, must_not_exist: bool = False) -> None:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GoalContinuationClaimError(
            "durable continuation claim is not JSON serializable"
        ) from exc
    if len(encoded) > MAX_CLAIM_BYTES:
        raise GoalContinuationClaimError("durable continuation claim is too large")
    if must_not_exist and path.exists():
        raise GoalContinuationClaimError("durable continuation claim already exists")
    installed = False
    try:
        descriptor, temp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            # Claim records must never use the shared busy-target/cross-device
            # copy fallbacks: an in-place rewrite is not crash atomic. Windows
            # readers can transiently deny replace, so retry only the atomic
            # operation and fail closed if the bounded budget is exhausted.
            for attempt in range(6):
                try:
                    os.replace(temp_name, path)
                    installed = True
                    break
                except OSError as exc:
                    if (
                        os.name != "nt"
                        or getattr(exc, "winerror", None) not in {5, 32, 33}
                        or attempt == 5
                    ):
                        raise
                    time.sleep(0.01 * (2**attempt))
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        _fsync_directory(path.parent)
    except GoalContinuationClaimError:
        raise
    except Exception as exc:
        if installed:
            try:
                if path.read_bytes() == encoded:
                    # The atomic namespace transition completed.  A parent
                    # directory fsync error makes power-loss persistence
                    # uncertain, but reporting publication failure would split
                    # caller ownership from the exact visible durable record.
                    # Adopt that record so same-process retry and process-crash
                    # recovery share one event identity.
                    return
            except OSError:
                pass
        raise GoalContinuationClaimError(
            "durable continuation claim publication failed"
        ) from exc


def _unlink_payload(path: Path) -> None:
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError as exc:
        raise GoalContinuationClaimError(
            "durable continuation claim retirement failed"
        ) from exc


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
