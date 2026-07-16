"""Strict mechanical protocol for the local privileged Discord connector.

The protocol carries normalized public-guild message events, bounded public
history reads, and fixed public message sends across a local Unix socket.  It
deliberately has no raw Discord URL/method surface and never interprets message
text.  Target visibility, allowlists, deadlines, frame bounds, idempotency, and
receipts are mechanical security/execution concerns owned by the connector.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

PROTOCOL_VERSION = "discord-connector.v2"
RECEIPT_VERSION = "discord-connector-receipt.v2"
MAX_FRAME_BYTES = 64 * 1024
MAX_CONTENT_CHARS = 2_000
MAX_CONTENT_BYTES = 8_000
MAX_NAME_CHARS = 160
MAX_HISTORY_MESSAGES = 25
MAX_HISTORY_CONTENT_BYTES = 2_000
MAX_IDEMPOTENCY_BYTES = 256
MAX_WAIT_MS = 5_000
MAX_DEADLINE_SECONDS = 30

_SNOWFLAKE_RE = re.compile(r"^[1-9][0-9]{0,24}$")
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CRON_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


class DiscordConnectorKind(StrEnum):
    HELLO = "hello"
    EVENT_NEXT = "event.next"
    EVENT_ACK = "event.ack"
    EVENT_ACK_READBACK = "event.ack.readback"
    TARGET_GET = "public.target.get"
    HISTORY_FETCH = "public.history.fetch"
    MESSAGE_SEND = "public.message.send"


class DiscordConnectorTargetType(StrEnum):
    PUBLIC_GUILD_CHANNEL = "public_guild_channel"
    PUBLIC_GUILD_THREAD = "public_guild_thread"
    GUILD_CHANNEL = "guild_channel"
    GUILD_THREAD = "guild_thread"


class DiscordConnectorHistoryAuthorityKind(StrEnum):
    AUTHENTICATED_USER = "authenticated_discord_user"
    REVIEWED_CRON = "reviewed_production_cron"


DISCORD_CONNECTOR_THREAD_TARGET_TYPES = frozenset({
    DiscordConnectorTargetType.PUBLIC_GUILD_THREAD,
    DiscordConnectorTargetType.GUILD_THREAD,
})


class DiscordConnectorProtocolError(ValueError):
    """Stable validation failure which never reflects message or credentials."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiscordConnectorProtocolError("non_canonical_json") from exc


def sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _non_json_constant(_value: str) -> None:
    raise ValueError("non JSON number")


def decode_frame(body: bytes) -> dict[str, Any]:
    if not isinstance(body, bytes) or not body or len(body) > MAX_FRAME_BYTES:
        raise DiscordConnectorProtocolError("invalid_frame_size")
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_duplicate_keys,
            parse_constant=_non_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DiscordConnectorProtocolError("invalid_frame_json") from exc
    if not isinstance(value, dict):
        raise DiscordConnectorProtocolError("invalid_frame_shape")
    return value


def _exact(
    value: Any,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise DiscordConnectorProtocolError(code)
    result = dict(value)
    if set(result) - required - optional or required - set(result):
        raise DiscordConnectorProtocolError(code)
    return result


def _uuid(value: Any, code: str) -> str:
    if not isinstance(value, str):
        raise DiscordConnectorProtocolError(code)
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise DiscordConnectorProtocolError(code) from exc
    if str(parsed) != value:
        raise DiscordConnectorProtocolError(code)
    return value


def _snowflake(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SNOWFLAKE_RE.fullmatch(value) is None:
        raise DiscordConnectorProtocolError(code)
    return value


def _optional_snowflake(value: Any, code: str) -> str | None:
    return None if value is None else _snowflake(value, code)


def _bounded_text(
    value: Any,
    *,
    code: str,
    allow_empty: bool = False,
    max_chars: int = MAX_CONTENT_CHARS,
    max_bytes: int = MAX_CONTENT_BYTES,
) -> str:
    if not isinstance(value, str) or (not value and not allow_empty) or "\x00" in value:
        raise DiscordConnectorProtocolError(code)
    if len(value) > max_chars or len(value.encode("utf-8")) > max_bytes:
        raise DiscordConnectorProtocolError(code)
    return value


def _bounded_history_text(value: Any, *, code: str, allow_empty: bool) -> str:
    text = _bounded_text(
        value,
        code=code,
        allow_empty=allow_empty,
        max_chars=MAX_CONTENT_CHARS,
        max_bytes=MAX_HISTORY_CONTENT_BYTES,
    )
    # Keep the response frame mathematically bounded: JSON can expand control
    # characters up to sixfold even when the raw UTF-8 byte count is small.
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        raise DiscordConnectorProtocolError(code)
    return text


def _bounded_history_author_name(value: Any, *, code: str) -> str:
    text = _bounded_text(
        value,
        code=code,
        max_chars=MAX_NAME_CHARS,
        max_bytes=MAX_NAME_CHARS * 4,
    )
    if any(ord(char) < 32 for char in text):
        raise DiscordConnectorProtocolError(code)
    return text


def _idempotency(value: Any) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_RE.fullmatch(value) is None:
        raise DiscordConnectorProtocolError("invalid_idempotency_key")
    if not 1 <= len(value.encode("utf-8")) <= MAX_IDEMPOTENCY_BYTES:
        raise DiscordConnectorProtocolError("invalid_idempotency_key")
    return value


def _integer(value: Any, *, minimum: int, maximum: int, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiscordConnectorProtocolError(code)
    if not minimum <= value <= maximum:
        raise DiscordConnectorProtocolError(code)
    return value


@dataclass(frozen=True)
class DiscordConnectorTarget:
    target_type: DiscordConnectorTargetType
    guild_id: str
    channel_id: str
    parent_channel_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordConnectorTarget":
        raw = _exact(
            value,
            required=frozenset({"target_type", "guild_id", "channel_id"}),
            optional=frozenset({"parent_channel_id"}),
            code="invalid_public_target",
        )
        try:
            target_type = DiscordConnectorTargetType(raw["target_type"])
        except (TypeError, ValueError) as exc:
            raise DiscordConnectorProtocolError("forbidden_or_invalid_target") from exc
        guild_id = _snowflake(raw["guild_id"], "invalid_public_target")
        channel_id = _snowflake(raw["channel_id"], "invalid_public_target")
        parent = _optional_snowflake(
            raw.get("parent_channel_id"), "invalid_public_target"
        )
        if target_type in DISCORD_CONNECTOR_THREAD_TARGET_TYPES:
            if parent is None or parent == channel_id:
                raise DiscordConnectorProtocolError("invalid_public_target")
        elif parent is not None:
            raise DiscordConnectorProtocolError("invalid_public_target")
        return cls(target_type, guild_id, channel_id, parent)

    def to_mapping(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "target_type": self.target_type.value,
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
        }
        if self.parent_channel_id is not None:
            value["parent_channel_id"] = self.parent_channel_id
        return value


@dataclass(frozen=True)
class DiscordConnectorHistoryAuthority:
    """Non-secret authority attached internally to one history request."""

    kind: DiscordConnectorHistoryAuthorityKind
    requester_user_id: str | None = None
    cron_job_id: str | None = None

    @classmethod
    def authenticated_user(
        cls, requester_user_id: Any
    ) -> "DiscordConnectorHistoryAuthority":
        return cls.from_mapping(
            {
                "kind": DiscordConnectorHistoryAuthorityKind.AUTHENTICATED_USER.value,
                "requester_user_id": requester_user_id,
            }
        )

    @classmethod
    def reviewed_cron(cls, job_id: Any) -> "DiscordConnectorHistoryAuthority":
        return cls.from_mapping(
            {
                "kind": DiscordConnectorHistoryAuthorityKind.REVIEWED_CRON.value,
                "cron_job_id": job_id,
            }
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordConnectorHistoryAuthority":
        if not isinstance(value, Mapping):
            raise DiscordConnectorProtocolError("invalid_history_authority")
        try:
            kind = DiscordConnectorHistoryAuthorityKind(value.get("kind"))
        except (TypeError, ValueError) as exc:
            raise DiscordConnectorProtocolError("invalid_history_authority") from exc
        if kind is DiscordConnectorHistoryAuthorityKind.AUTHENTICATED_USER:
            raw = _exact(
                value,
                required=frozenset({"kind", "requester_user_id"}),
                code="invalid_history_authority",
            )
            return cls(
                kind=kind,
                requester_user_id=_snowflake(
                    raw["requester_user_id"], "invalid_history_authority"
                ),
            )
        raw = _exact(
            value,
            required=frozenset({"kind", "cron_job_id"}),
            code="invalid_history_authority",
        )
        job_id = raw["cron_job_id"]
        if not isinstance(job_id, str) or _CRON_JOB_ID_RE.fullmatch(job_id) is None:
            raise DiscordConnectorProtocolError("invalid_history_authority")
        return cls(kind=kind, cron_job_id=job_id)

    def to_mapping(self) -> dict[str, str]:
        if self.kind is DiscordConnectorHistoryAuthorityKind.AUTHENTICATED_USER:
            if self.requester_user_id is None:
                raise DiscordConnectorProtocolError("invalid_history_authority")
            return {
                "kind": self.kind.value,
                "requester_user_id": self.requester_user_id,
            }
        if self.cron_job_id is None:
            raise DiscordConnectorProtocolError("invalid_history_authority")
        return {"kind": self.kind.value, "cron_job_id": self.cron_job_id}

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_mapping())


@dataclass(frozen=True)
class DiscordConnectorEvent:
    event_id: str
    target: DiscordConnectorTarget
    author_id: str
    author_name: str
    author_is_bot: bool
    content: str
    created_at_unix_ms: int
    reply_to_message_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordConnectorEvent":
        raw = _exact(
            value,
            required=frozenset(
                {
                    "event_id",
                    "target",
                    "author_id",
                    "author_name",
                    "author_is_bot",
                    "content",
                    "created_at_unix_ms",
                    "reply_to_message_id",
                }
            ),
            code="invalid_public_event",
        )
        if type(raw["author_is_bot"]) is not bool:
            raise DiscordConnectorProtocolError("invalid_public_event")
        return cls(
            event_id=_snowflake(raw["event_id"], "invalid_public_event"),
            target=DiscordConnectorTarget.from_mapping(raw["target"]),
            author_id=_snowflake(raw["author_id"], "invalid_public_event"),
            author_name=_bounded_text(
                raw["author_name"],
                code="invalid_public_event",
                max_chars=MAX_NAME_CHARS,
                max_bytes=MAX_NAME_CHARS * 4,
            ),
            author_is_bot=raw["author_is_bot"],
            content=_bounded_text(raw["content"], code="invalid_public_event"),
            created_at_unix_ms=_integer(
                raw["created_at_unix_ms"],
                minimum=1,
                maximum=(1 << 63) - 1,
                code="invalid_public_event",
            ),
            reply_to_message_id=_optional_snowflake(
                raw["reply_to_message_id"], "invalid_public_event"
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "target": self.target.to_mapping(),
            "author_id": self.author_id,
            "author_name": self.author_name,
            "author_is_bot": self.author_is_bot,
            "content": self.content,
            "created_at_unix_ms": self.created_at_unix_ms,
            "reply_to_message_id": self.reply_to_message_id,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_mapping())


@dataclass(frozen=True)
class DiscordConnectorHistoryMessage:
    """One bounded public-guild text observation returned by the connector."""

    message_id: str
    author_id: str
    author_name: str
    author_is_bot: bool
    content: str
    content_truncated: bool
    created_at_unix_ms: int
    reply_to_message_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordConnectorHistoryMessage":
        raw = _exact(
            value,
            required=frozenset(
                {
                    "message_id",
                    "author_id",
                    "author_name",
                    "author_is_bot",
                    "content",
                    "content_truncated",
                    "created_at_unix_ms",
                    "reply_to_message_id",
                }
            ),
            code="invalid_public_history_message",
        )
        if (
            type(raw["author_is_bot"]) is not bool
            or type(raw["content_truncated"]) is not bool
        ):
            raise DiscordConnectorProtocolError("invalid_public_history_message")
        return cls(
            message_id=_snowflake(
                raw["message_id"], "invalid_public_history_message"
            ),
            author_id=_snowflake(raw["author_id"], "invalid_public_history_message"),
            author_name=_bounded_history_author_name(
                raw["author_name"],
                code="invalid_public_history_message",
            ),
            author_is_bot=raw["author_is_bot"],
            content=_bounded_history_text(
                raw["content"],
                code="invalid_public_history_message",
                allow_empty=True,
            ),
            content_truncated=raw["content_truncated"],
            created_at_unix_ms=_integer(
                raw["created_at_unix_ms"],
                minimum=1,
                maximum=(1 << 63) - 1,
                code="invalid_public_history_message",
            ),
            reply_to_message_id=_optional_snowflake(
                raw["reply_to_message_id"], "invalid_public_history_message"
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "author_is_bot": self.author_is_bot,
            "content": self.content,
            "content_truncated": self.content_truncated,
            "created_at_unix_ms": self.created_at_unix_ms,
            "reply_to_message_id": self.reply_to_message_id,
        }


@dataclass(frozen=True)
class DiscordConnectorHistoryPage:
    """Exact chronological page from one proven public Discord target."""

    target: DiscordConnectorTarget
    messages: tuple[DiscordConnectorHistoryMessage, ...]
    limit: int
    before_message_id: str | None
    after_message_id: str | None
    has_more: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "DiscordConnectorHistoryPage":
        raw = _exact(
            value,
            required=frozenset(
                {
                    "target",
                    "messages",
                    "query",
                    "has_more",
                    "order",
                }
            ),
            code="invalid_public_history_page",
        )
        query = _exact(
            raw["query"],
            required=frozenset(
                {"limit", "before_message_id", "after_message_id"}
            ),
            code="invalid_public_history_page",
        )
        limit = _integer(
            query["limit"],
            minimum=1,
            maximum=MAX_HISTORY_MESSAGES,
            code="invalid_public_history_page",
        )
        before = _optional_snowflake(
            query["before_message_id"], "invalid_public_history_page"
        )
        after = _optional_snowflake(
            query["after_message_id"], "invalid_public_history_page"
        )
        if before is not None and after is not None:
            raise DiscordConnectorProtocolError("invalid_public_history_page")
        if (
            not isinstance(raw["messages"], list)
            or len(raw["messages"]) > limit
            or type(raw["has_more"]) is not bool
            or (raw["has_more"] and len(raw["messages"]) != limit)
            or raw["order"] != "oldest_to_newest"
        ):
            raise DiscordConnectorProtocolError("invalid_public_history_page")
        messages = tuple(
            DiscordConnectorHistoryMessage.from_mapping(item)
            for item in raw["messages"]
        )
        ids = [item.message_id for item in messages]
        timestamps = [item.created_at_unix_ms for item in messages]
        if len(ids) != len(set(ids)) or timestamps != sorted(timestamps):
            raise DiscordConnectorProtocolError("invalid_public_history_page")
        return cls(
            target=DiscordConnectorTarget.from_mapping(raw["target"]),
            messages=messages,
            limit=limit,
            before_message_id=before,
            after_message_id=after,
            has_more=raw["has_more"],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "target": self.target.to_mapping(),
            "messages": [item.to_mapping() for item in self.messages],
            "query": {
                "limit": self.limit,
                "before_message_id": self.before_message_id,
                "after_message_id": self.after_message_id,
            },
            "has_more": self.has_more,
            "order": "oldest_to_newest",
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_mapping())


@dataclass(frozen=True)
class DiscordConnectorRequest:
    kind: DiscordConnectorKind
    request_id: str
    payload: Mapping[str, Any]


def parse_request(value: Mapping[str, Any]) -> DiscordConnectorRequest:
    envelope = _exact(
        value,
        required=frozenset({"protocol", "kind", "request_id", "payload"}),
        code="invalid_request_shape",
    )
    if envelope["protocol"] != PROTOCOL_VERSION:
        raise DiscordConnectorProtocolError("unsupported_protocol")
    try:
        kind = DiscordConnectorKind(envelope["kind"])
    except (TypeError, ValueError) as exc:
        raise DiscordConnectorProtocolError("unknown_request_kind") from exc
    request_id = _uuid(envelope["request_id"], "invalid_request_id")
    payload = _parse_payload(kind, envelope["payload"])
    return DiscordConnectorRequest(kind, request_id, payload)


def _parse_payload(kind: DiscordConnectorKind, value: Any) -> dict[str, Any]:
    if kind is DiscordConnectorKind.HELLO:
        return _exact(
            value,
            required=frozenset({"consumer"}),
            code="invalid_hello",
        ) | {
            "consumer": _bounded_text(
                dict(value)["consumer"],
                code="invalid_hello",
                max_chars=64,
                max_bytes=64,
            )
        }
    if kind is DiscordConnectorKind.EVENT_NEXT:
        raw = _exact(
            value,
            required=frozenset({"wait_ms"}),
            code="invalid_event_next",
        )
        return {
            "wait_ms": _integer(
                raw["wait_ms"], minimum=0, maximum=MAX_WAIT_MS, code="invalid_event_next"
            )
        }
    if kind in {
        DiscordConnectorKind.EVENT_ACK,
        DiscordConnectorKind.EVENT_ACK_READBACK,
    }:
        raw = _exact(
            value,
            required=frozenset({"delivery_id", "event_id", "event_sha256"}),
            code=(
                "invalid_event_ack"
                if kind is DiscordConnectorKind.EVENT_ACK
                else "invalid_event_ack_readback"
            ),
        )
        code = (
            "invalid_event_ack"
            if kind is DiscordConnectorKind.EVENT_ACK
            else "invalid_event_ack_readback"
        )
        digest = raw["event_sha256"]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise DiscordConnectorProtocolError(code)
        return {
            "delivery_id": _uuid(raw["delivery_id"], code),
            "event_id": _snowflake(raw["event_id"], code),
            "event_sha256": digest,
        }
    if kind is DiscordConnectorKind.TARGET_GET:
        raw = _exact(
            value,
            required=frozenset({"channel_id"}),
            code="invalid_target_query",
        )
        return {"channel_id": _snowflake(raw["channel_id"], "invalid_target_query")}
    if kind is DiscordConnectorKind.HISTORY_FETCH:
        raw = _exact(
            value,
            required=frozenset(
                {
                    "channel_id",
                    "limit",
                    "before_message_id",
                    "after_message_id",
                    "authority",
                }
            ),
            code="invalid_history_query",
        )
        before = _optional_snowflake(
            raw["before_message_id"], "invalid_history_query"
        )
        after = _optional_snowflake(
            raw["after_message_id"], "invalid_history_query"
        )
        if before is not None and after is not None:
            raise DiscordConnectorProtocolError("invalid_history_query")
        return {
            "channel_id": _snowflake(raw["channel_id"], "invalid_history_query"),
            "limit": _integer(
                raw["limit"],
                minimum=1,
                maximum=MAX_HISTORY_MESSAGES,
                code="invalid_history_query",
            ),
            "before_message_id": before,
            "after_message_id": after,
            "authority": DiscordConnectorHistoryAuthority.from_mapping(
                raw["authority"]
            ).to_mapping(),
        }
    raw = _exact(
        value,
        required=frozenset(
            {
                "idempotency_key",
                "target",
                "content",
                "reply_to_message_id",
                "deadline_unix_ms",
            }
        ),
        code="invalid_send",
    )
    deadline = _integer(
        raw["deadline_unix_ms"],
        minimum=1,
        maximum=(1 << 63) - 1,
        code="invalid_send_deadline",
    )
    now_ms = int(time.time() * 1000)
    if deadline <= now_ms or deadline > now_ms + MAX_DEADLINE_SECONDS * 1000:
        raise DiscordConnectorProtocolError("invalid_send_deadline")
    return {
        "idempotency_key": _idempotency(raw["idempotency_key"]),
        "target": DiscordConnectorTarget.from_mapping(raw["target"]).to_mapping(),
        "content": _bounded_text(raw["content"], code="invalid_send"),
        "reply_to_message_id": _optional_snowflake(
            raw["reply_to_message_id"], "invalid_send"
        ),
        "deadline_unix_ms": deadline,
    }


def request_message(
    kind: DiscordConnectorKind,
    payload: Mapping[str, Any],
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    message = {
        "protocol": PROTOCOL_VERSION,
        "kind": kind.value,
        "request_id": request_id or str(uuid.uuid4()),
        "payload": dict(payload),
    }
    parsed = parse_request(message)
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": parsed.kind.value,
        "request_id": parsed.request_id,
        "payload": dict(parsed.payload),
    }


def receipt(
    *,
    request: DiscordConnectorRequest,
    status: str,
    result: Mapping[str, Any],
    replayed: bool = False,
) -> dict[str, Any]:
    if status not in {"ok", "idle", "blocked", "failed", "dispatch_uncertain"}:
        raise ValueError("invalid connector receipt status")
    unsigned = {
        "protocol": PROTOCOL_VERSION,
        "receipt_version": RECEIPT_VERSION,
        "kind": request.kind.value,
        "request_id": request.request_id,
        "status": status,
        "replayed": bool(replayed),
        "result": dict(result),
    }
    return {**unsigned, "receipt_sha256": sha256_json(unsigned)}


def validate_receipt(
    value: Any,
    *,
    expected_kind: DiscordConnectorKind,
    expected_request_id: str,
) -> dict[str, Any]:
    raw = _exact(
        value,
        required=frozenset(
            {
                "protocol",
                "receipt_version",
                "kind",
                "request_id",
                "status",
                "replayed",
                "result",
                "receipt_sha256",
            }
        ),
        code="invalid_connector_receipt",
    )
    digest = raw.pop("receipt_sha256")
    if (
        raw["protocol"] != PROTOCOL_VERSION
        or raw["receipt_version"] != RECEIPT_VERSION
        or raw["kind"] != expected_kind.value
        or raw["request_id"] != expected_request_id
        or type(raw["replayed"]) is not bool
        or raw["status"]
        not in {"ok", "idle", "blocked", "failed", "dispatch_uncertain"}
        or not isinstance(raw["result"], Mapping)
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or sha256_json(raw) != digest
    ):
        raise DiscordConnectorProtocolError("invalid_connector_receipt")
    return {**raw, "result": dict(raw["result"]), "receipt_sha256": digest}


__all__ = [
    "MAX_DEADLINE_SECONDS",
    "MAX_FRAME_BYTES",
    "MAX_HISTORY_CONTENT_BYTES",
    "MAX_HISTORY_MESSAGES",
    "MAX_WAIT_MS",
    "PROTOCOL_VERSION",
    "DiscordConnectorEvent",
    "DiscordConnectorHistoryMessage",
    "DiscordConnectorHistoryPage",
    "DiscordConnectorHistoryAuthority",
    "DiscordConnectorHistoryAuthorityKind",
    "DiscordConnectorKind",
    "DiscordConnectorProtocolError",
    "DiscordConnectorRequest",
    "DiscordConnectorTarget",
    "DiscordConnectorTargetType",
    "canonical_json_bytes",
    "decode_frame",
    "parse_request",
    "receipt",
    "request_message",
    "sha256_json",
    "validate_receipt",
]
