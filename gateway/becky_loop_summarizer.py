"""Safe, bounded inputs and result contracts for Becky loop summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import unquote

try:
    from agent.redact import redact_sensitive_text as _force_redact
except Exception:  # pragma: no cover - incomplete Hermes installations only
    _force_redact = None


_INTERNAL_ENTRY_KINDS = frozenset({
    "system",
    "tool",
    "cron",
    "approval",
    "progress",
    "delegate",
    "branch-marker",
})
_MAX_VISIBLE_MESSAGES = 2_000
_MAX_VISIBLE_BYTES = 2 * 1024 * 1024
_DEFAULT_CHUNK_BYTES = 48 * 1024
_URL_USERINFO_PATTERN = re.compile(r"(?P<prefix>https?://)(?P<userinfo>[^@/\s]+)@")
_URL_QUERY_VALUE_PATTERN = re.compile(
    r"(?P<prefix>[?&][^=\s&#]+)(?P<equals>=)(?P<value>[^&#\s]*)"
)


@dataclass(frozen=True)
class VisibleMessage:
    """A redacted public conversation turn, with only local references."""

    ref: str
    role: Literal["user", "assistant"]
    text: str
    occurred_at: datetime


@dataclass(frozen=True)
class StructuredLoopSummary:
    about: str
    action_needed: str | None
    decisions: list[str]
    unresolved_items: list[str]
    waiting_on: Literal["user", "becky", "external", "none", "unknown"]
    key_event_refs: list[str]
    final_outcome: str | None


@dataclass(frozen=True)
class LoopSummary:
    summary: str
    decisions: list[str]
    unresolved_items: list[str]
    next_action: str | None
    waiting_on: str
    key_events: list[dict[str, str]]
    final_outcome: str | None


class _ConversationTooLarge(ValueError):
    """Private signal used by the bridge to return conversation_too_large."""

    def __init__(self) -> None:
        super().__init__("conversation_too_large")


class LoopSummarizer:
    """Provider seam; provider integration is intentionally added separately."""

    def summarize(
        self,
        *,
        row: dict[str, Any],
        transcript: list[dict[str, Any]],
        deadline: float,
    ) -> LoopSummary:
        del row, transcript, deadline
        raise NotImplementedError("Loop summary provider is not configured")


def extract_visible_messages(
    transcript: list[dict[str, Any]], hidden_values: set[str]
) -> list[VisibleMessage]:
    """Return only active, redacted user and assistant turns from a transcript."""
    visible: list[VisibleMessage] = []
    for entry in transcript:
        if not _is_visible_entry(entry):
            continue
        timestamp = _parse_timestamp(entry.get("timestamp"))
        if timestamp is None:
            continue
        text = _safe_public_text(entry["content"], hidden_values)
        if not text:
            continue
        visible.append(
            VisibleMessage(
                ref=f"m{len(visible) + 1:06d}",
                role=entry["role"],
                text=text,
                occurred_at=timestamp,
            )
        )
    return visible


def chunk_visible_messages(
    messages: list[VisibleMessage], max_bytes: int = _DEFAULT_CHUNK_BYTES
) -> list[list[VisibleMessage]]:
    """Group complete visible messages into bounded UTF-8 chunks."""
    if len(messages) > _MAX_VISIBLE_MESSAGES:
        raise _ConversationTooLarge()
    if _visible_bytes(messages) > _MAX_VISIBLE_BYTES:
        raise _ConversationTooLarge()
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    chunks: list[list[VisibleMessage]] = []
    chunk: list[VisibleMessage] = []
    chunk_bytes = 0
    for message in messages:
        message_bytes = len(message.text.encode("utf-8"))
        if message_bytes > max_bytes:
            raise _ConversationTooLarge()
        if chunk and chunk_bytes + message_bytes > max_bytes:
            chunks.append(chunk)
            chunk = []
            chunk_bytes = 0
        chunk.append(message)
        chunk_bytes += message_bytes
    if chunk:
        chunks.append(chunk)
    return chunks


def _is_visible_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    role = entry.get("role")
    if role not in {"user", "assistant"}:
        return False
    if entry.get("active", True) in {False, 0} or entry.get("deleted"):
        return False
    if entry.get("inactive"):
        return False
    if entry.get("rewound") or entry.get("is_rewound"):
        return False
    if any(
        entry.get(flag)
        for flag in ("branch_marker", "is_branch_marker", "delegate", "is_delegate")
    ):
        return False
    for field in ("type", "kind", "message_type", "entry_type", "event_type"):
        value = entry.get(field)
        if (
            isinstance(value, str)
            and value.lower().replace("_", "-") in _INTERNAL_ENTRY_KINDS
        ):
            return False
    return isinstance(entry.get("content"), str) and bool(entry["content"].strip())


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else None
    return None


def _safe_public_text(value: str, hidden_values: set[str]) -> str:
    if _force_redact is None:
        return "[REDACTED]"
    try:
        text = _force_redact(value, force=True)
    except Exception:
        return "[REDACTED]"
    hidden_values = _nonempty_hidden_values(hidden_values)
    text = _redact_url_components(text, hidden_values)
    for hidden in hidden_values:
        if hidden.isdigit():
            pattern = rf"(?<!\d){re.escape(hidden)}(?!\d)"
        else:
            pattern = rf"(?<![A-Za-z0-9_-]){re.escape(hidden)}(?![A-Za-z0-9_-])"
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


def _nonempty_hidden_values(hidden_values: set[str]) -> list[str]:
    return sorted(
        (item for item in hidden_values if isinstance(item, str) and item.strip()),
        key=len,
        reverse=True,
    )


def _redact_url_components(text: str, hidden_values: list[str]) -> str:
    def redact_userinfo(match: re.Match[str]) -> str:
        return (
            match["prefix"] + _redact_url_value(match["userinfo"], hidden_values) + "@"
        )

    def redact_query_value(match: re.Match[str]) -> str:
        return (
            match["prefix"]
            + match["equals"]
            + _redact_url_value(match["value"], hidden_values)
        )

    text = _URL_USERINFO_PATTERN.sub(redact_userinfo, text)
    return _URL_QUERY_VALUE_PATTERN.sub(redact_query_value, text)


def _redact_url_value(value: str, hidden_values: list[str]) -> str:
    if any(unquote(value).casefold() == hidden.casefold() for hidden in hidden_values):
        return "[REDACTED]"
    return value


def _visible_bytes(messages: list[VisibleMessage]) -> int:
    return sum(len(message.text.encode("utf-8")) for message in messages)
