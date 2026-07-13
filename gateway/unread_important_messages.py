"""Unread important-message detection for gateway text nudges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import re
from typing import Any

from gateway.important_contacts import is_important_sender
from gateway.platforms.base import MessageEvent, MessageType


@dataclass(frozen=True)
class ImportantUnreadMessageEvent:
    """Internal event emitted when an unread important text is nudge-eligible."""

    platform: str
    sender_id: str | None
    sender_name: str | None
    chat_id: str | None
    chat_id_alt: str | None
    stable_message_id: str | None
    dedupe_key: str
    text: str
    timestamp: datetime | None
    importance_score: int | None = None
    importance_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class MessageImportanceScore:
    """Local, non-LLM importance score for an incoming text message."""

    score: int
    reasons: tuple[str, ...]


@dataclass
class _NudgeState:
    """In-memory cadence state for one unread important message."""

    attempts: int = 0
    next_due_at: datetime | None = None


class ImportantUnreadNudgeScheduler:
    """Deduplicate important unread-message nudges behind an unread check.

    ``detect_due_nudge`` is intentionally synchronous and side-effect-light so
    gateway adapters can call it immediately before generating or scheduling a
    user-visible nudge.  A caller must provide the freshest unread state it has;
    read messages are tombstoned so a later duplicate delivery for the same
    message id cannot restart the nudge series.
    """

    def __init__(self, cadence_seconds: tuple[int, ...] | list[int] = (300, 900, 1800)):
        cleaned = tuple(int(value) for value in cadence_seconds if int(value) > 0)
        self._cadence_seconds = cleaned or (300,)
        self._states: dict[str, _NudgeState] = {}
        self._read_keys: set[str] = set()

    def mark_read(self, dedupe_key: str) -> None:
        """Stop all future nudges for a message once it is known read."""
        key = _clean(dedupe_key)
        if not key:
            return
        self._states.pop(key, None)
        self._read_keys.add(key)

    def mark_read_event(self, event: MessageEvent) -> None:
        """Stop nudges for a raw gateway message event that is now read."""
        if event is None:
            return
        platform = _platform_value(event)
        stable_id = _message_id(event)
        key = f"{platform}:{stable_id}" if stable_id else _unstable_dedupe_key(event, platform)
        self.mark_read(key)

    def detect_due_nudge(
        self,
        event: ImportantUnreadMessageEvent,
        *,
        is_unread: bool,
        now: datetime | None = None,
    ) -> ImportantUnreadMessageEvent | None:
        """Return *event* only when its next nudge is currently due.

        The unread state is checked before every generated nudge.  Repeated
        detections of the same unread message before ``next_due_at`` are
        swallowed, preventing webhook replay/burst spam while preserving the
        intended cadence.
        """
        if event is None:
            return None
        key = _clean(getattr(event, "dedupe_key", None))
        if not key or key in self._read_keys:
            return None
        if not is_unread:
            self.mark_read(key)
            return None

        current_time = now or datetime.now()
        state = self._states.get(key)
        if state is not None and state.next_due_at is not None and current_time < state.next_due_at:
            return None

        if state is None:
            state = _NudgeState()
            self._states[key] = state

        delay_index = min(state.attempts, len(self._cadence_seconds) - 1)
        state.next_due_at = current_time + timedelta(seconds=self._cadence_seconds[delay_index])
        state.attempts += 1
        return event


@dataclass(frozen=True)
class UnreadImportantMessageResult:
    """Decision result for the unread important-message nudge path."""

    eligible: bool
    reason: str
    internal_event: ImportantUnreadMessageEvent | None = None


def _platform_value(event: MessageEvent) -> str:
    platform = getattr(getattr(event, "source", None), "platform", None)
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _payload_record(raw_message: Any) -> dict[str, Any]:
    if not isinstance(raw_message, dict):
        return {}
    data = raw_message.get("data")
    if isinstance(data, dict):
        return data
    message = raw_message.get("message")
    if isinstance(message, dict):
        return message
    return raw_message


def _is_explicitly_read(raw_message: Any) -> bool:
    """Return True only for explicit read markers; missing status stays unread.

    Current text adapters generally receive new inbound webhook events and do
    not maintain external unread state. We therefore assume a new inbound text
    is unread unless the normalized/raw payload contains a clear read marker.
    """
    record = _payload_record(raw_message)
    if not record:
        return False

    for key in ("read", "isRead", "is_read"):
        value = record.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
            return True

    for key in ("unread", "isUnread", "is_unread"):
        value = record.get(key)
        if value is False:
            return True
        if isinstance(value, str) and value.strip().lower() in {"false", "0", "no"}:
            return True

    for key in ("dateRead", "date_read", "readAt", "read_at"):
        value = record.get(key)
        if value not in (None, "", 0, "0"):
            return True

    return False


def _message_id(event: MessageEvent) -> str | None:
    if _clean(getattr(event, "message_id", None)):
        return _clean(getattr(event, "message_id", None))
    source = getattr(event, "source", None)
    if _clean(getattr(source, "message_id", None)):
        return _clean(getattr(source, "message_id", None))
    record = _payload_record(getattr(event, "raw_message", None))
    for key in ("MessageSid", "messageSid", "guid", "messageGuid", "id"):
        if _clean(record.get(key)):
            return _clean(record.get(key))
    return None


_URGENT_RE = re.compile(r"\b(urgent|emergency|asap|important|911)\b", re.IGNORECASE)
_HELP_RE = re.compile(r"\b(help|stuck|need you|please respond|respond)\b", re.IGNORECASE)
_CALL_RE = re.compile(r"\b(call me|call|phone|facetime)\b", re.IGNORECASE)
_LOGISTICS_RE = re.compile(
    r"\b(pick\s*up|pickup|school|doctor|hospital|medicine|medication|appointment|ride|where are you)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"\?\s*$|\b(can you|could you|will you|are you|do you|did you|where|when|what|who|how|why)\b",
    re.IGNORECASE,
)


def _importance_scoring_config(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    value = config.get("importance_scoring")
    return value if isinstance(value, Mapping) else {}


def _importance_scoring_enabled(config: Any) -> bool:
    scoring = _importance_scoring_config(config)
    return scoring.get("enabled") is True


def _importance_threshold(config: Any) -> int:
    scoring = _importance_scoring_config(config)
    try:
        return max(1, int(scoring.get("threshold", 70)))
    except (TypeError, ValueError):
        return 70


def score_message_importance(text: str, config: Any = None) -> MessageImportanceScore:
    """Score message text for local content-based nudge importance.

    The scorer is intentionally deterministic and local-only. It does not call
    an LLM or persist message text; callers decide whether to use the score.
    """
    body = (text or "").strip()
    if not body:
        return MessageImportanceScore(0, ())

    score = 0
    reasons: list[str] = []
    if _URGENT_RE.search(body):
        score += 35
        reasons.append("urgent_keyword")
    if _HELP_RE.search(body):
        score += 25
        reasons.append("help_request")
    if _CALL_RE.search(body):
        score += 20
        reasons.append("call_request")
    if _LOGISTICS_RE.search(body):
        score += 25
        reasons.append("logistics_or_care")
    if _QUESTION_RE.search(body):
        score += 15
        reasons.append("direct_question")

    # Long multi-sentence messages that already hit an actionable signal tend
    # to be more intentional than one-word pings; cap keeps scoring legible.
    if reasons and len(body) >= 80:
        score += 5
        reasons.append("substantial_message")

    return MessageImportanceScore(min(score, 100), tuple(reasons))


def _unstable_dedupe_key(event: MessageEvent, platform: str) -> str:
    source = getattr(event, "source", None)
    timestamp = getattr(event, "timestamp", None)
    parts = [
        platform,
        _clean(getattr(source, "chat_id", None)) or "",
        _clean(getattr(source, "user_id", None)) or "",
        _clean(getattr(source, "chat_id_alt", None)) or "",
        timestamp.isoformat() if timestamp else "",
        event.text or "",
    ]
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{platform}:unstable:{digest}"


def evaluate_unread_important_message(
    event: MessageEvent,
    important_contacts_config: Any,
) -> UnreadImportantMessageResult:
    """Evaluate whether *event* should enter the important unread nudge path."""
    if event is None or getattr(event, "source", None) is None:
        return UnreadImportantMessageResult(False, "missing_source")
    if event.message_type is not MessageType.TEXT:
        return UnreadImportantMessageResult(False, "unsupported_message_type")
    if not (event.text or "").strip():
        return UnreadImportantMessageResult(False, "missing_text")
    if _is_explicitly_read(getattr(event, "raw_message", None)):
        return UnreadImportantMessageResult(False, "already_read")

    sender_matched = is_important_sender(event.source, important_contacts_config)
    importance_score: MessageImportanceScore | None = None
    if not sender_matched:
        if not _importance_scoring_enabled(important_contacts_config):
            return UnreadImportantMessageResult(False, "not_important_sender")
        importance_score = score_message_importance(event.text, important_contacts_config)
        if importance_score.score < _importance_threshold(important_contacts_config):
            return UnreadImportantMessageResult(False, "not_important_sender")

    platform = _platform_value(event)
    stable_id = _message_id(event)
    dedupe_key = f"{platform}:{stable_id}" if stable_id else _unstable_dedupe_key(event, platform)
    source = event.source
    internal_event = ImportantUnreadMessageEvent(
        platform=platform,
        sender_id=_clean(getattr(source, "user_id", None)),
        sender_name=_clean(getattr(source, "user_name", None)),
        chat_id=_clean(getattr(source, "chat_id", None)),
        chat_id_alt=_clean(getattr(source, "chat_id_alt", None)),
        stable_message_id=stable_id,
        dedupe_key=dedupe_key,
        text=event.text,
        timestamp=getattr(event, "timestamp", None),
        importance_score=importance_score.score if importance_score is not None else None,
        importance_reasons=importance_score.reasons if importance_score is not None else (),
    )
    reason = "eligible" if sender_matched else "scored_important"
    return UnreadImportantMessageResult(True, reason, internal_event)


def detect_unread_important_message(
    event: MessageEvent,
    important_contacts_config: Any,
) -> ImportantUnreadMessageEvent | None:
    """Return the internal nudge event for eligible messages, otherwise None."""
    return evaluate_unread_important_message(event, important_contacts_config).internal_event
