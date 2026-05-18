"""Profile-local pending interaction handoff records.

These records are intentionally small and local to the active Hermes profile.
They bridge visible Discord follow-ups across cron jobs, goal continuation, and
gateway session splits without writing to user memory or external knowledge
stores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
import threading
import uuid
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write


STORE_VERSION = 1
DEFAULT_TTL_SECONDS = 24 * 60 * 60
PENDING_DIRNAME = "pending_interactions"
PENDING_FILENAME = "records.json"

STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"
STATUS_EXPIRED = "expired"
STATUS_AMBIGUOUS = "ambiguous"

_LOCK = threading.RLock()

_QUESTION_HINTS = (
    "?",
    "need your input",
    "need input",
    "blocked",
    "stuck",
    "let me know",
    "reply with",
    "respond with",
    "which",
    "choose",
    "confirm",
    "approve",
    "deny",
    "should i",
    "continue?",
    "proceed?",
    "어떻게",
    "어느",
    "무엇",
    "뭐",
    "선택",
    "확인",
    "답장",
    "알려",
    "진행할까요",
    "계속할까요",
    "할까요",
    "까요",
    "나요",
)

_CONFIRMATION_HINTS = (
    "yes",
    "no",
    "confirm",
    "approve",
    "deny",
    "proceed",
    "continue",
    "go ahead",
    "계속",
    "진행",
    "확인",
    "승인",
    "거절",
)

_REPLY_REFERENCE_HINTS = _CONFIRMATION_HINTS + (
    "next action",
    "do it",
    "that one",
    "option",
    "choice",
    "go with",
    "use that",
    "start",
    "다음",
    "액션",
    "그거",
    "그걸로",
    "이걸로",
    "그대로",
    "시작",
    "해줘",
)


@dataclass(frozen=True)
class PendingResolution:
    """Result of matching an inbound user reply to pending interactions."""

    status: str
    message: str
    record: Optional[dict[str, Any]] = None
    matches: tuple[dict[str, Any], ...] = ()


def pending_store_path() -> Path:
    """Return the active profile's pending interaction store path."""

    return get_hermes_home() / PENDING_DIRNAME / PENDING_FILENAME


def current_origin_profile() -> str:
    """Best-effort profile name for the active HERMES_HOME."""

    home = get_hermes_home()
    if home.parent.name == "profiles" and home.name:
        return home.name
    return "default"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normal_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _looks_like_pending_reply(text: str) -> bool:
    """Return True when text is plausibly a reply to a pending prompt.

    Pending interactions are scoped to a Discord channel/thread, but public
    channels can contain unrelated chatter. Never consume a pending record for
    arbitrary text just because it is the only open item in that location.
    """

    body = (text or "").strip()
    if not body or len(body) > 500:
        return False
    lowered = body.lower()
    if any(hint in lowered for hint in _REPLY_REFERENCE_HINTS):
        return True
    return bool(re.match(r"^(?:[1-9]\d?|[a-z])(?:[).:-]|\s*$)", lowered))


def _participant_matches(record: dict[str, Any], user_id: Optional[str]) -> bool:
    record_user = _normal_id(record.get("user_id"))
    if not record_user:
        return True
    return record_user == _normal_id(user_id)


def _normal_artifact_paths(paths: Optional[Iterable[Any]]) -> list[str]:
    if not paths:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in paths:
        if item is None:
            continue
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _read_records_unlocked() -> list[dict[str, Any]]:
    path = pending_store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("records", [])
    else:
        records = []
    return [dict(item) for item in records if isinstance(item, dict)]


def _write_records_unlocked(records: list[dict[str, Any]]) -> None:
    atomic_json_write(
        pending_store_path(),
        {
            "version": STORE_VERSION,
            "records": records,
        },
        sort_keys=True,
    )


def load_pending_interactions() -> list[dict[str, Any]]:
    """Return all pending interaction records for the active profile."""

    with _LOCK:
        return _read_records_unlocked()


def _expire_records(records: list[dict[str, Any]], now: datetime) -> bool:
    changed = False
    for record in records:
        if record.get("status") != STATUS_OPEN:
            continue
        expires_at = _parse_time(record.get("expires_at"))
        if expires_at is not None and expires_at <= now:
            record["status"] = STATUS_EXPIRED
            changed = True
    return changed


def _location_matches(record: dict[str, Any], platform: str, channel_id: Optional[str], thread_id: Optional[str]) -> bool:
    if str(record.get("platform", "")).lower() != platform.lower():
        return False

    record_channel = _normal_id(record.get("channel_id"))
    record_thread = _normal_id(record.get("thread_id"))
    incoming_channel = _normal_id(channel_id)
    incoming_thread = _normal_id(thread_id)

    if record_thread:
        return record_thread == incoming_thread or record_thread == incoming_channel
    if not record_channel:
        return False
    return record_channel == incoming_channel or record_channel == incoming_thread


def _record_dedupe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("platform", "")).lower(),
        _normal_id(record.get("channel_id")),
        _normal_id(record.get("thread_id")),
        _normal_id(record.get("user_id")),
        _normal_id(record.get("source_session_id")),
        _normal_id(record.get("job_id")),
        str(record.get("question_summary", "")).strip(),
    )


def record_pending_interaction(
    *,
    platform: str,
    channel_id: Any,
    thread_id: Any = None,
    user_id: Any = None,
    source_session_id: Any = None,
    job_id: Any = None,
    question_summary: str,
    expected_reply_shape: str,
    artifact_paths: Optional[Iterable[Any]] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    created_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Create or refresh a profile-local pending interaction record."""

    now = created_at or _utc_now()
    expires_at = now + timedelta(seconds=max(1, int(ttl_seconds)))
    record = {
        "id": f"pi_{uuid.uuid4().hex[:16]}",
        "origin_profile": current_origin_profile(),
        "platform": str(platform).lower(),
        "channel_id": _normal_id(channel_id),
        "thread_id": _normal_id(thread_id),
        "user_id": _normal_id(user_id),
        "source_session_id": _normal_id(source_session_id),
        "job_id": _normal_id(job_id),
        "question_summary": (question_summary or "").strip()[:500],
        "expected_reply_shape": (expected_reply_shape or "free-form reply").strip()[:240],
        "artifact_paths": _normal_artifact_paths(artifact_paths),
        "created_at": _format_time(now),
        "expires_at": _format_time(expires_at),
        "status": STATUS_OPEN,
    }

    if not record["platform"] or not (record["channel_id"] or record["thread_id"]):
        raise ValueError("platform and a Discord channel/thread id are required")
    if not record["question_summary"]:
        raise ValueError("question_summary is required")

    with _LOCK:
        records = _read_records_unlocked()
        _expire_records(records, now)
        incoming_key = _record_dedupe_key(record)
        for existing in records:
            if existing.get("status") == STATUS_OPEN and _record_dedupe_key(existing) == incoming_key:
                existing["expires_at"] = record["expires_at"]
                existing["artifact_paths"] = _normal_artifact_paths(
                    list(existing.get("artifact_paths") or []) + record["artifact_paths"]
                )
                _write_records_unlocked(records)
                return dict(existing)
        records.append(record)
        _write_records_unlocked(records)
        return dict(record)


def _interesting_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in lines:
        lowered = line.lower()
        if any(hint in lowered for hint in _QUESTION_HINTS):
            return line[:500]
    return lines[0][:500]


def infer_pending_question(text: str) -> Optional[tuple[str, str]]:
    """Return (summary, expected reply shape) when text appears to ask for input."""

    body = (text or "").strip()
    if not body or body.strip().upper() == "[SILENT]":
        return None
    lowered = body.lower()
    if not any(hint in lowered for hint in _QUESTION_HINTS):
        return None
    summary = _interesting_line(body)
    if not summary:
        return None
    expected = "confirmation or instruction" if any(hint in lowered for hint in _CONFIRMATION_HINTS) else "free-form reply"
    return summary, expected


def extract_artifact_paths(text: str) -> list[str]:
    """Extract local artifact hints from response text."""

    if not text:
        return []
    paths: list[str] = []
    for match in re.finditer(r"MEDIA:([^\s]+)", text):
        paths.append(match.group(1).strip())
    for match in re.finditer(r"(?<!\w)(/[^ \n\r\t]+)", text):
        candidate = match.group(1).rstrip(".,);]")
        if candidate and ("/docs/" in candidate or "/.hermes/" in candidate or "/tmp/" in candidate):
            paths.append(candidate)
    return _normal_artifact_paths(paths)


def maybe_record_pending_interaction(
    *,
    platform: str,
    channel_id: Any,
    thread_id: Any = None,
    user_id: Any = None,
    source_session_id: Any = None,
    job_id: Any = None,
    response_text: str,
    artifact_paths: Optional[Iterable[Any]] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> Optional[dict[str, Any]]:
    """Record a pending interaction if the response appears to request input."""

    inferred = infer_pending_question(response_text)
    if inferred is None:
        return None
    summary, expected = inferred
    combined_artifacts = _normal_artifact_paths(list(artifact_paths or []) + extract_artifact_paths(response_text))
    return record_pending_interaction(
        platform=platform,
        channel_id=channel_id,
        thread_id=thread_id,
        user_id=user_id,
        source_session_id=source_session_id,
        job_id=job_id,
        question_summary=summary,
        expected_reply_shape=expected,
        artifact_paths=combined_artifacts,
        ttl_seconds=ttl_seconds,
    )


def _build_handoff_message(record: dict[str, Any], reply_text: str) -> str:
    source_line = ""
    if record.get("source_session_id"):
        source_line = f"Source session id: {record['source_session_id']}\n"
    elif record.get("job_id"):
        source_line = f"Cron job id: {record['job_id']}\n"

    artifacts = record.get("artifact_paths") or []
    artifact_line = f"Artifact paths: {', '.join(str(p) for p in artifacts)}\n" if artifacts else ""

    return (
        "[Pending interaction handoff]\n"
        "The user's message is a reply to a visible pending interaction in this Discord thread/channel.\n"
        f"Pending interaction id: {record.get('id')}\n"
        f"Origin profile: {record.get('origin_profile')}\n"
        f"{source_line}"
        f"Question summary: {record.get('question_summary')}\n"
        f"Expected reply shape: {record.get('expected_reply_shape')}\n"
        f"{artifact_line}"
        "Use this visible pending interaction as the primary context. Do not infer a different task from runtime recall unless the user explicitly says so.\n\n"
        "User reply:\n"
        f"{reply_text}"
    )


def _build_ambiguity_message(matches: list[dict[str, Any]], reply_text: str) -> str:
    lines = []
    for record in matches[:6]:
        lines.append(
            f"- {record.get('id')}: {record.get('question_summary')} "
            f"(expected: {record.get('expected_reply_shape')})"
        )
    return (
        "[Pending interaction ambiguity]\n"
        "The user's message could refer to multiple open pending interactions in this Discord thread/channel.\n"
        "Ask the user to specify which pending item they want to continue before acting.\n\n"
        "Open pending interactions:\n"
        f"{chr(10).join(lines)}\n\n"
        "User reply:\n"
        f"{reply_text}"
    )


def resolve_pending_reply(
    *,
    platform: str,
    channel_id: Any,
    thread_id: Any = None,
    user_id: Any = None,
    reply_text: str,
    now: Optional[datetime] = None,
) -> PendingResolution:
    """Resolve an inbound Discord reply against visible pending records."""

    current_time = now or _utc_now()
    with _LOCK:
        records = _read_records_unlocked()
        changed = _expire_records(records, current_time)
        same_location = [
            record
            for record in records
            if _location_matches(record, str(platform), _normal_id(channel_id), _normal_id(thread_id))
        ]
        open_matches = [record for record in same_location if record.get("status") == STATUS_OPEN]
        open_matches = [record for record in open_matches if _participant_matches(record, _normal_id(user_id))]
        expired_matches = [record for record in same_location if record.get("status") == STATUS_EXPIRED]

        if open_matches and not _looks_like_pending_reply(reply_text):
            if changed:
                _write_records_unlocked(records)
            return PendingResolution(status="none", message=reply_text, matches=tuple(dict(record) for record in open_matches))

        if len(open_matches) == 1:
            record = open_matches[0]
            record["status"] = STATUS_RESOLVED
            changed = True
            _write_records_unlocked(records)
            return PendingResolution(
                status=STATUS_RESOLVED,
                message=_build_handoff_message(record, reply_text),
                record=dict(record),
                matches=(dict(record),),
            )

        if len(open_matches) > 1:
            if changed:
                _write_records_unlocked(records)
            match_copies = tuple(dict(record) for record in open_matches)
            return PendingResolution(
                status=STATUS_AMBIGUOUS,
                message=_build_ambiguity_message(open_matches, reply_text),
                matches=match_copies,
            )

        if changed:
            _write_records_unlocked(records)
        if expired_matches:
            return PendingResolution(
                status=STATUS_EXPIRED,
                message=reply_text,
                matches=tuple(dict(record) for record in expired_matches),
            )

    return PendingResolution(status="none", message=reply_text)
