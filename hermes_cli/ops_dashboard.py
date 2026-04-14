"""Normalization helpers for the Jax ops dashboard read model.

This module builds lightweight session/run records directly from Hermes session
JSON files in ``HERMES_HOME/sessions``. It is intentionally read-only and keeps
just enough metadata for overview and active-work surfaces without loading full
transcripts into memory-heavy UI models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home

_ACTIVE_WINDOW_SECONDS = 300
_DEFAULT_SUMMARY_LENGTH = 240
_MAX_SUMMARY_SOURCE_LENGTH = 4000
_CRON_FILE_RE = re.compile(r"^session_cron_([^_]+)_")
_CRON_SESSION_RE = re.compile(r"^cron_([^_]+)_")
_ISSUE_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_OBSIDIAN_PROJECT_RE = re.compile(r"/Projects/([^/\n]+)/")
_WHITESPACE_RE = re.compile(r"\s+")
_ERROR_SNIPPET_RE = re.compile(
    r'("status"\s*:\s*"error"|"success"\s*:\s*false|traceback|exception|http\s+[45]\d\d|"exit_code"\s*:\s*[1-9])',
    re.IGNORECASE,
)


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    source_path: str
    transcript_path: str
    platform: Optional[str]
    model: Optional[str]
    started_at: Optional[str]
    updated_at: Optional[str]
    message_count: int
    assistant_message_count: int
    tool_result_count: int
    tool_call_count: int
    last_role: Optional[str]
    summary: str
    issue_identifiers: list[str]
    project_hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    session_id: str
    run_type: str
    platform: Optional[str]
    status: str
    cron_job_id: Optional[str]
    transcript_path: str
    started_at: Optional[str]
    updated_at: Optional[str]
    duration_seconds: Optional[int]
    message_count: int
    tool_call_count: int
    latest_tool_name: Optional[str]
    summary: str
    failure_reason: Optional[str]
    issue_identifiers: list[str]
    project_hints: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionRunBundle:
    session: SessionRecord
    run: RunRecord


def get_sessions_dir() -> Path:
    return get_hermes_home() / "sessions"


def list_session_run_bundles(
    sessions_dir: Path | None = None,
    *,
    now: datetime | None = None,
    limit: int | None = None,
) -> list[SessionRunBundle]:
    """Load and normalize session JSON files from disk.

    Results are ordered by ``updated_at`` descending, then filename descending.
    Invalid JSON files are skipped to keep dashboard ingestion resilient.
    """
    base = sessions_dir or get_sessions_dir()
    if not base.exists():
        return []

    bundles: list[SessionRunBundle] = []
    for path in sorted(base.glob("session*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        bundles.append(normalize_session_file(path, payload=payload, now=now))

    bundles.sort(
        key=lambda bundle: (
            bundle.run.updated_at or "",
            bundle.session.source_path,
        ),
        reverse=True,
    )
    if limit is not None:
        bundles = bundles[:limit]
    return bundles


def list_session_records(
    sessions_dir: Path | None = None,
    *,
    now: datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return [bundle.session.to_dict() for bundle in list_session_run_bundles(sessions_dir, now=now, limit=limit)]


def list_run_records(
    sessions_dir: Path | None = None,
    *,
    now: datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    return [bundle.run.to_dict() for bundle in list_session_run_bundles(sessions_dir, now=now, limit=limit)]


def normalize_session_file(
    path: Path,
    *,
    payload: Optional[dict[str, Any]] = None,
    now: datetime | None = None,
) -> SessionRunBundle:
    data = payload if payload is not None else _load_json(path) or {}
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []

    session_id = str(data.get("session_id") or path.stem)
    started_dt = _parse_dt(data.get("session_start"))
    updated_dt = _parse_dt(data.get("last_updated"))
    last_message = messages[-1] if messages else {}

    tool_call_names = _tool_call_names(messages)
    summary = _derive_summary(messages)
    issue_identifiers = _extract_issue_identifiers(_summary_text_sources(messages, summary))
    project_hints = _extract_project_hints(_summary_text_sources(messages, summary))

    session_record = SessionRecord(
        session_id=session_id,
        source_path=str(path),
        transcript_path=str(path),
        platform=_clean_string(data.get("platform")),
        model=_clean_string(data.get("model")),
        started_at=_to_iso(started_dt),
        updated_at=_to_iso(updated_dt),
        message_count=int(data.get("message_count") or len(messages)),
        assistant_message_count=sum(1 for message in messages if message.get("role") == "assistant"),
        tool_result_count=sum(1 for message in messages if message.get("role") == "tool"),
        tool_call_count=len(tool_call_names),
        last_role=_clean_string(last_message.get("role")),
        summary=summary,
        issue_identifiers=issue_identifiers,
        project_hints=project_hints,
    )

    run_type, cron_job_id = _derive_run_type(path.name, session_id, _clean_string(data.get("platform")))
    status = _derive_run_status(messages, updated_dt=updated_dt, now=now)
    failure_reason = _derive_failure_reason(messages, status)
    duration_seconds = _duration_seconds(started_dt, updated_dt)

    run_record = RunRecord(
        run_id=session_id,
        session_id=session_id,
        run_type=run_type,
        platform=_clean_string(data.get("platform")),
        status=status,
        cron_job_id=cron_job_id,
        transcript_path=str(path),
        started_at=_to_iso(started_dt),
        updated_at=_to_iso(updated_dt),
        duration_seconds=duration_seconds,
        message_count=session_record.message_count,
        tool_call_count=session_record.tool_call_count,
        latest_tool_name=tool_call_names[-1] if tool_call_names else None,
        summary=summary,
        failure_reason=failure_reason,
        issue_identifiers=issue_identifiers,
        project_hints=project_hints,
    )
    return SessionRunBundle(session=session_record, run=run_record)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _duration_seconds(started_at: Optional[datetime], updated_at: Optional[datetime]) -> Optional[int]:
    if not started_at or not updated_at:
        return None
    return max(0, int((updated_at - started_at).total_seconds()))


def _derive_run_type(filename: str, session_id: str, platform: Optional[str]) -> tuple[str, Optional[str]]:
    match = _CRON_FILE_RE.match(filename) or _CRON_SESSION_RE.match(session_id)
    if match:
        return "cron", match.group(1)
    if platform in {"cli", "discord", "telegram", "slack", "whatsapp", "signal", "matrix", "mattermost", "feishu", "wecom", "dingtalk", "sms", "email"}:
        return "interactive", None
    if platform == "cron":
        return "cron", None
    return "interactive", None


def _derive_run_status(
    messages: list[dict[str, Any]],
    *,
    updated_dt: Optional[datetime],
    now: datetime | None,
) -> str:
    if not messages:
        return "failed"

    current_time = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    last_message = messages[-1]
    last_role = last_message.get("role")
    finish_reason = _clean_string(last_message.get("finish_reason"))

    if last_role == "assistant" and finish_reason == "stop":
        return "completed"

    if updated_dt is not None:
        age_seconds = (current_time - updated_dt).total_seconds()
        if age_seconds <= _ACTIVE_WINDOW_SECONDS:
            if last_role == "assistant" and finish_reason in {"incomplete", "tool_calls"}:
                return "running"
            if last_role != "assistant":
                return "running"

    if last_role == "assistant" and finish_reason is None and _clean_string(last_message.get("content")):
        return "completed"

    return "failed"


def _derive_failure_reason(messages: list[dict[str, Any]], status: str) -> Optional[str]:
    if status != "failed":
        return None

    for message in reversed(messages):
        text = _message_text(message.get("content"))
        if text and _ERROR_SNIPPET_RE.search(text):
            return _truncate(_collapse_whitespace(text), _DEFAULT_SUMMARY_LENGTH)

    last_message = messages[-1] if messages else {}
    last_role = _clean_string(last_message.get("role")) or "unknown"
    return f"Run ended without a final assistant response (last role: {last_role})."


def _derive_summary(messages: list[dict[str, Any]]) -> str:
    final_assistant = _latest_message_text(messages, role="assistant")
    if final_assistant:
        return _truncate(final_assistant, _DEFAULT_SUMMARY_LENGTH)

    last_tool_text = _latest_message_text(messages, role="tool")
    if last_tool_text:
        return _truncate(last_tool_text, _DEFAULT_SUMMARY_LENGTH)

    last_error = None
    for message in reversed(messages):
        text = _message_text(message.get("content"))
        if text and _ERROR_SNIPPET_RE.search(text):
            last_error = text
            break
    if last_error:
        return _truncate(last_error, _DEFAULT_SUMMARY_LENGTH)

    first_user = _latest_message_text(list(reversed(messages)), role="user")
    if first_user:
        return _truncate(first_user, _DEFAULT_SUMMARY_LENGTH)

    return "No transcript summary available."


def _latest_message_text(messages: Iterable[dict[str, Any]], *, role: str) -> Optional[str]:
    for message in reversed(list(messages)):
        if message.get("role") != role:
            continue
        text = _message_text(message.get("content"))
        if text:
            return _collapse_whitespace(text)
    return None


def _tool_call_names(messages: Iterable[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            name = _clean_string(((tool_call or {}).get("function") or {}).get("name"))
            if name:
                names.append(name)
    return names


def _message_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return _collapse_whitespace(content[:_MAX_SUMMARY_SOURCE_LENGTH]) or None
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    fragments.append(text)
        joined = " ".join(fragments)
        return _collapse_whitespace(joined[:_MAX_SUMMARY_SOURCE_LENGTH]) or None
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return _collapse_whitespace(text[:_MAX_SUMMARY_SOURCE_LENGTH]) or None
        return _collapse_whitespace(json.dumps(content)[:_MAX_SUMMARY_SOURCE_LENGTH]) or None
    return _collapse_whitespace(str(content)[:_MAX_SUMMARY_SOURCE_LENGTH]) or None


def _summary_text_sources(messages: list[dict[str, Any]], summary: str) -> list[str]:
    sources: list[str] = [summary]
    if messages:
        first_user = next((msg for msg in messages if msg.get("role") == "user"), None)
        if first_user:
            text = _message_text(first_user.get("content"))
            if text:
                sources.append(text)
        last_assistant = next((msg for msg in reversed(messages) if msg.get("role") == "assistant"), None)
        if last_assistant:
            text = _message_text(last_assistant.get("content"))
            if text:
                sources.append(text)
    return sources


def _extract_issue_identifiers(texts: Iterable[str]) -> list[str]:
    candidates: list[str] = []
    for text in texts:
        candidates.extend(_ISSUE_RE.findall(text or ""))
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result[:10]


def _extract_project_hints(texts: Iterable[str]) -> list[str]:
    candidates: list[str] = []
    for text in texts:
        candidates.extend(match.group(1) for match in _OBSIDIAN_PROJECT_RE.finditer(text or ""))
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result[:10]


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _truncate(text: str, limit: int) -> str:
    collapsed = _collapse_whitespace(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _clean_string(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
