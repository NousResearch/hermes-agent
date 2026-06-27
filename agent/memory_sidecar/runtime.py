from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from .models import Observation, ObservationFile, SessionFact
from .store import MemorySidecarStore

logger = logging.getLogger(__name__)

_EXPLICIT_OBSERVATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("decision:", "decision"),
    ("verdict:", "decision"),
    ("recommendation:", "decision"),
    ("next step:", "next_step"),
    ("next steps:", "next_step"),
    ("todo:", "next_step"),
    ("issue:", "bug"),
    ("bug:", "bug"),
    ("problem:", "bug"),
    ("changed:", "change"),
    ("updated:", "change"),
    ("implemented:", "change"),
    ("added:", "change"),
    ("removed:", "change"),
    ("patched:", "change"),
    ("how it works:", "how_it_works"),
)
_PATH_RE = re.compile(r"(?:~?/)?[A-Za-z0-9_./-]+\.(?:py|md|json|yaml|yml|toml|js|ts|tsx|jsx|sql|sh)")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _normalize_title(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return _truncate(first_line or text.strip(), 120)


def _extract_files(text: str) -> tuple[ObservationFile, ...]:
    seen: set[str] = set()
    files: list[ObservationFile] = []
    for match in _PATH_RE.findall(text):
        path = match.rstrip(".,:;)\"]'")
        if path in seen:
            continue
        seen.add(path)
        files.append(ObservationFile(file_path=path))
        if len(files) >= 8:
            break
    return tuple(files)


def _classify_observation(text: str) -> Optional[tuple[str, float]]:
    first_line = next((line.strip().lower() for line in text.splitlines() if line.strip()), "")
    for prefix, kind in _EXPLICIT_OBSERVATION_PATTERNS:
        if first_line.startswith(prefix):
            return kind, 0.9
    return None


def ingest_message(
    store: MemorySidecarStore,
    *,
    session_id: str,
    message_id: int,
    role: str,
    content: object,
) -> None:
    if not isinstance(content, str):
        return
    text = content.strip()
    if not text:
        return

    now_iso = _now_iso()
    current_fact = store.get_session_fact(session_id)
    user_goal = current_fact.user_goal if current_fact else None
    latest_summary = current_fact.latest_summary if current_fact else None

    if role == "user" and not user_goal and len(text) >= 12:
        user_goal = _truncate(text, 280)
    if role == "assistant":
        latest_summary = _truncate(text, 280)

    if user_goal or latest_summary or current_fact:
        store.upsert_session_fact(
            SessionFact(
                session_id=session_id,
                user_goal=user_goal,
                latest_summary=latest_summary,
                last_seen_at=now_iso,
            )
        )

    classified = _classify_observation(text)
    if not classified:
        return
    observation_type, confidence = classified
    store.insert_observation(
        Observation(
            session_id=session_id,
            message_id=str(message_id),
            role=role,
            event_ts=now_iso,
            observation_type=observation_type,
            title=_normalize_title(text),
            summary=_truncate(text, 240),
            detail=text[:4000],
            concepts=(observation_type, role),
            files=_extract_files(text),
            privacy_status="public",
            confidence=confidence,
        )
    )
