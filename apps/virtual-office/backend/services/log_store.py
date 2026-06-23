import json
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("VIRTUAL_OFFICE_DATA_ROOT", str(PROJECT_ROOT / "data")))
LOGS_PATH = DATA_ROOT / "logs" / "events.jsonl"


def _ensure_store() -> None:
    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOGS_PATH.exists():
        LOGS_PATH.touch()


def append_event(
    level: str,
    message: str,
    agent: str,
    task_id: str | None = None,
    handoff_id: str | None = None,
) -> dict[str, Any]:
    _ensure_store()
    event = {
        "id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level,
        "message": message,
        "metadata": {
            "agent": agent,
            "task_id": task_id,
            "handoff_id": handoff_id,
        },
    }
    with LOGS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return event


def list_events(
    limit: int = 100,
    level: str | None = None,
    agent: str | None = None,
    task_id: str | None = None,
    handoff_id: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_store()
    matched: deque[dict[str, Any]] = deque(maxlen=max(1, limit))
    normalized_level = level.lower() if level else None
    normalized_agent = agent.lower() if agent else None

    with LOGS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if normalized_level and str(event.get("level", "")).lower() != normalized_level:
                continue
            metadata = event.get("metadata") or {}
            if normalized_agent and str(metadata.get("agent", "")).lower() != normalized_agent:
                continue
            if task_id and str(metadata.get("task_id") or "") != task_id:
                continue
            if handoff_id and str(metadata.get("handoff_id") or "") != handoff_id:
                continue
            matched.append(event)

    return list(matched)


def get_event(log_id: str) -> dict[str, Any] | None:
    _ensure_store()
    with LOGS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if str(event.get("id") or "") == log_id:
                return event
    return None
