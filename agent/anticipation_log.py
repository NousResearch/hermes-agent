"""Sanitized audit logging for anticipation decisions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from agent.anticipation_policy import AnticipationDecision


LOG_DIR_NAME = "anticipation"
LOG_FILE_NAME = "decisions.jsonl"


def decision_log_path() -> Path:
    return get_hermes_home() / LOG_DIR_NAME / LOG_FILE_NAME


def append_decision_log(decision: AnticipationDecision, *, now: datetime | None = None) -> Path:
    """Append one sanitized anticipation decision record and return the path."""

    timestamp = now or datetime.now(timezone.utc)
    path = decision_log_path()
    _ensure_secure_parent(path.parent)

    record = _decision_to_record(decision, timestamp)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    _secure_file(path)
    return path


def read_recent_decision_logs(*, limit: int = 10) -> list[dict[str, Any]]:
    """Read recent sanitized decision records, newest first."""

    path = decision_log_path()
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))[: max(0, limit)]


def _decision_to_record(decision: AnticipationDecision, timestamp: datetime) -> dict[str, Any]:
    candidate = decision.candidate
    return {
        "ts": timestamp.isoformat(),
        "loop_id": candidate.loop_id,
        "action": decision.action,
        "reason": decision.reason,
        "confidence": candidate.confidence if isfinite(candidate.confidence) else None,
        "dedupe_key_hash": sha256(candidate.dedupe_key.encode("utf-8")).hexdigest(),
        "title_hash": sha256(candidate.title.encode("utf-8")).hexdigest(),
    }


def _ensure_secure_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass
