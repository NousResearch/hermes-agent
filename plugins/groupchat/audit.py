"""Private, bounded JSONL logs for Groupchat filter decisions."""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import uuid

MAX_BYTES = 10 * 1024 * 1024
SAFE_FIELDS = frozenset({"decision", "reason_code", "pattern_index", "pattern_sha256",
                         "provider", "model", "fallback_used", "fail_open"})


def log_paths(profile_home, platform):
    directory = Path(profile_home) / "logs"
    return {"relevance": str(directory / f"{platform}-relevance-decisions.jsonl"),
            "pingpong": str(directory / f"{platform}-groupchat-outbound.jsonl")}


def record_outbound(profile_home, platform, room_id, scope, decision):
    """Log decisions, not delivery receipts. Failure must never block a reply."""
    try:
        path = Path(log_paths(profile_home, platform)["pingpong"])
        record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "profile": Path(profile_home).name, "platform": platform, "room_id": room_id,
                  "scope": scope, "decision_id": uuid.uuid4().hex, "direction": "outbound"}
        record.update({key: value for key, value in decision.items() if key in SAFE_FIELDS
                       and isinstance(value, (str, int, float, bool))})
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size >= MAX_BYTES:
            path.replace(path.with_suffix(".jsonl.1"))
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logging.getLogger(__name__).warning("Groupchat decision audit failed (%s)", type(exc).__name__)
