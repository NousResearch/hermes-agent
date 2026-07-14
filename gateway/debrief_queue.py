"""Minimal append-only queue for batched (non-urgent) alerts, consumed by
the daily debrief job. Full debrief content assembly is owned by the
2026-07-04 hermes-autonomy-overhaul spec (§4.7) — this just guarantees
batched alerts land somewhere durable instead of being dropped.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

QUEUE_PATH = Path.home() / ".hermes" / "debrief_queue.jsonl"


def queue_for_debrief(*, source: str, alert_type: str, message: str) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "source": source,
        "alert_type": alert_type,
        "message": message,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
    with QUEUE_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
