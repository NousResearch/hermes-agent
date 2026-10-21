from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from robin.models import Entry


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _most_recent_surfaced_topic(index: dict) -> str | None:
    latest_topic: str | None = None
    latest_timestamp: datetime | None = None
    for item in index.get("items", {}).values():
        last_surfaced = item.get("last_surfaced")
        if not last_surfaced:
            continue
        parsed = parse_timestamp(last_surfaced)
        if latest_timestamp is None or parsed > latest_timestamp:
            latest_timestamp = parsed
            latest_topic = item.get("topic")
    return latest_topic


def _candidate_sort_key(item: dict, entry: Entry) -> tuple[int, int, datetime, tuple[int, int], str]:
    last_surfaced = item.get("last_surfaced")
    rating = item.get("rating")
    return (
        item.get("times_surfaced", 0),
        0 if last_surfaced is None else 1,
        parse_timestamp(last_surfaced) if last_surfaced else datetime.min.replace(tzinfo=timezone.utc),
        (0, rating) if rating is not None else (1, 6),
        entry.entry_id,
    )


def pick_best_candidate(index: dict, entries: list[Entry], config: dict) -> tuple[dict, Entry] | None:
    entries_by_id = {entry.entry_id: entry for entry in entries}
    cooldown_days = config.get("review_cooldown_days", 60)
    cutoff = datetime.now(timezone.utc) - timedelta(days=cooldown_days)

    candidates: list[tuple[dict, Entry]] = []
    for entry_id, item in index.get("items", {}).items():
        entry = entries_by_id.get(entry_id)
        if entry is None:
            continue

        last_surfaced = item.get("last_surfaced")
        if last_surfaced and parse_timestamp(last_surfaced) > cutoff:
            continue

        candidates.append((item, entry))

    if not candidates:
        return None

    most_recent_topic = _most_recent_surfaced_topic(index)
    if most_recent_topic is not None:
        alternate_topic_candidates = [candidate for candidate in candidates if candidate[1].topic != most_recent_topic]
        if alternate_topic_candidates:
            candidates = alternate_topic_candidates

    candidates.sort(key=lambda candidate: _candidate_sort_key(candidate[0], candidate[1]))
    minimum_times_surfaced = candidates[0][0].get("times_surfaced", 0)
    top_pool = [candidate for candidate in candidates if candidate[0].get("times_surfaced", 0) == minimum_times_surfaced][:10]
    return random.choice(top_pool)


def mark_surfaced(index: dict, entry_id: str, *, awaiting_rating: bool = False) -> dict:
    if entry_id not in index.get("items", {}):
        raise KeyError(entry_id)

    item = index["items"][entry_id]
    item["last_surfaced"] = now_iso()
    item["times_surfaced"] = item.get("times_surfaced", 0) + 1
    item["_awaiting_rating"] = awaiting_rating
    return item


def rate_item(index: dict, entry_id: str, rating: int) -> dict:
    if entry_id not in index.get("items", {}):
        raise KeyError(entry_id)
    if rating < 1 or rating > 5:
        raise ValueError("Rating must be between 1 and 5.")

    item = index["items"][entry_id]
    item["rating"] = rating
    if not item.get("_awaiting_rating"):
        item["last_surfaced"] = now_iso()
        item["times_surfaced"] = item.get("times_surfaced", 0) + 1
    item["_awaiting_rating"] = False
    return item
