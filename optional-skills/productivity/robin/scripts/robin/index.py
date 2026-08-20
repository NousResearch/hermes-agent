from __future__ import annotations

from collections.abc import Iterable

from robin.models import Entry


def _legacy_key(topic: str, date_added: str, seq: int) -> str:
    return f"{topic}:{date_added}:{seq:03d}"


def is_legacy_key(key: str) -> bool:
    parts = key.rsplit(":", 2)
    if len(parts) != 3:
        return False
    _, date_part, seq_part = parts
    return len(date_part) == 10 and date_part.count("-") == 2 and seq_part.isdigit() and len(seq_part) == 3


def rebuild_index(entries: Iterable[Entry], old_index: dict) -> dict:
    old_items = old_index.get("items", {})

    legacy_items: dict[str, dict] = {}
    id_items: dict[str, dict] = {}
    for key, value in old_items.items():
        if is_legacy_key(key):
            legacy_items[key] = value
        else:
            id_items[key] = value

    items: dict[str, dict] = {}
    seq_counters: dict[tuple[str, str], int] = {}

    for entry in entries:
        group = (entry.topic, entry.date_added)
        seq_counters[group] = seq_counters.get(group, 0) + 1
        seq = seq_counters[group]

        previous = id_items.get(entry.entry_id)
        if previous is None:
            previous = legacy_items.get(_legacy_key(entry.topic, entry.date_added, seq))

        items[entry.entry_id] = {
            "id": entry.entry_id,
            "topic": entry.topic,
            "date": entry.date_added,
            "rating": previous.get("rating") if previous else None,
            "last_surfaced": previous.get("last_surfaced") if previous else None,
            "times_surfaced": previous.get("times_surfaced", 0) if previous else 0,
            "_awaiting_rating": previous.get("_awaiting_rating", False) if previous else False,
        }

    return {"items": items}


def ensure_entry_in_index(entry: Entry, index: dict) -> None:
    index.setdefault("items", {})
    index["items"].setdefault(
        entry.entry_id,
        {
            "id": entry.entry_id,
            "topic": entry.topic,
            "date": entry.date_added,
            "rating": None,
            "last_surfaced": None,
            "times_surfaced": 0,
            "_awaiting_rating": False,
        },
    )
