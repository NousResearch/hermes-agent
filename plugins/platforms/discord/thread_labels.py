"""Local registry and parsing helpers for Discord thread labels.

These labels are intentionally independent of Discord forum tags so they work
for ordinary text-channel threads, forum posts, and test doubles alike.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).with_name("thread_labels.json")
_LABEL_TOKEN_RE = re.compile(r"\[([a-z][a-z0-9_-]{1,31})\]")


def _load_registry() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    labels = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for label_id, metadata in labels.items():
        if not isinstance(label_id, str) or not isinstance(metadata, dict):
            continue
        key = label_id.strip().lower()
        if not key:
            continue
        normalized[key] = {"id": key, **metadata}
    return normalized


DISCORD_THREAD_LABELS = _load_registry()


def extract_thread_labels(thread_name: str | None) -> list[dict[str, Any]]:
    """Return registered labels present as ``[label]`` tokens in a thread name."""
    if not isinstance(thread_name, str) or not thread_name:
        return []

    seen: set[str] = set()
    labels: list[dict[str, Any]] = []
    for match in _LABEL_TOKEN_RE.finditer(thread_name):
        label_id = match.group(1).lower()
        if label_id in seen:
            continue
        metadata = DISCORD_THREAD_LABELS.get(label_id)
        if metadata is None:
            continue
        seen.add(label_id)
        labels.append(dict(metadata))
    return labels
