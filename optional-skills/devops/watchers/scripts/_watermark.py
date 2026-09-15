"""Shared watermark helper used by the three watcher scripts.

A watermark is just a JSON file that records the IDs we've seen on previous
runs, so the next run only emits items we haven't seen before.

Contract:
- First run: record all IDs from the fetched batch, emit nothing.
- Subsequent runs: emit items whose ID isn't in the stored set.
- Bounded: keep at most `max_seen` IDs (default 500).
- Atomic: write to a .tmp file and rename, so a crashed script can't
  leave a half-written state file that permanently breaks dedup.

Import and use from any custom watcher script:

    from _watermark import Watermark

    wm = Watermark.load("my-feed-name")
    new_items = wm.filter_new(fetched_items, id_key="id")
    wm.save()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _state_dir() -> Path:
    """Where watermark files live — respects WATCHER_STATE_DIR override."""
    override = os.environ.get("WATCHER_STATE_DIR")
    if override:
        return Path(override)
    # Default: $HERMES_HOME/watcher-state/, falling back to ~/.hermes/watcher-state/.
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(hermes_home) / "watcher-state"


class Watermark:
    """Per-watcher state. Persisted to <state_dir>/<name>.json."""

    def __init__(self, name: str, *, max_seen: int = 500) -> None:
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"watermark name must be alphanumeric + '-'/'_' (got {name!r})"
            )
        self.name = name
        self.max_seen = max_seen
        self._path = _state_dir() / f"{name}.json"
        self._data: Dict[str, Any] = {"seen_ids": [], "first_run": True}

    @classmethod
    def load(cls, name: str, *, max_seen: int = 500) -> "Watermark":
        wm = cls(name, max_seen=max_seen)
        if wm._path.exists():
            try:
                wm._data = json.loads(wm._path.read_text(encoding="utf-8"))
                wm._data.setdefault("seen_ids", [])
                wm._data["first_run"] = False
            except (OSError, json.JSONDecodeError):
                # Corrupt state file — treat as a first run but don't crash.
                wm._data = {"seen_ids": [], "first_run": True}
        return wm

    @property
    def is_first_run(self) -> bool:
        return bool(self._data.get("first_run", True))

    @property
    def seen(self) -> List[str]:
        return list(self._data.get("seen_ids", []))

    def filter_new(
        self, items: Iterable[Dict[str, Any]], *, id_key: str = "id"
    ) -> List[Dict[str, Any]]:
        """Return items whose id isn't in the stored set.

        Side effect: updates the in-memory seen set with every id in the
        batch (so save() persists the full new watermark).  On first run,
        records every id but returns an empty list (baseline, no replay).
        """
        existing = set(str(x) for x in self._data.get("seen_ids", []))
        was_first_run = self.is_first_run

        new_items: List[Dict[str, Any]] = []
        batch_ids: List[str] = []
        for item in items:
            ident = item.get(id_key)
            if ident is None:
                continue
            ident_str = str(ident)
            batch_ids.append(ident_str)
            if ident_str in existing:
                continue
            if was_first_run:
                continue  # record but don't emit
            new_items.append(item)

        combined = list(existing) + [i for i in batch_ids if i not in existing]
        if len(combined) > self.max_seen:
            combined = combined[-self.max_seen:]
        self._data["seen_ids"] = combined
        self._data["first_run"] = False
        return new_items

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)


def format_items_as_markdown(
    items: List[Dict[str, Any]],
    *,
    title_key: str = "title",
    url_key: str = "url",
    body_key: Optional[str] = None,
    max_body_chars: int = 500,
) -> str:
    """Render a list of items as Markdown for cron delivery.

    One heading per item + its URL + optional snippet of body.  Output is
    empty string when items is empty — cron will then treat stdout as
    silent and skip delivery (existing behavior).
    """
    if not items:
        return ""
    lines: List[str] = []
    for item in items:
        title = (item.get(title_key) or "(no title)").strip()
        url = (item.get(url_key) or "").strip()
        lines.append(f"## {title}")
        if url:
            lines.append(url)
        if body_key:
            body = (item.get(body_key) or "").strip()
            if body:
                if len(body) > max_body_chars:
                    body = body[:max_body_chars].rstrip() + "…"
                lines.append("")
                lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
