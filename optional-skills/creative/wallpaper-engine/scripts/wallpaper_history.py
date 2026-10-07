#!/usr/bin/env python3
"""
Wallpaper generation history and feedback tracker.

Stores a JSON log of every generated wallpaper plus user ratings and tags.
Feeds into Hermes memory for long-term aesthetic preference learning.

Data directory: ``$HERMES_HOME/wallpaper-engine/``
History file:   ``$HERMES_HOME/wallpaper-engine/history.json``

Usage:
    python3 wallpaper_history.py add <image_path> <prompt> <workflow> [--meta '{"key":"val"}']
    python3 wallpaper_history.py feedback <id> <rating> [tags ...]
    python3 wallpaper_history.py list [--limit N] [--rated-only]
    python3 wallpaper_history.py stats
    python3 wallpaper_history.py get <id>

Stdlib-only; Python 3.10+.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _data_dir() -> Path:
    """Return the wallpaper-engine data directory (profile-aware)."""
    hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    return Path(hermes_home) / "wallpaper-engine"


def _history_path() -> Path:
    return _data_dir() / "history.json"


def _load() -> list[dict[str, Any]]:
    """Load the history JSON, returning an empty list if none exists."""
    hp = _history_path()
    if not hp.is_file():
        return []
    try:
        with open(hp, "r") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return data
    return []


def _save(entries: list[dict[str, Any]]) -> None:
    """Atomically write the history JSON."""
    hp = _history_path()
    hp.parent.mkdir(parents=True, exist_ok=True)
    tmp = hp.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(entries, fh, indent=2, default=str)
    tmp.replace(hp)


def cmd_add(args: list[str]) -> None:
    """Record a new wallpaper generation."""
    import argparse

    parser = argparse.ArgumentParser(prog="add")
    parser.add_argument("image_path", help="Path to the generated image file.")
    parser.add_argument("prompt", help="The generation prompt used.")
    parser.add_argument("workflow", help="Workflow filename used.")
    parser.add_argument("--meta", default="{}",
                       help="JSON string of extra metadata.")
    ns = parser.parse_args(args)

    try:
        meta = json.loads(ns.meta)
    except json.JSONDecodeError:
        meta = {}

    entry = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_path": str(Path(ns.image_path).resolve()),
        "prompt": ns.prompt,
        "workflow": ns.workflow,
        "rating": None,
        "tags": [],
        "rated_at": None,
        "meta": meta,
    }

    history = _load()
    history.append(entry)
    _save(history)

    print(json.dumps({"status": "ok", "id": entry["id"]}))


def cmd_feedback(args: list[str]) -> None:
    """Record user feedback (rating + tags) for an existing entry."""
    import argparse

    parser = argparse.ArgumentParser(prog="feedback")
    parser.add_argument("entry_id", help="ID of the entry to rate.")
    parser.add_argument("rating", type=int, choices=range(1, 6),
                       help="Rating 1-5 (1=dislike, 5=love).")
    parser.add_argument("tags", nargs="*", default=[],
                       help="Style/quality tags, e.g. 'dark' 'moody' 'mountains'.")
    ns = parser.parse_args(args)

    history = _load()
    for entry in history:
        if entry["id"] == ns.entry_id:
            entry["rating"] = ns.rating
            entry["tags"] = ns.tags
            entry["rated_at"] = datetime.now(timezone.utc).isoformat()
            _save(history)
            print(json.dumps({"status": "ok", "id": ns.entry_id,
                              "rating": ns.rating, "tags": ns.tags}))
            return

    print(json.dumps({"status": "error",
                      "error": f"Entry not found: {ns.entry_id}"}))
    sys.exit(1)


def cmd_list(args: list[str]) -> None:
    """List recent wallpaper generations."""
    import argparse

    parser = argparse.ArgumentParser(prog="list")
    parser.add_argument("--limit", "-n", type=int, default=20,
                       help="Max entries to show.")
    parser.add_argument("--rated-only", action="store_true",
                       help="Only show entries with feedback.")
    ns = parser.parse_args(args)

    history = _load()
    if ns.rated_only:
        history = [e for e in history if e.get("rating") is not None]

    # Most recent first
    history = list(reversed(history))[:ns.limit]

    # Strip image_path for compact display (paths are long)
    summary = []
    for e in history:
        summary.append({
            "id": e["id"],
            "timestamp": e["timestamp"],
            "prompt": e["prompt"][:120],
            "workflow": e["workflow"],
            "rating": e["rating"],
            "tags": e["tags"],
        })

    print(json.dumps({"status": "ok", "count": len(summary), "entries": summary}))


def cmd_get(args: list[str]) -> None:
    """Get full details for a single entry by ID."""
    import argparse

    parser = argparse.ArgumentParser(prog="get")
    parser.add_argument("entry_id", help="ID of the entry to retrieve.")
    ns = parser.parse_args(args)

    history = _load()
    for entry in history:
        if entry["id"] == ns.entry_id:
            print(json.dumps({"status": "ok", "entry": entry}))
            return

    print(json.dumps({"status": "error",
                      "error": f"Entry not found: {ns.entry_id}"}))
    sys.exit(1)


def cmd_stats(args: list[str]) -> None:
    """Compute preference statistics from rated entries."""
    history = _load()
    rated = [e for e in history if e.get("rating") is not None]

    if not rated:
        print(json.dumps({"status": "ok", "total": len(history),
                          "rated": 0, "message": "No ratings yet."}))
        return

    ratings = [e["rating"] for e in rated]
    avg_rating = sum(ratings) / len(ratings)

    # Tag frequency across rated entries
    tag_counter: Counter[str] = Counter()
    for e in rated:
        for tag in e.get("tags", []):
            tag_counter[tag.lower()] += 1

    # Group by rating bucket for preference insight
    loved = [e for e in rated if e["rating"] >= 4]
    disliked = [e for e in rated if e["rating"] <= 2]

    loved_tags: Counter[str] = Counter()
    for e in loved:
        for tag in e.get("tags", []):
            loved_tags[tag.lower()] += 1

    disliked_tags: Counter[str] = Counter()
    for e in disliked:
        for tag in e.get("tags", []):
            disliked_tags[tag.lower()] += 1

    # Recent trend (last 5 rated)
    recent = list(reversed(rated))[:5]
    recent_avg = (
        sum(e["rating"] for e in recent) / len(recent)
        if recent else None
    )

    result = {
        "status": "ok",
        "total_generated": len(history),
        "total_rated": len(rated),
        "average_rating": round(avg_rating, 2),
        "recent_average": round(recent_avg, 2) if recent_avg is not None else None,
        "top_tags": tag_counter.most_common(10),
        "loved_tags": loved_tags.most_common(5),
        "disliked_tags": disliked_tags.most_common(5),
    }
    print(json.dumps(result))


_COMMANDS = {
    "add": cmd_add,
    "feedback": cmd_feedback,
    "list": cmd_list,
    "get": cmd_get,
    "stats": cmd_stats,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(f"Usage: {sys.argv[0]} <{'|'.join(_COMMANDS)}> [...]", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    _COMMANDS[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()
