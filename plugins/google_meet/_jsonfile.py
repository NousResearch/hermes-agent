"""Tiny JSON file helpers shared by the bot, process manager, node registry and node server."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

_say_queue_lock_holder = threading.local()


def read_json(path: Path) -> Optional[Any]:
    """Parsed JSON from *path*, or None when missing/unreadable/malformed."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


@contextmanager
def say_queue_lock(queue_path: Path) -> Iterator[None]:
    """Cross-process lock around ``say_queue.jsonl`` edits. The bot's speaker re-reads and rewrites the
    queue to drop what it spoke; an agent-side append landing in between would be wiped by the rewrite."""
    from hermes_cli.auth import _file_lock

    lock_path = queue_path.with_name(queue_path.name + ".lock")
    with _file_lock(lock_path, _say_queue_lock_holder, 10.0, f"Timed out waiting for say queue lock ({lock_path})"):
        yield
