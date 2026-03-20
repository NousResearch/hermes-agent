"""Discord thread participation persistence helpers."""

from __future__ import annotations

import json
import logging

from hermes_cli.config import get_hermes_home


logger = logging.getLogger(__name__)


def thread_state_path():
    """Return the persisted thread participation state path."""
    return get_hermes_home() / "discord_threads.json"


def load_participated_threads() -> set[str]:
    """Load persisted thread IDs from disk."""
    path = thread_state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(data)
    except Exception as exc:
        logger.debug("Could not load discord thread state: %s", exc)
    return set()


def save_participated_threads(threads: set[str], max_threads: int = 500) -> set[str]:
    """Persist the current thread set to disk and return the trimmed set."""
    path = thread_state_path()
    try:
        thread_list = list(threads)
        if len(thread_list) > max_threads:
            thread_list = thread_list[-max_threads:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(thread_list), encoding="utf-8")
        return set(thread_list)
    except Exception as exc:
        logger.debug("Could not save discord thread state: %s", exc)
        return set(threads)


def track_thread(threads: set[str], thread_id: str, max_threads: int = 500) -> set[str]:
    """Add a thread to the participation set, persist it, and return the updated set."""
    if thread_id in threads:
        return set(threads)

    updated_threads = set(threads)
    updated_threads.add(thread_id)
    return save_participated_threads(updated_threads, max_threads=max_threads)
