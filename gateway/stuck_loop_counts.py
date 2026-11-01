"""Persisted per-session restart-failure counters (``.restart_failure_counts``, #7536).

Every load/mutate/save of the file runs under one lock: session completions clear their
counter from worker threads concurrently, and an unserialized pair rewrites a stale snapshot
that resurrects the other session's counter.
"""

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Nothing re-acquires it while held, so a plain Lock suffices.
_COUNTS_LOCK = threading.Lock()


def _read_counts(path: Path) -> dict:
    """Parsed counters, or {} when the file is missing, unreadable or not a JSON object."""
    try:
        counts = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return counts if isinstance(counts, dict) else {}


def increment_restart_failure_counts(path: Path, active_session_keys: set) -> None:
    """Increment the counters of sessions active at shutdown; drop the rest (their loop is broken)."""
    from utils import atomic_json_write
    with _COUNTS_LOCK:
        counts = _read_counts(path)
        try:
            atomic_json_write(path, {key: counts.get(key, 0) + 1 for key in active_session_keys}, indent=None)
        except Exception:
            logger.debug("Could not write restart-failure counters to %s", path, exc_info=True)


def clear_restart_failure_count(path: Path, session_key: str) -> None:
    """Drop a completed session's counter; removes the file once no counter is left.

    Blocking (``atomic_json_write`` fsyncs): callers on the event loop run it via ``asyncio.to_thread``.
    """
    from utils import atomic_json_write
    with _COUNTS_LOCK:
        counts = _read_counts(path)
        if session_key not in counts:
            return
        del counts[session_key]
        try:
            if counts:
                atomic_json_write(path, counts, indent=None)
            else:
                path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not clear restart-failure counter in %s", path, exc_info=True)
