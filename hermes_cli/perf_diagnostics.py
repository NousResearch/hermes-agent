"""Small, secret-free activity registry for gateway stall diagnostics."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import itertools
import logging
import os
import threading
import time
from typing import Iterator

_log = logging.getLogger(__name__)
_lock = threading.RLock()
_sequence = itertools.count(1)
_active: dict[int, dict] = {}
_recent_slow: deque[dict] = deque(maxlen=20)
_SLOW_ACTIVITY_SECONDS = 2.0


def begin_activity(category: str, name: str, **attributes: object) -> int:
    """Register an operation without retaining payloads, credentials, or IDs."""
    token = next(_sequence)
    safe_attributes = {
        str(key): value
        for key, value in attributes.items()
        if isinstance(value, (bool, int, float, str))
    }
    with _lock:
        _active[token] = {
            "category": str(category),
            "name": str(name),
            "started": time.monotonic(),
            "attributes": safe_attributes,
        }
    return token


def finish_activity(token: int, *, error: bool = False) -> None:
    with _lock:
        row = _active.pop(token, None)
    if row is None:
        return

    elapsed = time.monotonic() - row["started"]
    if elapsed < _SLOW_ACTIVITY_SECONDS:
        return

    completed = {
        "category": row["category"],
        "name": row["name"],
        "duration_ms": round(elapsed * 1000, 1),
        "error": bool(error),
        "attributes": row["attributes"],
    }
    with _lock:
        _recent_slow.append(completed)
    _log.info(
        "slow activity category=%s name=%s duration_ms=%.1f error=%s attributes=%s",
        completed["category"],
        completed["name"],
        completed["duration_ms"],
        completed["error"],
        completed["attributes"],
    )


@contextmanager
def track_activity(category: str, name: str, **attributes: object) -> Iterator[None]:
    token = begin_activity(category, name, **attributes)
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        finish_activity(token, error=failed)


def diagnostics_snapshot(*, active_limit: int = 8, recent_limit: int = 8) -> dict:
    """Return bounded process/profile identity and current/recent activities."""
    now = time.monotonic()
    with _lock:
        active = sorted(
            (
                {
                    "category": row["category"],
                    "name": row["name"],
                    "elapsed_ms": round((now - row["started"]) * 1000, 1),
                    "attributes": dict(row["attributes"]),
                }
                for row in _active.values()
            ),
            key=lambda row: row["elapsed_ms"],
            reverse=True,
        )[: max(0, active_limit)]
        recent_count = max(0, recent_limit)
        recent = list(_recent_slow)[-recent_count:] if recent_count else []

    return {
        "pid": os.getpid(),
        "profile": _profile_name(),
        "active": active,
        "recent_slow": recent,
    }


def _profile_name() -> str:
    explicit = os.environ.get("HERMES_PROFILE", "").strip()
    if explicit:
        return explicit
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _reset_for_tests() -> None:
    with _lock:
        _active.clear()
        _recent_slow.clear()
