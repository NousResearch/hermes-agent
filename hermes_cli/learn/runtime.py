"""In-process Learn sampler runtime for Desktop/web-server opt-in mode."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from hermes_constants import get_hermes_home

from . import analyzer, sampler, state

_runtime_lock = threading.Lock()
_runtimes: dict[str, tuple[threading.Event, threading.Thread]] = {}


def _worker(home: Path, stop_event: threading.Event, interval_seconds: float) -> None:
    while not stop_event.is_set():
        current = state._load_state(home)
        if current.get("mode") == "off" or current.get("state") != "running":
            break
        sampler.sample_once(home=home)
        analyzer.create_usage_suggestions(home=home)
        stop_event.wait(interval_seconds)


def ensure_running(*, home: Path | None = None, interval_seconds: float = 60.0) -> bool:
    """Start the profile-local Learn sampler loop if it is not already running."""
    if os.environ.get("HERMES_LEARN_DISABLE_RUNTIME"):
        return False

    resolved_home = (home or get_hermes_home()).resolve()
    key = str(resolved_home)
    with _runtime_lock:
        existing = _runtimes.get(key)
        if existing and existing[1].is_alive():
            return False

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_worker,
            args=(resolved_home, stop_event, max(1.0, float(interval_seconds))),
            daemon=True,
            name=f"learn-sampler-{resolved_home.name}",
        )
        _runtimes[key] = (stop_event, thread)
        thread.start()
        return True


def stop_runtime(*, home: Path | None = None) -> bool:
    """Signal the profile-local Learn sampler loop to stop."""
    resolved_home = (home or get_hermes_home()).resolve()
    key = str(resolved_home)
    with _runtime_lock:
        existing = _runtimes.pop(key, None)
        if not existing:
            return False
        existing[0].set()
        return True
