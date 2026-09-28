"""Startup / attach latency: one bucketed row per process start and surface.

In-process surfaces (classic CLI, messaging gateway, ``hermes serve``) measure from the OS process
creation time, the earliest timestamp available (it includes interpreter start and imports), to the
moment they are ready. The Ink TUI and the Desktop app measure on their own side and report through
the ``shared_metrics.startup_latency`` RPC. Recording goes through the shared ``_emit`` gate, so a
profile without shared metrics enabled records nothing.
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from typing import Any

from . import shared_metrics_contract as contract
from .shared_metrics_contract import _bucket, _non_negative_number

logger = logging.getLogger(__name__)

_THRESHOLDS_MS = (
    (500, "lt_500ms"), (1_000, "500ms_to_1s"), (2_000, "1s_to_2s"), (5_000, "2s_to_5s"), (10_000, "5s_to_10s"),
)
# (pid, surface): a forked child re-arms, a surface in this process counts once.
_recorded: set[tuple[int, str]] = set()
_lock = threading.Lock()


def latency_bucket(elapsed_ms: Any) -> str | None:
    value = _non_negative_number(elapsed_ms)
    return None if value is None else _bucket(value, _THRESHOLDS_MS, "gte_10s")


def startup_latency_fields(*, surface: Any, elapsed_ms: Any) -> dict[str, str] | None:
    bucket = latency_bucket(elapsed_ms)
    if surface not in contract.STARTUP_SURFACES or bucket is None:
        return None
    return {"latency_bucket": bucket, "surface": surface}


def record_startup_latency(*, surface: str, elapsed_ms: Any) -> None:
    """Record one measured startup (no-op unless shared metrics are on; never raises)."""
    from .shared_metrics_events import _emit

    _emit(contract.STARTUP_LATENCY_MARK, startup_latency_fields, surface=surface, elapsed_ms=elapsed_ms)


def process_started_at() -> float | None:
    """Epoch seconds at which the OS created this process, or ``None`` when unreadable."""
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return None


def _claim(surface: str) -> bool:
    key = (os.getpid(), surface)
    with _lock:
        if key in _recorded:
            return False
        _recorded.add(key)
        return True


def _record_since_process_start(surface: str, ready_at: float) -> None:
    started = process_started_at()
    if started is not None:
        record_startup_latency(surface=surface, elapsed_ms=max(0.0, ready_at - started) * 1000)


def record_process_ready(surface: str, *, background: bool = False) -> None:
    """``surface`` became ready now: record process-start -> now once per process.

    ``background`` hands the (possibly cold, ~seconds) metrics runtime start to a daemon thread
    under the caller's contextvars (profile binding), so an event loop never waits on it; the
    ready time is taken before the hand-off.
    """
    ready_at = time.time()
    try:
        if not _claim(surface):
            return
        if not background:
            _record_since_process_start(surface, ready_at)
            return
        context = contextvars.copy_context()
        threading.Thread(
            target=context.run, args=(_record_since_process_start, surface, ready_at),
            name="hermes-startup-latency", daemon=True,
        ).start()
    except Exception:
        logger.debug("Startup latency for %s not recorded", surface, exc_info=True)


def cli_prompt_ready_handler():
    """A prompt_toolkit ``after_render`` handler that records the first rendered prompt once."""

    def _on_render(_app: Any) -> None:
        record_process_ready("cli", background=True)

    return _on_render


def record_cli_one_shot_ready() -> None:
    """A ``-q`` run is ready to dispatch its query. Kanban workers are dispatched processes, not a
    user waiting on a prompt, so they stay out of the CLI startup distribution."""
    if not os.environ.get("HERMES_KANBAN_TASK"):
        record_process_ready("cli")


def record_rpc_startup_latency(*, client_surface: Any, elapsed_ms: Any) -> None:
    """The TUI / Desktop client's own launch -> ready measurement, reported once per launch by the
    client. ``client_surface`` is the client's declared surface or the backend's detection
    (``desktop``/``tui``); anything else records nothing."""
    surface = {"desktop": "desktop_attach", "desktop_attach": "desktop_attach", "tui": "tui"}.get(client_surface)
    if surface is not None:
        record_startup_latency(surface=surface, elapsed_ms=elapsed_ms)
