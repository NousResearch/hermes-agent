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
# (pid, surface, client launch id): a forked child re-arms, a surface in this process counts once,
# and an RPC-reporting client counts once per launch (a Desktop reconnect re-sends the same launch).
_recorded: set[tuple[int, str, str]] = set()
_lock = threading.Lock()
# A local client minting a fresh launch id per call must not grow this set without bound.
_MAX_RECORDED = 1024
_LAUNCH_ID_MAX_LEN = 64
# ``os.execvp`` keeps the PID, so a relaunched process (``sessions browse`` -> resume) would count
# the picker time as startup; relaunch() stamps its PID here right before the exec.
RELAUNCHED_PID_ENV = "HERMES_RELAUNCHED_PID"


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


def mark_in_place_relaunch() -> None:
    """Called right before an in-place ``exec``: the new program inherits this PID and env."""
    os.environ[RELAUNCHED_PID_ENV] = str(os.getpid())


def _relaunched_in_place() -> bool:
    # A child spawned later inherits the env but has its own PID, so it still counts.
    return os.environ.get(RELAUNCHED_PID_ENV) == str(os.getpid())


def _claim(surface: str, launch_id: str = "") -> bool:
    key = (os.getpid(), surface, launch_id)
    with _lock:
        if key in _recorded or len(_recorded) >= _MAX_RECORDED:
            return False
        _recorded.add(key)
        return True


def _collection_enabled() -> bool:
    from .relay_shared_metrics import enabled

    return enabled()


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
        # Claim before the gate so a later opt-in never records a stale "startup" mid-session.
        if _relaunched_in_place() or not _claim(surface) or not _collection_enabled():
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


def _launch_key(launch_id: Any) -> str:
    """The client's per-launch id, only ever a local latch key (never recorded). Older clients send
    none and latch on their side; they count once per backend process and surface."""
    return launch_id if isinstance(launch_id, str) and len(launch_id) <= _LAUNCH_ID_MAX_LEN else ""


def record_rpc_startup_latency(*, client_surface: Any, elapsed_ms: Any, launch_id: Any = None) -> None:
    """The TUI / Desktop client's own launch -> ready measurement, once per (surface, client launch)
    in this backend process. ``client_surface`` is the client's declared surface or the backend's
    detection (``desktop``/``tui``); anything else records nothing."""
    surface = {"desktop": "desktop_attach", "desktop_attach": "desktop_attach", "tui": "tui"}.get(client_surface)
    # Only a usable measurement spends the launch's claim.
    if surface is not None and latency_bucket(elapsed_ms) is not None and _claim(surface, _launch_key(launch_id)):
        record_startup_latency(surface=surface, elapsed_ms=elapsed_ms)
