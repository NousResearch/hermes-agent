"""The host's wait on a compression commit that already won admission.

Sibling of ``agent/conversation_compression.py`` (the facade), which imports it at module level; this
module never imports the facade at module level (import cycle). It logs under the facade's logger name
so log consumers keep one source.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("agent.conversation_compression")

# Overrun waits proceed in bounded slices so each window logs (escalating); clamped to ceiling for tiny values.
_COMMIT_OVERRUN_WAIT_SLICE_SECONDS = 30.0


def _await_in_flight_commit(
    future: Any, *, ceiling: float, wait_started: float, on_commit_overrun: Optional[Callable[[float, float], None]]
) -> Any:
    """begin_commit won the race: the SessionDB mutation cannot be fence-cancelled, so wait
    in bounded slices, logging (escalating) + surfacing once via ``on_commit_overrun``
    WHILE the commit hangs. Never silently hung or abandoned.
    """
    overrun_surfaced = False
    overrun_reports = 0
    while True:
        waited = time.monotonic() - wait_started
        remaining = ceiling - waited
        if remaining <= 0:
            # Bounded increments so each overrun window is visible in logs rather than one silent unbounded block.
            remaining = min(_COMMIT_OVERRUN_WAIT_SLICE_SECONDS, max(ceiling, 0.05))
            overrun_reports += 1
            log = logger.warning if overrun_reports <= 2 else logger.error
            log(
                "Context compression SessionDB commit still running "
                "%.1fs past the total ceiling (waited %.1fs, ceiling %.1fs); commit cannot be abandoned mid-flight — "
                "continuing to wait (check SessionDB health if this persists)", waited - ceiling, waited,
                ceiling,
            )
            if not overrun_surfaced and on_commit_overrun is not None:
                from agent.conversation_compression import _swallow
                overrun_surfaced = True
                with _swallow('compress_context commit-overrun callback failed', exc_info=True):
                    on_commit_overrun(waited, ceiling)
        try:
            return future.result(timeout=remaining)
        except concurrent.futures.TimeoutError:
            # Aliases builtin TimeoutError (3.11+): also fires when the commit worker died with one (#63892).
            # A settled future never unsettles — this ceiling-less loop spun forever; re-raise the worker's error.
            if future.done():
                return future.result()
            # Commit-phase progress is informative only — the commit must complete; loop
            # and re-report with the updated overrun window.
            continue
