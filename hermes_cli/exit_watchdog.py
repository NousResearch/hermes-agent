"""Shared bounded-exit watchdog for interactive and one-shot CLI shutdown."""

from __future__ import annotations

import os
import threading
import time


def arm_exit_watchdog(
    timeout_s: float | None = None,
    exit_code: int = 0,
) -> None:
    """Force process exit when cleanup or interpreter teardown wedges."""
    if timeout_s is None:
        try:
            timeout_s = float(os.getenv("HERMES_EXIT_WATCHDOG_S", "30"))
        except (TypeError, ValueError):
            timeout_s = 30.0
    if timeout_s <= 0:
        return
    # Tests invoke cleanup directly; a delayed os._exit would kill the worker.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    def _watchdog() -> None:
        time.sleep(timeout_s)
        # Nothing may run between the deadline and forced exit: logging,
        # handler locks, stream flushes, and user callbacks can all block.
        # Final one-shot stdout is flushed before this watchdog is armed.
        os._exit(exit_code)

    threading.Thread(
        target=_watchdog,
        daemon=True,
        name="exit-watchdog",
    ).start()
