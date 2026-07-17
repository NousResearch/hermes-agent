"""Shared bounded-exit watchdog for interactive and one-shot CLI shutdown."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time


logger = logging.getLogger(__name__)


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
        try:
            logger.warning(
                "Exit watchdog fired after %.0fs — forcing process exit "
                "(a cleanup step or non-daemon thread is wedged).",
                timeout_s,
            )
        except Exception:
            pass
        try:
            logging.shutdown()
        except Exception:
            pass
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os._exit(exit_code)

    try:
        threading.Thread(
            target=_watchdog,
            daemon=True,
            name="exit-watchdog",
        ).start()
    except Exception:
        pass
