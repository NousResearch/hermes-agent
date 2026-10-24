"""Ctrl+Z suspend scope for the CLI (#83006): signal only this process.

Suspending the whole process group (``os.kill(0, SIGTSTP)``) also stopped
every background job sharing it, so a stray 0x1A byte in pasted input could
look like a crash that killed background work. Mirroring shell job-control
semantics - only the foreground process stops - keeps background tasks
running while ``fg`` resumes the CLI.
"""

from __future__ import annotations

import logging
import os
import signal

logger = logging.getLogger(__name__)


def _suspend_cli_process() -> None:
    """Suspend only the CLI process, not its whole process group (Unix).

    Ctrl+Z used to signal the group via ``os.kill(0, SIGTSTP)``, stopping
    every background job sharing it (long-running terminal/OCR tasks) -
    which made a stray 0x1A byte in pasted input *look* like a crash
    (#83006). Shell job control stops only the foreground job; mirror that
    by signaling just this process so ``fg`` resumes the CLI while
    background tasks keep running.

    Known limit: this narrows, but does not eliminate, the paste-trigger
    half of #83006. A pasted 0x1A whose bytes reach the key parser (torn
    paste where the ESC[200~ start marker is lost, or a raw 0x1A keypress)
    still fires the ``c-z`` binding and suspends the CLI itself - only the
    collateral group suspension is gone. A raw 0x1A is byte-identical to a
    real Ctrl+Z, so it is inherently undetectable app-side; leave it as a
    documented terminal/job-control limit.
    """
    logger.info("Ctrl+Z: suspending Hermes CLI (pid %d)", os.getpid())
    os.kill(os.getpid(), signal.SIGTSTP)
