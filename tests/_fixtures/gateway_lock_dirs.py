"""Per-process gateway rendezvous directories for the test run.

``tests/conftest.py`` gives every pytest process its own
``HERMES_GATEWAY_LOCK_DIR`` named ``<prefix><pid>`` under the temp dir, and
sweeps the directories of processes that are gone at import time. The sweep
lives here so it can be tested without re-importing conftest.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import psutil

LOCK_DIR_PREFIX = "hermes-test-gateway-locks-"


def sweep_stale_lock_dirs(root: Path, prefix: str = LOCK_DIR_PREFIX) -> None:
    """Remove every ``<prefix><pid>`` directory under ``root`` whose PID is not running.

    Liveness is ``psutil.pid_exists``, never ``os.kill(pid, 0)``: on Windows
    signal 0 is ``CTRL_C_EVENT`` and goes through ``GenerateConsoleCtrlEvent``
    (bpo-14484), so probing a sibling worker would either interrupt its
    console process group or fail with ``OSError`` and have the sweep delete a
    live worker's directory. See CONTRIBUTING.md, "Cross-Platform
    Compatibility", critical rule 1.
    """
    for stale in root.glob(f"{prefix}*"):
        try:
            pid = int(stale.name[len(prefix):])
        except ValueError:
            continue
        if not psutil.pid_exists(pid):
            shutil.rmtree(stale, ignore_errors=True)
