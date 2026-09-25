"""Cron external worker: the PM dependency boot its own entry point never gets.

``sys.executable -m cron.scheduler --external-worker-file ...``
(``cron/scheduler.py::_launch_external_cron_worker``) is a full Hermes entry point that
does not go through ``hermes_bootstrap``, and it imports Hermes packages the moment it
starts. ``cron/scheduler_worker_env.py`` restores the committed generation's
``site-packages`` on its ``PYTHONPATH`` so those imports resolve, but a pinned path is not a
boot: the worker holds no lease on the generation, so the PM collector may remove it
between the gateway's exit and the worker's next import (#122290 review), and its
``sys.path`` never ran the generation's ``.pth`` files.

``pm.environments.activate_dependencies`` is exactly that boot -- the same call
``hermes_bootstrap`` makes for every other entry point -- and it leases the generation it
selects for the life of the process. ``worker_bootstrap()`` runs it at the top of
``cron/scheduler.py`` (after the stdlib-only ``sys.path`` pin, before any Hermes package
import) and does nothing unless ``_launch_external_cron_worker`` marked this child: the
gateway already booted through ``hermes_bootstrap``, and every other importer of
``cron.scheduler`` is an interpreter that owns its own dependencies.

Failure degrades to a log line: the worker still has the pinned dependency path, and a
dependency problem must not be reported as a lost job.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Set by ``_launch_external_cron_worker`` in the child's env and consumed here, so it never
# reaches the worker's own children -- they inherit the activated PYTHONPATH instead.
WORKER_MARKER = "HERMES_CRON_EXTERNAL_WORKER"

_root = Path(__file__).resolve().parent.parent


def worker_bootstrap() -> None:
    """Run PM's dependency boot in the marked external worker; idempotent, never raises."""
    if not os.environ.pop(WORKER_MARKER, None):
        return
    try:
        from pm.environments import activate_dependencies

        activate_dependencies(_root)
    except Exception as exc:
        logger.warning("cron external worker dependency activation failed: %s", exc)