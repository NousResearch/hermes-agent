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
``cron/__init__.py`` -- ``-m cron.scheduler`` executes the package before the module, and the
package's first import (``cron.jobs`` -> ``hermes_yaml`` -> ``ruamel``) is already a
dependency -- and does nothing unless ``_launch_external_cron_worker`` marked this child: the
gateway already booted through ``hermes_bootstrap``, and every other importer of
``cron.scheduler`` is an interpreter that owns its own dependencies.

A genuine activation failure propagates: continuing on the inherited, unleased path lets the
collector delete the generation under a live worker. The worker then exits before its ownership
acknowledgement, which ``_launch_external_cron_worker`` already reports as a failed dispatch with
the worker's stderr. PM's legitimate no-ops (no committed generation under a venv, wheel, Nix)
return normally.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set by ``_launch_external_cron_worker`` in the child's env and consumed here, so it never
# reaches the worker's own children -- they inherit the activated PYTHONPATH instead.
# Distinct from ``_HERMES_CRON_EXTERNAL_WORKER`` (the owning execution id) in scheduler.py.
WORKER_MARKER = "_HERMES_CRON_WORKER_BOOT"


def worker_bootstrap() -> None:
    """Run PM's dependency boot in the marked external worker, once."""
    if not os.environ.pop(WORKER_MARKER, None):
        return
    from pm.environments import activate_dependencies

    activate_dependencies(Path(__file__).resolve().parent.parent)
