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

That boot covers dependencies only. ``hermes_bootstrap`` also finishes a pending source update
and may relaunch the process into a fresh one (``hermes_cli/venv_sync.py::relaunch_command``, an
``-I`` interpreter that ignores the pinned ``PYTHONPATH``). ``finish_worker_boot()`` runs it from
the worker's ``__main__`` before the payload is read, so a relaunch replays the whole worker: the
payload is still on disk and the marker still set, and the new process boots its dependencies
again. Left to ``run_agent``'s import inside ``run_one_job``, the relaunch came after the ack, with
the payload deleted and the marker consumed: the new process died on ``ruamel`` and the adopted
run was left ``unknown``.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set by ``_launch_external_cron_worker`` in the child's env and consumed by
# ``finish_worker_boot()`` -- after any relaunch, before the ack -- so it never reaches the
# worker's own children; they inherit the activated PYTHONPATH instead.
# Distinct from ``_HERMES_CRON_EXTERNAL_WORKER`` (the owning execution id) in scheduler.py.
WORKER_MARKER = "_HERMES_CRON_WORKER_BOOT"


def worker_bootstrap() -> None:
    """Run PM's dependency boot in the marked external worker (again after a relaunch)."""
    if not os.environ.get(WORKER_MARKER):
        return
    from pm.environments import activate_dependencies

    activate_dependencies(Path(__file__).resolve().parent.parent)


def finish_worker_boot() -> None:
    """Run ``hermes_bootstrap`` before the worker reads its payload; may relaunch the worker.

    Not from ``worker_bootstrap()``: while ``-m cron.scheduler`` executes ``cron/__init__.py``,
    ``sys.argv[0]`` is still ``-m`` and ``__main__`` has no spec, so the relaunch command could
    not name the module to re-run. In ``__main__`` both are set.
    """
    import hermes_bootstrap  # noqa: F401

    os.environ.pop(WORKER_MARKER, None)
