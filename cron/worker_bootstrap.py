"""Cron external worker: PM dependency boot at the package prelude.

The restart-safe external worker is spawned as ``sys.executable -m cron.scheduler``.
Python runs the ``cron`` package prelude (``cron/__init__.py``) *before* the
scheduler module body, and ``cron/__init__``'s first import (``cron.jobs``)
reaches the dependency graph: ``cron.jobs`` -> ``utils`` -> ``hermes_yaml`` ->
``ruamel.yaml``. Any lifetime protection installed *inside* ``cron/scheduler.py``
therefore runs only after that first application import is already in flight
-- too late: the PM collector can reclaim the generation the worker is importing
from in the window between the gateway's exit and that first import.

So the boot happens in the package prelude, before ``cron.jobs`` loads, and it
delegates to ``pm.environments.activate_dependencies`` -- the same call
``hermes_bootstrap`` makes for every other entry point.  That single call
resolves the committed generation **once**, leases it for the life of this
process (the child-held kernel lease the collector checks via ``leases_held``),
and puts its ``site-packages`` on ``sys.path`` through ``site.addsitedir``.
Because resolution, leasing and activation happen in one call, there is no
separate pre-lease that could pin a different generation than the one
ultimately imported.  ``activate_dependencies`` never re-execs the scheduler,
so the prelude stays a genuine stdlib-only boot.

Gate: the boot runs only when this process is the externally spawned worker
(``--external-worker-file`` in ``sys.argv``).  The gateway and every other
importer of the ``cron`` package no-op here -- the gateway already booted
through ``hermes_bootstrap`` -> ``activate_dependencies`` -- so they keep their
exact launch contract.

Failure in the marked worker is **fatal**, not swallowed: ``activate_dependencies``
raising (a corrupt selection, a committed environment with no ``site-packages``)
means the retention the worker needs is not established, and continuing on an
ambient import path the collector may reclaim is exactly the hazard this boot
removes.  The raise takes the worker down before it can publish its ownership
ack; the gateway's existing pre-ack failure/recovery path (the spawn site
observing the pre-ack exit) handles it.  A no-committed-generation case
(pre-PM install, a runner that owns its dependencies) is a no-op: the
interpreter's own packages are used and the worker proceeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_BOOTED = False


def _is_restart_safe_worker() -> bool:
    """True only when this process is the externally spawned restart-safe worker.

    The spawn site (``_launch_external_cron_worker``) passes
    ``--external-worker-file``; the gateway runs in-process with no such flag,
    and no other CLI invocation of ``cron/scheduler`` uses it.
    """
    return "--external-worker-file" in sys.argv


def worker_bootstrap() -> None:
    """Run PM's dependency boot in the marked external worker, in the package prelude.

    Call from the top of ``cron/__init__.py`` -- the prelude that executes
    before the first application import (``cron.jobs``) and before the
    ``cron/scheduler.py`` module body.  Gated on ``--external-worker-file`` in
    ``sys.argv``; the gateway and every unmarked importer no-op here.

    A failure in the marked worker propagates; do not call this from a context
    that swallows the error, because continuing on an ambient, unleased import
    path is precisely what this boot exists to prevent.
    """
    global _BOOTED
    if _BOOTED:
        return
    if not _is_restart_safe_worker():
        return
    _BOOTED = True
    from pm.environments import activate_dependencies
    activate_dependencies(_root)
