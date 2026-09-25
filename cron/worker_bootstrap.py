"""Cron external worker: lifetime protection and dependency activation at spawn.

The restart-safe external worker (``sys.executable -m cron.scheduler``) is a fresh
process.  Its own ``sys.path`` is the store Python's, which owns no third-party
dependencies; the gateway's dependency generation is not on that path.

``pin_hermes_tree_on_pythonpath`` (``cron/scheduler_worker_env.py``) restores the
site-packages directory into the child's ``PYTHONPATH`` so the worker can import
Hermes dependencies.  That directory alone, however, is not a retention
permit: the PM collector removes an unselected, lease-managed generation once it
is old enough and no process holds its lease.  The gateway holds a lease on the
generation it is running; the freshly spawned worker does not, so between the
gateway's exit and the worker's next import the collector may delete the very
generation the worker needs (``#122290`` P2, ``andrexibiza`` review).

This module closes that gap at the worker entry, *before* the first
application/dependency import:

* ``lease_dependencies()`` acquires the generation's kernel lease (the same
  primitive ``pm.environments.activate_dependencies`` uses) so the collector
  cannot remove it while the worker imports or runs.
* ``activate_worker_dependencies()`` puts the committed generation's
  ``site-packages`` on this process's ``sys.path`` (``site.addsitedir``), so
  dependency imports resolve without waiting for the gateway's handoff.

Both are no-ops (or degrade to a warning) when no dependency generation is
committed — a runner that owns its dependencies (wheel / pipx / test /
developer venv) needs nothing added.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parent.parent
_LEASE_ACQUIRED = False
_ACTIVATED = False


def _activate_generation() -> Path | None:
    """Locate the committed dependency venv, or ``None`` when there is none.

    Uses the same resolution ``pm.environments.activate_dependencies`` uses
    (``committed_venv`` — never the in-tree ``venv``/``.venv``).  Returns the
    *venv* directory (the one containing ``pyvenv.cfg``) that
    ``lease_generation`` expects; the function's ``.parent`` is the
    lease-managed generation.
    Stdlib-only reads: a missing facts file means "nothing committed", not an
    error.
    """
    try:
        from pm.environments import committed_venv
    except Exception:
        # pm is not yet importable (pre-PM tree / sealed payload); nothing to lease.
        return None
    try:
        venv = committed_venv(_root)
    except Exception:
        # Corrupt facts file: degrade to a warning, never brick the worker.
        return None
    if venv is None:
        return None
    # A committed venv must carry the marker; a directory that merely happens to
    # be named like one is not a retention target.
    if not (venv / "pyvenv.cfg").is_file():
        return None
    return venv


def lease_dependencies() -> None:
    """Hold the committed dependency generation's kernel lease for this worker's lifetime.

    Must be called *before* any application or dependency import, because the
    collector's safety condition is "no process holds the lease"; a late lease
    only protects imports that happen after it.  ``lease_generation`` takes the
    committed venv path and leases its parent (the generation directory), so a
    generation that is no longer the selected one still survives the collector
    while the worker holds the lease.  The lease is released by ``atexit``
    (``hermes_cli.runtime_state.lease_directory``), so an early crash still
    releases it on exit.  Idempotent within a process.
    """
    global _LEASE_ACQUIRED
    if _LEASE_ACQUIRED:
        return
    venv = _activate_generation()
    if venv is None:
        return
    try:
        from hermes_cli.runtime_state import lease_generation
        lease_generation(venv)
        _LEASE_ACQUIRED = True
    except Exception as exc:  # pragma: no cover - degradation path
        logger.warning("cron worker could not lease dependency generation: %s", exc)


def activate_worker_dependencies() -> None:
    """Put the committed generation's ``site-packages`` on this process's path.

    Delegates to ``pm.environments.activate_dependencies`` — the same
    resolution, ``site.addsitedir`` and ``sys.path`` reordering the gateway's
    boot uses — so the worker imports exactly what the gateway was running.
    No-op when there is no committed generation.  Idempotent within a process.
    """
    global _ACTIVATED
    if _ACTIVATED:
        return
    try:
        from pm.environments import activate_dependencies
        activate_dependencies(_root)
        _ACTIVATED = True
    except Exception as exc:  # pragma: no cover - degradation path
        logger.warning("cron worker dependency activation failed: %s", exc)


def worker_bootstrap() -> None:
    """Entry-point hook: lease then activate, in that order.

    Leasing first guarantees the generation survives even if activation is
    later skipped; activating second puts ``site-packages`` on the path so the
    first dependency import succeeds.  Call this at the very top of
    ``cron/scheduler.py`` (after the stdlib-only ``sys.path`` pin, before any
    Hermes package import).

    Activation is gated on the process being a *restart-safe external worker*
    (one spawned as ``python -m cron.scheduler --external-worker-file ...``):
    such a worker is a fresh interpreter whose own ``sys.path`` is the store
    Python's and carries no dependency packages, so it must activate.  A
    gateway process already runs on its activated dependency graph (booted
    through ``hermes_bootstrap`` → ``activate_dependencies``); re-activating
    there is a harmless no-op but is skipped so the gateway keeps its exact
    launch contract.  Leasing is unconditional: it is idempotent and protects
    the generation the worker (or the gateway) will use regardless.
    """
    lease_dependencies()
    if _is_restart_safe_worker():
        activate_worker_dependencies()


def _is_restart_safe_worker() -> bool:
    """True only when this process is the externally spawned restart-safe worker.

    The spawn site (``_launch_external_cron_worker``) passes
    ``--external-worker-file``; the gateway runs in-process with no such flag,
    and other CLI invocations of ``cron/scheduler`` do not use it either.
    """
    return "--external-worker-file" in sys.argv
