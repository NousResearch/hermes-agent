"""Which claims' ``worker_pid`` this process may probe or signal (claim-provenance owner).

A claim lock is ``<hostname>:<pid>`` (``kanban_db._claimer_id``). Every reclaim
pass that reads ``worker_pid`` as evidence — the crash sweep, the stale-claim
release, the max-runtime kill, the operator and dashboard reclaims — first has
to decide whether that PID means anything *here*: a PID recorded on another
machine is a number, not a process. That qualification used to be an inline
``startswith(host_prefix)`` repeated in each pass; it lives here so every PID
probe and every SIGTERM goes through one predicate, on the same
``<stem>_<topic>`` boundary that already produced ``kanban_db_dispatch.py``.

Leaf module: it imports nothing from the other ``kanban_db_*`` modules at load
time, so the facades import it without a cycle. ``kanban_db._host_prefix`` is
resolved lazily, keeping ``kanban_db`` the single owner of claim-lock host
identity.
"""

from __future__ import annotations

from typing import Optional


def _claim_pid_checkable(
    claim_lock: Optional[str], *, host_prefix: Optional[str] = None,
) -> bool:
    """True when ``worker_pid`` from this claim may be probed or signalled here.

    The hostname test the reclaim passes have always made: the lock's
    ``<hostname>:`` prefix must be this host's (``kanban_db._host_prefix``).

    ``host_prefix`` lets a sweep hoist ``_host_prefix()`` out of its row loop
    (one ``gethostname`` per pass, as before). Otherwise it is read at call
    time — ``kanban_db`` imports this module, so importing it at load would be
    a cycle, and the late lookup keeps ``kanban_db._host_prefix`` the patch
    point for tests.
    """
    if host_prefix is None:
        from hermes_cli.kanban_db import _host_prefix
        host_prefix = _host_prefix()
    return str(claim_lock or "").startswith(host_prefix)
