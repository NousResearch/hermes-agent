"""PID-namespace identity for kanban worker claims (the claim-provenance owner).

A claim lock is ``<hostname>:<pid>`` (``kanban_db._claimer_id``) and the reclaim
passes historically used the hostname alone to decide that a PID they could not
see was a dead worker. Hostname and PID namespace are independent in
containers: Compose services joined with ``network_mode: "service:x"`` share
one hostname, and unless ``pid:`` is set too each keeps its own PID namespace,
where the same PID number is an unrelated process. ``/proc/<pid>`` only answers
for the caller's namespace, so "no such PID" is not evidence about a worker in
a container we cannot see into.

``tasks.claim_pidns`` records the namespace a claim was made in
(``kanban_db._claim_and_open_run``) and leaves with the lock on release. This
module owns the identity resolution (``_local_pid_namespace``) and the guard
every PID probe and every SIGTERM goes through (``_claim_pid_checkable``). It
is a leaf: it imports nothing from the other ``kanban_db_*`` modules at load
time, so those facades import it without a cycle.
"""

from __future__ import annotations

import os
import re
import sys
from typing import NamedTuple, Optional


class LocalPidNamespace(NamedTuple):
    """What this process knows about its own PID namespace.

    Three states, and :func:`_claim_pid_checkable` treats each differently:
    ``supported`` False — the platform has no namespace identity at all (no
    ``/proc/self/ns/pid``: macOS, Windows); ``supported`` True with an ``id``
    — Linux, resolved; ``supported`` True with ``id None`` — Linux, but the
    lookup failed (restricted or unmounted ``/proc``). The last is unknown
    authority, not the compatibility case: absence of our own provenance
    cannot become provenance because the read failed.
    """

    id: Optional[str]
    supported: bool


_UNSUPPORTED = LocalPidNamespace(None, False)

# Cached once there is a definite answer: a process cannot move itself into
# another PID namespace (``setns`` applies to its future children), so a
# successful read never changes for us. A failed read is NOT cached — it may
# be transient, and while it lasts every host-local claim is unverifiable.
_LOCAL_PID_NS: Optional[LocalPidNamespace] = None


def _parse_pid_namespace_link(link: str) -> Optional[str]:
    """``"pid:[4026532534]"`` -> ``"4026532534"``; None when unparseable.

    Split out from the resolver so the parsing can be tested as the pure
    string -> string mapping it is, on any host.
    """
    match = re.search(r"\[(\d+)\]", link or "")
    return match.group(1) if match else None


def _resolve_local_pid_namespace() -> LocalPidNamespace:
    if sys.platform != "linux":
        return _UNSUPPORTED
    try:
        link = os.readlink("/proc/self/ns/pid")
    except OSError:
        return LocalPidNamespace(None, True)
    return LocalPidNamespace(_parse_pid_namespace_link(link), True)


def _local_pid_namespace() -> LocalPidNamespace:
    """This process' PID-namespace identity (see :class:`LocalPidNamespace`)."""
    global _LOCAL_PID_NS
    if _LOCAL_PID_NS is None:
        local = _resolve_local_pid_namespace()
        if local.id is not None or not local.supported:
            _LOCAL_PID_NS = local
        return local
    return _LOCAL_PID_NS


def _pid_namespace_id() -> Optional[str]:
    """What a claim records: the ``/proc/self/ns/pid`` inode on Linux, ``None``
    where there is no namespace identity or the lookup failed. Stored as NULL
    either way — a claim never asserts a namespace it cannot prove."""
    return _local_pid_namespace().id


def _claim_pid_checkable(
    claim_lock: Optional[str], claim_pidns: Optional[str], *,
    host_prefix: Optional[str] = None,
) -> bool:
    """True when ``worker_pid`` from this claim means anything to *this* process.

    A PID is checkable only when the lock's host prefix is ours AND the
    claim's PID namespace is provably ours. The matrix:

    * host prefix differs -> False (as always);
    * this platform has no namespace identity (macOS/Windows) -> True: the
      hostname is the only evidence there is, unchanged from before this
      column existed, and an unknown claim namespace stays harmless there;
    * this platform has one but our own lookup failed -> False: unknown
      authority, see :class:`LocalPidNamespace`;
    * the claim recorded no namespace (pre-upgrade row, or a claim written by
      an older binary that never stamps it) -> False: an older writer sharing
      our hostname is indistinguishable from a sibling container sharing it,
      so hostname is not authority across the mixed-version boundary either;
    * both known -> equality.

    Every False leaves the claim to its TTL: never probed, never signalled.

    ``host_prefix`` lets a sweep hoist ``kanban_db._host_prefix`` out of its
    row loop (one ``gethostname`` per pass). Otherwise it is read at call time
    — ``kanban_db`` imports this module, so importing it at load would be a
    cycle, and the late lookup keeps ``kanban_db._host_prefix`` the patch
    point for tests.
    """
    if host_prefix is None:
        from hermes_cli.kanban_db import _host_prefix
        host_prefix = _host_prefix()
    if not str(claim_lock or "").startswith(host_prefix):
        return False
    local = _local_pid_namespace()
    if not local.supported:
        return True
    if local.id is None or claim_pidns is None:
        return False
    return str(claim_pidns) == str(local.id)
