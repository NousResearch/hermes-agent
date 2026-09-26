"""Who owns a scoped token lock — the one qualified self-ownership predicate.

A scoped lock (one Telegram token across profiles/homes) records the PID of the
process that holds it. Inside a PID namespace that number is only meaningful to
processes sharing the namespace, so two gateways in different namespaces can
both be PID 1. Every decision that asks "is this record mine?" or "is this record
dead?" must therefore qualify the recorded PID with the namespace that issued it
before treating any local process observation as evidence about it (#123081).

Every consumer routes through :func:`scoped_lock_owned_by_self` and
:func:`scoped_lock_stale_locally` so the qualification cannot be dropped again by
a path that forgets it — the numeric-PID shortcuts that predate the stamp were
exactly where it went missing.

Unstamped records keep main's numeric-PID behavior: a record written before the
stamp existed is not retroactively foreign, and refusing it would make every
pre-upgrade lock permanently unreclaimable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from hermes_platform.host.pid_namespace import (
    describe_pid_namespace,
    local_pid_namespace,
    pid_checkable_from,
)

logger = logging.getLogger(__name__)

#: ``scoped_lock_qualification`` results.
MINE = "mine"
FOREIGN = "foreign"
UNVERIFIABLE = "unverifiable"
UNREADABLE = "unreadable"


def _record_pid(record: Any) -> Optional[int]:
    try:
        return int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return None


def scoped_lock_qualification(record: Any) -> str:
    """Classify a scoped-lock record against THIS process' PID namespace.

    * :data:`MINE` — the record names this process and its namespace is one we can
      reason about, so a local liveness observation about that number is meaningful.
    * :data:`FOREIGN` — a stamped record from another namespace, or our own
      namespace could not be resolved. Either way the number does not describe a
      process we may act on.
    * :data:`UNVERIFIABLE` — checkable namespace, but a different live process.
    * :data:`UNREADABLE` — no usable PID (malformed or legacy record); callers keep
      handling it on their pre-existing terms.
    """
    if not isinstance(record, dict):
        return UNREADABLE
    pid = _record_pid(record)
    if pid is None or pid <= 0:
        return UNREADABLE
    if not pid_checkable_from(record.get("pidns")):
        logger.warning(
            "Scoped lock pid=%s was stamped in PID namespace %s and this process is in %s; "
            "its owner cannot be identified from here.",
            pid, record.get("pidns") or "unrecorded", describe_pid_namespace(),
        )
        return FOREIGN
    return MINE if pid == os.getpid() else UNVERIFIABLE


def scoped_lock_owned_by_self(record: Any) -> bool:
    """True when this process is the lock's owner — PID equality, namespace-qualified.

    The numeric shortcut this replaces was sound only inside one PID namespace.
    A record stamped elsewhere that happens to carry our own PID belongs to a
    different process and must never be overwritten as "self" or released as ours.
    """
    return scoped_lock_qualification(record) == MINE


def scoped_lock_stale_locally(
    record: Any, is_pid_alive: Callable[[int], bool]
) -> bool:
    """True when a lock record is reclaimable, treating a foreign stamp as alive.

    ``is_pid_alive`` is the local liveness probe, and it is consulted only for a
    record whose namespace we share: a local "not found" for a number issued in
    another namespace says nothing about that namespace's owner, so it can never
    authorize reclaiming its lock. A record with no usable PID keeps main's
    behavior and is reclaimable.
    """
    qualification = scoped_lock_qualification(record)
    if qualification == UNREADABLE:
        return True
    if qualification == FOREIGN:
        return False
    pid = _record_pid(record)
    # Not MINE and not UNREADABLE, so the PID parsed. Guarded anyway so a future
    # classification cannot hand None to the liveness probe.
    return True if pid is None else not is_pid_alive(pid)
