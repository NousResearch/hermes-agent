"""PID-namespace identity for state.db lock/lease holders (the probe-provenance owner).

A ``pid=<n>`` lock holder (compression lock, session turn lease) is written by the
process that owns it, relative to THAT process's PID namespace.  Readers sharing
one state.db from a different namespace — two containers on one volume, systemd
``PrivatePIDs=``, any container-per-process layout — see a disjoint PID set, so
a sibling's live ``pid=`` reads as absent and the namespace-blind probe (the
historic ``psutil.pid_exists(pid)``) declares a live holder dead and reclaims
its unexpired row.  For a session turn lease that ends the sibling's in-flight
turn ("another Hermes process took over this session"); for a compression lock
it splits the compression lineage.  The same class applies to the flock holder
records in ``hermes_state_common`` (``pid=`` recorded by a holder that forks
then dies).

This module owns the identity resolution and the qualification every structured
holder probe goes through.  It is a leaf: it imports nothing from the
``hermes_state*`` modules at load time, so ``hermes_state`` and
``hermes_state_common`` can import it without a cycle.  The identity is the
``/proc/self/ns/pid`` inode — the kernel's own answer to "which PID namespace am
I in" — the same identity the namespace-relative work for gateway records and
scoped locks (issue #123081) resolves for its records.

Two predicates, one qualification rule, two rollouts.  A record STAMPED with a
namespace is probed only where the stamp is ours; a foreign stamp is never
probed (the reader's absence reading says nothing about a process in another
namespace).  What differs between the predicates is what an UNSTAMPED record —
written by a build that predates the stamp — defaults to:

* :func:`holder_pid_checkable` — STRICT, for TTL-bounded rows (compression
  locks, session turn leases).  Unstamped holders are not probed; they defer to
  their own expiry — at most the remaining TTL (300 s), and a false defer
  self-heals there.  The asymmetry is deliberate (probe doubt already defers,
  for the same reason): a false reclaim ends a live sibling's turn and cannot
  be undone, while a false defer costs a bounded wait.  Strictness is also what
  makes the fix hold during a mixed-version rollout — the stealer side stops
  stealing as soon as the READER restarts, no writer restart required.

* :func:`persistent_record_pidns_checkable` — LEGACY ROLLOUT, for records with
  no expiry (the flock holder records).  An unstamped record keeps main's
  behavior: those records never expire, so refusing to probe an unstamped one
  would make it permanently unverifiable and silently disable orphaned-lock
  cleanup on every install that had not yet restarted — the rollout boundary
  the namespace-relative-identity work documents for gateway records.
  Protection begins as writers restart and stamp; a foreign stamp is still
  never probed.

Tri-state, mirroring how the platform reports other facts:

* ``supported=False`` — the platform has no PID-namespace concept (macOS,
  Windows).  There is exactly one namespace, so a numeric PID is still
  evidence and every predicate returns True (main's semantics).
* ``supported=True`` with an ``id`` — resolved.
* ``supported=True`` with an ``id`` of None — Linux, but the lookup failed
  (restricted or unmounted ``/proc``).  A failed lookup is NOT cached and NOT
  authority: absence of provenance cannot become provenance, so every
  structured holder is unverifiable and defers.
"""

from __future__ import annotations

import os
import re
import sys
from typing import NamedTuple, Optional

# ``pidns=`` token inside a structured holder string (``pid=123:pidns=4026533184:turn=…``).
_PIDNS_TOKEN_RE = re.compile(r"(?:^|:)pidns=(\d+)(?::|$)")


class LocalPidNamespace(NamedTuple):
    """What this process knows about its own PID namespace.

    Three states, and the ``checkable`` predicates treat each differently:
    ``supported`` False — the platform has no namespace identity at all;
    ``supported`` True with an ``id`` — resolved; ``supported`` True with an
    ``id`` of None — resolution failed, see the module docstring.
    """

    id: Optional[str]
    supported: bool


_UNSUPPORTED = LocalPidNamespace(None, False)

# Cached once there is a definite answer: a process cannot move itself into
# another PID namespace (``setns`` applies to its future children), so a
# successful read never changes for us.  A failed read is NOT cached — it may
# be transient, and while it lasts every structured holder is unverifiable.
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


def pid_namespace_id() -> Optional[str]:
    """What a holder records: the ``/proc/self/ns/pid`` inode on Linux, ``None``
    where there is no namespace identity or the lookup failed.  A holder never
    asserts a namespace it cannot prove."""
    return _local_pid_namespace().id


def holder_namespace_token() -> str:
    """The ``:pidns=<id>`` fragment to append to a new holder string, or ``""``.

    Callers build ``f"pid={os.getpid()}{holder_namespace_token()}:turn=…"``.
    """
    ns = pid_namespace_id()
    return f":pidns={ns}" if ns else ""


def recorded_namespace(holder: str) -> Optional[str]:
    """The ``pidns=`` stamp from a holder string, or None when it carries none."""
    match = _PIDNS_TOKEN_RE.search(holder or "")
    return match.group(1) if match else None


def _qualify(recorded: Optional[str], *, unstamped_checkable: bool) -> bool:
    """The shared matrix; *unstamped_checkable* is the policy for a record that
    carries no namespace (see the module docstring for why there are two)."""
    local = _local_pid_namespace()
    if not local.supported:
        return True  # single-namespace platform: a pid is evidence on its own
    if local.id is None:
        return False  # our own lookup failed: unknown authority is not authority
    if recorded is None:
        return unstamped_checkable
    return str(recorded) == str(local.id)


def holder_pid_checkable(holder: str) -> bool:
    """STRICT: may a structured holder string's ``pid=`` be probed here?

    The policy for TTL-bounded rows (compression locks, session turn leases):
    a foreign-namespace or unstamped holder is never probed — it defers to the
    row's own expiry instead (at most the remaining TTL; a false defer
    self-heals, a false reclaim ends a live turn).
    """
    return _qualify(recorded_namespace(holder), unstamped_checkable=False)


def persistent_record_pidns_checkable(recorded: Optional[str]) -> bool:
    """LEGACY ROLLOUT: may a persistent record (no expiry) be probed here?

    The policy for the flock holder records: a foreign-namespace record is
    never probed, while an unstamped one keeps main's behavior — those records
    never expire, so refusing to probe them would permanently disable
    orphaned-lock cleanup (see the module docstring).
    """
    return _qualify(recorded, unstamped_checkable=True)
