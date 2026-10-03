"""PID-namespace identity for state.db lock/lease holders.

A ``pid=<n>`` holder is relative to its writer's PID namespace; a reader in a
sibling namespace (two containers on one volume, systemd ``PrivatePIDs=``) sees
a live holder as absent and would reclaim its unexpired row. Holders are
stamped with ``pidns=<inode of /proc/self/ns/pid>`` and a foreign stamp is
never probed. Unstamped (pre-upgrade) records differ by policy:

* :func:`holder_pid_checkable` — STRICT, for TTL rows (compression locks,
  turn leases): unstamped defers to its expiry (<= TTL; a false defer
  self-heals, a false reclaim ends a live turn).
* :func:`persistent_record_pidns_checkable` — LEGACY, for no-expiry flock
  records: unstamped keeps probing so orphaned-lock cleanup still works.

Platforms without PID namespaces (macOS, Windows) keep the plain pid probe; a
failed Linux lookup is not cached and makes every holder unverifiable.
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
    if recorded is None:
        return unstamped_checkable
    if local.id is None:
        return False  # our own lookup failed: unknown authority is not authority
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
