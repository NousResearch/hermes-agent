"""This process' PID namespace identity.

A PID is only meaningful together with the namespace that issued it. A process
inside a PID namespace (``systemd PrivatePIDs=``, ``unshare --pid``, a container)
sees its own gateway as PID 1, and that number resolves to the host's init for
every process outside that namespace. Any "is that process still alive" check
that compares a recorded PID against ``/proc`` from a different namespace gets
a confident, wrong answer.

The identity is the ``/proc/self/ns/pid`` inode — the kernel's own answer to
"which PID namespace am I in", and what ``nsenter``/``unshare`` compare to decide
whether they are looking at the same set of processes. The number is stable for
the life of the namespace and uniquely identifies it on the machine, so two
processes may compare identities without a shared coordinate system.

Tri-state, mirroring how the platform reports other facts:

* ``supported=False`` — the platform has no namespace concept (macOS, Windows).
  There is exactly one namespace, so a PID needs no qualification.
* ``supported=True`` with an ``id`` — resolved.
* ``supported=True`` with ``id=None`` — this process is on a platform that has
  namespaces but the lookup failed. A failed lookup is NOT cached and NOT
  authority: absence of provenance cannot become provenance.
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LocalPidNamespace:
    """This process' PID namespace as a tri-state: unsupported / known / unknown."""

    id: Optional[str]
    supported: bool = True

    @property
    def known(self) -> bool:
        """True when this process can compare its own namespace against a recorded one."""
        return self.supported and self.id is not None


_NO_NAMESPACE = LocalPidNamespace(id=None, supported=False)
_UNRESOLVED = LocalPidNamespace(id=None, supported=True)


def _parse_pid_namespace_link(text: str) -> Optional[str]:
    """``pid:[4026531834]`` → ``4026531834``; anything else → ``None``."""
    text = text.strip()
    if not (text.startswith("pid:[") and text.endswith("]")):
        return None
    return text[5:-1] or None


def _resolve_local_pid_namespace() -> LocalPidNamespace:
    """Read ``/proc/self/ns/pid`` once. Never raises."""
    if not sys.platform.startswith("linux"):
        return _NO_NAMESPACE
    try:
        return LocalPidNamespace(id=_parse_pid_namespace_link(os.readlink("/proc/self/ns/pid")))
    except (OSError, ValueError):
        return _UNRESOLVED


@functools.cache
def _local_pid_namespace_cached() -> LocalPidNamespace:
    return _resolve_local_pid_namespace()


def local_pid_namespace() -> LocalPidNamespace:
    """This process' PID namespace identity, cached once a definite answer was obtained.

    A failed lookup is deliberately not cached, so a transient ``/proc`` problem
    (a permissions race, a partially mounted procfs) is retried on the next call
    instead of pinning the process to "unknown" for its whole life.
    """
    resolved = _local_pid_namespace_cached()
    if resolved.known:
        return resolved
    return _resolve_local_pid_namespace()


def pid_namespace_id(pid: int) -> Optional[str]:
    """The PID namespace of the process ``pid``, or ``None`` when it cannot be read.

    Used to qualify a PID that some *other* process recorded: a recorded
    ``(pid, pidns)`` pair only means something when the reader knows which
    namespace the number was issued in.
    """
    try:
        return _parse_pid_namespace_link(os.readlink(f"/proc/{int(pid)}/ns/pid"))
    except (OSError, ValueError, TypeError):
        return None


def pid_checkable_from(recorded_pidns: Optional[str], recorded_pid: Optional[int] = None) -> bool:
    """True when a recorded PID may be probed in THIS process' namespace.

    * No namespace on this platform → checkable: there is only one namespace, so
      the number carries its meaning on its own (#41173 keeps its hostname-only
      semantics for the same reason).
    * This process could not resolve its own identity → NOT checkable. Unknown
      authority is not authority.
    * The record names a different namespace → NOT checkable. The number it holds
      was issued there and means nothing here.
    * Everything else → checkable, including a record with no ``pidns`` at all.

    That last case is deliberate and is the rollout boundary, not an oversight: an
    unstamped record was written by a build that predates the stamp, and refusing
    to probe it would make every such record permanently unverifiable — silently
    disabling unclean-death detection (the ``state.db`` integrity check) for every
    install that had not yet restarted under this build. So an unstamped record
    keeps exactly main's behavior, and a gateway picks up namespace protection the
    moment it restarts on a build that stamps one.
    """
    ours = local_pid_namespace()
    if not ours.supported:
        return True
    if not ours.known:
        return False
    if recorded_pidns is None:
        return True
    return recorded_pidns == ours.id


def describe_pid_namespace() -> str:
    """Short human-readable form for logs and error messages."""
    ours = local_pid_namespace()
    if not ours.supported:
        return "none (single namespace platform)"
    return ours.id or "unknown"
