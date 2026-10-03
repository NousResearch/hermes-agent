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

import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional


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

#: Stamp written by a build that supports namespaces but could not resolve its own when it wrote
#: the record. Distinct from an ABSENT stamp on purpose: absence means "written by a build that
#: predates the stamp" and keeps main's behavior, while this says "written from inside a namespace
#: I could not name". Without it, a transient procfs makes the CURRENT build emit a record the
#: reader later reclassifies as legacy local authority (#123081).
PIDNS_UNRESOLVED = "unresolved"


def is_unresolved_pid_namespace(value: Any) -> bool:
    """True when ``value`` is the explicit "this build could not qualify it" stamp."""
    return value == PIDNS_UNRESOLVED


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


#: Process-local memo for a definite observation. ``functools.cache`` cannot be written to, so a
#: known answer needs an explicit home of its own. The lock covers the memo check and the probe as
#: one step, so concurrent cold callers share a single readlink. It does NOT by itself stop a
#: retain from landing mid-call: a caller that ends up unresolved re-reads the pointer before
#: returning, because a stale unknown reaches ``record_unlinkable_from`` as "cannot read a claim,
#: so unlink" and makes a KNOWN foreign record eligible for cleanup.
_local_namespace_memo: Optional[LocalPidNamespace] = None
_local_namespace_lock = threading.Lock()


def _set_local_namespace(value: LocalPidNamespace) -> None:
    """Retain a definite observation in the memo.

    Only ever called with a ``known`` value: memoizing ``_UNRESOLVED`` is precisely the bug this
    memo replaces, since it would pin the process to UNKNOWN for its whole life. The caller guards
    on ``.known`` before getting here, so an unresolved result stays absent and retryable.
    """
    global _local_namespace_memo
    with _local_namespace_lock:
        _local_namespace_memo = value


def _local_pid_namespace_cached() -> LocalPidNamespace:
    """The retained observation, or a fresh read that is retained when it is definite.

    Retention does not depend on which call observed a definite answer: a first read lands here
    and is retained, while ``local_pid_namespace``'s in-call retry retains its own value and
    returns it directly. Its unresolved branch does not come back through here -- it takes the
    memo pointer, so a failed probe never costs a third readlink.
    """
    global _local_namespace_memo
    with _local_namespace_lock:
        memo = _local_namespace_memo
        if memo is not None:
            return memo
        # Probe UNDER the lock so "check the memo, then probe" is one step: concurrent cold
        # callers make one probe between them instead of each paying their own. It does NOT stop a
        # retain from landing while this caller is still probing -- `local_pid_namespace`'s
        # unresolved branch is what catches that, by re-reading the memo before returning.
        resolved = _resolve_local_pid_namespace()
        if resolved.known:
            _local_namespace_memo = resolved
    return resolved


def _clear_local_namespace_cache() -> None:
    """Drop the retained observation. Test-only seam."""
    global _local_namespace_memo
    with _local_namespace_lock:
        _local_namespace_memo = None


def local_pid_namespace() -> LocalPidNamespace:
    """This process' PID namespace identity, cached once a definite answer was obtained.

    A failed lookup is deliberately not retained, so a transient ``/proc`` problem
    (a permissions race, a partially mounted procfs) is retried on the next call
    instead of pinning the process to "unknown" for its whole life.

    A retry that DOES learn the namespace is retained as-is. Without that, the memo stayed
    empty even after we had read a real id: the next lookup read again, failed again on a
    different transient, and dropped the process back to UNKNOWN for the rest of its life --
    contradicting the contract above and reactivating every unknown-namespace refusal
    mid-process, including the ones that refuse to signal.

    When this call's own probes fail but a concurrent thread retained the namespace while they
    were in flight, this returns THAT retained value rather than its own unresolved one. It is
    this process' namespace either way -- a PID namespace is immutable for a live process -- so
    the answer is correct, and handing back the stale unknown would be actively wrong.
    """
    resolved = _local_pid_namespace_cached()
    if resolved.known:
        return resolved
    retry = _resolve_local_pid_namespace()
    if not retry.known:
        # Unknown stays retryable, but a peer thread may have retained a definite namespace while
        # this probe was in flight. Returning our own failed read would hand that caller a stale
        # unknown, and `record_unlinkable_from` reads an unnameable own-namespace as "cannot read a
        # claim, so unlink" -- so a KNOWN foreign record becomes eligible for cleanup. Take the memo
        # POINTER, not another probe: a third readlink here would only re-discover the failure.
        # The lock here is REQUIRED, not defensive: it is the happens-before that makes a peer's
        # retained value visible on a free-threaded build. A bare read of the global passes every
        # test on GIL CPython and would drop that guarantee silently, so do not "simplify" it away.
        with _local_namespace_lock:
            memo = _local_namespace_memo
        return memo if memo is not None else retry
    # Retain the value we ALREADY obtained. Clearing a memo and re-reading /proc re-introduces
    # the very race this branch exists to close: if that extra read fails on its own transient,
    # the "promotion" stores _UNRESOLVED and pins the process to UNKNOWN for life -- and it
    # RETURNS that, so the call which just learned the id hands back "unknown" to its own
    # caller. A second probe is not a retained observation.
    _set_local_namespace(retry)
    return retry


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


def pid_checkable_from(recorded_pidns: Any) -> bool:
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

    A ``pidns`` that is not a bare ASCII digit string is treated as unstamped rather than as
    foreign. This record is written by us, so a non-canonical value is a corrupted or
    hand-edited field, and the only reader that can be sure what it meant is the writer.
    Failing closed on it would refuse to clean up a dead owner's files — and since
    ``write_pid_file`` is ``O_EXCL``, that turns one bad byte into an install that cannot
    start at all, with no CLI route back (``hermes gateway stop`` finds no live PID and
    doctor does not touch these files). Treating garbage as "no claim" keeps the boundary
    that matters (a *real* foreign stamp still refuses) and leaves the corrupt-record
    cleanup that main does. Canonicalization happens on the write side too, so nothing
    this build writes can reach the branch.

    CALLERS THAT DELETE MUST NOT USE THIS PREDICATE ON THE ``known=False`` ROW — see
    :func:`record_unlinkable_from`. Refusing to signal is right when our identity is
    unknown; refusing to delete a dead owner's identity files turns one unreadable
    ``/proc`` into an install that cannot start.
    """
    ours = local_pid_namespace()
    if not ours.supported:
        return True
    if not ours.known:
        return False
    if is_unresolved_pid_namespace(recorded_pidns):
        # The writer came from a namespace it could not name. Probing its PID here reads an
        # unrelated process, so this is "unqualified", not "legacy" — absence is the legacy case.
        return False
    if not _is_canonical_pid_namespace(recorded_pidns):
        return True
    return recorded_pidns == ours.id


def record_unlinkable_from(recorded_pidns: Any) -> bool:
    """True when this record's namespace stamp proves the identity files must stay.

    The unlink counterpart of :func:`pid_checkable_from`, and deliberately NOT the same
    predicate, because "I cannot verify this PID" and "I must not delete these files" are
    different claims with opposite costs. Failing closed is correct before a signal — an
    unknown authority is not authority, and a signal cannot be taken back. Failing closed
    before an unlink is a different trade entirely: ``write_pid_file`` is ``O_EXCL``, so
    declining to remove a dead owner's record makes every later start die on
    ``FileExistsError``, with no CLI route back. A failed ``/proc`` lookup — an unmounted
    procfs, a permissions race — must not be able to do that.

    So the unlink refuses only on a claim it can actually read: a canonical namespace id
    that is not ours. An unstamped record, a non-canonical one, or a moment when this
    process cannot name its own namespace all keep main's behavior.
    """
    ours = local_pid_namespace()
    if not ours.supported:
        return False
    if not ours.known:
        return False
    if not _is_canonical_pid_namespace(recorded_pidns):
        return False
    return recorded_pidns != ours.id


def _is_canonical_pid_namespace(value: Optional[str]) -> bool:
    """True when ``value`` is a namespace id this module could have written: ASCII digits.

    The kernel's ``/proc/<pid>/ns/pid`` symlink reads ``pid:[4026531836]``, so the id is a
    plain integer string. Anything else — a float, padded whitespace, an int that arrived
    as a JSON number, an empty string, arbitrary text — was not produced by
    :func:`local_pid_namespace` and carries no identity we can compare.

    ``isascii() and isdecimal()``, not ``isdigit()``: ``str.isdigit`` accepts non-ASCII digit
    forms (``'²'``, ``'٤٠٢٦'``, fullwidth ``'１２'``), so those would be classified CANONICAL,
    compare unequal, read as FOREIGN and reintroduce the exact wedge this tolerance removes —
    reachable only by hand-editing a record, but the predicate claims otherwise in its
    docstring and that claim has to hold.
    """
    return isinstance(value, str) and value.isascii() and value.isdecimal()


def describe_pid_namespace() -> str:
    """Short human-readable form for logs and error messages."""
    ours = local_pid_namespace()
    if not ours.supported:
        return "none (single namespace platform)"
    return ours.id or "unknown"
