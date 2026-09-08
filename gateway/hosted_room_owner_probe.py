"""Strict process-incarnation probe for hosted-room owner-death recovery.

Two pure functions and no state: capture the local process domain at admission, and later answer
whether a captured incarnation is ``alive``, ``ended`` or ``unknown``.

Deliberately NOT built on ``hermes_cli.active_sessions._process_start_time`` (which swallows every
exception into ``None``) or ``gateway.status._pid_exists`` (whose POSIX branch turns a generic
``OSError`` into absence). Recovery promotes an answer to a terminal transition, so an error may
never read as death: this module preserves psutil's exception types and fails closed.

**Supported-platform contract:** Linux only. The domain is ``/proc/sys/kernel/random/boot_id`` plus
``/proc/self/ns/pid``, so a database copied to another machine (or read from another PID namespace)
cannot match and its rows stay unrecoverable rather than being misjudged. Where the domain cannot
be read, no descriptor is captured and recovery is simply unavailable -- a declared capability
limit, not a fault, and never a reason to refuse ordinary work. ``host`` is a human label only.

v2 retains ``process_start`` as epoch metadata and adds positive integer ``process_start_ticks``
from Linux stat field 22 for identity. Persisted v1 descriptors are never upgraded or rewritten:
their epoch mismatch is ``unknown``, while verified absence or zombie status still proves death.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, TypeGuard

DESCRIPTOR_VERSION = 2

# Only legacy epoch equality uses tolerance. An epoch mismatch is not evidence of PID reuse:
# Linux btime (and therefore psutil create_time) can shift while the process remains alive.
_START_TIME_TOLERANCE_SECONDS = 0.001

ALIVE = "alive"
ENDED = "ended"
UNKNOWN = "unknown"

_DOMAIN_FIELDS = ("boot_id", "pid_ns", "pid", "process_start")
# What only the running process can say about the target it admitted. The durable key and the
# runtime handle are recorded separately: a successor's resume may legitimately return a different
# runtime id, so recovery compares the stored key and revalidates home/profile locally.
_NATIVE_TARGET_FIELDS = ("home", "profile", "runtime_session_id", "stored_session_key")


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _pid_namespace() -> str | None:
    try:
        return os.readlink("/proc/self/ns/pid") or None
    except OSError:
        return None


def capture_local_domain() -> dict[str, Any] | None:
    """This process's incarnation coordinates, or ``None`` when the platform cannot supply them.

    ``None`` means "recovery is unavailable here", never "something failed": the caller stores no
    descriptor and the attempt runs exactly as it does today.
    """
    boot_id, pid_ns = _boot_id(), _pid_namespace()
    if not boot_id or not pid_ns:
        return None
    pid = os.getpid()
    try:
        import psutil
        from hermes_state_common import _proc_start_ticks

        process_start = float(psutil.Process(pid).create_time())
        process_start_ticks = _proc_start_ticks(pid)
    except Exception:
        return None  # a descriptor that cannot be probed later is worse than none at all
    domain = {
        "descriptor_version": DESCRIPTOR_VERSION, "host": os.uname().nodename,
        "boot_id": boot_id, "pid_ns": pid_ns, "pid": pid, "process_start": process_start,
        "process_start_ticks": process_start_ticks}
    # Captured through the same validation the probe applies, so this module cannot mint a
    # descriptor its own verdict would later misread.
    return domain if _domain_is_current(domain) else None


def validate_native_descriptor(native: Any) -> dict[str, Any] | None:
    """The complete native half of a descriptor, or ``None`` — never a partial map.

    ``None`` means "no descriptor is stored", which costs only death-recovery eligibility. A
    half-written descriptor would be worse than none: it can neither be probed nor matched, but it
    would still look like recovery evidence.
    """
    if not isinstance(native, Mapping):
        return None
    if not _domain_is_current(native) or not isinstance(native.get("host"), str):
        return None
    validated = {field: native[field] for field in (*_DOMAIN_FIELDS, "host", "descriptor_version")}
    if native["descriptor_version"] == 2:
        validated["process_start_ticks"] = native["process_start_ticks"]
    for field in _NATIVE_TARGET_FIELDS:
        value = native.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        validated[field] = value
    return validated


def _exact_int(value: Any) -> TypeGuard[int]:
    """A real ``int``. ``bool`` is excluded explicitly: ``True == 1`` would otherwise pass."""
    return isinstance(value, int) and not isinstance(value, bool)


def _domain_is_current(descriptor: Mapping[str, Any]) -> bool:
    """Whether the descriptor was written by this host's current boot and pid namespace."""
    version = descriptor.get("descriptor_version")
    # Exact type AND value: `True == 1` and `1.0 == 1` are both true in Python, so a bare
    # comparison would admit a bool or a float as version 1.
    if not _exact_int(version) or version not in (1, 2):
        return False
    if version == 2:
        ticks = descriptor.get("process_start_ticks")
        if not _exact_int(ticks) or ticks <= 0:
            return False
    elif "process_start_ticks" in descriptor:
        return False  # v1 has epoch evidence only; never reinterpret or silently drop a stamp
    if any(descriptor.get(field) is None for field in _DOMAIN_FIELDS):
        return False
    pid, start = descriptor.get("pid"), descriptor.get("process_start")
    if not _exact_int(pid) or pid <= 0:
        return False
    # Keep epoch evidence finite and positive in both versions; v2 must not launder corrupt v1
    # metadata merely because it also carries ticks.
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        return False
    if not math.isfinite(start) or start <= 0:
        return False
    if not isinstance(descriptor.get("boot_id"), str) or not isinstance(descriptor.get("pid_ns"), str):
        return False
    return (descriptor["boot_id"], descriptor["pid_ns"]) == (_boot_id(), _pid_namespace())


def probe_owner_incarnation(descriptor: Mapping[str, Any] | None) -> str:
    """``alive`` | ``ended`` | ``unknown`` for one recorded owner incarnation.

    ``ended`` is only ever returned for a descriptor whose domain is verified to be this host's
    current one AND whose process is verifiably gone or verifiably a different incarnation. Every
    other outcome -- a legacy ``NULL``, a corrupt or foreign descriptor, a missing psutil, a denied
    or unexpected error -- is ``unknown``, which authorizes nothing.
    """
    if not isinstance(descriptor, Mapping) or not _domain_is_current(descriptor):
        return UNKNOWN
    try:
        import psutil
    except Exception:
        return UNKNOWN
    pid = int(descriptor["pid"])
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return ENDED  # reaped-but-not-collected: its execution is over
        if descriptor["descriptor_version"] == 2:
            from hermes_state_common import _proc_start_ticks

            ticks = _proc_start_ticks(pid)
            if not _exact_int(ticks) or ticks <= 0:
                return UNKNOWN  # even a missing /proc file is not a NoSuchProcess verdict
            return ALIVE if ticks == descriptor["process_start_ticks"] else ENDED
        started = float(process.create_time())
    except psutil.NoSuchProcess:
        return ENDED
    except Exception:
        # AccessDenied, a zombie racing collection, an unexpected OSError: all unreadable, and an
        # unreadable process has NOT been shown to have ended.
        return UNKNOWN
    if not math.isfinite(started) or started <= 0:
        return UNKNOWN
    if abs(started - float(descriptor["process_start"])) > _START_TIME_TOLERANCE_SECONDS:
        return UNKNOWN  # v1 cannot distinguish PID reuse from wall-clock metadata drift
    return ALIVE
