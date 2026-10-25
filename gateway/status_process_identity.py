"""Process-identity readings that must be a property of the process, not of the reader (sibling of
``gateway/status``)."""

from __future__ import annotations

import sys


def raw_create_time(proc) -> float:
    """``proc.create_time()`` as the kernel recorded it, without psutil's clock-step correction.

    psutil >= 7 (``_psosx.adjust_proc_create_time``) shifts every macOS ``create_time()`` by
    ``INIT_BOOT_TIME - boot_time()`` — the difference between ``kern.boottime`` when psutil was
    imported into THIS process and ``kern.boottime`` now. ``kern.boottime`` moves with every
    ``settimeofday`` step (timed applies several per day), so the "same" process reads differently
    from two processes that imported psutil at different times, and differently within one
    long-lived process before and after a step. A liveness fingerprint built from that value is
    not a property of the process any more: the 2026-10-25 gmantra incident had the day-old
    gateway dispatcher read every worker +2 s off its own spawn record (seven live, heartbeating
    workers declared ``crashed`` in two sweeps). The kernel's ``p_starttime`` never changes after
    fork, so the unadjusted value is the stable identity. ``monotonic=True`` is the psutil knob
    that returns it on macOS (BSD too); on Linux the same keyword means seconds-since-boot, but
    Linux never reaches here (``/proc`` answered first). A psutil without the knob falls back to
    the adjusted value — the drift-tolerant comparators cover that case."""
    impl = getattr(proc, "_proc", None)
    if sys.platform == "darwin" and impl is not None and callable(getattr(impl, "create_time", None)):
        try:
            return float(impl.create_time(monotonic=True))
        except TypeError:
            pass
    return float(proc.create_time())
