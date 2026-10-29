"""Shared host memory budget for Kanban workers.

One invariant, computed once, honoured by BOTH sides of the worker lifecycle:

    effective_max_workers  x  per_worker_memory_max  <=  MEMORY_BUDGET_FRACTION x MemTotal

The two sides used to disagree:

* ``tools.process_registry._worker_memory_max_bytes()`` set each worker's cgroup
  ``MemoryMax`` to ``min(slice_limit, MemTotal/2, 4 GiB)`` — it had no idea how many
  workers would run.
* ``hermes_cli.kanban_db_dispatch.derive_default_max_in_progress()`` capped
  concurrency at ``clamp(MemTotal/512, 2, 8)`` — it assumed 512 MiB per worker.

On the homelab host (2026-10-28, card ``t_8e8ec4de``) the two produced
``3 workers x 3971 MiB = 11913 MiB`` of claimed memory against ``7942 MiB`` of RAM —
a 1.5x overcommit at 3 workers and 4x at the derived cap of 8. Each scope policed
only itself, so the kernel OOM-killed whichever scope crossed the line, taking a
healthy worker with it (``constraint=CONSTRAINT_MEMCG``).

## Why the per-worker limit is PRIMARY and the cap is derived from it

The two quantities cannot be chosen independently. A worker's cgroup limit has to
be large enough that a real worker does not get OOM-killed by its own scope — the
exact failure ``t_4d12c8eb`` fixed when a mis-inherited 1 GiB limit killed workers
mid-implementation. The concurrency cap then has to be small enough that the sum
of those limits fits the host.

Measured on the homelab host (2026-10-28, ``memory.peak`` of three live worker
scopes): **1343-1375 MiB** per worker — the Hermes runtime plus a pyright language
server. The old 512 MiB planning number understated that by ~2.7x, so the derived
cap of 8 was never achievable in practice.

So the derivation order is:

1. per-worker limit  = the worker's real footprint (measured, with margin),
   bounded by the host's per-worker ceiling and by the slice's own limit;
2. concurrency cap    = ``floor(budget / per_worker_limit)`` — the most workers
   whose limits still sum inside the budget.

Deriving the cap from the limit (rather than the limit from the cap) guarantees
the invariant holds AND that the limit is never too tight to run a worker.

This module is the single source of truth both sides import, so the invariant cannot
drift again. It is deliberately dependency-free (stdlib only) so ``tools`` can import it
without a circular import back into ``hermes_cli.kanban_db_dispatch``.
"""

from __future__ import annotations

import os
from typing import Optional

# Fraction of physical RAM the concurrent worker scopes may collectively claim.
# The remaining 30% covers the gateway/control plane, the dashboard, the MCP
# gateway, the OS page cache, and every non-worker process on the host. The
# dispatcher ALSO refuses to spawn under live memory pressure, so this is a
# ceiling, not a target.
MEMORY_BUDGET_FRACTION = 0.7

# Per-worker floor: below this a worker cannot usefully run (the Hermes runtime
# plus a language server need more). A host too small to satisfy the invariant at
# this floor is a host where the operator must lower the concurrency cap explicitly.
MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024

# Absolute ceiling for a single worker's scope. A single worker should never be
# able to claim the whole host even when it is the only one running.
MAX_WORKER_MEMORY_MAX_BYTES = 4 * 1024 * 1024 * 1024

# The per-worker footprint a worker scope is sized for.
#
# 1400 MiB is the MEASURED peak of a real worker scope on the homelab host
# (2026-10-28, ``memory.peak`` = 1350995968 / 1408446464 / 386093056 bytes across
# three live scopes): the Hermes runtime plus a pyright language server, which is
# what the kernel OOM-killed at 13:01 (``task=node-MainThread``, RSS 1.4 GB).
#
# The old value was 512 MiB — a 2.7x under-estimate that made the derived cap of 8
# workers unfundable on a 7942 MiB host. Sizing the limit from the real footprint
# means the cap falls to the number the host can actually carry, instead of the
# limit being cut to a size that kills workers (t_4d12c8eb).
MEMORY_GUARD_MB_PER_WORKER = 1400

# Derived default bounds: never below 2 (the smallest VM must still progress),
# never above 8 (more fan-out must be explicit in config).
DERIVED_MAX_IN_PROGRESS_FLOOR = 2
DERIVED_MAX_IN_PROGRESS_CEILING = 8


def host_memory_total_bytes() -> Optional[int]:
    """Physical RAM in bytes, or ``None`` when it cannot be determined.

    ``None`` means "unknown" and every caller fails open (no cap / default limit)
    rather than guessing, so macOS and Windows dev machines are unaffected.
    """
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def per_worker_memory_max_bytes(
    host_bytes: Optional[int],
    max_workers: Optional[int] = None,
    *,
    mb_per_worker: int = MEMORY_GUARD_MB_PER_WORKER,
    fraction: float = MEMORY_BUDGET_FRACTION,
    slice_limit: Optional[int] = None,
    live_scopes: Optional[int] = None,
) -> Optional[int]:
    """Per-worker cgroup ``MemoryMax`` for a host with ``host_bytes`` of RAM.

    The bound is the MINIMUM of:
      * ``mb_per_worker`` — the worker's real footprint (measured), so a worker is
        never OOM-killed by its own scope (the ``t_4d12c8eb`` failure mode),
      * ``fraction x host_bytes / max_workers`` — the host-wide budget. When
        ``max_workers`` is the concurrency cap the dispatcher will honour, this
        term is what keeps the SUM of concurrent scopes inside the host's budget,
      * ``fraction x host_bytes / live_scopes`` — the SAME host-wide budget, but
        counted over EVERY scope that draws a per-worker limit, not just the
        kanban ones. Cron executions use the same scope mechanism and the same
        limit, and cron concurrency is unbounded by default, so the kanban cap
        alone does not describe the real number of scopes claiming memory.
        ``live_scopes`` is the count of scopes already on the host (plus any
        reserved headroom); ``None`` means "not measured" and this term is
        skipped, preserving the pre-existing behaviour.
      * ``fraction x host_bytes / 2`` — even ONE worker may not claim more than
        half the host's budget (a single worker must not starve the control plane),
      * ``MAX_WORKER_MEMORY_MAX_BYTES`` — no single worker may claim the host,
      * ``slice_limit`` — the ``memory.max`` of the slice the scope is created
        under (the real parent cgroup; the gateway unit's own budget is
        deliberately NOT used — see ``t_4d12c8eb``).

    ``None`` when the host RAM is unknown — the caller then falls back to its own
    default rather than inventing a number.
    """
    if host_bytes is None:
        return None
    try:
        host = int(host_bytes)
    except (TypeError, ValueError):
        return None
    if host <= 0:
        return None
    budget = int(host * float(fraction))
    candidates = [
        int(mb_per_worker) * 1024 * 1024,
        budget // 2,
        MAX_WORKER_MEMORY_MAX_BYTES,
    ]
    if max_workers is not None:
        try:
            workers = int(max_workers)
        except (TypeError, ValueError):
            workers = 0
        if workers > 0:
            candidates.append(budget // workers)
    if live_scopes is not None:
        try:
            scopes = int(live_scopes)
        except (TypeError, ValueError):
            scopes = 0
        if scopes > 0:
            candidates.append(budget // scopes)
    if slice_limit is not None and slice_limit > 0:
        # A slice limit BELOW the per-worker floor is not a usable bound (the
        # worker could never run under it) — ignore it, exactly as
        # ``tools.process_registry`` did before the budget was shared. Clamping it
        # up to the floor would silently turn "no usable bound" into "a 64 MiB
        # limit", which kills every worker.
        if slice_limit >= MIN_WORKER_MEMORY_MAX_BYTES:
            candidates.append(int(slice_limit))
    return max(MIN_WORKER_MEMORY_MAX_BYTES, min(candidates))


def effective_max_workers(
    host_bytes: Optional[int],
    *,
    per_worker_bytes: Optional[int] = None,
    mb_per_worker: int = MEMORY_GUARD_MB_PER_WORKER,
    fraction: float = MEMORY_BUDGET_FRACTION,
    floor: int = DERIVED_MAX_IN_PROGRESS_FLOOR,
    ceiling: int = DERIVED_MAX_IN_PROGRESS_CEILING,
) -> Optional[int]:
    """The most workers whose per-worker limits still sum inside the host budget.

    ``floor(budget / per_worker_limit)`` where ``budget = fraction x host_bytes``.
    When ``per_worker_bytes`` is not supplied it is computed from
    :func:`per_worker_memory_max_bytes`.

    ``None`` when the host's RAM is unknown — the caller then applies no cap at
    all (fail open), so macOS and Windows dev machines are unaffected.
    """
    if host_bytes is None:
        return None
    try:
        host = int(host_bytes)
    except (TypeError, ValueError):
        return None
    if host <= 0:
        return None
    if per_worker_bytes is None:
        per_worker_bytes = per_worker_memory_max_bytes(host, mb_per_worker=mb_per_worker)
    if not per_worker_bytes or per_worker_bytes <= 0:
        return None
    budget = int(host * float(fraction))
    workers = budget // int(per_worker_bytes)
    return max(int(floor), min(int(workers), int(ceiling)))


def budget_is_consistent(
    max_workers: Optional[int],
    per_worker_bytes: Optional[int],
    host_bytes: Optional[int],
    *,
    fraction: float = MEMORY_BUDGET_FRACTION,
) -> bool:
    """Whether ``max_workers x per_worker_bytes`` fits the host budget.

    ``True`` when any input is unknown (nothing to assert) so a fail-open host
    does not trip the invariant check.
    """
    if max_workers is None or per_worker_bytes is None or host_bytes is None:
        return True
    try:
        total = int(max_workers) * int(per_worker_bytes)
    except (TypeError, ValueError):
        return True
    return total <= int(int(host_bytes) * float(fraction))


# ---------------------------------------------------------------------------
# Live scope accounting (t_8e8ec4de follow-up, 2026-10-29)
#
# The invariant above is expressed in terms of the KANBAN concurrency cap, but
# the scopes that actually claim memory are NOT only kanban workers. Cron jobs
# are dispatched through the SAME ``hermes-worker-<suffix>`` scope mechanism
# (``cron/scheduler.py`` -> ``restart_safe_gateway_child_argv``), so every cron
# execution holds a scope with the SAME per-worker ``MemoryMax`` — and cron
# concurrency is bounded by ``cron.max_parallel_jobs``, which is unset by
# default and therefore UNBOUNDED.
#
# Measured on the homelab host (2026-10-29): a 3-minute sample saw a peak of
# 24 concurrent ``hermes-worker-*.scope`` units claiming 33600 MiB against a
# 5559 MiB budget — a 6x overcommit — while the kanban-only accounting still
# reported "3 workers x 1400 MiB = 4200 MiB, consistent". The invariant was
# TRUE and the host was still oversubscribed, because the accounting did not
# cover the scopes that were really running.
#
# So the budget must count EVERY live scope that draws a per-worker limit, not
# just the kanban subset.
# ---------------------------------------------------------------------------

# Prefix shared by every scope that draws a per-worker ``MemoryMax``.
# ``process_registry._build_systemd_scope_argv`` names them all
# ``hermes-worker-<suffix>``; the kanban dispatcher uses the ``kanban-`` suffix
# and the cron scheduler uses ``cron-``.
WORKER_SCOPE_PREFIX = "hermes-worker-"

# The cgroup slice the worker scopes are created under. Every scope that draws
# a per-worker limit lives directly under it.
WORKER_SCOPE_SLICE = "app.slice"


def live_worker_scope_count(cgroup_root: str = "/sys/fs/cgroup") -> int:
    """Number of live ``hermes-worker-*`` scopes under the worker slice.

    Counts BOTH kanban and cron scopes, because both draw the same per-worker
    ``MemoryMax`` and both compete for the same host RAM. This is the number the
    host-wide budget has to fit, NOT the kanban concurrency cap.

    ``0`` when the cgroup tree cannot be read — the caller then falls back to
    the static cap rather than inventing a count.
    """
    import glob
    import os

    try:
        # ``user.slice/user-<uid>.slice/user@<uid>.service/app.slice`` is where
        # the scopes land for a systemd user session; glob for the slice rather
        # than hardcoding the uid so the path stays correct for any user.
        patterns = (
            f"{cgroup_root}/user.slice/user-*/user@*.service/{WORKER_SCOPE_SLICE}",
            f"{cgroup_root}/{WORKER_SCOPE_SLICE}",
        )
        for pattern in patterns:
            matches = glob.glob(pattern)
            if not matches:
                continue
            total = 0
            for slice_dir in matches:
                try:
                    entries = os.listdir(slice_dir)
                except OSError:
                    continue
                for entry in entries:
                    if not entry.startswith(WORKER_SCOPE_PREFIX):
                        continue
                    if not entry.endswith(".scope"):
                        continue
                    total += 1
            return total
    except Exception:  # noqa: BLE001
        return 0
    return 0


def effective_scope_count(
    host_bytes: Optional[int],
    *,
    per_worker_bytes: Optional[int] = None,
    reserved_scopes: int = 0,
    max_workers: Optional[int] = None,
    mb_per_worker: int = MEMORY_GUARD_MB_PER_WORKER,
    fraction: float = MEMORY_BUDGET_FRACTION,
) -> Optional[int]:
    """How many per-worker scopes the host budget can actually carry.

    This is the number the budget must satisfy, and it is the MAXIMUM of:

      * the kanban concurrency cap — the dispatcher will not run more workers
        than this, and
      * the number of scopes already live on the host — cron scopes are NOT
        counted by the kanban cap, so a host running many cron jobs has more
        scopes in flight than the kanban cap allows.

    ``reserved_scopes`` is the number of scopes the host is expected to keep
    running alongside (cron headroom); it is added to the live count so a
    freshly-spawned scope does not immediately push the total over the budget.

    ``None`` when the host RAM is unknown — the caller fails open, so macOS and
    Windows dev machines are unaffected.
    """
    if host_bytes is None:
        return None
    try:
        host = int(host_bytes)
    except (TypeError, ValueError):
        return None
    if host <= 0:
        return None
    if per_worker_bytes is None:
        per_worker_bytes = per_worker_memory_max_bytes(
            host, max_workers=max_workers, mb_per_worker=mb_per_worker, fraction=fraction
        )
    if not per_worker_bytes or per_worker_bytes <= 0:
        return None

    budget = int(host * float(fraction))
    carryable = budget // int(per_worker_bytes)

    live = live_worker_scope_count()
    if live:
        carryable = max(carryable, live + max(int(reserved_scopes), 0))
    return max(1, int(carryable))


__all__ = [
    "MEMORY_BUDGET_FRACTION",
    "MEMORY_GUARD_MB_PER_WORKER",
    "MIN_WORKER_MEMORY_MAX_BYTES",
    "MAX_WORKER_MEMORY_MAX_BYTES",
    "DERIVED_MAX_IN_PROGRESS_FLOOR",
    "DERIVED_MAX_IN_PROGRESS_CEILING",
    "WORKER_SCOPE_PREFIX",
    "WORKER_SCOPE_SLICE",
    "host_memory_total_bytes",
    "effective_max_workers",
    "per_worker_memory_max_bytes",
    "budget_is_consistent",
    "live_worker_scope_count",
    "effective_scope_count",
]
