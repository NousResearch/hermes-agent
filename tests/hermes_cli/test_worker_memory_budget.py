"""Host-wide worker memory budget: the per-worker scope limit and the
concurrency cap must not disagree (t_8e8ec4de, 2026-10-28).

The invariant under test:

    effective_max_workers x per_worker_memory_max <= 0.7 x MemTotal

Before the fix the two sides of the worker lifecycle computed unrelated numbers:

* ``tools.process_registry._worker_memory_max_bytes()`` -> ``min(slice, MemTotal/2,
  4 GiB)`` = 3971 MiB on the homelab host,
* ``hermes_cli.kanban_db_dispatch.derive_default_max_in_progress()`` ->
  ``clamp(MemTotal/512, 2, 8)`` = 8.

Three concurrent workers therefore CLAIMED 11913 MiB against 7942 MiB of RAM and
the kernel OOM-killed a healthy worker (``constraint=CONSTRAINT_MEMCG``; journal
2026-10-28 05:34 and 13:01).

This module asserts the invariant for the live host AND for synthetic hosts,
including the low-memory host the card calls out (4096 MiB, where the cap must
FALL rather than stay at 8).
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import memory_budget as mb
import tools.process_registry as pr

MIB = 1024 * 1024
GIB = 1024 * MIB


def _kib(mem_bytes: int) -> int:
    return mem_bytes // 1024


# ---------------------------------------------------------------------------
# the shared module
# ---------------------------------------------------------------------------


def test_per_worker_limit_covers_the_measured_worker_footprint():
    """The limit must be big enough for a real worker scope.

    Measured 2026-10-28 on the homelab host: three live worker scopes peaked at
    1350995968 / 1408446464 / 386093056 bytes (``memory.peak``) — the Hermes
    runtime plus a pyright language server. A limit below that re-creates the
    ``t_4d12c8eb`` failure (worker OOM-killed by its own scope mid-task).
    """
    host = 7942 * MIB
    limit = mb.per_worker_memory_max_bytes(host)
    assert limit is not None
    assert limit >= 1400 * MIB, "limit must cover the measured 1400 MiB worker peak"


def test_per_worker_limit_never_exceeds_half_the_host_budget():
    """A single worker may not claim more than half the host's budget."""
    host = 7942 * MIB
    limit = mb.per_worker_memory_max_bytes(host)
    assert limit is not None
    assert limit <= int(host * mb.MEMORY_BUDGET_FRACTION) // 2


def test_per_worker_limit_respects_a_tighter_slice():
    """The slice's own ``memory.max`` still wins over the budget term."""
    host = 7942 * MIB
    slice_limit = 512 * MIB
    limit = mb.per_worker_memory_max_bytes(host, slice_limit=slice_limit)
    assert limit == slice_limit


def test_per_worker_limit_shrinks_when_the_operator_raises_the_cap():
    """An explicit ``kanban.max_in_progress`` tightens the limit, not the cap."""
    host = 7942 * MIB
    tight = mb.per_worker_memory_max_bytes(host, max_workers=8)
    loose = mb.per_worker_memory_max_bytes(host, max_workers=3)
    assert tight is not None and loose is not None
    assert tight < loose


def test_effective_max_workers_is_derived_from_the_limit_not_a_guess():
    host = 7942 * MIB
    per_worker = mb.per_worker_memory_max_bytes(host)
    assert per_worker is not None
    cap = mb.effective_max_workers(host, per_worker_bytes=per_worker)
    assert cap is not None
    assert cap * per_worker <= int(host * mb.MEMORY_BUDGET_FRACTION)

def test_unknown_host_memory_fails_open():
    detected = mb.host_memory_total_bytes()
    assert detected is None or detected > 0
    assert mb.per_worker_memory_max_bytes(None) is None
    assert mb.effective_max_workers(None) is None
    # An unknown budget must not be reported as a violation.
    assert mb.budget_is_consistent(None, None, None) is True


def test_budget_is_consistent_detects_a_violation():
    host = 4096 * MIB
    too_many = 8
    per_worker = 1400 * MIB
    assert mb.budget_is_consistent(too_many, per_worker, host) is False
    ok = mb.effective_max_workers(host, per_worker_bytes=per_worker)
    assert ok is not None
    assert mb.budget_is_consistent(ok, per_worker, host) is True


# ---------------------------------------------------------------------------
# the dispatcher side
# ---------------------------------------------------------------------------


def test_derived_cap_falls_on_a_low_memory_host():
    """Acceptance criterion: at 4096 MiB the cap must FALL, not stay at 8.

    The old formula was ``clamp(MemTotal/512, 2, 8)``, which yields 8 on a 4096
    MiB host — 8 workers x a 3971 MiB scope limit on a 4096 MiB box.
    """
    cap = kbd.derive_default_max_in_progress({"mem_total_kib": 4 * 1024 * 1024})
    assert cap is not None
    assert cap < 8, f"4096 MiB host must not derive the ceiling cap, got {cap}"
    assert cap == 2


def test_derived_cap_on_the_live_host_shape():
    """7942 MiB host -> a cap the host can actually carry."""
    cap = kbd.derive_default_max_in_progress({"mem_total_kib": 7942 * 1024})
    assert cap is not None
    assert 2 <= cap <= 8
    # The board runs 3 workers; the host-level cap must not be below that or the
    # board's own max_spawn would be unreachable.
    assert cap >= 3


@pytest.mark.parametrize("mem_mib", [4096, 6144, 7942, 16384, 32768])
def test_derived_cap_keeps_the_invariant_on_every_host_size(mem_mib):
    """The dispatcher's cap and the scope limit must agree at every host size."""
    host = mem_mib * MIB
    cap = kbd.derive_default_max_in_progress({"mem_total_kib": mem_mib * 1024})
    assert cap is not None
    per_worker = mb.per_worker_memory_max_bytes(host, max_workers=cap)
    assert per_worker is not None
    assert cap * per_worker <= int(host * mb.MEMORY_BUDGET_FRACTION), (
        f"{mem_mib} MiB host: {cap} x {per_worker // MIB} MiB exceeds the budget"
    )


def test_derived_cap_fails_open_without_memtotal():
    assert kbd.derive_default_max_in_progress({}) is None
    assert kbd.derive_default_max_in_progress({"mem_total_kib": 0}) is None
    assert kbd.derive_default_max_in_progress({"mem_total_kib": -5}) is None
    assert kbd.derive_default_max_in_progress({"mem_total_kib": True}) is None
    assert kbd.derive_default_max_in_progress({"mem_total_kib": "1048576"}) is None


# ---------------------------------------------------------------------------
# the scope side (tools.process_registry)
# ---------------------------------------------------------------------------


def test_worker_scope_limit_agrees_with_the_dispatcher(monkeypatch):
    """``_worker_memory_max_bytes()`` and ``derive_default_max_in_progress()`` must
    satisfy the invariant together on the LIVE host.

    This is the end-to-end assertion of acceptance criterion 1: the two functions
    named in the card must agree.
    """
    host = mb.host_memory_total_bytes()
    if host is None:
        pytest.skip("host memory unknown")
    monkeypatch.setattr(
        mb, "host_memory_total_bytes", lambda: host, raising=False,
    )
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    # The dispatcher's sample comes from the gateway's ledger, which the test
    # conftest blanks; feed it the same host total so both sides see one host.
    monkeypatch.setattr(
        kbd, "_system_memory_sample",
        lambda: {"mem_total_kib": host // 1024},
        raising=False,
    )

    cap = kbd.resolve_max_in_progress(kbd.configured_max_in_progress())
    limit = pr._worker_memory_max_bytes()
    assert cap is not None
    assert mb.budget_is_consistent(cap, limit, host), (
        f"{cap} workers x {limit // MIB} MiB exceeds "
        f"{int(host * mb.MEMORY_BUDGET_FRACTION) // MIB} MiB of budget"
    )


def test_worker_scope_limit_is_not_too_tight_for_a_real_worker(monkeypatch):
    """The limit must not re-create the ``t_4d12c8eb`` 1 GiB failure.

    A worker scope peaked at 1375 MiB on the homelab host. A limit below the
    measured footprint kills workers mid-implementation with
    ``constraint=CONSTRAINT_MEMCG`` — the exact incident this card's sibling
    ``t_4d12c8eb`` fixed.
    """
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    host = mb.host_memory_total_bytes()
    if host is None:
        pytest.skip("host memory unknown")
    monkeypatch.setattr(mb, "host_memory_total_bytes", lambda: host, raising=False)
    limit = pr._worker_memory_max_bytes()
    assert limit >= 1400 * MIB, (
        f"limit {limit // MIB} MiB is below the measured worker footprint"
    )


def test_worker_scope_limit_never_exceeds_the_previous_generous_bound(monkeypatch):
    """Acceptance criterion 5: do not RAISE the scope limit.

    The pre-fix limit was ``min(slice, MemTotal/2, 4 GiB)``. The new limit may
    be lower (it must be, to fit the budget) but never higher.
    """
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    host = mb.host_memory_total_bytes()
    if host is None:
        pytest.skip("host memory unknown")
    monkeypatch.setattr(mb, "host_memory_total_bytes", lambda: host, raising=False)
    legacy_bound = min(4 * GIB, max(64 * MIB, host // 2))
    assert pr._worker_memory_max_bytes() <= legacy_bound


def test_worker_scope_limit_respects_the_slice_limit(monkeypatch):
    """The slice's own limit still constrains the scope (t_4d12c8eb)."""
    monkeypatch.setattr(
        pr, "_scope_parent_memory_max_bytes", lambda: 512 * MIB, raising=False,
    )
    assert pr._worker_memory_max_bytes() == 512 * MIB
