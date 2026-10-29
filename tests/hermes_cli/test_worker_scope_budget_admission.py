"""The host-wide scope budget must count EVERY scope, not just kanban workers
(t_8e8ec4de follow-up, 2026-10-29).

The sibling fix (commit ``dd03bf5a8b``) made the per-worker scope limit and the
kanban concurrency cap agree on ONE budget::

    effective_max_workers x per_worker_memory_max <= 0.7 x MemTotal

That invariant was TRUE on the live host, and the host was still oversubscribed.

Why: the scopes that actually claim memory are not only kanban workers. Cron
executions are dispatched through the SAME scope mechanism
(``cron/scheduler.py`` -> ``restart_safe_gateway_child_argv`` ->
``_build_systemd_scope_argv``), so every cron execution holds a scope with the
SAME per-worker ``MemoryMax``. Cron concurrency is bounded by
``cron.max_parallel_jobs``, which is unset by default and therefore UNBOUNDED.

Measured on the homelab host (2026-10-29), a 3-minute sample of the live
cgroup tree::

    peak 24 concurrent hermes-worker-*.scope  ->  33600 MiB claimed
    budget (0.7 x 7942 MiB)                   ->   5559 MiB
    kanban-only accounting said                ->   3 x 1400 = 4200 MiB  "consistent"

So the invariant was satisfied AND the host was 6x overcommitted, because the
accounting did not cover the scopes that were really running.

These tests cover the fix: the budget is now an ADMISSION check over the LIVE
scope count, and the per-worker limit is NOT shrunk (shrinking it would not
reduce the claim — existing scopes keep the limit they were created with — while
it WOULD starve new workers, re-creating the ``t_4d12c8eb`` failure).
"""

from __future__ import annotations

import pytest

from hermes_cli import memory_budget as mb
import tools.process_registry as pr

MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# live scope counting
# ---------------------------------------------------------------------------


def test_live_scope_count_counts_kanban_AND_cron_scopes(tmp_path):
    """Both scope families draw a per-worker limit, so both must be counted.

    The kanban dispatcher uses the ``kanban-`` unit suffix and the cron
    scheduler uses ``cron-``; ``process_registry`` prefixes both with
    ``hermes-worker-``. A count that only saw kanban scopes would keep
    reporting the host as comfortable while cron piled up.
    """
    slice_dir = tmp_path / "user.slice" / "user-1000.slice" / "user@1000.service" / "app.slice"
    slice_dir.mkdir(parents=True)
    for name in (
        "hermes-worker-kanban-t_abc-run-1.scope",
        "hermes-worker-kanban-t_def-run-2.scope",
        "hermes-worker-cron-70ad14397f81-exec-abc.scope",
        "hermes-worker-cron-f6547e7a6a23-exec-def.scope",
    ):
        (slice_dir / name).mkdir()
    # A NON-worker scope must not be counted: it draws no per-worker limit.
    (slice_dir / "hermes-gateway.service").mkdir()
    (slice_dir / "some-other.scope").mkdir()

    assert mb.live_worker_scope_count(cgroup_root=str(tmp_path)) == 4


def test_live_scope_count_is_zero_when_the_tree_cannot_be_read(tmp_path):
    """An unreadable cgroup tree yields 0, not a crash and not a guess."""
    assert mb.live_worker_scope_count(cgroup_root=str(tmp_path / "nope")) == 0


def test_live_scope_count_ignores_non_scope_entries(tmp_path):
    """Only ``*.scope`` units count — a stray file must not inflate the count."""
    slice_dir = tmp_path / "app.slice"
    slice_dir.mkdir(parents=True)
    (slice_dir / "hermes-worker-kanban-t_abc-run-1.scope").mkdir()
    (slice_dir / "hermes-worker-kanban-t_abc-run-1.scope.d").mkdir()
    (slice_dir / "hermes-worker-notes.txt").write_text("x")

    assert mb.live_worker_scope_count(cgroup_root=str(tmp_path)) == 1


def _fake_sysconf(host_bytes: int):
    """A ``os.sysconf`` stand-in that reports a host with ``host_bytes`` of RAM.

    ``SC_PHYS_PAGES`` and ``SC_PAGE_SIZE`` are DIFFERENT keys — a single-value
    mock would multiply the page count by the page size twice and produce a
    nonsensical total.
    """
    page_size = 4096
    pages = host_bytes // page_size

    def _sysconf(name):
        if name == "SC_PHYS_PAGES":
            return pages
        if name == "SC_PAGE_SIZE":
            return page_size
        raise ValueError(f"unexpected sysconf key {name!r}")

    return _sysconf


# ---------------------------------------------------------------------------
# the admission check
# ---------------------------------------------------------------------------


def test_budget_is_not_exhausted_while_scopes_fit(monkeypatch):
    """Below the carryable count, another scope is affordable."""
    monkeypatch.setattr(pr.os, "sysconf", _fake_sysconf(7942 * MIB), raising=False)
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    monkeypatch.setattr(pr, "_live_scope_count_for_budget", lambda: 3, raising=False)

    assert pr.worker_scope_budget_exhausted() is False


def test_budget_is_exhausted_when_another_scope_would_overcommit(monkeypatch):
    """At the carryable count, creating another scope must be refused.

    5559 MiB of budget / 1400 MiB per scope = 3 carryable scopes. A FOURTH
    scope would claim 5600 MiB — past the budget — which is the 6x overcommit
    the live host was actually running, just at smaller scale.
    """
    monkeypatch.setattr(pr.os, "sysconf", _fake_sysconf(7942 * MIB), raising=False)
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    monkeypatch.setattr(pr, "_live_scope_count_for_budget", lambda: 4, raising=False)

    assert pr.worker_scope_budget_exhausted() is True


def test_budget_check_fails_open_when_ram_is_unknown(monkeypatch):
    """Unknown RAM must not refuse every scope (macOS/Windows dev hosts)."""

    def _boom(_name):
        raise OSError("no sysconf")

    monkeypatch.setattr(pr.os, "sysconf", _boom, raising=False)
    monkeypatch.setattr(pr, "_live_scope_count_for_budget", lambda: 99, raising=False)

    assert pr.worker_scope_budget_exhausted() is False


def test_budget_check_never_refuses_the_very_first_scope(monkeypatch):
    """A host too small for two scopes must still run ONE.

    The floor of 1 keeps a small host progressing instead of deadlocking.
    """
    monkeypatch.setattr(pr.os, "sysconf", _fake_sysconf(256 * MIB), raising=False)
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    monkeypatch.setattr(pr, "_live_scope_count_for_budget", lambda: 1, raising=False)

    # carryable == max(1, budget // per_worker) == 1, and 1 > 1 is False.
    assert pr.worker_scope_budget_exhausted() is False


# ---------------------------------------------------------------------------
# the limit must NOT be shrunk to compensate for live scopes
# ---------------------------------------------------------------------------


def test_per_worker_limit_is_not_shrunk_by_live_scope_count(monkeypatch):
    """The fix bounds the SCOPE COUNT, not the per-scope size.

    Shrinking the new scope's limit would not reduce the collective claim —
    existing scopes keep the limit they were created with — while it WOULD
    starve the new worker and re-create the ``t_4d12c8eb`` failure (a healthy
    worker OOM-killed by its own scope mid-task).
    """
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    host = 7942 * MIB
    monkeypatch.setattr(pr.os, "sysconf", _fake_sysconf(host), raising=False)
    monkeypatch.setattr(pr, "_live_scope_count_for_budget", lambda: 24, raising=False)

    assert pr._worker_memory_max_bytes() >= 1400 * MIB


def test_live_scopes_term_shrinks_the_limit_only_when_explicitly_requested(monkeypatch):
    """The ``live_scopes`` term exists but is opt-in, and is never applied by
    ``_worker_memory_max_bytes`` — the admission check owns that concern."""
    host = 7942 * MIB
    default = mb.per_worker_memory_max_bytes(host, max_workers=3)
    with_term = mb.per_worker_memory_max_bytes(host, max_workers=3, live_scopes=24)

    assert default is not None and default >= 1400 * MIB
    assert with_term is not None and with_term < default


# ---------------------------------------------------------------------------
# the invariant, stated over the scopes that actually run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mem_total_mib",
    [2048, 4096, 6144, 7942, 16384],
)
def test_cap_falls_on_a_small_host(mem_total_mib):
    """Acceptance criterion 2: on a low-memory host the cap must FALL.

    The pre-fix code yielded 8 workers at every size, including 4096 MiB.
    """
    host = mem_total_mib * MIB
    per_worker = mb.per_worker_memory_max_bytes(host)
    cap = mb.effective_max_workers(host, per_worker_bytes=per_worker)

    assert cap is not None
    assert cap <= 8
    assert mb.budget_is_consistent(cap, per_worker, host)


def test_effective_scope_count_covers_cron_scopes(monkeypatch):
    """The scope budget must count cron scopes, not just the kanban cap.

    With 12 cron scopes already live, the number of scopes the host can carry
    is at least 12 — the kanban cap of 3 alone would say "plenty of room" while
    the host was running 12 extra scopes.
    """
    monkeypatch.setattr(mb, "live_worker_scope_count", lambda: 12, raising=False)
    host = 7942 * MIB
    per_worker = mb.per_worker_memory_max_bytes(host, max_workers=3)

    count = mb.effective_scope_count(host, per_worker_bytes=per_worker)
    assert count is not None
    assert count >= 12


# ---------------------------------------------------------------------------
# the dispatch path: no scope is CREATED when the budget is exhausted
# ---------------------------------------------------------------------------


def _as_managed_gateway(monkeypatch):
    """Put ``process_registry`` on the managed-gateway scope path.

    The real dispatch path returns ``in_process`` for a non-gateway caller, so
    the budget check is only reachable when the gateway's own preconditions
    hold: ``INVOCATION_ID`` set, a supervised gateway process, and a usable
    ``systemd-run --user --scope``.
    """
    monkeypatch.setenv("INVOCATION_ID", "test-invocation")
    monkeypatch.setattr(pr, "_is_supervised_gateway_process", lambda: True, raising=False)
    monkeypatch.setattr(pr, "_systemd_run_user_scope_available", lambda: True, raising=False)


def test_dispatch_refuses_to_create_a_scope_when_the_budget_is_exhausted(monkeypatch):
    """No scope is created once the host has no room for another one."""
    _as_managed_gateway(monkeypatch)
    monkeypatch.setattr(pr.os, "sysconf", _fake_sysconf(7942 * MIB), raising=False)
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    monkeypatch.setattr(pr, "_live_scope_count_for_budget", lambda: 4, raising=False)

    with pytest.raises(pr.WorkerScopeBudgetExhausted):
        pr.restart_safe_gateway_child_argv(
            ["/bin/true"],
            unit_suffix="probe-budget-test",
            require_restart_safe_scope=True,
        )


def test_dispatch_still_creates_a_scope_when_there_is_room(monkeypatch):
    """The check must not block normal operation."""
    _as_managed_gateway(monkeypatch)
    monkeypatch.setattr(pr.os, "sysconf", _fake_sysconf(7942 * MIB), raising=False)
    monkeypatch.setattr(pr, "_scope_parent_memory_max_bytes", lambda: None, raising=False)
    monkeypatch.setattr(pr, "_live_scope_count_for_budget", lambda: 2, raising=False)

    dispatch = pr.restart_safe_gateway_child_argv(
        ["/bin/true"],
        unit_suffix="probe-budget-test",
        require_restart_safe_scope=True,
    )
    assert dispatch.mode == "scoped"
    assert "systemd-run" in dispatch.argv[0]


def test_the_budget_exception_is_an_infrastructure_failure_for_the_dispatcher():
    """The kanban dispatcher must treat it like the unreachable-bus case.

    ``RestartSafeScopeUnavailable`` is already handled as an infrastructure
    failure that does NOT spend the card's retry budget (#114720). The budget
    refusal is the same kind of condition — the host is momentarily out of
    room, the card is fine — so it must be classified the same way.
    """
    from hermes_cli import kanban_db_dispatch as kbd

    assert issubclass(pr.WorkerScopeBudgetExhausted, RuntimeError)
    assert not issubclass(pr.WorkerScopeBudgetExhausted, pr.RestartSafeScopeUnavailable)

    # Both host-side refusals are infrastructure failures.
    assert kbd._is_infrastructure_spawn_failure(
        pr.WorkerScopeBudgetExhausted("no room on this host")
    ) is True
    assert kbd._is_infrastructure_spawn_failure(
        pr.RestartSafeScopeUnavailable("no user bus")
    ) is True
    # A genuine card failure is NOT.
    assert kbd._is_infrastructure_spawn_failure(RuntimeError("card is broken")) is False
