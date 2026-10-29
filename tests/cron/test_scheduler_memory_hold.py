"""Cron's turn in the gateway's memory-aware admission (cron/scheduler_memory_hold.py)."""

from __future__ import annotations

from gateway import admission_memory as am
from hermes_platform.memory import MemoryReading

MB = 1024 * 1024


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _gate(free_mb, **settings):
    clock = _Clock()
    gate = am.MemoryGate(
        settings=am.MemoryGuardSettings(**{"critical_mb": 256, "low_mb": 512, "reserve_mb": 0, **settings}),
        reader=lambda: MemoryReading(free_mb * MB, 4096 * MB, "cgroup"),
        clock=clock,
    )
    return gate, clock


def test_cron_run_is_recorded_failed_when_memory_never_frees(monkeypatch):
    from cron.scheduler_memory_hold import hold_for_memory

    gate, clock = _gate(10, max_wait_seconds=4)
    monkeypatch.setattr(am, "_gate", gate)
    monkeypatch.setattr(am.time, "sleep", lambda _: setattr(clock, "now", clock.now + 2))  # cron thread's sleep
    held = hold_for_memory({"id": "j1", "name": "nightly"}, "j1", "nightly", None)
    assert held is not None and held[0] is False and "memory" in held[3]


def test_a_script_only_cron_job_waits_before_its_script_runs(monkeypatch):
    from cron import scheduler

    gate, clock = _gate(10, max_wait_seconds=2)
    monkeypatch.setattr(am, "_gate", gate)
    monkeypatch.setattr(am.time, "sleep", lambda _: setattr(clock, "now", clock.now + 2))
    ran = []
    monkeypatch.setattr(scheduler, "_prepare_job_prompt", lambda *a, **k: ran.append(a) or ((True, "", "", None), None))
    ok, _, _, error = scheduler.run_job({"id": "j1", "name": "heavy", "no_agent": True, "script": "x.py"})
    assert ran == [], "the job's script ran before the memory hold"
    assert ok is False and "memory" in error
