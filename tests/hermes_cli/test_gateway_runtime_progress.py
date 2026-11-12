"""A default ensure waits out a slow cold start whose owner keeps burning CPU, never a stalled one."""
import pytest

from hermes_agent.hermes_cli import gateway_runtime_progress as progress


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(progress.time, "monotonic", clock)
    monkeypatch.setattr(progress, "_starting_owner_pid", lambda homes, child: 4242)
    return clock


def _run(clock, monkeypatch, runway, cpu, *, until=1000, step=1.0):
    """Drive ``extend`` once per ``step`` until the clock reaches ``until`` or the deadline passes.
    ``cpu(t)`` is the owner's CPU seconds at elapsed ``t``. Returns the elapsed time it gave up at."""
    start = clock.now
    monkeypatch.setattr(progress, "_cpu_seconds", lambda pid: cpu(clock.now - start))
    while clock.now - start < until:
        deadline = runway.extend(("home",), None)
        if clock.now >= deadline:
            return clock.now - start
        clock.now += step
    return None


def test_progressing_owner_outlives_the_base_deadline_until_the_cap(clock, monkeypatch):
    runway = progress.StartRunway(None, 60.0)
    # A boot at a 5% quota: 0.05 s of CPU per wall second, still not READY.
    gave_up = _run(clock, monkeypatch, runway, lambda t: 0.05 * t)
    assert gave_up == pytest.approx(progress.MAX_ADAPTIVE_TIMEOUT, abs=1.0)


def test_owner_with_no_cpu_progress_fails_at_the_base_deadline(clock, monkeypatch):
    runway = progress.StartRunway(None, 60.0)
    gave_up = _run(clock, monkeypatch, runway, lambda t: 2.0)  # stopped / blocked on a lock
    assert gave_up == pytest.approx(60.0, abs=1.0)


def test_owner_that_stalls_after_progressing_fails_within_two_windows(clock, monkeypatch):
    runway = progress.StartRunway(None, 60.0)
    gave_up = _run(clock, monkeypatch, runway, lambda t: 0.05 * min(t, 90.0))
    assert 90.0 < gave_up <= 90.0 + 2 * progress.PROGRESS_WINDOW_S + progress._LEAD_S + 1


def test_explicit_timeout_is_a_hard_deadline(clock, monkeypatch):
    runway = progress.StartRunway(10.0, 60.0)
    gave_up = _run(clock, monkeypatch, runway, lambda t: 0.05 * t)
    assert gave_up == pytest.approx(10.0, abs=1.0)


def test_unreadable_owner_cpu_never_extends(clock, monkeypatch):
    runway = progress.StartRunway(None, 60.0)
    gave_up = _run(clock, monkeypatch, runway, lambda t: None)  # AccessDenied / gone
    assert gave_up == pytest.approx(60.0, abs=1.0)


def test_owner_cpu_readable_only_after_the_first_sample_still_extends(clock, monkeypatch):
    runway = progress.StartRunway(None, 60.0)
    gave_up = _run(clock, monkeypatch, runway, lambda t: None if t < 3 else 0.05 * t)
    assert gave_up == pytest.approx(progress.MAX_ADAPTIVE_TIMEOUT, abs=1.0)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_refused(timeout):
    with pytest.raises(ValueError):
        progress.StartRunway(timeout, 60.0)


def test_cpu_seconds_reads_this_process():
    import os
    cpu = progress._cpu_seconds(os.getpid())
    assert cpu is not None and cpu > 0


def test_a_lock_record_without_a_running_gateway_is_never_the_owner(tmp_path, monkeypatch):
    """A backup/restore holding the home writes a lock record too; its CPU is not the start's."""
    from types import SimpleNamespace
    from hermes_agent.gateway import status
    monkeypatch.setattr(status, "get_running_pid", lambda pid_path, cleanup_stale: None)
    assert progress._starting_owner_pid((tmp_path,), SimpleNamespace(pid=77)) == 77
