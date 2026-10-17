"""Per-job jitter: a job with jitter_seconds sleeps 0–N s before firing; without it, no delay."""
import time

from cron.scheduler import _apply_job_jitter


def test_no_jitter_returns_zero_immediately():
    """A job without jitter_seconds (or with 0) must not sleep."""
    job = {"id": "j1", "name": "no-jitter", "prompt": "hi"}
    start = time.monotonic()
    slept = _apply_job_jitter(job)
    elapsed = time.monotonic() - start
    assert slept == 0
    assert elapsed < 0.1


def test_zero_jitter_returns_zero():
    """jitter_seconds=0 explicitly means no delay."""
    job = {"id": "j2", "name": "zero-jitter", "jitter_seconds": 0}
    start = time.monotonic()
    slept = _apply_job_jitter(job)
    elapsed = time.monotonic() - start
    assert slept == 0
    assert elapsed < 0.1


def test_jitter_waits_at_least_one_second():
    """A job with jitter_seconds=2 must sleep between 1 and 2 s (randint(0,2) can be 0,
    1, or 2; we test the floor: at least 1 s in the worst case of randint returning 1).
    To make this deterministic we monkeypatch random.randint to return 2."""
    import random as _random
    original = _random.randint
    _random.randint = lambda a, b: b  # always return the max
    try:
        job = {"id": "j3", "name": "jitter-2s", "jitter_seconds": 2}
        start = time.monotonic()
        slept = _apply_job_jitter(job)
        elapsed = time.monotonic() - start
        assert slept == 2
        assert elapsed >= 2.0
    finally:
        _random.randint = original


def test_jitter_returns_zero_when_randint_returns_zero():
    """randint(0, N) can return 0 — the function must handle that gracefully."""
    import random as _random
    original = _random.randint
    _random.randint = lambda a, b: 0
    try:
        job = {"id": "j4", "name": "jitter-0", "jitter_seconds": 5}
        start = time.monotonic()
        slept = _apply_job_jitter(job)
        elapsed = time.monotonic() - start
        assert slept == 0
        assert elapsed < 0.1
    finally:
        _random.randint = original
