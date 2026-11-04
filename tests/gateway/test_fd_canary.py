"""fd budget canary unit tests (MTK Vikunja card #150 follow-up).

Pure decision core + threshold parsing + a smoke of the housekeeping entry
point with no runner (logs, never pages, never raises).
"""
import logging

from gateway.fd_canary import (
    DEFAULT_THRESHOLD,
    PAGE_WINDOW_SECONDS,
    _threshold,
    evaluate,
    fd_count,
    housekeeping_tick,
)


def test_fd_count_returns_positive_int_for_self():
    assert isinstance(fd_count(), int)
    assert fd_count() > 0


def test_fd_count_missing_pid_returns_minus_one():
    assert fd_count(999_999_999) == -1


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("HERMES_FD_CANARY_THRESHOLD", "700")
    assert _threshold() == 700
    monkeypatch.setenv("HERMES_FD_CANARY_THRESHOLD", "not-a-number")
    assert _threshold() == DEFAULT_THRESHOLD
    monkeypatch.setenv("HERMES_FD_CANARY_THRESHOLD", "")
    assert _threshold() == DEFAULT_THRESHOLD
    monkeypatch.delenv("HERMES_FD_CANARY_THRESHOLD", raising=False)
    assert _threshold() == DEFAULT_THRESHOLD


def test_evaluate_healthy_never_pages():
    state = {"last_page_ts": 0.0, "last_count": -1}
    assert evaluate(100, 512, now=1000.0, state=state) == (False, False)
    assert state["last_page_ts"] == 0.0


def test_evaluate_pages_once_per_window_then_rate_limits():
    state = {"last_page_ts": 0.0, "last_count": -1}
    now = 1_000_000.0
    # First crossing pages.
    assert evaluate(600, 512, now=now + 1, state=state) == (True, True)
    # Still over budget inside the window: over, but rate-limited.
    assert evaluate(700, 512, now=now + PAGE_WINDOW_SECONDS - 1, state=state) == (False, True)
    # Window elapsed while still over budget: pages again.
    assert evaluate(650, 512, now=now + PAGE_WINDOW_SECONDS + 2, state=state) == (True, True)
    # Recovery resets the limiter, so the next crossing pages immediately.
    assert evaluate(200, 512, now=now + PAGE_WINDOW_SECONDS + 10, state=state) == (False, False)
    assert state["last_page_ts"] == 0.0
    assert evaluate(600, 512, now=now + PAGE_WINDOW_SECONDS + 20, state=state) == (True, True)


def test_evaluate_negative_count_is_not_over():
    state = {"last_page_ts": 0.0, "last_count": -1}
    assert evaluate(-1, 512, now=1000.0, state=state) == (False, False)


def test_housekeeping_tick_without_runner_never_raises(caplog, monkeypatch):
    monkeypatch.setenv("HERMES_FD_CANARY_THRESHOLD", "2")  # any real process is over 2 fds
    with caplog.at_level(logging.ERROR):
        housekeeping_tick(None, None)
    assert any("fd budget" in rec.message for rec in caplog.records)
    # Second tick inside the window: still over budget, rate-limited, no new page attempt.
    with caplog.at_level(logging.DEBUG):
        housekeeping_tick(None, None)
    assert any("rate-limited" in rec.message for rec in caplog.records)