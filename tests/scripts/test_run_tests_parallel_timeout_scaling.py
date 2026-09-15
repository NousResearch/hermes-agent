"""Duration-aware per-file timeout scaling in scripts/run_tests_parallel.py.

The flat --file-timeout cap (default 300s) falsely SIGKILL'd
known-slow large-collection files under CI load, then the automatic
retry passed — manufacturing FLAKY reports for healthy files
(tests/test_hermes_state.py, 2026-08-18 on main). The scaler gives a
file max(flat_cap, 3 × last observed duration) and never lowers the cap.
Only first-attempt-clean durations feed the cache, so a hang can never
compound its own bound.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scaler_never_lowers_the_flat_cap() -> None:
    mod = _load_runner()
    uncached = REPO_ROOT / "tests" / "test_example.py"
    fast = REPO_ROOT / "tests" / "test_fast.py"
    zero = REPO_ROOT / "tests" / "test_zero.py"
    durations = {
        mod._format_file(fast, REPO_ROOT): 4.2,
        mod._format_file(zero, REPO_ROOT): 0.0,
    }
    assert mod._effective_file_timeout(uncached, REPO_ROOT, 300.0, None) == 300.0
    assert mod._effective_file_timeout(uncached, REPO_ROOT, 300.0, durations) == 300.0
    assert mod._effective_file_timeout(fast, REPO_ROOT, 300.0, durations) == 300.0
    assert mod._effective_file_timeout(zero, REPO_ROOT, 300.0, durations) == 300.0


def test_slow_file_gets_proportional_headroom() -> None:
    mod = _load_runner()
    f = REPO_ROOT / "tests" / "test_hermes_state.py"
    durations = {mod._format_file(f, REPO_ROOT): 205.0}
    # 205s last run → 615s bound: a load-dilated healthy run survives,
    # a genuine hang is still killed.
    assert mod._effective_file_timeout(f, REPO_ROOT, 300.0, durations) == 615.0


def test_only_first_attempt_clean_durations_feed_the_cache() -> None:
    """A timed-out or retried file must not raise its own future bound."""
    mod = _load_runner()
    clean = REPO_ROOT / "tests" / "test_clean.py"
    hung = REPO_ROOT / "tests" / "test_hung.py"
    flaky = REPO_ROOT / "tests" / "test_flaky.py"
    file_times = [(clean, 12.0), (hung, 300.4), (flaky, 250.0)]
    failures = [(hung, "(300s exceeded; process tree SIGKILL'd)", {})]
    flaky_results = [(flaky, "⚠ FLAKY: failed on attempt 1, passed on retry")]

    kept = mod._clean_pass_durations(file_times, failures, flaky_results)

    assert kept == [(clean, 12.0)]
