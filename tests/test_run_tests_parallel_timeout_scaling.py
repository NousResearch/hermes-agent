"""Duration-aware per-file timeout scaling in scripts/run_tests_parallel.py.

The flat --file-timeout cap (default 300s) falsely SIGKILL'd
known-slow large-collection files under CI load, then the automatic
retry passed — manufacturing FLAKY reports for healthy files
(tests/test_hermes_state.py, 2026-08-18 on main). The scaler gives a
file max(flat_cap, 3 × last observed duration) and never lowers the cap.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_uncached_file_keeps_flat_cap() -> None:
    mod = _load_runner()
    f = REPO_ROOT / "tests" / "test_example.py"
    assert mod._effective_file_timeout(f, REPO_ROOT, 300.0, {}) == 300.0
    assert mod._effective_file_timeout(f, REPO_ROOT, 300.0, None) == 300.0


def test_fast_file_keeps_flat_cap() -> None:
    mod = _load_runner()
    f = REPO_ROOT / "tests" / "test_fast.py"
    durations = {mod._format_file(f, REPO_ROOT): 4.2}
    # 3 × 4.2 « 300 — the flat cap stays; the scaler never lowers a bound.
    assert mod._effective_file_timeout(f, REPO_ROOT, 300.0, durations) == 300.0


def test_slow_file_gets_proportional_headroom() -> None:
    mod = _load_runner()
    f = REPO_ROOT / "tests" / "test_hermes_state.py"
    durations = {mod._format_file(f, REPO_ROOT): 205.0}
    # 205s last run → 615s bound: a load-dilated healthy run survives,
    # a genuine hang is still killed.
    assert mod._effective_file_timeout(f, REPO_ROOT, 300.0, durations) == 615.0


def test_zero_or_missing_duration_is_ignored() -> None:
    mod = _load_runner()
    f = REPO_ROOT / "tests" / "test_zero.py"
    durations = {mod._format_file(f, REPO_ROOT): 0.0}
    assert mod._effective_file_timeout(f, REPO_ROOT, 300.0, durations) == 300.0
