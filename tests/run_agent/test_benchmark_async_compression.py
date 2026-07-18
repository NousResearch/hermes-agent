"""CI wrapper for the controlled benchmark (task 10).

Runs ``scripts/benchmark_async_compression.py`` with light parameters and
enforces the structural gates. Timing-sensitive thresholds are relaxed vs
the script's production-scale defaults (2s simulated summariser) so CI
variance can't flake: at 500ms the apply pause is already an order of
magnitude smaller than the pause it replaces.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_async_compression",
        REPO_ROOT / "scripts" / "benchmark_async_compression.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("benchmark_async_compression", module)
    spec.loader.exec_module(module)
    return module


def test_benchmark_gates_hold_at_reduced_scale():
    bench = _load_benchmark_module()
    report = bench.run_benchmark(iterations=5, summariser_ms=500.0,
                                 turn_work_ms=10.0)

    modes = report["modes"]
    # Structural gates — never timing-flaky.
    assert report["loss_or_dup"] == 0
    assert report["errors"] == []
    assert modes["background"]["ready_rate"] == 1.0

    # The apply window must stay far below both the plan's 500ms budget and
    # the synchronous pause it replaces.
    assert modes["background"]["pause_p95_ms"] < 500.0
    assert report["pause_reduction"] >= 0.80  # >=0.90 at production scale

    # Shadow and fallback must pay the SAME pause class as sync (they run
    # the synchronous path) — the feature never makes the baseline worse.
    assert modes["shadow"]["pause_p95_ms"] >= 500.0 * 0.9
    assert modes["fallback"]["pause_p95_ms"] >= 500.0 * 0.9
