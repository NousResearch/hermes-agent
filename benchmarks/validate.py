"""Quick validation script for the benchmark suite."""
import json
import os
import sys
import py_compile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Validate JSON fixtures
fixture_dir = os.path.join(os.path.dirname(__file__), "suite_a", "fixtures")
total = 0
for name in ["semantic_recall", "contradictions", "temporal_decay",
             "cross_reference", "importance_filtering"]:
    path = os.path.join(fixture_dir, f"{name}.json")
    with open(path) as f:
        data = json.load(f)
    total += len(data)
    print(f"  {name}: {len(data)} scenarios")
print(f"  TOTAL: {total} scenarios\n")

# 2. Compile check
base = os.path.dirname(__file__)
py_files = [
    "__init__.py", "__main__.py", "interface.py", "judge.py",
    "statistical.py", "runner.py", "baseline/flat_store.py",
]
for f in py_files:
    py_compile.compile(os.path.join(base, f), doraise=True)
print("  All Python files compile OK\n")

# 3. Test flat store
from benchmarks.baseline.flat_store import FlatMemoryStore

store = FlatMemoryStore()
store.store("PostgreSQL 15 with pgvector extension")
store.store("The backend runs on port 8080")
store.store("Redis is used for caching")

results = store.recall("What database do we use?")
assert results, "Recall returned empty!"
print(f"  Flat store recall: '{results[0]}'")

store.simulate_time(10)
store.simulate_access("PostgreSQL")
assert store.get_stats()["clock"] == 10.0

store.reset()
assert store.get_stats()["total_memories"] == 0
print("  FlatMemoryStore: all operations work\n")

# 4. Test interface imports
from benchmarks.interface import (
    BenchmarkableStore, BenchmarkConfig, RunResult,
    AggregateResult, CategoryResult, SignificanceResult, JudgeResult,
)
cfg = BenchmarkConfig(backend_name="test")
assert cfg.num_runs == 5
print("  Interface dataclasses: OK\n")

# 5. Test statistical
from benchmarks.statistical import compute_confidence_interval, aggregate_results
ci = compute_confidence_interval([0.8, 0.85, 0.82, 0.79, 0.83])
print(f"  CI test: [{ci[0]:.3f}, {ci[1]:.3f}]")
assert ci[0] < 0.82 < ci[1], "CI should contain the mean"
print("  Statistical module: OK\n")

print("=" * 50)
print("  VALIDATION PASSED — all checks green")
print("=" * 50)
