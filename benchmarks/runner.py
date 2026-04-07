"""
Memory Benchmark Runner — system-agnostic AI agent memory evaluation

Usage:
    python -m benchmarks.runner --backend baseline-flat --suite all --runs 3
    python -m benchmarks.runner --backend my-backend --suite a,b,c --runs 5
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from benchmarks.interface import (
    BenchmarkableStore, BenchmarkConfig, RunResult,
    AggregateResult, CategoryResult, SignificanceResult,
)
from benchmarks.capabilities import BackendCapabilities
from benchmarks.judge import MemoryJudge, HeuristicJudge
from benchmarks.statistical import aggregate_results, compare_runs
from benchmarks.metrics import compute_metric_suite, token_f1, exact_match, recall_at_k, mrr, compute_cost_metrics


# --- Backend Registry ---

BACKENDS: Dict[str, Type[BenchmarkableStore]] = {}
BACKEND_CAPABILITIES: Dict[str, "BackendCapabilities"] = {}  # type: ignore[name-defined]


def register_backend(name: str, cls: Type[BenchmarkableStore],
                    capabilities: "BackendCapabilities | None" = None):  # type: ignore[name-defined]
    """Register a memory backend for benchmarking."""
    BACKENDS[name] = cls
    if capabilities is not None:
        BACKEND_CAPABILITIES[name] = capabilities


def get_backend(name: str, config: BenchmarkConfig) -> BenchmarkableStore:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name](**config.parameters)


# Register built-in backends
from benchmarks.baseline.flat_store import FlatMemoryStore, BACKEND_CAPABILITIES as BASELINE_CAPS
register_backend("baseline-flat", FlatMemoryStore, BASELINE_CAPS)

# Auto-discover plugin backends from benchmarks/backends/
try:
    import importlib
    _backends_dir = Path(__file__).parent / "backends"
    if _backends_dir.exists():
        for _plugin_file in _backends_dir.glob("*.py"):
            if _plugin_file.name.startswith("_"):
                continue
            try:
                _mod = importlib.import_module(f"benchmarks.backends.{_plugin_file.stem}")
                if hasattr(_mod, "BACKEND_NAME") and hasattr(_mod, "BACKEND_CLASS"):
                    _caps = getattr(_mod, "BACKEND_CAPABILITIES", None)
                    register_backend(_mod.BACKEND_NAME, _mod.BACKEND_CLASS, _caps)
                else:
                    print(f"  [warn] benchmarks/backends/{_plugin_file.name}: "
                          f"missing BACKEND_NAME or BACKEND_CLASS export, skipped",
                          file=sys.stderr)
            except Exception as _exc:
                print(f"  [warn] benchmarks/backends/{_plugin_file.name}: "
                      f"failed to load: {_exc}", file=sys.stderr)
except Exception:
    pass


# --- Token Estimation ---

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text.
    Good enough for cost estimation. For exact counts, use tiktoken.
    """
    return max(len(text) // 4, 1)


def count_recall_tokens(results: list) -> tuple:
    """Count tokens and chars in recalled memory strings.
    Returns (token_count, char_count).
    """
    total_chars = sum(len(r) for r in results)
    total_tokens = sum(estimate_tokens(r) for r in results)
    return total_tokens, total_chars


def compute_scenario_metrics(results: list, gold_answer: str, query: str = "") -> dict:
    """Compute retrieval and answer metrics for a single scenario.

    Args:
        results: List of recalled memory strings (ranked by relevance)
        gold_answer: The expected answer
        query: The query used (for reference only)

    Returns:
        Dict with metric values
    """
    actual = results[0] if results else ""
    # For retrieval metrics, the 'relevant' set is the gold answer
    relevant = [gold_answer]

    return compute_metric_suite(
        retrieved=results,
        relevant=relevant,
        gold_answer=gold_answer,
        predicted_answer=actual,
    )


# --- Fixture Loading ---

SUITE_DIR = Path(__file__).parent


def load_fixtures(suite: str) -> Dict[str, list]:
    """Load all fixture JSON files for a given suite (e.g., 'a')."""
    fixture_dir = SUITE_DIR / f"suite_{suite}" / "fixtures"
    if not fixture_dir.exists():
        raise FileNotFoundError(f"No fixtures found at {fixture_dir}")

    fixtures = {}
    for f in sorted(fixture_dir.glob("*.json")):
        with open(f) as fh:
            fixtures[f.stem] = json.load(fh)
    return fixtures


# --- Scenario Runners ---

def run_semantic_recall(backend: BenchmarkableStore, scenarios: list,
                        judge: MemoryJudge) -> CategoryResult:
    """Run semantic recall scenarios (Suite A1)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()
        # Store distractor facts first (topically related noise that forces
        # genuine retrieval discrimination — without distractors, a backend
        # that returns "most recent memory" trivially scores 100%)
        for distractor in sc.get("distractors", []):
            backend.store(distractor, category="factual")
        # Store the target fact
        backend.store(sc["fact"], category="factual")
        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Sub-scores by difficulty
    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="semantic_recall",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_contradictions(backend: BenchmarkableStore, scenarios: list,
                       judge: MemoryJudge) -> CategoryResult:
    """Run contradiction handling scenarios (Suite A2).
    
    Harder than naive "store A then B": stores fact_a among distractors,
    advances time, then stores the contradicting fact_b. This means
    recency alone isn't enough — the system needs to either supersede
    fact_a or reliably rank fact_b higher even with many other recent
    memories in the store.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    # Distractor facts that persist across all scenarios
    distractors = [
        "The office has a ping pong table in the break room",
        "Team lunch is every Thursday at noon",
        "The wifi password is posted near the coffee machine",
        "Company all-hands meeting is the first Monday of each month",
        "The parking garage closes at 10 PM",
    ]

    for sc in scenarios:
        backend.reset()

        # Store distractors first
        for d in distractors:
            backend.store(d, category="factual")

        # Store the original fact
        backend.store(sc["fact_a"], category="factual")

        # More distractors to bury fact_a
        backend.store("Code reviews require at least two approvals", category="factual")
        backend.store("Sprint planning happens every other Monday", category="factual")

        # Advance time — simulates days passing
        backend.simulate_time(7)  # 1 week later

        # Add some recent distractors (before fact_b)
        backend.store("The new intern starts next week", category="factual")
        backend.store("Friday is demo day for the current sprint", category="factual")

        # Now store the contradicting/updated fact (not the most recent)
        backend.store(sc["fact_b"], category="factual")

        # One more distractor after — fact_b is NOT the newest thing
        backend.store("Remember to water the office plants", category="factual")

        results = backend.recall(sc["query"], top_k=5)
        # For subtle_update contradictions (facts that ADD to rather than replace),
        # the gold answer may require multiple facts — provide top-2
        if sc.get("contradiction_type") == "subtle_update" and len(results) > 1:
            actual = " | ".join(results[:2])
        else:
            actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "type": sc["contradiction_type"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="contradictions",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_temporal_decay(backend: BenchmarkableStore, scenarios: list,
                       judge: MemoryJudge) -> CategoryResult:
    """Run temporal decay scenarios (Suite A3)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Store facts with proper time simulation.
        # Sort by stored_days_ago descending (oldest first).
        # Advance simulated time between stores to create real recency gaps.
        sorted_facts = sorted(sc["facts"], key=lambda f: f["stored_days_ago"], reverse=True)
        prev_days_ago = sorted_facts[0]["stored_days_ago"] if sorted_facts else 0

        for fact in sorted_facts:
            # Advance time from previous fact to this one
            time_gap = prev_days_ago - fact["stored_days_ago"]
            if time_gap > 0:
                backend.simulate_time(time_gap)
            prev_days_ago = fact["stored_days_ago"]

            backend.store(fact["content"], category="factual")

            # Simulate rehearsals if present
            for r_day in fact.get("rehearsed_days_ago", []):
                backend.simulate_access(fact["content"])

        # Advance remaining time to "now" (days_ago=0)
        if prev_days_ago > 0:
            backend.simulate_time(prev_days_ago)

        results = backend.recall(sc["query"], top_k=5)
        # For hard temporal scenarios, top fact may be partial — pass top-2
        if sc.get("difficulty") == "hard" and len(results) > 1:
            actual = " | ".join(results[:2])
        else:
            actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="temporal_decay",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_cross_reference(backend: BenchmarkableStore, scenarios: list,
                        judge: MemoryJudge) -> CategoryResult:
    """Run cross-reference scenarios (Suite A4)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()
        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        results = backend.recall(sc["query"], top_k=10)
        # Concatenate top results as the answer context
        actual = " | ".join(results[:sc["num_facts_needed"]]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "num_facts_needed": sc["num_facts_needed"],
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="cross_reference",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_importance_filtering(backend: BenchmarkableStore, scenarios: list,
                             judge: MemoryJudge) -> CategoryResult:
    """Run importance filtering scenarios (Suite A5)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()
        for fact in sc["important_facts"]:
            backend.store(fact["content"], importance=fact["importance"])
        for fact in sc["noise_facts"]:
            backend.store(fact["content"], importance=fact["importance"])

        results = backend.recall(sc["query"], top_k=5)
        # Use top-k results to cover multi-fact scenarios (num_important >= 2)
        num_needed = len(sc.get("important_facts", []))
        if num_needed > 1 and len(results) > 1:
            actual = " | ".join(results[:num_needed])
        else:
            actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="importance_filtering",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite B: Consolidation & Compression ---

def run_consolidation(backend: BenchmarkableStore, scenarios: list,
                      judge: MemoryJudge) -> CategoryResult:
    """Run consolidation scenarios (Suite B1).

    Each scenario stores facts, rehearses some via access_sequence,
    advances time by time_gap_days, calls consolidate(), then checks
    whether the target fact is still recalled.  The expects_layer field
    (core/archive) is recorded in details for downstream analysis; the
    benchmark score is purely based on recall correctness.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        # Simulate rehearsals
        for access_hint in sc.get("access_sequence", []):
            backend.simulate_access(access_hint)

        # Advance time to trigger decay / consolidation pressure
        backend.simulate_time(sc.get("time_gap_days", 30))

        # Run consolidation cycle
        backend.consolidate()

        results = backend.recall(sc["query"], top_k=5)
        # Pass top-3 results to judge — consolidation is about whether facts
        # survive at all, not strict #1 ranking
        actual = " | ".join(results[:3]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "expects_layer": sc.get("expects_layer", "unknown"),
            "time_gap_days": sc.get("time_gap_days", 30),
            "rehearsals": len(sc.get("access_sequence", [])),
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Sub-scores: core (frequently accessed) vs archive (never accessed)
    sub_scores = {}
    for layer in ["core", "archive"]:
        subset = [d for d in details if d["expects_layer"] == layer]
        if subset:
            sub_scores[layer] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="consolidation",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_compression(backend: BenchmarkableStore, scenarios: list,
                    judge: MemoryJudge) -> CategoryResult:
    """Run compression scenarios (Suite B2).

    Stores a set of facts (some redundant), advances time, consolidates,
    then checks that critical information is preserved in recall.
    Scoring: correct if gold_answer recalled AND all must_preserve
    substrings appear in the top-5 recalled memories.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        # Let time pass and consolidate
        backend.simulate_time(30)
        backend.consolidate()

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        # Check gold answer
        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)

        # Check must_preserve substrings in top-5
        combined = " ".join(results).lower()
        must_preserve = sc.get("must_preserve", [])
        preserved = all(mp.lower() in combined for mp in must_preserve)

        scenario_correct = jr.correct and preserved
        if scenario_correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "correct": scenario_correct,
            "answer_correct": jr.correct,
            "preserved": preserved,
            "must_preserve": must_preserve,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="compression",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite C: Multi-scope memory ---

def run_scopes(backend: BenchmarkableStore, scenarios: list,
               judge: MemoryJudge) -> CategoryResult:
    """Run multi-scope memory scenarios (Suite C).

    Stores facts with different scope tags, recalls with query_scope,
    checks that the gold answer appears AND the should_not_contain
    string is absent from the top-5 results.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        for fact_obj in sc["facts"]:
            backend.store(
                fact_obj["content"],
                category="factual",
                scope=fact_obj.get("scope", "global"),
            )

        query_scope = sc.get("query_scope", "global")
        results = backend.recall(sc["query"], top_k=5, scope=query_scope)
        # Pass top-3 results to judge — scope isolation is tested by
        # should_not_contain, retrieval quality by answer correctness
        actual = " | ".join(results[:3]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)

        # Check that the wrong-scope content did not leak in
        combined = " ".join(results).lower()
        should_not = sc.get("should_not_contain", "")
        no_leak = should_not.lower() not in combined if should_not else True

        scenario_correct = jr.correct and no_leak
        if scenario_correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "query_scope": query_scope,
            "correct": scenario_correct,
            "answer_correct": jr.correct,
            "no_leak": no_leak,
            "should_not_contain": should_not,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Sub-scores: answer correctness vs leak prevention
    sub_scores = {
        "answer_correct": (
            sum(1 for d in details if d["answer_correct"]) / len(details)
            if details else 0
        ),
        "no_scope_leak": (
            sum(1 for d in details if d["no_leak"]) / len(details)
            if details else 0
        ),
    }

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="scopes",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite D: Adversarial robustness ---

def run_adversarial(backend: BenchmarkableStore, scenarios: list,
                    judge: MemoryJudge) -> CategoryResult:
    """Run adversarial robustness scenarios (Suite D).

    Stores normal facts + one malicious fact, recalls with a legitimate
    query, checks that:
      1. The gold answer is present in the top result.
      2. If should_be_blocked=True, the malicious_fact content does NOT
         appear as the top-1 result.

    Note: the cognitive store does not sanitize injections (that is
    hermes-aegis's job).  This benchmark measures whether embedding-based
    relevance scoring naturally de-prioritises adversarial payloads.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Store normal facts first
        for fact in sc.get("normal_facts", []):
            backend.store(fact, category="factual")

        # Store the adversarial fact (at standard importance — no special treatment)
        backend.store(sc["malicious_fact"], category="factual")

        results = backend.recall(sc["query"], top_k=5)
        # Pass top-5 results to judge — adversarial tests care about
        # (1) blocking injections from top-1 and (2) retrieving the correct answer
        actual = " | ".join(results[:5]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)

        # For should_be_blocked cases: malicious payload must not be top-1
        should_be_blocked = sc.get("should_be_blocked", False)
        malicious = sc["malicious_fact"].lower()
        top1 = results[0].lower().strip() if results else ""
        top1_is_malicious = top1 == malicious.strip()
        blocked_ok = (not should_be_blocked) or (not top1_is_malicious)

        scenario_correct = jr.correct and blocked_ok
        if scenario_correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "type": sc.get("type", "unknown"),
            "should_be_blocked": should_be_blocked,
            "top1_is_malicious": top1_is_malicious,
            "blocked_ok": blocked_ok,
            "correct": scenario_correct,
            "answer_correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Sub-scores by adversarial type
    sub_scores = {}
    for atype in set(d["type"] for d in details):
        subset = [d for d in details if d["type"] == atype]
        if subset:
            sub_scores[atype] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Also report overall block rate
    blockable = [d for d in details if d["should_be_blocked"]]
    if blockable:
        sub_scores["block_rate"] = sum(1 for d in blockable if d["blocked_ok"]) / len(blockable)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="adversarial",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite E: Scale & Performance ---

def run_scale(backend: BenchmarkableStore, scenarios: list,
              judge: MemoryJudge) -> CategoryResult:
    """Run scale & performance scenarios (Suite E).

    Each scenario has either a flat list of facts or a time_series list.
    The noise_count / noise_template fields generate synthetic padding to
    reach the target memory_count without bloating the fixture file.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Time-series scenarios: store facts at different simulated ages
        if "time_series" in sc:
            # Sort oldest first so time advances forward
            series = sorted(sc["time_series"], key=lambda x: -x["stored_days_ago"])
            prev_days = series[0]["stored_days_ago"]
            for item in series:
                gap = prev_days - item["stored_days_ago"]
                if gap > 0:
                    backend.simulate_time(gap)
                backend.store(item["content"], category="factual")
                prev_days = item["stored_days_ago"]
            # Store noise facts at current time
            for fact in sc.get("noise_facts", []):
                backend.store(fact, category="factual")
        else:
            # Flat scenarios: store target + noise.
            # Target is stored first so it gets the same small virtual-clock gap
            # as noise facts (0.1ms per insert in the adapter).  After all facts
            # are stored we advance time by 30 days so every fact has the same
            # age at recall — this isolates semantic/importance discrimination
            # from ACT-R recency effects (recency is Suite B's domain).
            # Support multi-needle scenarios (multiple target facts)
            if sc.get("multi_needle") and "target_facts" in sc:
                for fact in sc["target_facts"]:
                    backend.store(fact, category="factual", importance=0.8)
            else:
                backend.store(sc["target_fact"], category="factual", importance=0.8)
            for fact in sc.get("noise_facts", []):
                backend.store(fact, category="factual", importance=0.3)

            # Generate synthetic padding facts if noise_count specified
            noise_count = sc.get("noise_count", 0)
            template = sc.get("noise_template", "Configuration parameter {i} has value {val}")
            existing_noise = len(sc.get("noise_facts", []))
            for i in range(noise_count - existing_noise):
                content = template.format(i=i, val=f"value_{i}")
                backend.store(content, category="factual", importance=0.1)

            # Age all facts equally so recall tests semantic discrimination,
            # not insertion order (ACT-R recency)
            backend.simulate_time(30)

        results = backend.recall(sc["query"], top_k=5)
        # Pass top-3 results for answer matching — scale tests whether the
        # correct fact is retrieved at all, not strict #1 ranking
        actual = " | ".join(results[:3]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc.get("difficulty", "medium"),
            "memory_count": sc.get("memory_count", "?"),
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="scale",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite F: Integration ---

def run_integration(backend: BenchmarkableStore, scenarios: list,
                    judge: MemoryJudge, capabilities=None) -> CategoryResult:
    """Run integration scenarios (Suite F).

    Each scenario is a sequence of steps (store / recall / advance_time /
    simulate_access / consolidate).  A scenario passes if all recall steps
    find their expected_in_result string in the result set.

    Args:
        capabilities: BackendCapabilities instance for per-scenario filtering.
    """
    correct = 0
    skipped = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        # Check per-scenario capability requirements
        requires = sc.get("requires", [])
        if capabilities:
            missing = [cap for cap in requires if not getattr(capabilities, cap, False)]
            if missing:
                details.append({
                    "id": sc["id"],
                    "difficulty": sc.get("difficulty", "medium"),
                    "correct": False,
                    "skipped": True,
                    "reason": f"missing capabilities: {', '.join(missing)}",
                })
                skipped += 1
                continue

        backend.reset()
        scenario_pass = True
        recall_results_log = []

        for step in sc["steps"]:
            action = step["action"]

            if action == "store":
                kwargs = {
                    "category": step.get("category", "factual"),
                    "scope": step.get("scope", "global"),
                    "importance": step.get("importance", 0.5),
                }
                backend.store(step["content"], **kwargs)

            elif action == "recall":
                top_k = step.get("top_k", 5)
                scope = step.get("scope")
                results = backend.recall(step["query"], top_k=top_k, scope=scope)
                rt, rc = count_recall_tokens(results)
                total_recall_tokens += rt
                total_recall_chars += rc

                combined = " ".join(results).lower()
                expected = step.get("expected_in_result", "").lower()
                step_pass = expected in combined if expected else True
                recall_results_log.append({
                    "query": step["query"],
                    "expected": expected,
                    "found": step_pass,
                    "top_result": results[0] if results else "",
                })
                if not step_pass:
                    scenario_pass = False

            elif action == "advance_time":
                backend.simulate_time(step["days"])

            elif action == "simulate_access":
                backend.simulate_access(step["content_substring"])

            elif action == "consolidate":
                backend.consolidate()

        if scenario_pass:
            correct += 1

        # Use the judge on the final recall step's gold answer for details
        final_recall = recall_results_log[-1] if recall_results_log else {}
        details.append({
            "id": sc["id"],
            "difficulty": sc.get("difficulty", "medium"),
            "correct": scenario_pass,
            "steps": len(sc["steps"]),
            "recall_steps": len(recall_results_log),
            "recalls": recall_results_log,
            "gold": sc["gold_answer"],
        })
        # Use the last recall step's results list for metrics; fall back to top_result string
        last_results = ([final_recall["top_result"]] if final_recall.get("top_result") else [])
        scenario_metrics = compute_scenario_metrics(last_results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="integration",
        total=len(scenarios),
        correct=correct,
        score=correct / (len(scenarios) - skipped) if (len(scenarios) - skipped) > 0 else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite G: Q-Learning / Reranking ---

def run_qlearning(backend: BenchmarkableStore, scenarios: list,
                  judge: MemoryJudge, capabilities=None) -> CategoryResult:
    """Run Q-learning reranking scenarios (Suite G).

    Each scenario runs multiple rounds. In each round:
      1. Recall the query
      2. Reward gold memories (+1.0), penalize non-gold top-3 (-0.15)
    Score: did gold memories move UP in rank across rounds?
    Metric: rank improvement ratio = (initial_avg_rank - final_avg_rank) / initial_avg_rank
    """
    total_scenarios = 0
    improved_scenarios = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    has_reward = hasattr(backend, 'reward_memory')
    has_ids = hasattr(backend, 'recall_with_ids')

    for sc in scenarios:
        backend.reset()

        # Store all memories
        memory_ids = []
        for mem in sc["memories"]:
            # Use store() and track stored IDs by doing recall_with_ids after
            backend.store(
                mem["content"],
                category=mem.get("category", "factual"),
                importance=0.5,
            )

        # Get IDs by recalling with a broad query and matching content
        # We need memory IDs for rewarding - use recall_with_ids if available
        if has_ids:
            # Get all stored memories via a broad match
            all_results = backend.recall_with_ids(sc["query"], top_k=len(sc["memories"]) + 5)
        else:
            all_results = None

        query = sc["query"]
        gold_contents = {sc["memories"][i]["content"] for i in sc["gold_ids"]}
        rounds = sc.get("rounds", 3)

        round_gold_ranks = []

        for round_num in range(rounds):
            if has_ids:
                results_with_ids = backend.recall_with_ids(query, top_k=len(sc["memories"]) + 5)
                results = [content for content, _ in results_with_ids]
            else:
                results = backend.recall(query, top_k=len(sc["memories"]) + 5)
                results_with_ids = [(content, None) for content in results]

            rt, rc = count_recall_tokens(results)
            total_recall_tokens += rt
            total_recall_chars += rc

            # Compute gold ranks for this round (1-indexed)
            gold_ranks = []
            for rank, (content, mem_id) in enumerate(results_with_ids, start=1):
                if content in gold_contents:
                    gold_ranks.append(rank)

            round_gold_ranks.append(gold_ranks)

            # Apply rewards/penalties if supported
            if has_reward and has_ids:
                for rank, (content, mem_id) in enumerate(results_with_ids, start=1):
                    if mem_id is None:
                        continue
                    if content in gold_contents:
                        backend.reward_memory(mem_id, 1.0)
                    elif rank <= 3:
                        # Penalize non-gold memories in top-3 (dead ends)
                        backend.reward_memory(mem_id, -0.15)

        # Score: did gold memories improve in rank from round 1 to final round?
        initial_ranks = round_gold_ranks[0] if round_gold_ranks else []
        final_ranks = round_gold_ranks[-1] if round_gold_ranks else []

        if initial_ranks:
            initial_avg = sum(initial_ranks) / len(initial_ranks)
        else:
            initial_avg = float(len(sc["memories"]))

        if final_ranks:
            final_avg = sum(final_ranks) / len(final_ranks)
        else:
            final_avg = float(len(sc["memories"]))

        # Rank improvement ratio (positive = improved)
        if initial_avg > 0:
            rank_improvement = (initial_avg - final_avg) / initial_avg
        else:
            rank_improvement = 0.0

        # "Correct" = gold memories are well-ranked in the final round.
        # The benchmark tests whether the Q-learning pipeline surfaces the right
        # memories over repeated feedback rounds. We count a scenario as correct if:
        #   (a) gold memories improved in rank across rounds, OR
        #   (b) gold memories ended up in the top-(len(gold_ids)+1) positions,
        #       meaning the system reliably retrieves relevant memories.
        # This is intentionally lenient because Q-learning cold-start protection
        # means impact is subtle in 3 rounds but grows with more interactions.
        target_rank = len(sc["gold_ids"]) + 1  # top-k+1 positions for k gold items
        gold_well_ranked = len(final_ranks) > 0 and final_avg <= target_rank
        improved = rank_improvement > 0 or gold_well_ranked

        # Also judge the final recall quality
        final_results = results if results else []
        actual = final_results[0] if final_results else ""
        jr = judge.judge_answer(query, sc["gold_answer"], actual)

        total_scenarios += 1
        if improved:
            improved_scenarios += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc.get("difficulty", "medium"),
            "correct": improved,
            "rank_improvement": rank_improvement,
            "initial_avg_rank": initial_avg,
            "final_avg_rank": final_avg,
            "rounds": rounds,
            "gold_ids_count": len(sc["gold_ids"]),
            "recall_correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(final_results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Aggregate retrieval metrics
    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="qlearning",
        total=total_scenarios,
        correct=improved_scenarios,
        score=improved_scenarios / total_scenarios if total_scenarios > 0 else 0.0,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# Category runner dispatch
# --- Suite H: Advanced Memory Features ---


def run_supersession(backend: BenchmarkableStore, scenarios: list,
                     judge: MemoryJudge) -> CategoryResult:
    """Run supersession scenarios (Suite H1).

    Tests that storing a new fact with the same type+target supersedes the old one.
    Backends that support MEMORY_SPEC notation and supersession will excel;
    flat backends will have both facts competing.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Store distractors first
        for d in sc.get("distractors", []):
            backend.store(d, category="factual")

        # Store fact_a (the original)
        backend.store(sc["fact_a"], category="factual")

        # Store fact_b (the superseding fact)
        backend.store(sc["fact_b"], category="factual")

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="supersession",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_typed_decay(backend: BenchmarkableStore, scenarios: list,
                    judge: MemoryJudge) -> CategoryResult:
    """Run typed decay scenarios (Suite H2).

    Tests that constraints (C-type) persist longer than unknowns (?-type)
    due to metabolic decay rates. After long time periods, C-type facts
    should still be retrievable while ?-type facts should have faded.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Store facts at different simulated ages (oldest first)
        sorted_facts = sorted(sc["facts"], key=lambda f: -f["stored_days_ago"])
        prev_days = sorted_facts[0]["stored_days_ago"] if sorted_facts else 0

        for fact in sorted_facts:
            gap = prev_days - fact["stored_days_ago"]
            if gap > 0:
                backend.simulate_time(gap)
            prev_days = fact["stored_days_ago"]

            # Construct MEMORY_SPEC notation
            notation = f"{fact['type']}[{fact['target']}]: {fact['content']}"
            backend.store(notation, category="factual")

        # Advance remaining time to "now"
        if prev_days > 0:
            backend.simulate_time(prev_days)

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "expected_type": sc["expected_type"],
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="typed_decay",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_scope_lifecycle(backend: BenchmarkableStore, scenarios: list,
                        judge: MemoryJudge) -> CategoryResult:
    """Run scope lifecycle scenarios (Suite H3).

    Tests scope isolation, global fact accessibility, and scope closing.
    Backends with scope lifecycle management should outperform flat stores.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Store facts with scopes
        for fact in sc["facts"]:
            backend.store(
                fact["content"],
                category="factual",
                scope=fact.get("scope", "global"),
            )

        # Apply scope actions (close/cool)
        for action in sc.get("scope_actions", []):
            # Closing a scope: advance time slightly then consolidate
            # This triggers scope cooling in backends that support it
            backend.simulate_time(1)
            backend.consolidate()

        query_scope = sc.get("query_scope", "global")
        results = backend.recall(sc["query"], top_k=5, scope=query_scope)
        actual = " | ".join(results[:3]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)

        # Check scope leakage
        combined = " ".join(results).lower()
        should_not = sc.get("should_not_contain", "")
        no_leak = should_not.lower() not in combined if should_not else True

        scenario_correct = jr.correct and no_leak
        if scenario_correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": scenario_correct,
            "answer_correct": jr.correct,
            "no_leak": no_leak,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="scope_lifecycle",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_notation_parsing(backend: BenchmarkableStore, scenarios: list,
                         judge: MemoryJudge) -> CategoryResult:
    """Run notation parsing scenarios (Suite H4).

    Tests that MEMORY_SPEC notation (C[x]:, D[x]:, V[x]:) is correctly
    parsed and the content is retrievable. Flat backends store the notation
    as-is; smart backends parse type/target/content.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="notation_parsing",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_deduplication(backend: BenchmarkableStore, scenarios: list,
                      judge: MemoryJudge) -> CategoryResult:
    """Run deduplication scenarios (Suite H5).

    Tests that duplicate/near-duplicate facts are handled gracefully.
    Backends with dedup should store fewer facts; the gold answer should
    still be retrievable.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "num_results": len(results),
            "max_expected": sc.get("max_unique_results", 5),
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="deduplication",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite I: Conversation & Stress ---


def run_conversation_memory(backend: BenchmarkableStore, scenarios: list,
                            judge: MemoryJudge) -> CategoryResult:
    """Run conversation memory scenarios (Suite I1).

    Simulates multi-turn conversations where facts are stated, updated,
    and must be recalled later. Tests real-world dialogue patterns.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Process turns: store content from designated store_turns
        store_turns = set(sc.get("store_turns", []))
        for i, turn in enumerate(sc["turns"]):
            if i in store_turns:
                backend.store(turn["content"], category="factual")

        # Query on the final turn
        query = sc["turns"][sc["query_turn"]]["content"]
        results = backend.recall(query, top_k=5)
        actual = " | ".join(results[:2]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(query, sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="conversation_memory",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


def run_capacity_stress(backend: BenchmarkableStore, scenarios: list,
                        judge: MemoryJudge) -> CategoryResult:
    """Run capacity stress scenarios (Suite I2).

    Tests retrieval quality under high fact counts (50-1000 facts).
    Measures how well the backend discriminates signal from noise at scale.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        num_facts = sc["num_facts"]
        template = sc.get("noise_template", "Fact {i}: value is {val}")
        noise_importance = sc.get("noise_importance", 0.1)

        # Store target fact
        if "old_fact" in sc:
            # Time-series: old fact stored first
            backend.store(sc["old_fact"], category="factual", importance=0.5)
            backend.simulate_time(sc.get("old_days_ago", 90) - sc.get("new_days_ago", 5))

        target_fact = sc.get("target_fact") or sc.get("new_fact", "")
        target_importance = sc.get("target_importance", 0.8)
        backend.store(target_fact, category="factual", importance=target_importance)

        # Store noise
        for i in range(num_facts - 1):
            content = template.format(i=i, val=f"value_{i}")
            backend.store(content, category="factual", importance=noise_importance)

        # Time-series: advance remaining days
        if "new_days_ago" in sc:
            backend.simulate_time(sc["new_days_ago"])

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "num_facts": num_facts,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="capacity_stress",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite M: Format Sensitivity (Omni-SimpleMem) ---

def run_format_sensitivity(backend: BenchmarkableStore, scenarios: list,
                           judge: MemoryJudge) -> CategoryResult:
    """Run format sensitivity scenarios (Suite M).

    Based on Omni-SimpleMem paper findings that LLM memory systems are sensitive
    to output format constraints embedded in stored facts and queries.

    Three sub-categories:
      clean_structured  — facts stored as JSON/YAML/structured strings; tests
                          whether the backend preserves and retrieves structured
                          content without mangling it.
      refusal           — no stored fact answers the query; correct behaviour is
                          returning an empty result set (gold_answer == 'NONE').
      constraint_start  — format directive prepended to the key fact; tests
                          whether retrieval survives a noisy preamble.
      constraint_end    — format directive appended after the key fact; tests
                          whether retrieval survives a noisy suffix.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        fmt_type = sc.get("format_type", "clean_structured")
        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        if fmt_type == "refusal":
            # Correct if the backend returns nothing relevant (empty result set
            # or the top result genuinely does not contain the queried info).
            if not results:
                scenario_correct = True
            elif actual == "" or actual.strip().upper() in ("NONE", "N/A", "UNKNOWN"):
                scenario_correct = True
            else:
                # If backend returned something, judge whether it actually answers
                # the query (it shouldn't). Correct = judge says NOT answered.
                jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
                scenario_correct = not jr.correct
        else:
            jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
            scenario_correct = jr.correct

        if scenario_correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "format_type": fmt_type,
            "difficulty": sc["difficulty"],
            "correct": scenario_correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Sub-scores by format type and by difficulty
    sub_scores = {}
    for ftype in ["clean_structured", "refusal", "constraint_start", "constraint_end"]:
        fsubset = [d for d in details if d["format_type"] == ftype]
        if fsubset:
            sub_scores[ftype] = sum(1 for d in fsubset if d["correct"]) / len(fsubset)
    for diff in ["easy", "medium", "hard"]:
        dsubset = [d for d in details if d["difficulty"] == diff]
        if dsubset:
            sub_scores[diff] = sum(1 for d in dsubset if d["correct"]) / len(dsubset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="format_sensitivity",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite N: Retrieval Ablation (Omni-SimpleMem) ---

def run_retrieval_ablation(backend: BenchmarkableStore, scenarios: list,
                           judge: MemoryJudge) -> CategoryResult:
    """Run retrieval ablation scenarios (Suite N).

    Based on Omni-SimpleMem findings that hybrid retrieval (BM25 + dense) outperforms
    either signal alone. Tests three retrieval signal conditions:

      keyword  — target fact shares exact lexical tokens with the query (BM25 advantage);
                 distractors are semantic paraphrases that dense retrieval might rank higher.
      semantic — target fact is conceptually aligned but uses different vocabulary (dense
                 advantage); distractors share keywords with the query but are wrong.
      hybrid   — both lexical and semantic signals are required to disambiguate the correct
                 fact from strong distractors; neither BM25-only nor dense-only succeeds.

    Sub-scores by retrieval_signal allow ablation of each signal's contribution.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        signal = sc.get("retrieval_signal", "hybrid")
        results = backend.recall(sc["query"], top_k=5)
        # Pass top-3 for hybrid scenarios where the correct fact may not rank #1
        # in a single-signal backend but should appear in top-3
        if signal == "hybrid" and len(results) > 1:
            actual = " | ".join(results[:3])
        else:
            actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "retrieval_signal": signal,
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    # Sub-scores by retrieval signal (ablation analysis)
    sub_scores = {}
    for sig in ["keyword", "semantic", "hybrid"]:
        ssubset = [d for d in details if d["retrieval_signal"] == sig]
        if ssubset:
            sub_scores[sig] = sum(1 for d in ssubset if d["correct"]) / len(ssubset)
    for diff in ["easy", "medium", "hard"]:
        dsubset = [d for d in details if d["difficulty"] == diff]
        if dsubset:
            sub_scores[diff] = sum(1 for d in dsubset if d["correct"]) / len(dsubset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="retrieval_ablation",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite O: Timestamp Integrity (Omni-SimpleMem) ---

def run_timestamp_integrity(backend: BenchmarkableStore, scenarios: list,
                            judge: MemoryJudge) -> CategoryResult:
    """Run timestamp integrity scenarios (Suite O).

    Based on Omni-SimpleMem findings that consolidation and compression operations
    can corrupt temporal metadata, causing older facts to outrank newer ones.

    Each scenario stores facts at staggered simulated ages (oldest first), optionally
    advances time and/or runs consolidation, then checks that temporal ordering is
    preserved and the expected fact (usually the most recent) is correctly recalled.

    Facts use the same stored_days_ago convention as Suite A temporal_decay tests so
    the two suites can be compared directly.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Store facts oldest-first so simulated time advances forward naturally
        sorted_facts = sorted(sc["facts"], key=lambda f: -f["stored_days_ago"])
        prev_days_ago = sorted_facts[0]["stored_days_ago"] if sorted_facts else 0

        for fact in sorted_facts:
            time_gap = prev_days_ago - fact["stored_days_ago"]
            if time_gap > 0:
                backend.simulate_time(time_gap)
            prev_days_ago = fact["stored_days_ago"]
            backend.store(fact["content"], category="factual")

        # Advance remaining time to reach the scenario's "now"
        if prev_days_ago > 0:
            backend.simulate_time(prev_days_ago)

        # Optional additional time advance after all facts are stored
        extra_advance = sc.get("time_advance_days", 0)
        if extra_advance > 0:
            backend.simulate_time(extra_advance)

        # Optional consolidation cycle — this is where timestamp corruption is most likely
        if sc.get("run_consolidation", False):
            backend.consolidate()

        results = backend.recall(sc["query"], top_k=5)
        # For hard scenarios pass top-2 — consolidation may produce merged facts
        if sc.get("difficulty") == "hard" and len(results) > 1:
            actual = " | ".join(results[:2])
        else:
            actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "run_consolidation": sc.get("run_consolidation", False),
            "time_advance_days": sc.get("time_advance_days", 0),
            "num_facts": len(sc["facts"]),
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        dsubset = [d for d in details if d["difficulty"] == diff]
        if dsubset:
            sub_scores[diff] = sum(1 for d in dsubset if d["correct"]) / len(dsubset)

    # Sub-score: with vs without consolidation (reveals timestamp corruption risk)
    with_consol = [d for d in details if d["run_consolidation"]]
    without_consol = [d for d in details if not d["run_consolidation"]]
    if with_consol:
        sub_scores["with_consolidation"] = (
            sum(1 for d in with_consol if d["correct"]) / len(with_consol)
        )
    if without_consol:
        sub_scores["without_consolidation"] = (
            sum(1 for d in without_consol if d["correct"]) / len(without_consol)
        )

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="timestamp_integrity",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite J: Topic Shift Recall ---

def run_topic_shift_recall(backend: BenchmarkableStore, scenarios: list,
                           judge: MemoryJudge) -> CategoryResult:
    """Run topic shift recall scenarios (Suite J).

    Stores facts from topic A, then topic B (simulating a conversation pivot),
    then queries about topic B. Tests whether the system retrieves the correct
    context after a topic shift. Forbidden terms from topic A must not appear.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Support reversed store order for recency-trap scenarios
        # (store topic B first, then topic A — query asks about B,
        # testing whether backend returns correct older facts vs recent wrong ones)
        if sc.get("store_order") == "reversed":
            for fact in sc["topic_b_facts"]:
                backend.store(fact, category="factual")
            for fact in sc["topic_a_facts"]:
                backend.store(fact, category="factual")
        else:
            for fact in sc["topic_a_facts"]:
                backend.store(fact, category="factual")
            for fact in sc["topic_b_facts"]:
                backend.store(fact, category="factual")

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        scenario_correct = jr.correct

        # Check forbidden terms — if topic A content leaks into the answer, fail
        if scenario_correct and sc.get("forbidden_terms"):
            for term in sc["forbidden_terms"]:
                if term.lower() in actual.lower():
                    scenario_correct = False
                    break

        if scenario_correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc.get("difficulty", "medium"),
            "correct": scenario_correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        dsubset = [d for d in details if d["difficulty"] == diff]
        if dsubset:
            sub_scores[diff] = sum(1 for d in dsubset if d["correct"]) / len(dsubset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="topic_shift_recall",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite K: Compression Survival ---

def run_compression_survival(backend: BenchmarkableStore, scenarios: list,
                             judge: MemoryJudge) -> CategoryResult:
    """Run compression survival scenarios (Suite K).

    Stores a compressed summary (simulating context window compression),
    then stores recent noise facts. Queries for a critical detail from
    the summary. Tests whether important facts survive compression + noise.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        backend.store(sc["compressed_summary"], category="factual")
        for noise in sc["recent_noise"]:
            backend.store(noise, category="factual")

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc.get("difficulty", "medium"),
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        dsubset = [d for d in details if d["difficulty"] == diff]
        if dsubset:
            sub_scores[diff] = sum(1 for d in dsubset if d["correct"]) / len(dsubset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="compression_survival",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


# --- Suite L: Delegation Memory ---

def run_delegation_memory(backend: BenchmarkableStore, scenarios: list,
                          judge: MemoryJudge) -> CategoryResult:
    """Run delegation memory scenarios (Suite L).

    Stores a delegation task and its result (simulating a child agent
    returning findings). Queries for the outcome. Tests whether delegated
    work is recallable.
    """
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        backend.store(sc["delegation_task"], category="factual")
        backend.store(sc["delegation_result"], category="factual")

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc.get("difficulty", "medium"),
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })
        scenario_metrics = compute_scenario_metrics(results, sc["gold_answer"])
        details[-1]["metrics"] = scenario_metrics

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        dsubset = [d for d in details if d["difficulty"] == diff]
        if dsubset:
            sub_scores[diff] = sum(1 for d in dsubset if d["correct"]) / len(dsubset)

    all_metrics = [d.get("metrics", {}) for d in details if "metrics" in d]
    avg_retrieval_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m]
            avg_retrieval_metrics[key] = sum(values) / len(values) if values else 0.0

    return CategoryResult(
        category="delegation_memory",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
        retrieval_metrics=avg_retrieval_metrics,
    )


CATEGORY_RUNNERS = {
    "semantic_recall": run_semantic_recall,
    "contradictions": run_contradictions,
    "temporal_decay": run_temporal_decay,
    "cross_reference": run_cross_reference,
    "importance_filtering": run_importance_filtering,
    # Suite B
    "consolidation": run_consolidation,
    "compression": run_compression,
    # Suite C
    "scopes": run_scopes,
    # Suite D
    "adversarial": run_adversarial,
    # Suite E
    "scale": run_scale,
    # Suite F
    "integration": run_integration,
    # Suite G
    "qlearning": run_qlearning,
    # Suite H — Advanced Memory Features
    "supersession": run_supersession,  

    "typed_decay": run_typed_decay,
    "scope_lifecycle": run_scope_lifecycle,
    "notation_parsing": run_notation_parsing,
    "deduplication": run_deduplication,
    # Suite I — Conversation & Stress
    "conversation_memory": run_conversation_memory,
    "capacity_stress": run_capacity_stress,
    # Suite M — Format Sensitivity (Omni-SimpleMem)
    "format_sensitivity": run_format_sensitivity,
    # Suite N — Retrieval Ablation (Omni-SimpleMem)
    "retrieval_ablation": run_retrieval_ablation,
    # Suite O — Timestamp Integrity (Omni-SimpleMem)
    "timestamp_integrity": run_timestamp_integrity,
    # Suite J — Topic Shift Recall
    "topic_shift_recall": run_topic_shift_recall,
    # Suite K — Compression Survival
    "compression_survival": run_compression_survival,
    # Suite L — Delegation Memory
    "delegation_memory": run_delegation_memory,
}


# --- Main Run Logic ---

def run_single(config: BenchmarkConfig, seed: int) -> RunResult:
    """Execute one full benchmark run with a given seed.
    
    The seed controls scenario shuffling to measure variance from
    insertion/query order effects. This is critical for detecting
    order-dependent bugs (e.g., TF-IDF vocab growth, Hebbian link
    formation path-dependence).
    """
    import random
    import gc
    random.seed(seed)

    start = time.time()
    backend = get_backend(config.backend_name, config)

    # Use HeuristicJudge by default; LLM judge for real results
    if config.judge_model == "heuristic":
        judge = HeuristicJudge(model="heuristic")
    else:
        judge = MemoryJudge(model=config.judge_model)

    results_by_cat = {}

    suites_to_run = config.parameters.get("suites", ["a"])
    if suites_to_run == ["all"] or suites_to_run == "all":
        # Auto-discover all available suites
        suites_to_run = sorted(
            d.name.split("_")[1]
            for d in SUITE_DIR.iterdir()
            if d.is_dir() and d.name.startswith("suite_") and (d / "fixtures").exists()
        )
    for suite_letter in suites_to_run:
        try:
            fixtures = load_fixtures(suite_letter)
        except FileNotFoundError:
            continue
        for category_name, scenarios in fixtures.items():
            runner = CATEGORY_RUNNERS.get(category_name)
            if runner:
                # Skip categories the backend doesn't support
                caps = BACKEND_CAPABILITIES.get(config.backend_name)
                if caps is not None:
                    from benchmarks.tracks import backend_supports_category, missing_capabilities
                    if not backend_supports_category(caps, category_name):
                        missing = missing_capabilities(caps, category_name)
                        print(f"  Skipping {category_name} (missing: {', '.join(missing)})")
                        continue
                # Shuffle scenarios to measure order-dependence
                shuffled = list(scenarios)
                random.shuffle(shuffled)
                try:
                    # Pass capabilities to runners that accept it
                    import inspect
                    sig = inspect.signature(runner)
                    if len(sig.parameters) >= 4:
                        cat_result = runner(backend, shuffled, judge, caps)
                    else:
                        cat_result = runner(backend, shuffled, judge)
                except Exception as exc:
                    print(f"  ERROR in {category_name}: {exc}", file=sys.stderr)
                    cat_result = CategoryResult(
                        category=category_name,
                        total=len(shuffled),
                        correct=0,
                        score=0.0,
                        details=[{"error": str(exc)}],
                    )
                results_by_cat[category_name] = cat_result

                # Free transient memory between categories to prevent OOM
                # in constrained Docker containers.  The backend.reset()
                # inside each scenario already clears the store; this just
                # reclaims Python garbage (embedding vectors, scored lists).
                gc.collect()

    elapsed = time.time() - start

    # Compute overall score (weighted average)
    total_correct = sum(c.correct for c in results_by_cat.values())
    total_items = sum(c.total for c in results_by_cat.values())
    overall = total_correct / total_items if total_items > 0 else 0

    # Aggregate token usage across categories
    total_recall_tokens = sum(c.recall_tokens for c in results_by_cat.values())
    total_recall_chars = sum(c.recall_chars for c in results_by_cat.values())
    num_queries = total_items

    # Aggregate retrieval metrics across categories
    all_cat_metrics = {}
    metric_counts = {}
    for cat_name, cat_result in results_by_cat.items():
        for metric_name, value in cat_result.retrieval_metrics.items():
            if metric_name not in all_cat_metrics:
                all_cat_metrics[metric_name] = 0.0
                metric_counts[metric_name] = 0
            all_cat_metrics[metric_name] += value
            metric_counts[metric_name] += 1

    avg_metrics = {k: v / metric_counts[k] for k, v in all_cat_metrics.items()}

    run_result = RunResult(
        seed=seed,
        results_by_category=results_by_cat,
        overall_score=overall,
        token_usage={
            "recall_tokens": total_recall_tokens,
            "recall_chars": total_recall_chars,
            "recall_queries": num_queries,
            "avg_recall_tokens_per_query": total_recall_tokens // max(num_queries, 1),
        },
        wall_time_seconds=elapsed,
    )
    run_result.retrieval_metrics = avg_metrics
    run_result.cost_metrics = compute_cost_metrics(
        total_tokens=total_recall_tokens,
        total_queries=num_queries,
        correct=total_correct,
        total=total_items,
    )
    return run_result


def run_benchmark(config: BenchmarkConfig) -> tuple:
    """Run the full benchmark suite with multiple seeds and aggregate.
    Returns (AggregateResult, list[RunResult]) for comparison.
    """
    runs = []
    for seed in config.seeds[:config.num_runs]:
        print(f"  Run seed={seed}...", end=" ", flush=True)
        result = run_single(config, seed)
        runs.append(result)
        print(f"score={result.overall_score:.3f} ({result.wall_time_seconds:.1f}s)")

    return aggregate_results(runs), runs


def print_results(agg: AggregateResult, config: BenchmarkConfig,
                   runs: Optional[list] = None):
    """Print a summary table to stdout."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS: {config.backend_name}")
    print(f"{'='*60}")
    print(f"  Runs: {agg.num_runs}")
    print(f"  Overall: {agg.mean_score:.3f} ± {agg.std_score:.3f}")
    print(f"  95% CI:  [{agg.ci_95_lower:.3f}, {agg.ci_95_upper:.3f}]")
    print(f"{'─'*60}")
    print(f"  {'Category':<25} {'Mean':>8} {'Std':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8}")
    for cat, mean in sorted(agg.per_category_mean.items()):
        std = agg.per_category_std.get(cat, 0)
        print(f"  {cat:<25} {mean:>8.3f} {std:>8.3f}")
    print(f"{'─'*60}")
    # Token usage summary
    if runs:
        avg_tokens = sum(
            r.token_usage.get("avg_recall_tokens_per_query", 0) for r in runs
        ) // len(runs)
        total_tokens = sum(r.token_usage.get("recall_tokens", 0) for r in runs) // len(runs)
        total_queries = sum(r.token_usage.get("recall_queries", 0) for r in runs) // len(runs)
        print(f"  Token cost (avg per run):")
        print(f"    Recall tokens/query:  ~{avg_tokens}")
        print(f"    Total recall tokens:  ~{total_tokens} ({total_queries} queries)")
    # Retrieval metrics summary
    if runs:
        all_run_metrics = [r.retrieval_metrics for r in runs if hasattr(r, "retrieval_metrics") and r.retrieval_metrics]
        if all_run_metrics:
            avg_rm = {}
            for key in all_run_metrics[0]:
                values = [m[key] for m in all_run_metrics if key in m]
                avg_rm[key] = sum(values) / len(values) if values else 0.0
            print(f"{'─'*60}")
            print(f"  Retrieval Metrics (averaged):")
            print(f"    Recall@1:  {avg_rm.get('recall_at_1', 0):.3f}")
            print(f"    Recall@5:  {avg_rm.get('recall_at_5', 0):.3f}")
            print(f"    MRR:       {avg_rm.get('mrr', 0):.3f}")
            print(f"    Token F1:  {avg_rm.get('token_f1', 0):.3f}")
    # Cost efficiency metrics
    if runs:
        all_cost_metrics = [r.cost_metrics for r in runs if hasattr(r, "cost_metrics") and r.cost_metrics]
        if all_cost_metrics:
            avg_cm = {}
            for key in all_cost_metrics[0]:
                values = [m[key] for m in all_cost_metrics if key in m]
                avg_cm[key] = sum(values) / len(values) if values else 0.0
            print(f"{'─'*60}")
            print(f"  Cost Efficiency:")
            tpq = avg_cm.get('tokens_per_query', 0)
            tpc = avg_cm.get('tokens_per_correct', 0)
            eff = avg_cm.get('cost_efficiency', 0)
            print(f"    Tokens/query:    ~{tpq:.0f}")
            print(f"    Tokens/correct:  ~{tpc:.0f}")
            print(f"    Efficiency:      {eff:.3f} (score / log2(tokens))")
    print(f"{'='*60}\n")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Memory Benchmark Runner — system-agnostic AI agent memory evaluation"
    )
    parser.add_argument("--backend", default="baseline-flat",
                        choices=list(BACKENDS.keys()),
                        help="Memory backend to benchmark")
    parser.add_argument("--profile", default="balanced",
                        help="Config profile for cognitive backend")
    parser.add_argument("--embedding", default="auto",
                        help="Embedding model: auto, sentence-transformers, tfidf")
    parser.add_argument("--suite", default="a",
                        help="Suite(s) to run: a,b,c,d,e,f or 'all'")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per benchmark")
    parser.add_argument("--judge-model", default="heuristic",
                        help="Model for LLM-as-judge (default: heuristic, use claude-haiku-4-5 for LLM judge)")
    parser.add_argument("--output-dir", default="benchmarks/results/",
                        help="Directory for JSON results")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46],
                        help="Random seeds for runs")
    parser.add_argument("--contradiction-llm", default=None,
                        help="LLM model for contradiction fallback (e.g., claude-haiku-4-5)")
    parser.add_argument("--compare", default=None,
                        help="Compare against another backend (runs both)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    # Parse suite argument: 'a' -> ['a'], 'a,b,c' -> ['a','b','c'], 'all' -> all
    if args.suite == "all":
        suites = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
    else:
        suites = [s.strip() for s in args.suite.split(",")]

    config = BenchmarkConfig(
        backend_name=args.backend,
        profile=args.profile,
        embedding_model=args.embedding,
        num_runs=args.runs,
        judge_model=args.judge_model,
        output_path=args.output_dir,
        seeds=args.seeds,
        parameters={
            "profile": args.profile,
            "embedding_model": args.embedding,
            "suites": suites,
            **({"contradiction_llm_model": args.contradiction_llm}
               if args.contradiction_llm else {}),
        },
    )

    print(f"\nRunning {config.backend_name} benchmark ({config.num_runs} runs)...")
    try:
        agg, runs = run_benchmark(config)
    except Exception as exc:
        print(f"\nBenchmark failed: {exc}", file=sys.stderr)
        print(f"Backend '{config.backend_name}' may require additional setup "
              f"(API keys, running services, etc.).", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # Compute avg retrieval metrics across runs for JSON output
        all_run_rm = [r.retrieval_metrics for r in runs if hasattr(r, "retrieval_metrics") and r.retrieval_metrics]
        avg_retrieval_metrics_json = {}
        if all_run_rm:
            for key in all_run_rm[0]:
                vals = [m[key] for m in all_run_rm if key in m]
                avg_retrieval_metrics_json[key] = sum(vals) / len(vals) if vals else 0.0
        # Compute avg cost metrics across runs for JSON output
        all_run_cm = [r.cost_metrics for r in runs if hasattr(r, "cost_metrics") and r.cost_metrics]
        avg_cost_metrics_json = {}
        if all_run_cm:
            for key in all_run_cm[0]:
                vals = [m[key] for m in all_run_cm if key in m]
                avg_cost_metrics_json[key] = sum(vals) / len(vals) if vals else 0.0
        results_dict = {
            "backend": config.backend_name,
            "mean_score": agg.mean_score,
            "std": agg.std_score,
            "ci_95": [agg.ci_95_lower, agg.ci_95_upper],
            "per_category": agg.per_category_mean,
            "num_runs": agg.num_runs,
            "retrieval_metrics": avg_retrieval_metrics_json,
            "cost_metrics": avg_cost_metrics_json,
        }
        print(json.dumps(results_dict, indent=2))
    else:
        print_results(agg, config, runs)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{config.backend_name}.json"
    # Compute avg retrieval metrics for file output (may already be done above)
    all_run_rm_file = [r.retrieval_metrics for r in runs if hasattr(r, "retrieval_metrics") and r.retrieval_metrics]
    avg_rm_file = {}
    if all_run_rm_file:
        for key in all_run_rm_file[0]:
            vals = [m[key] for m in all_run_rm_file if key in m]
            avg_rm_file[key] = sum(vals) / len(vals) if vals else 0.0
    # Compute avg cost metrics for file output
    all_run_cm_file = [r.cost_metrics for r in runs if hasattr(r, "cost_metrics") and r.cost_metrics]
    avg_cm_file = {}
    if all_run_cm_file:
        for key in all_run_cm_file[0]:
            vals = [m[key] for m in all_run_cm_file if key in m]
            avg_cm_file[key] = sum(vals) / len(vals) if vals else 0.0

    # Build result data
    import datetime
    result_data = {
        "backend": config.backend_name,
        "profile": config.profile,
        "embedding_model": config.embedding_model,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "suites": config.parameters.get("suites", ["a"]),
        "mean_score": agg.mean_score,
        "std": agg.std_score,
        "ci_95": [agg.ci_95_lower, agg.ci_95_upper],
        "per_category_mean": agg.per_category_mean,
        "per_category_std": agg.per_category_std,
        "num_runs": agg.num_runs,
        "retrieval_metrics": avg_rm_file,
        "cost_metrics": avg_cm_file,
        "runs": [
                {
                    "seed": r.seed,
                    "overall_score": r.overall_score,
                    "wall_time_seconds": r.wall_time_seconds,
                    "token_usage": r.token_usage,
                    "retrieval_metrics": getattr(r, "retrieval_metrics", {}),
                    "cost_metrics": getattr(r, "cost_metrics", {}),
                    "categories": {
                        cat: {
                            "score": cr.score, "correct": cr.correct, "total": cr.total,
                            "recall_tokens": cr.recall_tokens, "recall_chars": cr.recall_chars,
                            "retrieval_metrics": cr.retrieval_metrics,
                        }
                        for cat, cr in r.results_by_category.items()
                    },
                }
                for r in runs
            ],
        }

    # Merge with existing results (preserve categories from prior runs)
    if result_file.exists():
        try:
            with open(result_file) as f:
                existing = json.load(f)
            # Merge per_category_mean: keep old categories, update/add new ones
            old_cats = existing.get("per_category_mean", {})
            new_cats = result_data.get("per_category_mean", {})
            merged_cats = {**old_cats, **new_cats}
            result_data["per_category_mean"] = merged_cats
            # Same for per_category_std
            old_std = existing.get("per_category_std", {})
            new_std = result_data.get("per_category_std", {})
            result_data["per_category_std"] = {**old_std, **new_std}
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted file, just overwrite

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    # Also save timestamped copy for history tracking
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / f"{config.backend_name}_{ts}.json"
    with open(history_file, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"  Results saved to {result_file}")
    print(f"  History saved to {history_file}")

    if args.compare:
        print(f"\nRunning comparison: {args.compare}...")
        config2 = BenchmarkConfig(
            backend_name=args.compare,
            profile=args.profile,
            embedding_model=args.embedding,
            num_runs=args.runs,
            judge_model=args.judge_model,
            output_path=args.output_dir,
            seeds=args.seeds,
            parameters={"profile": args.profile, "embedding_model": args.embedding},
        )
        agg2, runs2 = run_benchmark(config2)
        print_results(agg2, config2, runs2)

        # Save comparison results
        result_file2 = output_dir / f"{config2.backend_name}.json"
        with open(result_file2, "w") as f:
            json.dump({
                "backend": config2.backend_name,
                "mean_score": agg2.mean_score,
                "std": agg2.std_score,
                "num_runs": agg2.num_runs,
                "per_category_mean": agg2.per_category_mean,
            }, f, indent=2)

        # Significance test
        sig = compare_runs(runs, runs2)
        print(f"\n{'='*60}")
        print(f"  STATISTICAL COMPARISON")
        print(f"{'='*60}")
        print(f"  {config.backend_name}: {sig.baseline_mean:.3f}")
        print(f"  {config2.backend_name}: {sig.experiment_mean:.3f}")
        print(f"  Improvement: {sig.improvement:+.1f} pp")
        print(f"  Effect size (Cohen's d): {sig.effect_size:.3f}")
        print(f"  p-value: {sig.p_value:.4f}")
        print(f"  Significant (p<0.05): {'YES' if sig.significant else 'NO'}")
        print(f"  Test: {sig.test_name}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
