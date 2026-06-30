"""Evaluation framework for memory retrieval relevance.

Runs a golden dataset of query→expected-relevant-entries pairs through the
scored pipeline and computes standard IR metrics: MAP@K, MRR, Recall@K.

Usage:
    python3 -m agent.memory.evaluate [--golden PATH] [--weights FTS,EMB,CTX,SIG]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def average_precision_at_k(
    relevant: List[int],
    ranked: List[int],
    k: int,
) -> float:
    """Average Precision at K.

    AP@K = (1/min(K, |relevant|)) * sum_{i=1..K} P@i * rel(i)

    where P@i is precision at cutoff i, and rel(i) = 1 if ranked[i] is relevant.
    """
    if not relevant or not ranked:
        return 0.0

    k = min(k, len(ranked))
    relevant_set = set(relevant)
    hits = 0
    sum_precision = 0.0

    for i in range(k):
        if ranked[i] in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)

    min_rel = min(k, len(relevant))
    return sum_precision / min_rel if min_rel > 0 else 0.0


def mean_reciprocal_rank(
    relevant: List[int],
    ranked: List[int],
) -> float:
    """Mean Reciprocal Rank — position of first relevant result.

    Returns 1/rank if found, 0 otherwise.
    """
    if not relevant or not ranked:
        return 0.0

    relevant_set = set(relevant)
    for i, rank in enumerate(ranked):
        if rank in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    relevant: List[int],
    ranked: List[int],
    k: int,
) -> float:
    """Recall at K — fraction of relevant entries found in top K."""
    if not relevant:
        return 1.0  # vacuously true — no relevant entries means nothing to recall
    if not ranked:
        return 0.0

    relevant_set = set(relevant)
    found = sum(1 for r in ranked[:k] if r in relevant_set)
    return found / len(relevant)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def load_golden(path: str) -> List[Dict]:
    """Load the golden dataset from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    logger.info("Loaded %d golden queries from %s", len(data), path)
    return data


def load_corpus() -> Tuple[List[str], str]:
    """Load the actual memory entries from MEMORY.md as the test corpus.

    Returns (entries_list, source_description).
    """
    from hermes_constants import get_hermes_home

    mem_path = get_hermes_home() / "memories" / "MEMORY.md"
    if not mem_path.exists():
        logger.warning("MEMORY.md not found at %s — using fallback test corpus", mem_path)
        return _fallback_corpus()

    with open(mem_path, "r", encoding="utf-8") as f:
        raw = f.read()

    entries = [e.strip() for e in raw.split("\n§\n") if e.strip()]
    logger.info("Loaded %d memory entries from %s", len(entries), mem_path)
    return entries, str(mem_path)


def _fallback_corpus() -> Tuple[List[str], str]:
    """Return a hardcoded test corpus when MEMORY.md isn't available."""
    entries = [
        "[context: deploy] [signal: 0.8]API: api.xentropy.ai (GKE). Mobile: /root/xentropy/client/apps/mobile/ — Expo 54. EXPO_TOKEN in GCP SM.",
        "[context: deploy] [signal: 0.8]GKE Hermes parity complete 2026-06-29: Dockerfile + gcloud CLI, K8s manifests (gateway+dashboard+kanban-daemon), WLI SA + init container GCS sync for skills/scripts/plugins/cron. VPS→GCS cron every 5min. Image: hermes-agent:v1 in europe-central2 Artifact Registry.",
        "[context: feature] [signal: 0.9]Memory context relevance framework: skill at ~/.hermes/skills/memory/context-relevance-framework/SKILL.md. 5-stage pipeline: Intent extraction → Multi-provider hybrid recall (FTS5+embedding+context-tag) → Cross-provider fusion (dedup+trust weighting) → Relevance gating (min_score+window budget) → Structured injection.",
        "[context: deploy] [signal: 0.8]Container naming migration DONE 2026-06-30. 13 GKE deployments updated to canonical images. 4 orphan repos deleted. Gitops committed.",
        "[context: feature] [signal: 0.9]Storage service at xentropy/ai/apps/storage/ — NestJS, Neo4j+GCS, GKE port 3006. Neo4j nodes: (:Tenant,:Workspace,:File,:IdempotencyKey). GCS: {project}-workspaces + {project}-assets.",
        "[context: feature] [signal: 0.9]Memory context relevance framework IMPLEMENTED. 5-stage pipeline: Intent extraction → FTS5+ONNX-embedding+context-tag hybrid recall → Cross-provider fusion → Relevance gating → Structured injection. all-MiniLM-L6-v2 via ONNX Runtime (22MB, no PyTorch). 162 tests pass.",
    ]
    return entries, "fallback corpus"


def run_evaluation(
    golden: List[Dict],
    entries: List[str],
    weights: Optional[Dict[str, float]] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run the golden dataset through the scored pipeline and compute metrics.

    Args:
        golden: List of golden query dicts.
        entries: The memory entry corpus to search against.
        weights: Optional override for HYBRID_WEIGHTS.
        k_values: List of K values for MAP@K / Recall@K (default [3, 5, 10]).

    Returns:
        Dict with per-query results and aggregate metrics.
    """
    if k_values is None:
        k_values = [3, 5, 10]

    from tools.memory_tool import MemoryStore
    from agent.memory.intent_extractor import extract_intent

    per_query: List[Dict] = []
    map_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    mrr_scores: List[float] = []
    recall_scores: Dict[int, List[float]] = {k: [] for k in k_values}
    task_accuracy: Dict[str, List[bool]] = {}

    for i, item in enumerate(golden):
        query = item["query"]
        expected_task = item.get("task_type")
        relevant = item["relevant_indices"]

        # Stage 1: Intent extraction
        intent = extract_intent(query)
        task_match = intent.task_type == expected_task if expected_task else True

        if expected_task:
            task_accuracy.setdefault(expected_task, []).append(task_match)

        # Stage 2-4: Scored prefetch
        scored = MemoryStore.scored_prefetch(
            entries,
            task_type=intent.task_type,
            task_confidence=intent.task_confidence,
            keywords=intent.keywords,
            expanded_query=intent.expanded_query,
            fts_scores=None,
            embedding_scores=None,
        )

        # Ranked entry indices
        ranked_indices = [s["entry_index"] for s in scored if s.get("entry_index") is not None]

        # Compute metrics
        result = {
            "query_index": i,
            "query": query[:60],
            "task_type": intent.task_type,
            "task_confidence": round(intent.task_confidence, 2),
            "task_match": task_match,
            "n_candidates": len(scored),
            "relevant_count": len(relevant),
            "ranked_indices": ranked_indices[:10],
            "scores": [round(s["score"], 3) for s in scored[:10]],
        }

        for k in k_values:
            apk = average_precision_at_k(relevant, ranked_indices, k)
            rk = recall_at_k(relevant, ranked_indices, k)
            map_scores[k].append(apk)
            recall_scores[k].append(rk)
            result[f"ap@{k}"] = round(apk, 4)
            result[f"recall@{k}"] = round(rk, 4)

        mrr = mean_reciprocal_rank(relevant, ranked_indices)
        mrr_scores.append(mrr)
        result["mrr"] = round(mrr, 4)

        per_query.append(result)

    # Aggregate
    aggregate: Dict[str, Any] = {
        "n_queries": len(golden),
        "n_entries": len(entries),
        "map": {},
        "recall": {},
        "mrr_avg": round(sum(mrr_scores) / max(len(mrr_scores), 1), 4),
        "task_accuracy": {},
    }

    for k in k_values:
        scores = map_scores[k]
        aggregate["map"][f"map@{k}"] = round(sum(scores) / max(len(scores), 1), 4)
        aggregate["recall"][f"recall@{k}"] = round(
            sum(recall_scores[k]) / max(len(recall_scores[k]), 1), 4
        )

    for task, matches in task_accuracy.items():
        accuracy = sum(matches) / len(matches) * 100
        aggregate["task_accuracy"][task] = round(accuracy, 1)

    return {
        "aggregate": aggregate,
        "per_query": per_query,
    }


def format_report(results: Dict[str, Any]) -> str:
    """Format evaluation results as a human-readable report."""
    agg = results["aggregate"]
    per_query = results["per_query"]

    lines = [
        "=" * 50,
        "Memory Relevance Evaluation Report",
        "=" * 50,
        f"Queries:     {agg['n_queries']}",
        f"Corpus:      {agg['n_entries']} entries",
        f"MRR:         {agg['mrr_avg']:.4f}",
        "",
        "MAP@K:",
    ]

    for k, v in agg["map"].items():
        lines.append(f"  {k:8s}: {v:.4f}")

    lines.append("")
    lines.append("Recall@K:")
    for k, v in agg["recall"].items():
        lines.append(f"  {k:9s}: {v:.4f}")

    if agg.get("task_accuracy"):
        lines.append("")
        lines.append("Task classification accuracy:")
        for task, pct in sorted(agg["task_accuracy"].items()):
            lines.append(f"  {task:12s}: {pct:.0f}%")

    lines.append("")
    lines.append("Per-query summary (first 3 failures / all):")
    failures = [q for q in per_query if q.get("ap@5", 1.0) < 0.5 and q["relevant_count"] > 0]
    for q in failures[:3]:
        lines.append(f"  ✗ [{q['query_index']}] {q['query']}")
        lines.append(f"      task={q['task_type']} "
                      f"AP@5={q.get('ap@5', 0):.3f} "
                      f"matched={q['relevant_count']} "
                      f"candidates={q['n_candidates']}")

    successes = [q for q in per_query if q.get("ap@5", 0) >= 0.5 or q["relevant_count"] == 0]
    lines.append("")
    lines.append(f"Passing queries: {len(successes)}/{len(per_query)}")

    return "\n".join(lines)


def format_json_report(results: Dict[str, Any], path: str) -> None:
    """Write JSON-format evaluation report to file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("JSON report written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory relevance evaluation")
    parser.add_argument(
        "--golden", default=None,
        help="Path to golden dataset JSON (default: tests/golden/memory_relevance_golden.json)",
    )
    parser.add_argument(
        "--json", default=None,
        help="Write detailed JSON report to file",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Override hybrid weights: FTS,EMB,CTX,SIG (e.g. 0.25,0.45,0.20,0.10)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-query details",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Load golden dataset
    golden_path = args.golden or (
        Path(__file__).parent.parent.parent / "tests" / "golden" / "memory_relevance_golden.json"
    )
    golden = load_golden(str(golden_path))

    # Load corpus (real memory entries)
    entries, src = load_corpus()
    print(f"Corpus: {len(entries)} entries from {src}")

    # Parse custom weights
    weights = None
    if args.weights:
        parts = [float(x) for x in args.weights.split(",")]
        if len(parts) == 4:
            from agent.memory.scored_memory import HYBRID_WEIGHTS
            weights = dict(HYBRID_WEIGHTS)
            weights["fts"] = parts[0]
            weights["embedding"] = parts[1]
            weights["context_tag"] = parts[2]
            weights["signal"] = parts[3]
            print(f"Using custom weights: FTS={parts[0]} EMB={parts[1]} CTX={parts[2]} SIG={parts[3]}")

    # Run evaluation
    results = run_evaluation(golden, entries, weights=weights)

    # Report
    print()
    print(format_report(results))

    if args.json:
        format_json_report(results, args.json)

    if args.verbose:
        print()
        print("-" * 50)
        print("Per-query details:")
        print("-" * 50)
        for q in results["per_query"]:
            print(f"\n[{q['query_index']}] q='{q['query']}'")
            print(f"    task={q['task_type']} conf={q['task_confidence']}"
                  f"  AP@5={q.get('ap@5', 0):.3f}  MRR={q['mrr']:.3f}")
            print(f"    ranked: {q['ranked_indices'][:5]}")
            print(f"    scores: {q['scores'][:5]}")


if __name__ == "__main__":
    main()
