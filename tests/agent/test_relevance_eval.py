"""CI gate for memory relevance: enforces metric floors on the golden dataset.

Fails when any metric drops below its threshold.
Run: pytest tests/agent/test_relevance_eval.py -v
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from tools.memory_tool import MemoryStore

# Metric floor thresholds — failing these blocks merges.
# These are set conservatively above the current baseline so the gate
# catches regressions before they reach production.
METRIC_FLOORS = {
    "mrr_avg": 0.45,         # first relevant result within top 3
    "map@3": 0.40,           # AP at top 3
    "map@5": 0.45,           # AP at top 5
    "recall@3": 0.80,        # 80% of relevant entries found in top 3
    "recall@5": 0.90,        # 90% in top 5
}

GOLDEN_PATH = Path(__file__).parent.parent / "golden" / "memory_relevance_golden.json"


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _load_corpus():
    """Load actual memory entries for realistic evaluation."""
    from hermes_constants import get_hermes_home
    mem_path = get_hermes_home() / "memories" / "MEMORY.md"
    if mem_path.exists():
        raw = mem_path.read_text(encoding="utf-8")
        entries = [e.strip() for e in raw.split("\n§\n") if e.strip()]
        if entries:
            return entries
    # Fallback: hardcoded test corpus matching golden dataset indices
    return [
        "[context: deploy] [signal: 0.8]API: api.xentropy.ai (GKE). Mobile: /root/xentropy/client/apps/mobile/ — Expo 54. EXPO_TOKEN in GCP SM.",
        "[context: deploy] [signal: 0.8]GKE Hermes parity complete 2026-06-29: Dockerfile + gcloud CLI, K8s manifests (gateway+dashboard+kanban-daemon), WLI SA + init container GCS sync for skills/scripts/plugins/cron. VPS→GCS cron every 5min. Image: hermes-agent:v1 in europe-central2 Artifact Registry.",
        "[context: feature] [signal: 0.8]Memory context relevance framework...",
        "[context: deploy] [signal: 0.8]Container naming migration DONE 2026-06-30. 13 GKE deployments updated to canonical images. 4 orphan repos deleted. Gitops committed.",
        "[context: feature] [signal: 0.9]Storage service at xentropy/ai/apps/storage/ — NestJS, Neo4j+GCS, GKE port 3006...",
        "[context: feature] [signal: 0.9]Memory context relevance framework fully IMPLEMENTED. all-MiniLM-L6-v2 via ONNX Runtime (22MB, no PyTorch). 162 tests pass...",
    ]


def test_relevance_metrics() -> None:
    """Golden dataset evaluation — assert all metric floors hold."""
    from agent.memory.evaluate import run_evaluation
    from agent.memory.intent_extractor import extract_intent

    golden = _load_golden()
    entries = _load_corpus()

    assert len(golden) >= 20, f"Golden dataset too small: {len(golden)}"
    assert len(entries) >= 4, f"Corpus too small: {len(entries)}"

    results = run_evaluation(golden, entries)
    agg = results["aggregate"]
    per_query = results["per_query"]

    # Log per-query failures for debugging
    failures = [
        q for q in per_query
        if q.get("ap@5", 1.0) < 0.3 and q["relevant_count"] > 0
    ]
    if failures:
        print(f"\n⚠ Low-scoring queries ({len(failures)}):")
        for q in failures[:5]:
            print(f"  [{q['query_index']}] '{q['query']}' AP@5={q.get('ap@5', 0):.3f} "
                  f"task={q['task_type']} relevant={q['relevant_count']}")

    # Assert metric floors
    mrr = agg["mrr_avg"]
    assert mrr >= METRIC_FLOORS["mrr_avg"], (
        f"MRR {mrr:.4f} < floor {METRIC_FLOORS['mrr_avg']:.2f}"
    )

    for metric, floor in METRIC_FLOORS.items():
        if metric == "mrr_avg":
            continue
        # metric is like "map@3" or "recall@5"
        category, k = metric.split("@")
        k = int(k)
        if category == "map":
            value = agg["map"].get(f"map@{k}", 0)
        elif category == "recall":
            value = agg["recall"].get(f"recall@{k}", 0)
        else:
            continue
        assert value >= floor, (
            f"{metric} {value:.4f} < floor {floor:.2f}"
        )

    print(f"\n✅ All metric floors passed ({len(per_query)} queries, {len(entries)} entries)")
    print(f"   MRR={mrr:.4f}  MAP@5={agg['map']['map@5']:.4f}  Recall@5={agg['recall']['recall@5']:.4f}")
