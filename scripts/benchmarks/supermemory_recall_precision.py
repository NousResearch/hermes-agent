#!/usr/bin/env python3
"""Offline precision benchmark for the current Supermemory prefetch path.

The fixture contains sanitized fake responses. This runner never constructs the
Supermemory SDK client and cannot read, write, or delete live memories.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.memory.supermemory import SupermemoryMemoryProvider  # noqa: E402

DEFAULT_FIXTURE = ROOT / "tests/fixtures/supermemory_recall_benchmark.json"


class FixtureClient:
    """Read-only fake client returning one checked-in response per query."""

    def __init__(self, cases: list[dict[str, Any]], documents: dict[str, dict[str, Any]]):
        self._cases = {case["query"]: case for case in cases}
        self._documents = documents

    def get_profile(self, query: str | None = None, *, container_tag: str | None = None) -> dict[str, Any]:
        del container_tag
        case = self._cases[query or ""]
        content = lambda doc_id: self._documents[doc_id]["content"]
        return {
            "static": [content(doc_id) for doc_id in case.get("static", [])],
            "dynamic": [content(doc_id) for doc_id in case.get("dynamic", [])],
            "search_results": [
                {
                    "id": doc_id,
                    "memory": content(doc_id),
                    "similarity": similarity,
                    "metadata": {
                        "benchmark_label": self._documents[doc_id]["label"],
                        "benchmark_class": self._documents[doc_id]["class"],
                    },
                }
                for doc_id, similarity in case.get("search", [])
            ],
        }


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_ids(context: str, documents: dict[str, dict[str, Any]]) -> list[str]:
    positions = []
    for doc_id, document in documents.items():
        position = context.find(document["content"])
        if position >= 0:
            positions.append((position, doc_id))
    return [doc_id for _, doc_id in sorted(positions)]


def evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
    settings = fixture["settings"]
    documents = {document["id"]: document for document in fixture["documents"]}
    cases = fixture["cases"]

    provider = SupermemoryMemoryProvider()
    provider._active = True
    provider._auto_recall = True
    provider._max_recall_results = settings["max_recall_results"]
    provider._profile_frequency = 50
    provider._config = dict(settings)
    provider._client = FixtureClient(cases, documents)  # type: ignore[assignment]

    k = settings["k"]
    relevant_selected_at_k = 0
    selected_at_k = 0
    required_selected = 0
    required_total = 0
    null_with_injection = 0
    null_total = 0
    transient_selected = Counter()
    total_selected = 0
    per_case = []
    category_counts: dict[str, Counter] = defaultdict(Counter)

    for case in cases:
        provider.on_turn_start(case.get("turn", 2), case["query"])
        context = provider.prefetch(case["query"])
        selected = _selected_ids(context, documents)
        selected_k = selected[:k]
        expected = set(case["expected_ids"])

        relevant_selected_at_k += sum(doc_id in expected for doc_id in selected_k)
        selected_at_k += len(selected_k)
        required_selected += len(expected.intersection(selected))
        required_total += len(expected)
        total_selected += len(selected)

        if not expected:
            null_total += 1
            null_with_injection += bool(selected)

        for doc_id in selected:
            document = documents[doc_id]
            if document["label"] == "transient":
                transient_selected[document["class"]] += 1

        category = category_counts[case["category"]]
        category["cases"] += 1
        category["selected"] += len(selected)
        category["required"] += len(expected)
        category["required_selected"] += len(expected.intersection(selected))

        per_case.append({
            "id": case["id"],
            "category": case["category"],
            "expected_ids": case["expected_ids"],
            "selected_ids": selected,
            "false_positive_ids": [doc_id for doc_id in selected if doc_id not in expected],
        })

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    contamination_by_class = {
        class_name: {
            "selected": count,
            "rate": ratio(count, total_selected),
        }
        for class_name, count in sorted(transient_selected.items())
    }
    category_summary = {
        name: {
            "cases": counts["cases"],
            "selected": counts["selected"],
            "required_recall": ratio(counts["required_selected"], counts["required"]),
        }
        for name, counts in sorted(category_counts.items())
    }
    by_id = {case["id"]: case for case in per_case}

    return {
        "schema_version": fixture["schema_version"],
        "case_count": len(cases),
        "document_count": len(documents),
        "settings_under_test": settings,
        "metrics": {
            f"precision_at_{k}": ratio(relevant_selected_at_k, selected_at_k),
            "false_positive_rate_on_null_queries": ratio(null_with_injection, null_total),
            "required_durable_recall": ratio(required_selected, required_total),
            "transient_contamination_rate": ratio(sum(transient_selected.values()), total_selected),
            "transient_contamination_by_class": contamination_by_class,
        },
        "exact_false_positive_modes": {
            "low_similarity_result_injected": "session_transcript" in by_id["null-01"]["selected_ids"],
            "profile_injected_when_disabled": "identity_timezone" in by_id["null-05"]["selected_ids"],
        },
        "categories": category_summary,
        "cases": per_case,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--assert-baseline",
        action="store_true",
        help="fail unless the current unfiltered-prefetch false positives are reproduced",
    )
    args = parser.parse_args()

    report = evaluate(load_fixture(args.fixture))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.assert_baseline and not all(report["exact_false_positive_modes"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
