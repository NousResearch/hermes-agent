#!/usr/bin/env python3
"""Offline selection-policy benchmark for the current Supermemory prefetch path.

The fixture contains sanitized fake responses. This runner never constructs the
Supermemory SDK client and cannot read, write, or delete live memories. It tests
Hermes' policy over supplied responses, not Supermemory retrieval quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.memory.supermemory import SupermemoryMemoryProvider  # noqa: E402

DEFAULT_FIXTURE = ROOT / "tests/fixtures/supermemory_recall_benchmark.json"


def _require_unique(items: list[dict[str, Any]], field: str, collection: str) -> None:
    counts = Counter(item[field] for item in items)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {field} in {collection}: {duplicates}")


def _validate_fixture(fixture: dict[str, Any]) -> None:
    _require_unique(fixture["documents"], "id", "documents")
    _require_unique(fixture["cases"], "id", "cases")
    _require_unique(fixture["cases"], "query", "cases")


class _TracedContent(str):
    """Fixture text that reports when the production formatter renders it."""

    _doc_id: str
    _on_render: Callable[[str], None]

    def __new__(cls, content: str, doc_id: str, on_render: Callable[[str], None]):
        instance = super().__new__(cls, content)
        instance._doc_id = doc_id
        instance._on_render = on_render
        return instance

    def __format__(self, format_spec: str) -> str:
        rendered = super().__format__(format_spec)
        self._on_render(self._doc_id)
        return rendered


class FixtureClient:
    """Read-only fake client returning one checked-in response per query."""

    def __init__(self, cases: list[dict[str, Any]], documents: dict[str, dict[str, Any]]):
        self._selection_trace: Callable[[str], None] | None = None
        self._cases = {
            case["query"]: {
                "static": list(case.get("static", [])),
                "dynamic": list(case.get("dynamic", [])),
                "search": [list(result) for result in case.get("search", [])],
            }
            for case in cases
        }
        self._documents = {
            doc_id: {
                "content": document["content"],
                "metadata": dict(document.get("metadata", {})),
            }
            for doc_id, document in documents.items()
        }

    def get_profile(self, query: str | None = None, *, container_tag: str | None = None) -> dict[str, Any]:
        del container_tag
        case = self._cases[query or ""]

        def content(doc_id: str) -> str:
            value = self._documents[doc_id]["content"]
            if self._selection_trace is None:
                return value
            return _TracedContent(value, doc_id, self._selection_trace)

        def search_result(result: list[Any]) -> dict[str, Any]:
            doc_id = result[0]
            payload = {
                "id": doc_id,
                "memory": content(doc_id),
                "metadata": dict(self._documents[doc_id].get("metadata", {})),
            }
            if len(result) > 1:
                payload["similarity"] = result[1]
            return payload

        return {
            "static": [content(doc_id) for doc_id in case.get("static", [])],
            "dynamic": [content(doc_id) for doc_id in case.get("dynamic", [])],
            "search_results": [search_result(result) for result in case.get("search", [])],
        }


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    _validate_fixture(fixture)
    return fixture


def _prefetch_with_selection_trace(
    provider: SupermemoryMemoryProvider,
    query: str,
) -> tuple[str, list[str]]:
    """Run prefetch while fixture values report actual formatter rendering."""
    selected_ids: list[str] = []
    client = provider._client
    if not isinstance(client, FixtureClient):
        raise TypeError("selection tracing requires FixtureClient")
    client._selection_trace = selected_ids.append
    try:
        context = provider.prefetch(query)
    finally:
        client._selection_trace = None
    return context, selected_ids if context else []


def evaluate(fixture: dict[str, Any]) -> dict[str, Any]:
    _validate_fixture(fixture)
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
    null_false_positives = 0
    transient_selected = Counter()
    relevance_false_positives = 0
    durable_irrelevant_selections = 0
    total_selected = 0
    per_case = []
    category_counts: dict[str, Counter] = defaultdict(Counter)

    for case in cases:
        provider.on_turn_start(case.get("turn", 2), case["query"])
        _, selected = _prefetch_with_selection_trace(provider, case["query"])
        selected_k = selected[:k]
        expected = set(case["expected_ids"])
        false_positive_ids = [doc_id for doc_id in selected if doc_id not in expected]

        relevant_selected_at_k += sum(doc_id in expected for doc_id in selected_k)
        selected_at_k += len(selected_k)
        required_selected += len(expected.intersection(selected))
        required_total += len(expected)
        total_selected += len(selected)
        relevance_false_positives += len(false_positive_ids)
        durable_irrelevant_selections += sum(
            documents[doc_id]["label"] == "durable" for doc_id in false_positive_ids
        )

        if not expected:
            null_total += 1
            null_with_injection += bool(selected)
            null_false_positives += len(selected)

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
            "false_positive_ids": false_positive_ids,
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
            f"precision_at_returned_up_to_{k}": ratio(relevant_selected_at_k, selected_at_k),
            f"recall_at_{k}": ratio(relevant_selected_at_k, required_total),
            "injection_rate_on_null_queries": ratio(null_with_injection, null_total),
            "mean_false_positives_per_null_query": ratio(null_false_positives, null_total),
            "relevance_false_positive_rate": ratio(relevance_false_positives, total_selected),
            "durable_irrelevant_selections": durable_irrelevant_selections,
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
