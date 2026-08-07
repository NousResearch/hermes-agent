from copy import deepcopy
from pathlib import Path

import pytest

from plugins.memory import supermemory as supermemory_module
from plugins.memory.supermemory import SupermemoryMemoryProvider
from scripts.benchmarks.supermemory_recall_precision import (
    FixtureClient,
    _prefetch_with_selection_trace,
    evaluate,
    load_fixture,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/supermemory_recall_benchmark.json"
EXPECTED_CATEGORIES = {
    "durable_identity_preferences",
    "cross_project_durable_conventions",
    "project_specific_durable_facts",
    "unrelated_null_queries",
    "adversarial_transient",
}


def test_recall_benchmark_fixture_has_compact_expected_label_schema():
    fixture = load_fixture(FIXTURE)
    documents = {document["id"]: document for document in fixture["documents"]}
    cases = fixture["cases"]

    assert fixture["schema_version"] == 1
    assert len(cases) >= 20
    assert {case["category"] for case in cases} == EXPECTED_CATEGORIES
    assert all(sum(case["category"] == category for case in cases) >= 4 for category in EXPECTED_CATEGORIES)
    assert len({document["content"] for document in documents.values()}) == len(documents)

    for case in cases:
        assert set(case) >= {"id", "category", "query", "expected_ids"}
        assert set(case["expected_ids"]).issubset(documents)
        assert all(documents[doc_id]["label"] == "durable" for doc_id in case["expected_ids"])
        response_ids = set(case.get("static", [])) | set(case.get("dynamic", []))
        response_ids.update(doc_id for doc_id, *_ in case.get("search", []))
        assert response_ids.issubset(documents)


def test_recall_benchmark_covers_selection_policy_edge_cases():
    fixture = load_fixture(FIXTURE)
    documents = {document["id"]: document for document in fixture["documents"]}
    cases = fixture["cases"]
    k = fixture["settings"]["k"]

    assert any(len(case.get("search", [])) > k for case in cases)
    assert any(len(result) == 1 for case in cases for result in case.get("search", []))
    assert any(
        documents[doc_id]["label"] == "durable" and doc_id not in case["expected_ids"]
        for case in cases
        for doc_id, *_ in case.get("search", [])
    )
    assert any(
        "paraphrase" in documents[doc_id]["class"]
        for case in cases
        for doc_id, *_ in case.get("search", [])
    )
    adversarial_queries = [case["query"].lower() for case in cases if case["category"] == "adversarial_transient"]
    assert any(all(marker not in query for marker in ("durable", "lasting", "persistent", "stable")) for query in adversarial_queries)
    assert any(any(ord(character) > 127 for character in document["content"]) for document in documents.values())
    assert any("documentType" in document.get("metadata", {}) for document in documents.values())
    assert any(document.get("metadata", {}).get("type") == "full session" for document in documents.values())
    malformed_scores = {
        result[1]
        for case in cases
        for result in case.get("search", [])
        if len(result) > 1
        and (isinstance(result[1], str) or not 0 <= result[1] <= 1)
    }
    assert {"NaN", "Infinity", "-Infinity", -0.01, 1.01}.issubset(malformed_scores)


def test_fixture_client_keeps_evaluator_truth_out_of_provider_responses():
    fixture = load_fixture(FIXTURE)
    documents = {document["id"]: document for document in fixture["documents"]}
    client = FixtureClient(fixture["cases"], documents)

    assert all("expected_ids" not in case for case in client._cases.values())
    assert all({"label", "class"}.isdisjoint(document) for document in client._documents.values())

    for case in fixture["cases"]:
        response = client.get_profile(case["query"])
        for result in response["search_results"]:
            evaluator_fields = {"label", "class", "benchmark_label", "benchmark_class"}
            assert evaluator_fields.isdisjoint(result["metadata"])
            assert result["metadata"] == documents[result["id"]].get("metadata", {})


@pytest.mark.parametrize(
    ("collection", "field"),
    [("documents", "id"), ("cases", "id"), ("cases", "query")],
)
def test_evaluate_rejects_duplicate_fixture_identifiers(collection, field):
    fixture = deepcopy(load_fixture(FIXTURE))
    duplicate = deepcopy(fixture[collection][0])
    if collection == "cases":
        other_field = "query" if field == "id" else "id"
        duplicate[other_field] += "-otherwise-unique"
    fixture[collection].append(duplicate)

    with pytest.raises(ValueError, match=f"duplicate {field}"):
        evaluate(fixture)


def test_precision_gated_prefetch_meets_offline_recall_contract():
    report = evaluate(load_fixture(FIXTURE))
    metrics = report["metrics"]

    assert report["exact_false_positive_modes"] == {
        "low_similarity_result_injected": False,
        "profile_injected_when_disabled": False,
    }
    assert "precision_at_5" not in metrics
    assert metrics["precision_at_returned_up_to_5"] >= 0.95
    assert metrics["injection_rate_on_null_queries"] <= 0.05
    assert metrics["mean_false_positives_per_null_query"] == 0.0
    assert metrics["relevance_false_positive_rate"] <= 0.05
    assert metrics["durable_irrelevant_selections"] == 0
    assert metrics["transient_contamination_rate"] == 0.0
    assert metrics["transient_contamination_by_class"] == {}
    # Profile recall is explicitly disabled by this fixture, so six required
    # profile-only documents trade recall for no unsolicited profile injection.
    assert metrics["required_durable_recall"] >= 0.7


def test_evaluate_applies_non_default_policy_settings():
    fixture = deepcopy(load_fixture(FIXTURE))
    fixture["settings"]["recall_min_similarity"] = 1.0
    fixture["settings"]["prefetch_include_profile"] = True
    fixture["settings"]["max_recall_results"] = 1

    report = evaluate(fixture)

    assert report["settings_under_test"] == fixture["settings"]
    assert report["metrics"]["required_durable_recall"] < 0.7857
    assert report["exact_false_positive_modes"]["profile_injected_when_disabled"] is True
    assert all(len(case["selected_ids"]) <= 1 for case in report["cases"])


def test_selected_ids_are_traced_without_content_substring_matching():
    fixture = deepcopy(load_fixture(FIXTURE))
    documents = {document["id"]: document for document in fixture["documents"]}
    documents["identity_language"]["content"] = "Avery"

    report = evaluate(fixture)
    identity_case = next(case for case in report["cases"] if case["id"] == "identity-02")

    assert identity_case["selected_ids"] == ["identity_style"]


def test_selected_ids_match_items_emitted_by_formatter_policy(monkeypatch):
    documents = {
        "shared": {"content": "Shared durable fact", "label": "durable", "class": "identity"},
        "dynamic": {"content": "Recent transient status", "label": "transient", "class": "status"},
        "search": {"content": "Search-only result", "label": "transient", "class": "search"},
    }
    cases = [{
        "query": "trace actual formatter output",
        "static": ["shared"],
        "dynamic": ["dynamic"],
        "search": [["shared", 0.99], ["search", 0.98]],
    }]
    provider = SupermemoryMemoryProvider()
    provider._active = True
    provider._auto_recall = True
    provider._max_recall_results = 5
    provider._prefetch_include_profile = True
    provider._profile_frequency = 50
    provider._client = FixtureClient(cases, documents)  # type: ignore[assignment]
    provider.on_turn_start(1, cases[0]["query"])

    production_formatter = supermemory_module._format_prefetch_context
    emitted_ids = []

    def one_total_result_formatter(static_facts, dynamic_facts, search_results, max_results):
        del dynamic_facts, search_results, max_results
        emitted_ids.append("shared")
        return production_formatter(static_facts[:1], [], [], 1)

    monkeypatch.setattr(supermemory_module, "_format_prefetch_context", one_total_result_formatter)

    context, selected_ids = _prefetch_with_selection_trace(provider, cases[0]["query"])

    assert "Shared durable fact" in context
    assert "Recent transient status" not in context
    assert "Search-only result" not in context
    assert selected_ids == emitted_ids == ["shared"]


@pytest.mark.parametrize("failure_mode", ["empty", "raise"])
def test_empty_prefetch_discards_partial_selection_trace_and_next_call_is_clean(monkeypatch, failure_mode):
    documents = {
        "shared": {"content": "Shared durable fact", "label": "durable", "class": "identity"},
    }
    cases = [{
        "query": "discard partial trace",
        "static": ["shared"],
    }]
    provider = SupermemoryMemoryProvider()
    provider._active = True
    provider._auto_recall = True
    provider._max_recall_results = 5
    provider._prefetch_include_profile = True
    provider._profile_frequency = 50
    client = FixtureClient(cases, documents)
    provider._client = client  # type: ignore[assignment]
    provider.on_turn_start(1, cases[0]["query"])

    production_formatter = supermemory_module._format_prefetch_context

    def unsuccessful_formatter(static_facts, dynamic_facts, search_results, max_results):
        rendered = f"{static_facts[0]}"
        if failure_mode == "raise":
            raise RuntimeError(f"failed after rendering {rendered}")
        return ""

    monkeypatch.setattr(supermemory_module, "_format_prefetch_context", unsuccessful_formatter)

    context, selected_ids = _prefetch_with_selection_trace(provider, cases[0]["query"])

    assert context == ""
    assert selected_ids == []
    assert client._selection_trace is None

    monkeypatch.setattr(supermemory_module, "_format_prefetch_context", production_formatter)

    next_context, next_selected_ids = _prefetch_with_selection_trace(provider, cases[0]["query"])

    assert "Shared durable fact" in next_context
    assert next_selected_ids == ["shared"]
    assert client._selection_trace is None
