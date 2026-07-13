from copy import deepcopy
from pathlib import Path

import pytest

from scripts.benchmarks.supermemory_recall_precision import FixtureClient, evaluate, load_fixture

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


def test_current_prefetch_baseline_exposes_precision_problem():
    report = evaluate(load_fixture(FIXTURE))
    metrics = report["metrics"]

    # RED-capable baseline: current prefetch injects all fake results regardless
    # of configured similarity and still injects profile data when the fixture's
    # prefetch_include_profile setting is false.
    assert report["exact_false_positive_modes"] == {
        "low_similarity_result_injected": True,
        "profile_injected_when_disabled": True,
    }
    assert "precision_at_5" not in metrics
    assert metrics["precision_at_returned_up_to_5"] < 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["injection_rate_on_null_queries"] > 0.0
    assert metrics["mean_false_positives_per_null_query"] > 0.0
    assert metrics["relevance_false_positive_rate"] > metrics["transient_contamination_rate"]
    assert metrics["durable_irrelevant_selections"] > 0
    assert metrics["transient_contamination_rate"] > 0.0
    assert metrics["required_durable_recall"] == 1.0

    contaminated_classes = metrics["transient_contamination_by_class"]
    assert {"kanban_task", "pull_request_status", "test_run", "project_status", "full_session"}.issubset(
        contaminated_classes
    )


def test_selected_ids_are_traced_without_content_substring_matching():
    fixture = deepcopy(load_fixture(FIXTURE))
    documents = {document["id"]: document for document in fixture["documents"]}
    documents["identity_language"]["content"] = "Avery"

    report = evaluate(fixture)
    identity_case = next(case for case in report["cases"] if case["id"] == "identity-02")

    assert identity_case["selected_ids"] == ["identity_style", "status_worker"]
