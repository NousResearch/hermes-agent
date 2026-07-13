from pathlib import Path

from scripts.benchmarks.supermemory_recall_precision import evaluate, load_fixture

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
        response_ids.update(doc_id for doc_id, _ in case.get("search", []))
        assert response_ids.issubset(documents)


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
    assert metrics["precision_at_5"] < 1.0
    assert metrics["false_positive_rate_on_null_queries"] > 0.0
    assert metrics["transient_contamination_rate"] > 0.0
    assert metrics["required_durable_recall"] == 1.0

    contaminated_classes = metrics["transient_contamination_by_class"]
    assert {"kanban_task", "pull_request_status", "test_run", "project_status", "full_session"}.issubset(
        contaminated_classes
    )
