from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import CategoryResult, RunResult
from benchmarks.statistical import aggregate_results
from benchmarks.tracks import backend_supports_category, missing_capabilities


def test_missing_capabilities_for_temporal_decay():
    caps = BackendCapabilities()
    assert backend_supports_category(caps, "semantic_recall") is True
    assert backend_supports_category(caps, "temporal_decay") is False
    assert missing_capabilities(caps, "temporal_decay") == ["time_simulation"]


def test_structured_backend_supports_structured_categories_only():
    caps = BackendCapabilities(scopes=True, typed_facts=True, supersession=True)
    assert backend_supports_category(caps, "scopes") is True
    assert backend_supports_category(caps, "notation_parsing") is True
    assert backend_supports_category(caps, "supersession") is True
    assert backend_supports_category(caps, "qlearning") is False


def test_aggregate_results_computes_mean_and_per_category():
    run1 = RunResult(
        seed=1,
        results_by_category={
            "semantic_recall": CategoryResult("semantic_recall", total=10, correct=9, score=0.9),
            "scopes": CategoryResult("scopes", total=5, correct=5, score=1.0),
        },
        overall_score=14 / 15,
    )
    run2 = RunResult(
        seed=2,
        results_by_category={
            "semantic_recall": CategoryResult("semantic_recall", total=10, correct=10, score=1.0),
            "scopes": CategoryResult("scopes", total=5, correct=4, score=0.8),
        },
        overall_score=14 / 15,
    )

    agg = aggregate_results([run1, run2])

    assert agg.num_runs == 2
    assert agg.per_category_mean["semantic_recall"] == 0.95
    assert agg.per_category_mean["scopes"] == 0.9
