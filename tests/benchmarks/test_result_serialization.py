import json

from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import AggregateResult, BenchmarkConfig, CategoryResult, RunResult
from benchmarks.runner import (
    build_result_data,
    build_skipped_category_reasons,
    print_results,
    shared_category_view,
)


def _sample_run(seed=42):
    return RunResult(
        seed=seed,
        results_by_category={
            "semantic_recall": CategoryResult(
                "semantic_recall",
                total=10,
                correct=8,
                score=0.8,
                sub_scores={"easy": 1.0},
                retrieval_metrics={"recall_at_5": 0.8},
                recall_tokens=50,
                recall_chars=200,
            )
        },
        overall_score=0.8,
        wall_time_seconds=1.25,
        token_usage={"recall_tokens": 50},
        retrieval_metrics={"recall_at_5": 0.8},
        cost_metrics={"tokens_per_query": 5.0},
    )


def test_build_skipped_category_reasons_explains_missing_capabilities():
    skipped = build_skipped_category_reasons(
        requested_categories=["semantic_recall", "temporal_decay"],
        executed_categories=["semantic_recall"],
        capabilities=BackendCapabilities(),
    )

    assert skipped == {"temporal_decay": "missing capabilities: time_simulation"}


def test_build_result_data_uses_rich_schema_and_runtime_metadata():
    config = BenchmarkConfig(
        backend_name="baseline-flat",
        profile="balanced",
        embedding_model="tfidf",
        parameters={"suites": ["a"]},
        num_runs=1,
        seeds=[42],
    )
    agg = AggregateResult(
        num_runs=1,
        mean_score=0.8,
        std_score=0.0,
        ci_95_lower=0.8,
        ci_95_upper=0.8,
        per_category_mean={"semantic_recall": 0.8},
        per_category_std={"semantic_recall": 0.0},
    )

    data = build_result_data(config, agg, [_sample_run()])
    encoded = json.dumps(data)

    assert data["schema_version"] == "2.0"
    assert data["backend"] == "baseline-flat"
    assert "requested_categories" in data
    assert data["executed_categories"] == ["semantic_recall"]
    assert isinstance(data["skipped_categories"], dict)
    assert "score_views" in data
    assert data["score_views"]["core"]["score"] == 0.8
    assert "runtime" in data
    assert "python" in data["runtime"]
    assert data["runs"][0]["categories"]["semantic_recall"]["sub_scores"] == {"easy": 1.0}
    assert "secret" not in encoded.lower()


def test_print_results_includes_fair_comparison_views(capsys):
    config = BenchmarkConfig(
        backend_name="baseline-flat",
        parameters={"suites": ["a"]},
        num_runs=1,
        seeds=[42],
    )
    agg = AggregateResult(
        num_runs=1,
        mean_score=0.8,
        std_score=0.0,
        ci_95_lower=0.8,
        ci_95_upper=0.8,
        per_category_mean={"semantic_recall": 0.8},
        per_category_std={"semantic_recall": 0.0},
    )

    print_results(agg, config, [_sample_run()])

    out = capsys.readouterr().out
    assert "Fair comparison views:" in out
    assert "Executed score: 0.800 over 1 categories" in out
    assert "Core score:     0.800 over 1 categories" in out


def test_shared_category_view_scores_only_intersection():
    payloads = [
        {
            "backend": "a",
            "executed_categories": ["semantic_recall", "temporal_decay"],
            "runs": [
                {
                    "categories": {
                        "semantic_recall": {"correct": 8, "total": 10},
                        "temporal_decay": {"correct": 1, "total": 10},
                    }
                }
            ],
        },
        {
            "backend": "b",
            "executed_categories": ["semantic_recall", "scopes"],
            "runs": [
                {
                    "categories": {
                        "semantic_recall": {"correct": 6, "total": 10},
                        "scopes": {"correct": 10, "total": 10},
                    }
                }
            ],
        },
    ]

    view = shared_category_view(payloads)

    assert view["categories"] == ["semantic_recall"]
    assert view["backends"]["a"] == {"score": 0.8, "correct": 8, "total": 10}
    assert view["backends"]["b"] == {"score": 0.6, "correct": 6, "total": 10}
