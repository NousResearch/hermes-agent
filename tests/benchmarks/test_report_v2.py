from benchmarks.visualize.report import generate_comparison_report, generate_report


def _payload(backend, score, correct, total, skipped=None):
    return {
        "schema_version": "2.0",
        "backend": backend,
        "profile": "balanced",
        "embedding_model": "tfidf",
        "suites": ["all"],
        "requested_categories": ["semantic_recall", "privacy_forgetting"],
        "executed_categories": ["semantic_recall"],
        "skipped_categories": skipped or {},
        "score_views": {
            "executed": {
                "score": score,
                "correct": correct,
                "total": total,
                "categories": ["semantic_recall"],
            },
            "core": {
                "score": score,
                "correct": correct,
                "total": total,
                "categories": ["semantic_recall"],
            },
            "tracks": {
                "core": {
                    "score": score,
                    "correct": correct,
                    "total": total,
                    "categories": ["semantic_recall"],
                }
            },
        },
        "mean_score": score,
        "std": 0.0,
        "ci_95": [score, score],
        "per_category_mean": {"semantic_recall": score},
        "per_category_std": {"semantic_recall": 0.0},
        "num_runs": 1,
        "runtime": {
            "python": "3.x",
            "platform": "test-platform",
            "packages": {"mnemoria": "0.2.2"},
            "git": {"branch": "test", "commit": "abc123", "dirty": False},
            "credentials": {"MEM0_API_KEY": "missing"},
        },
        "runs": [
            {
                "seed": 42,
                "overall_score": score,
                "wall_time_seconds": 1.0,
                "token_usage": {},
                "categories": {
                    "semantic_recall": {
                        "score": score,
                        "correct": correct,
                        "total": total,
                    }
                },
            }
        ],
    }


def test_generate_report_includes_schema_v2_fairness_metadata():
    report = generate_report(
        _payload(
            "mnemoria",
            0.8,
            8,
            10,
            skipped={"privacy_forgetting": "missing capabilities: forgetting"},
        )
    )

    assert "Schema Version" in report
    assert "Fair Comparison Views" in report
    assert "Core" in report
    assert "Skipped Categories" in report
    assert "privacy_forgetting" in report
    assert "Runtime and Reproducibility Metadata" in report
    assert "MEM0_API_KEY: missing" in report


def test_generate_comparison_report_includes_core_and_shared_scores():
    before = _payload("baseline-flat", 0.6, 6, 10)
    after = _payload("mnemoria", 0.8, 8, 10)

    report = generate_comparison_report(before, after)

    assert "Core Score" in report
    assert "Shared-Category Score" in report
    assert "1 cats" in report
    assert "baseline-flat" in report
    assert "mnemoria" in report
