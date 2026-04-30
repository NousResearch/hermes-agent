from benchmarks.capabilities import BackendCapabilities
from benchmarks.interface import CategoryResult, RunResult
from benchmarks.statistical import aggregate_results
from benchmarks.runner import BACKEND_CAPABILITIES, CATEGORY_RUNNERS, SUITE_DIR, build_score_views, discover_suites, requested_categories_for_suites
from benchmarks.tracks import CATEGORY_SPECS, backend_supports_category, missing_capabilities


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
    # qlearning is now core track (no required capabilities) — all backends can run it
    assert backend_supports_category(caps, "qlearning") is True
    # integration is now core track (per-scenario capability checks instead)
    assert backend_supports_category(caps, "integration") is True
    assert backend_supports_category(caps, "privacy_forgetting") is False
    assert missing_capabilities(caps, "privacy_forgetting") == ["forgetting"]


def test_every_runner_category_has_track_metadata():
    assert set(CATEGORY_RUNNERS) - set(CATEGORY_SPECS) == set()


def test_every_fixture_category_has_runner_and_track_metadata():
    fixture_categories = {
        fixture.stem
        for suite in SUITE_DIR.glob("suite_*/fixtures")
        for fixture in suite.glob("*.json")
    }
    assert fixture_categories - set(CATEGORY_RUNNERS) == set()
    assert fixture_categories - set(CATEGORY_SPECS) == set()


def test_backend_capability_declarations_include_all_known_fields():
    fields = set(BackendCapabilities.__dataclass_fields__)
    assert "forgetting" in fields
    for backend_name, capabilities in BACKEND_CAPABILITIES.items():
        assert isinstance(capabilities, BackendCapabilities), backend_name
        missing_fields = [field for field in fields if not hasattr(capabilities, field)]
        assert missing_fields == []


def test_discover_suites_finds_all_fixture_suites():
    suites = discover_suites()
    assert suites[:3] == ["a", "b", "c"]
    assert suites[-6:] == ["o", "p", "q", "r", "s", "t"]


def test_new_category_fixtures_are_full_sized():
    import json

    expected_counts = {
        "p/fixtures/abstention.json": 12,
        "q/fixtures/preference_memory.json": 15,
        "r/fixtures/privacy_forgetting.json": 10,
        "s/fixtures/multi_hop_exploration.json": 12,
        "t/fixtures/long_conversation.json": 10,
    }
    for rel, expected in expected_counts.items():
        data = json.loads((SUITE_DIR / f"suite_{rel}").read_text())
        assert len(data) == expected

    preference_subtypes = {sc["sub_type"] for sc in json.loads((SUITE_DIR / "suite_q/fixtures/preference_memory.json").read_text())}
    assert preference_subtypes == {
        "stable_preference",
        "corrected_preference",
        "negative_preference",
        "project_convention",
        "identity_fact",
    }

    long_subtypes = {sc["sub_type"] for sc in json.loads((SUITE_DIR / "suite_t/fixtures/long_conversation.json").read_text())}
    assert long_subtypes == {
        "preference_evolution",
        "project_decision_history",
        "correction_after_delay",
        "multi_session_synthesis",
        "rejected_option_recall",
    }


def test_requested_categories_and_score_views():
    requested = requested_categories_for_suites(["d"])
    assert requested == ["adversarial"]

    run = RunResult(
        seed=42,
        results_by_category={
            "adversarial": CategoryResult("adversarial", total=15, correct=14, score=14 / 15),
            "timestamp_integrity": CategoryResult("timestamp_integrity", total=8, correct=7, score=7 / 8),
        },
        overall_score=21 / 23,
    )

    views = build_score_views([run], ["adversarial", "timestamp_integrity"])

    assert views["executed"]["correct"] == 21
    assert views["executed"]["total"] == 23
    assert views["core"]["categories"] == ["adversarial"]
    assert views["tracks"]["temporal"]["categories"] == ["timestamp_integrity"]


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
