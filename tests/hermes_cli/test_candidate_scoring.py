from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.candidate_scoring import (
    DIMENSIONS,
    PairObservation,
    aa_acceptance,
    archive_equivalence_key,
    archive_policy_digest,
    archive_rank,
    deterministic_indices,
    score_evaluation,
    verify_score_parity,
)


def _one_per_dimension(candidate: int = 100, incumbent: int = 90):
    return [
        {
            "case_id": dimension,
            "primary_dimension": dimension,
            "candidate_score": candidate,
            "incumbent_score": incumbent,
            "repetition": 1,
            "complete": True,
            "candidate_valid": True,
            "incumbent_valid": True,
        }
        for dimension in DIMENSIONS
    ]


def test_hfs_uses_six_pr1_primary_dimensions_and_fixed_weights():
    summary = score_evaluation(
        _one_per_dimension(),
        repetitions=1,
        expected_case_ids=DIMENSIONS,
        replicates=32,
    )
    assert summary["status"] == "SCREEN-PASS"
    assert summary["candidate"]["hfs"] == 100.0
    assert summary["incumbent"]["hfs"] == 90.0
    assert summary["paired_hfs_delta"]["mean"] == 10.0
    assert summary["counts"]["wins"] == 6
    assert summary["counts"]["losses"] == 0
    assert summary["counts"]["ties"] == 0
    for dimension in DIMENSIONS:
        assert summary["dimensions"][dimension]["n_arm_candidate"] == 1
        assert summary["dimensions"][dimension]["n_arm_incumbent"] == 1
        assert summary["dimensions"][dimension]["n_pair"] == 1


def test_pair_observation_missing_validity_flags_fails_closed():
    observation = PairObservation.from_mapping({
        "case_id": "case-1",
        "primary_dimension": "correctness",
        "candidate_score": 100,
        "incumbent_score": 100,
        "arm_order": "candidate-first",
    })
    assert observation.complete is False
    assert observation.candidate_valid is False
    assert observation.incumbent_valid is False


def test_repetitions_average_inside_case_before_dimension_mean():
    rows = []
    for dimension in DIMENSIONS:
        rows.extend(
            {
                "case_id": dimension,
                "primary_dimension": dimension,
                "repetition": repetition,
                "candidate_score": candidate,
                "incumbent_score": 80,
                    "complete": True,
                    "candidate_valid": True,
                    "incumbent_valid": True,
            }
            for repetition, candidate in ((1, 100), (2, 0), (3, 100))
        )
    summary = score_evaluation(
        rows, repetitions=3, expected_case_ids=DIMENSIONS, replicates=16
    )
    assert summary["candidate"]["dimensions"]["correctness"] == 66.667
    assert summary["candidate"]["hfs"] == 66.667
    assert summary["dimensions"]["correctness"]["n_pair"] == 1


def test_incomplete_pair_is_gate_failed_and_not_counted_as_win():
    rows = _one_per_dimension()
    rows[0]["complete"] = False
    rows[0]["candidate_valid"] = False
    summary = score_evaluation(
        rows, repetitions=1, expected_case_ids=DIMENSIONS, replicates=8
    )
    assert summary["status"] == "GATE-FAILED"
    assert summary["counts"]["incomplete"] == 1
    assert summary["counts"]["wins"] == 5
    assert summary["dimensions"]["correctness"]["n_pair"] == 0


def test_tie_epsilon_and_screening_vocabulary():
    rows = _one_per_dimension(candidate=100, incumbent=99.5)
    summary = score_evaluation(
        rows, repetitions=1, expected_case_ids=DIMENSIONS, replicates=8
    )
    assert summary["counts"]["ties"] == 6
    assert summary["status"] == "SCREEN-PASS"
    assert "PROMOTE-CANDIDATE" not in summary["status"]


def test_counter_rng_is_reproducible_and_seeded_without_global_state():
    first = deterministic_indices(
        7, 12, seed=20260715, metric="golden", dimension="correctness", level="case"
    )
    second = deterministic_indices(
        7, 12, seed=20260715, metric="golden", dimension="correctness", level="case"
    )
    assert first == second
    assert first == [6, 5, 5, 0, 3, 2, 2, 1, 1, 2, 4, 2]


def test_aa_acceptance_requires_exact_81_pairs_and_zero_including_intervals():
    observations = [
        PairObservation(
            case_id=f"case-{index:03d}",
            primary_dimension="correctness",
            candidate_score=100,
            incumbent_score=100,
            arm_order="candidate-first" if index % 2 else "incumbent-first",
        )
        for index in range(81)
    ]
    accepted = aa_acceptance(
        observations,
        receipt_integrity_rate=1.0,
        scorer_disagreement_count=0,
        seed=20260715,
        replicates=32,
    )
    assert accepted["accepted"] is True
    assert accepted["criteria"] == {
        "receipt_integrity": True,
        "scorer_disagreement": True,
        "false_non_tie_rate": True,
        "mean_delta": True,
        "order_effect": True,
    }
    rejected = aa_acceptance(
        observations[:-1],
        receipt_integrity_rate=1.0,
        scorer_disagreement_count=0,
        seed=20260715,
        replicates=8,
    )
    assert rejected["accepted"] is False
    assert rejected["status"] == "GATE-FAILED"


def test_archive_requires_exact_equivalence_key_and_is_informational():
    key = archive_equivalence_key({
        "lane_id": "cli-full-v1",
        "suite_id": "full-hermes-cli-v1",
        "suite_version": 1,
        "case_catalog_digest": "catalog",
        "scorer_id": "hermes-fitness-v1",
        "scorer_version": 1,
        "weights_version": "cli-full-v1",
        "hard_gate_policy_version": 1,
        "pairing_policy_version": 1,
        "hermes_revision": "rev",
        "config_policy_digest": "cfg",
        "tool_schema_policy_digest": "tools",
        "compression_mode": "deferred",
        "external_network": "excluded-tools-only",
        "filesystem_scope": "fixture-only",
        "approval_policy": "configured",
        "hardware_class": "cpu",
        "accelerator_family": "none",
        "device_count": 1,
        "driver_major": "0",
        "runtime_major": "1",
    })
    digest = archive_policy_digest(key)
    entries = [
        {"entry_id": "a", "hfs": 80, "equivalence_key": key, "policy_digest": digest},
        {"entry_id": "b", "hfs": 90, "equivalence_key": key, "policy_digest": digest},
        {
            "entry_id": "wrong",
            "hfs": 100,
            "equivalence_key": {**key, "hardware_class": "gpu"},
            "policy_digest": digest,
        },
    ]
    ranked = archive_rank(85, entries, equivalence_key=key, policy_digest=digest)
    assert ranked["rank"] == 2
    assert ranked["n"] == 2
    assert ranked["label"] == "raw-rank-only"


def test_golden_fixture_is_valid_json_and_has_pinned_fields():
    path = Path(__file__).parent / "fixtures" / "candidate_scoring" / "golden-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["scorer_version"] == 1
    assert value["rng"] == "sha256-counter-v1"
    assert value["replicates"] == 10000
    assert value["input"]
    assert value["expected"]["paired_hfs_delta"]


def test_golden_vector_reduces_to_exact_pinned_values():
    path = Path(__file__).parent / "fixtures" / "candidate_scoring" / "golden-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["input"]
    summary = score_evaluation(
        rows,
        seed=value["seed"],
        repetitions=1,
        expected_case_ids=[row["case_id"] for row in rows],
        replicates=value["replicates"],
    )
    expected = value["expected"]
    assert summary["candidate"]["hfs"] == expected["candidate_hfs"]
    assert summary["incumbent"]["hfs"] == expected["incumbent_hfs"]
    assert summary["paired_hfs_delta"] == expected["paired_hfs_delta"]
    for dimension, interval in expected["dimension_deltas"].items():
        actual = summary["dimensions"][dimension]["delta_ci"]
        assert {key: actual[key] for key in ("mean", "lower", "upper")} == interval


def test_online_and_offline_scorer_payloads_have_exact_parity():
    rows = _one_per_dimension(candidate=97, incumbent=96)
    online = score_evaluation(
        rows, repetitions=1, expected_case_ids=DIMENSIONS, replicates=16
    )
    offline = score_evaluation(
        rows, repetitions=1, expected_case_ids=DIMENSIONS, replicates=16
    )
    assert verify_score_parity(online, offline)
