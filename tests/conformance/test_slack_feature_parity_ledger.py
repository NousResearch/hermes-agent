"""Slack-specific semantic guard for the canonical 24-row parity ledger."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
sys.path.insert(0, str(SCRIPTS_CI))

from feature_parity_ledger import validate_ledger  # noqa: E402

LEDGER_PATH = (
    REPO_ROOT / "docs" / "architecture" / "feature-parity" / "slack.json"
)
REGISTRY_PATH = (
    REPO_ROOT / "docs" / "architecture" / "feature-parity" / "contracts.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(document: dict) -> dict[str, dict]:
    return {row["id"]: row for row in document["capabilities"]}


def test_slack_ledger_passes_generic_contract() -> None:
    assert validate_ledger(
        _load(LEDGER_PATH),
        contract_registry=_load(REGISTRY_PATH),
    ) == []


def test_slack_contract_has_exactly_24_canonical_rows() -> None:
    document = _load(LEDGER_PATH)
    expected = [f"SLK{index}" for index in range(1, 25)]
    assert document["campaign"]["expected_capability_ids"] == expected
    assert [row["id"] for row in document["capabilities"]] == expected


def test_packet_evidence_never_promotes_delivery() -> None:
    document = _load(LEDGER_PATH)
    evidence = document["snapshot"]["packet_evidence"]
    assert evidence == [
        {
            "name": "slack-feature-parity-2026-08-14",
            "disposition": "historical_evidence_only",
        },
        {
            "name": "hermes-tag-horizontal-layer-2026-08-14",
            "commit": "dcf05b8ff59a81709f044c5aa6c8ce1026bc8d19",
            "disposition": "additive_kernel_not_upstream_wired",
        },
    ]
    assert not any(row["artifact_evidence"] for row in document["capabilities"])
    assert all(
        row["delivery_state"] not in {
            "candidate_unwired",
            "candidate_open",
            "on_main_unverified",
            "released",
        }
        for row in document["capabilities"]
    )


def test_snapshot_records_absent_runtime_and_god_files() -> None:
    snapshot = _load(LEDGER_PATH)["snapshot"]
    assert snapshot["slack_adapter_bytes"] == 424946
    assert snapshot["channel_governance_on_main"] is False
    assert snapshot["hermes_tag_kernel_on_main"] is False
    assert snapshot["packet_artifacts_are_delivery"] is False
    assert snapshot["open_extraction_prs"] == [79712, 79713, 79714, 79800, 80303]
    assert snapshot["non_mergeable_extraction_prs"] == [79800]


def test_structural_prs_remain_dependencies_not_false_completion() -> None:
    rows = _rows(_load(LEDGER_PATH))
    expected = {79712, 79713, 79714, 79800, 80303}
    for capability_id in ("SLK5", "SLK6"):
        publications = rows[capability_id]["publications"]
        assert {p["number"] for p in publications} == expected
        assert all(p["role"] == "dependency" for p in publications)
        assert rows[capability_id]["delivery_state"] == "gap"


def test_flagship_and_native_stop_gaps_are_interlocked() -> None:
    rows = _rows(_load(LEDGER_PATH))
    assert rows["SLK14"]["publications"] == [
        {"kind": "issue", "number": 80338, "role": "dependency", "state": "open"}
    ]
    assert rows["SLK18"]["publications"] == [
        {"kind": "issue", "number": 90978, "role": "dependency", "state": "open"}
    ]


def test_epic_reconciliation_has_one_authority_state() -> None:
    rows = _rows(_load(LEDGER_PATH))
    assert rows["SLK3"]["delivery_state"] == "candidate_blocked"
    authoritative = [
        p for p in rows["SLK3"]["publications"]
        if p.get("role") == "authoritative"
    ]
    assert authoritative == [{
        "kind": "pull_request",
        "number": 91036,
        "role": "authoritative",
        "state": "open",
        "author": "andrexibiza",
        "url": "https://github.com/NousResearch/hermes-agent/pull/91036",
    }]


def test_status_counts_are_an_explicit_non_completion_receipt() -> None:
    counts = Counter(
        row["delivery_state"] for row in _load(LEDGER_PATH)["capabilities"]
    )
    assert counts["released"] == 0
    assert counts["on_main_unverified"] == 0
    assert counts["candidate_open"] == 0
    assert counts["candidate_unwired"] == 0
    assert counts["candidate_blocked"] == 1
    assert counts["gap"] == 23
    assert sum(counts.values()) == 24


def test_forbidden_growth_surfaces_are_locked() -> None:
    forbidden = _load(LEDGER_PATH)["campaign"]["forbidden_growth_paths"]
    assert forbidden == [
        "plugins/platforms/slack/adapter.py",
        "tests/gateway/test_slack.py",
        "gateway/run.py",
    ]
