"""Behavioral tests for the executable Feature Parity campaign ledger contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
sys.path.insert(0, str(SCRIPTS_CI))

from validate_feature_parity_ledger import (  # noqa: E402
    canonical_contract_digest,
    validate_ledger,
)


def _row(
    capability_id: str,
    *,
    name: str | None = None,
    product_state: str = "accepted",
    delivery_state: str = "candidate_open",
    pr: int | None = None,
) -> dict:
    row = {
        "id": capability_id,
        "name": name or f"Capability {capability_id}",
        "source_anchor": f"spec#{capability_id}",
        "product_state": product_state,
        "delivery_state": delivery_state,
        "implementation_paths": [
            f"plugins/platforms/example/{capability_id.lower()}.py"
        ],
        "test_paths": [f"tests/example/test_{capability_id.lower()}.py"],
        "consumers": [f"plugins/platforms/example/runtime.py:{capability_id}"],
        "publications": [],
        "artifact_evidence": [],
    }
    if pr is not None:
        row["publications"] = [
            {
                "kind": "pull_request",
                "number": pr,
                "role": "authoritative",
            }
        ]
    if product_state in {"pair_gap", "conditional", "deferred", "rejected"}:
        row["decision"] = "Explicit product decision."
    if product_state == "rejected":
        row["implementation_paths"] = []
        row["test_paths"] = []
        row["consumers"] = []
        row["delivery_state"] = "gap"
    return row


def _ledger(*rows: dict) -> dict:
    capabilities = list(rows)
    return {
        "schema_version": 1,
        "campaign": {
            "id": "example-parity",
            "tracker": 123,
            "expected_capability_ids": [row["id"] for row in capabilities],
            "forbidden_growth_paths": ["plugins/platforms/example/adapter.py"],
            "contract_sha256": canonical_contract_digest(capabilities),
        },
        "snapshot": {
            "upstream_sha": "a" * 40,
            "captured_at": "2026-08-19T21:45:14Z",
        },
        "capabilities": capabilities,
    }


def test_valid_candidate_contract_passes() -> None:
    document = _ledger(_row("M1", pr=1001), _row("T1", pr=1002))
    assert validate_ledger(document) == []


def test_packet_green_is_evidence_not_delivery_state() -> None:
    document = _ledger(_row("M1", pr=1001))
    document["capabilities"][0]["delivery_state"] = "implemented_in_packet"
    assert any(
        "confuses artifact evidence with delivery" in error
        for error in validate_ledger(document)
    )


def test_contract_digest_makes_row_remapping_visible() -> None:
    document = _ledger(
        _row("W1", name="Native webhooks", product_state="rejected")
    )
    document["capabilities"][0]["name"] = "Multi-profile routing"
    assert any("contract_sha256" in error for error in validate_ledger(document))


def test_expected_ids_are_exact_and_ordered() -> None:
    document = _ledger(_row("M1", pr=1001), _row("M2", pr=1002))
    document["capabilities"].reverse()
    document["campaign"]["contract_sha256"] = canonical_contract_digest(
        document["capabilities"]
    )
    assert any(
        "capability ids do not exactly match" in error
        for error in validate_ledger(document)
    )


def test_one_authoritative_pr_cannot_own_two_capabilities() -> None:
    document = _ledger(_row("M1", pr=1001), _row("M2", pr=1001))
    assert any(
        "claimed by multiple capabilities" in error
        for error in validate_ledger(document)
    )


def test_candidate_open_requires_runtime_consumer() -> None:
    document = _ledger(_row("M1", pr=1001))
    document["capabilities"][0]["consumers"] = []
    assert any(
        "requires runtime consumers" in error for error in validate_ledger(document)
    )


def test_candidate_state_requires_exactly_one_authoritative_publication() -> None:
    document = _ledger(_row("M1"))
    assert any(
        "requires exactly one authoritative publication" in error
        for error in validate_ledger(document)
    )


def test_rejected_capability_cannot_sneak_in_production_code() -> None:
    document = _ledger(_row("W1", product_state="rejected"))
    document["capabilities"][0]["implementation_paths"] = [
        "tools/example/webhooks.py"
    ]
    assert any(
        "rejected but declares implementation_paths" in error
        for error in validate_ledger(document)
    )


def test_god_file_growth_is_rejected() -> None:
    document = _ledger(_row("M1", pr=1001))
    document["capabilities"][0]["implementation_paths"] = [
        "plugins/platforms/example/adapter.py"
    ]
    assert any("grows forbidden surface" in error for error in validate_ledger(document))


def test_released_requires_terminal_evidence() -> None:
    row = _row("M1", pr=1001, delivery_state="released")
    row["merged"] = {"commit_sha": "b" * 40}
    document = _ledger(row)
    assert any(
        "release_evidence is required" in error
        for error in validate_ledger(document)
    )


def test_released_with_head_bound_evidence_passes() -> None:
    row = _row("M1", pr=1001, delivery_state="released")
    row["merged"] = {"commit_sha": "b" * 40}
    row["release_evidence"] = {
        "ci_url": "https://github.com/example/repo/actions/runs/1",
        "live_receipt": "receipts/example-live.json",
        "review_a": "approval at bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "review_b": "approval at bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    }
    document = _ledger(row)
    assert validate_ledger(document) == []


def test_cli_reports_all_errors(tmp_path, capsys) -> None:
    from validate_feature_parity_ledger import main

    document = _ledger(_row("M1"))
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main([str(path)]) == 1
    stderr = capsys.readouterr().err
    assert "INVALID" in stderr
    assert "authoritative publication" in stderr
