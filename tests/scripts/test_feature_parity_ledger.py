"""Core contract and publication-authority tests for Feature Parity ledgers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_parity_ledger_test_support import (  # noqa: E402
    _errors,
    _ledger,
    _registry,
    _row,
    canonical_contract_digest,

)

def test_valid_candidate_contract_passes() -> None:
    document = _ledger(_row("M1", pr=1001), _row("T1", pr=1002))
    assert _errors(document, _registry(document)) == []


def test_packet_green_is_evidence_not_delivery_state() -> None:
    document = _ledger(_row("M1", pr=1001))
    document["capabilities"][0]["delivery_state"] = "implemented_in_packet"
    assert any("confuses artifact evidence with delivery" in error for error in _errors(document))


def test_external_registry_prevents_self_authorized_row_remap() -> None:
    document = _ledger(_row("W1", name="Native webhooks", product_state="rejected"))
    registry = _registry(document)
    document["capabilities"][0]["name"] = "Multiplex routing"
    document["campaign"]["contract_sha256"] = canonical_contract_digest(document["capabilities"])
    assert any("external registry" in error for error in _errors(document, registry))


def test_contract_digest_includes_source_anchor() -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["source_anchor"] = "A different spec"
    assert any("contract_sha256" in error for error in _errors(document))


def test_expected_ids_are_required_non_empty_exact_and_ordered() -> None:
    document = _ledger(_row("M1"), _row("M2", pr=1002))
    document["campaign"]["expected_capability_ids"] = []
    errors = _errors(document)
    assert any("must not be empty" in error for error in errors)
    assert any("do not exactly match" in error for error in errors)


def test_expected_id_order_drift_is_rejected_even_with_new_digest() -> None:
    document = _ledger(_row("M1"), _row("M2", pr=1002))
    document["capabilities"].reverse()
    document["campaign"]["contract_sha256"] = canonical_contract_digest(document["capabilities"])
    assert any("order differs" in error for error in _errors(document))


def test_duplicate_capability_ids_are_rejected() -> None:
    document = _ledger(_row("M1", pr=1001), _row("M1", pr=1002))
    assert any("duplicate capability ids" in error for error in _errors(document))


def test_malformed_capability_is_reported_without_digest_crash() -> None:
    document = _ledger(_row("M1"))
    document["capabilities"].append("not-an-object")
    errors = _errors(document)
    assert any("capabilities[1] must be an object" in error for error in errors)


def test_required_row_lists_cannot_be_omitted() -> None:
    document = _ledger(_row("M1"))
    del document["capabilities"][0]["artifact_evidence"]
    assert any("artifact_evidence is required" in error for error in _errors(document))


def test_authoritative_publication_must_be_a_pull_request() -> None:
    document = _ledger(_row("M1"))
    publication = document["capabilities"][0]["publications"][0]
    publication.update({"kind": "issue", "url": "https://github.com/example/project/issues/1001"})
    assert any("must be a pull request" in error for error in _errors(document))


def test_candidate_authority_must_be_open() -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["publications"][0]["state"] = "closed"
    assert any("must be open for candidate delivery" in error for error in _errors(document))


def test_candidate_open_requires_exact_head_sha() -> None:
    document = _ledger(_row("M1"))
    del document["capabilities"][0]["publications"][0]["head_sha"]
    assert any("head_sha" in error for error in _errors(document))


def test_one_authoritative_pr_cannot_own_two_capabilities() -> None:
    document = _ledger(_row("M1", pr=1001), _row("M2", pr=1001))
    assert any("claimed by multiple capabilities" in error for error in _errors(document))
