"""Delivery-state and repository-path tests for Feature Parity ledgers."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_parity_ledger_test_support import (  # noqa: E402
    _errors,
    _ledger,
    _publication,
    _row,

)

def test_candidate_open_requires_runtime_consumer() -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["consumers"] = []
    assert any("requires runtime consumers" in error for error in _errors(document))


def test_candidate_unwired_requires_gap_and_forbids_consumers() -> None:
    document = _ledger(_row("M1", delivery_state="candidate_unwired"))
    row = document["capabilities"][0]
    row["consumers"] = ["plugins/example/runtime.py:M1"]
    del row["wiring_gap"]
    errors = _errors(document)
    assert any("wiring_gap" in error for error in errors)
    assert any("must have no consumers" in error for error in errors)


def test_candidate_blocked_requires_explicit_blocker() -> None:
    document = _ledger(_row("M1", delivery_state="candidate_blocked"))
    del document["capabilities"][0]["blocker"]
    assert any("blocker" in error for error in _errors(document))


def test_gap_requires_reason_or_product_decision() -> None:
    document = _ledger(_row("M1", delivery_state="gap"))
    del document["capabilities"][0]["gap_reason"]
    assert any("gap_reason" in error for error in _errors(document))


def test_superseded_row_has_no_authority_and_names_successor() -> None:
    document = _ledger(_row("M1", delivery_state="superseded"))
    del document["capabilities"][0]["superseded_by"]
    assert any("superseded_by" in error for error in _errors(document))


def test_gap_cannot_retain_authoritative_publication() -> None:
    document = _ledger(_row("M1", delivery_state="gap"))
    document["capabilities"][0]["publications"] = [_publication(1001)]
    assert any("cannot retain authoritative publication" in error for error in _errors(document))


def test_rejected_and_deferred_rows_cannot_accumulate_code() -> None:
    for product_state in ("rejected", "deferred"):
        document = _ledger(_row("W1", product_state=product_state))
        document["capabilities"][0]["implementation_paths"] = ["plugins/example/w1.py"]
        assert any("cannot declare production" in error for error in _errors(document))


def test_decision_gated_row_cannot_promote() -> None:
    document = _ledger(_row("I1", product_state="conditional"))
    assert any("cannot advance" in error for error in _errors(document))


@pytest.mark.parametrize(
    "path",
    [
        "../plugins/example/m1.py",
        "/plugins/example/m1.py",
        "C:/plugins/example/m1.py",
        "plugins\\example\\m1.py",
        "plugins//example/m1.py",
        "plugins/example/",
    ],
)
def test_non_canonical_repository_paths_are_rejected(path: str) -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["implementation_paths"] = [path]
    assert any("implementation_paths[0]" in error for error in _errors(document))


def test_consumer_identifier_requires_path_and_symbol() -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["consumers"] = ["plugins/example/runtime.py"]
    assert any("<path>:<symbol>" in error for error in _errors(document))


def test_forbidden_growth_is_rejected_after_canonicalization() -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["implementation_paths"] = ["plugins/example/adapter.py"]
    assert any("grows forbidden surface" in error for error in _errors(document))


def test_boolean_schema_version_does_not_equal_integer_version() -> None:
    document = _ledger(_row("M1"))
    document["schema_version"] = True
    assert any("schema_version must be 1" in error for error in _errors(document))


def test_invalid_timestamp_is_rejected() -> None:
    document = _ledger(_row("M1"))
    document["snapshot"]["captured_at"] = "not-a-timeZ"
    assert any("valid RFC 3339" in error for error in _errors(document))
