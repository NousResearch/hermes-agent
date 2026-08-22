"""Repository-wide ownership tests for Feature Parity ledgers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_parity_ledger_repository_test_support import (  # noqa: E402
    _ledger,
    _write_ledger,
    _write_registry,
    discover_ledgers,
    validate_repository,

)

def test_empty_registry_and_no_ledgers_is_valid(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    assert validate_repository(tmp_path) == []


def test_discovery_excludes_contract_registry(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    document = _ledger("alpha-parity", 101, "M1", 1001)
    _write_ledger(tmp_path, "alpha", document)
    assert [path.name for path in discover_ledgers(tmp_path)] == ["alpha.json"]


def test_registered_repository_ledger_passes(tmp_path: Path) -> None:
    document = _ledger("alpha-parity", 101, "M1", 1001)
    _write_registry(
        tmp_path,
        ("alpha-parity", document["campaign"]["contract_sha256"], 101, 101),
    )
    _write_ledger(tmp_path, "alpha", document)
    assert validate_repository(tmp_path) == []


def test_duplicate_campaign_ids_are_rejected_across_ledgers(tmp_path: Path) -> None:
    first = _ledger("alpha-parity", 101, "M1", 1001)
    second = _ledger("alpha-parity", 102, "M2", 1002)
    _write_registry(
        tmp_path,
        ("alpha-parity", first["campaign"]["contract_sha256"], 101, 101),
    )
    _write_ledger(tmp_path, "alpha", first)
    _write_ledger(tmp_path, "beta", second)
    assert any("campaign id 'alpha-parity' is duplicated" in error for error in validate_repository(tmp_path))


def test_duplicate_tracker_is_rejected_across_ledgers(tmp_path: Path) -> None:
    first = _ledger("alpha-parity", 101, "M1", 1001)
    second = _ledger("beta-parity", 101, "M2", 1002)
    _write_registry(
        tmp_path,
        ("alpha-parity", first["campaign"]["contract_sha256"], 101, 101),
        ("beta-parity", second["campaign"]["contract_sha256"], 101, 101),
    )
    _write_ledger(tmp_path, "alpha", first)
    _write_ledger(tmp_path, "beta", second)
    assert any("tracker issue #101 is duplicated" in error for error in validate_repository(tmp_path))


def test_authoritative_pr_cannot_be_claimed_across_ledgers(tmp_path: Path) -> None:
    first = _ledger("alpha-parity", 101, "M1", 1001)
    second = _ledger("beta-parity", 102, "M2", 1001)
    _write_registry(
        tmp_path,
        ("alpha-parity", first["campaign"]["contract_sha256"], 101, 101),
        ("beta-parity", second["campaign"]["contract_sha256"], 102, 102),
    )
    _write_ledger(tmp_path, "alpha", first)
    _write_ledger(tmp_path, "beta", second)
    assert any("example/project#1001 is claimed across ledgers" in error for error in validate_repository(tmp_path))
