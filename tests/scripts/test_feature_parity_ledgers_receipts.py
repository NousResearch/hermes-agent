"""Repository-wide receipt tests for Feature Parity ledgers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_parity_ledger_repository_test_support import (  # noqa: E402
    _ledger,
    _write_ledger,
    _write_registry,
    validate_repository,

)

def test_released_receipt_must_exist(tmp_path: Path) -> None:
    document = _ledger("alpha-parity", 101, "M1", 1001, released=True)
    _write_registry(
        tmp_path,
        ("alpha-parity", document["campaign"]["contract_sha256"], 101, 101),
    )
    _write_ledger(tmp_path, "alpha", document)
    assert any("live receipt does not exist" in error for error in validate_repository(tmp_path))


def test_released_receipt_hash_is_verified(tmp_path: Path) -> None:
    receipt = tmp_path / "receipts" / "live.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"live": true}', encoding="utf-8")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    document = _ledger(
        "alpha-parity",
        101,
        "M1",
        1001,
        released=True,
        receipt_digest=digest,
    )
    _write_registry(
        tmp_path,
        ("alpha-parity", document["campaign"]["contract_sha256"], 101, 101),
    )
    _write_ledger(tmp_path, "alpha", document)
    assert validate_repository(tmp_path) == []


def test_released_receipt_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    receipt = tmp_path / "receipts" / "live.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"live": true}', encoding="utf-8")
    document = _ledger("alpha-parity", 101, "M1", 1001, released=True)
    _write_registry(
        tmp_path,
        ("alpha-parity", document["campaign"]["contract_sha256"], 101, 101),
    )
    _write_ledger(tmp_path, "alpha", document)
    assert any("live receipt hash mismatch" in error for error in validate_repository(tmp_path))
