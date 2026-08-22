"""Contract-registry and strict JSON tests for Feature Parity ledgers."""

from __future__ import annotations

from pathlib import Path
import sys

import json

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_parity_ledger_test_support import (  # noqa: E402
    LedgerValidationError,
    _errors,
    _ledger,
    _registry,
    _row,
    _validator,
    load_json_document,
    validate_contract_registry,

)

def test_registry_revision_one_is_valid() -> None:
    document = _ledger(_row("M1"))
    assert validate_contract_registry(_registry(document)) == []


def test_registry_revisions_are_contiguous_and_linked() -> None:
    document = _ledger(_row("M1"))
    registry = _registry(document)
    first = registry["contracts"]["example-feature-parity"][0]
    second = dict(first)
    second.update(
        {
            "revision": 3,
            "contract_sha256": "b" * 64,
            "previous_contract_sha256": "c" * 64,
            "reason": "Changed semantics.",
        }
    )
    registry["contracts"]["example-feature-parity"].append(second)
    errors = validate_contract_registry(registry)
    assert any("contiguous" in error for error in errors)
    assert any("must match revision 1" in error for error in errors)


def test_registry_revision_after_first_requires_reason() -> None:
    document = _ledger(_row("M1"))
    registry = _registry(document)
    first = registry["contracts"]["example-feature-parity"][0]
    second = dict(first)
    second.update(
        {
            "revision": 2,
            "contract_sha256": "b" * 64,
            "previous_contract_sha256": first["contract_sha256"],
        }
    )
    registry["contracts"]["example-feature-parity"].append(second)
    assert any("reason is required" in error for error in validate_contract_registry(registry))


def test_registry_repository_and_tracker_are_stable() -> None:
    document = _ledger(_row("M1"))
    registry = _registry(document)
    first = registry["contracts"]["example-feature-parity"][0]
    second = dict(first)
    second.update(
        {
            "revision": 2,
            "repository": "other/project",
            "tracker": 999,
            "contract_sha256": "b" * 64,
            "previous_contract_sha256": first["contract_sha256"],
            "reason": "No.",
            "authority": {
                "kind": "issue",
                "number": 999,
                "url": "https://github.com/other/project/issues/999",
            },
        }
    )
    registry["contracts"]["example-feature-parity"].append(second)
    errors = validate_contract_registry(registry)
    assert any("repository cannot change" in error for error in errors)
    assert any("tracker cannot change" in error for error in errors)


def test_unregistered_contract_revision_is_rejected() -> None:
    document = _ledger(_row("M1"))
    document["campaign"]["contract_revision"] = 2
    assert any("is not registered" in error for error in _errors(document, _registry(document)))


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(LedgerValidationError, match="duplicate JSON key"):
        load_json_document(path)


def test_non_utf8_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff")
    with pytest.raises(LedgerValidationError, match="invalid UTF-8"):
        load_json_document(path)


def test_document_size_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_text('{"x": 1}', encoding="utf-8")
    with pytest.raises(LedgerValidationError, match="document limit"):
        load_json_document(path, max_bytes=4)


def test_cli_reports_all_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    document = _ledger(_row("M1"))
    document["capabilities"][0]["consumers"] = []
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert _validator.main([str(path), "--repository-root", str(tmp_path)]) == 1
    stderr = capsys.readouterr().err
    assert "INVALID" in stderr
    assert "runtime consumers" in stderr
