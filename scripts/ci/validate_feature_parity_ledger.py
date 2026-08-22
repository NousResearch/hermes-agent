#!/usr/bin/env python3
"""Validate Feature Parity & Alignment campaign ledgers."""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from feature_parity_ledger import (  # noqa: E402
    LedgerValidationError,
    canonical_contract_digest,
    canonical_contract_payload,
    discover_ledgers,
    load_json_document,
    main,
    validate_contract_registry,
    validate_ledger,
    validate_path,
    validate_repository,
)

__all__ = [
    "LedgerValidationError",
    "canonical_contract_digest",
    "canonical_contract_payload",
    "discover_ledgers",
    "load_json_document",
    "main",
    "validate_contract_registry",
    "validate_ledger",
    "validate_path",
    "validate_repository",
]

if __name__ == "__main__":
    raise SystemExit(main())
