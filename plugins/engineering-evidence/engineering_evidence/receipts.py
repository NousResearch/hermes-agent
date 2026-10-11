"""Validation for immutable, diagnostic-only engineering receipts."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any


class ReceiptValidationError(ValueError):
    """Raised when a receipt cannot be trusted as a bounded evidence record."""


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED = ("schema_name", "receipt_type", "receipt_id", "repository", "head_sha", "authority")
_RECEIPT_TYPES = {
    "codebase_snapshot",
    "test_discovery",
    "experience_candidate",
    "deployment_correlation",
}


def validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive copy of a valid v1 receipt.

    Receipts are deliberately diagnostic-only. They can support review and learning,
    but never satisfy an authority gate by themselves.
    """

    if not isinstance(receipt, Mapping):
        raise ReceiptValidationError("receipt must be an object")
    missing = [key for key in _REQUIRED if not str(receipt.get(key) or "").strip()]
    if missing:
        raise ReceiptValidationError(f"missing receipt fields: {', '.join(missing)}")
    if receipt["schema_name"] != "engineering_evidence_v1":
        raise ReceiptValidationError("unsupported schema_name")
    if receipt["receipt_type"] not in _RECEIPT_TYPES:
        raise ReceiptValidationError("unsupported receipt_type")
    if receipt["authority"] != "diagnostic-only":
        raise ReceiptValidationError("authority must be diagnostic-only")
    if not _SHA_RE.fullmatch(str(receipt["head_sha"])):
        raise ReceiptValidationError("head_sha must be a 40-character hexadecimal commit")
    for field in ("receipt_id", "repository"):
        if len(str(receipt[field]).strip()) > 240:
            raise ReceiptValidationError(f"{field} is too long")
    return copy.deepcopy(dict(receipt))


def validate_exact_head(receipt: Mapping[str, Any], current_head: str) -> dict[str, Any]:
    """Validate a receipt and reject it when it is stale for ``current_head``."""

    validated = validate_receipt(receipt)
    if not _SHA_RE.fullmatch(str(current_head)):
        raise ReceiptValidationError("current_head must be a 40-character hexadecimal commit")
    if validated["head_sha"] != current_head:
        raise ReceiptValidationError("stale head: receipt does not match current repository HEAD")
    return validated
