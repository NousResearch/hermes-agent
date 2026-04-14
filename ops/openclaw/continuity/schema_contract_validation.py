#!/usr/bin/env python3
"""Shared fail-close schema validation helpers for continuity operator surfaces."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

try:  # pragma: no cover - dependency wiring is validated in caller tests
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


def _json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(part) for part in seq)


def validate_contract_payload_schema(
    payload: Dict[str, Any],
    *,
    schema_path: pathlib.Path,
    contract_prefix: str,
) -> None:
    """Validate payload against schema with deterministic fail-close error shape.

    Raises RuntimeError with one of the following shapes:
    - "<prefix>_validator_unavailable"
    - "<prefix>_schema_missing:<schema_path>"
    - "<prefix>_schema_not_object"
    - "<prefix>_schema_validation_failed:data_path=<ptr>:schema_path=<ptr>:error=<msg>"
    """

    if Draft202012Validator is None or FormatChecker is None:
        raise RuntimeError(f"{contract_prefix}_validator_unavailable")
    if not schema_path.exists():
        raise RuntimeError(f"{contract_prefix}_schema_missing:{schema_path}")

    schema_doc = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema_doc, dict):
        raise RuntimeError(f"{contract_prefix}_schema_not_object")

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return

    err = errors[0]
    data_ptr = _json_ptr(err.absolute_path)
    schema_ptr = _json_ptr(err.absolute_schema_path)
    raise RuntimeError(
        f"{contract_prefix}_schema_validation_failed:"
        f"data_path={data_ptr}:schema_path={schema_ptr}:error={err.message}"
    )
