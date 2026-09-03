"""Canonical serialization and digest projections for Task 28 Phase 28A."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from htr.state import BoundedActionValidationError

CANONICAL_FIXTURE_BYTES = (
    b'{\n  "record_type": "bounded_action_proposal",\n  "schema_version": "1"\n}\n'
)


def canonical_json(obj: Any) -> str:
    if isinstance(obj, float):
        raise BoundedActionValidationError("floats rejected before serialization")
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        separators=(",", ": "),
    ) + "\n"


def canonical_json_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_digest(payload: dict[str, Any]) -> str:
    encoded = canonical_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def projection_a(intent: dict[str, Any]) -> dict[str, Any]:
    """Caller-intent projection (A) — input must exclude protocol-derived fields."""
    return deepcopy(intent)


def projection_b(record: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(record)
    body.pop("record_digest", None)
    return body


def projection_c(record: dict[str, Any]) -> dict[str, Any]:
    body = projection_b(record)
    body.pop("created_at", None)
    return body


def compute_request_intent_digest(intent: dict[str, Any]) -> str:
    return sha256_digest(projection_a(intent))


def compute_record_digest(record: dict[str, Any]) -> str:
    return sha256_digest(projection_b(record))


def validate_record_digest(record: dict[str, Any]) -> None:
    stored = record.get("record_digest")
    if not isinstance(stored, str) or not stored.startswith("sha256:"):
        raise BoundedActionValidationError("missing or invalid record_digest")
    computed = compute_record_digest(record)
    if stored != computed:
        raise BoundedActionValidationError("record_digest mismatch")
