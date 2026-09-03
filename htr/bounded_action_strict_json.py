"""Strict JSON parsing for Task 28 Phase 28A records."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from htr.state import BoundedActionValidationError

MAX_RECORD_BYTES = 262144
MAX_JSON_DEPTH = 32
MAX_ARRAY_LENGTH = 256
MAX_OBJECT_FIELDS = 128
MAX_STRING_CODEPOINTS = 8192
MAX_TOTAL_DECODED_STRING_CODEPOINTS = 512000
MAX_REASON_CODES = 16
MAX_PROPOSAL_SUMMARY_CODEPOINTS = 4096
MAX_REASON_DETAIL_CODEPOINTS = 1024

_PROHIBITED_CONTROLS = frozenset(range(0, 32)) - {9, 10, 13}


def _count_codepoints(value: str) -> int:
    return len(value)


def _validate_string(value: str, *, field: str, max_codepoints: int | None = None) -> str:
    if "\x00" in value:
        raise BoundedActionValidationError(f"{field}: NUL prohibited")
    for ch in value:
        if ord(ch) in _PROHIBITED_CONTROLS:
            raise BoundedActionValidationError(f"{field}: prohibited control character")
    if unicodedata.normalize("NFC", value) != value:
        raise BoundedActionValidationError(f"{field}: string must be NFC")
    limit = max_codepoints if max_codepoints is not None else MAX_STRING_CODEPOINTS
    if _count_codepoints(value) > limit:
        raise BoundedActionValidationError(f"{field}: exceeds max codepoints")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise BoundedActionValidationError("duplicate JSON keys rejected")
        seen.add(key)
        obj[key] = value
    return obj


def _walk_validate(
    obj: Any,
    *,
    depth: int = 0,
    string_budget: list[int],
) -> None:
    if depth > MAX_JSON_DEPTH:
        raise BoundedActionValidationError("JSON depth exceeds limit")
    if isinstance(obj, dict):
        if len(obj) > MAX_OBJECT_FIELDS:
            raise BoundedActionValidationError("object field count exceeds limit")
        for key, val in obj.items():
            if not isinstance(key, str):
                raise BoundedActionValidationError("object keys must be strings")
            _validate_string(key, field="object key")
            _walk_validate(val, depth=depth + 1, string_budget=string_budget)
    elif isinstance(obj, list):
        if len(obj) > MAX_ARRAY_LENGTH:
            raise BoundedActionValidationError("array length exceeds limit")
        for item in obj:
            _walk_validate(item, depth=depth + 1, string_budget=string_budget)
    elif isinstance(obj, str):
        _validate_string(obj, field="string value")
        string_budget[0] += _count_codepoints(obj)
        if string_budget[0] > MAX_TOTAL_DECODED_STRING_CODEPOINTS:
            raise BoundedActionValidationError("total decoded string codepoints exceeds limit")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, float):
        raise BoundedActionValidationError("floats rejected")
    elif isinstance(obj, int):
        if obj < 0 or obj > (2**63 - 1):
            raise BoundedActionValidationError("integer out of range")
    elif obj is None:
        return
    else:
        raise BoundedActionValidationError(f"unsupported JSON type: {type(obj)!r}")


def parse_strict_json_bytes(raw: bytes) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise BoundedActionValidationError("UTF-8 BOM rejected")
    if len(raw) > MAX_RECORD_BYTES:
        raise BoundedActionValidationError("record exceeds MAX_RECORD_BYTES")
    if not raw.endswith(b"\n"):
        raise BoundedActionValidationError("record must end with LF")
    if len(raw) >= 2 and raw.endswith(b"\n\n"):
        raise BoundedActionValidationError("record must end with exactly one trailing LF")
    if raw.rstrip(b"\n") != raw[:-1]:
        raise BoundedActionValidationError("no bytes permitted after final LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BoundedActionValidationError("malformed UTF-8") from exc
    try:
        obj = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=lambda _: (_ for _ in ()).throw(ValueError("float")),
        )
    except json.JSONDecodeError as exc:
        raise BoundedActionValidationError(f"malformed JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise BoundedActionValidationError("record root must be object")
    _walk_validate(obj, string_budget=[0])
    return obj


def require_exact_canonical_bytes(raw: bytes, canonical: bytes) -> None:
    if raw != canonical:
        raise BoundedActionValidationError("noncanonical stored bytes")
