"""Canonical request encoding and hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def signed_material(
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    request_sha256: str,
) -> bytes:
    return "\n".join(
        (method.upper(), path, timestamp, nonce, request_sha256)
    ).encode("utf-8")
