"""Canonical routing identities shared by pure selection and persistence."""
import hashlib
import json


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
