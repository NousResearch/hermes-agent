"""Canonical payload identity after authorization and immutable media capture."""
import hashlib
import json


def admission_fingerprint(*, canonical_target: str, payload: dict) -> str:
    encoded = json.dumps(
        {'target': canonical_target, 'payload': payload}, ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8', errors='surrogatepass')
    return hashlib.sha256(encoded).hexdigest()
