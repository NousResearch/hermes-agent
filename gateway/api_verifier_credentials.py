"""Mechanical verifier-only credentials for the Hermes API boundary.

Production gateway processes must not retain a reusable API bearer or owner
approval passkey.  This module defines two strict, versioned verifier formats:

* a domain-separated SHA-256 verifier for a high-entropy API bearer; and
* a domain-separated, fixed-parameter scrypt verifier for an owner passkey.

The builders belong on a trusted staging edge.  Runtime code only parses the
public verifier and compares a caller-supplied candidate in constant time.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


API_BEARER_VERIFIER_SCHEMA = "hermes.api.bearer-sha256-verifier.v1"
API_APPROVAL_VERIFIER_SCHEMA = "hermes.api.approval-scrypt-verifier.v1"
API_BEARER_DOMAIN = b"hermes.api.bearer.v1\x00"
API_APPROVAL_DOMAIN = b"hermes.api.approval-passkey.v1\x00"

SCRYPT_N = 16_384
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SCRYPT_SALT_BYTES = 32
SCRYPT_MAXMEM = 64 * 1024 * 1024

MIN_BEARER_BYTES = 32
MIN_PASSKEY_BYTES = 32
MAX_CANDIDATE_BYTES = 8_192
MAX_VERIFIER_BYTES = 8_192

_HEX_32 = re.compile(r"^[0-9a-f]{64}$")


class APIVerifierCredentialError(ValueError):
    """Stable, secret-free verifier credential failure."""


@dataclass(frozen=True)
class APIBearerVerifier:
    sha256_hex: str


@dataclass(frozen=True)
class APIApprovalScryptVerifier:
    salt: bytes
    verifier: bytes


def _candidate_bytes(value: Any, *, minimum: int, label: str) -> bytes:
    if not isinstance(value, str) or value != value.strip():
        raise APIVerifierCredentialError(f"{label}_invalid")
    try:
        raw = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise APIVerifierCredentialError(f"{label}_invalid") from exc
    if (
        not minimum <= len(raw) <= MAX_CANDIDATE_BYTES
        or any(byte < 0x20 or byte == 0x7F for byte in raw)
    ):
        raise APIVerifierCredentialError(f"{label}_invalid")
    return raw


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _strict_object(raw: str | bytes, *, label: str) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            payload = raw.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise APIVerifierCredentialError(f"{label}_invalid") from exc
    elif isinstance(raw, bytes):
        payload = raw
    else:
        raise APIVerifierCredentialError(f"{label}_invalid")
    if not 1 <= len(payload) <= MAX_VERIFIER_BYTES:
        raise APIVerifierCredentialError(f"{label}_invalid")
    try:
        value = json.loads(payload.decode("ascii", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise APIVerifierCredentialError(f"{label}_invalid") from exc
    if not isinstance(value, Mapping) or _canonical_bytes(value) != payload:
        raise APIVerifierCredentialError(f"{label}_invalid")
    return value


def _bearer_digest(token: bytes) -> str:
    return hashlib.sha256(API_BEARER_DOMAIN + token).hexdigest()


def build_api_bearer_verifier(token: str) -> bytes:
    """Build canonical public verifier bytes on a trusted staging edge."""

    candidate = _candidate_bytes(
        token,
        minimum=MIN_BEARER_BYTES,
        label="api_bearer",
    )
    return _canonical_bytes(
        {
            "schema": API_BEARER_VERIFIER_SCHEMA,
            "algorithm": "sha256",
            "domain": API_BEARER_DOMAIN[:-1].decode("ascii"),
            "sha256": _bearer_digest(candidate),
        }
    )


def parse_api_bearer_verifier(raw: str | bytes) -> APIBearerVerifier:
    value = _strict_object(raw, label="api_bearer_verifier")
    if (
        set(value) != {"schema", "algorithm", "domain", "sha256"}
        or value.get("schema") != API_BEARER_VERIFIER_SCHEMA
        or value.get("algorithm") != "sha256"
        or value.get("domain") != API_BEARER_DOMAIN[:-1].decode("ascii")
        or not isinstance(value.get("sha256"), str)
        or _HEX_32.fullmatch(value["sha256"]) is None
    ):
        raise APIVerifierCredentialError("api_bearer_verifier_invalid")
    return APIBearerVerifier(sha256_hex=value["sha256"])


def api_bearer_matches(verifier: APIBearerVerifier, candidate: Any) -> bool:
    try:
        raw = _candidate_bytes(
            candidate,
            minimum=MIN_BEARER_BYTES,
            label="api_bearer",
        )
    except APIVerifierCredentialError:
        return False
    return hmac.compare_digest(_bearer_digest(raw), verifier.sha256_hex)


def _approval_scrypt(passkey: bytes, salt: bytes) -> bytes:
    try:
        return hashlib.scrypt(
            API_APPROVAL_DOMAIN + passkey,
            salt=salt,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
            dklen=SCRYPT_DKLEN,
            maxmem=SCRYPT_MAXMEM,
        )
    except (TypeError, ValueError) as exc:
        raise APIVerifierCredentialError("api_approval_verifier_invalid") from exc


def build_api_approval_scrypt_verifier(
    passkey: str,
    *,
    salt: bytes | None = None,
) -> bytes:
    """Build a canonical scrypt verifier on a trusted staging edge."""

    candidate = _candidate_bytes(
        passkey,
        minimum=MIN_PASSKEY_BYTES,
        label="api_approval_passkey",
    )
    actual_salt = os.urandom(SCRYPT_SALT_BYTES) if salt is None else salt
    if not isinstance(actual_salt, bytes) or len(actual_salt) != SCRYPT_SALT_BYTES:
        raise APIVerifierCredentialError("api_approval_salt_invalid")
    verifier = _approval_scrypt(candidate, actual_salt)
    return _canonical_bytes(
        {
            "schema": API_APPROVAL_VERIFIER_SCHEMA,
            "algorithm": "scrypt",
            "domain": API_APPROVAL_DOMAIN[:-1].decode("ascii"),
            "n": SCRYPT_N,
            "r": SCRYPT_R,
            "p": SCRYPT_P,
            "dklen": SCRYPT_DKLEN,
            "salt_hex": actual_salt.hex(),
            "verifier_hex": verifier.hex(),
        }
    )


def parse_api_approval_scrypt_verifier(
    raw: str | bytes,
) -> APIApprovalScryptVerifier:
    value = _strict_object(raw, label="api_approval_verifier")
    if (
        set(value)
        != {
            "schema",
            "algorithm",
            "domain",
            "n",
            "r",
            "p",
            "dklen",
            "salt_hex",
            "verifier_hex",
        }
        or value.get("schema") != API_APPROVAL_VERIFIER_SCHEMA
        or value.get("algorithm") != "scrypt"
        or value.get("domain") != API_APPROVAL_DOMAIN[:-1].decode("ascii")
        or type(value.get("n")) is not int
        or value.get("n") != SCRYPT_N
        or type(value.get("r")) is not int
        or value.get("r") != SCRYPT_R
        or type(value.get("p")) is not int
        or value.get("p") != SCRYPT_P
        or type(value.get("dklen")) is not int
        or value.get("dklen") != SCRYPT_DKLEN
        or not isinstance(value.get("salt_hex"), str)
        or _HEX_32.fullmatch(value["salt_hex"]) is None
        or not isinstance(value.get("verifier_hex"), str)
        or _HEX_32.fullmatch(value["verifier_hex"]) is None
    ):
        raise APIVerifierCredentialError("api_approval_verifier_invalid")
    return APIApprovalScryptVerifier(
        salt=bytes.fromhex(value["salt_hex"]),
        verifier=bytes.fromhex(value["verifier_hex"]),
    )


def api_approval_passkey_matches(
    verifier: APIApprovalScryptVerifier,
    candidate: Any,
) -> bool:
    try:
        raw = _candidate_bytes(
            candidate,
            minimum=MIN_PASSKEY_BYTES,
            label="api_approval_passkey",
        )
        observed = _approval_scrypt(raw, verifier.salt)
    except APIVerifierCredentialError:
        return False
    return hmac.compare_digest(observed, verifier.verifier)


__all__ = [
    "API_APPROVAL_VERIFIER_SCHEMA",
    "APIApprovalScryptVerifier",
    "API_BEARER_VERIFIER_SCHEMA",
    "APIBearerVerifier",
    "APIVerifierCredentialError",
    "api_approval_passkey_matches",
    "api_bearer_matches",
    "build_api_approval_scrypt_verifier",
    "build_api_bearer_verifier",
    "parse_api_approval_scrypt_verifier",
    "parse_api_bearer_verifier",
]
