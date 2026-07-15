from __future__ import annotations

import json

import pytest

from gateway import api_verifier_credentials as verifier


BEARER = "api-bearer-for-tests-only-0123456789abcdef"
PASSKEY = "owner-passkey-for-tests-only-0123456789abcdef"


def test_bearer_verifier_is_domain_separated_and_not_reusable() -> None:
    payload = verifier.build_api_bearer_verifier(BEARER)
    parsed = verifier.parse_api_bearer_verifier(payload)

    assert BEARER.encode() not in payload
    assert verifier.api_bearer_matches(parsed, BEARER) is True
    assert verifier.api_bearer_matches(parsed, "x" * 40) is False
    assert verifier.api_bearer_matches(parsed, parsed.sha256_hex) is False
    raw = json.loads(payload)
    assert raw["domain"] == "hermes.api.bearer.v1"
    assert raw["sha256"] != __import__("hashlib").sha256(BEARER.encode()).hexdigest()


def test_bearer_verifier_rejects_noncanonical_or_parameter_drift() -> None:
    payload = verifier.build_api_bearer_verifier(BEARER)
    raw = json.loads(payload)
    raw["algorithm"] = "sha512"
    with pytest.raises(verifier.APIVerifierCredentialError):
        verifier.parse_api_bearer_verifier(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        )
    with pytest.raises(verifier.APIVerifierCredentialError):
        verifier.parse_api_bearer_verifier(payload + b"\n")


def test_approval_verifier_is_salted_bounded_and_not_reusable() -> None:
    payload = verifier.build_api_approval_scrypt_verifier(
        PASSKEY,
        salt=b"s" * 32,
    )
    parsed = verifier.parse_api_approval_scrypt_verifier(payload)

    assert PASSKEY.encode() not in payload
    assert verifier.api_approval_passkey_matches(parsed, PASSKEY) is True
    assert verifier.api_approval_passkey_matches(parsed, "x" * 40) is False
    assert verifier.api_approval_passkey_matches(
        parsed,
        parsed.verifier.hex(),
    ) is False


@pytest.mark.parametrize("field,value", [("n", 2**20), ("r", 9), ("p", 2), ("dklen", 64)])
def test_approval_verifier_rejects_kdf_parameter_drift(field: str, value: int) -> None:
    raw = json.loads(
        verifier.build_api_approval_scrypt_verifier(PASSKEY, salt=b"s" * 32)
    )
    raw[field] = value
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(verifier.APIVerifierCredentialError):
        verifier.parse_api_approval_scrypt_verifier(payload)


def test_builders_reject_short_or_control_bearing_secrets() -> None:
    with pytest.raises(verifier.APIVerifierCredentialError):
        verifier.build_api_bearer_verifier("short")
    with pytest.raises(verifier.APIVerifierCredentialError):
        verifier.build_api_approval_scrypt_verifier("p" * 31)
    with pytest.raises(verifier.APIVerifierCredentialError):
        verifier.build_api_approval_scrypt_verifier("p" * 32 + "\n")
