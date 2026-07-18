#!/usr/bin/env python3
"""Pinned py_webauthn verification boundary for passkey v2.

Production verification is delegated exclusively to py_webauthn 3.0.0.  The
credential public key stays in its WebAuthn COSE representation, matching the
live credential.  No handwritten assertion verifier is selectable.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import re
from typing import Any, Mapping

from scripts.canary import passkey_v2_protocol as protocol


CREDENTIAL_SCHEMA = "muncho-passkey-v2-migrated-credential.v1"
ASSERTION_SCHEMA = "muncho-passkey-v2-assertion.v1"
VERIFICATION_SCHEMA = "muncho-passkey-v2-assertion-verification.v1"
SELECTED_WEBAUTHN_PACKAGE = "webauthn==3.0.0"
SELECTED_CBOR2_PACKAGE = "cbor2==6.1.3"
SELECTED_CRYPTOGRAPHY_PACKAGE = "cryptography==49.0.0"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DISCORD_ID = re.compile(r"^[1-9][0-9]{16,21}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]{2,65536}$")
_CREDENTIAL_FIELDS = frozenset({
    "schema",
    "owner_discord_user_id",
    "expected_user_handle_sha256",
    "credential_id_b64url",
    "credential_id_sha256",
    "public_key_cose_b64url",
    "public_key_cose_sha256",
    "source_public_key_sha256",
    "public_key_byte_count",
    "initial_sign_count",
    "initial_credential_backed_up",
    "rp_id",
    "origin",
    "imported_at_unix",
    "migration_receipt_sha256",
    "status",
    "credential_record_sha256",
})
_ASSERTION_FIELDS = frozenset({"schema", "credential"})


class PasskeyV2WebAuthnError(RuntimeError):
    """Stable py_webauthn boundary failure."""


def _decode_b64url(value: Any, *, label: str, maximum: int) -> bytes:
    if not isinstance(value, str) or _B64URL.fullmatch(value) is None:
        raise PasskeyV2WebAuthnError(f"passkey_v2_{label}_invalid")
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (TypeError, ValueError) as exc:
        raise PasskeyV2WebAuthnError(f"passkey_v2_{label}_invalid") from None
    if not raw or len(raw) > maximum:
        raise PasskeyV2WebAuthnError(f"passkey_v2_{label}_invalid")
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value:
        raise PasskeyV2WebAuthnError(f"passkey_v2_{label}_not_canonical")
    return raw


def _load_selected_verifier() -> Any:
    try:
        from webauthn import verify_authentication_response
        from webauthn.helpers.exceptions import InvalidAuthenticationResponse
    except (ImportError, ModuleNotFoundError) as exc:
        raise PasskeyV2WebAuthnError(
            "passkey_v2_selected_webauthn_runtime_unavailable"
        ) from None
    if (
        importlib.metadata.version("webauthn") != "3.0.0"
        or importlib.metadata.version("cbor2") != "6.1.3"
        or importlib.metadata.version("cryptography") != "49.0.0"
    ):
        raise PasskeyV2WebAuthnError(
            "passkey_v2_selected_webauthn_runtime_mismatch"
        )
    return verify_authentication_response, InvalidAuthenticationResponse


def build_migrated_credential(
    *,
    owner_discord_user_id: str,
    credential_id: bytes,
    public_key_cose: bytes,
    rp_id: str,
    origin: str,
    imported_at_unix: int,
    migration_receipt_sha256: str,
    initial_sign_count: int,
    initial_credential_backed_up: bool,
    expected_user_handle: bytes,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": CREDENTIAL_SCHEMA,
        "owner_discord_user_id": owner_discord_user_id,
        # This comes from the signed v1 registration/migration evidence.  It
        # is deliberately not derived from the Discord owner id: WebAuthn's
        # opaque user.id is registration data and may use another encoding.
        "expected_user_handle_sha256": hashlib.sha256(
            expected_user_handle
        ).hexdigest(),
        # A WebAuthn credential id is public routing material, not a secret.
        # Keeping the exact value makes allowCredentials deterministic and
        # avoids assuming that the migrated credential is discoverable.
        "credential_id_b64url": base64.urlsafe_b64encode(credential_id)
        .rstrip(b"=")
        .decode("ascii"),
        "credential_id_sha256": hashlib.sha256(credential_id).hexdigest(),
        "public_key_cose_b64url": base64.urlsafe_b64encode(public_key_cose)
        .rstrip(b"=")
        .decode("ascii"),
        "public_key_cose_sha256": hashlib.sha256(public_key_cose).hexdigest(),
        "source_public_key_sha256": hashlib.sha256(public_key_cose).hexdigest(),
        "public_key_byte_count": len(public_key_cose),
        "initial_sign_count": initial_sign_count,
        "initial_credential_backed_up": initial_credential_backed_up,
        "rp_id": rp_id,
        "origin": origin,
        "imported_at_unix": imported_at_unix,
        "migration_receipt_sha256": migration_receipt_sha256,
        "status": "active",
    }
    return validate_migrated_credential({
        **unsigned,
        "credential_record_sha256": protocol.sha256_json(unsigned),
    })


def validate_migrated_credential(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CREDENTIAL_FIELDS:
        raise PasskeyV2WebAuthnError("passkey_v2_credential_fields_invalid")
    credential = dict(value)
    if (
        credential.get("schema") != CREDENTIAL_SCHEMA
        or not isinstance(credential.get("owner_discord_user_id"), str)
        or _DISCORD_ID.fullmatch(credential["owner_discord_user_id"]) is None
        or not isinstance(credential.get("credential_id_sha256"), str)
        or _SHA256.fullmatch(credential["credential_id_sha256"]) is None
        or not isinstance(credential.get("expected_user_handle_sha256"), str)
        or _SHA256.fullmatch(credential["expected_user_handle_sha256"]) is None
        or not isinstance(credential.get("public_key_cose_sha256"), str)
        or _SHA256.fullmatch(credential["public_key_cose_sha256"]) is None
        or not isinstance(credential.get("migration_receipt_sha256"), str)
        or _SHA256.fullmatch(credential["migration_receipt_sha256"]) is None
        or credential.get("rp_id") != protocol.PRODUCTION_RP_ID
        or credential.get("origin") != protocol.PRODUCTION_ORIGIN
        or credential.get("status") != "active"
        or not isinstance(credential.get("imported_at_unix"), int)
        or isinstance(credential.get("imported_at_unix"), bool)
        or credential["imported_at_unix"] < 1
        or not isinstance(credential.get("initial_sign_count"), int)
        or isinstance(credential.get("initial_sign_count"), bool)
        or credential["initial_sign_count"] < 0
        or not isinstance(credential.get("initial_credential_backed_up"), bool)
        or not isinstance(credential.get("public_key_byte_count"), int)
        or isinstance(credential.get("public_key_byte_count"), bool)
        or credential["public_key_byte_count"] < 1
        or not isinstance(credential.get("source_public_key_sha256"), str)
        or _SHA256.fullmatch(credential["source_public_key_sha256"]) is None
    ):
        raise PasskeyV2WebAuthnError("passkey_v2_credential_identity_invalid")
    public_key = _decode_b64url(
        credential.get("public_key_cose_b64url"),
        label="credential_public_key",
        maximum=4096,
    )
    credential_id = _decode_b64url(
        credential.get("credential_id_b64url"),
        label="credential_id",
        maximum=4096,
    )
    if (
        hashlib.sha256(credential_id).hexdigest()
        != credential["credential_id_sha256"]
        or hashlib.sha256(public_key).hexdigest()
        != credential["public_key_cose_sha256"]
        or hashlib.sha256(public_key).hexdigest()
        != credential["source_public_key_sha256"]
        or len(public_key) != credential["public_key_byte_count"]
    ):
        raise PasskeyV2WebAuthnError("passkey_v2_credential_public_key_hash_invalid")
    expected = protocol.sha256_json({
        name: item
        for name, item in credential.items()
        if name != "credential_record_sha256"
    })
    if credential.get("credential_record_sha256") != expected:
        raise PasskeyV2WebAuthnError("passkey_v2_credential_record_hash_invalid")
    return credential


def _checked_assertion(assertion: Any) -> Mapping[str, Any]:
    if not isinstance(assertion, Mapping) or set(assertion) != _ASSERTION_FIELDS:
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_fields_invalid")
    if assertion.get("schema") != ASSERTION_SCHEMA:
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_schema_invalid")
    credential = assertion.get("credential")
    if not isinstance(credential, Mapping):
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_credential_invalid")
    # Round-trip through the strict protocol JSON domain before the third-party
    # parser sees the browser object.  Duplicate keys must already have been
    # rejected by the authority HTTP decoder.
    protocol.canonical_json_bytes(credential)
    return dict(credential)


def assertion_credential_id_sha256(assertion: Mapping[str, Any]) -> str:
    credential = _checked_assertion(assertion)
    raw = _decode_b64url(
        credential.get("rawId"),
        label="assertion_credential_id",
        maximum=4096,
    )
    if credential.get("id") != credential.get("rawId"):
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_credential_id_mismatch")
    return hashlib.sha256(raw).hexdigest()


def verify_assertion(
    assertion: Mapping[str, Any],
    *,
    credential: Mapping[str, Any],
    challenge: Mapping[str, Any],
    envelope: Mapping[str, Any],
    prior_sign_count: int,
) -> Mapping[str, Any]:
    action = protocol.validate_action_envelope(envelope)
    protocol.require_production_webauthn_identity(action)
    checked_challenge = protocol.validate_challenge_record(
        challenge, envelope=action
    )
    checked_credential = validate_migrated_credential(credential)
    browser_credential = _checked_assertion(assertion)
    credential_id_sha256 = assertion_credential_id_sha256(assertion)
    if (
        checked_credential["owner_discord_user_id"]
        != action["required_approver_discord_user_id"]
        or credential_id_sha256 != checked_credential["credential_id_sha256"]
    ):
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_credential_mismatch")
    response_for_user_handle = browser_credential.get("response")
    if not isinstance(response_for_user_handle, Mapping):
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_response_invalid")
    encoded_user_handle = response_for_user_handle.get("userHandle")
    if encoded_user_handle is not None:
        user_handle = _decode_b64url(
            encoded_user_handle,
            label="assertion_user_handle",
            maximum=256,
        )
        if (
            hashlib.sha256(user_handle).hexdigest()
            != checked_credential["expected_user_handle_sha256"]
        ):
            raise PasskeyV2WebAuthnError(
                "passkey_v2_assertion_user_handle_mismatch"
            )
    if (
        not isinstance(prior_sign_count, int)
        or isinstance(prior_sign_count, bool)
        or prior_sign_count < 0
    ):
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_counter_invalid")
    public_key = _decode_b64url(
        checked_credential["public_key_cose_b64url"],
        label="credential_public_key",
        maximum=4096,
    )
    challenge_bytes = _decode_b64url(
        checked_challenge["challenge_b64url"],
        label="challenge",
        maximum=4096,
    )
    response_for_client_data = browser_credential.get("response")
    if not isinstance(response_for_client_data, Mapping):
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_response_invalid")
    client_data_precheck = _decode_b64url(
        response_for_client_data.get("clientDataJSON"),
        label="client_data",
        maximum=16 * 1024,
    )

    def client_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in items:
            if name in result:
                raise PasskeyV2WebAuthnError(
                    "passkey_v2_client_data_duplicate_key"
                )
            result[name] = item
        return result

    def reject_constant(_value: str) -> None:
        raise PasskeyV2WebAuthnError("passkey_v2_client_data_number_invalid")

    try:
        client_object = json.loads(
            client_data_precheck.decode("utf-8", errors="strict"),
            object_pairs_hook=client_pairs,
            parse_constant=reject_constant,
            parse_float=reject_constant,
        )
    except PasskeyV2WebAuthnError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PasskeyV2WebAuthnError("passkey_v2_client_data_invalid") from None
    if (
        not isinstance(client_object, Mapping)
        or (
            "crossOrigin" in client_object
            and client_object.get("crossOrigin") is not False
        )
        or "topOrigin" in client_object
    ):
        raise PasskeyV2WebAuthnError("passkey_v2_cross_origin_forbidden")
    verifier, invalid_response = _load_selected_verifier()
    try:
        verified = verifier(
            credential=dict(browser_credential),
            expected_challenge=challenge_bytes,
            expected_rp_id=protocol.PRODUCTION_RP_ID,
            expected_origin=protocol.PRODUCTION_ORIGIN,
            credential_public_key=public_key,
            credential_current_sign_count=prior_sign_count,
            require_user_verification=True,
        )
    except invalid_response as exc:
        raise PasskeyV2WebAuthnError(
            "passkey_v2_assertion_cryptographic_verification_failed"
        ) from None
    if (
        hashlib.sha256(bytes(verified.credential_id)).hexdigest()
        != credential_id_sha256
        or verified.user_verified is not True
    ):
        raise PasskeyV2WebAuthnError("passkey_v2_assertion_result_invalid")
    response = response_for_client_data
    client_data = _decode_b64url(
        response.get("clientDataJSON"), label="client_data", maximum=16 * 1024
    )
    authenticator_data = _decode_b64url(
        response.get("authenticatorData"),
        label="authenticator_data",
        maximum=16 * 1024,
    )
    signature = _decode_b64url(
        response.get("signature"), label="assertion_signature", maximum=1024
    )
    unsigned = {
        "schema": VERIFICATION_SCHEMA,
        "request_id": action["request_id"],
        "action_envelope_sha256": action["envelope_sha256"],
        "challenge_id": checked_challenge["challenge_id"],
        "challenge_record_sha256": checked_challenge[
            "challenge_record_sha256"
        ],
        "approver_discord_user_id": checked_credential[
            "owner_discord_user_id"
        ],
        "credential_id_sha256": credential_id_sha256,
        "credential_record_sha256": checked_credential[
            "credential_record_sha256"
        ],
        "credential_sign_count": int(verified.new_sign_count),
        "credential_device_type": verified.credential_device_type.value,
        "credential_backed_up": bool(verified.credential_backed_up),
        "user_verified": True,
        "rp_id": protocol.PRODUCTION_RP_ID,
        "origin": protocol.PRODUCTION_ORIGIN,
        "verifier_package": SELECTED_WEBAUTHN_PACKAGE,
        "client_data_sha256": hashlib.sha256(client_data).hexdigest(),
        "authenticator_data_sha256": hashlib.sha256(authenticator_data).hexdigest(),
        "assertion_signature_sha256": hashlib.sha256(signature).hexdigest(),
    }
    return {**unsigned, "verification_sha256": protocol.sha256_json(unsigned)}
