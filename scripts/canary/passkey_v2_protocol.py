#!/usr/bin/env python3
"""Canonical, exact-action protocol for the Muncho passkey v2 boundary.

This module contains no routing or task semantics.  It defines the immutable
documents passed between an unprivileged WebAuthn UI and the privileged,
single-use grant consumer.  Every protocol document is strict canonical JSON:
unknown fields, duplicate object keys, floating point values, non-finite
numbers, non-UTF-8 input, and non-canonical encodings are rejected.

The v2 protocol deliberately does not support TOTP for dangerous actions.
Recovery or credential enrolment is a separate workflow and cannot mint an
execution grant.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


CANONICALIZATION_VERSION = "muncho-json-utf8-v1"
ACTION_ENVELOPE_SCHEMA = "muncho-dangerous-action-envelope.v2"
CHALLENGE_SCHEMA = "muncho-dangerous-action-passkey-challenge.v2"
GRANT_SCHEMA = "muncho-dangerous-action-passkey-grant.v2"
AUTHORIZATION_RECEIPT_SCHEMA = (
    "muncho-dangerous-action-passkey-authorization-receipt.v2"
)
UI_VIEW_SCHEMA = "muncho-dangerous-action-passkey-ui-view.v2"

MINIMUM_TTL_SECONDS = 30
MAXIMUM_TTL_SECONDS = 300
MAXIMUM_JSON_BYTES = 1024 * 1024
MAXIMUM_ACTION_PAYLOAD_BYTES = 512 * 1024
DANGEROUS_TOTP_ENABLED = False
GENESIS_JOURNAL_HEAD_SHA256 = "0" * 64
EXECUTION_WINDOW_SECONDS = 3600
PRODUCTION_RP_ID = "lomliev.com"
PRODUCTION_ORIGIN = "https://auth.lomliev.com"

DANGEROUS_SCOPES = frozenset({
    "cloud_secret_change",
    "db_write",
    "deploy_restart",
    "fiscal_command",
    "gateway_restart_or_owner_switch",
    "production_write",
    "raw_export",
    "runtime_config_mutation",
})

UI_SECURITY_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self'; form-action 'self'; "
        "frame-ancestors 'none'; base-uri 'none'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Pragma": "no-cache",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
_GRANT_ID = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
_CHALLENGE_ID = re.compile(r"^[A-Za-z0-9_-]{32,64}$")
_DISCORD_ID = re.compile(r"^[1-9][0-9]{16,21}$")
_CASE_ID = re.compile(r"^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,190}$")
_STAGE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,254}$")
_RP_ID = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_ORIGIN = re.compile(
    r"^https://(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_B64URL = re.compile(r"^[A-Za-z0-9_-]{32,1024}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

_ACTION_FIELDS = frozenset({
    "schema",
    "canonicalization",
    "request_id",
    "requester_discord_user_id",
    "required_approver_discord_user_id",
    "scope",
    "case_id",
    "target_system",
    "action_summary",
    "risk",
    "rollback",
    "action_payload",
    "action_payload_sha256",
    "executor_release_sha",
    "executor_plan_sha256",
    "transaction_id",
    "stage",
    "webauthn_rp_id",
    "webauthn_origin",
    "authority_release_sha",
    "authority_manifest_sha256",
    "authority_host_receipt_sha256",
    "source_preflight_sha256",
    "live_projection_sha256",
    "external_iam_receipt_sha256",
    "prior_authoritative_receipt_sha256",
    "prior_event_head_sha256",
    "execution_window_seconds",
    "issued_at_unix",
    "expires_at_unix",
    "approval_ttl_seconds",
    "envelope_sha256",
})
_CHALLENGE_FIELDS = frozenset({
    "schema",
    "canonicalization",
    "state",
    "request_id",
    "action_envelope_sha256",
    "challenge_id",
    "challenge_b64url",
    "challenge_sha256",
    "rp_id",
    "origin",
    "created_at_unix",
    "expires_at_unix",
    "challenge_record_sha256",
})
_GRANT_FIELDS = frozenset({
    "schema",
    "canonicalization",
    "state",
    "method",
    "single_use",
    "request_id",
    "action_envelope_sha256",
    "challenge_id",
    "challenge_record_sha256",
    "grant_id",
    "approver_discord_user_id",
    "credential_id_sha256",
    "credential_record_sha256",
    "credential_migration_receipt_sha256",
    "assertion_verification_sha256",
    "credential_sign_count",
    "credential_backed_up",
    "user_verified",
    "rp_id",
    "origin",
    "granted_at_unix",
    "expires_at_unix",
    "grant_sha256",
})
_RUNTIME_FIELDS = frozenset({
    "executor_release_sha",
    "executor_plan_sha256",
    "executor_binary_sha256",
    "mutation_wrapper_sha256",
    "remote_transport_sha256",
    "runtime_binding_sha256",
})
_RECEIPT_FIELDS = frozenset({
    "schema",
    "canonicalization",
    "outcome",
    "mutation_authorized",
    "mutation_executed",
    "authorization_disposition",
    "consume_attempt_id",
    "request_id",
    "action_envelope_sha256",
    "action_payload_sha256",
    "scope",
    "case_id",
    "target_system",
    "transaction_id",
    "stage",
    "grant_id",
    "grant_sha256",
    "approver_discord_user_id",
    "credential_id_sha256",
    "credential_record_sha256",
    "credential_migration_receipt_sha256",
    "assertion_verification_sha256",
    "credential_sign_count",
    "credential_backed_up",
    "approval_method",
    "challenge_id",
    "challenge_record_sha256",
    "granted_at_unix",
    "grant_expires_at_unix",
    "consumed_at_unix",
    "execution_window_seconds",
    "execution_window_expires_at_unix",
    "authority_release_sha",
    "authority_manifest_sha256",
    "authority_host_receipt_sha256",
    "source_preflight_sha256",
    "live_projection_sha256",
    "external_iam_receipt_sha256",
    "prior_authoritative_receipt_sha256",
    "prior_event_head_sha256",
    "runtime_binding",
    "prior_journal_head_sha256",
    "receipt_public_key_id",
    "signature_ed25519_b64url",
    "receipt_sha256",
})


class PasskeyV2ProtocolError(RuntimeError):
    """A stable fail-closed protocol validation error."""


def _decode_canonical_b64url(
    value: Any,
    *,
    label: str,
    minimum_bytes: int,
    maximum_bytes: int,
) -> bytes:
    if not isinstance(value, str) or _B64URL.fullmatch(value) is None:
        raise PasskeyV2ProtocolError(f"passkey_v2_{label}_invalid")
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (TypeError, ValueError) as exc:
        raise PasskeyV2ProtocolError(f"passkey_v2_{label}_invalid") from None
    canonical = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if (
        not minimum_bytes <= len(raw) <= maximum_bytes
        or canonical != value
    ):
        raise PasskeyV2ProtocolError(f"passkey_v2_{label}_invalid")
    return raw


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in items:
        if name in value:
            raise PasskeyV2ProtocolError("passkey_v2_duplicate_json_key")
        value[name] = item
    return value


def _reject_constant(_value: str) -> None:
    raise PasskeyV2ProtocolError("passkey_v2_nonfinite_number")


def _reject_float(_value: str) -> None:
    raise PasskeyV2ProtocolError("passkey_v2_floating_number_forbidden")


def _validate_json_domain(value: Any, *, depth: int = 0) -> None:
    if depth > 64:
        raise PasskeyV2ProtocolError("passkey_v2_json_nesting_invalid")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        raise PasskeyV2ProtocolError("passkey_v2_floating_number_forbidden")
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise PasskeyV2ProtocolError("passkey_v2_string_invalid") from None
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise PasskeyV2ProtocolError("passkey_v2_string_invalid")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_domain(item, depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise PasskeyV2ProtocolError("passkey_v2_object_key_invalid")
            _validate_json_domain(name, depth=depth + 1)
            _validate_json_domain(item, depth=depth + 1)
        return
    raise PasskeyV2ProtocolError("passkey_v2_json_type_invalid")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the v1 canonical JSON profile.

    The profile intentionally excludes floating point numbers.  All protocol
    times, counters, and sizes are integers, so excluding floats removes
    cross-runtime number formatting ambiguity.
    """

    _validate_json_domain(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PasskeyV2ProtocolError("passkey_v2_json_invalid") from None


def decode_canonical_json(
    raw: bytes,
    *,
    maximum_bytes: int = MAXIMUM_JSON_BYTES,
) -> Any:
    """Strictly decode one byte-exact v1 canonical JSON document."""

    if not isinstance(raw, bytes) or not raw or len(raw) > maximum_bytes:
        raise PasskeyV2ProtocolError("passkey_v2_json_size_invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
        )
    except PasskeyV2ProtocolError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PasskeyV2ProtocolError("passkey_v2_json_invalid") from None
    if raw != canonical_json_bytes(value):
        raise PasskeyV2ProtocolError("passkey_v2_json_not_canonical")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _unsigned(value: Mapping[str, Any], digest_name: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != digest_name}


def _text(value: Any, *, label: str, minimum: int, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or value != value.strip()
        or _CONTROL.search(value) is not None
    ):
        raise PasskeyV2ProtocolError(f"passkey_v2_{label}_invalid")
    return value


def validate_request_id(value: Any) -> str:
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        raise PasskeyV2ProtocolError("passkey_v2_request_id_invalid")
    return value


def validate_grant_id(value: Any) -> str:
    if not isinstance(value, str) or _GRANT_ID.fullmatch(value) is None:
        raise PasskeyV2ProtocolError("passkey_v2_grant_id_invalid")
    return value


def _validate_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PasskeyV2ProtocolError(f"passkey_v2_{label}_invalid")
    return value


def _validate_discord_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _DISCORD_ID.fullmatch(value) is None:
        raise PasskeyV2ProtocolError(f"passkey_v2_{label}_invalid")
    return value


def _validate_time(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PasskeyV2ProtocolError(f"passkey_v2_{label}_invalid")
    return value


def build_action_envelope(
    *,
    request_id: str,
    requester_discord_user_id: str,
    required_approver_discord_user_id: str,
    scope: str,
    case_id: str,
    target_system: str,
    action_summary: str,
    risk: str,
    rollback: str,
    action_payload: Mapping[str, Any],
    executor_release_sha: str,
    executor_plan_sha256: str,
    transaction_id: str,
    stage: str,
    webauthn_rp_id: str,
    webauthn_origin: str,
    authority_release_sha: str,
    authority_manifest_sha256: str,
    authority_host_receipt_sha256: str,
    source_preflight_sha256: str,
    live_projection_sha256: str,
    external_iam_receipt_sha256: str,
    prior_authoritative_receipt_sha256: str,
    prior_event_head_sha256: str,
    issued_at_unix: int,
    approval_ttl_seconds: int,
    execution_window_seconds: int = EXECUTION_WINDOW_SECONDS,
) -> Mapping[str, Any]:
    payload = dict(action_payload)
    unsigned = {
        "schema": ACTION_ENVELOPE_SCHEMA,
        "canonicalization": CANONICALIZATION_VERSION,
        "request_id": request_id,
        "requester_discord_user_id": requester_discord_user_id,
        "required_approver_discord_user_id": required_approver_discord_user_id,
        "scope": scope,
        "case_id": case_id,
        "target_system": target_system,
        "action_summary": action_summary,
        "risk": risk,
        "rollback": rollback,
        "action_payload": payload,
        "action_payload_sha256": sha256_json(payload),
        "executor_release_sha": executor_release_sha,
        "executor_plan_sha256": executor_plan_sha256,
        "transaction_id": transaction_id,
        "stage": stage,
        "webauthn_rp_id": webauthn_rp_id,
        "webauthn_origin": webauthn_origin,
        "authority_release_sha": authority_release_sha,
        "authority_manifest_sha256": authority_manifest_sha256,
        "authority_host_receipt_sha256": authority_host_receipt_sha256,
        "source_preflight_sha256": source_preflight_sha256,
        "live_projection_sha256": live_projection_sha256,
        "external_iam_receipt_sha256": external_iam_receipt_sha256,
        "prior_authoritative_receipt_sha256": (
            prior_authoritative_receipt_sha256
        ),
        "prior_event_head_sha256": prior_event_head_sha256,
        "execution_window_seconds": execution_window_seconds,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": issued_at_unix + approval_ttl_seconds,
        "approval_ttl_seconds": approval_ttl_seconds,
    }
    value = {**unsigned, "envelope_sha256": sha256_json(unsigned)}
    return validate_action_envelope(value)


def validate_action_envelope(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ACTION_FIELDS:
        raise PasskeyV2ProtocolError("passkey_v2_action_fields_invalid")
    if (
        value.get("schema") != ACTION_ENVELOPE_SCHEMA
        or value.get("canonicalization") != CANONICALIZATION_VERSION
    ):
        raise PasskeyV2ProtocolError("passkey_v2_action_schema_invalid")
    validate_request_id(value.get("request_id"))
    _validate_discord_id(
        value.get("requester_discord_user_id"), label="requester_discord_id"
    )
    _validate_discord_id(
        value.get("required_approver_discord_user_id"),
        label="required_approver_discord_id",
    )
    if value.get("scope") not in DANGEROUS_SCOPES:
        raise PasskeyV2ProtocolError("passkey_v2_scope_invalid")
    if (
        not isinstance(value.get("case_id"), str)
        or _CASE_ID.fullmatch(value["case_id"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_case_id_invalid")
    if (
        not isinstance(value.get("target_system"), str)
        or _TARGET.fullmatch(value["target_system"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_target_invalid")
    _text(value.get("action_summary"), label="summary", minimum=8, maximum=2048)
    _text(value.get("risk"), label="risk", minimum=8, maximum=4096)
    _text(value.get("rollback"), label="rollback", minimum=8, maximum=4096)
    payload = value.get("action_payload")
    if not isinstance(payload, Mapping) or not payload:
        raise PasskeyV2ProtocolError("passkey_v2_action_payload_invalid")
    if len(canonical_json_bytes(payload)) > MAXIMUM_ACTION_PAYLOAD_BYTES:
        raise PasskeyV2ProtocolError("passkey_v2_action_payload_too_large")
    if value.get("action_payload_sha256") != sha256_json(payload):
        raise PasskeyV2ProtocolError("passkey_v2_action_payload_hash_invalid")
    if (
        not isinstance(value.get("executor_release_sha"), str)
        or _REVISION.fullmatch(value["executor_release_sha"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_release_invalid")
    _validate_sha(value.get("executor_plan_sha256"), label="plan_sha256")
    _validate_sha(value.get("transaction_id"), label="transaction_id")
    if (
        not isinstance(value.get("stage"), str)
        or _STAGE.fullmatch(value["stage"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_stage_invalid")
    if (
        not isinstance(value.get("webauthn_rp_id"), str)
        or _RP_ID.fullmatch(value["webauthn_rp_id"]) is None
        or not isinstance(value.get("webauthn_origin"), str)
        or _ORIGIN.fullmatch(value["webauthn_origin"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_webauthn_identity_invalid")
    if (
        not isinstance(value.get("authority_release_sha"), str)
        or _REVISION.fullmatch(value["authority_release_sha"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_authority_release_invalid")
    for name in (
        "authority_manifest_sha256",
        "authority_host_receipt_sha256",
        "source_preflight_sha256",
        "live_projection_sha256",
        "external_iam_receipt_sha256",
        "prior_authoritative_receipt_sha256",
        "prior_event_head_sha256",
    ):
        _validate_sha(value.get(name), label=name)
    if value.get("execution_window_seconds") != EXECUTION_WINDOW_SECONDS:
        raise PasskeyV2ProtocolError("passkey_v2_execution_window_invalid")
    issued = _validate_time(value.get("issued_at_unix"), label="issued_at")
    expires = _validate_time(value.get("expires_at_unix"), label="expires_at")
    ttl = value.get("approval_ttl_seconds")
    if (
        not isinstance(ttl, int)
        or isinstance(ttl, bool)
        or not MINIMUM_TTL_SECONDS <= ttl <= MAXIMUM_TTL_SECONDS
        or expires != issued + ttl
    ):
        raise PasskeyV2ProtocolError("passkey_v2_ttl_invalid")
    expected = sha256_json(_unsigned(value, "envelope_sha256"))
    if value.get("envelope_sha256") != expected:
        raise PasskeyV2ProtocolError("passkey_v2_action_hash_invalid")
    return dict(value)


def build_challenge_record(
    *,
    envelope: Mapping[str, Any],
    challenge_id: str,
    challenge_b64url: str,
    rp_id: str,
    origin: str,
    created_at_unix: int,
) -> Mapping[str, Any]:
    action = validate_action_envelope(envelope)
    challenge_bytes = _decode_canonical_b64url(
        challenge_b64url,
        label="challenge",
        minimum_bytes=32,
        maximum_bytes=768,
    )
    unsigned = {
        "schema": CHALLENGE_SCHEMA,
        "canonicalization": CANONICALIZATION_VERSION,
        "state": "challenge_created",
        "request_id": action["request_id"],
        "action_envelope_sha256": action["envelope_sha256"],
        "challenge_id": challenge_id,
        "challenge_b64url": challenge_b64url,
        "challenge_sha256": sha256_bytes(challenge_bytes),
        "rp_id": rp_id,
        "origin": origin,
        "created_at_unix": created_at_unix,
        "expires_at_unix": action["expires_at_unix"],
    }
    value = {**unsigned, "challenge_record_sha256": sha256_json(unsigned)}
    return validate_challenge_record(value, envelope=action)


def validate_challenge_record(
    value: Any,
    *,
    envelope: Mapping[str, Any],
) -> Mapping[str, Any]:
    action = validate_action_envelope(envelope)
    if not isinstance(value, Mapping) or set(value) != _CHALLENGE_FIELDS:
        raise PasskeyV2ProtocolError("passkey_v2_challenge_fields_invalid")
    challenge = dict(value)
    if (
        challenge.get("schema") != CHALLENGE_SCHEMA
        or challenge.get("canonicalization") != CANONICALIZATION_VERSION
        or challenge.get("state") != "challenge_created"
        or challenge.get("request_id") != action["request_id"]
        or challenge.get("action_envelope_sha256")
        != action["envelope_sha256"]
        or challenge.get("expires_at_unix") != action["expires_at_unix"]
        or challenge.get("rp_id") != action["webauthn_rp_id"]
        or challenge.get("origin") != action["webauthn_origin"]
    ):
        raise PasskeyV2ProtocolError("passkey_v2_challenge_binding_invalid")
    if (
        not isinstance(challenge.get("challenge_id"), str)
        or _CHALLENGE_ID.fullmatch(challenge["challenge_id"]) is None
        or not isinstance(challenge.get("challenge_b64url"), str)
        or _B64URL.fullmatch(challenge["challenge_b64url"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_challenge_invalid")
    raw = _decode_canonical_b64url(
        challenge["challenge_b64url"],
        label="challenge",
        minimum_bytes=32,
        maximum_bytes=768,
    )
    if (
        challenge.get("challenge_sha256") != sha256_bytes(raw)
        or not isinstance(challenge.get("rp_id"), str)
        or _RP_ID.fullmatch(challenge["rp_id"]) is None
        or not isinstance(challenge.get("origin"), str)
        or _ORIGIN.fullmatch(challenge["origin"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_challenge_identity_invalid")
    created = _validate_time(challenge.get("created_at_unix"), label="challenge_time")
    if not action["issued_at_unix"] <= created < action["expires_at_unix"]:
        raise PasskeyV2ProtocolError("passkey_v2_challenge_time_invalid")
    expected = sha256_json(_unsigned(challenge, "challenge_record_sha256"))
    if challenge.get("challenge_record_sha256") != expected:
        raise PasskeyV2ProtocolError("passkey_v2_challenge_hash_invalid")
    return challenge


def build_passkey_grant(
    *,
    envelope: Mapping[str, Any],
    challenge: Mapping[str, Any],
    grant_id: str,
    approver_discord_user_id: str,
    credential_id_sha256: str,
    credential_record_sha256: str,
    credential_migration_receipt_sha256: str,
    assertion_verification_sha256: str,
    credential_sign_count: int,
    credential_backed_up: bool,
    granted_at_unix: int,
) -> Mapping[str, Any]:
    action = validate_action_envelope(envelope)
    checked_challenge = validate_challenge_record(challenge, envelope=action)
    unsigned = {
        "schema": GRANT_SCHEMA,
        "canonicalization": CANONICALIZATION_VERSION,
        "state": "granted",
        "method": "passkey",
        "single_use": True,
        "request_id": action["request_id"],
        "action_envelope_sha256": action["envelope_sha256"],
        "challenge_id": checked_challenge["challenge_id"],
        "challenge_record_sha256": checked_challenge[
            "challenge_record_sha256"
        ],
        "grant_id": grant_id,
        "approver_discord_user_id": approver_discord_user_id,
        "credential_id_sha256": credential_id_sha256,
        "credential_record_sha256": credential_record_sha256,
        "credential_migration_receipt_sha256": (
            credential_migration_receipt_sha256
        ),
        "assertion_verification_sha256": assertion_verification_sha256,
        "credential_sign_count": credential_sign_count,
        "credential_backed_up": credential_backed_up,
        "user_verified": True,
        "rp_id": checked_challenge["rp_id"],
        "origin": checked_challenge["origin"],
        "granted_at_unix": granted_at_unix,
        "expires_at_unix": action["expires_at_unix"],
    }
    value = {**unsigned, "grant_sha256": sha256_json(unsigned)}
    return validate_passkey_grant(
        value,
        envelope=action,
        challenge=checked_challenge,
    )


def validate_passkey_grant(
    value: Any,
    *,
    envelope: Mapping[str, Any],
    challenge: Mapping[str, Any],
) -> Mapping[str, Any]:
    action = validate_action_envelope(envelope)
    checked_challenge = validate_challenge_record(challenge, envelope=action)
    if not isinstance(value, Mapping) or set(value) != _GRANT_FIELDS:
        raise PasskeyV2ProtocolError("passkey_v2_grant_fields_invalid")
    grant = dict(value)
    if (
        grant.get("schema") != GRANT_SCHEMA
        or grant.get("canonicalization") != CANONICALIZATION_VERSION
        or grant.get("state") != "granted"
        or grant.get("method") != "passkey"
        or grant.get("single_use") is not True
        or grant.get("user_verified") is not True
        or grant.get("request_id") != action["request_id"]
        or grant.get("action_envelope_sha256") != action["envelope_sha256"]
        or grant.get("challenge_id") != checked_challenge["challenge_id"]
        or grant.get("challenge_record_sha256")
        != checked_challenge["challenge_record_sha256"]
        or grant.get("approver_discord_user_id")
        != action["required_approver_discord_user_id"]
        or grant.get("rp_id") != checked_challenge["rp_id"]
        or grant.get("origin") != checked_challenge["origin"]
        or grant.get("expires_at_unix") != action["expires_at_unix"]
    ):
        raise PasskeyV2ProtocolError("passkey_v2_grant_binding_invalid")
    validate_grant_id(grant.get("grant_id"))
    _validate_discord_id(
        grant.get("approver_discord_user_id"), label="grant_approver_discord_id"
    )
    _validate_sha(grant.get("credential_id_sha256"), label="credential_id")
    _validate_sha(
        grant.get("credential_record_sha256"), label="credential_record_sha256"
    )
    _validate_sha(
        grant.get("credential_migration_receipt_sha256"),
        label="credential_migration_receipt_sha256",
    )
    _validate_sha(
        grant.get("assertion_verification_sha256"),
        label="assertion_verification_sha256",
    )
    count = grant.get("credential_sign_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise PasskeyV2ProtocolError("passkey_v2_credential_counter_invalid")
    if not isinstance(grant.get("credential_backed_up"), bool):
        raise PasskeyV2ProtocolError("passkey_v2_credential_backup_invalid")
    granted = _validate_time(grant.get("granted_at_unix"), label="granted_at")
    if not checked_challenge["created_at_unix"] <= granted < action[
        "expires_at_unix"
    ]:
        raise PasskeyV2ProtocolError("passkey_v2_grant_time_invalid")
    expected = sha256_json(_unsigned(grant, "grant_sha256"))
    if grant.get("grant_sha256") != expected:
        raise PasskeyV2ProtocolError("passkey_v2_grant_hash_invalid")
    return grant


def validate_runtime_binding(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_FIELDS:
        raise PasskeyV2ProtocolError("passkey_v2_runtime_binding_fields_invalid")
    binding = dict(value)
    if (
        not isinstance(binding.get("executor_release_sha"), str)
        or _REVISION.fullmatch(binding["executor_release_sha"]) is None
    ):
        raise PasskeyV2ProtocolError("passkey_v2_runtime_release_invalid")
    for name in (
        "executor_plan_sha256",
        "executor_binary_sha256",
        "mutation_wrapper_sha256",
        "remote_transport_sha256",
    ):
        _validate_sha(binding.get(name), label=name)
    if binding.get("runtime_binding_sha256") != sha256_json(
        _unsigned(binding, "runtime_binding_sha256")
    ):
        raise PasskeyV2ProtocolError("passkey_v2_runtime_binding_hash_invalid")
    return binding


def build_runtime_binding(
    *,
    executor_release_sha: str,
    executor_plan_sha256: str,
    executor_binary_sha256: str,
    mutation_wrapper_sha256: str,
    remote_transport_sha256: str,
) -> Mapping[str, Any]:
    unsigned = {
        "executor_release_sha": executor_release_sha,
        "executor_plan_sha256": executor_plan_sha256,
        "executor_binary_sha256": executor_binary_sha256,
        "mutation_wrapper_sha256": mutation_wrapper_sha256,
        "remote_transport_sha256": remote_transport_sha256,
    }
    value = {**unsigned, "runtime_binding_sha256": sha256_json(unsigned)}
    return validate_runtime_binding(value)


def build_authorization_receipt_unsigned(
    *,
    envelope: Mapping[str, Any],
    grant: Mapping[str, Any],
    challenge: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    consume_attempt_id: str,
    consumed_at_unix: int,
    prior_journal_head_sha256: str,
    receipt_public_key_id: str,
) -> Mapping[str, Any]:
    action = validate_action_envelope(envelope)
    checked_challenge = validate_challenge_record(challenge, envelope=action)
    checked_grant = validate_passkey_grant(
        grant,
        envelope=action,
        challenge=checked_challenge,
    )
    runtime = validate_runtime_binding(runtime_binding)
    _validate_sha(prior_journal_head_sha256, label="prior_journal_head")
    _validate_sha(receipt_public_key_id, label="receipt_public_key_id")
    _validate_sha(consume_attempt_id, label="consume_attempt_id")
    consumed = _validate_time(consumed_at_unix, label="consumed_at")
    if not checked_grant["granted_at_unix"] <= consumed < checked_grant[
        "expires_at_unix"
    ]:
        raise PasskeyV2ProtocolError("passkey_v2_consumption_time_invalid")
    if (
        runtime["executor_release_sha"] != action["executor_release_sha"]
        or runtime["executor_plan_sha256"] != action["executor_plan_sha256"]
    ):
        raise PasskeyV2ProtocolError("passkey_v2_runtime_action_binding_invalid")
    return {
        "schema": AUTHORIZATION_RECEIPT_SCHEMA,
        "canonicalization": CANONICALIZATION_VERSION,
        "outcome": "ALLOW",
        "mutation_authorized": True,
        "mutation_executed": False,
        "authorization_disposition": "authorized_once",
        "consume_attempt_id": consume_attempt_id,
        "request_id": action["request_id"],
        "action_envelope_sha256": action["envelope_sha256"],
        "action_payload_sha256": action["action_payload_sha256"],
        "scope": action["scope"],
        "case_id": action["case_id"],
        "target_system": action["target_system"],
        "transaction_id": action["transaction_id"],
        "stage": action["stage"],
        "grant_id": checked_grant["grant_id"],
        "grant_sha256": checked_grant["grant_sha256"],
        "approver_discord_user_id": checked_grant[
            "approver_discord_user_id"
        ],
        "credential_id_sha256": checked_grant["credential_id_sha256"],
        "credential_record_sha256": checked_grant[
            "credential_record_sha256"
        ],
        "credential_migration_receipt_sha256": checked_grant[
            "credential_migration_receipt_sha256"
        ],
        "assertion_verification_sha256": checked_grant[
            "assertion_verification_sha256"
        ],
        "credential_sign_count": checked_grant["credential_sign_count"],
        "credential_backed_up": checked_grant["credential_backed_up"],
        "approval_method": "passkey",
        "challenge_id": checked_grant["challenge_id"],
        "challenge_record_sha256": checked_grant[
            "challenge_record_sha256"
        ],
        "granted_at_unix": checked_grant["granted_at_unix"],
        "grant_expires_at_unix": checked_grant["expires_at_unix"],
        "consumed_at_unix": consumed,
        "execution_window_seconds": action["execution_window_seconds"],
        "execution_window_expires_at_unix": (
            consumed + action["execution_window_seconds"]
        ),
        "authority_release_sha": action["authority_release_sha"],
        "authority_manifest_sha256": action["authority_manifest_sha256"],
        "authority_host_receipt_sha256": action[
            "authority_host_receipt_sha256"
        ],
        "source_preflight_sha256": action["source_preflight_sha256"],
        "live_projection_sha256": action["live_projection_sha256"],
        "external_iam_receipt_sha256": action[
            "external_iam_receipt_sha256"
        ],
        "prior_authoritative_receipt_sha256": action[
            "prior_authoritative_receipt_sha256"
        ],
        "prior_event_head_sha256": action["prior_event_head_sha256"],
        "runtime_binding": runtime,
        "prior_journal_head_sha256": prior_journal_head_sha256,
        "receipt_public_key_id": receipt_public_key_id,
    }


def validate_authorization_receipt(
    value: Any,
    *,
    envelope: Mapping[str, Any],
    grant: Mapping[str, Any],
    challenge: Mapping[str, Any],
    receipt_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        raise PasskeyV2ProtocolError("passkey_v2_receipt_fields_invalid")
    receipt = dict(value)
    signature = receipt.pop("signature_ed25519_b64url")
    receipt_sha = receipt.pop("receipt_sha256")
    if not isinstance(signature, str) or _B64URL.fullmatch(signature) is None:
        raise PasskeyV2ProtocolError("passkey_v2_receipt_signature_invalid")
    runtime_binding = receipt.get("runtime_binding")
    consume_attempt_id = receipt.get("consume_attempt_id")
    consumed_at_unix = receipt.get("consumed_at_unix")
    prior_journal_head_sha256 = receipt.get("prior_journal_head_sha256")
    receipt_public_key_id = receipt.get("receipt_public_key_id")
    if (
        not isinstance(runtime_binding, Mapping)
        or not isinstance(consume_attempt_id, str)
        or type(consumed_at_unix) is not int
        or not isinstance(prior_journal_head_sha256, str)
        or not isinstance(receipt_public_key_id, str)
    ):
        raise PasskeyV2ProtocolError("passkey_v2_receipt_binding_invalid")
    unsigned = build_authorization_receipt_unsigned(
        envelope=envelope,
        grant=grant,
        challenge=challenge,
        runtime_binding=runtime_binding,
        consume_attempt_id=consume_attempt_id,
        consumed_at_unix=consumed_at_unix,
        prior_journal_head_sha256=prior_journal_head_sha256,
        receipt_public_key_id=receipt_public_key_id,
    )
    if receipt != unsigned:
        raise PasskeyV2ProtocolError("passkey_v2_receipt_binding_invalid")
    expected_key_id = sha256_bytes(receipt_public_key.public_bytes_raw())
    if unsigned["receipt_public_key_id"] != expected_key_id:
        raise PasskeyV2ProtocolError("passkey_v2_receipt_key_invalid")
    try:
        signature_bytes = _decode_canonical_b64url(
            signature,
            label="receipt_signature",
            minimum_bytes=64,
            maximum_bytes=64,
        )
        receipt_public_key.verify(signature_bytes, canonical_json_bytes(unsigned))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise PasskeyV2ProtocolError("passkey_v2_receipt_signature_invalid") from None
    signed = {**unsigned, "signature_ed25519_b64url": signature}
    if receipt_sha != sha256_json(signed):
        raise PasskeyV2ProtocolError("passkey_v2_receipt_hash_invalid")
    return {**signed, "receipt_sha256": receipt_sha}


def build_ui_view(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the exact, non-truncated values a passkey UI must render."""

    action = validate_action_envelope(envelope)
    payload_json = canonical_json_bytes(action["action_payload"]).decode("utf-8")
    envelope_json = canonical_json_bytes(action).decode("utf-8")
    return {
        "schema": UI_VIEW_SCHEMA,
        "request_id": action["request_id"],
        "requester_discord_user_id": action["requester_discord_user_id"],
        "required_approver_discord_user_id": action[
            "required_approver_discord_user_id"
        ],
        "scope": action["scope"],
        "case_id": action["case_id"],
        "target_system": action["target_system"],
        "action_summary": action["action_summary"],
        "risk": action["risk"],
        "rollback": action["rollback"],
        "exact_action_payload_canonical_json": payload_json,
        "exact_action_envelope_canonical_json": envelope_json,
        "action_payload_sha256": action["action_payload_sha256"],
        "full_action_envelope_sha256": action["envelope_sha256"],
        "executor_release_sha": action["executor_release_sha"],
        "executor_plan_sha256": action["executor_plan_sha256"],
        "transaction_id": action["transaction_id"],
        "stage": action["stage"],
        "webauthn_rp_id": action["webauthn_rp_id"],
        "webauthn_origin": action["webauthn_origin"],
        "expires_at_unix": action["expires_at_unix"],
        "execution_window_seconds": action["execution_window_seconds"],
        "approval_method": "passkey",
        "totp_available_for_dangerous_action": False,
        "values_are_complete_and_untruncated": True,
    }


def validate_ui_headers(headers: Mapping[str, str]) -> None:
    for name, expected in UI_SECURITY_HEADERS.items():
        if headers.get(name) != expected:
            raise PasskeyV2ProtocolError("passkey_v2_ui_headers_invalid")


def require_dangerous_approval_method(method: str) -> None:
    if method != "passkey":
        raise PasskeyV2ProtocolError("passkey_v2_dangerous_totp_disabled")


def require_production_webauthn_identity(envelope: Mapping[str, Any]) -> None:
    action = validate_action_envelope(envelope)
    if (
        action["webauthn_rp_id"] != PRODUCTION_RP_ID
        or action["webauthn_origin"] != PRODUCTION_ORIGIN
    ):
        raise PasskeyV2ProtocolError("passkey_v2_production_webauthn_invalid")
