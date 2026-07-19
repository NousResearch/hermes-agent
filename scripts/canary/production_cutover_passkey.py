#!/usr/bin/env python3
"""Closed passkey-v2 contract for one exact production cutover freeze.

The module is deliberately an edge contract.  It cannot select a command,
path, VM, plan type, or rollback policy.  One passkey grant claims one exact
owner-signed :class:`FreezePlan`; all later recovery is constrained to that
same plan's already-reviewed state machine.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_trust as release_trust
from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_storage_growth as storage
from scripts.canary import production_cutover_portable_contract as portable


CUTOVER_ACTION_SCHEMA = "muncho-passkey-v2-production-cutover-action.v1"
CUTOVER_FACTS_SCHEMA = "muncho-passkey-v2-production-cutover-facts.v1"
CUTOVER_PROOF_SCHEMA = "muncho-passkey-v2-production-cutover-proof.v1"
CUTOVER_TRUST_SCHEMA = "muncho-passkey-v2-production-cutover-trust.v1"
CUTOVER_CLAIM_FRAME_SCHEMA = (
    "muncho-production-cutover-passkey-claim-frame.v1"
)
CUTOVER_TRUST_BUNDLE_PATH = Path(
    "/etc/muncho-owner-gate/public/production-cutover-passkey-trust.json"
)

PRODUCTION_PROJECT = "adventico-ai-platform"
PRODUCTION_ZONE = "europe-west3-a"
PRODUCTION_VM_NAME = "ai-platform-runtime-01"
PRODUCTION_VM_INSTANCE_ID = "1094477181810932795"
OWNER_DISCORD_USER_ID = storage.OWNER_DISCORD_USER_ID
ACTION_SCOPE = "production_write"
ACTION_STAGE = "freeze"
ACTION_CASE_ID = "case:muncho-production-canonical-brain-cutover"
ACTION_TARGET_SYSTEM = (
    "gce:adventico-ai-platform/europe-west3-a/"
    "ai-platform-runtime-01/canonical-brain"
)
ACTION_SUMMARY = (
    "Claim and execute the exact owner-signed production Canonical Brain "
    "freeze and cutover plan on ai-platform-runtime-01."
)
ACTION_RISK = (
    "The exact legacy gateway, writer, and connector are stopped before the "
    "reviewed final tail is captured; the fixed cutover may then mutate the "
    "production database and host activation state."
)
ACTION_ROLLBACK = (
    "Before a cutover plan exists, restore the exact legacy gateway through "
    "abort-freeze. After cutover staging, use only the rollback transitions "
    "already encoded in the claimed CutoverPlan state machine."
)
ALLOWED_OPERATIONS = (
    "stage_exact_freeze_authority",
    "stop_exact_legacy_services",
    "capture_exact_final_tail",
    "abort_exact_freeze_before_cutover_plan",
    "stage_exact_cron_continuity",
    "stage_exact_cutover_plan",
    "run_exact_phase_b_preflight",
    "apply_or_recover_exact_cutover",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PUBLICATION_FIELDS = frozenset({
    "schema",
    "action",
    "release_revision",
    "documents",
    "secret_material_recorded",
    "secret_digest_recorded",
    "publication_sha256",
})
_ACTION_FIELDS = frozenset({
    "schema",
    "operation",
    "production_project",
    "production_zone",
    "production_vm_name",
    "production_vm_instance_id",
    "production_release_revision",
    "freeze_plan",
    "freeze_plan_sha256",
    "freeze_approval",
    "freeze_approval_sha256",
    "freeze_publication_sha256",
    "allowed_operations",
    "caller_selected_commands_allowed",
    "caller_selected_paths_allowed",
    "caller_selected_targets_allowed",
    "generic_shell_fallback_allowed",
})
_TRUST_FIELDS = frozenset({
    "schema",
    "authority_release_sha",
    "release_trust_manifest_b64url",
    "release_trust_public_key_b64url",
    "host_observation_public_key_b64url",
    "post_iam_host_observation",
    "authority_receipt_public_key_pem_b64url",
    "authority_receipt_public_key_sha256",
    "trust_bundle_sha256",
})
_PROOF_FIELDS = frozenset({
    "schema",
    "freeze_publication_sha256",
    "action_envelope",
    "challenge_record",
    "grant_record",
    "authorization_receipt",
    "trust_bundle",
    "proof_sha256",
})
_CLAIM_FIELDS = frozenset({
    "schema", "publication", "passkey_proof", "claim_sha256"
})


class ProductionCutoverPasskeyError(RuntimeError):
    """Stable, secret-free production passkey boundary error."""


class DedicatedOwnerGateTransport(Protocol):
    def invoke_owner_gate(self, canonical_frame: bytes) -> bytes: ...


def _canonical(value: Any) -> bytes:
    return protocol.canonical_json_bytes(value)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode_b64url(value: Any, *, maximum: int, label: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > maximum * 2:
        raise ProductionCutoverPasskeyError(
            f"production_cutover_{label}_invalid"
        )
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (UnicodeError, ValueError) as exc:
        raise ProductionCutoverPasskeyError(
            f"production_cutover_{label}_invalid"
        ) from None
    if not raw or len(raw) > maximum or _b64url(raw) != value:
        raise ProductionCutoverPasskeyError(
            f"production_cutover_{label}_invalid"
        )
    return raw


def _validate_publication(
    value: Any,
    *,
    now_unix: int,
) -> tuple[Mapping[str, Any], Any, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != _PUBLICATION_FIELDS:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_publication_invalid"
        )
    unsigned = {
        name: item for name, item in value.items()
        if name != "publication_sha256"
    }
    documents = value.get("documents")
    if (
        value.get("schema") != "muncho-production-cutover-publication.v1"
        or value.get("action") != "freeze-authority"
        or _REVISION.fullmatch(str(value.get("release_revision"))) is None
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("publication_sha256") != protocol.sha256_json(unsigned)
        or not isinstance(documents, Mapping)
        or set(documents) != {"plan", "approval"}
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_publication_invalid"
        )
    try:
        _portable_publication, plan, approval = (
            portable.validate_freeze_publication(value, now_unix=now_unix)
        )
    except (KeyError, TypeError, portable.PortableCutoverContractError):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_publication_invalid"
        ) from None
    target = plan.value["target"]
    if (
        plan.value["release_revision"] != value["release_revision"]
        or target.get("project") != PRODUCTION_PROJECT
        or target.get("zone") != PRODUCTION_ZONE
        or target.get("vm") != PRODUCTION_VM_NAME
        or plan.value.get("secret_material_recorded") is not False
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_target_invalid"
        )
    return copy.deepcopy(dict(value)), plan, copy.deepcopy(dict(approval))


def build_cutover_action_envelope(
    *,
    freeze_publication: Mapping[str, Any],
    authority_release_sha: str,
    authority_manifest_sha256: str,
    authority_host_receipt_sha256: str,
    issued_at_unix: int,
) -> Mapping[str, Any]:
    publication, plan, approval = _validate_publication(
        freeze_publication, now_unix=issued_at_unix
    )
    if (
        _REVISION.fullmatch(authority_release_sha or "") is None
        or _SHA256.fullmatch(authority_manifest_sha256 or "") is None
        or _SHA256.fullmatch(authority_host_receipt_sha256 or "") is None
        or type(issued_at_unix) is not int
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_authority_invalid"
        )
    remaining = approval["expires_at_unix"] - issued_at_unix
    if remaining < protocol.MINIMUM_TTL_SECONDS:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_approval_expired"
        )
    ttl = min(protocol.MAXIMUM_TTL_SECONDS, remaining)
    payload = {
        "schema": CUTOVER_ACTION_SCHEMA,
        "operation": "claim_exact_freeze_plan",
        "production_project": PRODUCTION_PROJECT,
        "production_zone": PRODUCTION_ZONE,
        "production_vm_name": PRODUCTION_VM_NAME,
        "production_vm_instance_id": PRODUCTION_VM_INSTANCE_ID,
        "production_release_revision": publication["release_revision"],
        "freeze_plan": plan.to_mapping(),
        "freeze_plan_sha256": plan.sha256,
        "freeze_approval": approval,
        "freeze_approval_sha256": approval["approval_sha256"],
        "freeze_publication_sha256": publication["publication_sha256"],
        "allowed_operations": list(ALLOWED_OPERATIONS),
        "caller_selected_commands_allowed": False,
        "caller_selected_paths_allowed": False,
        "caller_selected_targets_allowed": False,
        "generic_shell_fallback_allowed": False,
    }
    transaction_id = protocol.sha256_json({
        "schema": "muncho-production-cutover-passkey-transaction.v1",
        "production_release_revision": publication["release_revision"],
        "freeze_plan_sha256": plan.sha256,
        "freeze_approval_sha256": approval["approval_sha256"],
        "freeze_publication_sha256": publication["publication_sha256"],
    })
    request_id = protocol.sha256_json({
        "schema": "muncho-production-cutover-passkey-request.v1",
        "transaction_id": transaction_id,
        "authority_release_sha": authority_release_sha,
        "issued_at_unix": issued_at_unix,
    })
    return protocol.build_action_envelope(
        request_id=request_id,
        requester_discord_user_id=OWNER_DISCORD_USER_ID,
        required_approver_discord_user_id=OWNER_DISCORD_USER_ID,
        scope=ACTION_SCOPE,
        case_id=ACTION_CASE_ID,
        target_system=ACTION_TARGET_SYSTEM,
        action_summary=ACTION_SUMMARY,
        risk=ACTION_RISK,
        rollback=ACTION_ROLLBACK,
        action_payload=payload,
        executor_release_sha=authority_release_sha,
        executor_plan_sha256=plan.sha256,
        transaction_id=transaction_id,
        stage=ACTION_STAGE,
        webauthn_rp_id=protocol.PRODUCTION_RP_ID,
        webauthn_origin=protocol.PRODUCTION_ORIGIN,
        authority_release_sha=authority_release_sha,
        authority_manifest_sha256=authority_manifest_sha256,
        authority_host_receipt_sha256=authority_host_receipt_sha256,
        source_preflight_sha256=plan.sha256,
        live_projection_sha256=approval["approval_sha256"],
        external_iam_receipt_sha256=publication["publication_sha256"],
        prior_authoritative_receipt_sha256=(
            protocol.GENESIS_JOURNAL_HEAD_SHA256
        ),
        prior_event_head_sha256=protocol.GENESIS_JOURNAL_HEAD_SHA256,
        issued_at_unix=issued_at_unix,
        approval_ttl_seconds=ttl,
    )


def validate_cutover_action_envelope(
    envelope: Any,
    *,
    freeze_publication: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    try:
        action = protocol.validate_action_envelope(envelope)
        protocol.require_production_webauthn_identity(action)
    except protocol.PasskeyV2ProtocolError:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_action_invalid"
        ) from None
    payload = action["action_payload"]
    if not isinstance(payload, Mapping) or set(payload) != _ACTION_FIELDS:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_action_invalid"
        )
    try:
        plan = portable.validate_freeze_plan(payload["freeze_plan"])
        approval = portable.validate_freeze_approval(
            payload["freeze_approval"],
            plan=plan,
            now_unix=action["issued_at_unix"],
        )
    except (KeyError, TypeError, portable.PortableCutoverContractError):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_action_invalid"
        ) from None
    target = plan.value["target"]
    if (
        payload.get("schema") != CUTOVER_ACTION_SCHEMA
        or payload.get("operation") != "claim_exact_freeze_plan"
        or payload.get("production_project") != PRODUCTION_PROJECT
        or payload.get("production_zone") != PRODUCTION_ZONE
        or payload.get("production_vm_name") != PRODUCTION_VM_NAME
        or payload.get("production_vm_instance_id")
        != PRODUCTION_VM_INSTANCE_ID
        or payload.get("production_release_revision")
        != plan.value["release_revision"]
        or target.get("project") != PRODUCTION_PROJECT
        or target.get("zone") != PRODUCTION_ZONE
        or target.get("vm") != PRODUCTION_VM_NAME
        or payload.get("freeze_plan_sha256") != plan.sha256
        or payload.get("freeze_approval_sha256")
        != approval["approval_sha256"]
        or payload.get("allowed_operations") != list(ALLOWED_OPERATIONS)
        or any(
            payload.get(name) is not False
            for name in (
                "caller_selected_commands_allowed",
                "caller_selected_paths_allowed",
                "caller_selected_targets_allowed",
                "generic_shell_fallback_allowed",
            )
        )
        or action["scope"] != ACTION_SCOPE
        or action["case_id"] != ACTION_CASE_ID
        or action["target_system"] != ACTION_TARGET_SYSTEM
        or action["action_summary"] != ACTION_SUMMARY
        or action["risk"] != ACTION_RISK
        or action["rollback"] != ACTION_ROLLBACK
        or action["stage"] != ACTION_STAGE
        or action["executor_plan_sha256"] != plan.sha256
        or action["source_preflight_sha256"] != plan.sha256
        or action["live_projection_sha256"]
        != approval["approval_sha256"]
        or action["external_iam_receipt_sha256"]
        != payload.get("freeze_publication_sha256")
        or action["requester_discord_user_id"] != OWNER_DISCORD_USER_ID
        or action["required_approver_discord_user_id"]
        != OWNER_DISCORD_USER_ID
        or _SHA256.fullmatch(str(action.get("request_id", ""))) is None
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_action_invalid"
        )
    if freeze_publication is not None:
        publication, expected_plan, expected_approval = _validate_publication(
            freeze_publication, now_unix=action["issued_at_unix"]
        )
        if (
            publication["publication_sha256"]
            != payload["freeze_publication_sha256"]
            or expected_plan.sha256 != plan.sha256
            or expected_approval["approval_sha256"]
            != approval["approval_sha256"]
            or _canonical(publication["documents"]["plan"])
            != _canonical(payload["freeze_plan"])
            or _canonical(publication["documents"]["approval"])
            != _canonical(payload["freeze_approval"])
        ):
            raise ProductionCutoverPasskeyError(
                "production_cutover_passkey_publication_binding_invalid"
            )
    return action


def mechanical_approval_facts(envelope: Any) -> Mapping[str, Any]:
    action = validate_cutover_action_envelope(envelope)
    payload = action["action_payload"]
    return {
        "schema": CUTOVER_FACTS_SCHEMA,
        "production_project": PRODUCTION_PROJECT,
        "production_zone": PRODUCTION_ZONE,
        "production_vm_name": PRODUCTION_VM_NAME,
        "production_vm_instance_id": PRODUCTION_VM_INSTANCE_ID,
        "production_release_revision": payload[
            "production_release_revision"
        ],
        "freeze_plan_sha256": payload["freeze_plan_sha256"],
        "freeze_approval_sha256": payload["freeze_approval_sha256"],
        "freeze_publication_sha256": payload[
            "freeze_publication_sha256"
        ],
        "exact_allowed_operations": list(ALLOWED_OPERATIONS),
        "single_use": True,
        "user_verification_required": True,
        "totp_available": False,
        "caller_selected_commands_allowed": False,
        "caller_selected_paths_allowed": False,
        "caller_selected_targets_allowed": False,
    }


def build_trust_bundle(
    *,
    authority_release_sha: str,
    release_trust_manifest_raw: bytes,
    release_trust_public_key_raw: bytes,
    host_observation_public_key_raw: bytes,
    post_iam_host_observation: Mapping[str, Any],
    authority_receipt_public_key_pem: bytes,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": CUTOVER_TRUST_SCHEMA,
        "authority_release_sha": authority_release_sha,
        "release_trust_manifest_b64url": _b64url(
            release_trust_manifest_raw
        ),
        "release_trust_public_key_b64url": _b64url(
            release_trust_public_key_raw
        ),
        "host_observation_public_key_b64url": _b64url(
            host_observation_public_key_raw
        ),
        "post_iam_host_observation": copy.deepcopy(
            dict(post_iam_host_observation)
        ),
        "authority_receipt_public_key_pem_b64url": _b64url(
            authority_receipt_public_key_pem
        ),
        "authority_receipt_public_key_sha256": _sha(
            authority_receipt_public_key_pem
        ),
    }
    value = {
        **unsigned,
        "trust_bundle_sha256": protocol.sha256_json(unsigned),
    }
    validate_trust_bundle(value)
    return value


def validate_trust_bundle(
    value: Any,
) -> tuple[Mapping[str, Any], Ed25519PublicKey]:
    if not isinstance(value, Mapping) or set(value) != _TRUST_FIELDS:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_trust_invalid"
        )
    bundle = copy.deepcopy(dict(value))
    unsigned = {
        name: item for name, item in bundle.items()
        if name != "trust_bundle_sha256"
    }
    if (
        bundle.get("schema") != CUTOVER_TRUST_SCHEMA
        or _REVISION.fullmatch(str(bundle.get("authority_release_sha")))
        is None
        or bundle.get("trust_bundle_sha256")
        != protocol.sha256_json(unsigned)
        or not isinstance(bundle.get("post_iam_host_observation"), Mapping)
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_trust_invalid"
        )
    manifest_raw = _decode_b64url(
        bundle["release_trust_manifest_b64url"],
        maximum=4 * 1024 * 1024,
        label="release_trust_manifest",
    )
    trust_public_raw = _decode_b64url(
        bundle["release_trust_public_key_b64url"],
        maximum=32,
        label="release_trust_public_key",
    )
    host_public_raw = _decode_b64url(
        bundle["host_observation_public_key_b64url"],
        maximum=32,
        label="host_observation_public_key",
    )
    receipt_pem = _decode_b64url(
        bundle["authority_receipt_public_key_pem_b64url"],
        maximum=16 * 1024,
        label="authority_receipt_public_key",
    )
    try:
        trust = release_trust.decode_pinned_release_trust(
            manifest_raw=manifest_raw,
            public_key_raw=trust_public_raw,
        )
        host_public_key = Ed25519PublicKey.from_public_bytes(
            host_public_raw
        )
        receipt_public_key = serialization.load_pem_public_key(receipt_pem)
    except (
        release_trust.OwnerGateTrustError,
        TypeError,
        ValueError,
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_trust_invalid"
        ) from None
    if not isinstance(receipt_public_key, Ed25519PublicKey):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_trust_invalid"
        )
    host_key_id = protocol.sha256_bytes(host_public_raw)
    collector_ids = trust.get("collector_public_key_ids")
    observation = bundle["post_iam_host_observation"]
    try:
        preflight._verify_seal(observation, label="host_observation")
        preflight._verify_attestation(
            observation,
            public_key=host_public_key,
            expected_public_key_id=host_key_id,
            label="host_observation",
        )
    except preflight.OwnerGatePreflightError:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_host_observation_invalid"
        ) from None
    expected_webauthn = {
        "rp_id": protocol.PRODUCTION_RP_ID,
        "origin": protocol.PRODUCTION_ORIGIN,
        "user_verification_required": True,
        "forged_assertion_blocked": True,
        "wrong_challenge_blocked": True,
        "wrong_origin_blocked": True,
        "wrong_rp_blocked": True,
        "no_uv_blocked": True,
        "replay_blocked": True,
        "concurrent_exactly_one_authorized": True,
        "web_raw_grant_api_absent": True,
    }
    executor = observation.get("executor")
    release = observation.get("release")
    migration = observation.get("migration")
    receipt_sha = _sha(receipt_pem)
    if (
        not isinstance(collector_ids, Mapping)
        or collector_ids.get("host") != host_key_id
        or trust.get("release_revision")
        != bundle["authority_release_sha"]
        or observation.get("schema") != preflight.HOST_OBSERVATION_SCHEMA
        or observation.get("phase") != "post_iam"
        or observation.get("secret_material_recorded") is not False
        or not isinstance(release, Mapping)
        or release.get("revision") != bundle["authority_release_sha"]
        or release.get("immutable") is not True
        or _SHA256.fullmatch(str(release.get("package_sha256"))) is None
        or observation.get("webauthn") != expected_webauthn
        or not isinstance(migration, Mapping)
        or migration.get("owner_discord_user_id")
        != OWNER_DISCORD_USER_ID
        or migration.get("credential_count") != 1
        or migration.get("enabled_owner_count") != 1
        or migration.get("public_key_only") is not True
        or migration.get("totp_seed_migrated") is not False
        or not isinstance(executor, Mapping)
        or executor.get("uid") != storage.OWNER_GATE_EXECUTOR_UID
        or executor.get("mutation_iam_binding_present") is not True
        or executor.get("authorization_receipt_signature_self_verified")
        is not True
        or executor.get("receipt_action_binding_self_verified") is not True
        or executor.get("local_gcloud_present") is not False
        or executor.get("generic_shell_fallback_present") is not False
        or executor.get("receipt_public_key_sha256") != receipt_sha
        or executor.get("receipt_public_key_owner") != "root:root"
        or executor.get("receipt_public_key_mode") != "0444"
        or bundle.get("authority_receipt_public_key_sha256") != receipt_sha
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_trust_invalid"
        )
    return bundle, receipt_public_key


def build_passkey_proof(
    *,
    freeze_publication: Mapping[str, Any],
    action_envelope: Mapping[str, Any],
    challenge_record: Mapping[str, Any],
    grant_record: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
    trust_bundle: Mapping[str, Any],
) -> Mapping[str, Any]:
    publication, _plan, _approval = _validate_publication(
        freeze_publication,
        now_unix=int(action_envelope.get("issued_at_unix", 0)),
    )
    unsigned = {
        "schema": CUTOVER_PROOF_SCHEMA,
        "freeze_publication_sha256": publication["publication_sha256"],
        "action_envelope": copy.deepcopy(dict(action_envelope)),
        "challenge_record": copy.deepcopy(dict(challenge_record)),
        "grant_record": copy.deepcopy(dict(grant_record)),
        "authorization_receipt": copy.deepcopy(
            dict(authorization_receipt)
        ),
        "trust_bundle": copy.deepcopy(dict(trust_bundle)),
    }
    value = {**unsigned, "proof_sha256": protocol.sha256_json(unsigned)}
    validate_passkey_proof(
        value,
        freeze_publication=publication,
        now_unix=authorization_receipt.get("consumed_at_unix"),
    )
    return value


def validate_passkey_proof(
    value: Any,
    *,
    freeze_publication: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _PROOF_FIELDS
        or type(now_unix) is not int
        or now_unix <= 0
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_proof_invalid"
        )
    proof = copy.deepcopy(dict(value))
    unsigned = {
        name: item for name, item in proof.items() if name != "proof_sha256"
    }
    if (
        proof.get("schema") != CUTOVER_PROOF_SCHEMA
        or proof.get("proof_sha256") != protocol.sha256_json(unsigned)
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_proof_invalid"
        )
    publication, _plan, approval = _validate_publication(
        freeze_publication,
        now_unix=int(proof.get("action_envelope", {}).get(
            "issued_at_unix", 0
        )),
    )
    action = validate_cutover_action_envelope(
        proof["action_envelope"], freeze_publication=publication
    )
    bundle, receipt_public_key = validate_trust_bundle(
        proof["trust_bundle"]
    )
    try:
        challenge = protocol.validate_challenge_record(
            proof["challenge_record"], envelope=action
        )
        grant = protocol.validate_passkey_grant(
            proof["grant_record"],
            envelope=action,
            challenge=challenge,
        )
        receipt = protocol.validate_authorization_receipt(
            proof["authorization_receipt"],
            envelope=action,
            grant=grant,
            challenge=challenge,
            receipt_public_key=receipt_public_key,
        )
    except protocol.PasskeyV2ProtocolError:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_proof_invalid"
        ) from None
    if (
        proof.get("freeze_publication_sha256")
        != publication["publication_sha256"]
        or action["authority_release_sha"]
        != bundle["authority_release_sha"]
        or action["authority_host_receipt_sha256"]
        != proof["trust_bundle"]["post_iam_host_observation"][
            "report_sha256"
        ]
        or action["authority_manifest_sha256"]
        != bundle["post_iam_host_observation"]["release"][
            "package_sha256"
        ]
        or grant["method"] != "passkey"
        or grant["single_use"] is not True
        or grant["user_verified"] is not True
        or receipt["outcome"] != "ALLOW"
        or receipt["mutation_authorized"] is not True
        or receipt["mutation_executed"] is not False
        or receipt["authorization_disposition"] != "authorized_once"
        or receipt["approval_method"] != "passkey"
        or not receipt["consumed_at_unix"] <= now_unix
        < receipt["execution_window_expires_at_unix"]
        or now_unix >= approval["expires_at_unix"]
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_proof_invalid"
        )
    return proof


def build_claim_frame(
    *,
    publication: Mapping[str, Any],
    passkey_proof: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    validate_passkey_proof(
        passkey_proof,
        freeze_publication=publication,
        now_unix=now_unix,
    )
    unsigned = {
        "schema": CUTOVER_CLAIM_FRAME_SCHEMA,
        "publication": copy.deepcopy(dict(publication)),
        "passkey_proof": copy.deepcopy(dict(passkey_proof)),
    }
    return {**unsigned, "claim_sha256": protocol.sha256_json(unsigned)}


def validate_claim_frame(
    value: Any,
    *,
    now_unix: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    publication, raw_proof = _claim_frame_parts(value)
    proof = validate_passkey_proof(
        raw_proof,
        freeze_publication=publication,
        now_unix=now_unix,
    )
    return publication, proof


def validate_claim_frame_for_recorded_replay(
    value: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate exact old bytes at their signed single-use consume time.

    This never authorizes a first write.  The root stager may use it only
    after matching every compact proof/publication digest against the one
    already-durable claim and proving that claim was recorded in-window.
    """

    publication, raw_proof = _claim_frame_parts(value)
    try:
        consumed_at = raw_proof["authorization_receipt"][
            "consumed_at_unix"
        ]
    except (KeyError, TypeError):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_claim_invalid"
        ) from None
    proof = validate_passkey_proof(
        raw_proof,
        freeze_publication=publication,
        now_unix=consumed_at,
    )
    return publication, proof


def _claim_frame_parts(
    value: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != _CLAIM_FIELDS:
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_claim_invalid"
        )
    unsigned = {
        name: item for name, item in value.items() if name != "claim_sha256"
    }
    if (
        value.get("schema") != CUTOVER_CLAIM_FRAME_SCHEMA
        or value.get("claim_sha256") != protocol.sha256_json(unsigned)
        or not isinstance(value.get("publication"), Mapping)
        or not isinstance(value.get("passkey_proof"), Mapping)
    ):
        raise ProductionCutoverPasskeyError(
            "production_cutover_passkey_claim_invalid"
        )
    return (
        copy.deepcopy(dict(value["publication"])),
        copy.deepcopy(dict(value["passkey_proof"])),
    )


class ProductionCutoverPasskeyBoundary:
    """Narrow owner-side exchange with the fixed owner-gate intake."""

    def __init__(
        self, authority_release_sha: str, transport: DedicatedOwnerGateTransport
    ) -> None:
        if (
            _REVISION.fullmatch(authority_release_sha or "") is None
            or not callable(getattr(transport, "invoke_owner_gate", None))
            or callable(getattr(transport, "run_local_compute_mutation", None))
        ):
            raise ProductionCutoverPasskeyError(
                "production_cutover_owner_gate_transport_invalid"
            )
        self.authority_release_sha = authority_release_sha
        self._transport = transport

    def _invoke(
        self, operation: str, document: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if operation not in {
            "request_production_cutover",
            "consume_production_cutover",
        }:
            raise ProductionCutoverPasskeyError(
                "production_cutover_owner_gate_operation_invalid"
            )
        unsigned = {
            "schema": storage.REMOTE_FRAME_SCHEMA,
            "operation": operation,
            "release_sha": self.authority_release_sha,
            "document": dict(document),
        }
        frame = {**unsigned, "frame_sha256": protocol.sha256_json(unsigned)}
        raw = self._transport.invoke_owner_gate(_canonical(frame))
        try:
            response = protocol.decode_canonical_json(raw)
        except protocol.PasskeyV2ProtocolError:
            raise ProductionCutoverPasskeyError(
                "production_cutover_owner_gate_response_invalid"
            ) from None
        response_unsigned = {
            name: item for name, item in response.items()
            if name != "response_sha256"
        } if isinstance(response, Mapping) else {}
        if (
            not isinstance(response, Mapping)
            or set(response) != {
                "schema", "operation", "release_sha", "ok", "document",
                "response_sha256",
            }
            or response.get("schema") != storage.REMOTE_RESPONSE_SCHEMA
            or response.get("operation") != operation
            or response.get("release_sha") != self.authority_release_sha
            or response.get("ok") is not True
            or not isinstance(response.get("document"), Mapping)
            or response.get("response_sha256")
            != protocol.sha256_json(response_unsigned)
        ):
            raise ProductionCutoverPasskeyError(
                "production_cutover_owner_gate_response_invalid"
            )
        return dict(response["document"])

    def request(
        self, freeze_publication: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self._invoke(
            "request_production_cutover",
            {"freeze_publication": dict(freeze_publication)},
        )

    def consume(
        self,
        *,
        freeze_publication: Mapping[str, Any],
        request_id: str,
        consume_attempt_id: str,
    ) -> Mapping[str, Any]:
        if (
            _SHA256.fullmatch(request_id or "") is None
            or _SHA256.fullmatch(consume_attempt_id or "") is None
        ):
            raise ProductionCutoverPasskeyError(
                "production_cutover_consume_attempt_invalid"
            )
        result = self._invoke(
            "consume_production_cutover",
            {
                "freeze_publication": dict(freeze_publication),
                "request_id": request_id,
                "consume_attempt_id": consume_attempt_id,
            },
        )
        if not isinstance(result.get("passkey_proof"), Mapping):
            raise ProductionCutoverPasskeyError(
                "production_cutover_owner_gate_response_invalid"
            )
        return result


__all__ = [
    "ACTION_CASE_ID",
    "ACTION_SCOPE",
    "ACTION_STAGE",
    "ACTION_TARGET_SYSTEM",
    "ALLOWED_OPERATIONS",
    "CUTOVER_ACTION_SCHEMA",
    "CUTOVER_CLAIM_FRAME_SCHEMA",
    "CUTOVER_FACTS_SCHEMA",
    "CUTOVER_PROOF_SCHEMA",
    "CUTOVER_TRUST_SCHEMA",
    "CUTOVER_TRUST_BUNDLE_PATH",
    "PRODUCTION_PROJECT",
    "PRODUCTION_ZONE",
    "PRODUCTION_VM_NAME",
    "PRODUCTION_VM_INSTANCE_ID",
    "ProductionCutoverPasskeyBoundary",
    "ProductionCutoverPasskeyError",
    "build_claim_frame",
    "build_cutover_action_envelope",
    "build_passkey_proof",
    "build_trust_bundle",
    "mechanical_approval_facts",
    "validate_claim_frame",
    "validate_cutover_action_envelope",
    "validate_passkey_proof",
    "validate_trust_bundle",
]
