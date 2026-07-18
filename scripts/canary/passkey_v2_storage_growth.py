#!/usr/bin/env python3
"""Exact storage-growth envelope and dedicated owner-gate remote boundary.

This module contains no local Compute mutation path.  It validates the one
40 -> 80 GB canary action and exchanges canonical frames with an explicitly
injected dedicated-owner-gate transport.  A path to ``gcloud`` or a generic
command runner is rejected.
"""

from __future__ import annotations

import base64
import re
import subprocess
from typing import Any, Mapping, Protocol

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import storage_growth_contract as contract
from scripts.canary import storage_growth_evidence as evidence


STORAGE_ACTION_SCHEMA = "muncho-passkey-v2-storage-growth-action.v1"
STORAGE_FACTS_SCHEMA = "muncho-passkey-v2-storage-growth-facts.v1"
REMOTE_FRAME_SCHEMA = "muncho-passkey-v2-owner-gate-frame.v1"
REMOTE_RESPONSE_SCHEMA = "muncho-passkey-v2-owner-gate-response.v1"
OWNER_GATE_VM_NAME = "muncho-owner-gate-01"
OWNER_GATE_WEB_UID = 29101
OWNER_GATE_AUTHORITY_UID = 29102
OWNER_GATE_EXECUTOR_UID = 29103
AUTHORITY_SOCKET = "/run/muncho-owner-gate/passkey-authority.sock"
EXECUTOR_SOCKET = "/run/muncho-owner-gate/privileged-executor.sock"
AUTHORITY_DB = "/var/lib/muncho-owner-gate/authority/passkey-v2.sqlite3"
EXECUTOR_DB = "/var/lib/muncho-owner-gate/executor/execution-v2.sqlite3"

# These are the only Cloud resources the v2 package knows.  Keep this module
# independent of the legacy storage approval/journal implementation: importing
# that graph into the privileged executor would reintroduce a parallel
# authority path.
PROJECT = contract.PROJECT
ZONE = contract.ZONE
VM_NAME = contract.VM_NAME
VM_INSTANCE_ID = contract.VM_INSTANCE_ID
DISK_NAME = contract.DISK_NAME
DISK_ID = contract.DISK_ID
BOOT_DEVICE_NAME = contract.BOOT_DEVICE_NAME
OWNER_ACCOUNT = contract.OWNER_ACCOUNT
OWNER_DISCORD_USER_ID = "1279454038731264061"
SOURCE_SIZE_GB = contract.SOURCE_SIZE_GB
TARGET_SIZE_GB = contract.TARGET_SIZE_GB
ACTION_SCOPE = "runtime_config_mutation"
ACTION_CASE_ID = "case:canary-storage-growth-p0"
ACTION_TARGET_SYSTEM = (
    "gce:adventico-ai-platform/europe-west3-a/"
    "muncho-canary-v2-01/boot-disk"
)
ACTION_SUMMARY = (
    "Increase the exact canary boot disk muncho-canary-v2-01 from 40 GB "
    "to 80 GB, with one conditional stop/start only if fresh evidence "
    "shows the filesystem still requires a reboot."
)
ACTION_RISK = (
    "This permanently increases the exact disk capacity and may stop then "
    "start the exact isolated canary instance; interruption is bounded to "
    "that stopped canary and disk shrink is not supported."
)
ACTION_ROLLBACK = (
    "Before any mutation, drift blocks execution. After resize, recovery "
    "reconciles the exact 80 GB disk and conditional instance lifecycle; "
    "capacity is not rolled back by disk shrink."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTION_FIELDS = frozenset({
    "schema",
    "operation",
    "project",
    "zone",
    "vm_name",
    "vm_instance_id",
    "disk_name",
    "disk_id",
    "boot_device_name",
    "source_size_gb",
    "target_size_gb",
    "conditional_reboot_approved",
    "allowed_operations",
    "storage_plan",
    "storage_plan_sha256",
    "source_preflight",
    "source_preflight_sha256",
    "live_projection",
    "live_projection_sha256",
    "external_iam_receipt",
    "external_iam_receipt_sha256",
    "current_host_receipt_file_sha256",
    "current_host_receipt_sha256",
    "current_stopped_release_receipt_file_sha256",
    "current_stopped_release_receipt_sha256",
})
_OPERATIONS_BY_STAGE = {
    "intent": [
        "resize_exact_disk_40_to_80",
        "conditional_stop_exact_instance",
        "conditional_start_exact_instance",
        "read_only_postflight",
    ],
    "resize": [
        "resize_exact_disk_40_to_80",
        "conditional_stop_exact_instance",
        "conditional_start_exact_instance",
        "read_only_postflight",
    ],
    "stop": [
        "conditional_stop_exact_instance",
        "conditional_start_exact_instance",
        "read_only_postflight",
    ],
    "start": ["conditional_start_exact_instance", "read_only_postflight"],
}


class PasskeyV2StorageBoundaryError(RuntimeError):
    """Stable storage owner-gate boundary failure."""


class DedicatedOwnerGateTransport(Protocol):
    """Narrow IAP transport; it cannot expose a local mutation callback."""

    def invoke_owner_gate(self, canonical_frame: bytes) -> bytes: ...


def exact_storage_plan() -> Mapping[str, Any]:
    """Return the single reviewed plan from the pure shared contract."""

    return contract.canonical_plan_report()


def validate_storage_plan(value: Any) -> Mapping[str, Any]:
    expected = exact_storage_plan()
    if (
        not isinstance(value, Mapping)
        or protocol.canonical_json_bytes(value)
        != protocol.canonical_json_bytes(expected)
    ):
        raise PasskeyV2StorageBoundaryError("passkey_v2_storage_plan_invalid")
    return expected


def canonical_bytes(value: Any) -> bytes:
    return protocol.canonical_json_bytes(value)


def validate_storage_growth_envelope(
    envelope: Mapping[str, Any],
) -> Mapping[str, Any]:
    action = protocol.validate_action_envelope(envelope)
    protocol.require_production_webauthn_identity(action)
    payload = action["action_payload"]
    if not isinstance(payload, Mapping) or set(payload) != _ACTION_FIELDS:
        raise PasskeyV2StorageBoundaryError("passkey_v2_storage_action_fields_invalid")
    plan = exact_storage_plan()
    stage = action["stage"]
    if (
        stage not in _OPERATIONS_BY_STAGE
        or payload.get("schema") != STORAGE_ACTION_SCHEMA
        or payload.get("operation") != "canary_boot_disk_growth_40_to_80"
        or payload.get("project") != PROJECT
        or payload.get("zone") != ZONE
        or payload.get("vm_name") != VM_NAME
        or payload.get("vm_instance_id") != VM_INSTANCE_ID
        or payload.get("disk_name") != DISK_NAME
        or payload.get("disk_id") != DISK_ID
        or payload.get("boot_device_name") != BOOT_DEVICE_NAME
        or payload.get("source_size_gb") != SOURCE_SIZE_GB
        or payload.get("target_size_gb") != TARGET_SIZE_GB
        or payload.get("conditional_reboot_approved") is not True
        or payload.get("allowed_operations") != _OPERATIONS_BY_STAGE[stage]
        or protocol.canonical_json_bytes(payload.get("storage_plan"))
        != protocol.canonical_json_bytes(plan)
        or payload.get("storage_plan_sha256") != plan["plan_sha256"]
        or action["executor_plan_sha256"] != plan["plan_sha256"]
        or action["scope"] != ACTION_SCOPE
        or action["case_id"] != ACTION_CASE_ID
        or action["target_system"] != ACTION_TARGET_SYSTEM
        or action["action_summary"] != ACTION_SUMMARY
        or action["risk"] != ACTION_RISK
        or action["rollback"] != ACTION_ROLLBACK
        or action["requester_discord_user_id"] != OWNER_DISCORD_USER_ID
        or action["required_approver_discord_user_id"] != OWNER_DISCORD_USER_ID
    ):
        raise PasskeyV2StorageBoundaryError("passkey_v2_storage_action_identity_invalid")
    for name, envelope_name in (
        ("source_preflight", "source_preflight_sha256"),
        ("live_projection", "live_projection_sha256"),
        ("external_iam_receipt", "external_iam_receipt_sha256"),
    ):
        item = payload.get(name)
        digest = payload.get(f"{name}_sha256")
        if (
            not isinstance(item, Mapping)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or protocol.sha256_json(item) != digest
            or digest != action[envelope_name]
        ):
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_storage_evidence_invalid"
            )
    try:
        source_bundle = evidence.validate_attested_observation_structure(
            payload["source_preflight"]
        )
        source = evidence.validate_observation(
            source_bundle["observation"],
            now_unix=action["issued_at_unix"],
            allowed_states=frozenset({
                "source_ready",
                "resize_complete_boot_required",
                "terminated_after_growth_intent",
            }),
        )
    except evidence.StorageGrowthEvidenceError as exc:
        raise PasskeyV2StorageBoundaryError(
            "passkey_v2_storage_source_evidence_invalid"
        ) from None
    allowed_checkpoints = {
        "intent": frozenset({"source"}),
        "resize": frozenset({"source", "post_resize"}),
        "stop": frozenset({"post_resize", "post_stop"}),
        "start": frozenset({"post_stop", "post_start"}),
    }[action["stage"]]
    expected_state = {
        "intent": "source_ready",
        "resize": "source_ready",
        "stop": "resize_complete_boot_required",
        "start": "terminated_after_growth_intent",
    }[action["stage"]]
    expected_source_request_binding = (
        evidence.observation_request_binding_sha256(
            transaction_id=action["transaction_id"],
            checkpoint=source_bundle["checkpoint"],
            prior_event_head_sha256=action["prior_event_head_sha256"],
            release_sha=action["executor_release_sha"],
            plan_sha256=action["executor_plan_sha256"],
        )
    )
    if (
        source_bundle["transaction_id"] != action["transaction_id"]
        or source["state"] != expected_state
        or source_bundle["checkpoint"] not in allowed_checkpoints
        or source_bundle["prior_event_head_sha256"]
        != action["prior_event_head_sha256"]
        or source_bundle["request_binding_sha256"]
        != expected_source_request_binding
        or source_bundle["observation_nonce_sha256"]
        != evidence.observation_nonce_sha256(
            request_binding_sha256=expected_source_request_binding,
            transaction_id=action["transaction_id"],
            checkpoint=source_bundle["checkpoint"],
        )
        or payload["live_projection"]
        != evidence.observation_projection(source)
        or payload["external_iam_receipt"] != source["external_iam_receipt"]
        or payload["external_iam_receipt_sha256"]
        != protocol.sha256_json(source["external_iam_receipt"])
    ):
        raise PasskeyV2StorageBoundaryError(
            "passkey_v2_storage_evidence_binding_invalid"
        )
    for name in (
        "current_host_receipt_file_sha256",
        "current_host_receipt_sha256",
        "current_stopped_release_receipt_file_sha256",
        "current_stopped_release_receipt_sha256",
    ):
        value = payload.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_storage_receipt_binding_invalid"
            )
    return action


def mechanical_approval_facts(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    """Derive the human-critical facts from the validated payload itself."""

    action = validate_storage_growth_envelope(envelope)
    payload = action["action_payload"]
    return {
        "schema": STORAGE_FACTS_SCHEMA,
        "project": payload["project"],
        "zone": payload["zone"],
        "instance_name": payload["vm_name"],
        "instance_numeric_id": payload["vm_instance_id"],
        "disk_name": payload["disk_name"],
        "disk_numeric_id": payload["disk_id"],
        "boot_device_name": payload["boot_device_name"],
        "source_size_gb": payload["source_size_gb"],
        "target_size_gb": payload["target_size_gb"],
        "conditional_stop_start_approved": payload[
            "conditional_reboot_approved"
        ],
        "exact_allowed_operations": list(payload["allowed_operations"]),
        "disk_shrink_rollback_available": False,
        "totp_available": False,
    }


def transaction_id_for_observation(observation: Mapping[str, Any]) -> str:
    checked = evidence.validate_initial_observation(
        observation,
        now_unix=int(observation.get("collected_at_unix", 0)),
    )
    return protocol.sha256_json({
        "schema": "muncho-storage-growth-transaction-id.v1",
        "plan_sha256": exact_storage_plan()["plan_sha256"],
        "source_observation_sha256": protocol.sha256_json(checked),
        "project": PROJECT,
        "zone": ZONE,
        "instance_id": VM_INSTANCE_ID,
        "disk_id": DISK_ID,
    })


def transaction_id_for_source(source_preflight: Mapping[str, Any]) -> str:
    bundle = evidence.validate_attested_observation_structure(source_preflight)
    return transaction_id_for_observation(bundle["observation"])


def build_storage_growth_envelope(
    *,
    source_preflight: Mapping[str, Any],
    transaction_id: str,
    stage: str,
    release_sha: str,
    authority_manifest_sha256: str,
    authority_host_receipt_sha256: str,
    prior_authoritative_receipt_sha256: str,
    prior_event_head_sha256: str,
    issued_at_unix: int,
) -> Mapping[str, Any]:
    """Build the one exact action remotely from trusted mechanical facts."""

    bundle = evidence.validate_attested_observation_structure(source_preflight)
    observation = evidence.validate_observation(
        bundle["observation"],
        now_unix=issued_at_unix,
        allowed_states=frozenset({
            "source_ready",
            "resize_complete_boot_required",
            "terminated_after_growth_intent",
            "target_ready",
        }),
    )
    if (
        stage not in _OPERATIONS_BY_STAGE
        or not isinstance(transaction_id, str)
        or _SHA256.fullmatch(transaction_id) is None
        or bundle["transaction_id"] != transaction_id
        or (
            stage == "intent"
            and transaction_id_for_source(bundle) != transaction_id
        )
    ):
        raise PasskeyV2StorageBoundaryError(
            "passkey_v2_storage_action_binding_invalid"
        )
    allowed_checkpoints = {
        "intent": frozenset({"source"}),
        "resize": frozenset({"source", "post_resize"}),
        "stop": frozenset({"post_resize", "post_stop"}),
        "start": frozenset({"post_stop", "post_start"}),
    }[stage]
    expected_state = {
        "intent": "source_ready",
        "resize": "source_ready",
        "stop": "resize_complete_boot_required",
        "start": "terminated_after_growth_intent",
    }[stage]
    expected_binding = evidence.observation_request_binding_sha256(
        transaction_id=transaction_id,
        checkpoint=bundle["checkpoint"],
        prior_event_head_sha256=prior_event_head_sha256,
        release_sha=release_sha,
        plan_sha256=exact_storage_plan()["plan_sha256"],
    )
    if (
        observation["state"] != expected_state
        or bundle["checkpoint"] not in allowed_checkpoints
        or bundle["prior_event_head_sha256"] != prior_event_head_sha256
        or bundle["request_binding_sha256"] != expected_binding
        or bundle["observation_nonce_sha256"]
        != evidence.observation_nonce_sha256(
            request_binding_sha256=expected_binding,
            transaction_id=transaction_id,
            checkpoint=bundle["checkpoint"],
        )
    ):
        raise PasskeyV2StorageBoundaryError(
            "passkey_v2_storage_observation_binding_invalid"
        )
    plan = exact_storage_plan()
    projection = evidence.observation_projection(observation)
    payload = {
        "schema": STORAGE_ACTION_SCHEMA,
        "operation": "canary_boot_disk_growth_40_to_80",
        "project": PROJECT,
        "zone": ZONE,
        "vm_name": VM_NAME,
        "vm_instance_id": VM_INSTANCE_ID,
        "disk_name": DISK_NAME,
        "disk_id": DISK_ID,
        "boot_device_name": BOOT_DEVICE_NAME,
        "source_size_gb": SOURCE_SIZE_GB,
        "target_size_gb": TARGET_SIZE_GB,
        "conditional_reboot_approved": True,
        "allowed_operations": list(_OPERATIONS_BY_STAGE[stage]),
        "storage_plan": plan,
        "storage_plan_sha256": plan["plan_sha256"],
        "source_preflight": dict(bundle),
        "source_preflight_sha256": protocol.sha256_json(bundle),
        "live_projection": projection,
        "live_projection_sha256": protocol.sha256_json(projection),
        "external_iam_receipt": observation["external_iam_receipt"],
        "external_iam_receipt_sha256": observation[
            "external_iam_receipt_sha256"
        ],
        "current_host_receipt_file_sha256": observation[
            "current_host_receipt_file_sha256"
        ],
        "current_host_receipt_sha256": observation[
            "current_host_receipt_sha256"
        ],
        "current_stopped_release_receipt_file_sha256": observation[
            "current_stopped_release_receipt_file_sha256"
        ],
        "current_stopped_release_receipt_sha256": observation[
            "current_stopped_release_receipt_sha256"
        ],
    }
    request_id = protocol.sha256_json({
        "schema": "muncho-passkey-v2-storage-request-id.v1",
        "transaction_id": transaction_id,
        "stage": stage,
        "source_preflight_sha256": payload["source_preflight_sha256"],
        "prior_authoritative_receipt_sha256": (
            prior_authoritative_receipt_sha256
        ),
        "prior_event_head_sha256": prior_event_head_sha256,
        "release_sha": release_sha,
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
        executor_release_sha=release_sha,
        executor_plan_sha256=plan["plan_sha256"],
        transaction_id=transaction_id,
        stage=stage,
        webauthn_rp_id=protocol.PRODUCTION_RP_ID,
        webauthn_origin=protocol.PRODUCTION_ORIGIN,
        authority_release_sha=release_sha,
        authority_manifest_sha256=authority_manifest_sha256,
        authority_host_receipt_sha256=authority_host_receipt_sha256,
        source_preflight_sha256=payload["source_preflight_sha256"],
        live_projection_sha256=payload["live_projection_sha256"],
        external_iam_receipt_sha256=payload[
            "external_iam_receipt_sha256"
        ],
        prior_authoritative_receipt_sha256=(
            prior_authoritative_receipt_sha256
        ),
        prior_event_head_sha256=prior_event_head_sha256,
        issued_at_unix=issued_at_unix,
        approval_ttl_seconds=300,
    )


class ProductionStorageGrowthBoundary:
    def __init__(self, release_sha: str, transport: DedicatedOwnerGateTransport) -> None:
        if not isinstance(release_sha, str) or re.fullmatch(r"[0-9a-f]{40}", release_sha) is None:
            raise PasskeyV2StorageBoundaryError("passkey_v2_boundary_release_invalid")
        if not callable(getattr(transport, "invoke_owner_gate", None)):
            raise PasskeyV2StorageBoundaryError("passkey_v2_owner_gate_transport_required")
        if callable(getattr(transport, "run_local_compute_mutation", None)):
            raise PasskeyV2StorageBoundaryError("passkey_v2_local_mutation_forbidden")
        self.release_sha = release_sha
        self._transport = transport

    def _invoke(self, operation: str, document: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation not in {
            "preflight",
            "request_initial",
            "request_resume",
            "execute_or_recover",
            "verify_terminal",
            "observation_request",
            "attest_cloud_observation",
        }:
            raise PasskeyV2StorageBoundaryError("passkey_v2_remote_operation_invalid")
        unsigned = {
            "schema": REMOTE_FRAME_SCHEMA,
            "operation": operation,
            "release_sha": self.release_sha,
            "document": dict(document),
        }
        frame = {**unsigned, "frame_sha256": protocol.sha256_json(unsigned)}
        raw = self._transport.invoke_owner_gate(protocol.canonical_json_bytes(frame))
        value = protocol.decode_canonical_json(raw)
        if not isinstance(value, Mapping):
            raise PasskeyV2StorageBoundaryError("passkey_v2_remote_response_invalid")
        response = dict(value)
        unsigned_response = {
            key: item for key, item in response.items() if key != "response_sha256"
        }
        if (
            set(response)
            != {"schema", "operation", "release_sha", "ok", "document", "response_sha256"}
            or response.get("schema") != REMOTE_RESPONSE_SCHEMA
            or response.get("operation") != operation
            or response.get("release_sha") != self.release_sha
            or response.get("ok") is not True
            or not isinstance(response.get("document"), Mapping)
            or response.get("response_sha256")
            != protocol.sha256_json(unsigned_response)
        ):
            raise PasskeyV2StorageBoundaryError("passkey_v2_remote_response_invalid")
        return dict(response["document"])

    def require_ready(self) -> Mapping[str, Any]:
        receipt = self._invoke("preflight", {})
        required = {
            "owner_gate_vm_name": OWNER_GATE_VM_NAME,
            "web_uid": OWNER_GATE_WEB_UID,
            "authority_uid": OWNER_GATE_AUTHORITY_UID,
            "executor_uid": OWNER_GATE_EXECUTOR_UID,
            "authority_socket": AUTHORITY_SOCKET,
            "executor_socket": EXECUTOR_SOCKET,
            "authority_db": AUTHORITY_DB,
            "executor_db": EXECUTOR_DB,
            "rp_id": protocol.PRODUCTION_RP_ID,
            "origin": protocol.PRODUCTION_ORIGIN,
            "iap_only": True,
            "local_compute_mutation_available": False,
            "sqlite_synchronous": "FULL",
            "sqlite_begin_immediate": True,
            "totp_dangerous_actions": False,
        }
        if any(receipt.get(name) != expected for name, expected in required.items()):
            raise PasskeyV2StorageBoundaryError("passkey_v2_owner_gate_not_ready")
        attestors = receipt.get("observation_attestors")
        if not isinstance(attestors, Mapping) or set(attestors) != {
            "cloud", "host"
        }:
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_observation_attestor_trust_invalid"
            )
        for role in ("cloud", "host"):
            record = attestors[role]
            if not isinstance(record, Mapping) or set(record) != {
                "public_key_id", "public_key_b64url"
            }:
                raise PasskeyV2StorageBoundaryError(
                    "passkey_v2_observation_attestor_trust_invalid"
                )
            encoded = record["public_key_b64url"]
            if not isinstance(encoded, str):
                raise PasskeyV2StorageBoundaryError(
                    "passkey_v2_observation_attestor_trust_invalid"
                )
            try:
                raw = base64.urlsafe_b64decode(
                    (encoded + "=" * (-len(encoded) % 4)).encode("ascii")
                )
            except (AttributeError, UnicodeError, ValueError) as exc:
                raise PasskeyV2StorageBoundaryError(
                    "passkey_v2_observation_attestor_trust_invalid"
                ) from None
            if (
                len(raw) != 32
                or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
                != encoded
                or protocol.sha256_bytes(raw) != record["public_key_id"]
            ):
                raise PasskeyV2StorageBoundaryError(
                    "passkey_v2_observation_attestor_trust_invalid"
                )
        return receipt

    def _validate_fixed_call(
        self,
        *,
        release_sha: str,
        plan: Mapping[str, Any],
        transaction_id: str,
    ) -> None:
        if (
            release_sha != self.release_sha
            or validate_storage_plan(plan) != exact_storage_plan()
            or not isinstance(transaction_id, str)
            or _SHA256.fullmatch(transaction_id) is None
        ):
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_boundary_call_binding_invalid"
            )

    def request_initial(
        self,
        *,
        release_sha: str,
        plan: Mapping[str, Any],
        source_preflight: Mapping[str, Any],
        transaction_id: str,
    ) -> Mapping[str, Any]:
        self._validate_fixed_call(
            release_sha=release_sha,
            plan=plan,
            transaction_id=transaction_id,
        )
        bundle = evidence.validate_attested_observation_structure(
            source_preflight
        )
        if transaction_id_for_source(bundle) != transaction_id:
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_transaction_id_invalid"
            )
        return self._invoke(
            "request_initial",
            {
                "plan_sha256": exact_storage_plan()["plan_sha256"],
                "transaction_id": transaction_id,
                "source_preflight": bundle,
            },
        )

    def observation_request(
        self,
        *,
        release_sha: str,
        plan: Mapping[str, Any],
        transaction_id: str,
    ) -> Mapping[str, Any]:
        self._validate_fixed_call(
            release_sha=release_sha,
            plan=plan,
            transaction_id=transaction_id,
        )
        result = self._invoke(
            "observation_request",
            {
                "plan_sha256": exact_storage_plan()["plan_sha256"],
                "transaction_id": transaction_id,
            },
        )
        unsigned = {
            key: item for key, item in result.items()
            if key != "observation_request_sha256"
        }
        if result.get("observation_request_sha256") != protocol.sha256_json(
            unsigned
        ):
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_observation_request_invalid"
            )
        return result

    def request_resume(
        self,
        *,
        release_sha: str,
        plan: Mapping[str, Any],
        transaction_id: str,
        continuation_preflight: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._validate_fixed_call(
            release_sha=release_sha,
            plan=plan,
            transaction_id=transaction_id,
        )
        bundle = evidence.validate_attested_observation_structure(
            continuation_preflight
        )
        if bundle["transaction_id"] != transaction_id:
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_transaction_id_invalid"
            )
        return self._invoke(
            "request_resume",
            {
                "plan_sha256": exact_storage_plan()["plan_sha256"],
                "transaction_id": transaction_id,
                "continuation_preflight": bundle,
            },
        )

    def attest_cloud_observation(
        self,
        *,
        release_sha: str,
        plan: Mapping[str, Any],
        transaction_id: str,
        attestation_request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._validate_fixed_call(
            release_sha=release_sha,
            plan=plan,
            transaction_id=transaction_id,
        )
        if (
            not isinstance(attestation_request, Mapping)
            or attestation_request.get("role") != "cloud"
            or not isinstance(
                attestation_request.get("observation_request"), Mapping
            )
            or attestation_request["observation_request"].get(
                "transaction_id"
            )
            != transaction_id
        ):
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_cloud_attestation_request_invalid"
            )
        return self._invoke(
            "attest_cloud_observation",
            {
                "plan_sha256": exact_storage_plan()["plan_sha256"],
                "transaction_id": transaction_id,
                "attestation_request": dict(attestation_request),
            },
        )

    def execute_or_recover(
        self,
        *,
        release_sha: str,
        plan: Mapping[str, Any],
        transaction_id: str,
        request_id: str,
        consume_attempt_id: str,
        continuation_preflight: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._validate_fixed_call(
            release_sha=release_sha,
            plan=plan,
            transaction_id=transaction_id,
        )
        if not isinstance(consume_attempt_id, str) or _SHA256.fullmatch(consume_attempt_id) is None:
            raise PasskeyV2StorageBoundaryError("passkey_v2_execution_attempt_invalid")
        protocol.validate_request_id(request_id)
        bundle = evidence.validate_attested_observation_structure(
            continuation_preflight
        )
        if bundle["transaction_id"] != transaction_id:
            raise PasskeyV2StorageBoundaryError(
                "passkey_v2_transaction_id_invalid"
            )
        return self._invoke(
            "execute_or_recover",
            {
                "request_id": request_id,
                "consume_attempt_id": consume_attempt_id,
                "transaction_id": transaction_id,
                "continuation_preflight": bundle,
            },
        )

    def verify_terminal(
        self,
        *,
        release_sha: str,
        plan: Mapping[str, Any],
        transaction_id: str,
    ) -> Mapping[str, Any]:
        self._validate_fixed_call(
            release_sha=release_sha,
            plan=plan,
            transaction_id=transaction_id,
        )
        return self._invoke(
            "verify_terminal",
            {
                "plan_sha256": exact_storage_plan()["plan_sha256"],
                "transaction_id": transaction_id,
            },
        )


def production_storage_growth_boundary(
    release_sha: str,
    gcloud_executable: Any,
    gcloud_configuration: Any,
    owner_identity: Any,
) -> ProductionStorageGrowthBoundary:
    """Build only from an already pinned dedicated-owner-gate transport.

    The broad historical parameters remain in the signature for launcher
    compatibility, but paths/credentials/configurations are never interpreted
    here.  Exactly one object must implement the narrow transport protocol.
    """

    candidates = [gcloud_executable, gcloud_configuration, owner_identity]
    transports = [
        item for item in candidates if callable(getattr(item, "invoke_owner_gate", None))
    ]
    if len(transports) != 1:
        raise PasskeyV2StorageBoundaryError("passkey_v2_owner_gate_transport_required")
    return ProductionStorageGrowthBoundary(release_sha, transports[0])
