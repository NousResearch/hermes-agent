#!/usr/bin/env python3
"""Pure validators for storage-growth source and continuation evidence.

The collector and privileged executor share these validators.  This module is
data/schema validation only: it contains no approval, journal, transport, or
mutation authority.
"""

from __future__ import annotations

import base64
import re
from typing import Any, Callable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import storage_growth_contract as contract
from scripts.canary.runtime_units import CANARY_RUNTIME_UNITS


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECK_NAMES = {
    "resource.project_zone_exact",
    "resource.instance_attachment_exact",
    "resource.disk_identity_and_size_exact",
    "runtime.units_inventory_exact",
    "runtime.units_stopped_exact",
    "canonical.receipts_exact",
    "iam.policy_live_exact",
    "filesystem.root_identity_exact",
    "filesystem.capacity_and_headroom_exact",
}
_PREFLIGHT_FIELDS = {
    "schema", "ok", "state", "collected_at_unix", "plan_sha256",
    "project", "zone", "owner_account", "vm_name", "vm_instance_id",
    "disk_name", "disk_id", "boot_device_name", "instance_status",
    "disk_size_gb", "boot_id_sha256", "root_source", "root_filesystem",
    "root_mountpoint", "root_size_bytes", "root_available_bytes",
    "canonical_receipt_source", "current_stopped_release_sha",
    "current_host_receipt_file_sha256", "current_host_receipt_sha256",
    "current_stopped_release_receipt_file_sha256",
    "current_stopped_release_receipt_sha256", "external_iam_receipt_sha256",
    "external_iam_receipt", "external_iam_policy_sha256",
    "external_iam_collected_at_unix", "external_iam_expires_at_unix",
    "runtime_units", "service_states", "service_states_sha256", "checks",
    "report_sha256",
}
_IAM_FIELDS = {
    "schema", "project", "zone", "instance", "service_account", "scopes",
    "roles", "permissions", "foundation_plan_sha256", "host_plan_sha256",
    "foundation_report_sha256", "host_report_sha256",
    "source_approval_sha256", "collected_at_unix", "expires_at_unix",
}
_SERVICE_PROPERTIES = (
    "LoadState", "ActiveState", "SubState", "UnitFileState", "MainPID",
    "FragmentPath", "DropInPaths",
)
ATTESTED_OBSERVATION_SCHEMA = "muncho-storage-growth-attested-observation.v1"
ATTESTATION_SCHEMA = "muncho-owner-gate-observation-attestation.v1"
OBSERVATION_BUNDLE_TTL_SECONDS = 300
OBSERVATION_CHECKPOINTS = frozenset({
    "source", "post_resize", "post_stop", "post_start",
})


class StorageGrowthEvidenceError(RuntimeError):
    """Stable strict-evidence validation failure."""


def observation_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the exact facts used for live equivalence and UI binding."""

    return {
        "schema": "muncho-storage-growth-live-observation-projection.v1",
        "report_sha256": value["report_sha256"],
        "state": value["state"],
        "collected_at_unix": value["collected_at_unix"],
        "project": value["project"],
        "zone": value["zone"],
        "vm_name": value["vm_name"],
        "vm_instance_id": value["vm_instance_id"],
        "disk_name": value["disk_name"],
        "disk_id": value["disk_id"],
        "boot_device_name": value["boot_device_name"],
        "instance_status": value["instance_status"],
        "disk_size_gb": value["disk_size_gb"],
        "boot_id_sha256": value["boot_id_sha256"],
        "root_size_bytes": value["root_size_bytes"],
        "root_available_bytes": value["root_available_bytes"],
        "service_states_sha256": value["service_states_sha256"],
        "external_iam_receipt_sha256": value["external_iam_receipt_sha256"],
        "external_iam_policy_sha256": value["external_iam_policy_sha256"],
    }


def observation_request_binding_sha256(
    *,
    transaction_id: str,
    checkpoint: str,
    prior_event_head_sha256: str,
    release_sha: str,
    plan_sha256: str,
) -> str:
    """Deterministic remote observation request bound to one checkpoint."""

    return protocol.sha256_json({
        "schema": "muncho-storage-growth-observation-request.v1",
        "transaction_id": transaction_id,
        "checkpoint": checkpoint,
        "prior_event_head_sha256": prior_event_head_sha256,
        "release_sha": release_sha,
        "plan_sha256": plan_sha256,
    })


def observation_nonce_sha256(
    *, request_binding_sha256: str, transaction_id: str, checkpoint: str
) -> str:
    return protocol.sha256_json({
        "schema": "muncho-storage-growth-observation-nonce.v1",
        "request_binding_sha256": request_binding_sha256,
        "transaction_id": transaction_id,
        "checkpoint": checkpoint,
    })


def _attestation_payload(bundle: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
    return {
        "schema": ATTESTED_OBSERVATION_SCHEMA,
        "role": role,
        **{
            name: bundle[name]
            for name in (
                "transaction_id",
                "checkpoint",
                "observation_nonce_sha256",
                "request_binding_sha256",
                "prior_event_head_sha256",
                "observation_request_sha256",
                "collection_attempt_id",
                "collection_attempt_sequence",
                "collection_attempt_issued_at_unix",
                "collection_attempt_expires_at_unix",
                "observed_at_unix",
                "expires_at_unix",
                "observation_sha256",
                "observation",
                "trusted_iam_projection_sha256",
                "trusted_iam_projection",
            )
        },
    }


def build_attested_observation(
    *,
    observation: Mapping[str, Any],
    transaction_id: str,
    checkpoint: str,
    prior_event_head_sha256: str,
    request_binding_sha256: str,
    observation_nonce_sha256_value: str,
    observation_request_sha256: str,
    collection_attempt_id: str,
    collection_attempt_sequence: int,
    collection_attempt_issued_at_unix: int,
    collection_attempt_expires_at_unix: int,
    trusted_iam_projection: Mapping[str, Any],
    cloud_public_key_id: str,
    cloud_signer: Callable[[bytes], bytes],
    host_public_key_id: str | None,
    host_signer: Callable[[bytes], bytes] | None,
) -> Mapping[str, Any]:
    """Seal one already-collected report with dedicated collector signers."""

    report = dict(observation)
    unsigned_bundle: dict[str, Any] = {
        "schema": ATTESTED_OBSERVATION_SCHEMA,
        "transaction_id": transaction_id,
        "checkpoint": checkpoint,
        "observation_nonce_sha256": observation_nonce_sha256_value,
        "request_binding_sha256": request_binding_sha256,
        "prior_event_head_sha256": prior_event_head_sha256,
        "observation_request_sha256": observation_request_sha256,
        "collection_attempt_id": collection_attempt_id,
        "collection_attempt_sequence": collection_attempt_sequence,
        "collection_attempt_issued_at_unix": (
            collection_attempt_issued_at_unix
        ),
        "collection_attempt_expires_at_unix": (
            collection_attempt_expires_at_unix
        ),
        "observed_at_unix": report.get("collected_at_unix"),
        "expires_at_unix": min(
            int(report.get("collected_at_unix", 0))
            + OBSERVATION_BUNDLE_TTL_SECONDS,
            collection_attempt_expires_at_unix,
            int(trusted_iam_projection.get("expires_at_unix", 0)),
        ),
        "observation": report,
        "observation_sha256": protocol.sha256_json(report),
        "trusted_iam_projection": dict(trusted_iam_projection),
        "trusted_iam_projection_sha256": protocol.sha256_json(
            trusted_iam_projection
        ),
    }

    def seal(role: str, key_id: str, signer: Callable[[bytes], bytes]) -> Mapping[str, Any]:
        signature = signer(protocol.canonical_json_bytes(
            _attestation_payload(unsigned_bundle, role=role)
        ))
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise StorageGrowthEvidenceError(
                "storage_growth_attestation_signature_invalid"
            )
        return {
            "schema": ATTESTATION_SCHEMA,
            "public_key_id": key_id,
            "signature_ed25519_b64url": base64.urlsafe_b64encode(
                signature
            ).rstrip(b"=").decode("ascii"),
        }

    cloud = seal("cloud", cloud_public_key_id, cloud_signer)
    host: Mapping[str, Any] | None
    if host_signer is None:
        if not (
            checkpoint == "post_stop"
            and report.get("state") == "terminated_after_growth_intent"
            and report.get("canonical_receipt_source")
            == "durable_signed_source_snapshot_for_stopped_vm"
            and host_public_key_id is None
        ):
            raise StorageGrowthEvidenceError(
                "storage_growth_host_attestation_required"
            )
        host = None
    else:
        if not isinstance(host_public_key_id, str):
            raise StorageGrowthEvidenceError(
                "storage_growth_host_attestation_required"
            )
        host = seal("host", host_public_key_id, host_signer)
    body = {
        **unsigned_bundle,
        "cloud_attestation": cloud,
        "host_attestation": host,
    }
    bundle = {**body, "bundle_sha256": protocol.sha256_json(body)}
    return validate_attested_observation_structure(bundle)


def validate_attested_observation_structure(value: Any) -> Mapping[str, Any]:
    """Validate exact bundle shape before the executor loads trusted keys."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema", "transaction_id", "checkpoint",
        "observation_nonce_sha256", "request_binding_sha256",
        "prior_event_head_sha256", "observation_request_sha256",
        "collection_attempt_id", "collection_attempt_sequence",
        "collection_attempt_issued_at_unix",
        "collection_attempt_expires_at_unix",
        "observed_at_unix", "expires_at_unix",
        "observation", "observation_sha256", "cloud_attestation",
        "trusted_iam_projection", "trusted_iam_projection_sha256",
        "host_attestation", "bundle_sha256",
    }:
        raise StorageGrowthEvidenceError("storage_growth_attestation_fields_invalid")
    bundle = dict(value)
    observation = bundle.get("observation")
    trusted_iam_projection = bundle.get("trusted_iam_projection")
    if not isinstance(observation, Mapping) or not isinstance(
        trusted_iam_projection, Mapping
    ):
        raise StorageGrowthEvidenceError(
            "storage_growth_attestation_bundle_invalid"
        )
    observed_at_unix = bundle.get("observed_at_unix")
    expires_at_unix = bundle.get("expires_at_unix")
    collection_attempt_expires_at_unix = bundle.get(
        "collection_attempt_expires_at_unix"
    )
    trusted_iam_expires_at_unix = trusted_iam_projection.get(
        "expires_at_unix"
    )
    if (
        type(observed_at_unix) is not int
        or type(expires_at_unix) is not int
        or type(collection_attempt_expires_at_unix) is not int
        or type(trusted_iam_expires_at_unix) is not int
    ):
        raise StorageGrowthEvidenceError(
            "storage_growth_attestation_bundle_invalid"
        )
    if (
        bundle.get("schema") != ATTESTED_OBSERVATION_SCHEMA
        or not _sha(bundle.get("transaction_id"))
        or bundle.get("checkpoint") not in OBSERVATION_CHECKPOINTS
        or not _sha(bundle.get("observation_nonce_sha256"))
        or not _sha(bundle.get("request_binding_sha256"))
        or not _sha(bundle.get("prior_event_head_sha256"))
        or not _sha(bundle.get("observation_request_sha256"))
        or not _sha(bundle.get("collection_attempt_id"))
        or type(bundle.get("collection_attempt_sequence")) is not int
        or bundle["collection_attempt_sequence"] < 1
        or type(bundle.get("collection_attempt_issued_at_unix")) is not int
        or bundle["collection_attempt_issued_at_unix"] < 1
        or bundle.get("collection_attempt_expires_at_unix")
        != bundle["collection_attempt_issued_at_unix"]
        + OBSERVATION_BUNDLE_TTL_SECONDS
        or bundle.get("observation_sha256") != protocol.sha256_json(observation)
        or bundle.get("trusted_iam_projection_sha256")
        != protocol.sha256_json(trusted_iam_projection)
        or observed_at_unix != observation.get("collected_at_unix")
        or expires_at_unix
        != min(
            observed_at_unix + OBSERVATION_BUNDLE_TTL_SECONDS,
            collection_attempt_expires_at_unix,
            trusted_iam_expires_at_unix,
        )
        or expires_at_unix <= observed_at_unix
        or bundle.get("bundle_sha256")
        != protocol.sha256_json(
            {key: item for key, item in bundle.items() if key != "bundle_sha256"}
        )
    ):
        raise StorageGrowthEvidenceError("storage_growth_attestation_bundle_invalid")
    for role in ("cloud", "host"):
        attestation = bundle.get(f"{role}_attestation")
        if (
            role == "host"
            and bundle["checkpoint"] == "post_stop"
            and observation.get("state") == "terminated_after_growth_intent"
            and attestation is None
        ):
            continue
        if (
            not isinstance(attestation, Mapping)
            or set(attestation)
            != {"schema", "public_key_id", "signature_ed25519_b64url"}
            or attestation.get("schema") != ATTESTATION_SCHEMA
            or not _sha(attestation.get("public_key_id"))
            or not isinstance(attestation.get("signature_ed25519_b64url"), str)
            or not attestation["signature_ed25519_b64url"]
            or "=" in attestation["signature_ed25519_b64url"]
        ):
            raise StorageGrowthEvidenceError("storage_growth_attestation_invalid")
    return bundle


def validate_attested_observation(
    value: Any,
    *,
    cloud_public_key: Ed25519PublicKey,
    cloud_public_key_id: str,
    host_public_key: Ed25519PublicKey,
    host_public_key_id: str,
    now_unix: int,
    allowed_states: frozenset[str],
    expected_transaction_id: str,
    expected_checkpoint: str,
    expected_request_binding_sha256: str,
    expected_prior_event_head_sha256: str,
    expected_observation_request_sha256: str,
    expected_collection_attempt_id: str,
    expected_collection_attempt_sequence: int,
    expected_collection_attempt_issued_at_unix: int,
    expected_collection_attempt_expires_at_unix: int,
) -> Mapping[str, Any]:
    bundle = validate_attested_observation_structure(value)
    keys = {
        "cloud": (cloud_public_key, cloud_public_key_id),
        "host": (host_public_key, host_public_key_id),
    }
    observation = dict(bundle["observation"])
    if (
        bundle["transaction_id"] != expected_transaction_id
        or bundle["checkpoint"] != expected_checkpoint
        or bundle["request_binding_sha256"]
        != expected_request_binding_sha256
        or bundle["prior_event_head_sha256"]
        != expected_prior_event_head_sha256
        or bundle["observation_request_sha256"]
        != expected_observation_request_sha256
        or bundle["collection_attempt_id"]
        != expected_collection_attempt_id
        or bundle["collection_attempt_sequence"]
        != expected_collection_attempt_sequence
        or bundle["collection_attempt_issued_at_unix"]
        != expected_collection_attempt_issued_at_unix
        or bundle["collection_attempt_expires_at_unix"]
        != expected_collection_attempt_expires_at_unix
        or bundle["observation_nonce_sha256"]
        != observation_nonce_sha256(
            request_binding_sha256=expected_request_binding_sha256,
            transaction_id=expected_transaction_id,
            checkpoint=expected_checkpoint,
        )
        or not bundle["observed_at_unix"] <= now_unix
        < bundle["expires_at_unix"]
    ):
        raise StorageGrowthEvidenceError(
            "storage_growth_attestation_binding_invalid"
        )
    for role, (public_key, expected_id) in keys.items():
        if not isinstance(public_key, Ed25519PublicKey):
            raise StorageGrowthEvidenceError("storage_growth_attestation_key_invalid")
        actual_id = protocol.sha256_bytes(public_key.public_bytes_raw())
        attestation = bundle[f"{role}_attestation"]
        if attestation is None:
            if (
                role == "host"
                and expected_checkpoint == "post_stop"
                and observation.get("state")
                == "terminated_after_growth_intent"
                and observation.get("canonical_receipt_source")
                == "durable_signed_source_snapshot_for_stopped_vm"
            ):
                continue
            raise StorageGrowthEvidenceError(
                "storage_growth_attestation_invalid"
            )
        if actual_id != expected_id or attestation["public_key_id"] != expected_id:
            raise StorageGrowthEvidenceError("storage_growth_attestation_key_invalid")
        encoded = attestation["signature_ed25519_b64url"]
        try:
            signature = base64.urlsafe_b64decode(
                (encoded + "=" * (-len(encoded) % 4)).encode("ascii")
            )
            if (
                len(signature) != 64
                or base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
                != encoded
            ):
                raise ValueError("noncanonical signature")
            public_key.verify(
                signature,
                protocol.canonical_json_bytes(
                    _attestation_payload(bundle, role=role)
                ),
            )
        except (InvalidSignature, UnicodeError, ValueError, TypeError) as exc:
            raise StorageGrowthEvidenceError(
                "storage_growth_attestation_signature_invalid"
            ) from None
    return validate_observation(
        observation,
        now_unix=now_unix,
        require_fresh=True,
        allowed_states=allowed_states,
    )


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_services(value: Any) -> list[Mapping[str, Any]] | None:
    if not isinstance(value, list) or len(value) != len(CANARY_RUNTIME_UNITS):
        return None
    result: list[Mapping[str, Any]] = []
    for unit, item in zip(CANARY_RUNTIME_UNITS, value, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"unit", "state", "properties"}
            or item.get("unit") != unit
            or item.get("state") not in {"absent", "disabled_inactive"}
            or not isinstance(item.get("properties"), Mapping)
            or set(item["properties"]) != set(_SERVICE_PROPERTIES)
            or any(not isinstance(item["properties"][name], str) for name in _SERVICE_PROPERTIES)
        ):
            return None
        expected = (
            {
                "LoadState": "not-found", "ActiveState": "inactive",
                "SubState": "dead", "UnitFileState": "", "MainPID": "0",
                "FragmentPath": "", "DropInPaths": "",
            }
            if item["state"] == "absent"
            else {
                "LoadState": "loaded", "ActiveState": "inactive",
                "SubState": "dead", "UnitFileState": "disabled",
                "MainPID": "0", "FragmentPath": f"/etc/systemd/system/{unit}",
                "DropInPaths": "",
            }
        )
        if dict(item["properties"]) != expected:
            return None
        result.append(dict(item))
    return result


def _validate_checks(value: Any, *, all_passed: bool) -> bool:
    if not isinstance(value, list) or len(value) != len(_CHECK_NAMES):
        return False
    names: set[str] = set()
    for item in value:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"name", "passed", "detail"}
            or item.get("name") not in _CHECK_NAMES
            or type(item.get("passed")) is not bool
            or (all_passed and item.get("passed") is not True)
            or not isinstance(item.get("detail"), str)
            or not item["detail"]
            or item["name"] in names
        ):
            return False
        names.add(str(item["name"]))
    return names == _CHECK_NAMES


def validate_external_iam(
    value: Any,
    *,
    receipt_sha256: str,
    now_unix: int,
    minimum_remaining_seconds: int = 720,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IAM_FIELDS:
        raise StorageGrowthEvidenceError("storage_growth_iam_fields_invalid")
    receipt = dict(value)
    if (
        receipt.get("schema") != "muncho-writer-external-iam-evidence.v1"
        or receipt.get("project") != contract.PROJECT
        or receipt.get("zone") != contract.ZONE
        or receipt.get("instance") != contract.VM_NAME
        or receipt.get("service_account") != contract.RUNTIME_SERVICE_ACCOUNT
        or tuple(receipt.get("scopes") or ()) != contract.RUNTIME_SCOPES
        or tuple(receipt.get("roles") or ()) != contract.RUNTIME_ROLES
        or tuple(receipt.get("permissions") or ()) != contract.RUNTIME_PERMISSIONS
        or any(
            not _sha(receipt.get(name))
            for name in (
                "foundation_plan_sha256", "host_plan_sha256",
                "foundation_report_sha256", "host_report_sha256",
                "source_approval_sha256",
            )
        )
        or receipt.get("source_approval_sha256")
        != contract.canonical_plan_sha256()
        or receipt_sha256 != protocol.sha256_json(receipt)
    ):
        raise StorageGrowthEvidenceError("storage_growth_iam_identity_invalid")
    policy = {
        name: receipt[name]
        for name in (
            "project", "zone", "instance", "service_account", "scopes",
            "roles", "permissions", "foundation_plan_sha256",
            "host_plan_sha256",
        )
    }
    collected = receipt.get("collected_at_unix")
    expires = receipt.get("expires_at_unix")
    if (
        protocol.sha256_json(policy) != contract.EXTERNAL_IAM_POLICY_SHA256
        or type(collected) is not int
        or type(expires) is not int
        or expires - collected != 1200
        or not collected <= now_unix
        or now_unix + minimum_remaining_seconds > expires
    ):
        raise StorageGrowthEvidenceError("storage_growth_iam_freshness_invalid")
    return receipt


def validate_observation(
    value: Any,
    *,
    now_unix: int,
    require_fresh: bool = True,
    allowed_states: frozenset[str] = frozenset({
        "source_ready", "resize_complete_boot_required",
        "terminated_after_growth_intent", "target_ready",
    }),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PREFLIGHT_FIELDS:
        raise StorageGrowthEvidenceError("storage_growth_observation_fields_invalid")
    report = dict(value)
    unsigned = {key: item for key, item in report.items() if key != "report_sha256"}
    collected = report.get("collected_at_unix")
    if (
        report.get("report_sha256") != protocol.sha256_json(unsigned)
        or report.get("schema") != contract.STORAGE_GROWTH_PREFLIGHT_SCHEMA
        or report.get("plan_sha256") != contract.canonical_plan_sha256()
        or report.get("project") != contract.PROJECT
        or report.get("zone") != contract.ZONE
        or report.get("owner_account") != contract.OWNER_ACCOUNT
        or report.get("vm_name") != contract.VM_NAME
        or report.get("vm_instance_id") != contract.VM_INSTANCE_ID
        or report.get("disk_name") != contract.DISK_NAME
        or report.get("disk_id") != contract.DISK_ID
        or report.get("boot_device_name") != contract.BOOT_DEVICE_NAME
        or report.get("runtime_units") != list(CANARY_RUNTIME_UNITS)
        or type(collected) is not int
        or collected < 1
        or collected > now_unix
        or (require_fresh and now_unix - collected > contract.PREFLIGHT_MAX_AGE_SECONDS)
        or report.get("state") not in allowed_states
    ):
        raise StorageGrowthEvidenceError("storage_growth_observation_identity_invalid")
    for name, expected in (
        ("current_stopped_release_sha", contract.CURRENT_STOPPED_RELEASE_SHA),
        ("current_host_receipt_file_sha256", contract.CURRENT_HOST_RECEIPT_FILE_SHA256),
        ("current_host_receipt_sha256", contract.CURRENT_HOST_RECEIPT_SHA256),
        ("current_stopped_release_receipt_file_sha256", contract.CURRENT_STOPPED_RELEASE_RECEIPT_FILE_SHA256),
        ("current_stopped_release_receipt_sha256", contract.CURRENT_STOPPED_RELEASE_RECEIPT_SHA256),
        ("external_iam_policy_sha256", contract.EXTERNAL_IAM_POLICY_SHA256),
    ):
        if report.get(name) != expected:
            raise StorageGrowthEvidenceError("storage_growth_observation_receipt_invalid")
    iam = validate_external_iam(
        report.get("external_iam_receipt"),
        receipt_sha256=str(report.get("external_iam_receipt_sha256")),
        # A historical source report remains valid as durable transaction
        # truth during replay.  Fresh continuation authority is always
        # validated against the current clock below/on the new observation.
        now_unix=now_unix if require_fresh else collected,
    )
    if (
        report.get("external_iam_collected_at_unix") != iam["collected_at_unix"]
        or report.get("external_iam_expires_at_unix") != iam["expires_at_unix"]
        or iam["collected_at_unix"] > collected
        or report.get("external_iam_policy_sha256")
        != contract.EXTERNAL_IAM_POLICY_SHA256
    ):
        raise StorageGrowthEvidenceError("storage_growth_observation_iam_invalid")

    state = report["state"]
    running = state != "terminated_after_growth_intent"
    if running:
        services = _validate_services(report.get("service_states"))
        root_size = report.get("root_size_bytes")
        root_available = report.get("root_available_bytes")
        if type(root_size) is not int or type(root_available) is not int:
            valid_state = False
        else:
            common = (
                report.get("instance_status") == "RUNNING"
                and report.get("canonical_receipt_source")
                == "fresh_running_vm_receipt_bytes"
                and _sha(report.get("boot_id_sha256"))
                and report.get("root_source") == contract.ROOT_SOURCE
                and report.get("root_filesystem") == contract.ROOT_FILESYSTEM
                and report.get("root_mountpoint") == contract.ROOT_MOUNTPOINT
                and 0 <= root_available <= root_size
                and services is not None
                and report.get("service_states_sha256")
                == protocol.sha256_json(services)
                and _validate_checks(report.get("checks"), all_passed=True)
            )
            source = (
                state == "source_ready"
                and report.get("ok") is True
                and report.get("disk_size_gb") == contract.SOURCE_SIZE_GB
                and contract.MINIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
                <= root_size
                <= contract.MAXIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
                and root_available >= contract.MINIMUM_SOURCE_FREE_BYTES
            )
            pending = (
                state == "resize_complete_boot_required"
                and report.get("ok") is False
                and report.get("disk_size_gb") == contract.TARGET_SIZE_GB
                and contract.MINIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
                <= root_size
                <= contract.MAXIMUM_SOURCE_ROOT_FILESYSTEM_BYTES
            )
            target = (
                state == "target_ready"
                and report.get("ok") is True
                and report.get("disk_size_gb") == contract.TARGET_SIZE_GB
                and contract.MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES
                <= root_size
                <= contract.MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES
                and root_available >= contract.MINIMUM_FREE_BYTES
            )
            valid_state = common and (source or pending or target)
    else:
        valid_state = (
            report.get("instance_status") == "TERMINATED"
            and report.get("canonical_receipt_source")
            == "durable_signed_source_snapshot_for_stopped_vm"
            and report.get("ok") is False
            and report.get("disk_size_gb") == contract.TARGET_SIZE_GB
            and report.get("boot_id_sha256") is None
            and report.get("root_source") is None
            and report.get("root_filesystem") is None
            and report.get("root_mountpoint") is None
            and report.get("root_size_bytes") is None
            and report.get("root_available_bytes") is None
            and report.get("service_states") == []
            and report.get("service_states_sha256") == protocol.sha256_json([])
            and _validate_checks(report.get("checks"), all_passed=False)
        )
    if not valid_state:
        raise StorageGrowthEvidenceError("storage_growth_observation_state_invalid")
    return report


def validate_initial_observation(value: Any, *, now_unix: int) -> Mapping[str, Any]:
    return validate_observation(
        value,
        now_unix=now_unix,
        allowed_states=frozenset({"source_ready"}),
    )
