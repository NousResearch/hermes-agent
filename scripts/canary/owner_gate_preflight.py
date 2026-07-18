#!/usr/bin/env python3
"""Strict read-only preflight schemas for the private Muncho owner gate.

The module validates sealed observations supplied by a trusted collector.  It
does not acquire credentials, invoke ``gcloud``, SSH to a host, or mutate any
resource.  Cloud collection is described as exact read-only REST requests so a
caller cannot silently fall back to ambient local CLI authority.
"""

from __future__ import annotations

import base64
import ipaddress
import re
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import owner_gate_foundation as foundation


CLOUD_OBSERVATION_SCHEMA = "muncho-owner-gate-cloud-observation.v1"
HOST_OBSERVATION_SCHEMA = "muncho-owner-gate-host-observation.v1"
PREFLIGHT_SCHEMA = "muncho-owner-gate-inert-preflight.v1"
POST_IAM_PREFLIGHT_SCHEMA = "muncho-owner-gate-post-iam-preflight.v1"

WEB_UID = 29101
AUTHORITY_UID = 29102
EXECUTOR_UID = 29103
AUTHORITY_DB = "/var/lib/muncho-owner-gate/authority/passkey-v2.sqlite3"
EXECUTOR_DB = (
    "/var/lib/muncho-owner-gate/executor/execution-v2.sqlite3"
)
MUTATION_JOURNAL_ROOT = "/var/lib/muncho-owner-gate/executor"
EXPECTED_CREDENTIAL_ID_SHA256 = (
    "63bbfca0778101d21dddf2b53cc774460565042391b918eb2d1c87b9d6d19860"
)
EXPECTED_PUBLIC_KEY_SHA256 = (
    "478c0bd2ee54f733dbb63acd329ad35188a7f091f9c6bdc4b6e64e7d59d5db89"
)
EXPECTED_USER_HANDLE_SHA256 = (
    "a72512de5fcd7fa3e679fcca570c9b4db6ff1e403b6329586ddad90c093ad983"
)
REQUIRED_HARDENING = {
    "NoNewPrivileges": "yes",
    "ProtectSystem": "strict",
    "ProtectHome": "yes",
    "PrivateTmp": "yes",
    "PrivateDevices": "yes",
    "ProtectClock": "yes",
    "ProtectControlGroups": "yes",
    "ProtectKernelLogs": "yes",
    "ProtectKernelModules": "yes",
    "ProtectKernelTunables": "yes",
    "ProtectProc": "invisible",
    "LockPersonality": "yes",
    "MemoryDenyWriteExecute": "yes",
    "RestrictRealtime": "yes",
    "RestrictSUIDSGID": "yes",
    "RestrictNamespaces": "yes",
}
EXPECTED_UNIT_PROPERTIES = {
    "web": {
        **REQUIRED_HARDENING,
        "User": "muncho-passkey-web",
        "Group": "muncho-passkey-web",
        "ExecStart": (
            "/opt/muncho-owner-gate/current/venv/bin/python -I -B "
            "/opt/muncho-owner-gate/current/bin/muncho-passkey-v2-web "
            "serve-web --config /etc/muncho-owner-gate/web.json"
        ),
        "PrivateNetwork": "no",
        "ActiveState": "inactive",
        "UnitFileState": "disabled",
    },
    "authority": {
        **REQUIRED_HARDENING,
        "User": "muncho-passkey-authority",
        "Group": "muncho-passkey-authority",
        "ExecStart": (
            "/opt/muncho-owner-gate/current/venv/bin/python -I -B "
            "/opt/muncho-owner-gate/current/bin/muncho-passkey-v2-authority "
            "serve-authority --config /etc/muncho-owner-gate/authority.json "
            "--socket-activation"
        ),
        "PrivateNetwork": "yes",
        "ActiveState": "inactive",
        "UnitFileState": "disabled",
    },
    "executor": {
        **REQUIRED_HARDENING,
        "User": "muncho-storage-executor",
        "Group": "muncho-storage-executor",
        "ExecStart": (
            "/opt/muncho-owner-gate/current/venv/bin/python -I -B "
            "/opt/muncho-owner-gate/current/bin/muncho-passkey-v2-executor "
            "serve-executor --config /etc/muncho-owner-gate/executor.json "
            "--socket-activation"
        ),
        "PrivateNetwork": "no",
        "ActiveState": "inactive",
        "UnitFileState": "disabled",
    },
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,24}$")
_B64URL_UNPADDED = re.compile(r"^[A-Za-z0-9_-]+$")
INSTANCE_PERMISSION_PROBE = (
    "compute.instances.delete",
    "compute.instances.get",
    "compute.instances.setIamPolicy",
    "compute.instances.setMachineType",
    "compute.instances.setMetadata",
    "compute.instances.setServiceAccount",
    "compute.instances.start",
    "compute.instances.stop",
)
DISK_PERMISSION_PROBE = (
    "compute.disks.createSnapshot",
    "compute.disks.delete",
    "compute.disks.get",
    "compute.disks.resize",
    "compute.disks.setIamPolicy",
)
SERVICE_ACCOUNT_PERMISSION_PROBE = (
    "iam.serviceAccountKeys.create",
    "iam.serviceAccounts.actAs",
    "iam.serviceAccounts.setIamPolicy",
)
PROJECT_PERMISSION_PROBE = (
    "resourcemanager.projects.delete",
    "resourcemanager.projects.setIamPolicy",
)


class OwnerGatePreflightError(RuntimeError):
    """Stable, secret-free preflight failure."""


def _strict(raw: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise OwnerGatePreflightError(f"owner_gate_{label}_fields_invalid")
    return raw


def _verify_seal(raw: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if "report_sha256" not in raw:
        raise OwnerGatePreflightError(f"owner_gate_{label}_seal_invalid")
    unsigned = {
        key: value
        for key, value in raw.items()
        if key not in {"report_sha256", "attestation"}
    }
    if (
        not _SHA256.fullmatch(str(raw.get("report_sha256", "")))
        or foundation.sha256_json(unsigned) != raw["report_sha256"]
    ):
        raise OwnerGatePreflightError(f"owner_gate_{label}_seal_invalid")
    return unsigned


def _decode_canonical_ed25519_signature(value: Any, *, label: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 86
        or _B64URL_UNPADDED.fullmatch(value) is None
    ):
        raise OwnerGatePreflightError(
            f"owner_gate_{label}_attestation_invalid"
        )
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise OwnerGatePreflightError(
            f"owner_gate_{label}_attestation_invalid"
        ) from None
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
    ):
        raise OwnerGatePreflightError(
            f"owner_gate_{label}_attestation_invalid"
        )
    return raw


def expected_effective_permission_probe(
    mutation_binding_present: bool,
) -> Mapping[str, Any]:
    instance_granted = (
        [
            "compute.instances.get",
            "compute.instances.start",
            "compute.instances.stop",
        ]
        if mutation_binding_present
        else []
    )
    disk_granted = (
        ["compute.disks.get", "compute.disks.resize"]
        if mutation_binding_present
        else []
    )
    service_account_email = (
        f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
    )
    return {
        "schema": "muncho-owner-gate-effective-permission-probe.v1",
        "inherited_bindings_evaluated": True,
        "conditional_bindings_evaluated": True,
        "instance": {
            "resource": foundation.TARGET_INSTANCE_SELF_LINK,
            "requested_permissions": list(INSTANCE_PERMISSION_PROBE),
            "granted_permissions": instance_granted,
        },
        "disk": {
            "resource": foundation.TARGET_DISK_SELF_LINK,
            "requested_permissions": list(DISK_PERMISSION_PROBE),
            "granted_permissions": disk_granted,
        },
        "service_account": {
            "resource": (
                f"projects/{foundation.PROJECT}/serviceAccounts/"
                f"{service_account_email}"
            ),
            "requested_permissions": list(SERVICE_ACCOUNT_PERMISSION_PROBE),
            "granted_permissions": [],
        },
        "project": {
            "resource": f"projects/{foundation.PROJECT}",
            "requested_permissions": list(PROJECT_PERMISSION_PROBE),
            "granted_permissions": [],
        },
    }


def _verify_attestation(
    raw: Mapping[str, Any],
    *,
    public_key: Ed25519PublicKey,
    expected_public_key_id: str,
    label: str,
) -> str:
    if not isinstance(public_key, Ed25519PublicKey):
        raise OwnerGatePreflightError(f"owner_gate_{label}_key_invalid")
    attestation = _strict(
        raw.get("attestation"),
        {"schema", "public_key_id", "signature_ed25519_b64url"},
        f"{label}_attestation",
    )
    public_bytes = public_key.public_bytes_raw()
    key_id = __import__("hashlib").sha256(public_bytes).hexdigest()
    if (
        attestation["schema"] != "muncho-owner-gate-observation-attestation.v1"
        or attestation["public_key_id"] != key_id
        or key_id != expected_public_key_id
    ):
        raise OwnerGatePreflightError(f"owner_gate_{label}_attestation_invalid")
    try:
        signature = _decode_canonical_ed25519_signature(
            attestation["signature_ed25519_b64url"],
            label=label,
        )
        signed = {
            key: value for key, value in raw.items() if key != "attestation"
        }
        public_key.verify(signature, foundation.canonical_json_bytes(signed))
    except (InvalidSignature, OwnerGatePreflightError) as exc:
        raise OwnerGatePreflightError(
            f"owner_gate_{label}_attestation_invalid"
        ) from None
    return key_id


def read_only_cloud_requests() -> tuple[Mapping[str, str], ...]:
    """Return the complete read-only REST inventory; never execute it locally."""

    project = foundation.PROJECT
    zone = foundation.ZONE
    region = foundation.REGION
    compute = "https://compute.googleapis.com/compute/v1"
    iam = "https://iam.googleapis.com/v1"
    crm = "https://cloudresourcemanager.googleapis.com/v1"
    vpcaccess = "https://vpcaccess.googleapis.com/v1"
    paths = (
        f"{compute}/projects/{project}/zones/{zone}/instances/{foundation.PRODUCTION_SOURCE_VM}",
        f"{compute}/projects/{project}/zones/{zone}/instances/{foundation.VM_NAME}",
        f"{compute}/projects/{project}/zones/{zone}/instances/{foundation.VM_NAME}/getEffectiveFirewalls",
        f"{compute}/projects/{project}/zones/{zone}/instances/{foundation.TARGET_INSTANCE}",
        f"{compute}/projects/{project}/zones/{zone}/disks/{foundation.TARGET_DISK}",
        f"{compute}/projects/{project}/aggregated/subnetworks",
        f"{compute}/projects/{project}/regions/{region}/subnetworks/{foundation.OWNER_GATE_SUBNET_NAME}",
        f"{compute}/projects/{project}/global/networks/{foundation.NETWORK_NAME}",
        f"{compute}/projects/{project}/global/routes",
        f"{compute}/projects/{project}/global/addresses",
        f"{vpcaccess}/projects/{project}/locations/{region}/connectors",
        f"{compute}/projects/{project}/global/firewalls/allow-iap-ssh",
        f"{compute}/projects/{project}/global/firewalls/muncho-owner-gate-web-from-production",
        f"{compute}/projects/{project}/global/firewalls",
        f"{iam}/projects/{project}/serviceAccounts/{foundation.SERVICE_ACCOUNT_NAME}%40{project}.iam.gserviceaccount.com",
        f"{iam}/projects/{project}/serviceAccounts/{foundation.SERVICE_ACCOUNT_NAME}%40{project}.iam.gserviceaccount.com/keys",
        f"{iam}/projects/{project}/roles/{foundation.MUTATION_ROLE_ID}",
    )
    service_account_resource = (
        f"projects/{project}/serviceAccounts/"
        f"{foundation.SERVICE_ACCOUNT_NAME}@{project}.iam.gserviceaccount.com"
    )
    return (
        *({"method": "GET", "url": path} for path in paths),
        {
            "method": "POST",
            "url": (
                f"{compute}/projects/{project}/zones/{zone}/instances/"
                f"{foundation.TARGET_INSTANCE}/testIamPermissions"
            ),
            "body": foundation.canonical_json_bytes({
                "permissions": list(INSTANCE_PERMISSION_PROBE)
            }).decode("ascii"),
        },
        {
            "method": "POST",
            "url": (
                f"{compute}/projects/{project}/zones/{zone}/disks/"
                f"{foundation.TARGET_DISK}/testIamPermissions"
            ),
            "body": foundation.canonical_json_bytes({
                "permissions": list(DISK_PERMISSION_PROBE)
            }).decode("ascii"),
        },
        {
            "method": "POST",
            "url": f"{iam}/{service_account_resource}:testIamPermissions",
            "body": foundation.canonical_json_bytes({
                "permissions": list(SERVICE_ACCOUNT_PERMISSION_PROBE)
            }).decode("ascii"),
        },
        {
            "method": "POST",
            "url": f"{crm}/projects/{project}:testIamPermissions",
            "body": foundation.canonical_json_bytes({
                "permissions": list(PROJECT_PERMISSION_PROBE)
            }).decode("ascii"),
        },
        {
            "method": "POST",
            "url": f"{crm}/projects/{project}:getIamPolicy",
            "body": '{"options":{"requestedPolicyVersion":3}}',
        },
    )


def _validate_cloud(
    raw: Mapping[str, Any],
    *,
    plan_sha256: str,
    public_key: Ed25519PublicKey,
    expected_public_key_id: str,
    mutation_binding_present: bool,
) -> str:
    _strict(raw, {
        "schema", "collected_at_unix", "plan_sha256", "project", "zone",
        "source", "subnet", "instance", "service_account", "iam", "firewalls",
        "targets", "collector", "credential_values_read", "report_sha256",
        "attestation",
    }, "cloud_observation")
    _verify_seal(raw, label="cloud_observation")
    key_id = _verify_attestation(
        raw,
        public_key=public_key,
        expected_public_key_id=expected_public_key_id,
        label="cloud_observation",
    )
    if (
        raw["schema"] != CLOUD_OBSERVATION_SCHEMA
        or raw["plan_sha256"] != plan_sha256
        or raw["project"] != foundation.PROJECT
        or raw["zone"] != foundation.ZONE
        or not isinstance(raw["collected_at_unix"], int)
        or isinstance(raw["collected_at_unix"], bool)
        or raw["collected_at_unix"] <= 0
        or raw["credential_values_read"] is not False
        or raw["collector"] != "trusted_read_only_rest"
    ):
        raise OwnerGatePreflightError("owner_gate_cloud_observation_invalid")

    source = _strict(raw["source"], {
        "name", "numeric_id", "internal_ip", "service_account", "network",
        "subnetwork",
    }, "cloud_source")
    if (
        source["name"] != foundation.PRODUCTION_SOURCE_VM
        or source["numeric_id"] != foundation.PRODUCTION_SOURCE_VM_ID
        or source["internal_ip"] != "10.80.0.2"
        or source["service_account"] != foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT
        or not str(source["network"]).endswith(
            f"/projects/{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
        )
        or not str(source["subnetwork"]).endswith(
            f"/regions/{foundation.REGION}/subnetworks/{foundation.PRODUCTION_SUBNET_NAME}"
        )
    ):
        raise OwnerGatePreflightError("owner_gate_cloud_source_invalid")

    subnet = _strict(raw["subnet"], {
        "name", "network", "cidr", "private_google_access", "stack_type",
        "overlap_count", "route_inventory_sha256",
    }, "cloud_subnet")
    if (
        subnet["name"] != foundation.OWNER_GATE_SUBNET_NAME
        or subnet["cidr"] != foundation.OWNER_GATE_SUBNET_CIDR
        or subnet["private_google_access"] is not True
        or subnet["stack_type"] != "IPV4_ONLY"
        or subnet["overlap_count"] != 0
        or not _SHA256.fullmatch(str(subnet["route_inventory_sha256"]))
        or not str(subnet["network"]).endswith(
            f"/projects/{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
        )
    ):
        raise OwnerGatePreflightError("owner_gate_cloud_subnet_invalid")

    instance = _strict(raw["instance"], {
        "name", "numeric_id", "status", "network", "subnetwork", "internal_ip",
        "access_config_count", "service_accounts", "oauth_scopes", "tags",
        "shielded_secure_boot", "shielded_vtpm", "shielded_integrity_monitoring",
        "serial_port_enabled", "project_ssh_keys_blocked", "os_login_enabled",
        "startup_script_present",
    }, "cloud_instance")
    try:
        internal_ip = ipaddress.ip_address(instance["internal_ip"])
        owner_network = ipaddress.ip_network(
            foundation.OWNER_GATE_SUBNET_CIDR, strict=True
        )
    except (TypeError, ValueError) as exc:
        raise OwnerGatePreflightError("owner_gate_cloud_instance_invalid") from None
    if (
        instance["name"] != foundation.VM_NAME
        or not _NUMERIC_ID.fullmatch(str(instance["numeric_id"]))
        or instance["status"] != "RUNNING"
        or instance["internal_ip"] != foundation.OWNER_GATE_PRIVATE_IP
        or internal_ip not in owner_network
        or instance["access_config_count"] != 0
        or instance["service_accounts"] != [
            f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
        ]
        or instance["oauth_scopes"] != list(foundation.OWNER_GATE_OAUTH_SCOPES)
        or set(instance["tags"]) != {
            foundation.IAP_NETWORK_TAG,
            foundation.OWNER_GATE_NETWORK_TAG,
        }
        or instance["shielded_secure_boot"] is not True
        or instance["shielded_vtpm"] is not True
        or instance["shielded_integrity_monitoring"] is not True
        or instance["serial_port_enabled"] is not False
        or instance["project_ssh_keys_blocked"] is not True
        or instance["os_login_enabled"] is not True
        or instance["startup_script_present"] is not False
        or not str(instance["network"]).endswith(
            f"/global/networks/{foundation.NETWORK_NAME}"
        )
        or not str(instance["subnetwork"]).endswith(
            f"/subnetworks/{foundation.OWNER_GATE_SUBNET_NAME}"
        )
    ):
        raise OwnerGatePreflightError("owner_gate_cloud_instance_invalid")

    service_account = _strict(raw["service_account"], {
        "email", "disabled", "user_managed_key_count", "project_roles",
        "effective_sensitive_permissions", "effective_permissions_probe_verified",
        "effective_permission_probe",
    }, "cloud_service_account")
    if (
        service_account["email"]
        != f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
        or service_account["disabled"] is not False
        or service_account["user_managed_key_count"] != 0
        or service_account["project_roles"]
        != sorted([
            f"projects/{foundation.PROJECT}/roles/{foundation.READ_ONLY_IAM_ROLE_ID}",
            *(
                [f"projects/{foundation.PROJECT}/roles/{foundation.MUTATION_ROLE_ID}"]
                if mutation_binding_present
                else []
            ),
        ])
        or service_account["effective_sensitive_permissions"]
        != (sorted(foundation.EXECUTION_PERMISSIONS) if mutation_binding_present else [])
        or service_account["effective_permissions_probe_verified"] is not True
        or service_account["effective_permission_probe"]
        != expected_effective_permission_probe(mutation_binding_present)
    ):
        raise OwnerGatePreflightError("owner_gate_cloud_service_account_invalid")

    iam = _strict(raw["iam"], {
        "custom_role_permissions", "mutation_binding_present", "forbidden_roles",
        "condition_expression", "read_only_role_permissions",
        "read_only_binding_present", "ancestor_read_only_permissions",
    }, "cloud_iam")
    if (
        iam["custom_role_permissions"] != sorted(foundation.EXECUTION_PERMISSIONS)
        or iam["mutation_binding_present"] is not mutation_binding_present
        or iam["forbidden_roles"] != []
        or iam["condition_expression"] != foundation._condition_expression()
        or iam["read_only_role_permissions"]
        != list(foundation.READ_ONLY_IAM_PERMISSIONS)
        or iam["read_only_binding_present"] is not True
        or iam["ancestor_read_only_permissions"]
        != list(foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS)
    ):
        raise OwnerGatePreflightError("owner_gate_cloud_iam_invalid")

    firewalls = _strict(raw["firewalls"], {
        "iap", "private_web", "public_owner_gate_rules",
        "effective_inventory_sha256", "effective_firewall_probe_verified",
    }, "cloud_firewalls")
    iap = _strict(firewalls["iap"], {
        "name", "source_ranges", "target_tags", "tcp_ports", "enabled",
    }, "cloud_iap_firewall")
    private_web = _strict(firewalls["private_web"], {
        "name", "source_service_accounts", "target_service_accounts", "tcp_ports",
        "enabled", "logging",
    }, "cloud_web_firewall")
    expected_executor_sa = (
        f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
    )
    if (
        iap != {
            "name": "allow-iap-ssh",
            "source_ranges": [foundation.IAP_SOURCE_RANGE],
            "target_tags": [foundation.IAP_NETWORK_TAG],
            "tcp_ports": [22],
            "enabled": True,
        }
        or private_web != {
            "name": "muncho-owner-gate-web-from-production",
            "source_service_accounts": [
                foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT
            ],
            "target_service_accounts": [expected_executor_sa],
            "tcp_ports": [foundation.WEB_LISTEN_PORT],
            "enabled": True,
            "logging": True,
        }
        or firewalls["public_owner_gate_rules"] != []
        or not _SHA256.fullmatch(str(firewalls["effective_inventory_sha256"]))
        or firewalls["effective_firewall_probe_verified"] is not True
    ):
        raise OwnerGatePreflightError("owner_gate_cloud_firewall_invalid")

    targets = _strict(raw["targets"], {
        "instance_name", "instance_numeric_id", "disk_name", "disk_numeric_id",
        "boot_device_name",
    }, "cloud_targets")
    if targets != {
        "instance_name": foundation.TARGET_INSTANCE,
        "instance_numeric_id": foundation.TARGET_INSTANCE_ID,
        "disk_name": foundation.TARGET_DISK,
        "disk_numeric_id": foundation.TARGET_DISK_ID,
        "boot_device_name": foundation.TARGET_BOOT_DEVICE,
    }:
        raise OwnerGatePreflightError("owner_gate_cloud_targets_invalid")
    return key_id


def _validate_host(
    raw: Mapping[str, Any],
    *,
    spec: foundation.OwnerGateSpec,
    plan_sha256: str,
    public_key: Ed25519PublicKey,
    expected_public_key_id: str,
    mutation_binding_present: bool,
) -> str:
    _strict(raw, {
        "schema", "collected_at_unix", "plan_sha256", "release", "identities",
        "sockets", "units", "filesystem_boundaries", "metadata_firewall",
        "sqlite", "migration", "webauthn", "request_intake", "executor",
        "old_v1", "caddy", "secret_material_recorded", "report_sha256",
        "attestation",
    }, "host_observation")
    _verify_seal(raw, label="host_observation")
    key_id = _verify_attestation(
        raw,
        public_key=public_key,
        expected_public_key_id=expected_public_key_id,
        label="host_observation",
    )
    if (
        raw["schema"] != HOST_OBSERVATION_SCHEMA
        or raw["plan_sha256"] != plan_sha256
        or not isinstance(raw["collected_at_unix"], int)
        or raw["collected_at_unix"] <= 0
        or raw["secret_material_recorded"] is not False
    ):
        raise OwnerGatePreflightError("owner_gate_host_observation_invalid")

    release = _strict(raw["release"], {
        "revision", "root", "uid", "gid", "mode", "immutable", "package_sha256",
        "package_inventory_sha256",
        "offline_wheelhouse_verified", "network_install_performed", "entrypoints",
        "python_version", "python_executable", "python_executable_sha256",
        "python_hash_revalidated_by_sha256sum",
    }, "host_release")
    if (
        release["revision"] != spec.release_revision
        or release["root"] != str(spec.release_root)
        or release["uid"] != 0
        or release["gid"] != 0
        or release["mode"] != "0555"
        or release["immutable"] is not True
        or not _SHA256.fullmatch(str(release["package_sha256"]))
        or release["package_inventory_sha256"]
        != spec.package_inventory_sha256
        or release["offline_wheelhouse_verified"] is not True
        or release["network_install_performed"] is not False
        or release["python_version"] != "3.11.2"
        or release["python_executable"]
        != f"{spec.release_root}/venv/bin/python"
        or release["python_executable_sha256"] != spec.interpreter_sha256
        or release["python_hash_revalidated_by_sha256sum"] is not True
        or sorted(release["entrypoints"])
        != sorted((
            "muncho-owner-gate-intake",
            "muncho-passkey-v2-web",
            "muncho-passkey-v2-authority",
            "muncho-passkey-v2-executor",
        ))
    ):
        raise OwnerGatePreflightError("owner_gate_host_release_invalid")

    identities = _strict(raw["identities"], {
        "web", "authority", "executor",
    }, "host_identities")
    expected_identities = {
        "web": ["muncho-passkey-web", WEB_UID, WEB_UID, "/usr/sbin/nologin"],
        "authority": [
            "muncho-passkey-authority", AUTHORITY_UID, AUTHORITY_UID,
            "/usr/sbin/nologin",
        ],
        "executor": [
            "muncho-storage-executor", EXECUTOR_UID, EXECUTOR_UID,
            "/usr/sbin/nologin",
        ],
    }
    for name, expected in expected_identities.items():
        identity = _strict(identities[name], {"name", "uid", "gid", "shell"}, f"host_{name}_identity")
        if [identity["name"], identity["uid"], identity["gid"], identity["shell"]] != expected:
            raise OwnerGatePreflightError("owner_gate_host_identity_invalid")
    if len({item["uid"] for item in identities.values()}) != 3:
        raise OwnerGatePreflightError("owner_gate_host_identity_invalid")

    sockets = _strict(raw["sockets"], {"web_authority", "authority_executor"}, "host_sockets")
    if sockets != {
        "web_authority": {
            "path": str(foundation.PASSKEY_AUTHORITY_SOCKET),
            "uid": AUTHORITY_UID,
            "gid": WEB_UID,
            "mode": "0660",
        },
        "authority_executor": {
            "path": str(foundation.PRIVILEGED_EXECUTOR_SOCKET),
            "uid": EXECUTOR_UID,
            "gid": AUTHORITY_UID,
            "mode": "0660",
        },
    }:
        raise OwnerGatePreflightError("owner_gate_host_socket_invalid")

    units = _strict(raw["units"], {"web", "authority", "executor"}, "host_units")
    for name, expected in EXPECTED_UNIT_PROPERTIES.items():
        unit = _strict(units[name], set(expected), f"host_{name}_unit")
        if unit != expected:
            raise OwnerGatePreflightError("owner_gate_host_unit_invalid")

    if raw["filesystem_boundaries"] != {
        "web_reads_authority_db": False,
        "web_writes_authority_db": False,
        "web_reads_mutation_journal": False,
        "authority_reads_mutation_journal": False,
        "executor_reads_authority_db": False,
        "authority_database_owner_uid": AUTHORITY_UID,
        "mutation_journal_owner_uid": EXECUTOR_UID,
    }:
        raise OwnerGatePreflightError("owner_gate_host_filesystem_boundary_invalid")
    if raw["metadata_firewall"] != {
        "web_blocked": True,
        "authority_blocked": True,
        "executor_metadata_allowed": True,
        "executor_private_google_api_allowed": True,
        "other_unprivileged_uids_blocked": True,
        "root_admin_metadata_allowed": True,
        "nft_ruleset_verified": True,
        "root_readiness_receipt_verified": True,
    }:
        raise OwnerGatePreflightError("owner_gate_host_metadata_firewall_invalid")

    sqlite = _strict(raw["sqlite"], {
        "runtime_module", "authority_schema", "executor_schema", "authority_db",
        "executor_db", "directory_mode", "database_mode", "journal_mode",
        "synchronous", "begin_immediate", "append_only_triggers", "runtime_eligible",
    }, "host_sqlite")
    if sqlite != {
        "runtime_module": "scripts.canary.passkey_v2_sqlite",
        "authority_schema": "muncho-passkey-v2-authority-sqlite.v1",
        "executor_schema": "muncho-passkey-v2-executor-sqlite.v1",
        "authority_db": AUTHORITY_DB,
        "executor_db": EXECUTOR_DB,
        "directory_mode": "0700",
        "database_mode": "0600",
        "journal_mode": "DELETE",
        "synchronous": "FULL",
        "begin_immediate": True,
        "append_only_triggers": True,
        "runtime_eligible": True,
    }:
        raise OwnerGatePreflightError("owner_gate_host_sqlite_invalid")

    migration = _strict(raw["migration"], {
        "credential_count", "enabled_owner_count", "owner_discord_user_id",
        "credential_id_sha256", "public_key_sha256", "public_key_byte_count",
        "sign_count", "backed_up", "active_request_count", "active_challenge_count",
        "active_grant_count", "totp_seed_migrated", "source_receipt_verified",
        "target_receipt_verified", "public_key_only", "user_handle_sha256",
        "credential_id_b64url_source_receipt_bound",
    }, "host_migration")
    if migration != {
        "credential_count": 1,
        "enabled_owner_count": 1,
        "owner_discord_user_id": "1279454038731264061",
        "credential_id_sha256": EXPECTED_CREDENTIAL_ID_SHA256,
        "public_key_sha256": EXPECTED_PUBLIC_KEY_SHA256,
        "user_handle_sha256": EXPECTED_USER_HANDLE_SHA256,
        "credential_id_b64url_source_receipt_bound": True,
        "public_key_byte_count": 77,
        "sign_count": 0,
        "backed_up": True,
        "active_request_count": 0,
        "active_challenge_count": 0,
        "active_grant_count": 0,
        "totp_seed_migrated": False,
        "source_receipt_verified": True,
        "target_receipt_verified": True,
        "public_key_only": True,
    }:
        raise OwnerGatePreflightError("owner_gate_host_migration_invalid")

    if raw["webauthn"] != {
        "rp_id": "lomliev.com",
        "origin": "https://auth.lomliev.com",
        "user_verification_required": True,
        "forged_assertion_blocked": True,
        "wrong_challenge_blocked": True,
        "wrong_origin_blocked": True,
        "wrong_rp_blocked": True,
        "no_uv_blocked": True,
        "replay_blocked": True,
        "concurrent_exactly_one_authorized": True,
        "web_raw_grant_api_absent": True,
    }:
        raise OwnerGatePreflightError("owner_gate_host_webauthn_invalid")
    if raw["request_intake"] != {
        "public_web_can_author_envelope": False,
        "iap_fixed_command_only": True,
        "signed_release_verified": True,
        "signed_source_preflight_verified": True,
        "signed_host_identity_verified": True,
        "signed_external_iam_verified": True,
        "release_plan_transaction_evidence_bound": True,
    }:
        raise OwnerGatePreflightError("owner_gate_host_request_intake_invalid")
    executor = raw["executor"]
    if (
        not isinstance(executor, Mapping)
        or _SHA256.fullmatch(
            str(executor.get("receipt_public_key_sha256", ""))
        ) is None
    ):
        raise OwnerGatePreflightError("owner_gate_host_executor_invalid")
    if executor != {
        "uid": EXECUTOR_UID,
        "mutation_iam_binding_present": mutation_binding_present,
        "activation_seal_present": False,
        "activation_seal_required_for_mutation_only": True,
        "activation_seal_exact_contract_verified": True,
        "activation_seal_expected_uid": 0,
        "activation_seal_expected_gid": EXECUTOR_UID,
        "activation_seal_expected_mode": "0440",
        "authorization_receipt_signature_self_verified": True,
        "receipt_action_binding_self_verified": True,
        "local_gcloud_present": False,
        "generic_shell_fallback_present": False,
        "compute_api_connectivity_verified": mutation_binding_present,
        "numeric_targets_reverified": mutation_binding_present,
        "target_instance_id": foundation.TARGET_INSTANCE_ID,
        "target_disk_id": foundation.TARGET_DISK_ID,
        "receipt_public_key_sha256": executor["receipt_public_key_sha256"],
        "receipt_public_key_owner": "root:root",
        "receipt_public_key_mode": "0444",
    }:
        raise OwnerGatePreflightError("owner_gate_host_executor_invalid")
    if raw["old_v1"] != {
        "unit": "muncho-passkey-stepup.service",
        "active": False,
        "masked": True,
        "trusted_for_v2": False,
    }:
        raise OwnerGatePreflightError("owner_gate_host_old_v1_invalid")
    if raw["caddy"] != {
        "public_origin": "https://auth.lomliev.com",
        "still_on_current_host": True,
        "private_v2_upstream_active": False,
        "config_validated": True,
        "rollback_mode": "pre_migration_v1_only",
    }:
        raise OwnerGatePreflightError("owner_gate_host_caddy_invalid")
    return key_id


def build_preflight_report(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    cloud_observation: Mapping[str, Any],
    host_observation: Mapping[str, Any],
    cloud_collector_public_key: Ed25519PublicKey,
    host_collector_public_key: Ed25519PublicKey,
    now_unix: int,
) -> Mapping[str, Any]:
    if (
        not isinstance(now_unix, int)
        or isinstance(now_unix, bool)
        or now_unix <= 0
    ):
        raise OwnerGatePreflightError("owner_gate_preflight_time_invalid")
    expected_cloud_key_id = plan.spec.cloud_collector_public_key_id
    expected_host_key_id = plan.spec.host_collector_public_key_id
    if (
        not isinstance(expected_cloud_key_id, str)
        or _SHA256.fullmatch(expected_cloud_key_id) is None
        or not isinstance(expected_host_key_id, str)
        or _SHA256.fullmatch(expected_host_key_id) is None
    ):
        raise OwnerGatePreflightError(
            "owner_gate_preflight_collector_key_invalid"
        )
    cloud_key_id = _validate_cloud(
        cloud_observation,
        plan_sha256=plan.sha256,
        public_key=cloud_collector_public_key,
        expected_public_key_id=expected_cloud_key_id,
        mutation_binding_present=False,
    )
    host_key_id = _validate_host(
        host_observation,
        spec=plan.spec,
        plan_sha256=plan.sha256,
        public_key=host_collector_public_key,
        expected_public_key_id=expected_host_key_id,
        mutation_binding_present=False,
    )
    for observed in (cloud_observation, host_observation):
        collected = observed["collected_at_unix"]
        if (
            collected > now_unix
            or now_unix - collected > foundation.PREFLIGHT_MAX_AGE_SECONDS
        ):
            raise OwnerGatePreflightError("owner_gate_preflight_stale")
    unsigned = {
        "schema": PREFLIGHT_SCHEMA,
        "plan_sha256": plan.sha256,
        "release_revision": plan.spec.release_revision,
        "cloud_observation_sha256": cloud_observation["report_sha256"],
        "host_observation_sha256": host_observation["report_sha256"],
        "cloud_collector_public_key_id": cloud_key_id,
        "host_collector_public_key_id": host_key_id,
        "receipt_public_key_sha256": host_observation["executor"][
            "receipt_public_key_sha256"
        ],
        "private_vm_ready": True,
        "offline_package_ready": True,
        "split_uid_boundary_ready": True,
        "public_key_migration_ready": True,
        "webauthn_verifier_ready": True,
        "atomic_single_use_ready": True,
        "request_intake_ready": True,
        "metadata_uid_firewall_ready": True,
        "old_v1_masked": True,
        "caddy_cutover_performed": False,
        "mutation_iam_binding_present": False,
        "executor_activation_seal_present": False,
        "inert_noop_security_smoke_ready": True,
        "mutation_performed": False,
        "secret_material_recorded": False,
        "observed_at_unix": now_unix,
    }
    return {**unsigned, "report_sha256": foundation.sha256_json(unsigned)}


def build_post_iam_preflight_report(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    cloud_observation: Mapping[str, Any],
    host_observation: Mapping[str, Any],
    cloud_collector_public_key: Ed25519PublicKey,
    host_collector_public_key: Ed25519PublicKey,
    now_unix: int,
) -> Mapping[str, Any]:
    if not isinstance(now_unix, int) or isinstance(now_unix, bool) or now_unix <= 0:
        raise OwnerGatePreflightError("owner_gate_preflight_time_invalid")
    expected_cloud_key_id = plan.spec.cloud_collector_public_key_id
    expected_host_key_id = plan.spec.host_collector_public_key_id
    if (
        not isinstance(expected_cloud_key_id, str)
        or _SHA256.fullmatch(expected_cloud_key_id) is None
        or not isinstance(expected_host_key_id, str)
        or _SHA256.fullmatch(expected_host_key_id) is None
    ):
        raise OwnerGatePreflightError(
            "owner_gate_preflight_collector_key_invalid"
        )
    cloud_key_id = _validate_cloud(
        cloud_observation,
        plan_sha256=plan.sha256,
        public_key=cloud_collector_public_key,
        expected_public_key_id=expected_cloud_key_id,
        mutation_binding_present=True,
    )
    host_key_id = _validate_host(
        host_observation,
        spec=plan.spec,
        plan_sha256=plan.sha256,
        public_key=host_collector_public_key,
        expected_public_key_id=expected_host_key_id,
        mutation_binding_present=True,
    )
    for observed in (cloud_observation, host_observation):
        collected = observed["collected_at_unix"]
        if collected > now_unix or now_unix - collected > foundation.PREFLIGHT_MAX_AGE_SECONDS:
            raise OwnerGatePreflightError("owner_gate_preflight_stale")
    unsigned = {
        "schema": POST_IAM_PREFLIGHT_SCHEMA,
        "plan_sha256": plan.sha256,
        "release_revision": plan.spec.release_revision,
        "cloud_observation_sha256": cloud_observation["report_sha256"],
        "host_observation_sha256": host_observation["report_sha256"],
        "cloud_collector_public_key_id": cloud_key_id,
        "host_collector_public_key_id": host_key_id,
        "receipt_public_key_sha256": host_observation["executor"][
            "receipt_public_key_sha256"
        ],
        "effective_permissions_exact_for_fixed_probe_set": True,
        "effective_permission_probe_sha256": foundation.sha256_json(
            cloud_observation["service_account"]["effective_permission_probe"]
        ),
        "operation_permission_absent": True,
        "compute_api_connectivity_verified": True,
        "target_instance_numeric_id": foundation.TARGET_INSTANCE_ID,
        "target_disk_numeric_id": foundation.TARGET_DISK_ID,
        "executor_activation_seal_present": False,
        "mutation_attempted": False,
        "topology_iam_readiness_seal_can_be_installed": True,
        "observed_at_unix": now_unix,
    }
    return {**unsigned, "report_sha256": foundation.sha256_json(unsigned)}
