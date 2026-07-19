#!/usr/bin/env python3
"""Owner-approved, reversible OS Login metadata migration for production.

This gate exists before :class:`ProductionCutoverTransport`: that transport
correctly refuses IAP/SSH while the production instance still carries legacy
``ssh-keys`` metadata or does not have ``enable-oslogin=TRUE``.  The gate uses
only fixed GCE metadata operations, verifies the exact instance, OS Login
profile/key, and effective IAM decisions first, and restores the exact prior
two-key metadata state if the immediate IAP/OS Login access probe fails.

No remote command is caller supplied.  The only remote access probe is
``/usr/bin/true`` through the already-pinned production IAP transport.  The
owner signing key is consumed locally and is never passed to this boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import full_canary_owner_launcher as canary_transport
from scripts.canary import production_cutover_owner_launcher as cutover_owner


PREFLIGHT_SCHEMA = "muncho-production-os-login-metadata-preflight.v1"
PLAN_SCHEMA = "muncho-production-os-login-metadata-plan.v1"
APPROVAL_SCHEMA = "muncho-production-os-login-metadata-approval.v1"
INTENT_SCHEMA = "muncho-production-os-login-metadata-intent.v1"
ACCESS_SCHEMA = "muncho-production-os-login-access.v1"
RECEIPT_SCHEMA = "muncho-production-os-login-metadata-receipt.v1"
AUTHORITY_SCHEMA = "muncho-production-os-login-metadata-authority.v1"
MAX_AGE_SECONDS = 900
MAX_METADATA_BYTES = 512 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT = re.compile(r"^[A-Za-z0-9+/=_-]{4,256}$")
_METADATA_KEY = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_TARGET = {
    "project": cutover_owner.PRODUCTION_PROJECT,
    "project_number": cutover_owner.PRODUCTION_PROJECT_NUMBER,
    "zone": cutover_owner.PRODUCTION_ZONE,
    "vm": cutover_owner.PRODUCTION_VM_NAME,
    "instance_id": cutover_owner.PRODUCTION_VM_INSTANCE_ID,
    "os_login_profile_id": cutover_owner.PRODUCTION_OS_LOGIN_PROFILE_ID,
    "os_login_username": cutover_owner.PRODUCTION_OS_LOGIN_USERNAME,
}
_IAM_PERMISSIONS = (
    "compute.instances.get",
    "compute.instances.osAdminLogin",
    "compute.instances.setMetadata",
    "iap.tunnelInstances.accessViaIAP",
)
_PREFLIGHT_FIELDS = frozenset({
    "schema",
    "target",
    "owner_subject_sha256",
    "owner_account_sha256",
    "state",
    "instance_metadata_fingerprint",
    "instance_metadata_keys",
    "enable_oslogin",
    "ssh_keys_present",
    "public_ssh_keys_sha256",
    "project_metadata_fingerprint",
    "project_metadata_keys",
    "os_login_profile_sha256",
    "os_login_public_key_fingerprint",
    "iam_permissions",
    "state_identity_sha256",
    "observed_at_unix",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})
_PLAN_FIELDS = frozenset({
    "schema",
    "target",
    "preflight_receipt_sha256",
    "preflight_state_identity_sha256",
    "prior_instance_metadata_fingerprint",
    "prior_enable_oslogin",
    "prior_public_ssh_keys_sha256",
    "operations",
    "owner_subject_sha256",
    "owner_public_key_ed25519_hex",
    "owner_key_id",
    "issued_at_unix",
    "expires_at_unix",
    "secret_material_recorded",
    "secret_digest_recorded",
    "plan_sha256",
})
_APPROVAL_FIELDS = frozenset({
    "schema",
    "purpose",
    "plan_sha256",
    "owner_subject_sha256",
    "owner_public_key_ed25519_hex",
    "owner_key_id",
    "nonce_sha256",
    "issued_at_unix",
    "expires_at_unix",
    "approved",
    "signature_ed25519_hex",
    "approval_sha256",
})
_INTENT_FIELDS = frozenset({
    "schema",
    "target",
    "plan_sha256",
    "approval_sha256",
    "prior_preflight",
    "prior_ssh_keys",
    "prior_enable_oslogin",
    "expected_intermediate_instance_metadata_keys",
    "expected_ready_instance_metadata_keys",
    "created_at_unix",
    "private_key_staged",
    "secret_material_recorded",
    "secret_digest_recorded",
    "intent_sha256",
})
_ACCESS_FIELDS = frozenset({
    "schema",
    "target",
    "owner_subject_sha256",
    "owner_account_sha256",
    "fixed_remote_command",
    "authorization_snapshot_sha256",
    "iap_os_login_succeeded",
    "observed_at_unix",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})
_RECEIPT_FIELDS = frozenset({
    "schema",
    "target",
    "plan_sha256",
    "approval_sha256",
    "intent_sha256",
    "prior_state_identity_sha256",
    "ready_state_identity_sha256",
    "access_receipt_sha256",
    "enable_oslogin",
    "instance_ssh_keys_present",
    "iap_os_login_succeeded",
    "rollback_used",
    "private_key_staged",
    "secret_material_recorded",
    "secret_digest_recorded",
    "completed_at_unix",
    "receipt_sha256",
})
_AUTHORITY_FIELDS = frozenset({
    "schema",
    "release_revision",
    "preflight",
    "plan",
    "approval",
    "private_key_staged",
    "secret_material_recorded",
    "secret_digest_recorded",
    "authority_sha256",
})


class OsLoginMetadataMigrationError(RuntimeError):
    """Stable, secret-free OS Login migration failure."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OsLoginMetadataMigrationError("os_login_metadata_json_invalid") from exc


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hashed(unsigned: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**unsigned, field: _sha(_canonical(unsigned))}


def _public_hex(private_key: Ed25519PrivateKey) -> str:
    if not isinstance(private_key, Ed25519PrivateKey):
        raise OsLoginMetadataMigrationError("os_login_owner_key_invalid")
    return (
        private_key
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )


def _metadata(
    value: Any,
    *,
    code: str,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not isinstance(value, Mapping) or set(value) != {"fingerprint", "items"}:
        raise OsLoginMetadataMigrationError(code)
    fingerprint = value.get("fingerprint")
    items = value.get("items")
    if (
        not isinstance(fingerprint, str)
        or _FINGERPRINT.fullmatch(fingerprint) is None
        or not isinstance(items, list)
    ):
        raise OsLoginMetadataMigrationError(code)
    result: dict[str, str] = {}
    total = 0
    for item in items:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"key", "value"}
            or not isinstance(item.get("key"), str)
            or _METADATA_KEY.fullmatch(item["key"]) is None
            or not isinstance(item.get("value"), str)
            or item["key"] in result
        ):
            raise OsLoginMetadataMigrationError(code)
        encoded = item["value"].encode("utf-8", errors="strict")
        total += len(item["key"].encode("ascii")) + len(encoded)
        if total > MAX_METADATA_BYTES:
            raise OsLoginMetadataMigrationError(code)
        result[item["key"]] = item["value"]
    return fingerprint, tuple(sorted(result.items()))


def _state_name(metadata: Mapping[str, str]) -> str:
    enabled = metadata.get("enable-oslogin")
    legacy = "ssh-keys" in metadata
    if legacy and enabled in {None, "FALSE"}:
        return "legacy_instance_ssh_keys"
    if legacy and enabled == "TRUE":
        return "os_login_enabled_with_legacy_keys"
    if not legacy and enabled == "TRUE":
        return "os_login_ready"
    raise OsLoginMetadataMigrationError("os_login_metadata_state_invalid")


@dataclass(frozen=True)
class ObservedState:
    """Public receipt plus private-in-process metadata used for exact rollback."""

    receipt: Mapping[str, Any]
    instance_metadata: tuple[tuple[str, str], ...]
    project_metadata: tuple[tuple[str, str], ...]


class MetadataMigrationBoundary(Protocol):
    def observe(self) -> ObservedState: ...

    def set_enable_oslogin_true(self) -> None: ...

    def remove_instance_ssh_keys(self) -> None: ...

    def restore_instance_ssh_keys(self, value: str) -> None: ...

    def restore_enable_oslogin(self, value: str | None) -> None: ...

    def probe_iap_os_login(self) -> Mapping[str, Any]: ...


class ProductionOsLoginMetadataTransport:
    """Fixed control-plane mutations around the pinned production transport."""

    def __init__(
        self,
        transport: cutover_owner.ProductionCutoverTransport,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(transport, cutover_owner.ProductionCutoverTransport):
            raise OsLoginMetadataMigrationError("os_login_metadata_transport_invalid")
        self._transport = transport
        self._clock = clock

    def _owner(self) -> tuple[str, str]:
        identity = self._transport._owner_identity
        account = identity.account_for_read_only_preflight()
        subject = identity.owner_subject_sha256
        if (
            not isinstance(account, str)
            or canary_transport.GcloudOwnerAccessToken._ACCOUNT.fullmatch(account)
            is None
            or not isinstance(subject, str)
            or _SHA256.fullmatch(subject) is None
        ):
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_owner_identity_invalid"
            )
        identity.require_stable()
        return account, subject

    def _instance(self, account: str) -> Mapping[str, Any]:
        return self._transport._run_read_only_gcloud_json((
            "compute",
            "instances",
            "describe",
            cutover_owner.PRODUCTION_VM_NAME,
            f"--project={cutover_owner.PRODUCTION_PROJECT}",
            f"--zone={cutover_owner.PRODUCTION_ZONE}",
            f"--account={account}",
            "--format=json(id,name,zone,metadata.fingerprint,metadata.items)",
            "--quiet",
        ))

    def _project(self, account: str) -> Mapping[str, Any]:
        return self._transport._run_read_only_gcloud_json((
            "compute",
            "project-info",
            "describe",
            f"--project={cutover_owner.PRODUCTION_PROJECT}",
            f"--account={account}",
            "--format=json(name,commonInstanceMetadata.fingerprint,"
            "commonInstanceMetadata.items)",
            "--quiet",
        ))

    def _profile(self, account: str) -> Mapping[str, Any]:
        return self._transport._run_read_only_gcloud_json((
            "compute",
            "os-login",
            "describe-profile",
            f"--project={cutover_owner.PRODUCTION_PROJECT}",
            f"--account={account}",
            "--format=json",
            "--quiet",
        ))

    def _iam(self, account: str, permission: str) -> Mapping[str, Any]:
        if permission not in _IAM_PERMISSIONS:
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_iam_permission_invalid"
            )
        project_resource = (
            "//cloudresourcemanager.googleapis.com/projects/"
            f"{cutover_owner.PRODUCTION_PROJECT}"
        )
        if permission.startswith("iap."):
            resource_name = (
                "//iap.googleapis.com/projects/"
                f"{cutover_owner.PRODUCTION_PROJECT_NUMBER}/iap_tunnel/zones/"
                f"{cutover_owner.PRODUCTION_ZONE}/instances/"
                f"{cutover_owner.PRODUCTION_VM_INSTANCE_ID}"
            )
            service = "iap.googleapis.com"
            resource_type = "iap.googleapis.com/TunnelInstance"
        else:
            resource_name = (
                "//compute.googleapis.com/projects/"
                f"{cutover_owner.PRODUCTION_PROJECT}/zones/"
                f"{cutover_owner.PRODUCTION_ZONE}/instances/"
                f"{cutover_owner.PRODUCTION_VM_NAME}"
            )
            service = "compute.googleapis.com"
            resource_type = "compute.googleapis.com/Instance"
        return self._transport._run_read_only_gcloud_json((
            "policy-intelligence",
            "troubleshoot-policy",
            "iam",
            project_resource,
            f"--principal-email={account}",
            f"--permission={permission}",
            f"--resource-name={resource_name}",
            f"--resource-service={service}",
            f"--resource-type={resource_type}",
            f"--project={cutover_owner.PRODUCTION_PROJECT}",
            f"--account={account}",
            "--format=json(overallAccessState)",
            "--quiet",
        ))

    def observe(self) -> ObservedState:
        account, subject = self._owner()
        instance = self._instance(account)
        if (
            set(instance) != {"id", "name", "zone", "metadata"}
            or instance.get("id") != cutover_owner.PRODUCTION_VM_INSTANCE_ID
            or instance.get("name") != cutover_owner.PRODUCTION_VM_NAME
            or instance.get("zone")
            != (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{cutover_owner.PRODUCTION_PROJECT}/zones/"
                f"{cutover_owner.PRODUCTION_ZONE}"
            )
        ):
            raise OsLoginMetadataMigrationError("os_login_metadata_instance_invalid")
        instance_fingerprint, instance_items = _metadata(
            instance["metadata"], code="os_login_metadata_instance_invalid"
        )
        instance_values = dict(instance_items)
        state = _state_name(instance_values)

        project = self._project(account)
        if (
            set(project) != {"name", "commonInstanceMetadata"}
            or project.get("name") != cutover_owner.PRODUCTION_PROJECT
        ):
            raise OsLoginMetadataMigrationError("os_login_metadata_project_invalid")
        project_fingerprint, project_items = _metadata(
            project["commonInstanceMetadata"],
            code="os_login_metadata_project_invalid",
        )

        profile = self._profile(account)
        posix_accounts = profile.get("posixAccounts")
        ssh_public_keys = profile.get("sshPublicKeys")
        if (
            set(profile) != {"name", "posixAccounts", "sshPublicKeys"}
            or profile.get("name") != cutover_owner.PRODUCTION_OS_LOGIN_PROFILE_ID
            or not isinstance(posix_accounts, list)
            or not isinstance(ssh_public_keys, Mapping)
            or any(not isinstance(item, Mapping) for item in posix_accounts)
            or any(
                not isinstance(key, str) or not isinstance(item, Mapping)
                for key, item in ssh_public_keys.items()
            )
        ):
            raise OsLoginMetadataMigrationError("os_login_metadata_profile_invalid")
        matching_accounts = [
            item
            for item in posix_accounts
            if item.get("username") == cutover_owner.PRODUCTION_OS_LOGIN_USERNAME
            and item.get("primary") is True
            and item.get("operatingSystemType") == "LINUX"
            and item.get("homeDirectory")
            == f"/home/{cutover_owner.PRODUCTION_OS_LOGIN_USERNAME}"
        ]
        public_key = self._transport._known_hosts.public_key_line()
        matching_keys = [
            fingerprint
            for fingerprint, item in ssh_public_keys.items()
            if _SHA256.fullmatch(fingerprint) is not None
            and item.get("fingerprint") == fingerprint
            and item.get("key") == public_key
        ]
        if len(matching_accounts) != 1 or len(matching_keys) != 1:
            raise OsLoginMetadataMigrationError("os_login_metadata_profile_invalid")
        normalized_profile = {
            "name": profile["name"],
            "posixAccounts": sorted(
                (dict(item) for item in posix_accounts), key=_canonical
            ),
            "sshPublicKeys": [
                {"fingerprint": fingerprint, "value": dict(item)}
                for fingerprint, item in sorted(ssh_public_keys.items())
            ],
        }
        iam: dict[str, str] = {}
        for permission in _IAM_PERMISSIONS:
            evidence = self._iam(account, permission)
            if (
                set(evidence) != {"overallAccessState"}
                or evidence.get("overallAccessState") != "CAN_ACCESS"
            ):
                raise OsLoginMetadataMigrationError("os_login_metadata_iam_invalid")
            iam[permission] = "GRANTED"
        if (
            self._instance(account) != instance
            or self._project(account) != project
            or self._profile(account) != profile
        ):
            raise OsLoginMetadataMigrationError("os_login_metadata_preflight_changed")
        ssh_keys = instance_values.get("ssh-keys")
        identity = {
            "target": copy.deepcopy(_TARGET),
            "owner_subject_sha256": subject,
            "owner_account_sha256": _sha(account.encode("utf-8")),
            "state": state,
            "instance_metadata_fingerprint": instance_fingerprint,
            "instance_metadata_keys": sorted(instance_values),
            "enable_oslogin": instance_values.get("enable-oslogin"),
            "ssh_keys_present": ssh_keys is not None,
            "public_ssh_keys_sha256": (
                None if ssh_keys is None else _sha(ssh_keys.encode("utf-8"))
            ),
            "project_metadata_fingerprint": project_fingerprint,
            "project_metadata_keys": sorted(dict(project_items)),
            "os_login_profile_sha256": _sha(_canonical(normalized_profile)),
            "os_login_public_key_fingerprint": matching_keys[0],
            "iam_permissions": iam,
        }
        unsigned = {
            "schema": PREFLIGHT_SCHEMA,
            **identity,
            "state_identity_sha256": _sha(_canonical(identity)),
            "observed_at_unix": int(self._clock()),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        receipt = validate_preflight_receipt(
            _hashed(unsigned, "receipt_sha256"), now_unix=int(self._clock())
        )
        self._transport._owner_identity.require_stable()
        return ObservedState(receipt, instance_items, project_items)

    def _mutate(self, arguments: tuple[str, ...]) -> None:
        account, _subject = self._owner()
        command_prefix = self._transport._gcloud_executable.trusted_command_prefix()
        environment = canary_transport._owner_gcloud_environment(
            self._transport._gcloud_configuration,
            command_prefix[0],
        )
        try:
            completed = self._transport._preflight_runner(
                (*command_prefix, *arguments, f"--account={account}", "--quiet"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=dict(environment),
                shell=False,
                timeout=120.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self._transport._postflight()
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_mutation_unavailable"
            ) from None
        self._transport._postflight()
        self._transport._owner_identity.require_stable()
        if (
            completed.returncode != 0
            or not isinstance(completed.stdout, bytes)
            or len(completed.stdout) > 64 * 1024
        ):
            raise OsLoginMetadataMigrationError("os_login_metadata_mutation_failed")

    @staticmethod
    def _instance_args(command: str) -> tuple[str, ...]:
        if command not in {"add-metadata", "remove-metadata"}:
            raise OsLoginMetadataMigrationError("os_login_metadata_command_invalid")
        return (
            "compute",
            "instances",
            command,
            cutover_owner.PRODUCTION_VM_NAME,
            f"--project={cutover_owner.PRODUCTION_PROJECT}",
            f"--zone={cutover_owner.PRODUCTION_ZONE}",
        )

    def set_enable_oslogin_true(self) -> None:
        self._mutate((
            *self._instance_args("add-metadata"),
            "--metadata=enable-oslogin=TRUE",
        ))

    def remove_instance_ssh_keys(self) -> None:
        self._mutate((
            *self._instance_args("remove-metadata"),
            "--keys=ssh-keys",
        ))

    def restore_instance_ssh_keys(self, value: str) -> None:
        if (
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8", errors="strict")) > MAX_METADATA_BYTES
        ):
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_rollback_value_invalid"
            )
        with tempfile.TemporaryDirectory(
            prefix="muncho-os-login-rollback-"
        ) as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            path = directory / "ssh-keys"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            try:
                raw = value.encode("utf-8", errors="strict")
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short ssh key metadata rollback write")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._mutate((
                *self._instance_args("add-metadata"),
                f"--metadata-from-file=ssh-keys={path}",
            ))

    def restore_enable_oslogin(self, value: str | None) -> None:
        if value is None:
            self._mutate((
                *self._instance_args("remove-metadata"),
                "--keys=enable-oslogin",
            ))
            return
        if value != "FALSE":
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_rollback_value_invalid"
            )
        self._mutate((
            *self._instance_args("add-metadata"),
            "--metadata=enable-oslogin=FALSE",
        ))

    def probe_iap_os_login(self) -> Mapping[str, Any]:
        account, subject = self._owner()
        completed = self._transport._run_remote(
            ("/usr/bin/true",),
            account=account,
            timeout_seconds=60.0,
            maximum_output_bytes=1,
        )
        if completed.returncode != 0 or completed.stdout != b"":
            raise OsLoginMetadataMigrationError("os_login_metadata_access_probe_failed")
        authorization = self._transport._authorization_snapshot(account)
        unsigned = {
            "schema": ACCESS_SCHEMA,
            "target": copy.deepcopy(_TARGET),
            "owner_subject_sha256": subject,
            "owner_account_sha256": _sha(account.encode("utf-8")),
            "fixed_remote_command": "/usr/bin/true",
            "authorization_snapshot_sha256": _sha(_canonical(authorization)),
            "iap_os_login_succeeded": True,
            "observed_at_unix": int(self._clock()),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        return _hashed(unsigned, "receipt_sha256")


def validate_preflight_receipt(
    value: Any,
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    current = int(time.time()) if now_unix is None else now_unix
    if not isinstance(value, Mapping) or set(value) != _PREFLIGHT_FIELDS:
        raise OsLoginMetadataMigrationError(
            "os_login_metadata_preflight_fields_invalid"
        )
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    identity = {
        name: value[name]
        for name in (
            "target",
            "owner_subject_sha256",
            "owner_account_sha256",
            "state",
            "instance_metadata_fingerprint",
            "instance_metadata_keys",
            "enable_oslogin",
            "ssh_keys_present",
            "public_ssh_keys_sha256",
            "project_metadata_fingerprint",
            "project_metadata_keys",
            "os_login_profile_sha256",
            "os_login_public_key_fingerprint",
            "iam_permissions",
        )
    }
    instance_keys = value.get("instance_metadata_keys")
    project_keys = value.get("project_metadata_keys")
    if (
        value.get("schema") != PREFLIGHT_SCHEMA
        or value.get("target") != _TARGET
        or _SHA256.fullmatch(str(value.get("owner_subject_sha256"))) is None
        or _SHA256.fullmatch(str(value.get("owner_account_sha256"))) is None
        or value.get("state")
        not in {
            "legacy_instance_ssh_keys",
            "os_login_enabled_with_legacy_keys",
            "os_login_ready",
        }
        or _FINGERPRINT.fullmatch(str(value.get("instance_metadata_fingerprint")))
        is None
        or not isinstance(instance_keys, list)
        or any(
            not isinstance(item, str) or _METADATA_KEY.fullmatch(item) is None
            for item in instance_keys
        )
        or instance_keys != sorted(set(instance_keys))
        or value.get("enable_oslogin") not in {None, "FALSE", "TRUE"}
        or type(value.get("ssh_keys_present")) is not bool
        or ("ssh-keys" in instance_keys) is not value["ssh_keys_present"]
        or ("enable-oslogin" in instance_keys)
        is not (value["enable_oslogin"] is not None)
        or (
            value["ssh_keys_present"]
            and _SHA256.fullmatch(str(value.get("public_ssh_keys_sha256"))) is None
        )
        or (
            not value["ssh_keys_present"]
            and value.get("public_ssh_keys_sha256") is not None
        )
        or _FINGERPRINT.fullmatch(str(value.get("project_metadata_fingerprint")))
        is None
        or not isinstance(project_keys, list)
        or any(
            not isinstance(item, str) or _METADATA_KEY.fullmatch(item) is None
            for item in project_keys
        )
        or project_keys != sorted(set(project_keys))
        or _SHA256.fullmatch(str(value.get("os_login_profile_sha256"))) is None
        or _SHA256.fullmatch(str(value.get("os_login_public_key_fingerprint"))) is None
        or value.get("iam_permissions")
        != {permission: "GRANTED" for permission in _IAM_PERMISSIONS}
        or value.get("state_identity_sha256") != _sha(_canonical(identity))
        or type(value.get("observed_at_unix")) is not int
        or not current - MAX_AGE_SECONDS <= value["observed_at_unix"] <= current + 30
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256") != _sha(_canonical(unsigned))
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_preflight_invalid")
    metadata = {
        "enable-oslogin": value["enable_oslogin"],
        "ssh-keys": value["ssh_keys_present"],
    }
    if metadata["ssh-keys"] and metadata["enable-oslogin"] in {None, "FALSE"}:
        expected_state = "legacy_instance_ssh_keys"
    elif metadata == {"enable-oslogin": "TRUE", "ssh-keys": True}:
        expected_state = "os_login_enabled_with_legacy_keys"
    elif metadata == {"enable-oslogin": "TRUE", "ssh-keys": False}:
        expected_state = "os_login_ready"
    else:
        raise OsLoginMetadataMigrationError("os_login_metadata_preflight_invalid")
    if value["state"] != expected_state:
        raise OsLoginMetadataMigrationError("os_login_metadata_preflight_invalid")
    return copy.deepcopy(dict(value))


def collect_migration_preflight(
    boundary: MetadataMigrationBoundary,
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    current = int(time.time()) if now_unix is None else now_unix
    receipt = validate_preflight_receipt(boundary.observe().receipt, now_unix=current)
    if receipt["state"] != "legacy_instance_ssh_keys":
        raise OsLoginMetadataMigrationError("os_login_metadata_migration_not_required")
    return receipt


def build_migration_plan(
    *,
    preflight_receipt: Mapping[str, Any],
    owner_subject_sha256: str,
    private_key: Ed25519PrivateKey,
    now_unix: int | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    current = int(time.time()) if now_unix is None else now_unix
    preflight = validate_preflight_receipt(preflight_receipt, now_unix=current)
    if (
        preflight["state"] != "legacy_instance_ssh_keys"
        or preflight["owner_subject_sha256"] != owner_subject_sha256
        or _SHA256.fullmatch(owner_subject_sha256 or "") is None
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_plan_invalid")
    public = _public_hex(private_key)
    unsigned = {
        "schema": PLAN_SCHEMA,
        "target": copy.deepcopy(_TARGET),
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "preflight_state_identity_sha256": preflight["state_identity_sha256"],
        "prior_instance_metadata_fingerprint": preflight[
            "instance_metadata_fingerprint"
        ],
        "prior_enable_oslogin": preflight["enable_oslogin"],
        "prior_public_ssh_keys_sha256": preflight["public_ssh_keys_sha256"],
        "operations": [
            {"action": "set", "key": "enable-oslogin", "value": "TRUE"},
            {"action": "remove", "key": "ssh-keys"},
        ],
        "owner_subject_sha256": owner_subject_sha256,
        "owner_public_key_ed25519_hex": public,
        "owner_key_id": _sha(bytes.fromhex(public)),
        "issued_at_unix": current,
        "expires_at_unix": current + 900,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    plan = validate_migration_plan(_hashed(unsigned, "plan_sha256"), now_unix=current)
    approval_unsigned = {
        "schema": APPROVAL_SCHEMA,
        "purpose": "production_os_login_metadata_apply",
        "plan_sha256": plan["plan_sha256"],
        "owner_subject_sha256": owner_subject_sha256,
        "owner_public_key_ed25519_hex": public,
        "owner_key_id": plan["owner_key_id"],
        "nonce_sha256": _sha(os.urandom(32)),
        "issued_at_unix": current,
        "expires_at_unix": current + 900,
        "approved": True,
    }
    signature = private_key.sign(_canonical(approval_unsigned)).hex()
    approval = validate_migration_approval(
        _hashed(
            {**approval_unsigned, "signature_ed25519_hex": signature},
            "approval_sha256",
        ),
        plan=plan,
        now_unix=current,
    )
    return plan, approval


def validate_migration_plan(
    value: Any,
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    current = int(time.time()) if now_unix is None else now_unix
    if not isinstance(value, Mapping) or set(value) != _PLAN_FIELDS:
        raise OsLoginMetadataMigrationError("os_login_metadata_plan_invalid")
    unsigned = {name: item for name, item in value.items() if name != "plan_sha256"}
    if (
        value.get("schema") != PLAN_SCHEMA
        or value.get("target") != _TARGET
        or _SHA256.fullmatch(str(value.get("preflight_receipt_sha256"))) is None
        or _SHA256.fullmatch(str(value.get("preflight_state_identity_sha256"))) is None
        or _FINGERPRINT.fullmatch(str(value.get("prior_instance_metadata_fingerprint")))
        is None
        or value.get("prior_enable_oslogin") not in {None, "FALSE"}
        or _SHA256.fullmatch(str(value.get("prior_public_ssh_keys_sha256"))) is None
        or value.get("operations")
        != [
            {"action": "set", "key": "enable-oslogin", "value": "TRUE"},
            {"action": "remove", "key": "ssh-keys"},
        ]
        or _SHA256.fullmatch(str(value.get("owner_subject_sha256"))) is None
        or not isinstance(value.get("owner_public_key_ed25519_hex"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["owner_public_key_ed25519_hex"]) is None
        or value.get("owner_key_id")
        != _sha(bytes.fromhex(value["owner_public_key_ed25519_hex"]))
        or type(value.get("issued_at_unix")) is not int
        or type(value.get("expires_at_unix")) is not int
        or not value["issued_at_unix"] <= current < value["expires_at_unix"]
        or not 1 <= value["expires_at_unix"] - value["issued_at_unix"] <= 900
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("plan_sha256") != _sha(_canonical(unsigned))
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_plan_invalid")
    return copy.deepcopy(dict(value))


def validate_migration_approval(
    value: Any,
    *,
    plan: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    current = int(time.time()) if now_unix is None else now_unix
    approved_plan = validate_migration_plan(plan, now_unix=current)
    if not isinstance(value, Mapping) or set(value) != _APPROVAL_FIELDS:
        raise OsLoginMetadataMigrationError("os_login_metadata_approval_invalid")
    unsigned = {name: item for name, item in value.items() if name != "approval_sha256"}
    signed = {
        name: item
        for name, item in value.items()
        if name not in {"signature_ed25519_hex", "approval_sha256"}
    }
    try:
        public = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(approved_plan["owner_public_key_ed25519_hex"])
        )
        public.verify(
            bytes.fromhex(str(value.get("signature_ed25519_hex"))),
            _canonical(signed),
        )
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise OsLoginMetadataMigrationError(
            "os_login_metadata_approval_invalid"
        ) from exc
    if (
        value.get("schema") != APPROVAL_SCHEMA
        or value.get("purpose") != "production_os_login_metadata_apply"
        or value.get("plan_sha256") != approved_plan["plan_sha256"]
        or value.get("owner_subject_sha256") != approved_plan["owner_subject_sha256"]
        or value.get("owner_public_key_ed25519_hex")
        != approved_plan["owner_public_key_ed25519_hex"]
        or value.get("owner_key_id") != approved_plan["owner_key_id"]
        or _SHA256.fullmatch(str(value.get("nonce_sha256"))) is None
        or type(value.get("issued_at_unix")) is not int
        or type(value.get("expires_at_unix")) is not int
        or not value["issued_at_unix"] <= current < value["expires_at_unix"]
        or not 1 <= value["expires_at_unix"] - value["issued_at_unix"] <= 900
        or value.get("approved") is not True
        or not isinstance(value.get("signature_ed25519_hex"), str)
        or re.fullmatch(r"[0-9a-f]{128}", value["signature_ed25519_hex"]) is None
        or value.get("approval_sha256") != _sha(_canonical(unsigned))
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_approval_invalid")
    return copy.deepcopy(dict(value))


def build_authority_bundle(
    *,
    release_revision: str,
    preflight: Mapping[str, Any],
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    current = int(time.time()) if now_unix is None else now_unix
    observed = validate_preflight_receipt(preflight, now_unix=current)
    approved_plan = validate_migration_plan(plan, now_unix=current)
    approved = validate_migration_approval(
        approval,
        plan=approved_plan,
        now_unix=current,
    )
    if (
        re.fullmatch(r"[0-9a-f]{40}", release_revision or "") is None
        or observed["state"] != "legacy_instance_ssh_keys"
        or approved_plan["preflight_receipt_sha256"] != observed["receipt_sha256"]
        or approved_plan["preflight_state_identity_sha256"]
        != observed["state_identity_sha256"]
        or approved_plan["owner_subject_sha256"] != observed["owner_subject_sha256"]
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_authority_invalid")
    unsigned = {
        "schema": AUTHORITY_SCHEMA,
        "release_revision": release_revision,
        "preflight": observed,
        "plan": approved_plan,
        "approval": approved,
        "private_key_staged": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _hashed(unsigned, "authority_sha256")


def validate_authority_bundle(
    value: Any,
    *,
    release_revision: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    current = int(time.time()) if now_unix is None else now_unix
    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
        raise OsLoginMetadataMigrationError("os_login_metadata_authority_invalid")
    unsigned = {
        name: item for name, item in value.items() if name != "authority_sha256"
    }
    if (
        value.get("schema") != AUTHORITY_SCHEMA
        or value.get("release_revision") != release_revision
        or value.get("private_key_staged") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("authority_sha256") != _sha(_canonical(unsigned))
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_authority_invalid")
    rebuilt = build_authority_bundle(
        release_revision=release_revision,
        preflight=value["preflight"],
        plan=value["plan"],
        approval=value["approval"],
        now_unix=current,
    )
    if rebuilt != value:
        raise OsLoginMetadataMigrationError("os_login_metadata_authority_invalid")
    return copy.deepcopy(dict(value))


def build_migration_intent(
    *,
    observed: ObservedState,
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Bind the exact recoverable pre-mutation state before any Cloud write."""

    current = int(time.time()) if now_unix is None else now_unix
    approved_plan = validate_migration_plan(plan, now_unix=current)
    approved = validate_migration_approval(
        approval,
        plan=approved_plan,
        now_unix=current,
    )
    prior = validate_preflight_receipt(observed.receipt, now_unix=current)
    instance_metadata = dict(observed.instance_metadata)
    project_metadata = dict(observed.project_metadata)
    prior_ssh_keys = instance_metadata.get("ssh-keys")
    if (
        len(instance_metadata) != len(observed.instance_metadata)
        or len(project_metadata) != len(observed.project_metadata)
        or prior["state"] != "legacy_instance_ssh_keys"
        or not _same_preflight_identity(prior, approved_plan)
        or sorted(instance_metadata) != prior["instance_metadata_keys"]
        or sorted(project_metadata) != prior["project_metadata_keys"]
        or instance_metadata.get("enable-oslogin")
        != approved_plan["prior_enable_oslogin"]
        or not isinstance(prior_ssh_keys, str)
        or not 0 < len(prior_ssh_keys.encode("utf-8")) <= MAX_METADATA_BYTES
        or _sha(prior_ssh_keys.encode("utf-8"))
        != approved_plan["prior_public_ssh_keys_sha256"]
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_intent_invalid")
    intermediate_keys = sorted({*instance_metadata, "enable-oslogin"})
    ready_keys = [key for key in intermediate_keys if key != "ssh-keys"]
    unsigned = {
        "schema": INTENT_SCHEMA,
        "target": copy.deepcopy(_TARGET),
        "plan_sha256": approved_plan["plan_sha256"],
        "approval_sha256": approved["approval_sha256"],
        "prior_preflight": prior,
        "prior_ssh_keys": prior_ssh_keys,
        "prior_enable_oslogin": approved_plan["prior_enable_oslogin"],
        "expected_intermediate_instance_metadata_keys": intermediate_keys,
        "expected_ready_instance_metadata_keys": ready_keys,
        "created_at_unix": current,
        "private_key_staged": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return validate_migration_intent(
        _hashed(unsigned, "intent_sha256"),
        plan=approved_plan,
        approval=approved,
        now_unix=current,
    )


def validate_migration_intent(
    value: Any,
    *,
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Validate a durable intent even after its one-shot approval expires."""

    current = int(time.time()) if now_unix is None else now_unix
    if not isinstance(value, Mapping) or set(value) != _INTENT_FIELDS:
        raise OsLoginMetadataMigrationError("os_login_metadata_intent_invalid")
    created_at = value.get("created_at_unix")
    if (
        type(created_at) is not int
        or created_at <= 0
        or created_at > current + 30
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_intent_invalid")
    approved_plan = validate_migration_plan(plan, now_unix=created_at)
    approved = validate_migration_approval(
        approval,
        plan=approved_plan,
        now_unix=created_at,
    )
    prior = validate_preflight_receipt(
        value.get("prior_preflight"),
        now_unix=created_at,
    )
    prior_keys = prior["instance_metadata_keys"]
    intermediate_keys = sorted({*prior_keys, "enable-oslogin"})
    ready_keys = [key for key in intermediate_keys if key != "ssh-keys"]
    prior_ssh_keys = value.get("prior_ssh_keys")
    unsigned = {name: item for name, item in value.items() if name != "intent_sha256"}
    if (
        value.get("schema") != INTENT_SCHEMA
        or value.get("target") != _TARGET
        or value.get("plan_sha256") != approved_plan["plan_sha256"]
        or value.get("approval_sha256") != approved["approval_sha256"]
        or prior["state"] != "legacy_instance_ssh_keys"
        or not _same_preflight_identity(prior, approved_plan)
        or not isinstance(prior_ssh_keys, str)
        or not 0 < len(prior_ssh_keys.encode("utf-8")) <= MAX_METADATA_BYTES
        or _sha(prior_ssh_keys.encode("utf-8"))
        != approved_plan["prior_public_ssh_keys_sha256"]
        or value.get("prior_enable_oslogin")
        != approved_plan["prior_enable_oslogin"]
        or value.get("expected_intermediate_instance_metadata_keys")
        != intermediate_keys
        or value.get("expected_ready_instance_metadata_keys") != ready_keys
        or value.get("private_key_staged") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("intent_sha256") != _sha(_canonical(unsigned))
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_intent_invalid")
    return copy.deepcopy(dict(value))


def _same_preflight_identity(
    observed: Mapping[str, Any], plan: Mapping[str, Any]
) -> bool:
    return (
        observed.get("state_identity_sha256")
        == plan.get("preflight_state_identity_sha256")
        and observed.get("instance_metadata_fingerprint")
        == plan.get("prior_instance_metadata_fingerprint")
        and observed.get("enable_oslogin") == plan.get("prior_enable_oslogin")
        and observed.get("public_ssh_keys_sha256")
        == plan.get("prior_public_ssh_keys_sha256")
        and observed.get("owner_subject_sha256") == plan.get("owner_subject_sha256")
    )


def validate_migration_receipt(
    value: Any,
    *,
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    if not isinstance(plan, Mapping) or not isinstance(approval, Mapping):
        raise OsLoginMetadataMigrationError("os_login_metadata_receipt_invalid")
    authorization_time = plan.get("issued_at_unix")
    approval_time = approval.get("issued_at_unix")
    if type(authorization_time) is not int or type(approval_time) is not int:
        raise OsLoginMetadataMigrationError("os_login_metadata_receipt_invalid")
    approved_plan = validate_migration_plan(plan, now_unix=authorization_time)
    approved = validate_migration_approval(
        approval,
        plan=approved_plan,
        now_unix=approval_time,
    )
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        raise OsLoginMetadataMigrationError("os_login_metadata_receipt_invalid")
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("target") != _TARGET
        or value.get("plan_sha256") != approved_plan["plan_sha256"]
        or value.get("approval_sha256") != approved["approval_sha256"]
        or _SHA256.fullmatch(str(value.get("intent_sha256"))) is None
        or any(
            _SHA256.fullmatch(str(value.get(field))) is None
            for field in (
                "prior_state_identity_sha256",
                "ready_state_identity_sha256",
                "access_receipt_sha256",
            )
        )
        or value.get("enable_oslogin") != "TRUE"
        or value.get("instance_ssh_keys_present") is not False
        or value.get("iap_os_login_succeeded") is not True
        or value.get("rollback_used") is not False
        or value.get("private_key_staged") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or type(value.get("completed_at_unix")) is not int
        or value["completed_at_unix"] < approved["issued_at_unix"]
        or value.get("receipt_sha256") != _sha(_canonical(unsigned))
    ):
        raise OsLoginMetadataMigrationError("os_login_metadata_receipt_invalid")
    return copy.deepcopy(dict(value))


_STABLE_RECOVERY_FIELDS = (
    "target",
    "owner_subject_sha256",
    "owner_account_sha256",
    "project_metadata_fingerprint",
    "project_metadata_keys",
    "os_login_profile_sha256",
    "os_login_public_key_fingerprint",
    "iam_permissions",
)


def _transition_matches_intent(
    receipt: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
) -> bool:
    prior = intent["prior_preflight"]
    state = receipt.get("state")
    if state == "os_login_enabled_with_legacy_keys":
        expected_keys = intent["expected_intermediate_instance_metadata_keys"]
        expected_ssh_keys = True
        expected_ssh_sha256 = _sha(intent["prior_ssh_keys"].encode("utf-8"))
    elif state == "os_login_ready":
        expected_keys = intent["expected_ready_instance_metadata_keys"]
        expected_ssh_keys = False
        expected_ssh_sha256 = None
    else:
        return False
    return (
        receipt.get("instance_metadata_keys") == expected_keys
        and receipt.get("enable_oslogin") == "TRUE"
        and receipt.get("ssh_keys_present") is expected_ssh_keys
        and receipt.get("public_ssh_keys_sha256") == expected_ssh_sha256
        and all(receipt.get(field) == prior[field] for field in _STABLE_RECOVERY_FIELDS)
    )


def _restore_prior_state(
    *,
    boundary: MetadataMigrationBoundary,
    intent: Mapping[str, Any],
    validation_now: Callable[[], int],
) -> ObservedState:
    failures: list[BaseException] = []
    try:
        boundary.restore_instance_ssh_keys(intent["prior_ssh_keys"])
    except BaseException as exc:
        failures.append(exc)
    try:
        boundary.restore_enable_oslogin(intent["prior_enable_oslogin"])
    except BaseException as exc:
        failures.append(exc)
    if failures:
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup(
            "OS Login metadata rollback operations were incomplete",
            failures,
        )
    restored = boundary.observe()
    restored_receipt = validate_preflight_receipt(
        restored.receipt,
        now_unix=validation_now(),
    )
    prior = intent["prior_preflight"]
    restored_metadata = dict(restored.instance_metadata)
    if (
        restored_receipt["state"] != "legacy_instance_ssh_keys"
        or restored_receipt["state_identity_sha256"]
        != prior["state_identity_sha256"]
        or restored_metadata.get("ssh-keys") != intent["prior_ssh_keys"]
        or restored_metadata.get("enable-oslogin")
        != intent["prior_enable_oslogin"]
    ):
        raise OsLoginMetadataMigrationError(
            "os_login_metadata_rollback_readback_invalid"
        )
    return restored


def execute_migration(
    *,
    boundary: MetadataMigrationBoundary,
    plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    intent: Mapping[str, Any],
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Apply the fixed transition with durable, crash-resumable rollback truth."""

    current = int(time.time()) if now_unix is None else now_unix
    validation_now = (
        (lambda: int(time.time())) if now_unix is None else (lambda: now_unix)
    )
    recovery = validate_migration_intent(
        intent,
        plan=plan,
        approval=approval,
        now_unix=current,
    )
    authorization_time = recovery["created_at_unix"]
    approved_plan = validate_migration_plan(plan, now_unix=authorization_time)
    approved = validate_migration_approval(
        approval,
        plan=approved_plan,
        now_unix=authorization_time,
    )
    prior_receipt = recovery["prior_preflight"]

    before = boundary.observe()
    before_receipt = validate_preflight_receipt(
        before.receipt,
        now_unix=validation_now(),
    )
    before_metadata = dict(before.instance_metadata)
    prior_ssh_keys = before_metadata.get("ssh-keys")
    if (
        before_receipt["state"] == "legacy_instance_ssh_keys"
        and _same_preflight_identity(before_receipt, approved_plan)
        and before_receipt["state_identity_sha256"]
        == prior_receipt["state_identity_sha256"]
        and prior_ssh_keys == recovery["prior_ssh_keys"]
    ):
        pass
    elif _transition_matches_intent(before_receipt, intent=recovery):
        before = _restore_prior_state(
            boundary=boundary,
            intent=recovery,
            validation_now=validation_now,
        )
        before_receipt = validate_preflight_receipt(
            before.receipt,
            now_unix=validation_now(),
        )
        before_metadata = dict(before.instance_metadata)
        prior_ssh_keys = before_metadata.get("ssh-keys")
    else:
        raise OsLoginMetadataMigrationError("os_login_metadata_pre_mutation_drifted")

    mutation_started = False
    try:
        mutation_started = True
        boundary.set_enable_oslogin_true()
        intermediate = boundary.observe()
        intermediate_receipt = validate_preflight_receipt(
            intermediate.receipt,
            now_unix=validation_now(),
        )
        expected_intermediate = {**before_metadata, "enable-oslogin": "TRUE"}
        if (
            dict(intermediate.instance_metadata) != expected_intermediate
            or intermediate.project_metadata != before.project_metadata
            or intermediate_receipt["state"] != "os_login_enabled_with_legacy_keys"
        ):
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_intermediate_invalid"
            )

        boundary.remove_instance_ssh_keys()
        ready = boundary.observe()
        ready_receipt = validate_preflight_receipt(
            ready.receipt,
            now_unix=validation_now(),
        )
        expected_ready = {
            key: value
            for key, value in expected_intermediate.items()
            if key != "ssh-keys"
        }
        if (
            dict(ready.instance_metadata) != expected_ready
            or ready.project_metadata != before.project_metadata
            or ready_receipt["state"] != "os_login_ready"
            or any(
                ready_receipt[field] != before_receipt[field]
                for field in _STABLE_RECOVERY_FIELDS
            )
        ):
            raise OsLoginMetadataMigrationError("os_login_metadata_ready_state_invalid")
        access = boundary.probe_iap_os_login()
        if not isinstance(access, Mapping):
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_access_probe_invalid"
            )
        access_unsigned = {
            name: item for name, item in access.items() if name != "receipt_sha256"
        }
        access_now = validation_now()
        if (
            set(access) != _ACCESS_FIELDS
            or access.get("schema") != ACCESS_SCHEMA
            or access.get("target") != _TARGET
            or access.get("owner_subject_sha256")
            != approved_plan["owner_subject_sha256"]
            or access.get("owner_account_sha256")
            != ready_receipt["owner_account_sha256"]
            or access.get("iap_os_login_succeeded") is not True
            or access.get("fixed_remote_command") != "/usr/bin/true"
            or _SHA256.fullmatch(str(access.get("authorization_snapshot_sha256")))
            is None
            or access.get("secret_material_recorded") is not False
            or access.get("secret_digest_recorded") is not False
            or type(access.get("observed_at_unix")) is not int
            or not access_now - MAX_AGE_SECONDS
            <= access["observed_at_unix"]
            <= access_now + 30
            or access.get("receipt_sha256") != _sha(_canonical(access_unsigned))
        ):
            raise OsLoginMetadataMigrationError(
                "os_login_metadata_access_probe_invalid"
            )
        final = boundary.observe()
        final_receipt = validate_preflight_receipt(
            final.receipt,
            now_unix=validation_now(),
        )
        if (
            final.instance_metadata != ready.instance_metadata
            or final.project_metadata != ready.project_metadata
            or final_receipt["state"] != "os_login_ready"
            or final_receipt["state_identity_sha256"]
            != ready_receipt["state_identity_sha256"]
        ):
            raise OsLoginMetadataMigrationError("os_login_metadata_post_access_drifted")
    except BaseException as primary:
        if mutation_started:
            try:
                _restore_prior_state(
                    boundary=boundary,
                    intent=recovery,
                    validation_now=validation_now,
                )
            except BaseException as rollback_error:
                raise BaseExceptionGroup(
                    "OS Login metadata migration failed and rollback was incomplete",
                    [primary, rollback_error],
                ) from None
        raise

    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "target": copy.deepcopy(_TARGET),
        "plan_sha256": approved_plan["plan_sha256"],
        "approval_sha256": approved["approval_sha256"],
        "intent_sha256": recovery["intent_sha256"],
        "prior_state_identity_sha256": prior_receipt["state_identity_sha256"],
        "ready_state_identity_sha256": final_receipt["state_identity_sha256"],
        "access_receipt_sha256": access["receipt_sha256"],
        "enable_oslogin": "TRUE",
        "instance_ssh_keys_present": False,
        "iap_os_login_succeeded": True,
        "rollback_used": False,
        "private_key_staged": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "completed_at_unix": int(time.time()) if now_unix is None else now_unix,
    }
    return validate_migration_receipt(
        _hashed(unsigned, "receipt_sha256"),
        plan=approved_plan,
        approval=approved,
        now_unix=current,
    )
