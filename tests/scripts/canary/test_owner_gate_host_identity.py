from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import struct
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_journal as foundation_journal
from scripts.canary import owner_gate_host_identity as host
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_trust as owner_trust
from scripts.canary import source_artifact_publication as source_publication


FOUNDATION_REVISION = "a" * 40
PINNING_REVISION = "b" * 40
INSTANCE_ID = "1234567890123456789"
SERVICE_ACCOUNT_ID = "2234567890123456789"
NETWORK_ID = "3234567890123456789"
SUBNETWORK_ID = "4234567890123456789"
DISK_ID = "5234567890123456789"
IMAGE_ID = "6234567890123456789"
IMAGE = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
    "global/images/debian-12-bookworm-v20260701"
)
_HOST_ALGORITHM = b"ssh-ed25519"
HOST_KEY = base64.b64encode(
    struct.pack(">I", len(_HOST_ALGORITHM))
    + _HOST_ALGORITHM
    + struct.pack(">I", 32)
    + b"H" * 32
).decode("ascii")
SECOND_HOST_KEY = base64.b64encode(
    struct.pack(">I", len(_HOST_ALGORITHM))
    + _HOST_ALGORITHM
    + struct.pack(">I", 32)
    + b"J" * 32
).decode("ascii")
_COLLECTION_REAUTH_KEY = Ed25519PrivateKey.generate()
_COLLECTION_REAUTH_KEY_ID = hashlib.sha256(
    _COLLECTION_REAUTH_KEY.public_key().public_bytes_raw()
).hexdigest()


def _collection_reauth_receipt() -> Mapping[str, Any]:
    body = {
        "schema": owner_reauth.RECEIPT_SCHEMA,
        "purpose": owner_reauth.RECEIPT_PURPOSE,
        "trusted_runtime_identity": {
            "release_revision": FOUNDATION_REVISION,
            "sealed_runtime_identity_sha256": "9" * 64,
            "command_prefix_sha256": "1" * 64,
            "python_executable_sha256": "2" * 64,
            "gcloud_module_sha256": "3" * 64,
            "sdk_root": (
                "/sealed/google-cloud-sdk-"
                f"{owner_reauth.GCLOUD_SDK_VERSION}"
            ),
            "sdk_python_config_identity_sha256": "4" * 64,
            "closed_environment_sha256": "5" * 64,
            "configuration": owner_reauth.GCLOUD_CONFIGURATION,
            "account": owner_reauth.OWNER_ACCOUNT,
            "project": foundation.PROJECT,
            "zone": foundation.ZONE,
        },
        "interactive_reauthentication": {
            "method": "gcloud_auth_login_force_interactive",
            "started_at_unix": 88,
            "completed_at_unix": 89,
            "command_sha256": "6" * 64,
            "interactive_tty_verified": True,
            "access_token_requested": False,
            "credential_material_captured": False,
        },
        "authenticated_probe": {
            "command_sha256": "7" * 64,
            "output_sha256": "8" * 64,
            "project_id": foundation.PROJECT,
            "project_number": launcher.OWNER_GATE_PROJECT_NUMBER,
        },
        "issued_at_unix": 90,
        "expires_at_unix": 200,
        "signer_key_id": _COLLECTION_REAUTH_KEY_ID,
    }
    return owner_reauth._sign_owner_reauth_receipt(
        body,
        private_key=_COLLECTION_REAUTH_KEY,
    )


@pytest.fixture
def collection_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> host._CollectionAuthorization:
    monkeypatch.setattr(
        owner_trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        _COLLECTION_REAUTH_KEY_ID,
    )
    return host._validate_collection_authorization(
        _collection_reauth_receipt(),
        public_key=_COLLECTION_REAUTH_KEY.public_key(),
        now_unix=100,
    )


def _foundation_journal_for_test(
    root: Path,
) -> foundation_journal.FoundationApplyJournal:
    os.chmod(root.parent, 0o700)
    os.chown(root.parent, os.geteuid(), os.getegid())
    return foundation_journal.FoundationApplyJournal(
        _root=root,
        _owner_uid=os.geteuid(),
        _owner_gid=os.getegid(),
    )


def _chain(**overrides: Any) -> host._FoundationChainProjection:
    values = {
        "foundation_source_revision": FOUNDATION_REVISION,
        "foundation_source_tree_oid": "c" * 40,
        "owner_reauthentication_receipt_sha256": "1" * 64,
        "owner_reauthentication_expires_at_unix": 1_000,
        "pre_foundation_authority_sha256": "2" * 64,
        "foundation_apply_receipt_sha256": "3" * 64,
        "direct_iam_authority_sha256": "4" * 64,
        "ancestry_evidence_sha256": "5" * 64,
        "ancestry_chain_sha256": "6" * 64,
        "signed_network_evidence_sha256": "7" * 64,
        "network_evidence_sha256": "8" * 64,
        "project_number": launcher.OWNER_GATE_PROJECT_NUMBER,
        "vm_numeric_id": INSTANCE_ID,
        "vm_self_link": host.EXPECTED_VM_SELF_LINK,
        "vm_creation_timestamp": "2026-07-15T10:00:00+00:00",
        "service_account_unique_id": SERVICE_ACCOUNT_ID,
        "network_numeric_id": NETWORK_ID,
        "subnetwork_numeric_id": SUBNETWORK_ID,
        "subnetwork_self_link": host.EXPECTED_SUBNETWORK_SELF_LINK,
        "network_self_link": host.EXPECTED_NETWORK_SELF_LINK,
        "boot_disk_numeric_id": DISK_ID,
        "boot_disk_architecture": "X86_64",
        "boot_disk_physical_block_size_bytes": 4096,
        "boot_disk_licenses": host.EXPECTED_DEBIAN_LICENSES,
        "boot_image_numeric_id": IMAGE_ID,
        "boot_image_self_link": IMAGE,
        **overrides,
    }
    return host._FoundationChainProjection(**values)


def _responses(**overrides: Any) -> Mapping[str, Any]:
    values: dict[str, Any] = {
        "instance": {
            "kind": "compute#instance",
            "id": INSTANCE_ID,
            "name": foundation.VM_NAME,
            "selfLink": host.EXPECTED_VM_SELF_LINK,
            "zone": host.EXPECTED_ZONE_SELF_LINK,
            "machineType": host.EXPECTED_MACHINE_TYPE_SELF_LINK,
            "status": "RUNNING",
            "creationTimestamp": "2026-07-15T10:00:00+00:00",
            "canIpForward": False,
            "deletionProtection": False,
            "scheduling": {
                key: value
                for key, value in host.EXPECTED_SCHEDULING.items()
                if key != "instanceTerminationAction"
            },
            "labels": {},
            "labelFingerprint": "label-fingerprint",
            "resourcePolicies": [],
            "minCpuPlatform": "Automatic",
            "confidentialInstanceConfig": {
                "enableConfidentialCompute": False,
            },
            "tags": {
                "fingerprint": "tag-fingerprint",
                "items": ["muncho-owner-gate", "iap-ssh"],
            },
            "metadata": {
                "fingerprint": "metadata-fingerprint",
                "items": [
                    {"key": "enable-oslogin", "value": "TRUE"},
                    {"key": "block-project-ssh-keys", "value": "TRUE"},
                    {"key": "serial-port-enable", "value": "FALSE"},
                ],
            },
            "shieldedInstanceConfig": dict(host.EXPECTED_SHIELDED_CONFIG),
            "serviceAccounts": [{
                "email": host.SERVICE_ACCOUNT_EMAIL,
                "scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
            }],
            "networkInterfaces": [{
                "network": host.EXPECTED_NETWORK_SELF_LINK,
                "subnetwork": host.EXPECTED_SUBNETWORK_SELF_LINK,
                "networkIP": foundation.OWNER_GATE_PRIVATE_IP,
                "aliasIpRanges": [],
                "accessConfigs": [],
                "ipv6AccessConfigs": [],
                "stackType": "IPV4_ONLY",
            }],
            "disks": [{
                "boot": True,
                "autoDelete": True,
                "mode": "READ_WRITE",
                "type": "PERSISTENT",
                "interface": "SCSI",
                "index": 0,
                "source": host.EXPECTED_BOOT_DISK_SELF_LINK,
                "deviceName": foundation.TARGET_BOOT_DEVICE,
                "diskSizeGb": str(foundation.BOOT_DISK_SIZE_GB),
            }],
        },
        "disk": {
            "kind": "compute#disk",
            "id": DISK_ID,
            "name": foundation.VM_NAME,
            "selfLink": host.EXPECTED_BOOT_DISK_SELF_LINK,
            "zone": host.EXPECTED_ZONE_SELF_LINK,
            "status": "READY",
            "type": host.EXPECTED_BOOT_DISK_TYPE_SELF_LINK,
            "sizeGb": str(foundation.BOOT_DISK_SIZE_GB),
            "sourceImage": IMAGE,
            "sourceImageId": IMAGE_ID,
            "users": [host.EXPECTED_VM_SELF_LINK],
            "architecture": "X86_64",
            "physicalBlockSizeBytes": 4096,
            "licenses": list(host.EXPECTED_DEBIAN_LICENSES),
        },
        "image": {
            "kind": "compute#image",
            "id": IMAGE_ID,
            "name": "debian-12-bookworm-v20260701",
            "selfLink": IMAGE,
            "status": "READY",
            "family": "debian-12",
            "architecture": "X86_64",
            "licenses": list(host.EXPECTED_DEBIAN_LICENSES),
        },
        "network": {
            "kind": "compute#network",
            "id": NETWORK_ID,
            "name": foundation.NETWORK_NAME,
            "selfLink": host.EXPECTED_NETWORK_SELF_LINK,
            "autoCreateSubnetworks": False,
        },
        "subnetwork": {
            "kind": "compute#subnetwork",
            "id": SUBNETWORK_ID,
            "name": foundation.OWNER_GATE_SUBNET_NAME,
            "selfLink": host.EXPECTED_SUBNETWORK_SELF_LINK,
            "network": host.EXPECTED_NETWORK_SELF_LINK,
            "region": host.EXPECTED_REGION_SELF_LINK,
            "ipCidrRange": foundation.OWNER_GATE_SUBNET_CIDR,
            "privateIpGoogleAccess": True,
            "stackType": "IPV4_ONLY",
            "purpose": "PRIVATE",
            "secondaryIpRanges": [],
        },
    }
    values.update(overrides)
    return values


def _owner_signer(
    tmp_path: Path,
) -> tuple[launcher._PhaseBOwnerExternalSigner, Ed25519PrivateKey]:
    key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "owner-key"
    public_path = tmp_path / "owner-key.pub"
    comment = "host-identity-v2-test"
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.OpenSSH,
            serialization.NoEncryption(),
        )
    )
    public_line = key.public_key().public_bytes(
        serialization.Encoding.OpenSSH,
        serialization.PublicFormat.OpenSSH,
    ) + b" " + comment.encode("ascii") + b"\n"
    public_path.write_bytes(public_line)
    os.chmod(private_path, 0o600)
    os.chmod(public_path, 0o600)
    public_blob = base64.b64decode(public_line.split(b" ", 2)[1], validate=True)
    fingerprint = "SHA256:" + base64.b64encode(
        hashlib.sha256(public_blob).digest()
    ).decode("ascii").rstrip("=")
    return (
        launcher._PhaseBOwnerExternalSigner(
            private_key_path=private_path,
            public_key_path=public_path,
            expected_comment=comment,
            expected_fingerprint=fingerprint,
        ),
        key,
    )


def _exact_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    launcher.TrustedGcloudExecutable,
    launcher.PinnedGcloudConfiguration,
    host.TrustedOwnerGateSshExecutable,
]:
    runtime = object.__new__(launcher.TrustedGcloudExecutable)
    configuration = object.__new__(launcher.PinnedGcloudConfiguration)
    ssh = object.__new__(host.TrustedOwnerGateSshExecutable)
    prefix = (
        "/trusted/python",
        *launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
        "/trusted/google-cloud-sdk/lib/gcloud.py",
    )
    monkeypatch.setattr(
        launcher.TrustedGcloudExecutable,
        "trusted_command_prefix",
        lambda _self: prefix,
    )
    monkeypatch.setattr(
        launcher.PinnedGcloudConfiguration,
        "account",
        property(lambda _self: host.OWNER_ACCOUNT),
    )
    monkeypatch.setattr(
        launcher.PinnedGcloudConfiguration,
        "assert_stable",
        lambda _self: None,
    )
    monkeypatch.setattr(
        launcher,
        "_owner_gcloud_environment",
        lambda _configuration, _python: {
            **{
                name: "fixed"
                for name in launcher.OwnerGateIapTransport._ENVIRONMENT_KEYS
            },
            "CLOUDSDK_CORE_PROJECT": foundation.PROJECT,
            "CLOUDSDK_COMPUTE_ZONE": foundation.ZONE,
        },
    )
    monkeypatch.setattr(
        host.TrustedOwnerGateSshExecutable,
        "paths",
        lambda _self: ("/usr/bin/ssh", "/bin/sh"),
    )
    monkeypatch.setattr(
        host.TrustedOwnerGateSshExecutable,
        "identity",
        lambda _self: {
            "ssh_executable_sha256": "a" * 64,
            "ssh_version": "OpenSSH_10.2p1, LibreSSL 3.3.6",
            "shell_executable_sha256": "b" * 64,
            "shell_version": "3.2.57(1)-release",
        },
    )
    return runtime, configuration, ssh


def _runner_for_keys(keys: list[str], calls: list[tuple[str, ...]]):
    def runner(argv: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(tuple(argv))
        path_arg = next(
            item for item in argv if item.startswith("-oUserKnownHostsFile=")
        )
        path = Path(path_arg.split("=", 1)[1])
        selected = keys[min(len(calls) - 1, len(keys) - 1)]
        path.write_text(
            f"compute.{INSTANCE_ID} ssh-ed25519 {selected}\n",
            encoding="ascii",
        )
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(argv, 255, b"", b"")

    return runner


def test_direct_compute_projection_binds_all_foundation_identities() -> None:
    identity = host._direct_compute_identity(_responses(), chain=_chain())

    assert identity.value["vm_numeric_id"] == INSTANCE_ID
    assert identity.value["owner_gate_service_account_unique_id"] == SERVICE_ACCOUNT_ID
    assert identity.value["network_numeric_id"] == NETWORK_ID
    assert identity.value["subnetwork_numeric_id"] == SUBNETWORK_ID
    assert identity.value["boot_disk_numeric_id"] == DISK_ID
    assert identity.value["boot_image_numeric_id"] == IMAGE_ID
    assert identity.value["external_ip_present"] is False


def test_projection_is_derived_from_real_signed_apply_and_canonical_direct_iam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.canary import direct_iam_identity_author as direct_author
    from scripts.canary import owner_gate_foundation_apply as foundation_apply
    from scripts.canary import owner_gate_trust as trust
    from tests.scripts.canary import test_direct_iam_identity_author as direct_fixture
    from tests.scripts.canary import (
        test_owner_gate_foundation_apply as apply_fixture,
    )
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    monkeypatch.setattr(
        fixture,
        "PROJECT_NUMBER",
        launcher.OWNER_GATE_PROJECT_NUMBER,
    )
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        fixture.RELEASE_KEY_ID,
    )
    authority, plan, _evidence = fixture._authority()
    owner_raw = foundation.canonical_json_bytes(fixture._owner_reauth_receipt())
    network_raw = foundation.canonical_json_bytes(
        fixture._signed_network_evidence()
    )
    ancestry_raw = fixture._signed_ancestry_raw()
    foundation_a = foundation_apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=foundation.canonical_json_bytes(authority),
        owner_reauthentication_receipt_raw=owner_raw,
        network_evidence_raw=network_raw,
        project_ancestry_evidence_raw=ancestry_raw,
        release_public_key=fixture.RELEASE_KEY.public_key(),
        network_collector_public_key=fixture.NETWORK_KEY.public_key(),
        project_ancestry_collector_public_key=fixture.NETWORK_KEY.public_key(),
        now_unix=fixture.NOW + 1,
    )
    os.chmod(tmp_path, 0o700)
    os.chown(tmp_path, -1, os.getegid())
    store = _foundation_journal_for_test(tmp_path / "journal")
    foundation_apply._apply_with_provider(
        chain=foundation_a,
        private_key=fixture.RELEASE_KEY,
        provider=apply_fixture._FakeProvider(foundation_a),
        journal=store,
        now_unix=lambda: fixture.NOW + 2,
    )
    monkeypatch.setattr(
        foundation_apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(
        foundation_apply.time,
        "time",
        lambda: float(fixture.NOW + 21),
    )
    apply_chain = foundation_apply.load_validated_foundation_apply_chain(
        foundation_a
    )

    facts = dict(direct_fixture._live_facts())
    instance = dict(facts["instance"])
    instance["id"] = apply_chain.owner_gate_vm_identity["numeric_id"]
    facts["instance"] = instance
    service_account = dict(facts["owner_gate_service_account"])
    service_account["uniqueId"] = apply_chain.service_account_identity["unique_id"]
    facts["owner_gate_service_account"] = service_account
    folder = f"folders/{fixture.FOLDER_ID}"
    organization = f"organizations/{fixture.ORGANIZATION_ID}"
    resources = [dict(item) for item in facts["resources"]]
    resources[0]["parent"] = folder
    resources.insert(1, {
        "name": folder,
        "parent": organization,
        "state": "ACTIVE",
        "etag": "folder-etag",
    })
    facts["resources"] = resources
    facts["resource_names"] = [direct_author.PROJECT_RESOURCE, folder, organization]
    policies = [dict(item) for item in facts["policies"]]
    policies.insert(1, {
        "version": 3,
        "etag": "folder-policy-etag",
        "bindings": [],
        "auditConfigs": [],
    })
    facts["policies"] = policies
    direct_raw = direct_author._build_authority_bytes(
        live_facts=facts,
        release_revision=apply_chain.foundation_source_revision,
        owner_reauthentication_receipt_sha256=(
            apply_chain.owner_reauthentication_receipt_sha256
        ),
        pre_foundation_authority_sha256=(
            apply_chain.pre_foundation_authority_sha256
        ),
        foundation_apply_receipt_sha256=(
            apply_chain.foundation_apply_receipt_sha256
        ),
        collected_at_unix=fixture.NOW + 21,
    )

    projection = host._projection_from_validated_chains(
        foundation_chain=apply_chain,
        direct_iam_authority_raw=direct_raw,
        now_unix=fixture.NOW + 21,
    )

    assert projection.vm_numeric_id == "3333333333333333333"
    assert projection.network_numeric_id == "6666666666666666666"
    assert projection.subnetwork_numeric_id == "2222222222222222222"
    assert projection.boot_disk_numeric_id == "5555555555555555555"
    assert projection.service_account_unique_id == "111111111111111111111"

    attacker_facts = dict(facts)
    attacker_instance = dict(attacker_facts["instance"])
    attacker_instance["id"] = "7777777777777777777"
    attacker_facts["instance"] = attacker_instance
    attacker_direct_raw = direct_author._build_authority_bytes(
        live_facts=attacker_facts,
        release_revision=apply_chain.foundation_source_revision,
        owner_reauthentication_receipt_sha256=(
            apply_chain.owner_reauthentication_receipt_sha256
        ),
        pre_foundation_authority_sha256=(
            apply_chain.pre_foundation_authority_sha256
        ),
        foundation_apply_receipt_sha256=(
            apply_chain.foundation_apply_receipt_sha256
        ),
        collected_at_unix=fixture.NOW + 21,
    )
    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_foundation_chain_mismatch",
    ):
        host._projection_from_validated_chains(
            foundation_chain=apply_chain,
            direct_iam_authority_raw=attacker_direct_raw,
            now_unix=fixture.NOW + 21,
        )


@pytest.mark.parametrize(
    ("resource", "field", "value"),
    [
        ("instance", "id", "7234567890123456789"),
        ("instance", "selfLink", "https://attacker.invalid/vm"),
        ("disk", "sourceImageId", "7234567890123456789"),
        ("network", "id", "not-numeric"),
        ("subnetwork", "id", "7234567890123456789"),
    ],
)
def test_direct_compute_projection_rejects_recreate_or_apply_mismatch(
    resource: str,
    field: str,
    value: str,
) -> None:
    responses = {name: dict(item) for name, item in _responses().items()}
    responses[resource][field] = value

    with pytest.raises(host.OwnerGateHostIdentityError):
        host._direct_compute_identity(responses, chain=_chain())


def test_iap_capture_is_no_auth_no_command_and_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)
    calls: list[tuple[str, ...]] = []

    observed = host._stable_iap_host_key(
        runtime=runtime,
        configuration=configuration,
        ssh=ssh,
        instance_id=INSTANCE_ID,
        runner=_runner_for_keys([HOST_KEY, HOST_KEY], calls),
    )

    assert observed == HOST_KEY
    assert len(calls) == 2
    for argv in calls:
        assert "-N" in argv
        assert "-oPreferredAuthentications=none" in argv
        assert "-oPubkeyAuthentication=no" in argv
        assert "-oPasswordAuthentication=no" in argv
        assert "-oHostKeyAlgorithms=ssh-ed25519" in argv
        proxy = next(item for item in argv if item.startswith("-oProxyCommand="))
        assert "start-iap-tunnel" in proxy
        assert foundation.VM_NAME in proxy
        assert "--listen-on-stdin" in proxy
        assert not any(item.startswith("--command=") for item in argv)


@pytest.mark.parametrize("capability", ["runtime", "ssh"])
def test_iap_capture_rejects_runtime_or_ssh_path_drift(
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)
    calls = 0

    if capability == "runtime":
        stable = runtime.trusted_command_prefix()

        def drifting_runtime(_self: Any) -> tuple[str, ...]:
            nonlocal calls
            calls += 1
            return stable if calls < 3 else (*stable[:-1], "/attacker/gcloud.py")

        monkeypatch.setattr(
            launcher.TrustedGcloudExecutable,
            "trusted_command_prefix",
            drifting_runtime,
        )
    else:
        stable_paths = ssh.paths()

        def drifting_ssh(_self: Any) -> tuple[str, str]:
            nonlocal calls
            calls += 1
            return (
                stable_paths
                if calls < 3
                else ("/attacker/ssh", stable_paths[1])
            )

        monkeypatch.setattr(
            host.TrustedOwnerGateSshExecutable,
            "paths",
            drifting_ssh,
        )

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_runtime_changed",
    ):
        host._capture_host_key_once(
            runtime=runtime,
            configuration=configuration,
            ssh=ssh,
            instance_id=INSTANCE_ID,
            runner=_runner_for_keys([HOST_KEY], []),
        )


def test_iap_capture_rechecks_gcloud_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)
    checked = False

    def changed_configuration(_self: Any) -> None:
        nonlocal checked
        checked = True
        raise launcher.OwnerLauncherError("test_configuration_changed")

    monkeypatch.setattr(
        launcher.PinnedGcloudConfiguration,
        "assert_stable",
        changed_configuration,
    )
    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_runtime_changed",
    ):
        host._capture_host_key_once(
            runtime=runtime,
            configuration=configuration,
            ssh=ssh,
            instance_id=INSTANCE_ID,
            runner=_runner_for_keys([HOST_KEY], []),
        )
    assert checked is True


def test_iap_capture_rejects_host_key_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)
    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_host_key_changed",
    ):
        host._stable_iap_host_key(
            runtime=runtime,
            configuration=configuration,
            ssh=ssh,
            instance_id=INSTANCE_ID,
            runner=_runner_for_keys([HOST_KEY, SECOND_HOST_KEY], []),
        )


@pytest.mark.parametrize(
    "line",
    [
        f"compute.{INSTANCE_ID} ssh-rsa AAAA\n",
        (
            f"compute.{INSTANCE_ID} ssh-ed25519 {HOST_KEY}\n"
            f"compute.{INSTANCE_ID} ssh-ed25519 {SECOND_HOST_KEY}\n"
        ),
    ],
)
def test_iap_capture_rejects_non_ed25519_or_extra_key(
    monkeypatch: pytest.MonkeyPatch,
    line: str,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)

    def runner(argv: tuple[str, ...], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        path_arg = next(
            item for item in argv if item.startswith("-oUserKnownHostsFile=")
        )
        Path(path_arg.split("=", 1)[1]).write_text(line, encoding="ascii")
        return subprocess.CompletedProcess(argv, 255, b"", b"")

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_known_hosts_invalid",
    ):
        host._capture_host_key_once(
            runtime=runtime,
            configuration=configuration,
            ssh=ssh,
            instance_id=INSTANCE_ID,
            runner=runner,
        )


def test_exact_capability_boundaries_reject_subclasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)

    class RuntimeSubclass(launcher.TrustedGcloudExecutable):
        pass

    attacker = object.__new__(RuntimeSubclass)
    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_capability_invalid",
    ):
        host._iap_ssh_argv(
            runtime=attacker,
            configuration=configuration,
            ssh=ssh,
            known_hosts_path="/tmp/known_hosts",
            instance_id=INSTANCE_ID,
        )

    assert type(runtime) is launcher.TrustedGcloudExecutable


def test_v3_owner_receipt_requires_distinct_pinning_revision(
    tmp_path: Path,
    collection_authorization: host._CollectionAuthorization,
) -> None:
    signer, _key = _owner_signer(tmp_path)
    identity = host._direct_compute_identity(_responses(), chain=_chain())
    receipt = host._author_receipt(
        chain=_chain(),
        collection_authorization=collection_authorization,
        identity=identity,
        host_key_base64=HOST_KEY,
        direct_observed_before_unix=100,
        host_key_observed_at_unix=101,
        direct_observed_after_unix=102,
        sealed_runtime_identity_sha256="9" * 64,
        toolchain_identity={
            "ssh_executable_sha256": "a" * 64,
            "ssh_version": "OpenSSH_10.2p1, LibreSSL 3.3.6",
            "shell_executable_sha256": "b" * 64,
            "shell_version": "3.2.57(1)-release",
        },
        owner_signer=signer,
    )
    path = tmp_path / "owner-gate-host-identity-v3.json"
    path.write_bytes(
        host.canonical_receipt_bytes(receipt, owner_signer=signer)
    )
    os.chmod(path, 0o600)

    pinned = launcher.PinnedOwnerGateHostIdentityReceipt(
        path=path,
        expected_receipt_sha256=receipt["receipt_sha256"],
        pinning_source_revision=PINNING_REVISION,
        owner_signer=signer,
        collection_reauthentication_public_key=(
            _COLLECTION_REAUTH_KEY.public_key()
        ),
    )
    assert pinned.snapshot().vm_numeric_id == INSTANCE_ID

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_identity_receipt_invalid",
    ):
        launcher.PinnedOwnerGateHostIdentityReceipt(
            path=path,
            expected_receipt_sha256=receipt["receipt_sha256"],
            pinning_source_revision=FOUNDATION_REVISION,
            owner_signer=signer,
            collection_reauthentication_public_key=(
                _COLLECTION_REAUTH_KEY.public_key()
            ),
        )


def test_candidate_receipt_is_canonical_chain_bound_and_verified(
    tmp_path: Path,
    collection_authorization: host._CollectionAuthorization,
) -> None:
    signer, _key = _owner_signer(tmp_path)
    identity = host._direct_compute_identity(_responses(), chain=_chain())
    receipt = host._author_receipt(
        chain=_chain(),
        collection_authorization=collection_authorization,
        identity=identity,
        host_key_base64=HOST_KEY,
        direct_observed_before_unix=100,
        host_key_observed_at_unix=101,
        direct_observed_after_unix=102,
        sealed_runtime_identity_sha256="9" * 64,
        toolchain_identity={
            "ssh_executable_sha256": "a" * 64,
            "ssh_version": "OpenSSH_10.2p1, LibreSSL 3.3.6",
            "shell_executable_sha256": "b" * 64,
            "shell_version": "3.2.57(1)-release",
        },
        owner_signer=signer,
    )
    expected = host.canonical_receipt_bytes(receipt, owner_signer=signer)
    assert host._decode_candidate_receipt(
        expected,
        chain=_chain(),
        collection_runtime_release_revision=FOUNDATION_REVISION,
        release_public_key=_COLLECTION_REAUTH_KEY.public_key(),
        owner_signer=signer,
    ) == receipt

    tampered = dict(receipt)
    tampered["vm_numeric_id"] = "7777777777777777777"
    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_receipt_invalid",
    ):
        host.canonical_receipt_bytes(tampered, owner_signer=signer)
    mismatched = _chain(vm_numeric_id="7777777777777777777")
    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_receipt_chain_mismatch",
    ):
        host._decode_candidate_receipt(
            expected,
            chain=mismatched,
            collection_runtime_release_revision=FOUNDATION_REVISION,
            release_public_key=_COLLECTION_REAUTH_KEY.public_key(),
            owner_signer=signer,
        )
    assert "publish_canonical_receipt_exclusive" not in host.__all__
    assert not hasattr(host, "publish_canonical_receipt_exclusive")


@pytest.mark.parametrize("checkpoint", ["after_candidate", "after_final_link"])
def test_signed_host_candidate_or_final_replays_without_recollection(
    tmp_path: Path,
    checkpoint: str,
    collection_authorization: host._CollectionAuthorization,
) -> None:
    signer, _key = _owner_signer(tmp_path)
    chain = _chain()
    identity = host._direct_compute_identity(_responses(), chain=chain)
    receipt = host._author_receipt(
        chain=chain,
        collection_authorization=collection_authorization,
        identity=identity,
        host_key_base64=HOST_KEY,
        direct_observed_before_unix=100,
        host_key_observed_at_unix=101,
        direct_observed_after_unix=102,
        sealed_runtime_identity_sha256="9" * 64,
        toolchain_identity={
            "ssh_executable_sha256": "a" * 64,
            "ssh_version": "OpenSSH_10.2p1, LibreSSL 3.3.6",
            "shell_executable_sha256": "b" * 64,
            "shell_version": "3.2.57(1)-release",
        },
        owner_signer=signer,
    )
    raw = host.canonical_receipt_bytes(receipt, owner_signer=signer)
    owner_home = tmp_path / "owner-home"
    trusted = owner_home / ".hermes" / "trusted"
    trusted.mkdir(parents=True, mode=0o700)
    os.chmod(trusted, 0o700)
    os.chown(trusted, -1, os.getegid())
    owner_key_id = signer.inspect().key_id

    def validate(value: bytes) -> source_publication._ValidatedArtifact:
        decoded = host._decode_candidate_receipt(
            value,
            chain=chain,
            collection_runtime_release_revision=FOUNDATION_REVISION,
            release_public_key=_COLLECTION_REAUTH_KEY.public_key(),
            owner_signer=signer,
        )
        return source_publication._ValidatedArtifact(
            value=decoded,
            logical_sha256=str(decoded["receipt_sha256"]),
        )

    class StopSeed(BaseException):
        pass

    def stop(name: str) -> None:
        if name == checkpoint:
            raise StopSeed

    publication_chain = host._host_publication_chain(
        chain,
        collection_runtime_release_revision=FOUNDATION_REVISION,
        owner_public_key_id=owner_key_id,
    )
    with pytest.raises(StopSeed):
        source_publication._run_host_identity_v3(
            owner_home=owner_home,
            chain=publication_chain,
            maximum=launcher.PinnedOwnerGateHostIdentityReceipt._MAX_BYTES,
            validator=validate,
            collector=lambda: raw,
            _checkpoint=stop,
        )
    replay = source_publication._run_host_identity_v3(
        owner_home=owner_home,
        chain=publication_chain,
        maximum=launcher.PinnedOwnerGateHostIdentityReceipt._MAX_BYTES,
        validator=validate,
        collector=lambda: pytest.fail("host evidence was recollected"),
        _recovery_only=True,
    )

    assert replay.raw == raw
    assert replay.value == receipt
    assert replay.value["signature_sshsig"] == receipt["signature_sshsig"]
    assert replay.value["direct_observed_before_unix"] == 100
    final = owner_home / source_publication._HOST_V3_RELATIVE
    assert final.read_bytes() == raw
    assert stat_mode(final) == 0o600
    assert final.stat().st_nlink == 1


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def test_compute_json_rejects_duplicate_keys() -> None:
    token = host._WipeableAccessToken("opaque-token")

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_compute_duplicate_key",
    ):
        host._read_compute_snapshot(
            chain=_chain(),
            access_token=token,
            requester=lambda _url, _headers, _timeout: (
                200,
                b'{"id":"123456","id":"654321"}',
            ),
        )
    token.wipe()
    assert token.retired is True


def test_compute_json_rejects_nonfinite_constant() -> None:
    token = host._WipeableAccessToken("opaque-token")

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_compute_invalid",
    ):
        host._read_compute_snapshot(
            chain=_chain(),
            access_token=token,
            requester=lambda _url, _headers, _timeout: (
                200,
                b'{"id":NaN}',
            ),
        )
    token.wipe()
    assert token.retired is True


class _ComputeHttpResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._body = body
        self._url = url
        self.status = status
        self.headers = dict(headers or {})

    def __enter__(self) -> "_ComputeHttpResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


class _ComputeHttpOpener:
    def __init__(self, response: _ComputeHttpResponse) -> None:
        self.response = response
        self.requests: list[Any] = []

    def open(self, request: Any, *, timeout: float) -> _ComputeHttpResponse:
        assert timeout == 30.0
        self.requests.append(request)
        return self.response


def _clear_compute_network_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in host._FORBIDDEN_NETWORK_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def test_default_compute_request_uses_fixed_tls_and_strict_no_redirect_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_compute_network_environment(monkeypatch)
    url = next(iter(host._compute_urls(_chain()).values()))
    body = b'{"id":"1234567890123456789"}'
    response = _ComputeHttpResponse(
        body,
        url=url,
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": str(len(body)),
        },
    )
    opener = _ComputeHttpOpener(response)
    context = object()
    handlers: list[Any] = []
    monkeypatch.setattr(
        launcher,
        "_pinned_system_tls_context",
        lambda: context,
    )

    def build_opener(*values: Any) -> _ComputeHttpOpener:
        handlers.extend(values)
        return opener

    monkeypatch.setattr(host.urllib.request, "build_opener", build_opener)

    status, payload = host._default_compute_request(
        url,
        {
            "Accept": "application/json",
            "Authorization": "Bearer opaque-token",
        },
        30.0,
    )

    assert status == 200
    assert payload == body
    assert len(opener.requests) == 1
    assert any(isinstance(item, host._NoRedirectHandler) for item in handlers)
    assert any(
        isinstance(item, urllib.request.ProxyHandler) for item in handlers
    )
    https = next(
        item
        for item in handlers
        if isinstance(item, urllib.request.HTTPSHandler)
    )
    assert https._context is context


@pytest.mark.parametrize(
    "name",
    [
        "HTTPS_PROXY",
        "SSL_CERT_DIR",
        "SSLKEYLOGFILE",
        "OPENSSL_CONF",
        "OPENSSL_MODULES",
    ],
)
def test_default_compute_request_rejects_ambient_network_influence(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    _clear_compute_network_environment(monkeypatch)
    monkeypatch.setenv(name, "/tmp/attacker-controlled")
    monkeypatch.setattr(
        host.urllib.request,
        "build_opener",
        lambda *_handlers: pytest.fail("network must not start"),
    )
    url = next(iter(host._compute_urls(_chain()).values()))

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_compute_tls_invalid",
    ):
        host._default_compute_request(
            url,
            {
                "Accept": "application/json",
                "Authorization": "Bearer opaque-token",
            },
            30.0,
        )


@pytest.mark.parametrize(
    ("status", "final_url", "headers", "body"),
    [
        (302, "https://attacker.invalid/", {"Content-Type": "application/json"}, b"{}"),
        (200, "https://attacker.invalid/", {"Content-Type": "application/json"}, b"{}"),
        (200, None, {"Content-Type": "text/html"}, b"<html></html>"),
        (
            200,
            None,
            {"Content-Type": "application/json", "Content-Length": "1"},
            b"{}",
        ),
        pytest.param(
            200,
            None,
            {"Content-Type": "application/json"},
            b"{" + b"x" * host.MAX_COMPUTE_RESPONSE_BYTES + b"}",
            id="oversize",
        ),
    ],
)
def test_default_compute_request_rejects_redirect_type_length_and_oversize(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    final_url: str | None,
    headers: Mapping[str, str],
    body: bytes,
) -> None:
    _clear_compute_network_environment(monkeypatch)
    url = next(iter(host._compute_urls(_chain()).values()))
    response = _ComputeHttpResponse(
        body,
        url=final_url or url,
        status=status,
        headers=headers,
    )
    monkeypatch.setattr(
        launcher,
        "_pinned_system_tls_context",
        lambda: object(),
    )
    monkeypatch.setattr(
        host.urllib.request,
        "build_opener",
        lambda *_handlers: _ComputeHttpOpener(response),
    )

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_compute_invalid",
    ):
        host._default_compute_request(
            url,
            {
                "Accept": "application/json",
                "Authorization": "Bearer opaque-token",
            },
            30.0,
        )


def test_malicious_owner_token_subclass_is_rejected_before_io(
    monkeypatch: pytest.MonkeyPatch,
    collection_authorization: host._CollectionAuthorization,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)

    class TokenSubclass(launcher.GcloudOwnerAccessToken):
        pass

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_capability_invalid",
    ):
        host._collect_with_capabilities(
            chain=_chain(),
            collection_authorization=collection_authorization,
            runtime=runtime,
            configuration=configuration,
            owner_identity=object.__new__(TokenSubclass),
            ssh=ssh,
            owner_signer=object(),  # type: ignore[arg-type]
            compute_requester=lambda *_args: pytest.fail("no Compute"),
            ssh_runner=lambda *_args, **_kwargs: pytest.fail("no process"),
            clock=lambda: 100.0,
        )


def test_token_is_wiped_and_all_capabilities_rechecked_on_compute_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collection_authorization: host._CollectionAuthorization,
) -> None:
    runtime, configuration, ssh = _exact_capabilities(monkeypatch)
    owner = object.__new__(launcher.GcloudOwnerAccessToken)
    owner._gcloud_configuration = configuration
    owner._gcloud_executable = runtime
    signer, _key = _owner_signer(tmp_path)
    checks = {"runtime": 0, "configuration": 0, "ssh": 0, "wipe": 0}

    def sealed_identity(_self: Any, *, expected_release_sha: str) -> Mapping[str, Any]:
        assert expected_release_sha == FOUNDATION_REVISION
        checks["runtime"] += 1
        return {"identity_sha256": "9" * 64}

    def bind(subject: launcher.GcloudOwnerAccessToken, expected: str) -> None:
        assert expected == hashlib.sha256(host.OWNER_ACCOUNT.encode()).hexdigest()
        subject._approved = True
        subject._pinned_account = host.OWNER_ACCOUNT
        subject.owner_subject_sha256 = expected

    def stable_configuration(_self: Any) -> None:
        checks["configuration"] += 1

    def ssh_identity(_self: Any) -> Mapping[str, str]:
        checks["ssh"] += 1
        return {
            "ssh_executable_sha256": "a" * 64,
            "ssh_version": "OpenSSH_10.2p1, LibreSSL 3.3.6",
            "shell_executable_sha256": "b" * 64,
            "shell_version": "3.2.57(1)-release",
        }

    original_wipe = host._WipeableAccessToken.wipe

    def tracked_wipe(token: host._WipeableAccessToken) -> None:
        checks["wipe"] += 1
        original_wipe(token)

    monkeypatch.setattr(
        launcher.TrustedGcloudExecutable,
        "sealed_runtime_identity",
        sealed_identity,
    )
    monkeypatch.setattr(
        launcher.PinnedGcloudConfiguration,
        "assert_stable",
        stable_configuration,
    )
    monkeypatch.setattr(
        host.TrustedOwnerGateSshExecutable,
        "identity",
        ssh_identity,
    )
    monkeypatch.setattr(launcher.GcloudOwnerAccessToken, "bind_approved_subject", bind)
    monkeypatch.setattr(
        launcher.GcloudOwnerAccessToken,
        "__call__",
        lambda _self: "opaque-owner-access-token",
    )
    monkeypatch.setattr(
        launcher.GcloudOwnerAccessToken,
        "require_stable",
        lambda _self: None,
    )
    monkeypatch.setattr(host._WipeableAccessToken, "wipe", tracked_wipe)

    with pytest.raises(host.OwnerGateHostIdentityError):
        host._collect_with_capabilities(
            chain=_chain(),
            collection_authorization=collection_authorization,
            runtime=runtime,
            configuration=configuration,
            owner_identity=owner,
            ssh=ssh,
            owner_signer=signer,
            compute_requester=lambda *_args: (_ for _ in ()).throw(
                host.OwnerGateHostIdentityError("forced_compute_failure")
            ),
            ssh_runner=lambda *_args, **_kwargs: pytest.fail("no IAP yet"),
            clock=lambda: 100.0,
        )

    assert checks["wipe"] == 1
    assert checks["runtime"] == 2
    assert checks["configuration"] >= 1
    assert checks["ssh"] == 2


def test_unpinned_consumer_starts_no_process_or_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("process must not start"),
    )
    monkeypatch.setattr(
        urllib.request.OpenerDirector,
        "open",
        lambda *_args, **_kwargs: pytest.fail("Compute must not be called"),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_identity_receipt_unpinned",
    ):
        launcher.PinnedOwnerGateHostIdentityReceipt(
            pinning_source_revision=PINNING_REVISION,
        )


def _owner_cli_inputs(
    tmp_path: Path,
    *,
    release_key: Ed25519PrivateKey,
    collector_key: Ed25519PrivateKey,
) -> tuple[list[str], Mapping[str, bytes]]:
    artifacts = {
        "pre-foundation-authority": b'{"artifact":"pre"}',
        "owner-reauth-receipt": b'{"artifact":"reauth"}',
        "network-evidence": b'{"artifact":"network"}',
        "project-ancestry-evidence": b'{"artifact":"ancestry"}',
        "direct-iam-authority": b'{"artifact":"direct"}',
    }
    argv: list[str] = []
    for name, raw in artifacts.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(raw)
        os.chmod(path, 0o444)
        argv.extend((f"--{name}", str(path)))
    release_path = tmp_path / "release.pub"
    release_path.write_bytes(release_key.public_key().public_bytes_raw())
    os.chmod(release_path, 0o444)
    collector_path = tmp_path / "collector.pub"
    collector_path.write_bytes(collector_key.public_key().public_bytes_raw())
    os.chmod(collector_path, 0o444)
    argv.extend(("--release-trust-public-key", str(release_path)))
    argv.extend(("--network-collector-public-key", str(collector_path)))
    argv.extend(
        ("--project-ancestry-collector-public-key", str(collector_path))
    )
    argv.extend(("--collection-release-revision", FOUNDATION_REVISION))
    return argv, artifacts


def test_host_identity_public_boundary_loads_apply_only_from_fixed_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.canary import owner_gate_foundation_apply as foundation_apply
    from scripts.canary import owner_gate_trust as trust
    from tests.scripts.canary import (
        test_owner_gate_foundation_apply as apply_fixture,
    )
    from tests.scripts.canary import test_owner_gate_pre_foundation as fixture

    parameters = inspect.signature(
        host.collect_and_publish_owner_gate_host_identity_v3
    ).parameters
    assert "foundation_apply_receipt_raw" not in parameters
    assert "foundation_apply_receipt_path" not in parameters
    assert "foundation_chain" not in parameters
    assert "output_path" not in parameters
    assert "candidate_path" not in parameters
    assert "journal_path" not in parameters
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        fixture.RELEASE_KEY_ID,
    )
    chain = apply_fixture._chain()
    os.chmod(tmp_path, 0o700)
    os.chown(tmp_path, -1, os.getegid())
    store = _foundation_journal_for_test(tmp_path / "journal")
    foundation_apply._apply_with_provider(
        chain=chain,
        private_key=fixture.RELEASE_KEY,
        provider=apply_fixture._FakeProvider(chain),
        journal=store,
        now_unix=lambda: fixture.NOW + 2,
    )
    monkeypatch.setattr(
        foundation_apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(
        foundation_apply.time,
        "time",
        lambda: float(fixture.NOW + 3),
    )
    seen: dict[str, Any] = {}

    class StopAfterFixedLoader(BaseException):
        pass

    def stop_after_fixed_loader(**kwargs: Any) -> host._FoundationChainProjection:
        seen.update(kwargs)
        raise StopAfterFixedLoader

    monkeypatch.setattr(
        host,
        "_projection_from_validated_chains",
        stop_after_fixed_loader,
    )

    with pytest.raises(StopAfterFixedLoader):
        host.collect_and_publish_owner_gate_host_identity_v3(
            pre_foundation_authority_raw=chain.pre_foundation_authority_raw,
            owner_reauthentication_receipt_raw=(
                chain.owner_reauthentication_receipt_raw
            ),
            network_evidence_raw=chain.network_evidence_raw,
            project_ancestry_evidence_raw=chain.ancestry_evidence_raw,
            direct_iam_authority_raw=b"{}",
            release_public_key=chain.release_public_key,
            network_collector_public_key=chain.network_collector_public_key,
            project_ancestry_collector_public_key=(
                chain.ancestry_collector_public_key
            ),
            collection_runtime_release_revision=FOUNDATION_REVISION,
        )

    loaded = seen["foundation_chain"]
    assert type(loaded) is foundation_apply.ValidatedFoundationApplyChain
    assert loaded.foundation_a.pre_foundation_authority_raw == (
        chain.pre_foundation_authority_raw
    )
    assert loaded.pre_foundation_authority_sha256 == (
        chain.pre_foundation_authority_sha256
    )
    assert seen["direct_iam_authority_raw"] == b"{}"
    assert seen["now_unix"] == fixture.NOW + 3


def test_owner_only_cli_reads_immutable_inputs_and_prints_only_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.canary import owner_gate_trust as trust

    release_key = Ed25519PrivateKey.generate()
    collector_key = Ed25519PrivateKey.generate()
    argv, artifacts = _owner_cli_inputs(
        tmp_path,
        release_key=release_key,
        collector_key=collector_key,
    )
    assert "--foundation-apply-receipt" not in argv
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        hashlib.sha256(release_key.public_key().public_bytes_raw()).hexdigest(),
    )
    owner_home = tmp_path / "owner-home"
    owner_home.mkdir(mode=0o700)
    monkeypatch.setattr(
        launcher,
        "_canonical_owner_home",
        lambda: str(owner_home),
    )
    expected_output = str(
        owner_home / launcher.OWNER_GATE_HOST_IDENTITY_RECEIPT_RELATIVE
    )
    called = False

    def collect(**kwargs: Any) -> Mapping[str, Any]:
        nonlocal called
        called = True
        assert "foundation_apply_receipt_raw" not in kwargs
        assert kwargs["pre_foundation_authority_raw"] == artifacts[
            "pre-foundation-authority"
        ]
        assert kwargs["owner_reauthentication_receipt_raw"] == artifacts[
            "owner-reauth-receipt"
        ]
        assert kwargs["network_evidence_raw"] == artifacts["network-evidence"]
        assert kwargs["project_ancestry_evidence_raw"] == artifacts[
            "project-ancestry-evidence"
        ]
        assert kwargs["direct_iam_authority_raw"] == artifacts[
            "direct-iam-authority"
        ]
        assert kwargs["release_public_key"].public_bytes_raw() == (
            release_key.public_key().public_bytes_raw()
        )
        assert kwargs["network_collector_public_key"].public_bytes_raw() == (
            collector_key.public_key().public_bytes_raw()
        )
        assert kwargs["collection_runtime_release_revision"] == (
            FOUNDATION_REVISION
        )
        return {
            "receipt": {"must_not_be_printed": "secret-marker"},
            "publication": {
                "path": expected_output,
                "receipt_sha256": "1" * 64,
                "receipt_file_sha256": "2" * 64,
            },
        }

    monkeypatch.setattr(
        host,
        "collect_and_publish_owner_gate_host_identity_v3",
        collect,
    )

    assert host.main(argv) == 0
    assert called is True
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "schema": "muncho-owner-gate-iap-host-identity-publication.v3",
        "receipt_published": True,
        "path": expected_output,
        "receipt_sha256": "1" * 64,
        "receipt_file_sha256": "2" * 64,
    }
    assert "secret-marker" not in json.dumps(report)


@pytest.mark.parametrize("path_attack", ["writable", "symlink"])
def test_owner_only_cli_rejects_mutable_or_aliased_artifact_before_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_attack: str,
) -> None:
    from scripts.canary import owner_gate_trust as trust

    release_key = Ed25519PrivateKey.generate()
    collector_key = Ed25519PrivateKey.generate()
    argv, _artifacts = _owner_cli_inputs(
        tmp_path,
        release_key=release_key,
        collector_key=collector_key,
    )
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        hashlib.sha256(release_key.public_key().public_bytes_raw()).hexdigest(),
    )
    authority_index = argv.index("--pre-foundation-authority") + 1
    authority_path = Path(argv[authority_index])
    if path_attack == "writable":
        os.chmod(authority_path, 0o600)
    else:
        real_path = tmp_path / "pre-authority-real.json"
        real_path.write_bytes(authority_path.read_bytes())
        os.chmod(real_path, 0o444)
        authority_path.unlink()
        authority_path.symlink_to(real_path)
    monkeypatch.setattr(
        host,
        "collect_and_publish_owner_gate_host_identity_v3",
        lambda **_kwargs: pytest.fail("collection must not start"),
    )

    with pytest.raises(
        host.OwnerGateHostIdentityError,
        match="owner_gate_host_identity_owner_input_invalid",
    ):
        host.main(argv)
