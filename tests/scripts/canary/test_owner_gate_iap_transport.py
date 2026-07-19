from __future__ import annotations

import os
import base64
import hashlib
import inspect
import json
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher


RELEASE = "a" * 40
ACCOUNT = "lomliev@adventico.com"
FRAME = b'{"document":{},"operation":"preflight","schema":"test"}'
OWNER_GATE_INSTANCE_ID = "1234567890123456789"
_HOST_ALGORITHM = b"ssh-ed25519"
HOST_KEY_BASE64 = base64.b64encode(
    struct.pack(">I", len(_HOST_ALGORITHM))
    + _HOST_ALGORITHM
    + struct.pack(">I", 32)
    + b"H" * 32
).decode("ascii")
HOST_KEY_LINE = (
    f"compute.{OWNER_GATE_INSTANCE_ID} ssh-ed25519 {HOST_KEY_BASE64}"
)


def _owner_gate_frame(
    operation: str,
    document: Mapping[str, Any],
) -> bytes:
    unsigned = {
        "schema": "muncho-passkey-v2-owner-gate-frame.v1",
        "operation": operation,
        "release_sha": RELEASE,
        "document": dict(document),
    }
    return launcher._canonical_bytes({
        **unsigned,
        "frame_sha256": hashlib.sha256(
            launcher._canonical_bytes(unsigned)
        ).hexdigest(),
    })


class _Executable:
    prefix = (
        "/trusted/python3.11",
        *launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
        "/trusted/google-cloud-sdk/lib/gcloud.py",
    )

    def trusted_command_prefix(self) -> tuple[str, ...]:
        return self.prefix


class _Configuration:
    def __init__(self, *, extra_environment: Mapping[str, str] | None = None) -> None:
        self._extra_environment = dict(extra_environment or {})
        self.stability_checks = 0

    @property
    def account(self) -> str:
        self.assert_stable()
        return ACCOUNT

    def assert_stable(self) -> None:
        self.stability_checks += 1

    def environment_values(self) -> Mapping[str, str]:
        self.assert_stable()
        return {
            "HOME": "/owner",
            "CLOUDSDK_CONFIG": "/owner/.config/gcloud",
            **self._extra_environment,
        }


class _Identity:
    def __init__(self, configuration: _Configuration) -> None:
        self.gcloud_configuration = configuration
        self.stability_checks = 0

    def account_for_read_only_preflight(self) -> str:
        return ACCOUNT

    def require_stable(self) -> None:
        self.stability_checks += 1


class _KnownHosts:
    def absolute_path(self) -> str:
        return "/owner/.ssh/google_compute_known_hosts"

    def private_key_path(self) -> str:
        return "/owner/.ssh/google_compute_engine"

    def public_key_line(self) -> str:
        return "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest"

    def server_host_key_line(self, instance_id: str) -> str:
        assert instance_id == OWNER_GATE_INSTANCE_ID
        return HOST_KEY_LINE


class _HostIdentity:
    def __init__(self) -> None:
        self.calls = 0
        self.value = launcher.OwnerGateHostIdentitySnapshot(
            vm_numeric_id=OWNER_GATE_INSTANCE_ID,
            host_key_algorithm="ssh-ed25519",
            host_key_base64=HOST_KEY_BASE64,
            known_hosts_line=HOST_KEY_LINE,
            receipt_sha256="9" * 64,
            receipt_file_sha256="8" * 64,
        )

    def snapshot(self) -> launcher.OwnerGateHostIdentitySnapshot:
        self.calls += 1
        return self.value


def _passthrough_factory(
    calls: list[tuple[tuple[str, ...], Mapping[str, Any]]],
):
    def factory(argv: tuple[str, ...], **kwargs: Any) -> subprocess.Popen[bytes]:
        calls.append((tuple(argv), dict(kwargs)))
        return subprocess.Popen(("/bin/cat",), **kwargs)

    return factory


def _script_factory(source: str):
    def factory(_argv: tuple[str, ...], **kwargs: Any) -> subprocess.Popen[bytes]:
        return subprocess.Popen((sys.executable, "-I", "-c", source), **kwargs)

    return factory


def _expected_ssh_flags() -> tuple[str, ...]:
    return (
        "--ssh-flag=-F/dev/null",
        "--ssh-flag=-T",
        "--ssh-flag=-i/owner/.ssh/google_compute_engine",
        "--ssh-flag=-oBatchMode=yes",
        "--ssh-flag=-oIdentitiesOnly=yes",
        "--ssh-flag=-oIdentityAgent=none",
        "--ssh-flag=-oCertificateFile=none",
        "--ssh-flag=-oPreferredAuthentications=publickey",
        "--ssh-flag=-oPubkeyAuthentication=yes",
        "--ssh-flag=-oPasswordAuthentication=no",
        "--ssh-flag=-oKbdInteractiveAuthentication=no",
        "--ssh-flag=-oGSSAPIAuthentication=no",
        "--ssh-flag=-oHostbasedAuthentication=no",
        "--ssh-flag=-oPermitLocalCommand=no",
        "--ssh-flag=-oClearAllForwardings=yes",
        "--ssh-flag=-oControlMaster=no",
        "--ssh-flag=-oControlPath=none",
        "--ssh-flag=-oKnownHostsCommand=none",
        "--ssh-flag=-oCanonicalizeHostname=no",
        "--ssh-flag=-oForwardAgent=no",
        "--ssh-flag=-oEscapeChar=none",
        "--ssh-flag=-oRequestTTY=no",
        "--ssh-flag=-oStrictHostKeyChecking=yes",
        f"--ssh-flag=-oHostKeyAlias=compute.{OWNER_GATE_INSTANCE_ID}",
        "--ssh-flag=-oUserKnownHostsFile=/owner/.ssh/google_compute_known_hosts",
        "--ssh-flag=-oGlobalKnownHostsFile=none",
        "--ssh-flag=-oUpdateHostKeys=no",
        "--ssh-flag=-oVerifyHostKeyDNS=no",
        "--ssh-flag=-oServerAliveInterval=15",
        "--ssh-flag=-oServerAliveCountMax=4",
    )


def _transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configuration: _Configuration | None = None,
    popen_factory: Any = None,
    timeout_seconds: float = 5.0,
    host_identity: _HostIdentity | None = None,
    known_hosts: Any = None,
    executable: Any = None,
) -> tuple[
    launcher.OwnerGateIapTransport,
    _Configuration,
    _Identity,
    _HostIdentity,
]:
    selected = configuration or _Configuration()
    identity = _Identity(selected)
    selected_host_identity = host_identity or _HostIdentity()
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda _release: "f" * 64,
    )
    return (
        launcher.OwnerGateIapTransport(
            release_sha=RELEASE,
            owner_identity=identity,
            gcloud_executable=executable or _Executable(),
            gcloud_configuration=selected,
            host_identity=selected_host_identity,
            known_hosts=known_hosts or _KnownHosts(),
            popen_factory=popen_factory or _passthrough_factory([]),
            timeout_seconds=timeout_seconds,
        ),
        selected,
        identity,
        selected_host_identity,
    )


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _sshsig(
    key: Ed25519PrivateKey,
    message: bytes,
    *,
    namespace: str,
) -> str:
    algorithm = b"ssh-ed25519"
    namespace_bytes = namespace.encode("ascii")
    signed = (
        b"SSHSIG"
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(hashlib.sha512(message).digest())
    )
    signature = key.sign(signed)
    public_raw = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    public_blob = _ssh_string(algorithm) + _ssh_string(public_raw)
    signature_blob = _ssh_string(algorithm) + _ssh_string(signature)
    envelope = (
        b"SSHSIG"
        + struct.pack(">I", 1)
        + _ssh_string(public_blob)
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(signature_blob)
    )
    body = base64.b64encode(envelope).decode("ascii")
    lines = [body[index : index + 70] for index in range(0, len(body), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    )


def _owner_signer(
    tmp_path: Path,
) -> tuple[launcher._PhaseBOwnerExternalSigner, Ed25519PrivateKey]:
    key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "owner-key"
    public_path = tmp_path / "owner-key.pub"
    comment = "owner-gate-host-identity-test"
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


def _write_host_identity_receipt(
    path: Path,
    *,
    signer: launcher._PhaseBOwnerExternalSigner,
    key: Ed25519PrivateKey,
    overrides: Mapping[str, Any] | None = None,
    corrupt_signature: bool = False,
) -> str:
    authority = signer.inspect()
    direct_identity = {
        "project": launcher.PROJECT,
        "project_number": launcher.OWNER_GATE_PROJECT_NUMBER,
        "zone": launcher.ZONE,
        "vm_name": "muncho-owner-gate-01",
        "vm_self_link": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{launcher.PROJECT}/zones/{launcher.ZONE}/instances/"
            "muncho-owner-gate-01"
        ),
        "vm_numeric_id": OWNER_GATE_INSTANCE_ID,
        "vm_creation_timestamp": "2026-07-15T10:00:00+00:00",
        "machine_type_self_link": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{launcher.PROJECT}/zones/{launcher.ZONE}/machineTypes/e2-small"
        ),
        "scheduling": {
            "automaticRestart": True,
            "instanceTerminationAction": "DELETE",
            "onHostMaintenance": "MIGRATE",
            "preemptible": False,
            "provisioningModel": "STANDARD",
        },
        "labels": {},
        "resource_policies": [],
        "minimum_cpu_platform": "Automatic",
        "confidential_compute": False,
        "owner_gate_service_account_email": (
            launcher.OWNER_GATE_SERVICE_ACCOUNT_EMAIL
        ),
        "owner_gate_service_account_unique_id": "2234567890123456789",
        "oauth_scopes": [
            "https://www.googleapis.com/auth/cloudplatformfolders.readonly",
            "https://www.googleapis.com/auth/cloudplatformorganizations.readonly",
            "https://www.googleapis.com/auth/cloudplatformprojects.readonly",
            "https://www.googleapis.com/auth/compute",
            "https://www.googleapis.com/auth/iam",
        ],
        "network_tags": ["iap-ssh", "muncho-owner-gate"],
        "instance_metadata": {
            "block-project-ssh-keys": "TRUE",
            "enable-oslogin": "TRUE",
            "serial-port-enable": "FALSE",
        },
        "shielded_instance_config": {
            "enableIntegrityMonitoring": True,
            "enableSecureBoot": True,
            "enableVtpm": True,
        },
        "can_ip_forward": False,
        "external_ip_present": False,
        "internal_ip": "10.80.3.2",
        "network_self_link": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{launcher.PROJECT}/global/networks/ai-platform-vpc"
        ),
        "network_numeric_id": "3234567890123456789",
        "subnetwork_self_link": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{launcher.PROJECT}/regions/europe-west3/subnetworks/"
            "muncho-owner-gate-europe-west3"
        ),
        "subnetwork_numeric_id": "4234567890123456789",
        "boot_disk_self_link": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{launcher.PROJECT}/zones/{launcher.ZONE}/disks/"
            "muncho-owner-gate-01"
        ),
        "boot_disk_numeric_id": "5234567890123456789",
        "boot_disk_type_self_link": (
            "https://www.googleapis.com/compute/v1/projects/"
            f"{launcher.PROJECT}/zones/{launcher.ZONE}/diskTypes/pd-balanced"
        ),
        "boot_disk_size_gb": 20,
        "boot_disk_auto_delete": True,
        "boot_disk_architecture": "X86_64",
        "boot_disk_physical_block_size_bytes": 4096,
        "boot_disk_licenses": [
            "https://www.googleapis.com/compute/v1/projects/"
            "debian-cloud/global/licenses/debian-12-bookworm"
        ],
        "boot_image_self_link": (
            "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
            "global/images/debian-12-bookworm-v20260701"
        ),
        "boot_image_numeric_id": "6234567890123456789",
        "boot_image_architecture": "X86_64",
        "boot_image_licenses": [
            "https://www.googleapis.com/compute/v1/projects/"
            "debian-cloud/global/licenses/debian-12-bookworm"
        ],
    }
    toolchain = {
        "sealed_runtime_identity_sha256": "9" * 64,
        "ssh_executable_sha256": "a" * 64,
        "ssh_version": "OpenSSH_10.2p1, LibreSSL 3.3.6",
        "shell_executable_sha256": "b" * 64,
        "shell_version": "3.2.57(1)-release",
    }
    unsigned = {
        "schema": launcher.OWNER_GATE_HOST_IDENTITY_RECEIPT_SCHEMA,
        "foundation_source_revision": "b" * 40,
        "foundation_source_tree_oid": "c" * 40,
        "owner_reauthentication_receipt_sha256": "1" * 64,
        "pre_foundation_authority_sha256": "2" * 64,
        "foundation_apply_receipt_sha256": "3" * 64,
        "direct_iam_authority_sha256": "4" * 64,
        "ancestry_evidence_sha256": "5" * 64,
        "ancestry_chain_sha256": "6" * 64,
        "signed_network_evidence_sha256": "7" * 64,
        "network_evidence_sha256": "8" * 64,
        **direct_identity,
        "owner_account": ACCOUNT,
        "host_key_algorithm": "ssh-ed25519",
        "host_key_base64": HOST_KEY_BASE64,
        "collection_method": launcher.OWNER_GATE_HOST_IDENTITY_COLLECTION_METHOD,
        "direct_identity_sha256": hashlib.sha256(
            launcher._canonical_bytes(direct_identity)
        ).hexdigest(),
        "direct_observed_before_unix": 100,
        "host_key_observed_at_unix": 101,
        "direct_observed_after_unix": 102,
        "owner_reauthentication_expires_at_unix": 200,
        **toolchain,
        "first_contact_toolchain_sha256": hashlib.sha256(
            launcher._canonical_bytes(toolchain)
        ).hexdigest(),
        "owner_public_key_id": authority.key_id,
        **dict(overrides or {}),
    }
    digest = hashlib.sha256(launcher._canonical_bytes(unsigned)).hexdigest()
    signed = {**unsigned, "receipt_sha256": digest}
    signature = _sshsig(
        key,
        launcher._canonical_bytes(signed),
        namespace=launcher.OWNER_GATE_HOST_IDENTITY_SSHSIG_NAMESPACE,
    )
    if corrupt_signature:
        signature = signature.replace("A", "B", 1)
    receipt = {**signed, "signature_sshsig": signature}
    path.write_bytes(launcher._canonical_bytes(receipt) + b"\n")
    os.chmod(path, 0o600)
    return digest


def test_transport_has_one_fixed_iap_stdin_command_and_closed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    transport, configuration, identity, host_identity = _transport(
        monkeypatch,
        popen_factory=_passthrough_factory(calls),
    )

    assert transport.invoke_owner_gate(FRAME) == FRAME
    assert len(calls) == 1
    argv, invocation = calls[0]
    expected_command = shlex.join((
        "/usr/bin/sudo",
        "--non-interactive",
        "--user=muncho-passkey-authority",
        "--",
        "/opt/muncho-owner-gate/current/venv/bin/python",
        "-I",
        "-B",
        "/opt/muncho-owner-gate/current/bin/muncho-owner-gate-intake",
    ))
    assert argv == (
        *_Executable.prefix,
        "compute",
        "ssh",
        f"{launcher.OS_LOGIN_USERNAME}@muncho-owner-gate-01",
        "--project=adventico-ai-platform",
        "--zone=europe-west3-a",
        f"--account={ACCOUNT}",
        "--plain",
        "--tunnel-through-iap",
        "--quiet",
        f"--command={expected_command}",
        *_expected_ssh_flags(),
    )
    assert not any(item in {"bash", "sh", "resize", "start", "stop"} for item in argv)
    assert invocation["shell"] is False
    assert invocation["start_new_session"] is True
    assert invocation["stderr"] is subprocess.DEVNULL
    environment = invocation["env"]
    assert set(environment) == launcher.OwnerGateIapTransport._ENVIRONMENT_KEYS
    assert not any("proxy" in name.casefold() for name in environment)
    assert environment["CLOUDSDK_CORE_PROJECT"] == "adventico-ai-platform"
    assert environment["CLOUDSDK_COMPUTE_ZONE"] == "europe-west3-a"
    assert configuration.stability_checks >= 5
    assert identity.stability_checks == 2
    assert host_identity.calls == 3
    assert not hasattr(transport, "run_local_compute_mutation")
    assert not hasattr(transport, "run")


def _activation_response(
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": launcher.OWNER_GATE_ACTIVATION_RESPONSE_SCHEMA,
        "release_revision": RELEASE,
        "disposition": "installed",
        "activation_seal_path": launcher.OWNER_GATE_ACTIVATION_SEAL_PATH,
        "activation_seal_sha256": "a" * 64,
        "activation_receipt_path": (
            f"{launcher.OWNER_GATE_ACTIVATION_RECEIPT_BASE}/{RELEASE}.json"
        ),
        "activation_receipt_sha256": "b" * 64,
        "service_contract_accepted": True,
        "cloud_mutation_performed": False,
        **dict(overrides or {}),
    }
    return {
        **unsigned,
        "response_sha256": hashlib.sha256(
            launcher._canonical_bytes(unsigned)
        ).hexdigest(),
    }


def test_activation_action_has_one_fixed_release_result_and_root_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    raw = launcher._canonical_bytes(_activation_response()) + b"\n"
    request = launcher._canonical_bytes({
        "schema": launcher.OWNER_GATE_ACTIVATION_REQUEST_SCHEMA,
        "release_revision": RELEASE,
    })
    factory = _script_factory(
        "import os,sys;"
        "received=sys.stdin.buffer.read();"
        f"sys.exit(9) if received!={request!r} else None;"
        f"os.write(1,{raw!r})"
    )

    def capture(
        argv: tuple[str, ...],
        **kwargs: Any,
    ) -> subprocess.Popen[bytes]:
        calls.append((tuple(argv), dict(kwargs)))
        return factory(argv, **kwargs)

    transport, configuration, identity, host_identity = _transport(
        monkeypatch,
        popen_factory=capture,
    )

    response = transport.install_activation_seal()

    assert response == _activation_response()
    assert len(calls) == 1
    argv, invocation = calls[0]
    expected_command = shlex.join((
        "/usr/bin/sudo",
        "--non-interactive",
        "--user=root",
        "--",
        "/opt/muncho-owner-gate/current/venv/bin/python",
        "-I",
        "-B",
        (
            "/opt/muncho-owner-gate/current/bin/"
            "muncho-owner-gate-activate-storage"
        ),
        "install",
    ))
    assert argv == (
        *_Executable.prefix,
        "compute",
        "ssh",
        f"{launcher.OS_LOGIN_USERNAME}@muncho-owner-gate-01",
        "--project=adventico-ai-platform",
        "--zone=europe-west3-a",
        f"--account={ACCOUNT}",
        "--plain",
        "--tunnel-through-iap",
        "--quiet",
        f"--command={expected_command}",
        *_expected_ssh_flags(),
    )
    assert invocation["shell"] is False
    assert invocation["start_new_session"] is True
    assert invocation["stderr"] is subprocess.DEVNULL
    assert set(invocation["env"]) == launcher.OwnerGateIapTransport._ENVIRONMENT_KEYS
    assert configuration.stability_checks >= 5
    assert identity.stability_checks == 2
    assert host_identity.calls == 3
    assert tuple(
        inspect.signature(
            launcher.OwnerGateIapTransport.install_activation_seal
        ).parameters
    ) == ("self",)


@pytest.mark.parametrize(
    "overrides",
    (
        {"schema": "wrong"},
        {"release_revision": "b" * 40},
        {"disposition": "forced"},
        {"activation_seal_path": "/tmp/seal"},
        {"activation_seal_sha256": "x" * 64},
        {"activation_receipt_path": "/tmp/receipt"},
        {"activation_receipt_sha256": "x" * 64},
        {"service_contract_accepted": False},
        {"cloud_mutation_performed": True},
    ),
)
def test_activation_result_rejects_every_release_or_receipt_mismatch(
    overrides: Mapping[str, Any],
) -> None:
    response = _activation_response(overrides=overrides)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_activation_iap_response_invalid",
    ):
        launcher.validate_owner_gate_activation_response(
            response,
            expected_release_sha=RELEASE,
        )


def test_activation_result_rejects_extra_field_and_wrong_self_hash() -> None:
    extra = {**_activation_response(), "caller_command": "/bin/sh"}
    wrong_hash = {**_activation_response(), "response_sha256": "0" * 64}
    for response in (extra, wrong_hash):
        with pytest.raises(
            launcher.OwnerLauncherError,
            match="owner_gate_activation_iap_response_invalid",
        ):
            launcher.validate_owner_gate_activation_response(
                response,
                expected_release_sha=RELEASE,
            )


def test_activation_transport_rejects_nonexact_ssh_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_passthrough_factory(calls),
    )
    monkeypatch.setattr(
        launcher.OwnerGateIapTransport,
        "_sealed_ssh_flags",
        staticmethod(lambda *_args: ("--ssh-flag=-oPermitLocalCommand=yes",)),
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_activation_iap_argv_invalid",
    ):
        transport.install_activation_seal()
    assert calls == []


def test_activation_factory_accepts_no_host_path_or_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _Configuration()
    identity = _Identity(configuration)
    executable = _Executable()
    captured: dict[str, Any] = {}
    expected = _activation_response(overrides={"disposition": "exact_replay"})

    class Transport:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def install_activation_seal(self) -> Mapping[str, Any]:
            return expected

    monkeypatch.setattr(launcher, "OwnerGateIapTransport", Transport)

    result = launcher.install_owner_gate_activation_seal(
        release_sha=RELEASE,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
    )

    assert result == expected
    assert captured == {
        "release_sha": RELEASE,
        "owner_identity": identity,
        "gcloud_executable": executable,
        "gcloud_configuration": configuration,
    }
    assert tuple(
        inspect.signature(launcher.install_owner_gate_activation_seal).parameters
    ) == (
        "release_sha",
        "gcloud_executable",
        "gcloud_configuration",
        "owner_identity",
    )



def test_production_host_identity_gate_is_deliberately_unpinned() -> None:
    assert launcher.OWNER_GATE_HOST_IDENTITY_RECEIPT_SHA256 is None
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_identity_receipt_unpinned",
    ):
        launcher.PinnedOwnerGateHostIdentityReceipt()


def test_signed_pinned_owner_gate_identity_binds_numeric_id_and_host_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signer, key = _owner_signer(tmp_path)
    path = tmp_path / "owner-gate-identity.json"
    digest = _write_host_identity_receipt(
        path,
        signer=signer,
        key=key,
    )
    import builtins

    real_import = builtins.__import__

    def import_without_gateway(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "gateway" or name.startswith("gateway."):
            raise AssertionError(
                "owner-gate identity verification must not import gateway"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gateway)

    receipt = launcher.PinnedOwnerGateHostIdentityReceipt(
        path=path,
        expected_receipt_sha256=digest,
        pinning_source_revision=RELEASE,
        owner_signer=signer,
    )

    snapshot = receipt.snapshot()
    assert snapshot.vm_numeric_id == OWNER_GATE_INSTANCE_ID
    assert snapshot.known_hosts_line == HOST_KEY_LINE
    assert snapshot.receipt_sha256 == digest
    assert snapshot.receipt_file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "overrides",
    [
        {"project": "attacker-project"},
        {"zone": "attacker-zone"},
        {"vm_name": "muncho-canary-v2-01"},
        {"vm_numeric_id": launcher.VM_INSTANCE_ID},
        {"vm_numeric_id": "7"},
        {"vm_creation_timestamp": "2026-07-15T::::::::"},
        {"owner_account": "attacker@example.com"},
        {"host_key_algorithm": "ssh-rsa"},
        {"host_key_base64": "AAAA"},
        {"owner_public_key_id": "0" * 64},
    ],
)
def test_signed_owner_gate_identity_rejects_wrong_target_or_key(
    tmp_path: Path,
    overrides: Mapping[str, Any],
) -> None:
    signer, key = _owner_signer(tmp_path)
    path = tmp_path / "owner-gate-identity.json"
    digest = _write_host_identity_receipt(
        path,
        signer=signer,
        key=key,
        overrides=overrides,
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_identity_receipt_invalid",
    ):
        launcher.PinnedOwnerGateHostIdentityReceipt(
            path=path,
            expected_receipt_sha256=digest,
            pinning_source_revision=RELEASE,
            owner_signer=signer,
        )


def test_owner_gate_identity_rejects_wrong_pin_and_signature(tmp_path: Path) -> None:
    signer, key = _owner_signer(tmp_path)
    path = tmp_path / "owner-gate-identity.json"
    digest = _write_host_identity_receipt(path, signer=signer, key=key)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_identity_receipt_invalid",
    ):
        launcher.PinnedOwnerGateHostIdentityReceipt(
            path=path,
            expected_receipt_sha256="0" * 64,
            pinning_source_revision=RELEASE,
            owner_signer=signer,
        )

    digest = _write_host_identity_receipt(
        path,
        signer=signer,
        key=key,
        corrupt_signature=True,
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_identity_receipt_signature_invalid",
    ):
        launcher.PinnedOwnerGateHostIdentityReceipt(
            path=path,
            expected_receipt_sha256=digest,
            pinning_source_revision=RELEASE,
            owner_signer=signer,
        )


def test_transport_rejects_known_host_not_bound_to_signed_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongKnownHosts(_KnownHosts):
        def server_host_key_line(self, instance_id: str) -> str:
            assert instance_id == OWNER_GATE_INSTANCE_ID
            return HOST_KEY_LINE.replace(HOST_KEY_BASE64, "AAAA")

    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        known_hosts=WrongKnownHosts(),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_host_key_mismatch",
    ):
        transport.invoke_owner_gate(FRAME)


def test_transport_does_not_inherit_ambient_proxy_or_python_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("PYTHONPATH", "/tmp/escape")
    calls: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_passthrough_factory(calls),
    )

    assert transport.invoke_owner_gate(FRAME) == FRAME
    environment = calls[0][1]["env"]
    assert "HTTPS_PROXY" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_transport_rejects_configuration_environment_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _Configuration(
        extra_environment={"HTTPS_PROXY": "http://127.0.0.1:9"}
    )
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        configuration=configuration,
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_environment_invalid",
    ):
        transport.invoke_owner_gate(FRAME)


@pytest.mark.parametrize(
    "extra",
    [
        ("--tunnel-through-iap",),
        ("--project=attacker-project",),
        ("--zone=attacker-zone",),
        ("--account=attacker@example.com",),
        ("--command=/bin/sh",),
        ("--ssh-flag=-oProxyCommand=/bin/sh",),
        ("--ssh-flag=-oPermitLocalCommand=yes",),
        tuple(reversed(_expected_ssh_flags())),
        _expected_ssh_flags()[:-1],
    ],
)
def test_transport_rejects_any_nonexact_ssh_argv(
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
) -> None:
    transport, _configuration, _identity, _host_identity = _transport(monkeypatch)
    monkeypatch.setattr(
        launcher.OwnerGateIapTransport,
        "_sealed_ssh_flags",
        staticmethod(lambda _known, _key, _instance: extra),
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_argv_invalid",
    ):
        transport.invoke_owner_gate(FRAME)


@pytest.mark.live_system_guard_bypass
def test_transport_kills_oversized_stdout_before_accepting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_count = (
        launcher.OwnerGateIapTransport._MAX_RESPONSE_BYTES // 65536
    ) + 1
    source = (
        "import os;"
        "os.read(0, 2_000_000);"
        "chunk=b'x'*65536;"
        f"[(os.write(1,chunk)) for _ in range({chunk_count})]"
    )
    processes: list[subprocess.Popen[bytes]] = []
    factory = _script_factory(source)

    def capture(argv: tuple[str, ...], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = factory(argv, **kwargs)
        processes.append(process)
        return process

    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=capture,
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_response_oversized",
    ):
        transport.invoke_owner_gate(FRAME)
    assert len(processes) == 1
    assert processes[0].poll() is not None


@pytest.mark.live_system_guard_bypass
def test_transport_fails_closed_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "import os,time; os.read(0,2_000_000); time.sleep(10)"
    processes: list[subprocess.Popen[bytes]] = []
    factory = _script_factory(source)

    def capture(argv: tuple[str, ...], **kwargs: Any) -> subprocess.Popen[bytes]:
        process = factory(argv, **kwargs)
        processes.append(process)
        return process

    transport, _configuration, _identity, host_identity = _transport(
        monkeypatch,
        popen_factory=capture,
        timeout_seconds=0.1,
    )

    with pytest.raises(launcher.OwnerLauncherError, match="owner_gate_iap_timeout"):
        transport.invoke_owner_gate(FRAME)
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert host_identity.calls == 3


def test_transport_reaps_when_process_group_signal_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stream:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Process:
        pid = 123456

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdin = Stream()
            self.stdout = Stream()
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.returncode = -signal.SIGKILL

        def wait(self, timeout: float | None = None) -> int:
            assert timeout == 5.0
            if self.terminated:
                self.returncode = -signal.SIGTERM
            assert self.returncode is not None
            return self.returncode

    import signal

    monkeypatch.setattr(
        launcher.os,
        "killpg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("group signal unavailable")
        ),
    )
    process = Process()

    launcher.OwnerGateIapTransport._terminate_process(process)  # type: ignore[arg-type]

    assert process.returncode == -signal.SIGTERM
    assert process.stdin.closed is True
    assert process.stdout.closed is True


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (
            "import os; os.read(0,2_000_000)",
            "owner_gate_iap_remote_failed",
        ),
        (
            "import os,sys; os.read(0,2_000_000); os.write(1,b'{}'); sys.exit(7)",
            "owner_gate_iap_remote_failed",
        ),
        (
            "import os; os.read(0,2_000_000); os.write(1,b'{\\\"a\\\": 1}')",
            "owner_gate_iap_response_invalid",
        ),
        (
            "import os; os.read(0,2_000_000); os.write(1,b'{\\\"a\\\":1}\\n')",
            "owner_gate_iap_response_invalid",
        ),
        (
            "import os; os.read(0,2_000_000); os.write(1,b'{}{}')",
            "owner_gate_iap_response_invalid",
        ),
    ],
)
def test_transport_rejects_empty_nonzero_or_noncanonical_response(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    error: str,
) -> None:
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_script_factory(source),
    )
    with pytest.raises(launcher.OwnerLauncherError, match=error):
        transport.invoke_owner_gate(FRAME)


def test_transport_concurrently_exchanges_near_limit_canonical_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = b'{"data":"' + b"r" * 900_000 + b'"}'
    response = b'{"data":"' + b"s" * 700_000 + b'"}'
    # Generate the frame in the child: embedding it in ``python -c`` exceeds
    # Linux MAX_ARG_STRLEN before the transport itself can be exercised.
    source = (
        "import os;"
        "payload=b'{\"data\":\"'+b's'*700_000+b'\"}';"
        "offset=0;"
        "\nwhile offset < len(payload):\n"
        " offset += os.write(1,payload[offset:offset+65536])\n"
        "\nwhile os.read(0,65536):\n pass\n"
    )
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_script_factory(source),
        timeout_seconds=10.0,
    )
    assert transport.invoke_owner_gate(request) == response


def test_transport_handles_partial_nonblocking_stdin_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = b'{"data":"' + b"p" * 128_000 + b'"}'
    original_write = launcher.os.write
    writes = 0

    def partial_write(fd: int, value: Any) -> int:
        nonlocal writes
        writes += 1
        return original_write(fd, value[:17])

    monkeypatch.setattr(launcher.os, "write", partial_write)
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_passthrough_factory([]),
    )
    assert transport.invoke_owner_gate(frame) == frame
    assert writes > 100


def test_transport_accepts_live_sized_full_iam_request_only_for_exact_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.canary import direct_iam_identity_author as direct_author

    document = {
        "trusted_iam_projection": {
            "canonical_role_inventory": "p" * (
                direct_author.MAX_HTTP_BODY_BYTES + 64 * 1024
            )
        }
    }
    frame = _owner_gate_frame("request_initial", document)
    response = launcher._canonical_bytes({"ok": True})
    assert (
        direct_author.MAX_HTTP_BODY_BYTES
        < len(frame)
        <= launcher.OwnerGateIapTransport._MAX_FRAME_BYTES
    )
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_script_factory(
            "import sys;sys.stdin.buffer.read();"
            f"sys.stdout.buffer.write({response!r})"
        ),
    )

    assert transport.invoke_owner_gate(frame) == response

    calls: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    generic, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_passthrough_factory(calls),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="^owner_gate_iap_frame_invalid$",
    ):
        generic.invoke_owner_gate(_owner_gate_frame("preflight", document))
    assert calls == []


def test_transport_accepts_live_sized_full_iam_response_only_for_exact_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = launcher._canonical_bytes({
        "trusted_iam_projection": "p" * (
            launcher.OwnerGateIapTransport._DEFAULT_RESPONSE_BYTES
            + 64 * 1024
        )
    })
    frame = _owner_gate_frame("attest_cloud_observation", {})
    response_overhead = len(b'{"trusted_iam_projection":""}')
    padding_bytes = len(response) - response_overhead
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_script_factory(
            "import sys;sys.stdin.buffer.read();"
            "sys.stdout.buffer.write(b'{\"trusted_iam_projection\":\"'"
            f" + b'p'*{padding_bytes}"
            " + b'\"}')"
        ),
    )

    assert transport.invoke_owner_gate(frame) == response


def test_host_attestor_transport_carries_live_sized_iam_projection() -> None:
    from scripts.canary import direct_iam_identity_author as direct_author
    from scripts.canary import storage_growth_trusted_collector as trusted

    request = {
        "schema": trusted.ATTESTATION_REQUEST_SCHEMA,
        "role": "host",
        "trusted_iam_projection": "p" * (
            direct_author.MAX_HTTP_BODY_BYTES + 64 * 1024
        ),
    }
    response = {
        "schema": trusted.ATTESTATION_RESPONSE_SCHEMA,
        "role": "host",
        "trusted_iam_projection": request["trusted_iam_projection"],
    }
    frame = trusted.protocol.canonical_json_bytes(request)
    response_raw = trusted.protocol.canonical_json_bytes(response) + b"\n"
    calls: list[Mapping[str, Any]] = []

    class Remote:
        def _run_remote_input(self, _argv: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            assert kwargs["input_bytes"] == frame + b"\n"
            return type("Completed", (), {"stdout": response_raw})()

    identity = _Identity(_Configuration())
    transport = object.__new__(
        launcher.CanaryHostObservationAttestorTransport
    )
    transport._release_sha = RELEASE
    transport._owner_identity = identity
    transport._account = ACCOUNT
    transport._transport = Remote()
    transport._remote_interpreter = "/trusted/python"
    transport._remote_entrypoint = "/trusted/host-attestor"

    assert (
        direct_author.MAX_HTTP_BODY_BYTES
        < len(frame)
        <= transport._MAX_FRAME_BYTES
    )
    assert transport.attest_host_observation(frame) == response
    assert calls == [{
        "account": ACCOUNT,
        "input_bytes": frame + b"\n",
        "timeout_seconds": 90.0,
        "maximum_input_bytes": transport._MAX_FRAME_BYTES + 1,
        "maximum_output_bytes": transport._MAX_FRAME_BYTES + 1,
    }]
    assert identity.stability_checks == 1


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b'{"a": 1}',
        b'{"a":1}\n',
        b"x" * (launcher.OwnerGateIapTransport._MAX_FRAME_BYTES + 1),
    ],
)
def test_transport_rejects_noncanonical_or_unbounded_input_before_process(
    monkeypatch: pytest.MonkeyPatch,
    frame: bytes,
) -> None:
    calls: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        popen_factory=_passthrough_factory(calls),
    )

    with pytest.raises(launcher.OwnerLauncherError, match="owner_gate_iap_frame_invalid"):
        transport.invoke_owner_gate(frame)
    assert calls == []


def test_transport_rejects_local_authority_drift_after_remote_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def provenance(_release: str) -> str:
        nonlocal calls
        calls += 1
        return ("a" if calls == 1 else "b") * 64

    transport, _configuration, _identity, _host_identity = _transport(monkeypatch)
    monkeypatch.setattr(launcher, "require_local_launcher_provenance", provenance)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_authority_changed",
    ):
        transport.invoke_owner_gate(FRAME)


def test_transport_rejects_gcloud_prefix_drift_after_remote_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DriftingExecutable(_Executable):
        def __init__(self) -> None:
            self.calls = 0

        def trusted_command_prefix(self) -> tuple[str, ...]:
            self.calls += 1
            if self.calls == 1:
                return self.prefix
            return (
                self.prefix[0],
                *self.prefix[1:-1],
                "/different/trusted-sdk/lib/gcloud.py",
            )

    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        executable=DriftingExecutable(),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_authority_changed",
    ):
        transport.invoke_owner_gate(FRAME)


def test_transport_rejects_pinned_configuration_account_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DriftingConfiguration(_Configuration):
        def __init__(self) -> None:
            super().__init__()
            self.account_calls = 0

        @property
        def account(self) -> str:
            self.account_calls += 1
            return ACCOUNT if self.account_calls == 1 else "attacker@example.com"

    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        configuration=DriftingConfiguration(),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_owner_identity_invalid",
    ):
        transport.invoke_owner_gate(FRAME)


def test_transport_rejects_known_hosts_material_drift_after_remote_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DriftingKnownHosts(_KnownHosts):
        def __init__(self) -> None:
            self.public_calls = 0

        def public_key_line(self) -> str:
            self.public_calls += 1
            return f"ssh-ed25519 client-key-{self.public_calls}"

    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        known_hosts=DriftingKnownHosts(),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_authority_changed",
    ):
        transport.invoke_owner_gate(FRAME)


def test_transport_rejects_signed_host_identity_drift_after_remote_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DriftingHostIdentity(_HostIdentity):
        def snapshot(self) -> launcher.OwnerGateHostIdentitySnapshot:
            self.calls += 1
            if self.calls < 3:
                return self.value
            return launcher.OwnerGateHostIdentitySnapshot(
                vm_numeric_id=self.value.vm_numeric_id,
                host_key_algorithm=self.value.host_key_algorithm,
                host_key_base64=self.value.host_key_base64,
                known_hosts_line=self.value.known_hosts_line,
                receipt_sha256="7" * 64,
                receipt_file_sha256=self.value.receipt_file_sha256,
            )

    transport, _configuration, _identity, _host_identity = _transport(
        monkeypatch,
        host_identity=DriftingHostIdentity(),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_authority_changed",
    ):
        transport.invoke_owner_gate(FRAME)


def test_factory_instantiates_explicit_boundary_with_dedicated_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _Configuration()
    identity = _Identity(configuration)
    executable = _Executable()
    captured: dict[str, Any] = {}

    class Boundary:
        def __init__(self, release_sha: str, transport: Any) -> None:
            captured["release_sha"] = release_sha
            captured["transport"] = transport

        def require_ready(self) -> Mapping[str, Any]:
            captured["ready"] = True
            return {"ok": True}

    from scripts.canary import passkey_v2_storage_growth

    dedicated_transport = object()

    def transport_factory(**kwargs: Any) -> object:
        captured["transport_kwargs"] = kwargs
        return dedicated_transport

    monkeypatch.setattr(
        passkey_v2_storage_growth,
        "ProductionStorageGrowthBoundary",
        Boundary,
    )
    monkeypatch.setattr(
        launcher,
        "OwnerGateIapTransport",
        transport_factory,
    )

    boundary = launcher._require_storage_growth_privileged_boundary(
        release_sha=RELEASE,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
    )

    assert boundary is not None
    assert captured["release_sha"] == RELEASE
    assert captured["transport"] is dedicated_transport
    assert "ready" not in captured
    assert captured["transport_kwargs"] == {
        "release_sha": RELEASE,
        "owner_identity": identity,
        "gcloud_executable": executable,
        "gcloud_configuration": configuration,
    }


def test_factory_defers_readiness_until_terminal_state_is_ruled_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _Configuration()
    identity = _Identity(configuration)
    executable = _Executable()
    constructed = 0

    class Boundary:
        def __init__(self, _release_sha: str, _transport: Any) -> None:
            nonlocal constructed
            constructed += 1

        def require_ready(self) -> Mapping[str, Any]:
            raise RuntimeError("not ready")

    from scripts.canary import passkey_v2_storage_growth

    monkeypatch.setattr(
        passkey_v2_storage_growth,
        "ProductionStorageGrowthBoundary",
        Boundary,
    )
    monkeypatch.setattr(
        launcher,
        "OwnerGateIapTransport",
        lambda **_kwargs: object(),
    )
    boundary = launcher._require_storage_growth_privileged_boundary(
        release_sha=RELEASE,
        gcloud_executable=executable,
        gcloud_configuration=configuration,
        owner_identity=identity,
    )
    assert isinstance(boundary, Boundary)
    assert constructed == 1


def test_real_factory_stops_at_unpinned_host_identity_before_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _Configuration()
    identity = _Identity(configuration)

    def forbidden_process(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("unpinned owner-gate identity must precede process start")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden_process)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_identity_receipt_unpinned",
    ):
        launcher._require_storage_growth_privileged_boundary(
            release_sha=RELEASE,
            gcloud_executable=_Executable(),
            gcloud_configuration=configuration,
            owner_identity=identity,
        )


def test_owner_gate_remote_paths_are_absolute_and_not_workspace_controlled() -> None:
    assert Path(launcher.OwnerGateIapTransport._REMOTE_PYTHON).is_absolute()
    assert Path(launcher.OwnerGateIapTransport._REMOTE_INTAKE).is_absolute()
    assert "/tmp/" not in launcher.OwnerGateIapTransport._REMOTE_PYTHON
    assert "/tmp/" not in launcher.OwnerGateIapTransport._REMOTE_INTAKE
