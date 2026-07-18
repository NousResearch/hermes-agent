from __future__ import annotations

import hashlib
import inspect
import io
import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_foundation_journal as foundation_journal
from scripts.canary import owner_gate_outer_stage0 as outer
from scripts.canary import owner_gate_stage0_iap as transport_module
from scripts.canary import owner_gate_trust as release_trust
from tests.scripts.canary import test_owner_gate_foundation_apply as apply_fixture
from tests.scripts.canary import test_owner_gate_pre_foundation as foundation_fixture


ROOT = Path(__file__).parents[3]
REVISION = "a" * 40
TREE = "b" * 40
INSTANCE_ID = "1234567890123456789"


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


def _foundation_projection() -> transport_module._FoundationProjection:
    return transport_module._FoundationProjection(
        pre_foundation_authority_sha256="4" * 64,
        foundation_apply_receipt_sha256="5" * 64,
        project_ancestry_evidence_sha256="9" * 64,
        project_ancestry_chain_sha256="a" * 64,
        resource_ancestor_chain=("organizations/123456789012",),
        interpreter_sha256="8" * 64,
    )


def _decode_foundation_a(
    *,
    owner_expires_at_unix: int = foundation_fixture.NOW + 300,
) -> foundation_apply.ValidatedFoundationAChain:
    authority, _plan, _evidence = foundation_fixture._authority(
        expires_at_unix=owner_expires_at_unix,
        owner_expires_at_unix=owner_expires_at_unix,
    )
    return foundation_apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=foundation.canonical_json_bytes(authority),
        owner_reauthentication_receipt_raw=foundation.canonical_json_bytes(
            foundation_fixture._owner_reauth_receipt(
                expires_at_unix=owner_expires_at_unix
            )
        ),
        network_evidence_raw=foundation.canonical_json_bytes(
            foundation_fixture._signed_network_evidence()
        ),
        project_ancestry_evidence_raw=(
            foundation_fixture._signed_ancestry_raw(
                owner_expires_at_unix=owner_expires_at_unix
            )
        ),
        release_public_key=foundation_fixture.RELEASE_KEY.public_key(),
        network_collector_public_key=(
            foundation_fixture.NETWORK_KEY.public_key()
        ),
        project_ancestry_collector_public_key=(
            foundation_fixture.NETWORK_KEY.public_key()
        ),
        now_unix=foundation_fixture.NOW + 1,
    )


def _write_foundation_artifacts(
    tmp_path: Path,
    chain: foundation_apply.ValidatedFoundationAChain,
    *,
    prefix: str = "foundation",
) -> transport_module.RawFoundationChainArtifacts:
    paths_and_raw = {
        "pre_foundation_authority_path": chain.pre_foundation_authority_raw,
        "owner_reauthentication_receipt_path": (
            chain.owner_reauthentication_receipt_raw
        ),
        "network_evidence_path": chain.network_evidence_raw,
        "network_collector_public_key_path": (
            chain.network_collector_public_key.public_bytes_raw()
        ),
        "project_ancestry_evidence_path": chain.ancestry_evidence_raw,
        "project_ancestry_collector_public_key_path": (
            chain.ancestry_collector_public_key.public_bytes_raw()
        ),
        "release_public_key_path": chain.release_public_key.public_bytes_raw(),
    }
    paths: dict[str, Path] = {}
    for name, raw in paths_and_raw.items():
        path = tmp_path / f"{prefix}-{name}"
        path.write_bytes(raw)
        path.chmod(0o444)
        paths[name] = path
    return transport_module.RawFoundationChainArtifacts(**paths)


def _prepare_foundation_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    transport_module.RawFoundationChainArtifacts,
    foundation_apply.ValidatedFoundationAChain,
]:
    monkeypatch.setattr(
        release_trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        foundation_fixture.RELEASE_KEY_ID,
    )
    chain = _decode_foundation_a()
    store = _foundation_journal_for_test(tmp_path / "journal")
    foundation_apply._apply_with_provider(
        chain=chain,
        private_key=foundation_fixture.RELEASE_KEY,
        provider=apply_fixture._FakeProvider(chain),
        journal=store,
        now_unix=lambda: foundation_fixture.NOW + 2,
    )
    monkeypatch.setattr(
        foundation_apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(
        foundation_apply.time,
        "time",
        lambda: float(foundation_fixture.NOW + 3),
    )
    return _write_foundation_artifacts(tmp_path, chain), chain


class _Sealer:
    def __init__(self) -> None:
        self.payload = (
            ROOT / "scripts/canary/owner_gate_outer_stage0.py"
        ).read_bytes()
        self.sha256 = hashlib.sha256(self.payload).hexdigest()

    def snapshot(self) -> tuple[bytes, str]:
        return self.payload, self.sha256


class _HostIdentity:
    def __init__(self) -> None:
        self.value = launcher.OwnerGateHostIdentitySnapshot(
            vm_numeric_id=INSTANCE_ID,
            host_key_algorithm="ssh-ed25519",
            host_key_base64="unused",
            known_hosts_line=(
                f"compute.{INSTANCE_ID} ssh-ed25519 unused"
            ),
            receipt_sha256="c" * 64,
            receipt_file_sha256="d" * 64,
        )

    def snapshot(self) -> launcher.OwnerGateHostIdentitySnapshot:
        return self.value


class _Configuration:
    account = "lomliev@adventico.com"

    def assert_stable(self) -> None:
        return None


class _OwnerIdentity:
    def __init__(self, configuration: _Configuration) -> None:
        self.gcloud_configuration = configuration

    def account_for_read_only_preflight(self) -> str:
        return self.gcloud_configuration.account

    def require_stable(self) -> None:
        return None


class _Gcloud:
    def trusted_command_prefix(self) -> tuple[str, ...]:
        return (
            "/trusted/python",
            *launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
            "/trusted/google-cloud-sdk/lib/gcloud.py",
        )


class _KnownHosts:
    def absolute_path(self) -> str:
        return "/trusted/google_compute_known_hosts"

    def private_key_path(self) -> str:
        return "/trusted/google_compute_engine"

    def public_key_line(self) -> str:
        return "ssh-ed25519 unused"

    def server_host_key_line(self, instance_id: str) -> str:
        assert instance_id == INSTANCE_ID
        return f"compute.{INSTANCE_ID} ssh-ed25519 unused"


def _streams(tmp_path: Path) -> tuple[
    transport_module.PinnedExactTreeStream,
    transport_module.PinnedExactTreeStream,
]:
    kit_manifest = outer.build_manifest(
        ROOT,
        release_revision=REVISION,
        source_tree_oid=TREE,
    )
    kit_manifest_sha256 = hashlib.sha256(
        outer.canonical_json_bytes(kit_manifest)
    ).hexdigest()
    kit = tmp_path / "kit"
    outer.materialize_kit(
        ROOT,
        kit,
        release_revision=REVISION,
        source_tree_oid=TREE,
    )
    kit_stream_path = tmp_path / "kit.stream"
    kit_build = outer.write_tree_stream(
        kit,
        kit_stream_path,
        purpose="outer-stage0-kit",
        release_id=kit_manifest_sha256,
    )

    bundle = tmp_path / "bundle"
    (bundle / "payload/bin").mkdir(parents=True)
    (bundle / "trust").mkdir()
    executable = bundle / "payload/bin/fixed"
    executable.write_bytes(b"#!/bin/false\n")
    executable.chmod(0o755)
    authority = bundle / "trust/release-trust.json"
    authority.write_bytes(b'{"signed":"later-stage0-verifies"}\n')
    authority.chmod(0o644)
    bundle_stream_path = tmp_path / "bundle.stream"
    bundle_build = outer.write_tree_stream(
        bundle,
        bundle_stream_path,
        purpose="owner-gate-bundle",
        release_id=REVISION,
    )
    return (
        transport_module.PinnedExactTreeStream(
            kit_stream_path,
            purpose="outer-stage0-kit",
            release_id=kit_manifest_sha256,
            expected_manifest_sha256=kit_build["stream_manifest_sha256"],
        ),
        transport_module.PinnedExactTreeStream(
            bundle_stream_path,
            purpose="owner-gate-bundle",
            release_id=REVISION,
            expected_manifest_sha256=bundle_build["stream_manifest_sha256"],
        ),
    )


def _transport(
    *,
    kit_stream: transport_module.PinnedExactTreeStream,
    bundle_stream: transport_module.PinnedExactTreeStream,
    wrong_receiver_receipt: bool = False,
) -> tuple[
    transport_module.OwnerGateStage0IapTransport,
    list[tuple[tuple[str, ...], bytes]],
]:
    sealer = _Sealer()
    host_identity = _HostIdentity()
    prefix = (
        "/trusted/python",
        *launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
        "/trusted/google-cloud-sdk/lib/gcloud.py",
    )
    snapshot = (
        prefix,
        "lomliev@adventico.com",
        "e" * 64,
        "/trusted/google_compute_known_hosts",
        "/trusted/google_compute_engine",
        "ssh-ed25519 unused",
        host_identity.value,
        host_identity.value.known_hosts_line,
    )
    calls: list[tuple[tuple[str, ...], bytes]] = []
    foundation_projection = _foundation_projection()

    def exchange(
        argv,
        _environment,
        input_source,
        **_kwargs,
    ) -> transport_module._ProcessResult:
        payload = input_source.read()
        calls.append((tuple(argv), payload))
        command_arg = next(item for item in argv if item.startswith("--command="))
        remote = shlex.split(command_arg.removeprefix("--command="))[3:]
        if remote[0] == "/usr/bin/readlink":
            stdout = b"python3.11\n"
        elif remote[0] == "/usr/bin/stat":
            stdout = (
                b"symbolic link|0|0|777|1\n"
                if remote[-1] == "/usr/bin/python3"
                else b"regular file|0|0|755|1\n"
            )
        elif remote[:2] == ["/usr/bin/python3", "--version"]:
            stdout = b"Python 3.11.2\n"
        elif remote[0] == "/usr/bin/sha256sum":
            selected = remote[-1]
            digest = (
                foundation_projection.interpreter_sha256
                if selected == "/usr/bin/python3.11"
                else sealer.sha256
            )
            stdout = f"{digest}  {selected}\n".encode("ascii")
        elif "stream-receive" in remote:
            purpose = remote[remote.index("--purpose") + 1]
            stream = kit_stream if purpose == "outer-stage0-kit" else bundle_stream
            receipt = transport_module.expected_tree_receipt(
                stream,
                receiver_self_sha256=sealer.sha256,
            )
            if wrong_receiver_receipt:
                receipt = {**receipt, "receipt_sha256": "0" * 64}
            stdout = outer.canonical_json_bytes(receipt) + b"\n"
        elif "seal" in remote:
            stdout = outer.canonical_json_bytes(
                transport_module.expected_seal_receipt(
                    kit_stream,
                    receiver_self_sha256=sealer.sha256,
                )
            ) + b"\n"
        else:
            stdout = b""
        return transport_module._ProcessResult(0, stdout, b"")

    transport = object.__new__(transport_module.OwnerGateStage0IapTransport)
    transport._release_sha = REVISION
    transport._timeout_seconds = 900.0
    transport._popen_factory = object()
    transport._stage0_sealer_source = sealer
    transport._stage0_exchange = exchange
    transport._foundation = foundation_projection
    transport._host_identity = host_identity
    transport._authority_snapshot = lambda: snapshot
    transport._environment = lambda _prefix: {"PINNED": "1"}
    return transport, calls


def test_fixed_iap_transport_materializes_and_receives_without_scp(
    tmp_path: Path,
) -> None:
    kit_stream, bundle_stream = _streams(tmp_path)
    transport, calls = _transport(
        kit_stream=kit_stream,
        bundle_stream=bundle_stream,
    )

    receipt = transport.transport_exact_stage0_and_bundle(
        kit_stream=kit_stream,
        bundle_stream=bundle_stream,
    )

    assert receipt["schema"] == transport_module.TRANSPORT_RECEIPT_SCHEMA
    assert receipt["recursive_scp_used"] is False
    assert receipt["caller_controlled_remote_command_used"] is False
    assert receipt["cloud_control_plane_mutation_performed"] is False
    assert receipt["host_filesystem_materialization_performed"] is True
    assert receipt["project_ancestry_evidence_sha256"] == "9" * 64
    assert receipt["project_ancestry_chain_sha256"] == "a" * 64
    assert receipt["resource_ancestor_chain"] == [
        "organizations/123456789012"
    ]
    commands = [
        shlex.split(
            next(item for item in argv if item.startswith("--command=")).removeprefix(
                "--command="
            )
        )[3:]
        for argv, _payload in calls
    ]
    assert commands
    assert all(command[0] != "scp" for command in commands)
    assert all(command[:3] != ["/bin/sh", "-c", "scp"] for command in commands)
    assert all("--project=adventico-ai-platform" in argv for argv, _ in calls)
    assert all("--zone=europe-west3-a" in argv for argv, _ in calls)
    assert all("--account=lomliev@adventico.com" in argv for argv, _ in calls)
    assert all(
        f"{launcher.OS_LOGIN_USERNAME}@muncho-owner-gate-01" in argv
        for argv, _ in calls
    )
    payloads = [payload for _argv, payload in calls if payload]
    assert payloads == [
        _Sealer().payload,
        Path(kit_stream.path).read_bytes(),
        Path(bundle_stream.path).read_bytes(),
    ]


def test_fixed_iap_transport_rejects_remote_receipt_drift(tmp_path: Path) -> None:
    kit_stream, bundle_stream = _streams(tmp_path)
    transport, _calls = _transport(
        kit_stream=kit_stream,
        bundle_stream=bundle_stream,
        wrong_receiver_receipt=True,
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_receive_outer_stage0_kit_failed",
    ):
        transport.transport_exact_stage0_and_bundle(
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )


def test_pinned_stream_rejects_symlink_or_trailing_bytes(tmp_path: Path) -> None:
    kit_stream, _bundle_stream = _streams(tmp_path)
    path = Path(kit_stream.path)
    symlink = tmp_path / "stream-link"
    symlink.symlink_to(path)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_stream_identity_invalid",
    ):
        transport_module.PinnedExactTreeStream(
            symlink,
            purpose=kit_stream.purpose,
            release_id=kit_stream.release_id,
            expected_manifest_sha256=kit_stream.expected_manifest_sha256,
        )

    path.chmod(0o600)
    with path.open("ab") as stream:
        stream.write(b"trailing")
    path.chmod(0o400)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_stream_identity_invalid",
    ):
        transport_module.PinnedExactTreeStream(
            path,
            purpose=kit_stream.purpose,
            release_id=kit_stream.release_id,
            expected_manifest_sha256=kit_stream.expected_manifest_sha256,
        )


def test_pinned_stream_second_open_rejects_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit_stream, _bundle_stream = _streams(tmp_path)
    path = Path(kit_stream.path)
    original_open = transport_module.os.open
    target_opens = 0

    def racing_open(selected, flags, *args):
        nonlocal target_opens
        if os.path.abspath(os.fspath(selected)) == str(path):
            target_opens += 1
            if target_opens == 2:
                raw = path.read_bytes()
                displaced = path.with_suffix(".displaced")
                path.replace(displaced)
                path.write_bytes(raw)
                path.chmod(0o400)
        return original_open(selected, flags, *args)

    monkeypatch.setattr(transport_module.os, "open", racing_open)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_stream_changed",
    ):
        kit_stream.open()


def test_bounded_exchange_has_local_stdin_e2e() -> None:
    payload = b"exact fixed stdin\x00binary\n"
    result = transport_module._bounded_process_exchange(
        ("/bin/cat",),
        {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        io.BytesIO(payload),
        maximum_input_bytes=len(payload),
        timeout_seconds=5.0,
    )

    assert result == transport_module._ProcessResult(0, payload, b"")


def test_transport_exposes_fixed_operations_without_generic_command_surface() -> None:
    transport = object.__new__(transport_module.OwnerGateStage0IapTransport)

    assert callable(transport.invoke_owner_gate)
    assert callable(transport.transport_exact_stage0_and_bundle)
    for generic_name in ("execute", "remote_command", "run_remote_command"):
        with pytest.raises(AttributeError):
            getattr(transport, generic_name)


def test_raw_foundation_seam_carries_only_unique_absolute_artifact_paths(
    tmp_path: Path,
) -> None:
    names = (
        "pre-foundation.json",
        "owner-reauth.json",
        "network-evidence.json",
        "network.pub",
        "ancestry.json",
        "ancestry.pub",
        "release.pub",
    )
    artifacts = transport_module.RawFoundationChainArtifacts(
        *tuple(tmp_path / name for name in names)
    )
    assert all(
        isinstance(getattr(artifacts, field), Path)
        for field in artifacts.__dataclass_fields__
    )
    assert not hasattr(artifacts, "validated")
    assert not hasattr(artifacts, "foundation_chain")
    assert not hasattr(artifacts, "foundation_apply_receipt_path")
    assert not hasattr(artifacts, "direct_iam_identity_authority_path")

    paths = [tmp_path / name for name in names]
    paths[-1] = paths[0]
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_raw_foundation_artifacts_invalid",
    ):
        transport_module.RawFoundationChainArtifacts(*paths)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_raw_foundation_artifacts_invalid",
    ):
        transport_module.RawFoundationChainArtifacts(
            Path("relative"),
            *tuple(tmp_path / name for name in names[1:]),
        )


def test_transport_actual_init_accepts_only_raw_fixed_journal_foundation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, chain = _prepare_foundation_artifacts(tmp_path, monkeypatch)
    configuration = _Configuration()
    identity = _OwnerIdentity(configuration)
    transport = transport_module.OwnerGateStage0IapTransport(
        release_sha=REVISION,
        owner_identity=identity,
        gcloud_executable=_Gcloud(),
        gcloud_configuration=configuration,
        foundation_artifacts=artifacts,
        sealer_source=_Sealer(),
        host_identity=_HostIdentity(),
        known_hosts=_KnownHosts(),
        exchange=lambda *_args, **_kwargs: transport_module._ProcessResult(
            0, b"", b""
        ),
    )

    assert transport._release_sha == REVISION
    assert transport._foundation.pre_foundation_authority_sha256 == (
        chain.pre_foundation_authority_sha256
    )
    parameters = inspect.signature(
        transport_module.OwnerGateStage0IapTransport
    ).parameters
    assert "foundation_artifacts" in parameters
    assert "foundation_chain" not in parameters
    assert "foundation_apply_receipt_raw" not in parameters
    assert "foundation_apply_receipt_path" not in parameters
    assert "journal" not in parameters
    assert "now_unix" not in parameters
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_iap_configuration_not_shared",
    ):
        transport_module.OwnerGateStage0IapTransport(
            release_sha=REVISION,
            owner_identity=identity,
            gcloud_executable=_Gcloud(),
            gcloud_configuration=_Configuration(),
            foundation_artifacts=artifacts,
            sealer_source=_Sealer(),
            host_identity=_HostIdentity(),
            known_hosts=_KnownHosts(),
        )


def test_transport_rejects_tampered_raw_foundation_before_iap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, _chain = _prepare_foundation_artifacts(tmp_path, monkeypatch)
    authority_path = artifacts.pre_foundation_authority_path
    authority_raw = authority_path.read_bytes()
    authority_path.chmod(0o600)
    authority_path.write_bytes(authority_raw + b"\n")
    authority_path.chmod(0o444)
    configuration = _Configuration()
    identity = _OwnerIdentity(configuration)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_foundation_chain_invalid",
    ):
        transport_module.OwnerGateStage0IapTransport(
            release_sha=REVISION,
            owner_identity=identity,
            gcloud_executable=_Gcloud(),
            gcloud_configuration=configuration,
            foundation_artifacts=artifacts,
            sealer_source=_Sealer(),
            host_identity=_HostIdentity(),
            known_hosts=_KnownHosts(),
            exchange=lambda *_args, **_kwargs: (
                transport_module._ProcessResult(0, b"", b"")
            ),
        )


@pytest.mark.parametrize(
    "attack",
    ("writable", "symlink", "hardlink", "collector-key", "release-pin"),
)
def test_transport_rejects_mutable_aliased_or_substituted_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    artifacts, _chain = _prepare_foundation_artifacts(tmp_path, monkeypatch)
    selected = artifacts.network_collector_public_key_path
    if attack == "writable":
        artifacts.network_evidence_path.chmod(0o600)
    elif attack == "symlink":
        raw = selected.read_bytes()
        target = tmp_path / "network-key-target"
        target.write_bytes(raw)
        target.chmod(0o444)
        selected.unlink()
        selected.symlink_to(target)
    elif attack == "hardlink":
        os.link(selected, tmp_path / "network-key-hardlink")
    elif attack == "collector-key":
        selected.chmod(0o600)
        selected.write_bytes(
            Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        )
        selected.chmod(0o444)
    else:
        monkeypatch.setattr(
            release_trust,
            "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
            "0" * 64,
        )
    configuration = _Configuration()

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_foundation_chain_invalid",
    ):
        transport_module.OwnerGateStage0IapTransport(
            release_sha=REVISION,
            owner_identity=_OwnerIdentity(configuration),
            gcloud_executable=_Gcloud(),
            gcloud_configuration=configuration,
            foundation_artifacts=artifacts,
            sealer_source=_Sealer(),
            host_identity=_HostIdentity(),
            known_hosts=_KnownHosts(),
        )


def test_transport_never_uses_success_journal_for_distinct_foundation_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        foundation_fixture.RELEASE_KEY_ID,
    )
    foundation_a = _decode_foundation_a()
    foundation_b = _decode_foundation_a(
        owner_expires_at_unix=foundation_fixture.NOW + 301
    )
    assert foundation_apply._transaction_id(foundation_a) != (
        foundation_apply._transaction_id(foundation_b)
    )
    store = _foundation_journal_for_test(tmp_path / "journal-b")
    foundation_apply._apply_with_provider(
        chain=foundation_b,
        private_key=foundation_fixture.RELEASE_KEY,
        provider=apply_fixture._FakeProvider(foundation_b),
        journal=store,
        now_unix=lambda: foundation_fixture.NOW + 2,
    )
    monkeypatch.setattr(
        foundation_apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(
        foundation_apply.time,
        "time",
        lambda: float(foundation_fixture.NOW + 3),
    )
    artifacts_a = _write_foundation_artifacts(
        tmp_path,
        foundation_a,
        prefix="foundation-a",
    )
    configuration = _Configuration()

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_foundation_chain_invalid",
    ):
        transport_module.OwnerGateStage0IapTransport(
            release_sha=REVISION,
            owner_identity=_OwnerIdentity(configuration),
            gcloud_executable=_Gcloud(),
            gcloud_configuration=configuration,
            foundation_artifacts=artifacts_a,
            sealer_source=_Sealer(),
            host_identity=_HostIdentity(),
            known_hosts=_KnownHosts(),
        )


def _minimal_transport(
    result: transport_module._ProcessResult,
    *,
    changing_authority: bool = False,
) -> transport_module.OwnerGateStage0IapTransport:
    transport = object.__new__(transport_module.OwnerGateStage0IapTransport)
    host = _HostIdentity().value
    prefix = (
        "/trusted/python",
        *launcher._GCLOUD_PYTHON_ISOLATION_ARGS,
        "/trusted/google-cloud-sdk/lib/gcloud.py",
    )
    stable = (
        prefix,
        "lomliev@adventico.com",
        "e" * 64,
        "/trusted/known-hosts",
        "/trusted/key",
        "ssh-ed25519 unused",
        host,
        host.known_hosts_line,
    )
    calls = 0

    def authority():
        nonlocal calls
        calls += 1
        return (*stable[:2], f"{calls if changing_authority else 'e' * 64}", *stable[3:])

    transport._authority_snapshot = authority
    transport._environment = lambda _prefix: {"PINNED": "1"}
    transport._stage0_exchange = lambda *_args, **_kwargs: result
    transport._popen_factory = object()
    return transport


def test_fixed_operation_rejects_authority_change() -> None:
    transport = _minimal_transport(
        transport_module._ProcessResult(0, b"", b""),
        changing_authority=True,
    )
    operation = transport_module._FixedOperation(
        "probe", ("/usr/bin/true",), b"", 0, 5.0
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_authority_changed",
    ):
        transport._execute_empty(operation)


def _interpreter_attestation_transport(
    *,
    drift: str | None = None,
) -> tuple[
    transport_module.OwnerGateStage0IapTransport,
    list[tuple[tuple[str, ...], bytes]],
]:
    transport = _minimal_transport(
        transport_module._ProcessResult(0, b"", b"")
    )
    foundation_projection = _foundation_projection()
    transport._foundation = foundation_projection
    calls: list[tuple[tuple[str, ...], bytes]] = []
    digest_calls = 0

    def exchange(argv, _environment, input_source, **_kwargs):
        nonlocal digest_calls
        payload = input_source.read()
        command_arg = next(item for item in argv if item.startswith("--command="))
        remote = tuple(
            shlex.split(command_arg.removeprefix("--command="))[3:]
        )
        calls.append((remote, payload))
        if remote[0] == "/usr/bin/readlink":
            stdout = b"python3.12\n" if drift == "link_target" else b"python3.11\n"
        elif remote[0] == "/usr/bin/stat" and remote[-1] == "/usr/bin/python3":
            stdout = (
                b"symbolic link|1000|0|777|1\n"
                if drift == "link_owner"
                else b"symbolic link|0|0|777|1\n"
            )
        elif remote[0] == "/usr/bin/stat":
            target_states = {
                "target_symlink": b"symbolic link|0|0|777|1\n",
                "target_hardlink": b"regular file|0|0|755|2\n",
                "target_mode": b"regular file|0|0|775|1\n",
                "target_owner": b"regular file|1000|0|755|1\n",
            }
            stdout = target_states.get(
                drift,
                b"regular file|0|0|755|1\n",
            )
        elif remote[0] == "/usr/bin/sha256sum":
            digest_calls += 1
            digest = (
                "9" * 64
                if drift == "before_after" and digest_calls == 2
                else foundation_projection.interpreter_sha256
            )
            stdout = f"{digest}  /usr/bin/python3.11\n".encode("ascii")
        elif remote == ("/usr/bin/python3", "--version"):
            stdout = b"Python 3.11.2\n"
        else:  # pragma: no cover - fixed-command tripwire
            raise AssertionError(remote)
        return transport_module._ProcessResult(0, stdout, b"")

    transport._stage0_exchange = exchange
    return transport, calls


def test_interpreter_attestation_hashes_before_version_and_is_stable() -> None:
    transport, calls = _interpreter_attestation_transport()

    receipt = transport._attest_remote_interpreter()

    assert receipt["python_executed_before_digest_match"] is False
    assert receipt["identity_stable_before_after_version_probe"] is True
    assert receipt["project_ancestry_evidence_sha256"] == "9" * 64
    assert receipt["project_ancestry_chain_sha256"] == "a" * 64
    assert receipt["resource_ancestor_chain"] == [
        "organizations/123456789012"
    ]
    assert all(payload == b"" for _remote, payload in calls)
    version_index = next(
        index
        for index, (remote, _payload) in enumerate(calls)
        if remote == ("/usr/bin/python3", "--version")
    )
    digest_indexes = [
        index
        for index, (remote, _payload) in enumerate(calls)
        if remote[0] == "/usr/bin/sha256sum"
    ]
    assert any(index < version_index for index in digest_indexes)
    assert any(index > version_index for index in digest_indexes)


@pytest.mark.parametrize(
    "drift",
    (
        "link_target",
        "link_owner",
        "target_symlink",
        "target_hardlink",
        "target_mode",
        "target_owner",
        "before_after",
    ),
)
def test_interpreter_attestation_rejects_identity_or_toctou_drift(
    drift: str,
) -> None:
    transport, _calls = _interpreter_attestation_transport(drift=drift)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_python_.*_failed",
    ):
        transport._attest_remote_interpreter()


@pytest.mark.parametrize(
    ("result", "code"),
    (
        (
            transport_module._ProcessResult(1, b"", b""),
            "owner_gate_stage0_iap_probe_failed",
        ),
        (
            transport_module._ProcessResult(0, b"", b"bounded error"),
            "owner_gate_stage0_iap_probe_failed",
        ),
        (
            transport_module._ProcessResult(0, b"drift", b""),
            "owner_gate_stage0_iap_probe_failed",
        ),
    ),
)
def test_fixed_operation_rejects_nonzero_stderr_or_stdout_drift(
    result: transport_module._ProcessResult,
    code: str,
) -> None:
    transport = _minimal_transport(result)
    operation = transport_module._FixedOperation(
        "probe", ("/usr/bin/true",), b"", 0, 5.0
    )

    with pytest.raises(launcher.OwnerLauncherError, match=code):
        transport._execute_empty(operation)


@pytest.mark.parametrize("case", ("regular", "symlink", "hardlink"))
def test_sealer_stage_replay_removes_stale_inode_without_following(
    tmp_path: Path,
    case: str,
) -> None:
    sealer = _Sealer()
    transport = _minimal_transport(
        transport_module._ProcessResult(0, b"", b"")
    )
    _remote_directory, remote_stage, remote_final = transport._sealer_paths(
        sealer.sha256
    )
    local_directory = tmp_path / "remote"
    local_directory.mkdir()
    local_stage = local_directory / "stage"
    local_final = local_directory / "final"
    victim = tmp_path / "victim"
    victim.write_bytes(b"must remain unchanged")
    if case == "regular":
        local_stage.write_bytes(b"partial stale bytes")
    elif case == "symlink":
        local_stage.symlink_to(victim)
    else:
        os.link(victim, local_stage)

    def local_path(remote: str) -> Path:
        if remote == remote_stage:
            return local_stage
        if remote == remote_final:
            return local_final
        return local_directory

    def exchange(argv, _environment, input_source, **_kwargs):
        payload = input_source.read()
        command_arg = next(item for item in argv if item.startswith("--command="))
        command = shlex.split(command_arg.removeprefix("--command="))[3:]
        binary = command[0]
        stdout = b""
        if binary == "/usr/bin/install":
            local_directory.mkdir(exist_ok=True)
        elif binary == "/bin/rm":
            local_path(command[-1]).unlink(missing_ok=True)
        elif binary == "/usr/bin/dd":
            assert "oflag=excl,nofollow" in command
            destination = local_path(
                next(item for item in command if item.startswith("of=")).split(
                    "=", 1
                )[1]
            )
            with destination.open("xb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        elif binary == "/bin/chmod":
            local_path(command[-1]).chmod(int(command[1], 8))
        elif binary == "/bin/chown" or binary == "/usr/bin/sync":
            pass
        elif binary == "/usr/bin/sha256sum":
            selected = local_path(command[1])
            stdout = (
                f"{hashlib.sha256(selected.read_bytes()).hexdigest()}  "
                f"{command[1]}\n"
            ).encode("ascii")
        elif binary == "/bin/cp":
            source = local_path(command[-2])
            destination = local_path(command[-1])
            if not destination.exists():
                shutil.copyfile(source, destination)
        else:
            raise AssertionError(command)
        return transport_module._ProcessResult(0, stdout, b"")

    transport._stage0_exchange = exchange
    assert transport._materialize_sealer(sealer.payload, sealer.sha256) == remote_final
    assert local_final.read_bytes() == sealer.payload
    assert not local_stage.exists()
    assert victim.read_bytes() == b"must remain unchanged"


class _ChangingSealer(_Sealer):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def snapshot(self) -> tuple[bytes, str]:
        self.calls += 1
        if self.calls == 1:
            return super().snapshot()
        changed = self.payload + b"\n# changed\n"
        return changed, hashlib.sha256(changed).hexdigest()


def test_transport_rejects_local_sealer_change(tmp_path: Path) -> None:
    kit_stream, bundle_stream = _streams(tmp_path)
    transport, _calls = _transport(
        kit_stream=kit_stream,
        bundle_stream=bundle_stream,
    )
    transport._stage0_sealer_source = _ChangingSealer()

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_sealer_changed",
    ):
        transport.transport_exact_stage0_and_bundle(
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )


def test_bounded_exchange_rejects_early_remote_close() -> None:
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_stdin_failed",
    ):
        transport_module._bounded_process_exchange(
            ("/usr/bin/true",),
            {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            io.BytesIO(b"x" * (2 * 1024 * 1024)),
            maximum_input_bytes=2 * 1024 * 1024,
            timeout_seconds=5.0,
        )


def _natural_reap(process: subprocess.Popen[bytes]) -> None:
    """Test cleanup that never sends a signal to a live system process."""

    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        if stream is not None and not stream.closed:
            stream.close()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - safety tripwire
        raise AssertionError("test child did not exit naturally") from exc


def test_bounded_exchange_rejects_input_oversize() -> None:
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_input_oversized",
    ):
        transport_module._bounded_process_exchange(
            ("/bin/cat",),
            {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            io.BytesIO(b"too large"),
            maximum_input_bytes=3,
            timeout_seconds=5.0,
            process_terminator=_natural_reap,
        )


def test_bounded_exchange_rejects_stdout_oversize() -> None:
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_stdout_oversized",
    ):
        transport_module._bounded_process_exchange(
            ("/bin/cat",),
            {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            io.BytesIO(b"12345"),
            maximum_input_bytes=5,
            maximum_stdout_bytes=3,
            timeout_seconds=5.0,
            process_terminator=_natural_reap,
        )


def test_bounded_exchange_rejects_stderr_oversize() -> None:
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_stderr_oversized",
    ):
        transport_module._bounded_process_exchange(
            (
                "/usr/bin/python3",
                "-c",
                "import os; os.write(2, b'x' * 32)",
            ),
            {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            io.BytesIO(b""),
            maximum_input_bytes=0,
            maximum_stderr_bytes=4,
            timeout_seconds=5.0,
            process_terminator=_natural_reap,
        )


def test_bounded_exchange_timeout_reaps_process() -> None:
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_iap_timeout",
    ):
        transport_module._bounded_process_exchange(
            (
                "/usr/bin/python3",
                "-c",
                "import time; time.sleep(0.15)",
            ),
            {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            io.BytesIO(b""),
            maximum_input_bytes=0,
            timeout_seconds=0.01,
            process_terminator=_natural_reap,
        )


def test_process_group_cleanup_is_term_then_kill_then_reap_without_live_signal() -> None:
    class _Stream:
        closed = False

        def close(self) -> None:
            self.closed = True

    class _Process:
        pid = 424242

        def __init__(self) -> None:
            self.stdin = _Stream()
            self.stdout = _Stream()
            self.stderr = _Stream()
            self.returncode = None
            self.waits: list[float] = []
            self.fallback_terminate_called = False
            self.fallback_kill_called = False

        def poll(self):
            return self.returncode

        def wait(self, timeout: float):
            self.waits.append(timeout)
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

        def terminate(self) -> None:
            self.fallback_terminate_called = True

        def kill(self) -> None:
            self.fallback_kill_called = True

    process = _Process()
    signals: list[tuple[int, int]] = []

    def fake_killpg(pid: int, selected_signal: int) -> None:
        signals.append((pid, selected_signal))
        if selected_signal == signal.SIGKILL:
            process.returncode = -signal.SIGKILL

    transport_module._terminate(process, kill_process_group=fake_killpg)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.waits == [5.0, 5.0]
    assert process.fallback_terminate_called is False
    assert process.fallback_kill_called is False
    assert process.poll() == -signal.SIGKILL
    assert process.stdin.closed
    assert process.stdout.closed
    assert process.stderr.closed
