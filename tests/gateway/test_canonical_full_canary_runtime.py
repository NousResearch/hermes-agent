"""Focused contracts for the isolated full-canary runtime foundation."""

from __future__ import annotations

import copy
import base64
import builtins
import errno
import hashlib
import json
import os
import re
import stat
import sys
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import gateway.canonical_full_canary_runtime as runtime

from gateway.canonical_full_canary_runtime import (
    API_SERVER_CREDENTIAL_NAME,
    DEFAULT_API_SERVER_CONTROL_KEY,
    DEFAULT_COLLECTOR_RUNTIME,
    DEFAULT_E2E_FIXTURE,
    DEFAULT_OBSERVER_CONFIG,
    EDGE_UNIT_NAME,
    CollectorReadiness,
    ExactArtifact,
    FullCanaryIdentities,
    FullCanaryPlan,
    FullCanarySystemdBundle,
    GATEWAY_UNIT_NAME,
    PHASE_B_READINESS_UNIT_NAME,
    WRITER_UNIT_NAME,
    _validate_gateway_config,
    _validate_writer_config,
    edge_start_command,
    evaluate_service_states,
    post_collector_start_commands,
    render_full_canary_systemd_bundle,
    stop_service_commands,
)


REVISION = "a" * 40
ARTIFACT_SHA256 = "b" * 64


def _phase_b_anchor() -> dict[str, object]:
    return {
        "phase_b_release_revision": REVISION,
        "phase_b_plan_sha256": "1" * 64,
        "phase_b_approval_sha256": "2" * 64,
        "phase_b_terminal_receipt_sha256": "3" * 64,
        "phase_b_foundation_generation_sha256": "4" * 64,
        "phase_b_readiness_receipt_sha256": "5" * 64,
        "phase_b_readiness_handoff_file_sha256": "6" * 64,
        "phase_b_readiness_sequence": 1,
    }


def _identities() -> FullCanaryIdentities:
    return FullCanaryIdentities.from_mapping(
        {
            "writer_user": "muncho_writer",
            "writer_group": "muncho_writer",
            "writer_uid": 2101,
            "writer_gid": 2201,
            "gateway_user": "hermes_gateway",
            "gateway_group": "hermes_gateway",
            "gateway_uid": 2102,
            "gateway_gid": 2202,
            "socket_client_group": "muncho_writer_clients",
            "socket_client_gid": 2203,
            "edge_user": "muncho-discord-egress",
            "edge_group": "muncho-discord-egress",
            "edge_uid": 2103,
            "edge_gid": 2204,
        }
    )


def _writer_only_service() -> str:
    interpreter = (
        Path("/opt/muncho-canary-releases") / REVISION / "venv/bin/python"
    )
    return f"""[Unit]
Description=Muncho privileged Canonical Writer (isolated canary)
Wants=network-online.target

[Service]
Type=notify
User=muncho_writer
Group=muncho_writer
WorkingDirectory=/opt/muncho-canary-releases/{REVISION}
ExecStart={interpreter} -B -I -m gateway.canonical_writer_bootstrap --config {runtime.DEFAULT_WRITER_CONFIG}
Restart=on-failure
RestartSec=5s
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=

[Install]
WantedBy=multi-user.target
"""


def _bundle() -> FullCanarySystemdBundle:
    root = Path("/opt/muncho-canary-releases") / REVISION
    return render_full_canary_systemd_bundle(
        revision=REVISION,
        artifact_sha256=ARTIFACT_SHA256,
        interpreter=root / "venv/bin/python",
        writer_only_service=_writer_only_service(),
        identities=_identities(),
        database_ip_allow="10.20.30.40/32",
    )


def _canonical_digest(value: dict) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _host_metadata_values() -> dict[str, str]:
    return {
        runtime._GCE_METADATA_PATHS["project_id"]: (
            runtime.DEDICATED_CANARY_PROJECT_ID
        ),
        runtime._GCE_METADATA_PATHS["project_number"]: (
            runtime.DEDICATED_CANARY_PROJECT_NUMBER
        ),
        runtime._GCE_METADATA_PATHS["zone"]: (
            "projects/39589465056/zones/europe-west3-a"
        ),
        runtime._GCE_METADATA_PATHS["instance_name"]: (
            runtime.DEDICATED_CANARY_INSTANCE_NAME
        ),
        runtime._GCE_METADATA_PATHS["instance_id"]: (
            runtime.DEDICATED_CANARY_INSTANCE_ID
        ),
        runtime._GCE_METADATA_PATHS["service_account_email"]: (
            runtime.DEDICATED_CANARY_SERVICE_ACCOUNT
        ),
    }


def _local_host_values() -> dict[str, str]:
    return {
        "machine_id": "1" * 32,
        "hostname": "muncho-canary-v2-01",
        "boot_id": "22222222-2222-4222-8222-222222222222",
    }


def _mapping_reader(values: dict[str, str]):
    return lambda name: values[name].encode("utf-8")


def _host_receipt_plan(raw: bytes) -> FullCanaryPlan:
    identities = _identities()
    return FullCanaryPlan(
        revision=REVISION,
        release={"artifact_sha256": ARTIFACT_SHA256},
        identities=identities,
        writer_activation_plan={},
        writer_activation_receipt={},
        writer_activation_receipt_file_sha256="c" * 64,
        phase_b_readiness_anchor=_phase_b_anchor(),
        artifacts={
            "host_identity_receipt": ExactArtifact(
                source_path=runtime.DEFAULT_HOST_IDENTITY_RECEIPT,
                target_path=runtime.DEFAULT_HOST_IDENTITY_RECEIPT,
                sha256=hashlib.sha256(raw).hexdigest(),
                mode=0o400,
                uid=0,
                gid=0,
                maximum_bytes=runtime._MAX_HOST_IDENTITY_RECEIPT_BYTES,
            ),
            "writer_config": ExactArtifact(
                source_path=Path("/tmp/full-canary-writer.json"),
                target_path=runtime.DEFAULT_WRITER_CONFIG,
                sha256="d" * 64,
                mode=0o440,
                uid=0,
                gid=identities.writer_gid,
            ),
        },
        allowed_previous_sha256={},
        unit_bundle=_bundle(),
        unit_paths={},
        e2e_verifier_module="gateway.canonical_full_canary_e2e",
        sha256="e" * 64,
    )


def _owner_approval(plan: FullCanaryPlan):
    now = int(time.time())
    return runtime.FullCanaryOwnerApproval.from_mapping(
        {
            "schema": runtime.FULL_CANARY_APPROVAL_SCHEMA,
            "scope": "full_canary_runtime_start",
            "plan_sha256": plan.sha256,
            "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
            "cryptographic_owner_proof": False,
            "owner_subject_sha256": "1" * 64,
            "approval_source_sha256": "2" * 64,
            "nonce_sha256": "3" * 64,
            "approved_at_unix": now - 1,
            "expires_at_unix": now + 300,
        }
    )


@pytest.mark.parametrize(
    "field,drifted",
    [
        ("project_id", "wrong-project"),
        ("project_number", "99999999999"),
        ("zone", "projects/39589465056/zones/europe-west3-b"),
        ("instance_name", "production-runtime"),
        ("instance_id", "1111111111111111111"),
        (
            "service_account_email",
            "production@adventico-ai-platform.iam.gserviceaccount.com",
        ),
    ],
)
def test_dedicated_host_gate_rejects_each_wrong_gce_tuple_member(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    drifted: str,
) -> None:
    metadata = _host_metadata_values()
    local = _local_host_values()
    receipt = runtime.collect_dedicated_canary_host_identity_receipt(
        metadata_reader=_mapping_reader(metadata),
        local_identity_reader=_mapping_reader(local),
        observed_at_unix=1_700_000_000,
    )
    raw = runtime._canonical_bytes(receipt)
    plan = _host_receipt_plan(raw)
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda _artifact, *, label: raw,
    )
    wrong = dict(metadata)
    wrong[runtime._GCE_METADATA_PATHS[field]] = drifted
    with pytest.raises(RuntimeError):
        runtime.validate_dedicated_canary_host(
            plan,
            metadata_reader=_mapping_reader(wrong),
            local_identity_reader=_mapping_reader(local),
        )


@pytest.mark.parametrize(
    "field,drifted",
    [
        ("machine_id", "3" * 32),
        ("hostname", "replacement-canary"),
        ("boot_id", "44444444-4444-4444-8444-444444444444"),
    ],
)
def test_dedicated_host_gate_rejects_stale_host_or_boot_binding(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    drifted: str,
) -> None:
    metadata = _host_metadata_values()
    local = _local_host_values()
    receipt = runtime.collect_dedicated_canary_host_identity_receipt(
        metadata_reader=_mapping_reader(metadata),
        local_identity_reader=_mapping_reader(local),
        observed_at_unix=1_700_000_000,
    )
    raw = runtime._canonical_bytes(receipt)
    plan = _host_receipt_plan(raw)
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda _artifact, *, label: raw,
    )
    wrong = dict(local)
    wrong[field] = drifted
    with pytest.raises(RuntimeError, match="stale or mismatched"):
        runtime.validate_dedicated_canary_host(
            plan,
            metadata_reader=_mapping_reader(metadata),
            local_identity_reader=_mapping_reader(wrong),
        )


@pytest.mark.parametrize(
    "mutation",
    ["source", "target", "mode", "uid", "gid", "maximum"],
)
def test_dedicated_host_gate_rejects_arbitrary_plan_artifact(
    mutation: str,
) -> None:
    metadata = _host_metadata_values()
    local = _local_host_values()
    receipt = runtime.collect_dedicated_canary_host_identity_receipt(
        metadata_reader=_mapping_reader(metadata),
        local_identity_reader=_mapping_reader(local),
        observed_at_unix=1_700_000_000,
    )
    raw = runtime._canonical_bytes(receipt)
    plan = _host_receipt_plan(raw)
    artifact = plan.artifacts["host_identity_receipt"]
    changes = {
        "source": {"source_path": Path("/tmp/self-asserted-host.json")},
        "target": {"target_path": Path("/tmp/self-asserted-host.json")},
        "mode": {"mode": 0o440},
        "uid": {"uid": 2101},
        "gid": {"gid": 2201},
        "maximum": {"maximum_bytes": 1024},
    }
    drifted = replace(artifact, **changes[mutation])
    drifted_plan = replace(
        plan,
        artifacts={**plan.artifacts, "host_identity_receipt": drifted},
    )
    with pytest.raises(RuntimeError, match="artifact is not pinned"):
        runtime.validate_dedicated_canary_host(
            drifted_plan,
            metadata_reader=_mapping_reader(metadata),
            local_identity_reader=_mapping_reader(local),
        )


@pytest.mark.parametrize("failure", ["symlink", "ownership", "mode"])
def test_sealed_host_receipt_rejects_untrusted_file_provenance(
    tmp_path: Path,
    failure: str,
) -> None:
    target = tmp_path / "host-identity-target.json"
    target.write_bytes(b"{}")
    target.chmod(0o400)
    path = target
    expected_uid = os.getuid()
    expected_mode = 0o400
    if failure == "symlink":
        path = tmp_path / "host-identity.json"
        path.symlink_to(target)
    elif failure == "ownership":
        expected_uid += 1
    else:
        target.chmod(0o600)
    with pytest.raises(RuntimeError, match="identity is invalid"):
        runtime._read_stable_file(
            path,
            maximum=1024,
            expected_uid=expected_uid,
            expected_gid=os.getgid(),
            allowed_modes=frozenset({expected_mode}),
        )


def test_sealed_host_receipt_rejects_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "host-identity.json"
    replacement = tmp_path / "replacement.json"
    path.write_bytes(b'{"receipt":"original"}')
    replacement.write_bytes(b'{"receipt":"replaced"}')
    path.chmod(0o400)
    replacement.chmod(0o400)
    expected_gid = path.lstat().st_gid
    real_read = os.read
    replaced = False

    def replacing_read(descriptor: int, maximum: int) -> bytes:
        nonlocal replaced
        value = real_read(descriptor, maximum)
        if value and not replaced:
            replaced = True
            os.replace(replacement, path)
        return value

    monkeypatch.setattr(runtime.os, "read", replacing_read)
    with pytest.raises(RuntimeError, match="changed during read"):
        runtime._read_stable_file(
            path,
            maximum=1024,
            expected_uid=os.getuid(),
            expected_gid=expected_gid,
            allowed_modes=frozenset({0o400}),
        )


def _publish_exclusive_for_test(
    directory: Path,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o400,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        runtime._write_exclusive_bytes_at(
            descriptor,
            name,
            payload,
            mode=mode,
            expected_uid=os.getuid() if expected_uid is None else expected_uid,
            expected_gid=(
                directory.stat().st_gid if expected_gid is None else expected_gid
            ),
        )
    finally:
        os.close(descriptor)


def test_exclusive_publisher_is_atomic_create_only_and_idempotent(
    tmp_path: Path,
) -> None:
    payload = b'{"canonical":"exact"}\n'
    path = tmp_path / "receipt.json"
    temp = tmp_path / runtime._exclusive_publication_temp_name(path.name)

    _publish_exclusive_for_test(tmp_path, path.name, payload)
    original = path.stat()
    _publish_exclusive_for_test(tmp_path, path.name, payload)

    assert path.read_bytes() == payload
    assert path.stat().st_ino == original.st_ino
    assert path.stat().st_nlink == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert not temp.exists()
    with pytest.raises(RuntimeError, match="identity is invalid|payload drifted"):
        _publish_exclusive_for_test(tmp_path, path.name, b'{"different":true}\n')
    assert path.read_bytes() == payload
    assert path.stat().st_ino == original.st_ino


@pytest.mark.parametrize("interrupted_entry", ["temp", "final"])
def test_exclusive_publisher_recovers_only_exact_truncated_prefix(
    tmp_path: Path,
    interrupted_entry: str,
) -> None:
    payload = b'{"canonical":"approval-or-receipt","value":123}\n'
    final = tmp_path / "approval.json"
    temp = tmp_path / runtime._exclusive_publication_temp_name(final.name)
    interrupted = temp if interrupted_entry == "temp" else final
    interrupted.write_bytes(payload[: len(payload) // 2])
    interrupted.chmod(0o400)

    _publish_exclusive_for_test(tmp_path, final.name, payload)

    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1
    assert not temp.exists()


def test_exclusive_publisher_fsyncs_complete_orphan_temp_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"canonical":"complete-before-fsync"}\n'
    final = tmp_path / "receipt.json"
    temp = tmp_path / runtime._exclusive_publication_temp_name(final.name)
    temp.write_bytes(payload)
    temp.chmod(0o400)
    orphan_identity = (temp.stat().st_dev, temp.stat().st_ino)
    flushed: set[tuple[int, int]] = set()
    real_fsync = runtime.os.fsync
    real_rename = runtime._rename_noreplace_at

    def recording_fsync(descriptor: int) -> None:
        item = os.fstat(descriptor)
        flushed.add((item.st_dev, item.st_ino))
        real_fsync(descriptor)

    def rename_after_flush(*args, **kwargs) -> None:
        assert orphan_identity in flushed
        real_rename(*args, **kwargs)

    monkeypatch.setattr(runtime.os, "fsync", recording_fsync)
    monkeypatch.setattr(runtime, "_rename_noreplace_at", rename_after_flush)

    _publish_exclusive_for_test(tmp_path, final.name, payload)

    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1
    assert not temp.exists()


def test_exclusive_publisher_fsyncs_exact_legacy_final_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"canonical":"legacy-direct-final"}\n'
    final = tmp_path / "receipt.json"
    final.write_bytes(payload)
    final.chmod(0o400)
    legacy_identity = (final.stat().st_dev, final.stat().st_ino)
    flushed: set[tuple[int, int]] = set()
    real_fsync = runtime.os.fsync

    def recording_fsync(descriptor: int) -> None:
        item = os.fstat(descriptor)
        flushed.add((item.st_dev, item.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(runtime.os, "fsync", recording_fsync)

    _publish_exclusive_for_test(tmp_path, final.name, payload)

    assert legacy_identity in flushed
    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1


def test_exclusive_publisher_recovers_durable_hardlink_fallback(
    tmp_path: Path,
) -> None:
    payload = b'{"canonical":"linked"}\n'
    final = tmp_path / "receipt.json"
    temp = tmp_path / runtime._exclusive_publication_temp_name(final.name)
    temp.write_bytes(payload)
    temp.chmod(0o400)
    os.link(temp, final)
    assert temp.stat().st_nlink == 2

    _publish_exclusive_for_test(tmp_path, final.name, payload)

    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1
    assert not temp.exists()


def test_exclusive_publisher_tolerates_concurrent_temp_cleanup_enoent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'{"canonical":"cleanup"}\n'
    final = tmp_path / "receipt.json"
    temp = tmp_path / runtime._exclusive_publication_temp_name(final.name)
    final.write_bytes(payload)
    final.chmod(0o400)
    temp.write_bytes(payload[: len(payload) // 2])
    temp.chmod(0o400)
    real_unlink = os.unlink
    disappeared = False

    def disappearing_unlink(name, *, dir_fd=None):
        nonlocal disappeared
        if name == temp.name and not disappeared:
            disappeared = True
            real_unlink(name, dir_fd=dir_fd)
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
        return real_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(runtime.os, "unlink", disappearing_unlink)
    _publish_exclusive_for_test(tmp_path, final.name, payload)

    assert disappeared is True
    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1
    assert not temp.exists()


@pytest.mark.parametrize("entry_name", ["final", "temp"])
@pytest.mark.parametrize(
    "drift",
    ["arbitrary-bytes", "symlink", "hardlink", "owner", "mode"],
)
def test_exclusive_publisher_rejects_untrusted_interrupted_state(
    tmp_path: Path,
    entry_name: str,
    drift: str,
) -> None:
    payload = b'{"canonical":"expected"}\n'
    final = tmp_path / "receipt.json"
    temp = tmp_path / runtime._exclusive_publication_temp_name(final.name)
    entry = final if entry_name == "final" else temp
    backing = tmp_path / "backing"
    expected_uid = os.getuid()

    if drift == "symlink":
        backing.write_bytes(payload)
        backing.chmod(0o400)
        entry.symlink_to(backing)
    elif drift == "hardlink":
        backing.write_bytes(payload)
        backing.chmod(0o400)
        os.link(backing, entry)
    else:
        entry.write_bytes(
            b"not-an-expected-prefix" if drift == "arbitrary-bytes" else payload
        )
        entry.chmod(0o600 if drift == "mode" else 0o400)
        if drift == "owner":
            expected_uid += 1

    with pytest.raises(RuntimeError, match="identity is invalid|payload drifted"):
        _publish_exclusive_for_test(
            tmp_path,
            final.name,
            payload,
            expected_uid=expected_uid,
        )
    assert os.path.lexists(entry)
    if entry_name == "temp":
        assert not final.exists()


def test_exclusive_publisher_serializes_concurrent_exact_retries(
    tmp_path: Path,
) -> None:
    payload = b'{"canonical":"concurrent"}\n'
    final = tmp_path / "receipt.json"
    temp = tmp_path / runtime._exclusive_publication_temp_name(final.name)
    barrier = threading.Barrier(4)
    failures: list[BaseException] = []

    def publish() -> None:
        try:
            barrier.wait()
            _publish_exclusive_for_test(tmp_path, final.name, payload)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    workers = [threading.Thread(target=publish) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert failures == []
    assert all(not worker.is_alive() for worker in workers)
    assert final.read_bytes() == payload
    assert final.stat().st_nlink == 1
    assert not temp.exists()


def test_stopped_preflight_host_mismatch_never_reaches_runner_or_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _host_metadata_values()
    local = _local_host_values()
    receipt = runtime.collect_dedicated_canary_host_identity_receipt(
        metadata_reader=_mapping_reader(metadata),
        local_identity_reader=_mapping_reader(local),
        observed_at_unix=1_700_000_000,
    )
    raw = runtime._canonical_bytes(receipt)
    plan = _host_receipt_plan(raw)
    wrong = dict(metadata)
    wrong[runtime._GCE_METADATA_PATHS["instance_name"]] = "production-runtime"
    runner_calls: list[tuple[str, ...]] = []
    install_calls: list[bool] = []
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda _artifact, *, label: raw,
    )
    monkeypatch.setattr(
        runtime,
        "_install_plan_artifacts",
        lambda _plan: install_calls.append(True),
    )

    def runner(command):
        runner_calls.append(command.argv)
        raise AssertionError("host mismatch reached a subprocess runner")

    with pytest.raises(runtime.FullCanaryPreflightError) as raised:
        runtime.collect_full_canary_preflight(
            plan,
            phase="stopped",
            runner=runner,
            metadata_reader=_mapping_reader(wrong),
            local_identity_reader=_mapping_reader(local),
        )
    assert raised.value.report["blockers"] == ["host.dedicated_canary_exact"]
    assert runner_calls == []
    assert install_calls == []


def test_host_is_revalidated_immediately_before_first_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _host_metadata_values()
    local = _local_host_values()
    receipt = runtime.collect_dedicated_canary_host_identity_receipt(
        metadata_reader=_mapping_reader(metadata),
        local_identity_reader=_mapping_reader(local),
        observed_at_unix=1_700_000_000,
    )
    raw = runtime._canonical_bytes(receipt)
    plan = _host_receipt_plan(raw)
    metadata_calls = 0
    install_calls: list[bool] = []
    runner_calls: list[tuple[str, ...]] = []

    def metadata_reader(path: str) -> bytes:
        nonlocal metadata_calls
        metadata_calls += 1
        if metadata_calls > len(runtime._GCE_METADATA_PATHS) and path == (
            runtime._GCE_METADATA_PATHS["instance_id"]
        ):
            return b"1111111111111111111"
        return metadata[path].encode("utf-8")

    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda _artifact, *, label: raw if label == "host_identity_receipt" else b"{}",
    )
    monkeypatch.setattr(runtime, "_validate_writer_config", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "_lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_preflight",
        lambda self, *, phase: {"report_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        runtime,
        "_install_plan_artifacts",
        lambda _plan: install_calls.append(True),
    )
    monkeypatch.setattr(
        runtime,
        "_write_append_only_receipt",
        lambda *_a, **_k: {"receipt_path": "/tmp/failure.json"},
    )

    def runner(command):
        runner_calls.append(command.argv)
        raise AssertionError("pre-install host mismatch reached mutation runner")

    lifecycle = runtime.FullCanaryLifecycle(
        plan,
        runner=runner,
        metadata_reader=metadata_reader,
        local_identity_reader=_mapping_reader(local),
    )
    with pytest.raises(RuntimeError, match="failed closed"):
        lifecycle.start(_owner_approval(plan))
    assert metadata_calls > len(runtime._GCE_METADATA_PATHS)
    assert install_calls == []
    assert runner_calls == []


def test_systemd_bundle_keeps_writer_credential_free_and_services_bounded() -> None:
    bundle = FullCanarySystemdBundle.from_mapping(_bundle().to_mapping())
    gateway_credential = (
        f"LoadCredential={API_SERVER_CREDENTIAL_NAME}:"
        f"{DEFAULT_API_SERVER_CONTROL_KEY}"
    )
    assert bundle.gateway_service.count(gateway_credential) == 1
    assert "LoadCredential=" not in bundle.edge_service
    assert "LoadCredential=" not in bundle.writer_service
    assert "ExecStartPre=" not in bundle.writer_service
    assert "ExecStopPost=" not in bundle.writer_service
    assert "EnvironmentFile=" not in "".join(
        (bundle.edge_service, bundle.writer_service, bundle.gateway_service)
    )
    for service in (
        bundle.edge_service,
        bundle.writer_service,
        bundle.gateway_service,
    ):
        assert "Restart=no\n" in service
        assert "Restart=on-failure" not in service
        assert "RuntimeMaxSec=900s\n" in service
    assert f"AssertPathExists={DEFAULT_OBSERVER_CONFIG}" in bundle.gateway_service
    assert f"ReadOnlyPaths={DEFAULT_OBSERVER_CONFIG}" in bundle.gateway_service
    assert f"ReadOnlyPaths={DEFAULT_E2E_FIXTURE}" in bundle.gateway_service
    assert (
        f"--config {runtime.DEFAULT_GATEWAY_CONFIG} "
        "--require-canonical-writer"
    ) in bundle.gateway_service
    assert (
        f"Environment=HERMES_CONFIG={runtime.DEFAULT_GATEWAY_CONFIG}"
    ) in bundle.gateway_service
    assert (
        f"Environment=HERMES_HOME={runtime.DEFAULT_GATEWAY_PROFILE_HOME}"
    ) in bundle.gateway_service
    assert bundle.gateway_service.count(
        runtime._GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE
    ) == 1
    assert (
        f"d {DEFAULT_COLLECTOR_RUNTIME} 0750 root "
        f"{_identities().gateway_group} - -"
    ) in bundle.tmpfiles

def test_writer_stays_credential_free_across_full_transition() -> None:
    writer_only = _writer_only_service()

    assert "LoadCredential=" not in writer_only
    assert "ExecStartPre=" not in writer_only
    assert "CapabilityBoundingSet=\n" in writer_only
    assert "Restart=on-failure\n" in writer_only

    transitioned = _bundle().writer_service
    assert "LoadCredential=" not in transitioned
    assert "ExecStartPre=" not in transitioned
    assert "ExecStopPost=" not in transitioned
    assert "CapabilityBoundingSet=\n" in transitioned
    assert "Restart=no\n" in transitioned


@pytest.mark.parametrize(
    "injection",
    [
        "LoadCredential=foreign:/tmp/secret\n",
        "ExecStartPre=/tmp/foreign-helper\n",
        "CapabilityBoundingSet=CAP_DAC_OVERRIDE\n",
        "Restart=no\n",
    ],
)
def test_full_writer_transition_rejects_precontaminated_writer_only_bytes(
    injection: str,
) -> None:
    writer_only = _writer_only_service()
    if injection.startswith("CapabilityBoundingSet"):
        writer_only = writer_only.replace("CapabilityBoundingSet=\n", injection, 1)
    elif injection.startswith("Restart"):
        writer_only = writer_only.replace("Restart=on-failure\n", injection, 1)
    else:
        writer_only = writer_only.replace("[Service]\n", f"[Service]\n{injection}", 1)
    interpreter = (
        Path("/opt/muncho-canary-releases") / REVISION / "venv/bin/python"
    )

    with pytest.raises(ValueError, match="cannot be extended exactly"):
        runtime._full_writer_service(
            writer_only,
            interpreter=interpreter,
            writer_user=_identities().writer_user,
            writer_group=_identities().writer_group,
        )


def test_systemd_bundle_rejects_any_second_credential() -> None:
    mapping = copy.deepcopy(_bundle().to_mapping())
    mapping["edge_service"] = mapping["edge_service"].replace(
        "[Service]\n",
        "[Service]\nLoadCredential=forbidden:/tmp/secret\n",
        1,
    )
    unsigned = {key: value for key, value in mapping.items() if key != "sha256"}
    mapping["sha256"] = _canonical_digest(unsigned)
    with pytest.raises(ValueError, match="credential boundary"):
        FullCanarySystemdBundle.from_mapping(mapping)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_unset",
        "missing_op_mask",
        "materialized_managed_scope",
        "missing_managed_parent",
        "missing_ssl_pin",
        "missing_user_plugin_mask",
        "missing_soul_mask",
        "missing_processes_mask",
        "missing_hooks_mask",
        "missing_cron_mask",
        "missing_scripts_mask",
        "missing_memories_mask",
        "missing_skills_mask",
        "missing_cursor_mask",
    ],
)
def test_systemd_bundle_rejects_environment_seal_drift(mutation: str) -> None:
    mapping = copy.deepcopy(_bundle().to_mapping())
    if mutation == "missing_unset":
        mapping["gateway_service"] = mapping["gateway_service"].replace(
            " HERMES_CODEX_BASE_URL", "", 1
        )
    elif mutation == "missing_op_mask":
        mapping["gateway_service"] = mapping["gateway_service"].replace(
            (
                "ReadOnlyPaths="
                f"{runtime.DEFAULT_GATEWAY_PROFILE_HOME}/.op.env\n"
            ),
            "",
            1,
        )
    elif mutation == "materialized_managed_scope":
        mapping["tmpfiles"] += (
            f"d {runtime.DEFAULT_DISABLED_MANAGED_SCOPE} "
            "0000 root root - -\n"
        )
    elif mutation == "missing_managed_parent":
        mapping["tmpfiles"] = mapping["tmpfiles"].replace(
            (
                f"d {runtime.DEFAULT_COLLECTOR_RUNTIME} 0750 root "
                f"{_identities().gateway_group} - -\n"
            ),
            "",
            1,
        )
    elif mutation == "missing_ssl_pin":
        mapping["gateway_service"] = mapping["gateway_service"].replace(
            f"Environment=SSL_CERT_FILE={runtime.DEFAULT_GATEWAY_CA_BUNDLE}\n",
            "",
            1,
        )
    elif mutation == "missing_user_plugin_mask":
        mapping["gateway_service"] = mapping["gateway_service"].replace(
            (
                "InaccessiblePaths="
                f"{runtime.DEFAULT_GATEWAY_USER_PLUGIN_ROOT}\n"
            ),
            "",
            1,
        )
    elif mutation == "missing_soul_mask":
        mapping["gateway_service"] = mapping["gateway_service"].replace(
            (
                "InaccessiblePaths="
                f"{runtime.DEFAULT_GATEWAY_PROFILE_HOME}/SOUL.md\n"
            ),
            "",
            1,
        )
    elif mutation == "missing_processes_mask":
        mapping["tmpfiles"] = mapping["tmpfiles"].replace(
            (
                f"f {runtime.DEFAULT_GATEWAY_PROFILE_HOME}/processes.json "
                "0000 root root - -\n"
            ),
            "",
            1,
        )
    elif mutation in {
        "missing_hooks_mask",
        "missing_cron_mask",
        "missing_scripts_mask",
        "missing_memories_mask",
        "missing_skills_mask",
    }:
        directory = mutation.removeprefix("missing_").removesuffix("_mask")
        mapping["gateway_service"] = mapping["gateway_service"].replace(
            (
                "InaccessiblePaths="
                f"{runtime.DEFAULT_GATEWAY_PROFILE_HOME}/{directory}\n"
            ),
            "",
            1,
        )
    else:
        mapping["tmpfiles"] = mapping["tmpfiles"].replace(
            (
                f"d {runtime.DEFAULT_GATEWAY_HOME}/.cursor "
                "0000 root root - -\n"
            ),
            "",
            1,
        )
    unsigned = {key: value for key, value in mapping.items() if key != "sha256"}
    mapping["sha256"] = _canonical_digest(unsigned)
    with pytest.raises(ValueError, match="configuration/environment boundary"):
        FullCanarySystemdBundle.from_mapping(mapping)


def test_gateway_startup_paths_require_readable_empty_env_and_absent_managed_child(
    tmp_path: Path,
) -> None:
    environment_file = tmp_path / ".env"
    soul_file = tmp_path / "SOUL.md"
    plugin_dir = tmp_path / "plugins"
    managed_parent = tmp_path / "runtime"
    managed_dir = managed_parent / "managed-scope-disabled"
    environment_file.touch(mode=0o444)
    environment_file.chmod(0o444)
    soul_file.touch(mode=0o600)
    soul_file.chmod(0)
    plugin_dir.mkdir(mode=0o700)
    plugin_dir.chmod(0)
    managed_parent.mkdir(mode=0o750)
    managed_parent.chmod(0o750)
    gateway_uid = os.getuid() + 1
    expected_gid = environment_file.lstat().st_gid
    try:
        assert runtime._validate_inert_gateway_paths(
            environment_files=(environment_file,),
            semantic_files=(soul_file,),
            semantic_directories=(plugin_dir,),
            managed_directory=managed_dir,
            expected_uid=os.getuid(),
            expected_gid=expected_gid,
            gateway_uid=gateway_uid,
            gateway_gid=expected_gid,
        )
        managed_dir.mkdir()
        with pytest.raises(RuntimeError, match="managed-scope child"):
            runtime._validate_inert_gateway_paths(
                environment_files=(environment_file,),
                semantic_files=(soul_file,),
                semantic_directories=(plugin_dir,),
                managed_directory=managed_dir,
                expected_uid=os.getuid(),
                expected_gid=expected_gid,
                gateway_uid=gateway_uid,
                gateway_gid=expected_gid,
            )
        managed_dir.rmdir()

        managed_parent.chmod(0o770)
        with pytest.raises(RuntimeError, match="parent boundary"):
            runtime._validate_inert_gateway_paths(
                environment_files=(environment_file,),
                semantic_files=(soul_file,),
                semantic_directories=(plugin_dir,),
                managed_directory=managed_dir,
                expected_uid=os.getuid(),
                expected_gid=expected_gid,
                gateway_uid=gateway_uid,
                gateway_gid=expected_gid,
            )
        managed_parent.chmod(0o750)

        soul_file.chmod(0o400)
        with pytest.raises(RuntimeError, match="file boundary"):
            runtime._validate_inert_gateway_paths(
                environment_files=(environment_file,),
                semantic_files=(soul_file,),
                semantic_directories=(plugin_dir,),
                managed_directory=managed_dir,
                expected_uid=os.getuid(),
                expected_gid=expected_gid,
                gateway_uid=gateway_uid,
                gateway_gid=expected_gid,
            )
    finally:
        environment_file.chmod(0o600)
        soul_file.chmod(0o600)
        plugin_dir.chmod(0o700)
        managed_parent.chmod(0o700)


def test_sealed_empty_environment_files_are_dotenv_loader_compatible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hermes_cli import env_loader

    homes = (tmp_path / "gateway-home", tmp_path / "gateway-profile")
    for home in homes:
        home.mkdir()
        for name in (".env", ".op.env"):
            path = home / name
            path.touch(mode=0o444)
            path.chmod(0o444)
    loaded: list[Path] = []

    def readable_empty_dotenv(*, dotenv_path, **_kwargs) -> None:
        path = Path(dotenv_path)
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise PermissionError("dotenv mask is unreadable")
        assert path.read_bytes() == b""
        loaded.append(path)

    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    monkeypatch.setattr(env_loader, "load_dotenv", readable_empty_dotenv)
    monkeypatch.setattr(
        env_loader,
        "_apply_external_secret_sources",
        lambda _home: None,
    )
    monkeypatch.setattr(env_loader, "_apply_managed_env", lambda: None)
    for home in homes:
        env_loader.load_hermes_dotenv(hermes_home=home)
    assert loaded == [
        homes[0] / ".env",
        homes[0] / ".op.env",
        homes[1] / ".env",
        homes[1] / ".op.env",
    ]

    unreadable = homes[0] / ".env"
    unreadable.chmod(0)
    with pytest.raises(PermissionError, match="unreadable"):
        env_loader.load_hermes_dotenv(hermes_home=homes[0])
    unreadable.chmod(0o444)


@pytest.mark.parametrize("mutation", ["missing", "different_config"])
def _gateway_config() -> dict:
    return {
        "canonical_brain": {
            "writer_boundary": {"enabled": True},
            "discord_edge": {"enabled": True},
            "tools_enabled": True,
        },
        "model": {"default": "gpt-5.6-sol", "provider": "openai-codex"},
        "agent": {
            "reasoning_effort": "high",
            "max_turns": 90,
            "adaptive_reasoning": {"enabled": True, "max_effort": "max"},
        },
        "memory": {
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
        "cron": {"enabled": False},
        "kanban": {
            "auxiliary_planning_enabled": False,
            "auto_decompose": False,
            "dispatch_in_gateway": False,
        },
        "curator": {"enabled": False, "prune_builtins": False},
        "plugins": {"enabled": ["muncho_canary_evidence"]},
        "platform_toolsets": {"api_server": ["canonical_brain", "todo"]},
        "gateway": {
            "api_server": {"max_concurrent_runs": 1},
            "isolated_runtime": True,
        },
        "platforms": {
            "api_server": {
                "enabled": True,
                "extra": {
                    "host": "127.0.0.1",
                    "port": 8642,
                    "key_credential": "api-server.key",
                },
            }
        },
    }


def _yaml_bytes(value: dict) -> bytes:
    return yaml.safe_dump(value, sort_keys=True).encode()


def test_gateway_config_pins_model_sovereignty_and_loopback_auth() -> None:
    assert _validate_gateway_config(_yaml_bytes(_gateway_config()))


def test_isolated_gateway_runtime_is_strict_opt_in_without_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.config import GatewayConfig

    monkeypatch.setenv("GATEWAY_ISOLATED_RUNTIME", "1")
    assert GatewayConfig.from_dict({}).isolated_runtime is False
    assert GatewayConfig.from_dict(
        {"gateway": {"isolated_runtime": "true"}}
    ).isolated_runtime is False
    assert GatewayConfig.from_dict(
        {"gateway": {"isolated_runtime": True}}
    ).isolated_runtime is True


def test_isolated_gateway_pins_exact_provider_registry_before_runtime_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run
    import providers

    calls: list[frozenset[str]] = []
    monkeypatch.setattr(
        providers,
        "configure_isolated_provider_discovery",
        lambda allowlist: calls.append(allowlist),
    )
    assert run._configure_gateway_provider_discovery(False) is False
    assert calls == []
    assert run._configure_gateway_provider_discovery(True) is True
    assert calls == [frozenset({"openai-codex"})]
    assert run._configure_gateway_provider_discovery(
        False,
        False,
        True,
    ) is True
    assert calls == [
        frozenset({"openai-codex"}),
        frozenset({"openai-codex"}),
    ]

    monkeypatch.setattr(
        providers,
        "configure_isolated_provider_discovery",
        lambda _allowlist: (_ for _ in ()).throw(
            RuntimeError("provider registry already broadened")
        ),
    )
    with pytest.raises(RuntimeError, match="already broadened"):
        run._configure_gateway_provider_discovery(True)


def test_explicit_cron_false_is_inert_and_default_remains_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run
    from hermes_cli import config as hermes_config

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"cron": {"enabled": False}},
    )
    assert run._gateway_cron_scheduler_enabled() is False

    monkeypatch.setattr(hermes_config, "load_config", lambda: {})
    assert run._gateway_cron_scheduler_enabled() is True
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"cron": {"enabled": 0}},
    )
    assert run._gateway_cron_scheduler_enabled() is True

    monkeypatch.setattr(
        run,
        "_gateway_cron_scheduler_enabled",
        lambda: pytest.fail("isolated cron must not reread config"),
    )
    assert run._gateway_cron_enabled_for_runtime(True) is False

    resolver_called = False

    def forbidden_resolver():
        nonlocal resolver_called
        resolver_called = True
        raise AssertionError("disabled cron must not resolve a provider")

    monkeypatch.setitem(
        sys.modules,
        "cron.scheduler_provider",
        SimpleNamespace(resolve_cron_scheduler=forbidden_resolver),
    )
    provider, thread = run._start_gateway_cron_scheduler(
        enabled=False,
        stop_event=threading.Event(),
        adapters={},
        loop=object(),
    )
    assert provider is None
    assert thread is None
    assert resolver_called is False


def test_isolated_runtime_blocks_hooks_process_recovery_and_all_auto_resume_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run

    runner = object.__new__(run.GatewayRunner)
    runner._isolated_runtime = True
    runner.hooks = SimpleNamespace(
        discover_and_load=lambda: pytest.fail("event hooks must not be discovered")
    )

    class ForbiddenSessionStore:
        @property
        def _lock(self):
            pytest.fail("session store must not be read for isolated auto-resume")

    runner.session_store = ForbiddenSessionStore()

    assert runner._load_gateway_startup_hooks() is False
    assert runner._recover_gateway_process_checkpoint() == 0
    assert runner._schedule_resume_pending_sessions() == 0
    assert runner._schedule_resume_pending_sessions(
        platform=run.Platform.API_SERVER
    ) == 0
    assert runner._agent_startup_isolation_kwargs() == {
        "skip_memory": True,
        "skip_context_files": True,
    }

    normal = object.__new__(run.GatewayRunner)
    normal._isolated_runtime = False
    assert normal._agent_startup_isolation_kwargs() == {
        "skip_memory": False,
        "skip_context_files": False,
    }


@pytest.mark.asyncio
async def test_isolated_startup_skips_session_mutations_and_background_watchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run

    runner = object.__new__(run.GatewayRunner)
    runner._isolated_runtime = True
    runner._async_session_store = SimpleNamespace(
        suspend_recently_active=lambda: pytest.fail(
            "isolated startup must not suspend prior sessions"
        )
    )
    runner._mark_runtime_status_active_sessions_resume_pending = lambda: pytest.fail(
        "isolated startup must not mark prior sessions"
    )
    runner._suspend_stuck_loop_sessions = lambda: pytest.fail(
        "isolated startup must not mutate stuck-loop sessions"
    )

    await runner._prepare_gateway_startup_restore()
    assert runner._startup_restore_in_progress is True
    assert runner._startup_restore_queue == []
    assert runner._startup_restore_tasks == []

    monkeypatch.setattr(
        run.asyncio,
        "create_task",
        lambda *_args, **_kwargs: pytest.fail(
            "isolated startup must not spawn continuity watchers"
        ),
    )
    assert runner._start_gateway_continuity_watchers() == ()


@pytest.mark.asyncio
async def test_disabled_kanban_watchers_return_before_db_import_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gateway.kanban_watchers as watchers
    from hermes_cli import config as hermes_config

    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": False}},
    )
    original_import = builtins.__import__
    db_imports: list[str] = []

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hermes_cli.kanban_db" or (
            name == "hermes_cli" and "kanban_db" in fromlist
        ):
            db_imports.append(name)
            raise AssertionError("disabled Kanban watcher imported its database")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        watchers.asyncio,
        "to_thread",
        lambda *_args, **_kwargs: pytest.fail(
            "disabled Kanban watcher dispatched background work"
        ),
    )
    target = SimpleNamespace()
    await watchers.GatewayKanbanWatchersMixin._kanban_notifier_watcher(target)
    await watchers.GatewayKanbanWatchersMixin._kanban_dispatcher_watcher(target)
    assert db_imports == []


def test_isolated_plugin_discovery_scans_and_loads_only_exact_bundled_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.plugins as plugins

    allowed = plugins.PluginManifest(
        name="allowed_observer",
        key="allowed_observer",
        source="bundled",
        kind="standalone",
        path=str(tmp_path / "allowed_observer"),
    )
    forbidden = plugins.PluginManifest(
        name="automatic_backend",
        key="image_gen/automatic_backend",
        source="bundled",
        kind="backend",
        path=str(tmp_path / "automatic_backend"),
    )
    manager = plugins.PluginManager()
    scans: list[tuple[Path, str]] = []
    loaded: list[str] = []

    def fake_scan(path, source, skip_names=None):
        scans.append((Path(path), source))
        if len(scans) > 1:
            pytest.fail("isolated discovery scanned a non-bundled source")
        return [allowed, forbidden]

    def fake_load(manifest):
        key = manifest.key or manifest.name
        loaded.append(key)
        manager._plugins[key] = plugins.LoadedPlugin(
            manifest=manifest,
            module=SimpleNamespace(),
            enabled=True,
        )

    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path)
    monkeypatch.setattr(manager, "_scan_directory", fake_scan)
    monkeypatch.setattr(
        manager,
        "_scan_entry_points",
        lambda: pytest.fail("isolated discovery enumerated entrypoints"),
    )
    monkeypatch.setattr(
        manager,
        "_load_plugin",
        fake_load,
    )

    manager.discover_and_load(
        isolated_allowlist=frozenset({"allowed_observer"})
    )
    assert scans == [(tmp_path, "bundled")]
    assert loaded == ["allowed_observer"]

    # A later generic caller cannot broaden an already-isolated manager.
    manager.discover_and_load()
    assert len(scans) == 1


def test_gateway_plugin_discovery_kwargs_never_fall_back_to_generic_mode() -> None:
    from gateway.config import _gateway_plugin_discovery_kwargs

    assert _gateway_plugin_discovery_kwargs(
        isolated_runtime=True,
        allowlist=("muncho_canary_evidence",),
    ) == {"isolated_allowlist": frozenset({"muncho_canary_evidence"})}
    with pytest.raises(RuntimeError, match="allowlist"):
        _gateway_plugin_discovery_kwargs(
            isolated_runtime=True,
            allowlist=None,
        )
    with pytest.raises(RuntimeError, match="allowlist"):
        _gateway_plugin_discovery_kwargs(
            isolated_runtime=True,
            allowlist=(),
        )


def test_isolated_gateway_config_propagates_plugin_discovery_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from gateway import config as gateway_config
    from hermes_cli import managed_scope
    from hermes_cli import plugins as plugin_module

    def fail_discovery(**_kwargs) -> None:
        raise RuntimeError("mandatory observer failed")

    monkeypatch.setattr(plugin_module, "discover_plugins", fail_discovery)
    direct = gateway_config.GatewayConfig(
        isolated_runtime=True,
        isolated_plugin_allowlist=("muncho_canary_evidence",),
    )
    with pytest.raises(RuntimeError, match="mandatory observer failed"):
        direct._is_platform_connected(
            gateway_config.Platform.LOCAL,
            gateway_config.PlatformConfig(enabled=True),
        )
    with pytest.raises(RuntimeError, match="mandatory observer failed"):
        gateway_config._apply_env_overrides(direct)

    normal = gateway_config.GatewayConfig()
    assert (
        normal._is_platform_connected(
            gateway_config.Platform.LOCAL,
            gateway_config.PlatformConfig(enabled=True),
        )
        is False
    )

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "gateway": {"isolated_runtime": True},
                "plugins": {"enabled": ["muncho_canary_evidence"]},
                "platforms": {"api_server": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_config, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        managed_scope,
        "apply_managed_overlay",
        lambda value: value,
    )
    with pytest.raises(RuntimeError, match="mandatory observer failed"):
        gateway_config.load_gateway_config()


def test_isolated_plugin_discovery_fails_on_prior_general_state_or_missing_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.plugins as plugins

    prior = plugins.PluginManager()
    prior._discovered = True
    prior._plugins["automatic_backend"] = plugins.LoadedPlugin(
        manifest=plugins.PluginManifest(
            name="automatic_backend",
            source="bundled",
            kind="backend",
        ),
        enabled=True,
    )
    with pytest.raises(RuntimeError, match="general plugins loaded"):
        prior.discover_and_load(
            isolated_allowlist=frozenset({"allowed_observer"})
        )

    missing = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path)
    monkeypatch.setattr(missing, "_scan_directory", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="allowlist is unavailable"):
        missing.discover_and_load(
            isolated_allowlist=frozenset({"allowed_observer"})
        )


def test_isolated_plugin_discovery_fails_closed_on_register_error_or_safe_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import hermes_cli.plugins as plugins

    manifest = plugins.PluginManifest(
        name="allowed_observer",
        key="allowed_observer",
        source="bundled",
        kind="standalone",
        path=str(tmp_path / "allowed_observer"),
    )
    failed = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path)
    monkeypatch.setattr(
        failed,
        "_scan_directory",
        lambda *_args, **_kwargs: [manifest],
    )

    def broken_register(_context) -> None:
        raise RuntimeError("collector registration failed")

    monkeypatch.setattr(
        failed,
        "_load_directory_module",
        lambda _manifest: SimpleNamespace(register=broken_register),
    )
    with pytest.raises(RuntimeError, match="failed to load"):
        failed.discover_and_load(
            isolated_allowlist=frozenset({"allowed_observer"})
        )
    assert failed._discovered is False
    assert failed._plugins["allowed_observer"].enabled is False
    assert "collector registration failed" in str(
        failed._plugins["allowed_observer"].error
    )
    with pytest.raises(RuntimeError, match="previously failed"):
        failed.discover_and_load()

    safe_mode = plugins.PluginManager()
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    with pytest.raises(RuntimeError, match="safe mode"):
        safe_mode.discover_and_load(
            isolated_allowlist=frozenset({"allowed_observer"})
        )
    assert safe_mode._discovered is False
    assert safe_mode._plugins == {}
    monkeypatch.delenv("HERMES_SAFE_MODE")
    with pytest.raises(RuntimeError, match="previously failed"):
        safe_mode.discover_and_load()


@pytest.mark.asyncio
async def test_isolated_gateway_stays_offline_when_mandatory_plugin_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from gateway import run
    from gateway.config import GatewayConfig
    from hermes_cli import plugins

    runner = object.__new__(run.GatewayRunner)
    runner.config = GatewayConfig(
        sessions_dir=tmp_path / "sessions",
        isolated_runtime=True,
        isolated_plugin_allowlist=("muncho_canary_evidence",),
    )
    runner._isolated_runtime = True
    runner._restart_drain_timeout = 30
    runner._startup_restore_in_progress = True

    async def keep_starting(*_args, **_kwargs) -> bool:
        return False

    runner._abort_startup_if_shutdown_requested = keep_starting
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        run,
        "_own_policy_open_startup_violation",
        lambda _config: None,
    )
    monkeypatch.setattr(
        plugins,
        "discover_plugins",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("observer register failed")
        ),
    )
    assert await runner.start() is False
    assert runner._startup_restore_in_progress is False


@pytest.mark.parametrize(
    "mutation",
    [
        "discord",
        "inline_key",
        "fallback",
        "kanban",
        "kanban_auxiliary",
        "kanban_dispatch",
        "cron_enabled",
        "memory_enabled",
        "isolated_runtime",
        "isolated_runtime_type",
        "terminal",
        "plugin",
        "concurrency",
        "turn_budget",
        "model_base_url",
        "model_custom_provider",
        "agent_extra",
        "adaptive_extra",
        "kanban_extra",
        "curator_extra",
        "canonical_extra",
        "plugin_extra",
        "gateway_extra",
        "root_extra",
    ],
)
def test_gateway_config_rejects_external_routing_or_semantic_authority(
    mutation: str,
) -> None:
    config = _gateway_config()
    if mutation == "discord":
        config["platforms"]["discord"] = {"enabled": True}
    elif mutation == "inline_key":
        config["platforms"]["api_server"]["extra"]["key"] = "secret"
    elif mutation == "fallback":
        config["fallback_model"] = "some-other-model"
    elif mutation == "kanban":
        config["kanban"]["auto_decompose"] = True
    elif mutation == "kanban_auxiliary":
        config["kanban"]["auxiliary_planning_enabled"] = True
    elif mutation == "kanban_dispatch":
        config["kanban"]["dispatch_in_gateway"] = True
    elif mutation == "cron_enabled":
        config["cron"]["enabled"] = True
    elif mutation == "memory_enabled":
        config["memory"]["memory_enabled"] = True
    elif mutation == "isolated_runtime":
        config["gateway"]["isolated_runtime"] = False
    elif mutation == "isolated_runtime_type":
        config["gateway"]["isolated_runtime"] = 1
    elif mutation == "terminal":
        config["platform_toolsets"]["api_server"].append("terminal")
    elif mutation == "plugin":
        config["plugins"]["enabled"] = []
    elif mutation == "concurrency":
        config["gateway"]["api_server"]["max_concurrent_runs"] = 2
    elif mutation == "turn_budget":
        config["agent"]["max_turns"] = 30
    elif mutation == "model_base_url":
        config["model"]["base_url"] = "https://attacker.invalid/v1"
    elif mutation == "model_custom_provider":
        config["model"]["custom_provider"] = "unreviewed"
    elif mutation == "agent_extra":
        config["agent"]["semantic_dispatch"] = True
    elif mutation == "adaptive_extra":
        config["agent"]["adaptive_reasoning"]["effort_router"] = "external"
    elif mutation == "kanban_extra":
        config["kanban"]["auxiliary_semantic_decomposition"] = True
    elif mutation == "curator_extra":
        config["curator"]["classifier"] = "external"
    elif mutation == "canonical_extra":
        config["canonical_brain"]["semantic_router"] = True
    elif mutation == "plugin_extra":
        config["plugins"]["autoload"] = True
    elif mutation == "gateway_extra":
        config["gateway"]["provider_override"] = "unreviewed"
    else:
        config["semantic_dispatcher"] = {"enabled": True}
    with pytest.raises(RuntimeError):
        _validate_gateway_config(_yaml_bytes(config))


def _sealed_gateway_environment_values() -> dict[str, str]:
    identity = _identities()
    return {
        "CREDENTIALS_DIRECTORY": f"/run/credentials/{GATEWAY_UNIT_NAME}",
        "HERMES_CONFIG": str(runtime.DEFAULT_GATEWAY_CONFIG),
        "HERMES_EXEC_ASK": "1",
        "HERMES_HOME": str(runtime.DEFAULT_GATEWAY_PROFILE_HOME),
        "HERMES_MANAGED_DIR": str(runtime.DEFAULT_DISABLED_MANAGED_SCOPE),
        "HERMES_MAX_ITERATIONS": "90",
        "HERMES_QUIET": "1",
        "HOME": str(runtime.DEFAULT_GATEWAY_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": identity.gateway_user,
        "NOTIFY_SOCKET": "@test-notify",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SHELL": "/usr/sbin/nologin",
        "SSL_CERT_FILE": str(runtime.DEFAULT_GATEWAY_CA_BUNDLE),
        "TERMINAL_CWD": str(runtime.DEFAULT_GATEWAY_HOME),
        "TZ": "UTC",
        "USER": identity.gateway_user,
        "_HERMES_GATEWAY": "1",
    }


def test_gateway_effective_environment_pins_exact_ca_bundle_hash() -> None:
    values = _sealed_gateway_environment_values()
    names = sorted(values)
    hashes = {
        name: hashlib.sha256(values[name].encode()).hexdigest()
        for name in names
    }
    plan = _host_receipt_plan(b"sealed-host-receipt")
    assert runtime._gateway_effective_environment_hashes_are_sealed(
        names,
        hashes,
        plan=plan,
    )
    hashes["SSL_CERT_FILE"] = "0" * 64
    assert not runtime._gateway_effective_environment_hashes_are_sealed(
        names,
        hashes,
        plan=plan,
    )


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "HERMES_CODEX_BASE_URL",
        "HERMES_ENVIRONMENT_HINT",
        "HERMES_KANBAN_TASK",
        "HERMES_ENABLE_PROJECT_PLUGINS",
        "HERMES_INFERENCE_MODEL",
        "HERMES_INFERENCE_PROVIDER",
        "HERMES_MAX_TOKENS",
        "HERMES_AGENT_TIMEOUT",
        "HERMES_CONCURRENT_TOOL_TIMEOUT_S",
        "OPENAI_API_KEY",
        "OP_SERVICE_ACCOUNT_TOKEN",
        "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY",
    ],
)
def test_live_readiness_rejects_forbidden_effective_gateway_environment_name(
    forbidden_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _sealed_gateway_environment_values()
    values[forbidden_name] = "must-never-enter-a-receipt"
    names = sorted(values)
    hashes = {
        name: hashlib.sha256(values[name].encode()).hexdigest()
        for name in names
    }
    identities = _identities()
    release_root = tmp_path / "release"
    plan = FullCanaryPlan(
        revision=REVISION,
        release={"artifact_root": str(release_root)},
        identities=identities,
        writer_activation_plan={},
        writer_activation_receipt={},
        writer_activation_receipt_file_sha256="c" * 64,
        phase_b_readiness_anchor=_phase_b_anchor(),
        artifacts={
            "edge_config": ExactArtifact(
                source_path=tmp_path / "edge.json",
                target_path=runtime.DEFAULT_EDGE_CONFIG,
                sha256="d" * 64,
                mode=0o440,
                uid=0,
                gid=identities.edge_gid,
            )
        },
        allowed_previous_sha256={},
        unit_bundle=_bundle(),
        unit_paths={},
        e2e_verifier_module=runtime.E2E_VERIFIER_MODULE,
        sha256="e" * 64,
    )
    receipts = {
        runtime.DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH: {
            "version": runtime.WRITER_RUNTIME_ATTESTATION_VERSION,
            "writer_pid": 101,
            "effective_environment_variable_names": [],
            "discord_edge_authority_enabled": True,
        },
        runtime.DEFAULT_GATEWAY_READINESS_PATH: {
            "version": runtime.READINESS_RECEIPT_VERSION,
            "gateway_pid": 102,
            "effective_environment_variable_names": names,
            "effective_environment_variable_value_sha256": hashes,
            "loaded_module_origins": [str(release_root / "gateway/run.py")],
        },
        runtime.DEFAULT_EDGE_READINESS_PATH: {
            "version": runtime.EDGE_READINESS_SCHEMA,
            "edge_pid": 103,
            "effective_environment_variable_names": [],
            "allowed_target_types": [
                "public_guild_channel",
                "public_guild_forum",
                "public_guild_thread",
            ],
            "forbidden_target_types": [
                "direct_message",
                "dm",
                "group_dm",
                "private_channel",
                "private_thread",
            ],
            "config_sha256": "d" * 64,
        },
    }
    states = {
        WRITER_UNIT_NAME: {"MainPID": 101, "StatusText": "ready"},
        GATEWAY_UNIT_NAME: {"MainPID": 102, "StatusText": "ready"},
        EDGE_UNIT_NAME: {"MainPID": 103, "StatusText": "ready"},
    }
    monkeypatch.setattr(
        runtime,
        "_readiness_receipt",
        lambda path, **_kwargs: receipts[path],
    )
    monkeypatch.setattr(
        runtime,
        "load_collector_readiness",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("absent")),
    )
    monkeypatch.setattr(
        runtime,
        "_api_loopback_listener_identity",
        lambda _pid: {
            "gateway_pid": 102,
            "host": "127.0.0.1",
            "port": 8642,
            "protocol": "tcp",
        },
    )

    checks = runtime._validate_live_readiness(plan, states)
    assert checks["readiness.gateway.sealed_effective_environment"] is False
    assert "must-never-enter-a-receipt" not in repr(receipts)


def _state(unit: str, *, live: bool) -> dict:
    path = {
        EDGE_UNIT_NAME: "/etc/systemd/system/muncho-discord-egress.service",
        WRITER_UNIT_NAME: "/etc/systemd/system/muncho-canonical-writer.service",
        GATEWAY_UNIT_NAME: "/etc/systemd/system/hermes-cloud-gateway.service",
    }[unit]
    return {
        "LoadState": "loaded",
        "ActiveState": "active" if live else "inactive",
        "SubState": "running" if live else "dead",
        "UnitFileState": "disabled",
        "MainPID": 111 if live else 0,
        "FragmentPath": path,
        "DropInPaths": "",
        "Type": "notify",
        "NotifyAccess": "main",
        "StatusText": "ready",
    }


def test_service_state_and_lifecycle_order_are_exact_and_disabled() -> None:
    units = (EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME)
    states = {unit: _state(unit, live=True) for unit in units}
    assert all(evaluate_service_states(states, phase="live").values())
    start = (edge_start_command(), *post_collector_start_commands())
    assert [command.argv[-1] for command in start] == list(units)
    assert [command.argv[-1] for command in stop_service_commands()] == list(
        reversed(units)
    ) + [PHASE_B_READINESS_UNIT_NAME]
    assert all("enable" not in command.argv for command in start)


@pytest.mark.parametrize(
    "unit",
    [EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME],
)
@pytest.mark.parametrize("phase", ["stopped", "live"])
def test_service_state_rejects_every_systemd_drop_in(
    unit: str,
    phase: str,
) -> None:
    assert "DropInPaths" in runtime._SERVICE_PROPERTIES
    states = {
        name: _state(name, live=phase == "live")
        for name in (EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME)
    }
    states[unit]["DropInPaths"] = "/run/systemd/system/override.conf"
    checks = evaluate_service_states(states, phase=phase)
    assert checks[f"service.{unit}.no_dropins"] is False


def test_edge_collector_gate_rejects_drop_in_before_receipt_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(EDGE_UNIT_NAME, live=True)
    state["DropInPaths"] = "/etc/systemd/system/override.conf"
    monkeypatch.setattr(
        runtime,
        "_readiness_receipt",
        lambda *_args, **_kwargs: pytest.fail(
            "drop-in must fail before readiness receipt access"
        ),
    )
    with pytest.raises(RuntimeError, match="not ready"):
        runtime._validate_edge_collector_gate(
            _host_receipt_plan(b"sealed-host-receipt"),
            state,
        )


def test_gateway_main_accepts_config_and_writer_readiness_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import run

    config = tmp_path / "gateway.yaml"
    config.write_text("{}\n", encoding="utf-8")
    captured = {}

    async def fake_start_gateway(parsed_config, **kwargs):
        captured["config"] = parsed_config
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(run, "start_gateway", fake_start_gateway)
    monkeypatch.setattr(
        run,
        "_load_required_canonical_gateway_config",
        lambda _path: run.GatewayConfig.from_dict({}),
    )
    monkeypatch.setattr(
        run,
        "_exit_after_graceful_shutdown",
        lambda code: captured.update(exit_code=code),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gateway.run",
            "--config",
            str(config),
            "--require-canonical-writer",
        ],
    )
    run.main()
    assert captured["kwargs"] == {"require_canonical_writer": True}
    assert captured["exit_code"] == 0


def _prepare_required_gateway_config_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from gateway import run
    from hermes_cli import config as config_module

    # The production pin is intentionally one-way for the life of a process.
    # Each test gets its own synthetic process authority via monkeypatch.
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)

    gateway_home = tmp_path / "gateway-home"
    profile_home = gateway_home / ".hermes"
    profile_home.mkdir(parents=True)
    config = profile_home / "config.yaml"
    config.write_bytes(_yaml_bytes(_gateway_config()))
    disabled_managed = tmp_path / "managed-scope-disabled"
    monkeypatch.setattr(runtime, "DEFAULT_GATEWAY_HOME", gateway_home)
    monkeypatch.setattr(runtime, "DEFAULT_GATEWAY_PROFILE_HOME", profile_home)
    monkeypatch.setattr(runtime, "DEFAULT_GATEWAY_CONFIG", config)
    monkeypatch.setattr(
        runtime,
        "DEFAULT_DISABLED_MANAGED_SCOPE",
        disabled_managed,
    )
    monkeypatch.setattr(run, "_gateway_config_home", lambda: profile_home)
    environment = _sealed_gateway_environment_values()
    environment.update(
        HOME=str(gateway_home),
        HERMES_CONFIG=str(config),
        HERMES_HOME=str(profile_home),
        HERMES_MANAGED_DIR=str(disabled_managed),
        TERMINAL_CWD=str(gateway_home),
    )
    monkeypatch.setattr(os, "environ", environment)
    return run, config, environment, disabled_managed


def test_required_gateway_config_accepts_only_exact_effective_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as config_module

    run, config, _environment, _disabled = _prepare_required_gateway_config_test(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setitem(
        config_module.DEFAULT_CONFIG,
        "future_semantic_default",
        {"must_not_enter_isolated_runtime": True},
    )
    parsed = run._load_required_canonical_gateway_config(str(config))
    assert parsed.platforms[run.Platform.API_SERVER].enabled is True
    assert parsed.isolated_runtime is True
    assert parsed.isolated_plugin_allowlist == ("muncho_canary_evidence",)

    expected = _gateway_config()
    effective = config_module.load_config()
    assert effective == expected
    assert config_module.load_config_readonly() == expected
    assert config_module.read_raw_config() == expected
    assert run._load_gateway_config() == expected
    assert run._load_gateway_runtime_config() == expected
    assert "future_semantic_default" not in effective
    assert effective["agent"]["adaptive_reasoning"] == {
        "enabled": True,
        "max_effort": "max",
    }
    assert effective["kanban"] == {
        "auxiliary_planning_enabled": False,
        "auto_decompose": False,
        "dispatch_in_gateway": False,
    }
    assert effective["platform_toolsets"] == {
        "api_server": ["canonical_brain", "todo"]
    }

    # Consumers receive defensive copies; no in-process caller can rewrite the
    # process authority without changing the sealed bytes.
    effective["agent"]["reasoning_effort"] = "low"
    assert config_module.load_config()["agent"]["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "HERMES_CODEX_BASE_URL",
        "HERMES_ENVIRONMENT_HINT",
        "HERMES_KANBAN_TASK",
        "HERMES_ENABLE_PROJECT_PLUGINS",
        "HERMES_INFERENCE_PROVIDER",
        "HERMES_MAX_TOKENS",
        "HERMES_AGENT_TIMEOUT",
        "HERMES_CONCURRENT_TOOL_TIMEOUT_S",
        "OPENAI_API_KEY",
        "OP_SERVICE_ACCOUNT_TOKEN",
        "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY",
    ],
)
def test_required_gateway_config_rejects_inherited_semantic_or_secret_env(
    forbidden_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, config, environment, _disabled = _prepare_required_gateway_config_test(
        tmp_path,
        monkeypatch,
    )
    environment[forbidden_name] = "must-never-be-logged"
    with pytest.raises(RuntimeError, match="environment is not sealed") as exc:
        run._load_required_canonical_gateway_config(str(config))
    assert "must-never-be-logged" not in str(exc.value)


def test_required_gateway_config_rejects_managed_overlay_or_effective_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as config_module

    run, config, environment, disabled = _prepare_required_gateway_config_test(
        tmp_path,
        monkeypatch,
    )
    disabled.mkdir()
    with pytest.raises(RuntimeError, match="managed scope is not disabled"):
        run._load_required_canonical_gateway_config(str(config))

    disabled.rmdir()
    drifted = _gateway_config()
    drifted["model"] = {"default": "other-model", "provider": "custom"}
    monkeypatch.setattr(config_module, "load_config", lambda: drifted)
    assert environment["HERMES_MANAGED_DIR"] == str(disabled)
    with pytest.raises(RuntimeError, match="effective config drifted"):
        run._load_required_canonical_gateway_config(str(config))


def test_effective_config_pin_rejects_claimed_raw_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as config_module

    _run, config, _environment, _disabled = _prepare_required_gateway_config_test(
        tmp_path,
        monkeypatch,
    )
    raw = config.read_bytes()
    sealed = _validate_gateway_config(raw)

    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="SHA-256 does not match raw bytes",
    ):
        config_module.pin_effective_config_projection(
            config_path=config,
            raw_bytes=raw,
            raw_sha256="0" * 64,
            exact_mapping=sealed,
        )
    assert config_module.effective_config_projection_is_pinned() is False


def test_post_pin_raw_or_path_drift_stays_out_of_snapshot_and_fails_at_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as config_module

    run, config, _environment, _disabled = _prepare_required_gateway_config_test(
        tmp_path,
        monkeypatch,
    )
    run._load_required_canonical_gateway_config(str(config))
    assert config_module.effective_config_projection_is_pinned() is True

    drifted = _gateway_config()
    drifted["future_semantic_default"] = {"enabled": True}
    config.write_bytes(_yaml_bytes(drifted))

    expected = _gateway_config()
    readers = (
        config_module.load_config,
        config_module.load_config_readonly,
        config_module.read_raw_config,
        run._load_gateway_config,
        run._load_gateway_runtime_config,
    )
    for reader in readers:
        assert reader() == expected
    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="raw content drifted",
    ):
        config_module.attest_pinned_effective_config_projection()

    config.write_bytes(_yaml_bytes(_gateway_config()))
    monkeypatch.setattr(
        config_module,
        "get_config_path",
        lambda: tmp_path / "other" / "config.yaml",
    )
    assert config_module.load_config() == expected
    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="path drifted",
    ):
        config_module.attest_pinned_effective_config_projection()


def test_post_pin_parse_failure_and_managed_scope_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as config_module

    run, config, _environment, disabled = _prepare_required_gateway_config_test(
        tmp_path,
        monkeypatch,
    )
    run._load_required_canonical_gateway_config(str(config))

    def fail_parse(_stream):
        raise ValueError("synthetic parser failure")

    with monkeypatch.context() as parse_patch:
        parse_patch.setattr(config_module, "fast_safe_load", fail_parse)
        assert config_module.load_config() == _gateway_config()
        with pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="parse failed",
        ):
            config_module.attest_pinned_effective_config_projection()

    disabled.mkdir()
    assert config_module.load_config() == _gateway_config()
    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="managed scope appeared",
    ):
        config_module.attest_pinned_effective_config_projection()


def test_post_pin_drift_cannot_bypass_gateway_budget_or_provider_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider

    run, config, environment, _disabled = _prepare_required_gateway_config_test(
        tmp_path,
        monkeypatch,
    )
    run._load_required_canonical_gateway_config(str(config))

    drifted = _gateway_config()
    drifted["agent"]["max_turns"] = 999
    drifted["provider_routing"] = {"order": ["attacker"]}
    drifted["fallback_model"] = {
        "provider": "custom",
        "model": "attacker-model",
    }
    config.write_bytes(_yaml_bytes(drifted))
    environment["HERMES_MAX_ITERATIONS"] = "90"

    run._bridge_max_turns_from_config(config.parent)
    assert environment["HERMES_MAX_ITERATIONS"] == "90"

    assert run.GatewayRunner._load_provider_routing() == {}
    assert run.GatewayRunner._load_fallback_model() is None

    prior_fallback = [{"provider": "sealed", "model": "sealed-model"}]
    runner = SimpleNamespace(_fallback_model=prior_fallback)
    refresh = run.GatewayRunner._refresh_fallback_model.__get__(runner)
    assert refresh() is None
    assert runner._fallback_model is None

    def must_not_resolve_provider(**_kwargs):
        raise AssertionError("drifted fallback route reached provider resolution")

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        must_not_resolve_provider,
    )
    assert run._try_resolve_fallback_provider() is None

    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="raw content drifted",
    ):
        config_module.attest_pinned_effective_config_projection()


def test_plugin_readiness_binds_authenticated_frame_and_live_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway_pid = os.getpid()
    gateway_uid = os.getuid()
    gateway_gid = os.getgid()
    release_root = tmp_path / "release"
    module_origin = release_root / "plugins/muncho_canary_evidence/__init__.py"
    module_origin.parent.mkdir(parents=True)
    module_origin.write_text("# sealed test plugin\n", encoding="utf-8")
    _, module_sha256 = runtime.module_file_identity(module_origin)

    identities = FullCanaryIdentities(
        writer_user="writer",
        writer_group="writer",
        writer_uid=gateway_uid + 101,
        writer_gid=gateway_gid + 101,
        gateway_user="gateway",
        gateway_group="gateway",
        gateway_uid=gateway_uid,
        gateway_gid=gateway_gid,
        socket_client_group="clients",
        socket_client_gid=gateway_gid + 102,
        edge_user="muncho-discord-egress",
        edge_group="muncho-discord-egress",
        edge_uid=gateway_uid + 103,
        edge_gid=gateway_gid + 103,
    )
    fixture_sha256 = "c" * 64
    plan = FullCanaryPlan(
        revision=REVISION,
        release={
            "artifact_root": str(release_root),
            "artifact_sha256": ARTIFACT_SHA256,
        },
        identities=identities,
        writer_activation_plan={},
        writer_activation_receipt={},
        writer_activation_receipt_file_sha256="d" * 64,
        phase_b_readiness_anchor=_phase_b_anchor(),
        artifacts={
            "e2e_fixture": ExactArtifact(
                source_path=tmp_path / "fixture.json",
                target_path=tmp_path / "fixture.json",
                sha256=fixture_sha256,
                mode=0o440,
                uid=0,
                gid=gateway_gid,
            )
        },
        allowed_previous_sha256={},
        unit_bundle=_bundle(),
        unit_paths={},
        e2e_verifier_module="gateway.canonical_full_canary_e2e",
        sha256="e" * 64,
    )
    collector_receipt = {"sealed": True}
    collector_raw = runtime._canonical_bytes(collector_receipt)
    collector = CollectorReadiness(
        receipt=collector_receipt,
        file_sha256=hashlib.sha256(collector_raw).hexdigest(),
        service_identity_sha256="1" * 64,
    )
    now_ms = int(time.time() * 1000)
    fixture = {
        "canary_run_id": "42cce3d5-9ee5-4ccc-a07c-fbc340bb58b0",
        "case_id": "case:full-canary",
        "api_session_key_sha256": "7" * 64,
        "valid_from_unix_ms": now_ms - 10_000,
        "valid_until_unix_ms": now_ms + 60_000,
    }
    observer = {
        "schema": "muncho-canary-evidence-config.v1",
    }
    observer_raw = runtime._canonical_bytes(observer)
    collector_socket_sha256 = "2" * 64
    edge_socket_sha256 = "3" * 64
    edge_identity_sha256 = "4" * 64
    edge_pid = 12345
    payload = {
        "plugin_name": "muncho_canary_evidence",
        "gateway_pid": gateway_pid,
        "config_sha256": hashlib.sha256(observer_raw).hexdigest(),
        "fixture_sha256": fixture_sha256,
        "release_sha": REVISION,
        "release_sha256": ARTIFACT_SHA256,
        "api_session_key_sha256": fixture["api_session_key_sha256"],
        "collector_service_identity_sha256": collector.service_identity_sha256,
        "collector_socket_identity_sha256": collector_socket_sha256,
        "discord_edge_service_identity_sha256": edge_identity_sha256,
        "discord_edge_socket_identity_sha256": edge_socket_sha256,
        "module_origin": str(module_origin),
        "module_sha256": module_sha256,
    }
    frame = {
        "schema": runtime.PLUGIN_FRAME_SCHEMA,
        "sequence": 1,
        "event": "plugin_ready",
        "release_sha": REVISION,
        "release_sha256": ARTIFACT_SHA256,
        "canary_run_id": fixture["canary_run_id"],
        "case_id": fixture["case_id"],
        "fixture_sha256": fixture_sha256,
        "collector_service_identity_sha256": collector.service_identity_sha256,
        "discord_edge_service_identity_sha256": edge_identity_sha256,
        "session_id": None,
        "turn_id": None,
        "observed_at_unix_ms": now_ms,
        "payload": payload,
    }
    boot_sha256 = "6" * 64
    boottime_ns = 9_000_000_000
    gateway_start_ticks = 987654
    monkeypatch.setattr(runtime, "boot_identity", lambda: (boot_sha256, boottime_ns))
    monkeypatch.setattr(
        runtime,
        "process_start_time_ticks",
        lambda _pid: gateway_start_ticks,
    )
    monkeypatch.setattr(
        runtime,
        "_process_owner_ids",
        lambda _pid: (gateway_uid, gateway_gid),
    )
    receipt = {
        "schema": runtime.PLUGIN_READINESS_SCHEMA,
        "full_canary_plan_sha256": plan.sha256,
        "canary_run_id": fixture["canary_run_id"],
        "collector_readiness_file_sha256": collector.file_sha256,
        "gateway_peer": {
            "pid": gateway_pid,
            "start_time_ticks": gateway_start_ticks,
            "uid": gateway_uid,
            "gid": gateway_gid,
        },
        "plugin_ready_frame": frame,
        "plugin_ready_frame_sha256": runtime._sha256_json(frame),
        "collector_hash_chain_head_sha256": "5" * 64,
        "boot_id_sha256": boot_sha256,
        "observed_at_unix": now_ms // 1000,
        "observed_at_boottime_ns": boottime_ns,
    }
    receipt["receipt_sha256"] = runtime._sha256_json(receipt)
    plugin_raw = runtime._canonical_bytes(receipt)
    collector_path = tmp_path / "collector-readiness.json"
    plugin_path = tmp_path / "plugin-readiness.json"
    observer_path = tmp_path / "observer.json"
    payloads = {
        collector_path: collector_raw,
        plugin_path: plugin_raw,
        observer_path: observer_raw,
    }

    def fake_read(path: Path, **_kwargs):
        return payloads[Path(path)], object()

    monkeypatch.setattr(runtime, "DEFAULT_COLLECTOR_READINESS_PATH", collector_path)
    monkeypatch.setattr(runtime, "DEFAULT_PLUGIN_READINESS_PATH", plugin_path)
    monkeypatch.setattr(runtime, "DEFAULT_OBSERVER_CONFIG", observer_path)
    monkeypatch.setattr(runtime, "_read_stable_file", fake_read)
    monkeypatch.setattr(runtime, "_validated_e2e_fixture", lambda _plan: fixture)
    monkeypatch.setattr(
        runtime,
        "_observer_config_mapping",
        lambda *_args, **_kwargs: observer,
    )
    monkeypatch.setattr(
        runtime,
        "_socket_identity_sha256",
        lambda path, **_kwargs: (
            collector_socket_sha256
            if Path(path) == runtime.DEFAULT_COLLECTOR_SOCKET
            else edge_socket_sha256
        ),
    )

    loaded = runtime.load_plugin_readiness(
        plan,
        collector=collector,
        gateway_pid=gateway_pid,
        edge_pid=edge_pid,
        edge_service_identity_sha256=edge_identity_sha256,
        path=plugin_path,
    )
    assert loaded.frame_sha256 == receipt["plugin_ready_frame_sha256"]
    assert loaded.file_sha256 == hashlib.sha256(plugin_raw).hexdigest()

    tampered = copy.deepcopy(receipt)
    tampered["plugin_ready_frame"]["payload"]["gateway_pid"] = gateway_pid + 1
    tampered["plugin_ready_frame_sha256"] = runtime._sha256_json(
        tampered["plugin_ready_frame"]
    )
    tampered["receipt_sha256"] = runtime._sha256_json(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    payloads[plugin_path] = runtime._canonical_bytes(tampered)
    with pytest.raises(RuntimeError, match="sealed module/config binding"):
        runtime.load_plugin_readiness(
            plan,
            collector=collector,
            gateway_pid=gateway_pid,
            edge_pid=edge_pid,
            edge_service_identity_sha256=edge_identity_sha256,
            path=plugin_path,
        )


def test_verify_and_stop_success_runs_live_verifier_before_fixed_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.canonical_full_canary_e2e import _INVARIANTS

    base_plan = _host_receipt_plan(b"{}")
    plan = replace(
        base_plan,
        release={**base_plan.release, "interpreter": "/usr/bin/python3"},
        artifacts={
            **base_plan.artifacts,
            "e2e_fixture": ExactArtifact(
                source_path=Path("/tmp/e2e-fixture.json"),
                target_path=Path("/tmp/e2e-fixture.json"),
                sha256="9" * 64,
                mode=0o440,
                uid=0,
                gid=base_plan.identities.gateway_gid,
            ),
        },
    )
    events: list[str] = []
    start = runtime.LoadedStartReceipt(
        value={"receipt_sha256": "a" * 64},
        file_sha256="b" * 64,
    )
    evidence_sha256 = "c" * 64

    def runner(command):
        if command.argv[0] == runtime.SYSTEMCTL:
            events.append(f"stop:{command.argv[-1]}")
            payload = b""
        else:
            events.append("verify_command")
            payload = runtime._canonical_bytes(
                {
                    "schema": "muncho-full-canary-e2e-verification.v1",
                    "ok": True,
                    "fixture_sha256": plan.artifacts["e2e_fixture"].sha256,
                    "evidence_sha256": evidence_sha256,
                    "full_canary_start_receipt_sha256": start.file_sha256,
                    "invariants": list(_INVARIANTS),
                    "invariant_receipt_sha256": "d" * 64,
                }
            )
        return runtime.subprocess.CompletedProcess(
            command.argv,
            0,
            stdout=payload,
            stderr=b"",
        )

    monkeypatch.setattr(
        runtime,
        "_observe_dedicated_canary_host",
        lambda **_kwargs: events.append("host_proof") or {},
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_require_dedicated_host",
        lambda self: events.append("plan_host_proof") or {},
    )
    monkeypatch.setattr(
        runtime,
        "load_start_receipt",
        lambda *_args, **_kwargs: events.append("load_start") or start,
    )
    monkeypatch.setattr(runtime, "_lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_preflight",
        lambda self, *, phase: events.append(f"preflight:{phase}")
        or {"report_sha256": ("e" if phase == "live" else "f") * 64},
    )
    def write_receipt(_plan, *, stage, value):
        events.append(f"write:{stage}")
        return {**value, "receipt_path": "/tmp/verified.json"}

    monkeypatch.setattr(runtime, "_write_append_only_receipt", write_receipt)
    lifecycle = runtime.FullCanaryLifecycle(plan, runner=runner)
    result = lifecycle.verify_and_stop(
        start_receipt_path=Path("/tmp/start.json"),
        evidence_path=runtime.expected_live_evidence_path(plan),
        evidence_sha256=evidence_sha256,
    )

    assert result["verified"] is True
    assert events.index("verify_command") < events.index(
        f"stop:{GATEWAY_UNIT_NAME}"
    )
    assert [event for event in events if event.startswith("stop:")] == [
        f"stop:{GATEWAY_UNIT_NAME}",
        f"stop:{WRITER_UNIT_NAME}",
        f"stop:{EDGE_UNIT_NAME}",
        f"stop:{PHASE_B_READINESS_UNIT_NAME}",
    ]
    assert events[-1] == "write:verified_stopped"


def test_verify_and_stop_start_receipt_failure_still_stops_before_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    events: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_observe_dedicated_canary_host",
        lambda **_kwargs: events.append("host_proof") or {},
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_require_dedicated_host",
        lambda self: events.append("plan_host_proof") or {},
    )
    monkeypatch.setattr(
        runtime,
        "load_start_receipt",
        lambda *_args, **_kwargs: events.append("load_start")
        or (_ for _ in ()).throw(RuntimeError("start receipt tampered")),
    )
    monkeypatch.setattr(
        runtime,
        "_stop_all",
        lambda **_kwargs: events.append("mechanical_stop")
        or (GATEWAY_UNIT_NAME, WRITER_UNIT_NAME, EDGE_UNIT_NAME),
    )
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_preflight",
        lambda self, *, phase: {"report_sha256": "f" * 64},
    )

    with pytest.raises(RuntimeError, match="start receipt tampered"):
        runtime.FullCanaryLifecycle(plan).verify_and_stop(
            start_receipt_path=Path("/tmp/start.json"),
            evidence_path=runtime.expected_live_evidence_path(plan),
            evidence_sha256="a" * 64,
        )
    assert "mechanical_stop" in events


def test_lifecycle_stop_plan_host_tamper_cannot_delay_mechanical_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    events: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_observe_dedicated_canary_host",
        lambda **_kwargs: events.append("compile_host_proof") or {},
    )
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime,
        "_stop_all",
        lambda **_kwargs: events.append("mechanical_stop")
        or (GATEWAY_UNIT_NAME, WRITER_UNIT_NAME, EDGE_UNIT_NAME),
    )
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_require_dedicated_host",
        lambda self: events.append("plan_host_validation")
        or (_ for _ in ()).throw(RuntimeError("plan host truth tampered")),
    )

    with pytest.raises(RuntimeError, match="plan host truth tampered"):
        runtime.FullCanaryLifecycle(plan).stop()
    assert events == [
        "compile_host_proof",
        "mechanical_stop",
        "plan_host_validation",
    ]
