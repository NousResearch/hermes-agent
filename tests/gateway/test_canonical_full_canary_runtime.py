"""Focused contracts for the isolated full-canary runtime foundation."""

from __future__ import annotations

import copy
import base64
import builtins
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
    return """[Unit]
Description=Muncho privileged Canonical Writer (isolated canary)
Wants=network-online.target

[Service]
Type=notify
Restart=on-failure
RestartSec=5s

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


def test_preflight_reports_never_authorized_without_reconciliation_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    plan = replace(
        plan,
        writer_activation_receipt={
            "activation_receipt_path": "/missing/writer-receipt.json"
        },
    )
    evidence = runtime.BootstrapNeverAuthorizedEvidence(
        value={"receipt_sha256": "9" * 64},
        path=Path("/var/lib/muncho-full-canary/never.json"),
        file_sha256="8" * 64,
    )
    monkeypatch.setattr(
        runtime,
        "validate_dedicated_canary_host",
        lambda *_args, **_kwargs: {"host": "exact"},
    )
    monkeypatch.setattr(
        runtime,
        "_validate_bootstrap_lifecycle_state",
        lambda *_args, **_kwargs: "never_authorized_reconciled",
    )

    with pytest.raises(runtime.FullCanaryPreflightError) as raised:
        runtime.collect_full_canary_preflight(
            plan,
            phase="stopped",
            bootstrap_reconciliation_evidence=evidence,
            runner=lambda _command: (_ for _ in ()).throw(
                RuntimeError("service state unavailable")
            ),
        )

    assert raised.value.report["bootstrap_credential_state"] == (
        "never_authorized_reconciled"
    )
    assert (
        raised.value.report["bootstrap_reconciliation_receipt_sha256"]
        is None
    )


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
        bootstrap_provisioner=SimpleNamespace(
            provision=lambda _request: {},
            reconcile=lambda _request, _receipt: {},
            abort=lambda: None,
        ),
    )
    with pytest.raises(RuntimeError, match="failed closed"):
        lifecycle.start(_owner_approval(plan))
    assert metadata_calls > len(runtime._GCE_METADATA_PATHS)
    assert install_calls == []
    assert runner_calls == []


def test_systemd_bundle_has_one_narrow_api_credential_and_no_restart() -> None:
    bundle = FullCanarySystemdBundle.from_mapping(_bundle().to_mapping())
    credential = (
        f"LoadCredential={API_SERVER_CREDENTIAL_NAME}:"
        f"{DEFAULT_API_SERVER_CONTROL_KEY}"
    )
    assert bundle.gateway_service.count(credential) == 1
    assert "LoadCredential=" not in bundle.edge_service
    assert "LoadCredential=" not in bundle.writer_service
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
    assert (
        "Environment=HERMES_MANAGED_DIR="
        f"{runtime.DEFAULT_DISABLED_MANAGED_SCOPE}"
    ) in bundle.gateway_service
    assert "InaccessiblePaths=-/etc/hermes" in bundle.gateway_service
    assert bundle.gateway_service.count(
        runtime._GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE
    ) == 1
    for name in (
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
    ):
        assert name in runtime._GATEWAY_SYSTEMD_UNSET_ENVIRONMENT_NAMES
    assert (
        f"Environment=SSL_CERT_FILE={runtime.DEFAULT_GATEWAY_CA_BUNDLE}"
        in bundle.gateway_service
    )
    assert (
        f"AssertPathExists={runtime.DEFAULT_GATEWAY_CA_BUNDLE}"
        in bundle.gateway_service
    )
    assert (
        f"ReadOnlyPaths={runtime.DEFAULT_GATEWAY_CA_BUNDLE}"
        in bundle.gateway_service
    )
    assert (
        f"InaccessiblePaths={runtime.DEFAULT_DISABLED_MANAGED_SCOPE}"
        not in bundle.gateway_service
    )
    assert str(runtime.DEFAULT_DISABLED_MANAGED_SCOPE) not in bundle.tmpfiles
    for path in runtime._GATEWAY_SEALED_EMPTY_ENVIRONMENT_FILES:
        assert f"ReadOnlyPaths={path}" in bundle.gateway_service
        assert f"InaccessiblePaths={path}" not in bundle.gateway_service
        assert f"f+ {path} 0444 root root - -" in bundle.tmpfiles
    assert (
        f"InaccessiblePaths={runtime.DEFAULT_GATEWAY_USER_PLUGIN_ROOT}"
        in bundle.gateway_service
    )
    assert (
        f"d {runtime.DEFAULT_GATEWAY_USER_PLUGIN_ROOT} 0000 root root - -"
        in bundle.tmpfiles
    )
    for path in runtime._GATEWAY_INACCESSIBLE_SEMANTIC_FILES:
        assert f"InaccessiblePaths={path}" in bundle.gateway_service
        assert f"f {path} 0000 root root - -" in bundle.tmpfiles
    for path in runtime._GATEWAY_INACCESSIBLE_SEMANTIC_DIRECTORIES:
        assert f"InaccessiblePaths={path}" in bundle.gateway_service
        assert f"d {path} 0000 root root - -" in bundle.tmpfiles
    interpreter = (
        Path("/opt/muncho-canary-releases") / REVISION / "venv/bin/python"
    )
    preclaim_reconciliation_argv = (
        f"{interpreter} -B -I -m "
        "gateway.canonical_writer_bootstrap "
        f"--config {runtime.DEFAULT_WRITER_CONFIG_SOURCE} "
        "--reconcile-canary-preclaim"
    )
    assert bundle.writer_service.count(
        f"ExecStartPre={preclaim_reconciliation_argv}"
    ) == 1
    assert bundle.writer_service.count(
        f"ExecStopPost={preclaim_reconciliation_argv}"
    ) == 1
    assert bundle.writer_service.count(
        f"ReadOnlyPaths={runtime.DEFAULT_WRITER_CONFIG_SOURCE}"
    ) == 1
    assert "--reconcile-canary-preclaim" not in bundle.edge_service
    assert "--reconcile-canary-preclaim" not in bundle.gateway_service
    assert (
        f"d {DEFAULT_COLLECTOR_RUNTIME} 0750 root "
        f"{_identities().gateway_group} - -"
    ) in bundle.tmpfiles


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
def test_systemd_bundle_rejects_preclaim_reconciliation_drift(
    mutation: str,
) -> None:
    mapping = copy.deepcopy(_bundle().to_mapping())
    if mutation == "missing":
        mapping["writer_service"] = "\n".join(
            line
            for line in mapping["writer_service"].splitlines()
            if "--reconcile-canary-preclaim" not in line
        ) + "\n"
    else:
        mapping["writer_service"] = mapping["writer_service"].replace(
            str(runtime.DEFAULT_WRITER_CONFIG_SOURCE),
            str(runtime.DEFAULT_WRITER_CONFIG),
            1,
        )
    unsigned = {key: value for key, value in mapping.items() if key != "sha256"}
    mapping["sha256"] = _canonical_digest(unsigned)
    with pytest.raises(ValueError, match="preclaim reconciliation boundary"):
        FullCanarySystemdBundle.from_mapping(mapping)


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
            "adaptive_reasoning": {"enabled": True, "max_effort": "xhigh"},
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
    )
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
        "max_effort": "xhigh",
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
        "valid_from_unix_ms": now_ms - 10_000,
        "valid_until_unix_ms": now_ms + 60_000,
    }
    observer = {
        "schema": "muncho-canary-evidence-config.v1",
        "canonical_scope": {"grant_id": "grant:test"},
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
        "canonical_scope_sha256": runtime._sha256_json(
            observer["canonical_scope"]
        ),
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


def test_writer_scope_expiry_is_exactly_fixture_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = _identities()
    fixture_sha256 = "7" * 64
    release_sha256 = "8" * 64
    valid_until_ms = (int(time.time()) + 600) * 1000
    fixture = {
        "case_id": "case:scope-expiry",
        "canary_run_id": "b3f27d4b-43f7-4cf5-9a72-4c1659352901",
        "owner_discord_user_id": "1279454038731264061",
        "valid_until_unix_ms": valid_until_ms,
    }
    plan = FullCanaryPlan(
        revision=REVISION,
        release={"artifact_sha256": release_sha256},
        identities=identities,
        writer_activation_plan={},
        writer_activation_receipt={},
        writer_activation_receipt_file_sha256="9" * 64,
        artifacts={
            "e2e_fixture": ExactArtifact(
                source_path=tmp_path / "fixture.json",
                target_path=tmp_path / "fixture.json",
                sha256=fixture_sha256,
                mode=0o440,
                uid=0,
                gid=identities.gateway_gid,
            )
        },
        allowed_previous_sha256={},
        unit_bundle=_bundle(),
        unit_paths={},
        e2e_verifier_module="gateway.canonical_full_canary_e2e",
        sha256="a" * 64,
    )
    approval_source = "b" * 64
    hba_receipt = {
        "version": "managed-cloudsqladmin-hba-rejection-v2",
        "host": "10.0.0.8",
        "tls_server_name": "db.internal",
        "port": 5432,
        "server_certificate_sha256": "d" * 64,
        "database": "cloudsqladmin",
        "user": "canonical_brain_canary_bootstrap_login",
        "observed_at_unix": int(time.time()),
        "expires_at_unix": int(time.time()) + 300,
        "sqlstate": "28000",
        "server_message": (
            'no pg_hba.conf entry for host "10.0.0.8", user '
            '"canonical_brain_canary_bootstrap_login", database '
            '"cloudsqladmin", SSL encryption'
        ),
        "result": "pg_hba_rejected",
        "tls_peer_verified": True,
    }
    hba_digest = runtime.managed_cloudsqladmin_hba_receipt_from_mapping(
        hba_receipt
    ).sha256
    config = {
        "service": {
            "socket_path": "/run/muncho-canonical-writer/writer.sock",
            "gateway_unit": GATEWAY_UNIT_NAME,
            "gateway_uid": identities.gateway_uid,
            "writer_uid": identities.writer_uid,
            "writer_gid": identities.writer_gid,
            "socket_gid": identities.socket_client_gid,
            "projector_gid": identities.socket_client_gid + 100,
            "owner_discord_user_ids": [fixture["owner_discord_user_id"]],
            "connection_timeout_seconds": 30.0,
            "max_connections": 8,
        },
        "database": {
            "host": "10.0.0.8",
            "tls_server_name": "db.internal",
            "port": 5432,
            "database": "muncho_canary_brain",
            "user": "canonical_brain_writer_login",
            "ca_file": "/etc/muncho/trust/cloudsql-server-ca.pem",
            "credential_file": "/etc/muncho/credentials/canonical-writer-db-password",
            "connect_timeout_seconds": 5.0,
            "io_timeout_seconds": 10.0,
        },
        "privileges": {
            "schema": "canonical_brain",
            "table_grants": [],
            "routine_identities": [],
            "helper_routine_identities": [],
            "schema_privileges": ["USAGE"],
            "database_privileges": ["CONNECT"],
            "role_memberships": ["canonical_brain_writer"],
            "private_schema_identity_sha256": "e" * 64,
            "managed_cloudsqladmin_hba_rejection_receipt": hba_receipt,
            "managed_cloudsqladmin_hba_rejection_sha256": hba_digest,
            "deployment_lock_key": 4_841_739_663_211_427_921,
        },
        "discord_edge_authority": {
            "enabled": True,
            "capability_private_key_file": (
                "/etc/muncho/keys/writer-capability-private.pem"
            ),
            "edge_receipt_public_key_file": (
                "/etc/muncho/keys/discord-edge-receipt-public.pem"
            ),
            "edge_receipt_public_key_id": "f" * 64,
            "request_timeout_seconds": 15,
        },
        "canary_scope_preapproval": {
            "grant_id": "grant:fixture-expiry",
            "case_id": fixture["case_id"],
            "release_sha256": release_sha256,
            "fixture_sha256": fixture_sha256,
            "run_id": fixture["canary_run_id"],
            "session_key_sha256": "c" * 64,
            "expires_at": datetime.fromtimestamp(
                valid_until_ms // 1000,
                tz=timezone.utc,
            ).isoformat(),
            "approved_by": fixture["owner_discord_user_id"],
            "approval_source_sha256": approval_source,
            "provisioning_receipt_sha256": "d" * 64,
            "bootstrap_database_user": (
                "canonical_brain_canary_bootstrap_login"
            ),
            "bootstrap_credential_file": str(
                runtime.DEFAULT_CANARY_BOOTSTRAP_CREDENTIAL
            ),
            "bootstrap_managed_cloudsqladmin_hba_rejection_receipt": (
                hba_receipt
            ),
            "bootstrap_managed_cloudsqladmin_hba_rejection_sha256": (
                hba_digest
            ),
        },
    }
    monkeypatch.setattr(runtime, "_validated_e2e_fixture", lambda _plan: fixture)
    assert _validate_writer_config(
        runtime._canonical_bytes(config),
        identities,
        plan=plan,
        expected_approval_source_sha256=approval_source,
    )
    for section in ("service", "database", "privileges"):
        secret_bearing = copy.deepcopy(config)
        secret_bearing[section]["password"] = "must-never-enter-evidence"
        with pytest.raises(RuntimeError, match="not exact|schema"):
            _validate_writer_config(
                runtime._canonical_bytes(secret_bearing),
                identities,
                plan=plan,
                expected_approval_source_sha256=approval_source,
            )

    expired_valid_until_ms = (int(time.time()) - 600) * 1000
    expired_fixture = {
        **fixture,
        "valid_until_unix_ms": expired_valid_until_ms,
    }
    expired_config = copy.deepcopy(config)
    expired_config["canary_scope_preapproval"]["expires_at"] = (
        datetime.fromtimestamp(
            expired_valid_until_ms // 1000,
            tz=timezone.utc,
        ).isoformat()
    )
    monkeypatch.setattr(
        runtime,
        "_validated_e2e_fixture",
        lambda _plan: expired_fixture,
    )
    with pytest.raises(RuntimeError, match="fixture-bound/fresh"):
        _validate_writer_config(
            runtime._canonical_bytes(expired_config),
            identities,
            plan=plan,
            expected_approval_source_sha256=approval_source,
        )
    assert _validate_writer_config(
        runtime._canonical_bytes(expired_config),
        identities,
        plan=plan,
        expected_approval_source_sha256=approval_source,
        require_fresh_canary_scope=False,
    )
    monkeypatch.setattr(runtime, "_validated_e2e_fixture", lambda _plan: fixture)

    drifted = copy.deepcopy(config)
    drifted_expiry = datetime.fromtimestamp(
        valid_until_ms // 1000 + 1,
        tz=timezone.utc,
    ).isoformat()
    drifted["canary_scope_preapproval"]["expires_at"] = drifted_expiry
    with pytest.raises(RuntimeError, match="fixture-bound"):
        _validate_writer_config(
            runtime._canonical_bytes(drifted),
            identities,
            plan=plan,
            expected_approval_source_sha256=approval_source,
        )


def _bootstrap_intent_config() -> dict:
    return {
        "database": {
            "host": "10.0.0.8",
            "tls_server_name": "db.internal",
            "port": 5432,
            "database": "muncho_canary_brain",
            "user": "canonical_brain_writer_login",
        },
        "canary_scope_preapproval": {
            "grant_id": "grant:bootstrap-intent",
            "case_id": "case:bootstrap-intent",
            "release_sha256": ARTIFACT_SHA256,
            "fixture_sha256": "4" * 64,
            "run_id": "run:bootstrap-intent",
            "session_key_sha256": "5" * 64,
            "expires_at": "2026-07-13T12:30:00+00:00",
            "approved_by": "1279454038731264061",
            "approval_source_sha256": "2" * 64,
            "provisioning_receipt_sha256": "0" * 64,
            "bootstrap_database_user": (
                "canonical_brain_canary_bootstrap_login"
            ),
        },
    }


def test_bootstrap_request_binds_fixed_sql_and_exact_eleven_gucs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    config = _bootstrap_intent_config()
    sql = b"BEGIN;\nSELECT 1;\nCOMMIT;\n"
    sql_sha256 = hashlib.sha256(sql).hexdigest()
    authorization = runtime.canonical_canary_bootstrap_authorization_sha256(
        config,
        bootstrap_sql_sha256=sql_sha256,
        bootstrap_retire_sql_sha256="8" * 64,
    )
    config["canary_scope_preapproval"][
        "provisioning_receipt_sha256"
    ] = authorization
    sql_path = Path(plan.release.get("artifact_root", "/opt/release")) / (
        runtime.DEFAULT_CANARY_BOOTSTRAP_SQL_RELATIVE
    )
    monkeypatch.setattr(
        runtime,
        "_validated_release_file",
        lambda _plan, relative, **_kwargs: (
            (sql_path, sql, sql_sha256)
            if relative == runtime.DEFAULT_CANARY_BOOTSTRAP_SQL_RELATIVE
            else (
                sql_path.with_name("canonical_writer_canary_bootstrap_retire_v1.sql"),
                b"BEGIN; COMMIT;",
                "8" * 64,
            )
        ),
    )

    request = runtime._build_canary_bootstrap_provisioning_request(
        plan,
        approval,
        config,
    )

    assert len(request.guc_bindings) == 11
    assert set(request.guc_bindings) == {
        "muncho.canonical_canary_bootstrap_database",
        "muncho.canonical_canary_bootstrap_grant_id",
        "muncho.canonical_canary_bootstrap_case_id",
        "muncho.canonical_canary_bootstrap_release_sha256",
        "muncho.canonical_canary_bootstrap_fixture_sha256",
        "muncho.canonical_canary_bootstrap_run_id",
        "muncho.canonical_canary_bootstrap_session_key_sha256",
        "muncho.canonical_canary_bootstrap_expires_at",
        "muncho.canonical_canary_bootstrap_approved_by",
        "muncho.canonical_canary_bootstrap_approval_source_sha256",
        "muncho.canonical_canary_bootstrap_provisioning_receipt_sha256",
    }
    assert request.authorization_receipt_sha256 == authorization
    assert request.guc_bindings[
        "muncho.canonical_canary_bootstrap_provisioning_receipt_sha256"
    ] == authorization
    assert request.sql_bytes == sql
    assert "SELECT 1" not in repr(request)


def test_bootstrap_request_rejects_free_provisioning_digest_before_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    sql = b"BEGIN; COMMIT;"
    monkeypatch.setattr(
        runtime,
        "_validated_release_file",
        lambda _plan, relative, **_kwargs: (
            Path("/opt/release") / relative,
            sql,
            hashlib.sha256(sql).hexdigest(),
        ),
    )
    with pytest.raises(RuntimeError, match="authorization digest drifted"):
        runtime._build_canary_bootstrap_provisioning_request(
            plan,
            _owner_approval(plan),
            _bootstrap_intent_config(),
        )


def test_default_missing_admin_session_blocks_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    mutations: list[str] = []
    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_require_dedicated_host",
        lambda self: {"exact": True},
    )
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda *_a, **_k: b"{}",
    )
    monkeypatch.setattr(runtime, "_validate_writer_config", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runtime,
        "_lifecycle_lock",
        lambda: mutations.append("lock"),
    )
    monkeypatch.setattr(
        runtime,
        "_install_plan_artifacts",
        lambda _plan: mutations.append("install"),
    )

    with pytest.raises(
        runtime.FullCanaryBootstrapAdminUnavailable,
        match="ephemeral bootstrap admin",
    ):
        runtime.FullCanaryLifecycle(plan).start(_owner_approval(plan))
    assert mutations == []


def test_writer_consumption_receipt_allows_deterministic_config_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    config = _bootstrap_intent_config()
    scope = config["canary_scope_preapproval"]
    consumption = {
        "version": "canonical-canary-bootstrap-consumption-v1",
        **{
            name: scope[name]
            for name in (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "session_key_sha256",
                "expires_at",
                "approved_by",
                "approval_source_sha256",
                "provisioning_receipt_sha256",
            )
        },
        "bootstrap_database_user": (
            "canonical_brain_canary_bootstrap_login"
        ),
        "preapproval_event_id": "11111111-1111-4111-8111-111111111111",
        "consumption_event_id": "22222222-2222-4222-8222-222222222222",
        "preapproved_at": "2026-07-13T12:00:00+00:00",
        "acl_revoked": True,
        "inserted": True,
    }
    consumption["receipt_sha256"] = runtime._sha256_json(consumption)
    readiness = {
        "version": runtime.WRITER_RUNTIME_ATTESTATION_VERSION,
        "canary_scope_bootstrap_consumption": consumption,
    }
    raw = runtime._canonical_bytes(config)
    installed: dict[str, object] = {}
    tombstones: list[bytes] = []
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda *_a, **_k: raw,
    )
    monkeypatch.setattr(
        runtime,
        "_validate_writer_config",
        lambda *_a, **_k: config,
    )
    retired = copy.deepcopy(config)
    retired.pop("canary_scope_preapproval")
    retired_payload = runtime._canonical_bytes(retired)
    monkeypatch.setattr(
        runtime,
        "_retired_writer_config_payload",
        lambda _plan: (
            retired_payload,
            hashlib.sha256(retired_payload).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_install_payload",
        lambda _plan, **kwargs: installed.update(kwargs) or {"changed": True},
    )
    monkeypatch.setattr(
        runtime,
        "_write_exclusive_bytes",
        lambda _path, payload, **_kwargs: tombstones.append(payload),
    )

    result = runtime._retire_writer_bootstrap_config(
        plan,
        writer_readiness=readiness,
    )

    assert b"canary_scope_preapproval" not in installed["payload"]
    tombstone = json.loads(tombstones[0])
    assert tombstone["bootstrap_consumption"] == consumption
    assert tombstone["retired_writer_config_sha256"] == hashlib.sha256(
        installed["payload"]
    ).hexdigest()
    assert result["tombstone"]["receipt_sha256"] == tombstone["receipt_sha256"]


def _preclaim_reconciliation_receipt(
    config: dict,
    *,
    outcome: str,
    inserted: bool = True,
) -> tuple[bytes, dict]:
    scope = config["canary_scope_preapproval"]
    event_ids = {
        "preapproval_event_id": "11111111-1111-4111-8111-111111111111",
        "bootstrap_consumption_event_id": (
            "22222222-2222-4222-8222-222222222222"
        ),
        "claim_event_id": "33333333-3333-4333-8333-333333333333",
        "retirement_event_id": "44444444-4444-4444-8444-444444444444",
        "revocation_event_id": "55555555-5555-4555-8555-555555555555",
    }
    result = {
        "success": True,
        "outcome": outcome,
        **{
            name: scope[name]
            for name in (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "session_key_sha256",
                "expires_at",
                "approved_by",
                "approval_source_sha256",
                "provisioning_receipt_sha256",
            )
        },
        **event_ids,
        "claimed_at": "2026-07-13T12:01:00+00:00",
        "retired_at": "2026-07-13T12:02:00+00:00",
        "reason": "",
        "scope_retired": False,
        "authority_active": False,
        "inserted": inserted,
        "deduped": not inserted,
    }
    if outcome == "not_preapproved":
        for name in event_ids:
            result[name] = None
        result.update(
            claimed_at=None,
            retired_at=None,
            reason="preapproval_not_committed",
            inserted=False,
            deduped=False,
        )
    elif outcome == "retired":
        result.update(
            claim_event_id=None,
            revocation_event_id=None,
            claimed_at=None,
            reason="activation_failed_before_first_claim",
            scope_retired=True,
        )
    elif outcome == "claimed":
        result.update(
            retirement_event_id=None,
            retired_at=None,
            reason="claim_already_committed_session_retired",
        )
    else:
        raise AssertionError(outcome)
    source = runtime._canonical_bytes(config)
    database_identity = {
        name: config["database"][name]
        for name in ("host", "tls_server_name", "port", "database", "user")
    }
    unsigned = {
        "version": runtime.CANARY_PRECLAIM_RECONCILIATION_VERSION,
        "observed_at_unix": 1_784_000_000,
        "source_config_path": str(runtime.DEFAULT_WRITER_CONFIG_SOURCE),
        "source_config_sha256": hashlib.sha256(source).hexdigest(),
        "database_identity": database_identity,
        "database_identity_sha256": runtime._sha256_json(database_identity),
        "result": result,
    }
    return source, {
        **unsigned,
        "receipt_sha256": runtime._sha256_json(unsigned),
    }


@pytest.mark.parametrize(
    "outcome,inserted",
    [
        ("not_preapproved", False),
        ("retired", True),
        ("retired", False),
        ("claimed", True),
        ("claimed", False),
    ],
)
def test_preclaim_reconciliation_accepts_only_exact_inactive_terminal_matrix(
    outcome: str,
    inserted: bool,
) -> None:
    config = _bootstrap_intent_config()
    source, receipt = _preclaim_reconciliation_receipt(
        config,
        outcome=outcome,
        inserted=inserted,
    )
    validated = runtime._validate_canary_preclaim_reconciliation_value(
        receipt,
        source_config_path=runtime.DEFAULT_WRITER_CONFIG_SOURCE,
        source_config_raw=source,
        writer_config=config,
        allowed_outcomes=frozenset({outcome}),
    )
    assert validated["result"]["outcome"] == outcome
    assert validated["result"]["authority_active"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "active",
        "source_digest",
        "database",
        "event_matrix",
        "wrong_outcome",
    ],
)
def test_preclaim_reconciliation_rejects_wrapper_or_terminal_drift(
    mutation: str,
) -> None:
    config = _bootstrap_intent_config()
    source, receipt = _preclaim_reconciliation_receipt(
        config,
        outcome="claimed",
    )
    if mutation == "active":
        receipt["result"]["authority_active"] = True
    elif mutation == "source_digest":
        receipt["source_config_sha256"] = "0" * 64
    elif mutation == "database":
        receipt["database_identity"]["database"] = "other"
    elif mutation == "event_matrix":
        receipt["result"]["retirement_event_id"] = (
            "66666666-6666-4666-8666-666666666666"
        )
    else:
        receipt["result"]["outcome"] = "retired"
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = runtime._sha256_json(unsigned)
    with pytest.raises(RuntimeError):
        runtime._validate_canary_preclaim_reconciliation_value(
            receipt,
            source_config_path=runtime.DEFAULT_WRITER_CONFIG_SOURCE,
            source_config_raw=source,
            writer_config=config,
            allowed_outcomes=frozenset({"claimed"}),
        )


def _direct_bootstrap_request(
    plan: FullCanaryPlan,
    approval: runtime.FullCanaryOwnerApproval,
) -> runtime.CanaryBootstrapProvisioningRequest:
    config = _bootstrap_intent_config()
    provision_sql = b"BEGIN;\nCOMMIT;\n"
    retire_sql = b"BEGIN;\nSELECT true;\nCOMMIT;\n"
    provision_digest = hashlib.sha256(provision_sql).hexdigest()
    retire_digest = hashlib.sha256(retire_sql).hexdigest()
    authorization = runtime.canonical_canary_bootstrap_authorization_sha256(
        config,
        bootstrap_sql_sha256=provision_digest,
        bootstrap_retire_sql_sha256=retire_digest,
    )
    config["canary_scope_preapproval"][
        "provisioning_receipt_sha256"
    ] = authorization
    gucs = {
        **runtime._canary_bootstrap_business_gucs(config),
        "muncho.canonical_canary_bootstrap_provisioning_receipt_sha256": (
            authorization
        ),
    }
    return runtime.CanaryBootstrapProvisioningRequest(
        plan_sha256=plan.sha256,
        owner_approval_sha256=approval.sha256,
        approval_source_sha256=str(
            approval.value["approval_source_sha256"]
        ),
        authorization_receipt_sha256=authorization,
        database={
            name: copy.deepcopy(config["database"][name])
            for name in ("host", "tls_server_name", "port", "database")
        },
        guc_bindings=gucs,
        guc_bindings_sha256=runtime._sha256_json(gucs),
        sql_path=Path("/opt/release/scripts/sql/canonical_writer_canary_bootstrap_v1.sql"),
        sql_sha256=provision_digest,
        sql_bytes=provision_sql,
        retire_sql_path=Path(
            "/opt/release/scripts/sql/"
            "canonical_writer_canary_bootstrap_retire_v1.sql"
        ),
        retire_sql_sha256=retire_digest,
        retire_sql_bytes=retire_sql,
    )


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        ("endpoint", "host"),
        ("endpoint", "tls_server_name"),
        ("endpoint", "port"),
        ("endpoint", "database"),
        *[
            ("guc", name)
            for name in sorted(
                {
                    "muncho.canonical_canary_bootstrap_database",
                    "muncho.canonical_canary_bootstrap_grant_id",
                    "muncho.canonical_canary_bootstrap_case_id",
                    "muncho.canonical_canary_bootstrap_release_sha256",
                    "muncho.canonical_canary_bootstrap_fixture_sha256",
                    "muncho.canonical_canary_bootstrap_run_id",
                    "muncho.canonical_canary_bootstrap_session_key_sha256",
                    "muncho.canonical_canary_bootstrap_expires_at",
                    "muncho.canonical_canary_bootstrap_approved_by",
                    "muncho.canonical_canary_bootstrap_approval_source_sha256",
                    "muncho.canonical_canary_bootstrap_provisioning_receipt_sha256",
                }
            )
        ],
    ],
)
def test_bootstrap_request_every_endpoint_and_guc_is_writer_artifact_derived(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    name: str,
) -> None:
    writer_raw = b"sealed-secret-free-writer-config"
    plan = _host_receipt_plan(b"{}")
    plan = replace(
        plan,
        artifacts={
            **dict(plan.artifacts),
            "writer_config": replace(
                plan.artifacts["writer_config"],
                sha256=hashlib.sha256(writer_raw).hexdigest(),
            ),
        },
    )
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    writer_config = _bootstrap_intent_config()
    writer_config["canary_scope_preapproval"][
        "provisioning_receipt_sha256"
    ] = request.authorization_receipt_sha256
    monkeypatch.setattr(
        runtime,
        "_validate_writer_config",
        lambda *_args, **_kwargs: writer_config,
    )
    runtime._bind_bootstrap_request_to_writer_config(
        plan=plan,
        approval=approval,
        request=request,
        writer_config_raw=writer_raw,
    )

    if kind == "endpoint":
        endpoint = dict(request.database)
        endpoint[name] = (
            endpoint[name] + 1
            if name == "port"
            else str(endpoint[name]) + "-tampered"
        )
        tampered = replace(request, database=endpoint)
    else:
        gucs = dict(request.guc_bindings)
        gucs[name] = gucs[name] + "-tampered"
        tampered = replace(
            request,
            guc_bindings=gucs,
            guc_bindings_sha256=runtime._sha256_json(gucs),
            authorization_receipt_sha256=(
                gucs[name]
                if name
                == "muncho.canonical_canary_bootstrap_provisioning_receipt_sha256"
                else request.authorization_receipt_sha256
            ),
        )
    with pytest.raises(RuntimeError, match="not derived"):
        runtime._bind_bootstrap_request_to_writer_config(
            plan=plan,
            approval=approval,
            request=tampered,
            writer_config_raw=writer_raw,
        )


@pytest.mark.parametrize(
    (
        "phase",
        "credential_present",
        "target_state",
        "tombstone_present",
        "outcome",
        "expected",
    ),
    [
        ("stopped", True, "configured", False, None, "armed"),
        ("live", True, "configured", False, None, None),
        (
            "stopped",
            False,
            "consumed_retired",
            True,
            "consumed",
            "consumed_retired",
        ),
        (
            "live",
            False,
            "consumed_retired",
            True,
            "consumed",
            "consumed_retired",
        ),
        (
            "stopped",
            False,
            "configured",
            False,
            "retired",
            "aborted_reconciled",
        ),
        ("live", False, "configured", False, "retired", None),
        ("stopped", False, "consumed_retired", True, None, None),
        (
            "stopped",
            True,
            "consumed_retired",
            True,
            "consumed",
            None,
        ),
    ],
)
def test_bootstrap_preflight_accepts_only_exact_lifecycle_matrix(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    credential_present: bool,
    target_state: str,
    tombstone_present: bool,
    outcome: str | None,
    expected: str | None,
) -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    evidence = (
        None
        if outcome is None
        else runtime.BootstrapReconciliationEvidence(
            plan_sha256=plan.sha256,
            approval=approval,
            request=request,
            provisioning_receipt=None,
            reconciliation_receipt={"outcome": outcome},
        )
    )
    monkeypatch.setattr(
        runtime,
        "_bootstrap_credential_is_present_and_exact",
        lambda _plan: credential_present,
    )
    monkeypatch.setattr(
        runtime,
        "_writer_target_bootstrap_state",
        lambda _plan: target_state,
    )
    monkeypatch.setattr(
        runtime.os.path,
        "lexists",
        lambda _path: tombstone_present,
    )
    monkeypatch.setattr(
        runtime,
        "_revalidate_bootstrap_reconciliation_evidence",
        lambda _plan, value: value.reconciliation_receipt,
    )

    if expected is None:
        with pytest.raises(RuntimeError, match="bootstrap credential"):
            runtime._validate_bootstrap_lifecycle_state(
                plan,
                phase=phase,
                evidence=evidence,
            )
    else:
        assert runtime._validate_bootstrap_lifecycle_state(
            plan,
            phase=phase,
            evidence=evidence,
        ) == expected


class _PreopenedSession:
    def __init__(
        self,
        *,
        cleanup_outcome: str = "retired",
        cleanup_inserted: bool = True,
        backend_pid: str = "42",
    ) -> None:
        self.queries: list[tuple[str, int]] = []
        self.closed = False
        self.close_count = 0
        self.identity_count = 0
        self.cleanup_outcome = cleanup_outcome
        self.cleanup_inserted = cleanup_inserted
        self.backend_pid = backend_pid

    def query(self, sql: str, *, maximum_rows: int):
        self.queries.append((sql, maximum_rows))
        if sql == runtime.PreopenedSessionBootstrapProvisioner._IDENTITY_SQL:
            self.identity_count += 1
            return SimpleNamespace(
                columns=(
                    "database_name",
                    "session_user",
                    "current_user",
                    "backend_pid",
                ),
                rows=(
                    (
                        "muncho_canary_brain",
                        "managed_admin",
                        "managed_admin",
                        self.backend_pid,
                    ),
                ),
                command_tag="SELECT 1",
            )
        if sql == "ROLLBACK":
            return SimpleNamespace(columns=(), rows=(), command_tag="ROLLBACK")
        encoded = re.findall(
            r"pg_catalog\.decode\('([A-Za-z0-9+/=]+)'",
            sql,
        )
        assert len(encoded) % 2 == 0
        settings = [
            (
                base64.b64decode(encoded[index]).decode(),
                base64.b64decode(encoded[index + 1]).decode(),
            )
            for index in range(0, len(encoded), 2)
        ]
        setting_rows = tuple((value,) for _name, value in settings)
        if len(settings) == 14:
            bindings = dict(settings)
            outcome_rows = {
                "retired": (
                    "11111111-1111-4111-8111-111111111111",
                    None,
                    "22222222-2222-4222-8222-222222222222",
                    "t" if self.cleanup_inserted else "f",
                    "activation_failed_before_consumption",
                ),
                "consumed": (
                    "11111111-1111-4111-8111-111111111111",
                    "33333333-3333-4333-8333-333333333333",
                    None,
                    "f",
                    "bootstrap_consumed",
                ),
                "not_authorized": (
                    None,
                    None,
                    None,
                    "f",
                    "provisioning_not_committed",
                ),
            }
            authorization_id, consumption_id, retirement_id, inserted, reason = (
                outcome_rows[self.cleanup_outcome]
            )
            return SimpleNamespace(
                columns=(
                    "outcome",
                    "grant_id",
                    "case_id",
                    "authorization_event_id",
                    "consumption_event_id",
                    "retirement_event_id",
                    "retired_inserted",
                    "plan_sha256",
                    "owner_approval_sha256",
                    "executor_session_identity_sha256",
                    "reason",
                    "bootstrap_acl_revoked",
                    "migration_owner_membership_absent",
                ),
                rows=setting_rows
                + (
                    ("",),
                    (
                        self.cleanup_outcome,
                        bindings[
                            "muncho.canonical_canary_bootstrap_grant_id"
                        ],
                        bindings[
                            "muncho.canonical_canary_bootstrap_case_id"
                        ],
                        authorization_id,
                        consumption_id,
                        retirement_id,
                        inserted,
                        bindings[
                            "muncho.canonical_canary_bootstrap_plan_sha256"
                        ],
                        bindings[
                            "muncho.canonical_canary_bootstrap_owner_approval_sha256"
                        ],
                        bindings[
                            "muncho.canonical_canary_bootstrap_executor_session_identity_sha256"
                        ],
                        reason,
                        "t",
                        "t",
                    ),
                ),
                command_tag="COMMIT",
            )
        assert len(settings) == 11
        return SimpleNamespace(
            columns=("set_config",),
            rows=setting_rows + (("",),),
            command_tag="COMMIT",
        )

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


def test_preopened_admin_session_applies_and_reconciles_without_secret_surface(
) -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    session = _PreopenedSession()
    provisioner = runtime.PreopenedSessionBootstrapProvisioner(
        session,
        tls_peer_certificate_sha256="a" * 64,
    )

    provision = provisioner.provision(request)
    validated_provision = runtime._validate_canary_bootstrap_provisioning_receipt(
        provision,
        request=request,
        approval=approval,
    )
    reconciliation = provisioner.reconcile(request, provision)
    validated_reconciliation = (
        runtime._validate_canary_bootstrap_reconciliation_receipt(
            reconciliation,
            request=request,
            provisioning_receipt=provision,
            approval=approval,
        )
    )

    assert validated_provision["executor_session_identity"]["session_user"] == (
        "managed_admin"
    )
    assert validated_reconciliation["bootstrap_acl_revoked"] is True
    assert validated_reconciliation["outcome"] == "retired"
    assert validated_reconciliation["session_continuity"] == (
        "same_provision_session"
    )
    assert session.identity_count == 2
    assert session.closed is True
    provisioner.abort()
    provisioner.abort()
    assert session.close_count == 1
    rendered = "\n".join(sql for sql, _maximum in session.queries)
    assert "password" not in rendered.casefold()
    assert "PGPASSWORD" not in rendered
    assert len(request.guc_bindings) == 11
    cleanup_sql = session.queries[-1][0]
    assert cleanup_sql.count("SELECT pg_catalog.set_config(") == 14


@pytest.mark.parametrize(
    ("outcome", "inserted", "reason"),
    [
        ("retired", True, "activation_failed_before_consumption"),
        ("retired", False, "activation_failed_before_consumption"),
        ("consumed", False, "bootstrap_consumed"),
        ("not_authorized", False, "provisioning_not_committed"),
    ],
)
def test_reconciliation_binds_each_exact_terminal_outcome(
    outcome: str,
    inserted: bool,
    reason: str,
) -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    session = _PreopenedSession(
        cleanup_outcome=outcome,
        cleanup_inserted=inserted,
    )
    provisioner = runtime.PreopenedSessionBootstrapProvisioner(
        session,
        tls_peer_certificate_sha256="a" * 64,
    )

    provision = provisioner.provision(request)
    reconciliation = provisioner.reconcile(request, provision)
    validated = runtime._validate_canary_bootstrap_reconciliation_receipt(
        reconciliation,
        request=request,
        provisioning_receipt=provision,
        approval=approval,
        expected_session_continuity="same_provision_session",
    )

    assert validated["outcome"] == outcome
    assert validated["retired_inserted"] is inserted
    assert validated["reason"] == reason


def test_reconciliation_reobserves_and_rejects_auto_reconnected_backend() -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)

    class ReconnectedSession(_PreopenedSession):
        def query(self, sql: str, *, maximum_rows: int):
            if (
                sql
                == runtime.PreopenedSessionBootstrapProvisioner._IDENTITY_SQL
                and self.identity_count == 1
            ):
                self.backend_pid = "43"
            return super().query(sql, maximum_rows=maximum_rows)

    session = ReconnectedSession()
    provisioner = runtime.PreopenedSessionBootstrapProvisioner(
        session,
        tls_peer_certificate_sha256="a" * 64,
    )
    provision = provisioner.provision(request)

    with pytest.raises(
        RuntimeError,
        match="admin session changed before reconciliation",
    ):
        provisioner.reconcile(request, provision)

    assert session.identity_count == 2
    assert session.closed is True


def test_recovery_session_can_replay_retired_truth_with_new_executor() -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    first_session = _PreopenedSession(backend_pid="42")
    first = runtime.PreopenedSessionBootstrapProvisioner(
        first_session,
        tls_peer_certificate_sha256="a" * 64,
    )
    provision = first.provision(request)
    first.abort()

    recovery_session = _PreopenedSession(
        cleanup_outcome="retired",
        cleanup_inserted=False,
        backend_pid="84",
    )
    recovery = runtime.PreopenedSessionBootstrapProvisioner(
        recovery_session,
        tls_peer_certificate_sha256="a" * 64,
    )
    receipt = recovery.reconcile(request, provision)
    validated = runtime._validate_canary_bootstrap_reconciliation_receipt(
        receipt,
        request=request,
        provisioning_receipt=provision,
        approval=approval,
        expected_session_continuity="recovery_session",
    )

    assert validated["outcome"] == "retired"
    assert validated["retired_inserted"] is False
    assert validated["session_continuity"] == "recovery_session"
    assert validated["executor_session_identity"]["backend_pid"] == 84
    assert (
        validated["executor_session_identity_sha256"]
        != provision["executor_session_identity_sha256"]
    )


def test_recovery_reconciliation_rejects_material_future_timestamp() -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    now_unix = int(time.time())
    recovery = runtime.PreopenedSessionBootstrapProvisioner(
        _PreopenedSession(cleanup_outcome="not_authorized"),
        tls_peer_certificate_sha256="a" * 64,
        now=lambda: (
            now_unix
            + runtime.BOOTSTRAP_RECONCILIATION_FUTURE_SKEW_SECONDS
            + 1
        ),
    )
    receipt = recovery.reconcile(request, None)

    with pytest.raises(RuntimeError, match="reconciliation receipt is not exact"):
        runtime._validate_canary_bootstrap_reconciliation_receipt(
            receipt,
            request=request,
            provisioning_receipt=None,
            approval=approval,
            expected_session_continuity="recovery_session",
            now_unix=now_unix,
        )


def test_expired_poststop_source_requires_exact_reconciliation_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    recovery_session = _PreopenedSession(cleanup_outcome="not_authorized")
    provisioner = runtime.PreopenedSessionBootstrapProvisioner(
        recovery_session,
        tls_peer_certificate_sha256="a" * 64,
    )
    reconciliation = provisioner.reconcile(request, None)
    evidence = runtime.validated_bootstrap_reconciliation_evidence(
        plan=plan,
        approval=approval,
        request=request,
        provisioning_receipt=None,
        reconciliation_receipt=reconciliation,
        expected_session_continuity="recovery_session",
    )
    freshness_checks: list[bool] = []

    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda *_args, **_kwargs: b"expired-writer-source",
    )

    def validate_writer(
        raw,
        _identities,
        *,
        require_fresh_canary_scope=True,
        **_kwargs,
    ):
        assert raw == b"expired-writer-source"
        freshness_checks.append(require_fresh_canary_scope)
        if require_fresh_canary_scope:
            raise RuntimeError("writer canary scope is fixture-bound/fresh invalid")
        return _bootstrap_intent_config()

    monkeypatch.setattr(runtime, "_validate_writer_config", validate_writer)
    monkeypatch.setattr(
        runtime,
        "validate_canary_preclaim_reconciliation_receipt",
        lambda **_kwargs: {"result": {"outcome": "retired"}},
    )

    with pytest.raises(RuntimeError, match="fixture-bound/fresh"):
        runtime.FullCanaryLifecycle(plan)._validate_poststop_preclaim_receipt(
            prior_generation=None,
            allow_not_preapproved=True,
        )
    receipt = runtime.FullCanaryLifecycle(
        plan,
        bootstrap_reconciliation_evidence=evidence,
    )._validate_poststop_preclaim_receipt(
        prior_generation=None,
        allow_not_preapproved=True,
    )

    assert receipt["result"]["outcome"] == "retired"
    assert freshness_checks == [True, False]


def test_existing_invalid_bootstrap_credential_is_tamper_not_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = tmp_path / "bootstrap-password"
    credential.write_bytes(b"present-but-invalid-mode")
    credential.chmod(0o600)
    plan = _host_receipt_plan(b"{}")
    plan = replace(
        plan,
        identities=replace(
            plan.identities,
            writer_uid=os.getuid(),
            writer_gid=os.getgid(),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "DEFAULT_CANARY_BOOTSTRAP_CREDENTIAL",
        credential,
    )

    with pytest.raises(RuntimeError, match="metadata is invalid"):
        runtime._bootstrap_credential_is_present_and_exact(plan)

    credential.unlink()
    assert runtime._bootstrap_credential_is_present_and_exact(plan) is False


def _bootstrap_envelope_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FullCanaryPlan, runtime.BootstrapReconciliationEvidence]:
    plan = _host_receipt_plan(b"{}")
    writer_raw = b'{"sealed":"writer"}'
    writer_path = tmp_path / "writer.json"
    writer_path.write_bytes(writer_raw)
    plan = replace(
        plan,
        artifacts={
            **dict(plan.artifacts),
            "writer_config": replace(
                plan.artifacts["writer_config"],
                source_path=writer_path,
                sha256=hashlib.sha256(writer_raw).hexdigest(),
            ),
            "e2e_fixture": ExactArtifact(
                source_path=tmp_path / "fixture.json",
                target_path=tmp_path / "fixture.json",
                sha256="4" * 64,
                mode=0o440,
                uid=0,
                gid=plan.identities.gateway_gid,
            ),
        },
    )
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    session = _PreopenedSession(cleanup_outcome="retired")
    provisioner = runtime.PreopenedSessionBootstrapProvisioner(
        session,
        tls_peer_certificate_sha256="a" * 64,
    )
    provision = provisioner.provision(request)
    reconciliation = provisioner.reconcile(request, provision)
    evidence = runtime.validated_bootstrap_reconciliation_evidence(
        plan=plan,
        approval=approval,
        request=request,
        provisioning_receipt=provision,
        reconciliation_receipt=reconciliation,
        expected_session_continuity="same_provision_session",
    )
    monkeypatch.setattr(runtime, "DEFAULT_EVIDENCE_ROOT", tmp_path / "evidence")

    def release_file(_plan, relative, **_kwargs):
        if relative == runtime.DEFAULT_CANARY_BOOTSTRAP_SQL_RELATIVE:
            return request.sql_path, request.sql_bytes, request.sql_sha256
        assert relative == runtime.DEFAULT_CANARY_BOOTSTRAP_RETIRE_SQL_RELATIVE
        return (
            request.retire_sql_path,
            request.retire_sql_bytes,
            request.retire_sql_sha256,
        )

    monkeypatch.setattr(runtime, "_validated_release_file", release_file)
    monkeypatch.setattr(
        runtime,
        "_bind_bootstrap_request_to_writer_config",
        lambda **_kwargs: {},
    )
    return plan, evidence


def _fake_root_envelope_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, int]]:
    writes: list[tuple[Path, int]] = []
    real_recover = runtime._recover_bootstrap_evidence_staging
    real_fallback = runtime._fallback_bootstrap_evidence_descriptor

    def write(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(mode)
        writes.append((path, mode))

    def read(path: Path, **_kwargs):
        return path.read_bytes(), path.lstat()

    original_lstat = Path.lstat

    def root_lstat(path: Path):
        item = original_lstat(path)
        if not any("bootstrap-reconciliation" in part for part in path.parts):
            return item
        return SimpleNamespace(
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_nlink=item.st_nlink,
        )

    monkeypatch.setattr(runtime, "_write_exclusive_bytes", write)
    monkeypatch.setattr(runtime, "_read_stable_file", read)
    monkeypatch.setattr(runtime, "_read_root_file_via_parent", read)
    monkeypatch.setattr(Path, "lstat", root_lstat)
    monkeypatch.setattr(
        runtime,
        "_validate_root_directory_chain",
        lambda _path: None,
    )
    monkeypatch.setattr(
        runtime,
        "_ensure_root_directory",
        lambda path, **_kwargs: path.mkdir(parents=True, exist_ok=True),
    )
    anchor = runtime.DEFAULT_EVIDENCE_ROOT.parent
    anchor.chmod(0o700)
    owner = anchor.lstat()
    monkeypatch.setattr(
        runtime,
        "_recover_bootstrap_evidence_staging",
        lambda plan: real_recover(
            plan,
            expected_uid=owner.st_uid,
            expected_gid=owner.st_gid,
            trusted_anchor=anchor,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_fallback_bootstrap_evidence_descriptor",
        lambda plan: real_fallback(
            plan,
            expected_uid=owner.st_uid,
            expected_gid=owner.st_gid,
            trusted_anchor=anchor,
        ),
    )
    return writes


def test_controlled_directory_chain_rejects_real_symlink_ancestor(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    anchor_item = tmp_path.lstat()
    uid = anchor_item.st_uid
    gid = anchor_item.st_gid
    safe = tmp_path / "safe" / "leaf"
    descriptor = runtime._open_root_directory_chain(
        safe,
        create=True,
        expected_uid=uid,
        expected_gid=gid,
        trusted_anchor=tmp_path,
    )
    os.close(descriptor)
    redirect = tmp_path / "redirect"
    redirect.symlink_to(tmp_path / "safe", target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink|invalid component"):
        runtime._open_root_directory_chain(
            redirect / "leaf",
            create=False,
            expected_uid=uid,
            expected_gid=gid,
            trusted_anchor=tmp_path,
        )


def test_held_parent_descriptor_rejects_real_ancestor_replacement(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    owner = tmp_path.lstat()
    canonical_parent = tmp_path / "tree" / "leaf"
    descriptor = runtime._open_root_directory_chain(
        canonical_parent,
        create=True,
        expected_uid=owner.st_uid,
        expected_gid=owner.st_gid,
        trusted_anchor=tmp_path,
    )
    retired_tree = tmp_path / "retired-tree"
    os.rename(tmp_path / "tree", retired_tree)
    canonical_parent.mkdir(parents=True, mode=0o700)
    entry = os.open(
        "evidence.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o400,
        dir_fd=descriptor,
    )
    os.write(entry, b"held-parent")
    os.fsync(entry)
    os.close(entry)
    os.fsync(descriptor)

    with pytest.raises(RuntimeError, match="reachability changed"):
        runtime._revalidate_root_directory_reachability(
            canonical_parent,
            descriptor,
            expected_uid=owner.st_uid,
            expected_gid=owner.st_gid,
            trusted_anchor=tmp_path,
        )
    os.close(descriptor)
    assert (retired_tree / "leaf" / "evidence.json").read_bytes() == (
        b"held-parent"
    )
    assert not (canonical_parent / "evidence.json").exists()


def test_wrong_host_proof_reaches_zero_systemctl_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[runtime.Command] = []
    monkeypatch.setattr(
        runtime,
        "_observe_dedicated_canary_host",
        lambda: (_ for _ in ()).throw(RuntimeError("wrong canary host")),
    )

    with pytest.raises(RuntimeError, match="wrong canary host"):
        runtime._stop_all(
            runner=lambda command: commands.append(command)
            or runtime.subprocess.CompletedProcess(
                command.argv,
                0,
                stdout=b"",
                stderr=b"",
            )
        )
    assert commands == []


def test_same_inode_hardlink_crash_is_finalized_on_real_filesystem(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging" / "evidence.tmp"
    final = tmp_path / "final" / "evidence.json"
    staging.parent.mkdir(mode=0o700)
    final.parent.mkdir(mode=0o700)
    staging.write_bytes(b'{"durable":true}')
    staging.chmod(0o400)
    before = staging.lstat()
    os.link(staging, final)
    assert staging.lstat().st_nlink == final.lstat().st_nlink == 2

    runtime._finalize_same_inode_hardlink_publication(
        staging_path=staging,
        final_path=final,
        staging_item=before,
        expected_uid=before.st_uid,
        expected_gid=before.st_gid,
    )

    assert not staging.exists()
    assert final.read_bytes() == b'{"durable":true}'
    assert final.lstat().st_nlink == 1


@pytest.mark.parametrize(
    "crash_state",
    ("orphan", "linked_before_cleanup", "torn"),
)
def test_never_authorized_staging_recovers_real_filesystem_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_state: str,
) -> None:
    plan = SimpleNamespace(revision="1" * 40, sha256="2" * 64)
    monkeypatch.setattr(runtime, "DEFAULT_EVIDENCE_ROOT", tmp_path / "evidence")
    staging = runtime._bootstrap_never_authorized_staging_path(plan)
    final = runtime._bootstrap_never_authorized_path(plan)
    staging.parent.mkdir(parents=True, mode=0o700)
    payload = b'{"state":"never_authorized"}'
    staging.write_bytes(b'{"state":' if crash_state == "torn" else payload)
    staging.chmod(0o400)
    owner = staging.lstat()

    def parse(_plan, *, path, raw):
        value = json.loads(raw)
        return runtime.BootstrapNeverAuthorizedEvidence(
            value=value,
            path=path,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        )

    monkeypatch.setattr(
        runtime,
        "_bootstrap_never_authorized_from_bytes",
        parse,
    )
    if crash_state == "linked_before_cleanup":
        final.parent.mkdir(parents=True, mode=0o700)
        os.link(staging, final)

    if crash_state == "torn":
        with pytest.raises(runtime.BootstrapEvidenceUnavailable):
            runtime._recover_bootstrap_never_authorized_staging(
                plan,
                    expected_uid=owner.st_uid,
                    expected_gid=owner.st_gid,
                trusted_anchor=tmp_path,
            )
        assert not staging.exists()
        assert not final.exists()
        return

    recovered = runtime._recover_bootstrap_never_authorized_staging(
        plan,
        expected_uid=owner.st_uid,
        expected_gid=owner.st_gid,
        trusted_anchor=tmp_path,
    )
    assert recovered is not None
    assert recovered.value == {"state": "never_authorized"}
    assert not staging.exists()
    assert final.read_bytes() == payload
    assert final.lstat().st_nlink == 1


def test_bootstrap_envelope_survives_crash_before_any_outcome_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, evidence = _bootstrap_envelope_fixture(tmp_path, monkeypatch)
    writes = _fake_root_envelope_filesystem(monkeypatch)

    persisted = runtime.persist_bootstrap_evidence_envelope(plan, evidence)
    descriptor = persisted.descriptor
    assert descriptor is not None
    assert writes == [
        (runtime._bootstrap_evidence_staging_path(plan, descriptor), 0o400)
    ]
    assert descriptor.path == (
        runtime.DEFAULT_EVIDENCE_ROOT
        / "plans"
        / plan.revision
        / plan.sha256
        / "bootstrap-reconciliation"
        / descriptor.attempt_id
        / f"{descriptor.envelope_sha256}.json"
    )
    envelope = json.loads(descriptor.path.read_text())
    assert envelope["predecessor_envelope_sha256"] is None
    assert envelope["owner_approval"] == evidence.approval.value
    assert len(envelope["bootstrap_request"]["guc_bindings"]) == 11

    # There is deliberately no start/failure/stopped receipt: fallback is the
    # crash/restart path and may accept only this sole plan-addressed envelope.
    restarted = runtime.load_bootstrap_evidence_envelope(plan)
    assert restarted.descriptor == descriptor
    assert restarted.reconciliation_receipt == evidence.reconciliation_receipt


def test_bootstrap_envelope_rejects_tamper_and_plan_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, evidence = _bootstrap_envelope_fixture(tmp_path, monkeypatch)
    _fake_root_envelope_filesystem(monkeypatch)
    persisted = runtime.persist_bootstrap_evidence_envelope(plan, evidence)
    descriptor = persisted.descriptor
    assert descriptor is not None

    original = descriptor.path.read_bytes()
    descriptor.path.chmod(0o600)
    descriptor.path.write_bytes(original + b"\n")
    descriptor.path.chmod(0o400)
    with pytest.raises(RuntimeError, match="file digest drifted"):
        runtime.load_bootstrap_evidence_envelope(plan, descriptor)
    descriptor.path.chmod(0o600)
    descriptor.path.write_bytes(original)
    descriptor.path.chmod(0o400)

    conflicting_attempt = "f" * 64
    conflicting_envelope = "e" * 64
    conflicting_path = (
        runtime._bootstrap_evidence_root(plan)
        / conflicting_attempt
        / f"{conflicting_envelope}.json"
    )
    conflicting_path.parent.mkdir(parents=True)
    conflicting_path.write_bytes(original)
    conflicting_path.chmod(0o400)
    with pytest.raises(
        runtime.BootstrapEvidenceAmbiguous,
        match="ambiguous",
    ):
        runtime.load_bootstrap_evidence_envelope(plan)


@pytest.mark.parametrize(
    "crash_point",
    ["orphan_fsynced_temp", "linked_before_temp_cleanup", "torn_temp"],
)
def test_bootstrap_envelope_recovers_power_loss_publication_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    plan, evidence = _bootstrap_envelope_fixture(tmp_path, monkeypatch)
    _fake_root_envelope_filesystem(monkeypatch)
    envelope = runtime._bootstrap_evidence_envelope(plan, evidence)
    payload = runtime._canonical_bytes(envelope)
    descriptor = runtime._descriptor_for_bootstrap_envelope(
        plan,
        envelope,
        file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    staging = runtime._bootstrap_evidence_staging_path(plan, descriptor)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_bytes(
        b'{"schema":' if crash_point == "torn_temp" else payload
    )
    staging.chmod(0o400)
    if crash_point == "linked_before_temp_cleanup":
        descriptor.path.parent.mkdir(parents=True, exist_ok=True)
        os.link(staging, descriptor.path)

    if crash_point == "torn_temp":
        with pytest.raises(runtime.BootstrapEvidenceUnavailable):
            runtime.load_bootstrap_evidence_envelope(plan)
        assert not staging.exists()
        assert not descriptor.path.exists()
        return

    loaded = runtime.load_bootstrap_evidence_envelope(plan)
    assert loaded.descriptor == descriptor
    assert descriptor.path.read_bytes() == payload
    assert not staging.exists()


def test_old_receipt_without_descriptor_or_envelope_requires_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _evidence = _bootstrap_envelope_fixture(tmp_path, monkeypatch)
    _fake_root_envelope_filesystem(monkeypatch)
    path = (
        runtime.DEFAULT_EVIDENCE_ROOT
        / "plans"
        / plan.revision
        / plan.sha256
        / "stopped"
        / "old.json"
    )
    unsigned = {
        "schema": runtime.FULL_CANARY_RECEIPT_SCHEMA,
        "stage": "stopped",
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.sha256,
        "receipt_path": str(path),
        "bootstrap_reconciliation": {},
    }
    path.parent.mkdir(parents=True)
    path.write_bytes(
        runtime._canonical_bytes(
            {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}
        )
    )
    path.chmod(0o400)

    with pytest.raises(RuntimeError, match="fields are incomplete"):
        runtime.load_bootstrap_evidence_from_receipt(path, plan=plan)


def test_failure_receipt_null_provisioning_resolves_durable_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, seeded = _bootstrap_envelope_fixture(tmp_path, monkeypatch)
    _fake_root_envelope_filesystem(monkeypatch)
    recovery = runtime.PreopenedSessionBootstrapProvisioner(
        _PreopenedSession(cleanup_outcome="not_authorized"),
        tls_peer_certificate_sha256="a" * 64,
    )
    reconciliation = recovery.reconcile(seeded.request, None)
    evidence = runtime.validated_bootstrap_reconciliation_evidence(
        plan=plan,
        approval=seeded.approval,
        request=seeded.request,
        provisioning_receipt=None,
        reconciliation_receipt=reconciliation,
        expected_session_continuity="recovery_session",
    )
    persisted = runtime.persist_bootstrap_evidence_envelope(plan, evidence)
    assert persisted.descriptor is not None
    path = (
        runtime.DEFAULT_EVIDENCE_ROOT
        / "plans"
        / plan.revision
        / plan.sha256
        / "failure"
        / "failure.json"
    )

    def write_receipt(provisioning):
        unsigned = {
            "schema": runtime.FULL_CANARY_RECEIPT_SCHEMA,
            "stage": "failure",
            "operation": "start",
            "revision": plan.revision,
            "full_canary_plan_sha256": plan.sha256,
            "receipt_path": str(path),
            "bootstrap_evidence_present": True,
            "bootstrap_never_authorized_evidence": None,
            "owner_approval_receipt": copy.deepcopy(
                dict(persisted.approval.value)
            ),
            "owner_approval_receipt_sha256": persisted.approval.sha256,
            "bootstrap_provisioning_receipt": provisioning,
            "bootstrap_reconciliation": copy.deepcopy(
                dict(persisted.reconciliation_receipt)
            ),
            "bootstrap_evidence_descriptor": (
                persisted.descriptor.to_mapping()
            ),
            "bootstrap_reconciliation_complete": True,
            "bootstrap_authority_may_require_owner_cleanup": False,
            "bootstrap_durable_evidence_recovery_required": False,
            "failed_at_unix": int(time.time()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.chmod(0o600)
        path.write_bytes(
            runtime._canonical_bytes(
                {
                    **unsigned,
                    "receipt_sha256": runtime._sha256_json(unsigned),
                }
            )
        )
        path.chmod(0o400)

    write_receipt(None)
    loaded = runtime.load_bootstrap_evidence_from_receipt(path, plan=plan)
    assert loaded.provisioning_receipt is None
    assert loaded.descriptor == persisted.descriptor

    write_receipt({})
    with pytest.raises(RuntimeError, match="copied bootstrap_provisioning"):
        runtime.load_bootstrap_evidence_from_receipt(path, plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_descriptor",
        "approval_sha",
        "approval_full",
        "provisioning",
        "reconciliation",
        "complete_flag",
        "cleanup_flag",
        "recovery_flag",
        "present_flag",
        "timestamp",
        "never_authorized",
    ],
)
def test_evidence_receipt_rejects_every_copied_truth_or_flag_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    plan, evidence = _bootstrap_envelope_fixture(tmp_path, monkeypatch)
    _fake_root_envelope_filesystem(monkeypatch)
    persisted = runtime.persist_bootstrap_evidence_envelope(plan, evidence)
    assert persisted.descriptor is not None
    path = (
        runtime.DEFAULT_EVIDENCE_ROOT
        / "plans"
        / plan.revision
        / plan.sha256
        / "failure"
        / f"{mutation}.json"
    )
    unsigned = {
        "schema": runtime.FULL_CANARY_RECEIPT_SCHEMA,
        "stage": "failure",
        "operation": "start",
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.sha256,
        "receipt_path": str(path),
        "bootstrap_evidence_present": True,
        "bootstrap_never_authorized_evidence": None,
        "bootstrap_evidence_descriptor": persisted.descriptor.to_mapping(),
        "owner_approval_receipt": copy.deepcopy(
            dict(persisted.approval.value)
        ),
        "owner_approval_receipt_sha256": persisted.approval.sha256,
        "bootstrap_provisioning_receipt": copy.deepcopy(
            dict(persisted.provisioning_receipt or {})
        ),
        "bootstrap_reconciliation": copy.deepcopy(
            dict(persisted.reconciliation_receipt)
        ),
        "bootstrap_reconciliation_complete": True,
        "bootstrap_authority_may_require_owner_cleanup": False,
        "bootstrap_durable_evidence_recovery_required": False,
        "failed_at_unix": int(time.time()),
    }
    if mutation == "missing_descriptor":
        unsigned["bootstrap_evidence_descriptor"] = None
    elif mutation == "approval_sha":
        unsigned["owner_approval_receipt_sha256"] = "f" * 64
    elif mutation == "approval_full":
        unsigned["owner_approval_receipt"]["owner_subject_sha256"] = "f" * 64
    elif mutation == "provisioning":
        unsigned["bootstrap_provisioning_receipt"] = None
    elif mutation == "reconciliation":
        unsigned["bootstrap_reconciliation"]["reason"] = "tampered"
    elif mutation == "complete_flag":
        unsigned["bootstrap_reconciliation_complete"] = False
    elif mutation == "cleanup_flag":
        unsigned["bootstrap_authority_may_require_owner_cleanup"] = True
    elif mutation == "recovery_flag":
        unsigned["bootstrap_durable_evidence_recovery_required"] = True
    elif mutation == "present_flag":
        unsigned["bootstrap_evidence_present"] = False
    elif mutation == "timestamp":
        unsigned["failed_at_unix"] = 0
    else:
        unsigned["bootstrap_never_authorized_evidence"] = {
            "path": "/forged",
            "file_sha256": "f" * 64,
            "receipt_sha256": "e" * 64,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        runtime._canonical_bytes(
            {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}
        )
    )
    path.chmod(0o400)

    with pytest.raises((RuntimeError, runtime.BootstrapEvidenceUnavailable)):
        runtime.load_bootstrap_evidence_from_receipt(path, plan=plan)


def test_start_receipt_resolves_bootstrap_envelope_before_other_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")
    monkeypatch.setattr(runtime, "DEFAULT_EVIDENCE_ROOT", tmp_path / "evidence")
    path = (
        runtime.DEFAULT_EVIDENCE_ROOT
        / "plans"
        / plan.revision
        / plan.sha256
        / "started"
        / "start.json"
    )
    unsigned = {
        "schema": runtime.FULL_CANARY_RECEIPT_SCHEMA,
        "stage": "started",
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.sha256,
        "receipt_path": str(path),
        "units_enabled": False,
        "runtime_max_seconds": 900,
        "start_order": [EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        runtime._canonical_bytes(
            {**unsigned, "receipt_sha256": runtime._sha256_json(unsigned)}
        )
    )
    path.chmod(0o400)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_read_stable_file",
        lambda observed, **_kwargs: (observed.read_bytes(), observed.lstat()),
    )
    monkeypatch.setattr(
        runtime,
        "load_bootstrap_evidence_from_receipt",
        lambda observed, **_kwargs: calls.append(str(observed)) or object(),
    )

    with pytest.raises(RuntimeError, match="identity receipts"):
        runtime.load_start_receipt(path, plan=plan)
    assert calls == [str(path)]


def test_runtime_cli_validate_passes_loaded_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = object()
    evidence = object()
    observed: list[object] = []
    monkeypatch.setattr(runtime, "load_full_canary_plan", lambda: plan)
    monkeypatch.setattr(
        runtime,
        "load_bootstrap_terminal_evidence",
        lambda observed_plan, *, phase: (
            evidence
            if observed_plan is plan and phase == "stopped"
            else pytest.fail("wrong terminal evidence load")
        ),
    )

    def preflight(observed_plan, *, phase, bootstrap_reconciliation_evidence):
        assert observed_plan is plan
        assert phase == "stopped"
        assert bootstrap_reconciliation_evidence is evidence
        return {"ok": True}

    monkeypatch.setattr(runtime, "collect_full_canary_preflight", preflight)
    monkeypatch.setattr(runtime, "_cli_result", lambda value: observed.append(value))

    assert runtime.main(["validate"]) == 0
    assert observed == [{"ok": True}]


def test_runtime_cli_stop_is_mechanical_before_evidence_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = object()
    evidence = object()
    events: list[str] = []
    stopped = (GATEWAY_UNIT_NAME, WRITER_UNIT_NAME, EDGE_UNIT_NAME)

    class Lifecycle:
        def attest_stopped_after_mechanical_stop(self, **kwargs):
            assert kwargs["stopped"] == stopped
            events.append("attest")
            return {"ok": True}

    monkeypatch.setattr(
        runtime,
        "load_full_canary_plan",
        lambda: events.append("load_plan") or plan,
    )
    monkeypatch.setattr(
        runtime,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: events.append("observe") or (1,),
    )
    monkeypatch.setattr(
        runtime,
        "mechanically_stop_full_canary_services",
        lambda: events.append("mechanical_stop") or stopped,
    )
    monkeypatch.setattr(
        runtime,
        "load_bootstrap_terminal_evidence",
        lambda *_args, **_kwargs: events.append("load_evidence") or evidence,
    )
    monkeypatch.setattr(
        runtime,
        "_lifecycle_for_bootstrap_terminal_evidence",
        lambda observed_plan, observed_evidence: (
            Lifecycle()
            if observed_plan is plan and observed_evidence is evidence
            else pytest.fail("terminal evidence was not passed to lifecycle")
        ),
    )
    monkeypatch.setattr(runtime, "_cli_result", lambda _value: None)

    assert runtime.main(["stop"]) == 0
    assert events == [
        "mechanical_stop",
        "observe",
        "load_plan",
        "load_evidence",
        "attest",
    ]


def test_runtime_cli_verify_tamper_still_mechanically_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_observe_dedicated_canary_host",
        lambda: events.append("host_proof") or {},
    )
    monkeypatch.setattr(
        runtime,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: events.append("observe_preclaim") or None,
    )
    monkeypatch.setattr(
        runtime,
        "load_full_canary_plan",
        lambda: events.append("load_plan") or object(),
    )

    def tampered(*_args, **_kwargs):
        events.append("load_evidence")
        raise RuntimeError("bootstrap evidence file digest drifted")

    monkeypatch.setattr(runtime, "load_bootstrap_terminal_evidence", tampered)
    monkeypatch.setattr(
        runtime,
        "mechanically_stop_full_canary_services",
        lambda: events.append("mechanical_stop")
        or (GATEWAY_UNIT_NAME, WRITER_UNIT_NAME, EDGE_UNIT_NAME),
    )
    monkeypatch.setattr(runtime, "_cli_result", lambda _value: None)

    assert (
        runtime.main(
            [
                "verify-and-stop",
                "--start-receipt",
                str(tmp_path / "start.json"),
                "--evidence-sha256",
                "a" * 64,
            ]
        )
        == 2
    )
    assert events == [
        "host_proof",
        "load_plan",
        "load_evidence",
        "mechanical_stop",
        "observe_preclaim",
    ]


def test_runtime_cli_verify_second_start_load_is_already_stopped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    plan = object()
    evidence = object.__new__(runtime.BootstrapReconciliationEvidence)
    start_path = tmp_path / "start.json"
    load_count = 0

    def load_start(path, *, plan):
        nonlocal load_count
        load_count += 1
        events.append(f"load_start_{load_count}")
        if load_count == 2:
            raise RuntimeError("second start receipt load drifted")
        return object()

    class Lifecycle:
        def verify_and_stop(self, **_kwargs):
            runtime.load_start_receipt(start_path, plan=plan)
            pytest.fail("second start receipt drift was accepted")

    monkeypatch.setattr(
        runtime,
        "_observe_dedicated_canary_host",
        lambda: events.append("host_proof") or {},
    )
    monkeypatch.setattr(
        runtime,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: events.append("observe_preclaim") or None,
    )
    monkeypatch.setattr(
        runtime,
        "mechanically_stop_full_canary_services",
        lambda: events.append("mechanical_stop")
        or (GATEWAY_UNIT_NAME, WRITER_UNIT_NAME, EDGE_UNIT_NAME),
    )
    monkeypatch.setattr(
        runtime,
        "load_full_canary_plan",
        lambda: events.append("load_plan") or plan,
    )
    monkeypatch.setattr(
        runtime,
        "load_bootstrap_terminal_evidence",
        lambda *_args, **_kwargs: events.append("load_evidence") or evidence,
    )
    monkeypatch.setattr(runtime, "load_start_receipt", load_start)
    monkeypatch.setattr(
        runtime,
        "expected_live_evidence_path",
        lambda _plan: tmp_path / "evidence.json",
    )
    monkeypatch.setattr(
        runtime,
        "_lifecycle_for_bootstrap_terminal_evidence",
        lambda *_args: events.append("construct_lifecycle") or Lifecycle(),
    )
    monkeypatch.setattr(runtime, "_cli_result", lambda _value: None)

    assert (
        runtime.main(
            [
                "verify-and-stop",
                "--start-receipt",
                str(start_path),
                "--evidence-sha256",
                "a" * 64,
            ]
        )
        == 2
    )
    assert events == [
        "host_proof",
        "load_plan",
        "load_evidence",
        "load_start_1",
        "construct_lifecycle",
        "load_start_2",
        "mechanical_stop",
        "observe_preclaim",
    ]


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
    monkeypatch.setattr(
        runtime,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: events.append("observe_preclaim") or (1,),
    )
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_validate_poststop_preclaim_receipt",
        lambda self, **_kwargs: events.append("validate_preclaim")
        or {"result": {"outcome": "claimed"}},
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
        runtime,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: events.append("observe_preclaim") or None,
    )
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_validate_poststop_preclaim_receipt",
        lambda self, **_kwargs: {"result": {"outcome": "retired"}},
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
    assert events.index("mechanical_stop") < events.index("observe_preclaim")


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
        runtime,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: events.append("observe_preclaim") or None,
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
        "observe_preclaim",
        "plan_host_validation",
    ]


def test_close_failure_remains_retryable_for_final_abort() -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)

    class CloseFailsOnceSession(_PreopenedSession):
        def close(self) -> None:
            self.close_count += 1
            if self.close_count == 1:
                raise RuntimeError("transient close failure")
            self.closed = True

    session = CloseFailsOnceSession()
    provisioner = runtime.PreopenedSessionBootstrapProvisioner(
        session,
        tls_peer_certificate_sha256="a" * 64,
    )
    provision = provisioner.provision(request)

    with pytest.raises(RuntimeError, match="transient close failure"):
        provisioner.reconcile(request, provision)
    provisioner.abort()

    assert session.close_count == 2
    assert session.closed is True


def test_noncanonical_provision_result_cannot_block_sealed_reconciliation() -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    session = _PreopenedSession(cleanup_outcome="retired")
    provisioner = runtime.PreopenedSessionBootstrapProvisioner(
        session,
        tls_peer_certificate_sha256="a" * 64,
    )
    provisioner.provision(request)
    noncanonical = {"unexpected_private_value": object()}

    reconciliation = provisioner.reconcile(request, noncanonical)
    validated = runtime._validate_canary_bootstrap_reconciliation_receipt(
        reconciliation,
        request=request,
        provisioning_receipt=noncanonical,
        approval=approval,
        expected_session_continuity="same_provision_session",
    )

    assert validated["reconciled"] is True
    assert validated["provisioning_receipt_present"] is True
    assert validated["provisioning_application_receipt_sha256"] == "0" * 64
    assert "unexpected_private_value" not in json.dumps(validated)


@pytest.mark.parametrize(
    "failure,cleanup_fails",
    [
        ("provision_raises", False),
        ("invalid_provision_receipt", False),
        ("writer_start", False),
        ("consumption", False),
        ("consumption", True),
        ("persistence", False),
    ],
)
def test_every_post_provision_failure_reconciles_before_exit(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    cleanup_fails: bool,
) -> None:
    plan = replace(
        _host_receipt_plan(b"{}"),
        writer_activation_receipt={"receipt_sha256": "7" * 64},
    )
    approval = _owner_approval(plan)
    request = _direct_bootstrap_request(plan, approval)
    calls: list[str] = []
    failure_receipts: list[dict] = []
    descriptor = runtime.BootstrapEvidenceDescriptor(
        schema=runtime.BOOTSTRAP_EVIDENCE_DESCRIPTOR_SCHEMA,
        path=Path("/var/lib/muncho/evidence.json"),
        file_sha256="8" * 64,
        envelope_sha256="9" * 64,
        attempt_id="a" * 64,
    )

    class Provisioner:
        def provision(self, observed):
            assert observed is request
            calls.append("provision")
            if failure == "provision_raises":
                raise RuntimeError("committed before transport failure")
            if failure == "invalid_provision_receipt":
                return {}
            return {"provision": "valid"}

        def reconcile(self, observed, provision):
            assert observed is request
            calls.append("reconcile")
            if failure == "provision_raises":
                assert provision is None
            if failure == "invalid_provision_receipt":
                assert provision == {}
            if cleanup_fails:
                raise RuntimeError("cleanup connection failed")
            return {"reconciled": True}

        def abort(self):
            calls.append("abort")

    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_require_dedicated_host",
        lambda self: calls.append("host") or {"exact": True},
    )
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda *_a, **_k: b"{}",
    )
    monkeypatch.setattr(runtime, "_validate_writer_config", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "_lifecycle_lock", lambda: nullcontext())
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_preflight",
        lambda self, *, phase: {"report_sha256": "6" * 64},
    )
    monkeypatch.setattr(runtime, "_install_plan_artifacts", lambda _plan: {})
    monkeypatch.setattr(
        runtime,
        "_validate_inert_gateway_paths",
        lambda **_kwargs: True,
    )

    def run_checked(command, **_kwargs):
        unit = command.argv[-1]
        calls.append("run:" + unit)
        if failure == "writer_start" and unit == runtime.WRITER_UNIT_NAME:
            raise RuntimeError("writer start failed")
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    monkeypatch.setattr(runtime, "_run_checked", run_checked)
    monkeypatch.setattr(
        runtime,
        "collect_service_state",
        lambda unit, **_kwargs: {"MainPID": 41, "unit": unit},
    )
    monkeypatch.setattr(runtime, "_validate_edge_collector_gate", lambda *_a: {})
    monkeypatch.setattr(
        runtime,
        "_await_collector_readiness",
        lambda *_a, **_k: SimpleNamespace(
            receipt={},
            file_sha256="5" * 64,
            service_identity_sha256="4" * 64,
        ),
    )
    monkeypatch.setattr(runtime, "materialize_observer_config", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runtime,
        "_build_canary_bootstrap_provisioning_request",
        lambda *_a, **_k: request,
    )

    def validate_provision(value, **_kwargs):
        if failure == "invalid_provision_receipt":
            raise RuntimeError("invalid provision receipt")
        return value

    monkeypatch.setattr(
        runtime,
        "_validate_canary_bootstrap_provisioning_receipt",
        validate_provision,
    )
    monkeypatch.setattr(runtime, "_readiness_receipt", lambda *_a, **_k: {})

    def validate_consumption(*_args):
        if failure == "consumption":
            raise RuntimeError("consumption receipt invalid")
        return {}

    monkeypatch.setattr(
        runtime,
        "_validate_writer_bootstrap_consumption",
        validate_consumption,
    )
    monkeypatch.setattr(
        runtime,
        "_retire_writer_bootstrap_config",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        runtime,
        "_validate_canary_bootstrap_reconciliation_receipt",
        lambda value, **_kwargs: value,
    )
    def persist(*_args, evidence, **_kwargs):
        calls.append("persist")
        if failure == "persistence":
            raise RuntimeError("evidence persistence failed")
        return replace(evidence, descriptor=descriptor)

    monkeypatch.setattr(runtime, "persist_bootstrap_evidence_envelope", persist)
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_bootstrap_evidence_descriptor_value",
        lambda self: (
            None
            if self._bootstrap_reconciliation_evidence is None
            or self._bootstrap_reconciliation_evidence.descriptor is None
            else descriptor.to_mapping()
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_stop_all",
        lambda **_kwargs: calls.append("stop") or (),
    )
    monkeypatch.setattr(
        runtime,
        "observe_canary_preclaim_reconciliation_generation",
        lambda: None,
    )
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_validate_preclaim_receipt",
        lambda self, **_kwargs: {
            "result": {"outcome": "not_preapproved"}
        },
    )
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_validate_poststop_preclaim_receipt",
        lambda self, **_kwargs: {"result": {"outcome": "retired"}},
    )

    def write_receipt(_plan, *, stage, value):
        if stage == "failure":
            failure_receipts.append(copy.deepcopy(dict(value)))
        return {"receipt_path": "/tmp/failure.json"}

    monkeypatch.setattr(runtime, "_write_append_only_receipt", write_receipt)
    lifecycle = runtime.FullCanaryLifecycle(
        plan,
        bootstrap_provisioner=Provisioner(),
    )
    expected = (
        ExceptionGroup
        if cleanup_fails or failure == "persistence"
        else RuntimeError
    )
    with pytest.raises(expected):
        lifecycle.start(approval)

    assert calls.count("reconcile") == 1
    assert calls.index("reconcile") < calls.index("stop")
    if cleanup_fails:
        assert "persist" not in calls
    elif failure == "persistence":
        assert calls.count("persist") == 2
        assert calls.index("reconcile") < calls.index("persist")
        assert calls.index("persist") < calls.index("stop")
    else:
        assert calls.index("reconcile") < calls.index("persist") < calls.index("stop")
    durable = not cleanup_fails and failure != "persistence"
    assert failure_receipts[-1]["bootstrap_reconciliation_complete"] is durable
    assert failure_receipts[-1][
        "bootstrap_authority_may_require_owner_cleanup"
    ] is (not durable)
    assert failure_receipts[-1][
        "bootstrap_durable_evidence_recovery_required"
    ] is (not durable)
    if failure in {"provision_raises", "invalid_provision_receipt"}:
        assert failure_receipts[-1]["bootstrap_provisioning_receipt"] is None


@pytest.mark.parametrize("failure", ["host", "validation", "preflight", "install", "edge"])
def test_every_pre_provision_failure_aborts_admin_session_once(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    plan = _host_receipt_plan(b"{}")
    approval = _owner_approval(plan)
    calls: list[str] = []

    class Provisioner:
        def provision(self, _request):
            calls.append("provision")
            raise AssertionError("pre-provision failure reached DB mutation")

        def reconcile(self, _request, _receipt):
            calls.append("reconcile")
            raise AssertionError("pre-provision failure reached DB reconciliation")

        def abort(self):
            calls.append("abort")

    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)

    def host(self):
        calls.append("host")
        if failure == "host":
            raise RuntimeError("host mismatch")
        return {"exact": True}

    monkeypatch.setattr(runtime.FullCanaryLifecycle, "_require_dedicated_host", host)
    monkeypatch.setattr(
        runtime,
        "_validate_artifact_source",
        lambda *_a, **_k: b"{}",
    )

    def validate(*_args, **_kwargs):
        if failure == "validation":
            raise RuntimeError("writer config invalid")
        return {}

    monkeypatch.setattr(runtime, "_validate_writer_config", validate)
    monkeypatch.setattr(runtime, "_lifecycle_lock", lambda: nullcontext())

    def preflight(self, *, phase):
        if failure == "preflight":
            raise RuntimeError("preflight failed")
        return {"report_sha256": "6" * 64}

    monkeypatch.setattr(runtime.FullCanaryLifecycle, "_preflight", preflight)

    def install(_plan):
        if failure == "install":
            raise RuntimeError("install failed")
        return {}

    monkeypatch.setattr(runtime, "_install_plan_artifacts", install)

    def run_checked(command, **_kwargs):
        if failure == "edge" and command.argv[-1] == runtime.EDGE_UNIT_NAME:
            raise RuntimeError("edge start failed")
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    monkeypatch.setattr(runtime, "_run_checked", run_checked)
    monkeypatch.setattr(runtime, "_stop_all", lambda **_kwargs: ())
    monkeypatch.setattr(
        runtime,
        "_write_append_only_receipt",
        lambda *_a, **_k: {"receipt_path": "/tmp/failure.json"},
    )
    lifecycle = runtime.FullCanaryLifecycle(
        plan,
        bootstrap_provisioner=Provisioner(),
    )

    with pytest.raises(RuntimeError):
        lifecycle.start(approval)

    assert calls.count("abort") == 1
    assert "provision" not in calls
    assert "reconcile" not in calls


def test_pre_provision_primary_and_admin_abort_errors_are_both_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _host_receipt_plan(b"{}")

    class Provisioner:
        def __init__(self):
            self.abort_count = 0

        def provision(self, _request):
            raise AssertionError

        def reconcile(self, _request, _receipt):
            raise AssertionError

        def abort(self):
            self.abort_count += 1
            raise RuntimeError(f"admin close failed {self.abort_count}")

    monkeypatch.setattr(runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        runtime.FullCanaryLifecycle,
        "_require_dedicated_host",
        lambda self: (_ for _ in ()).throw(RuntimeError("host mismatch")),
    )
    provisioner = Provisioner()
    lifecycle = runtime.FullCanaryLifecycle(
        plan,
        bootstrap_provisioner=provisioner,
    )

    with pytest.raises(ExceptionGroup) as captured:
        lifecycle.start(_owner_approval(plan))

    assert str(captured.value.exceptions[0]) == "host mismatch"
    abort_group = captured.value.exceptions[1]
    assert isinstance(abort_group, ExceptionGroup)
    assert [str(error) for error in abort_group.exceptions] == [
        "admin close failed 1",
        "admin close failed 2",
    ]
    assert provisioner.abort_count == 2
