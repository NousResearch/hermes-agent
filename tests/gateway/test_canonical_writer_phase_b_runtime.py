from __future__ import annotations

import inspect
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway import canonical_writer_phase_b_runtime as runtime


REVISION = "a" * 40


def _systemd_result(
    *,
    active: bool = False,
    unit: str = "muncho-canonical-writer.service",
) -> subprocess.CompletedProcess[str]:
    values = {
        "LoadState": "loaded",
        "ActiveState": "active" if active else "inactive",
        "SubState": "running" if active else "dead",
        "UnitFileState": "disabled",
        "MainPID": "123" if active else "0",
        "FragmentPath": "/etc/systemd/system/fixed.service",
        "DropInPaths": "",
        "TriggeredBy": "",
        "Triggers": "",
        "NextElapseUSecRealtime": "",
    }
    if unit in runtime._PIDLESS_SERVICE_UNITS:
        values.pop("MainPID")
    stdout = "".join(f"{name}={value}\n" for name, value in values.items())
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_zero_input_cli_and_collector_have_no_evidence_mapping_surface() -> None:
    assert runtime._parser().parse_args([]) is not None
    with pytest.raises(SystemExit):
        runtime._parser().parse_args(["--revision", REVISION])
    signature = inspect.signature(runtime._collect_fixed_phase_b_readiness_mapping)
    assert tuple(signature.parameters) == (
        "foundation_value",
        "current_release_revision",
        "observed_at_unix",
    )
    assert "mapping" not in signature.parameters
    assert "collector" not in signature.parameters
    assert "host" not in signature.parameters
    assert "database" not in signature.parameters
    assert "service" not in signature.parameters
    assert "publish_phase_b_readiness" not in phase_b.__all__
    assert tuple(
        inspect.signature(runtime.publish_fixed_phase_b_readiness).parameters
    ) == ()
    with pytest.raises(TypeError):
        runtime.publish_fixed_phase_b_readiness(  # type: ignore[call-arg]
            current_release_revision=REVISION,
            now_unix=1_700_000_000,
        )
    assert "build_phase_b_dependencies" not in runtime.__all__
    assert "load_fixed_phase_b_authority" not in runtime.__all__


def test_fixed_seven_service_collection_is_stopped_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def run(command, **_kwargs):
        calls.append(tuple(command))
        return _systemd_result(unit=command[-1])

    monkeypatch.setattr(runtime.subprocess, "run", run)
    observed = runtime._collect_services(REVISION, 1_700_000_000)
    assert [item["name"] for item in observed["services"]] == list(
        phase_b.SERVICE_UNITS
    )
    assert observed["services_stopped_and_disabled"] is True
    assert all(command[0] == "/usr/bin/systemctl" for command in calls)
    assert all(command[-2] == "--" for command in calls)
    assert [command[-1] for command in calls] == list(phase_b.SERVICE_UNITS)
    timer = next(
        item
        for item in observed["services"]
        if item["name"] == "muncho-canonical-writer-export.timer"
    )
    assert timer["main_pid"] == 0
    assert "password" not in json.dumps(observed).casefold()


def test_missing_main_pid_remains_invalid_for_service_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _systemd_result()
    result = subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.replace("MainPID=0\n", ""),
        result.stderr,
    )
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_args, **_kwargs: result)

    with pytest.raises(runtime.PhaseBRuntimeError, match="systemd_invalid"):
        runtime._collect_one_service("muncho-canonical-writer.service")


def test_active_service_blocks_readiness_before_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: _systemd_result(active=True),
    )
    with pytest.raises(phase_b.PhaseBError, match="service_snapshot"):
        phase_b._validate_services(
            runtime._collect_services(REVISION, 1_700_000_000),
            release_revision=REVISION,
        )


def test_fixed_authority_loader_rejects_noncanonical_or_writable_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The production loader intentionally rejects every world-writable
    # ancestor.  pytest's normal macOS temp root is /private/tmp (mode 1777),
    # so a tmp_path fixture cannot represent a valid authority hierarchy.
    repository = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(
        prefix="phase-b-authority-", dir=repository
    ) as directory:
        trusted_root = Path(directory)
        plan_path = trusted_root / "plan.json"
        approval_path = trusted_root / "approval.json"
        monkeypatch.setattr(runtime, "PHASE_B_PLAN_PATH", plan_path)
        monkeypatch.setattr(runtime, "PHASE_B_APPROVAL_PATH", approval_path)
        monkeypatch.setattr(runtime, "_ROOT_UID", os.geteuid())
        monkeypatch.setattr(runtime, "_ROOT_GID", os.getegid())

        plan_path.write_text('{"a":1}\n', encoding="utf-8")
        approval_path.write_text('{"b":2}\n', encoding="utf-8")
        plan_path.chmod(0o400)
        approval_path.chmod(0o400)
        assert runtime._read_fixed_root_json(plan_path) == {"a": 1}

        plan_path.chmod(0o600)
        with pytest.raises(runtime.PhaseBRuntimeError, match="untrusted"):
            runtime._read_fixed_root_json(plan_path)
        plan_path.write_text('{ "a": 1 }\n', encoding="utf-8")
        plan_path.chmod(0o400)
        with pytest.raises(runtime.PhaseBRuntimeError, match="not_canonical"):
            runtime._read_fixed_root_json(plan_path)


def test_production_dependencies_require_typed_plan_and_cloud_boundary() -> None:
    signature = inspect.signature(runtime.build_phase_b_dependencies)
    assert tuple(signature.parameters) == ("plan", "cloud")
    with pytest.raises(TypeError, match="PhaseBPlan"):
        runtime.build_phase_b_dependencies(object(), object())  # type: ignore[arg-type]


def test_authority_recollection_uses_signed_session_not_recovery_preapproval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import canonical_full_canary_coordinator as coordinator

    owner_subject_sha256 = "b" * 64
    owner_key_file_sha256 = "c" * 64
    owner_source = {
        "path": coordinator.PHASE_B_OWNER_PUBLIC_KEY_PATH,
        "file_sha256": owner_key_file_sha256,
        "device": 1,
        "inode": 2,
        "uid": coordinator.PHASE_B_OWNER_PUBLIC_KEY_UID,
        "gid": coordinator.PHASE_B_OWNER_PUBLIC_KEY_GID,
        "mode": "0600",
        "size": 64,
    }
    provenance = {
        "approval_source_sha256": coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
        "owner_subject_sha256": owner_subject_sha256,
        "owner_resume_public_key_ed25519_hex": None,
        "owner_resume_key_id": None,
        "owner_resume_public_key_file_sha256": None,
        "owner_resume_public_fingerprint": None,
        "authority_sources": {"owner_resume_public_key": None},
    }
    authority = {
        "approval_source_sha256": coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256,
        "owner_subject_sha256": owner_subject_sha256,
        "authority_sources": {"owner_resume_public_key": owner_source},
    }
    plan = SimpleNamespace(
        owner_subject_sha256=owner_subject_sha256,
        value={
            "owner_resume_public_key_ed25519_hex": "e" * 64,
            "owner_resume_key_id": "f" * 64,
            "owner_resume_public_key_file_sha256": owner_key_file_sha256,
        },
    )
    coordinator_input = object()
    monkeypatch.setattr(
        coordinator,
        "load_coordinator_input",
        lambda: coordinator_input,
    )
    monkeypatch.setattr(
        coordinator,
        "_phase_b_authority_provenance",
        lambda value: provenance if value is coordinator_input else None,
    )

    result = runtime._recollect_fixed_authority_provenance(
        plan=plan,  # type: ignore[arg-type]
        authority=authority,
    )

    assert result["approval_source_sha256"] == (
        coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
    )
    assert result["owner_subject_sha256"] == owner_subject_sha256
    assert result["authority_sources"]["owner_resume_public_key"] == owner_source


def test_authority_recollection_rejects_unpinned_session_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import canonical_full_canary_coordinator as coordinator

    monkeypatch.setattr(coordinator, "load_coordinator_input", lambda: object())
    monkeypatch.setattr(
        coordinator,
        "_phase_b_authority_provenance",
        lambda _value: {
            "approval_source_sha256": (
                coordinator.PHASE_B_PINNED_APPROVAL_SOURCE_SHA256
            ),
            "owner_subject_sha256": "b" * 64,
        },
    )
    plan = SimpleNamespace(owner_subject_sha256="b" * 64, value={})

    with pytest.raises(
        runtime.PhaseBRuntimeError,
        match="phase_b_runtime_authority_source_invalid",
    ):
        runtime._recollect_fixed_authority_provenance(
            plan=plan,  # type: ignore[arg-type]
            authority={
                "approval_source_sha256": "0" * 64,
                "owner_subject_sha256": "b" * 64,
            },
        )


def test_cloud_readiness_uses_only_condition_compatible_instance_and_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    operation = {
        "kind": "sql#operationsList",
        "items": [],
    }

    def get(_token: str, url: str):
        calls.append(url)
        if url.endswith("/instances/muncho-canary-pg18-v2"):
            return {
                "kind": "sql#instance",
                **runtime._FIXED_INSTANCE_PROJECTION,
            }
        assert "/operations?" in url
        assert "instance=muncho-canary-pg18-v2" in url
        return operation

    monkeypatch.setattr(runtime, "_cloud_get", get)
    snapshot = runtime._collect_stable_cloud_snapshot("opaque-token")
    assert snapshot.instance_projection == runtime._FIXED_INSTANCE_PROJECTION
    assert snapshot.relevant_user_operations == ()
    assert len(calls) == 6
    assert sum(url.endswith("/instances/muncho-canary-pg18-v2") for url in calls) == 2
    assert all("/users" not in url for url in calls)


def test_cloud_instance_projection_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "kind": "sql#instance",
        **runtime._FIXED_INSTANCE_PROJECTION,
    }
    payload["state"] = "SUSPENDED"
    monkeypatch.setattr(runtime, "_cloud_get", lambda *_args: payload)
    with pytest.raises(runtime.PhaseBRuntimeError, match="instance_invalid"):
        runtime._cloud_sql_instance("opaque-token")
