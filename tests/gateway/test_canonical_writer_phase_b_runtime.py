from __future__ import annotations

import inspect
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway import canonical_writer_phase_b_runtime as runtime


REVISION = "a" * 40


def _anonymous_stat(*, directory: bool, mode: int, nlink: int = 0):
    return SimpleNamespace(
        st_mode=(stat.S_IFDIR if directory else stat.S_IFREG) | mode,
        st_uid=0,
        st_gid=0,
        st_nlink=nlink,
    )


def _install_anonymous_descriptor_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "O_CLOEXEC": 0x01,
        "O_DIRECTORY": 0x02,
        "O_EXCL": 0x04,
        "O_NOFOLLOW": 0x08,
        "O_TMPFILE": 0x10,
    }.items():
        monkeypatch.setattr(runtime.os, name, value, raising=False)


def test_secret_descriptor_uses_unlinked_otmpfile_when_memfd_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_anonymous_descriptor_flags(monkeypatch)
    monkeypatch.setattr(runtime.os, "memfd_create", None, raising=False)
    opens: list[tuple[object, ...]] = []
    closed: list[int] = []
    writes: list[tuple[int, bytes]] = []
    inheritable: dict[int, bool] = {}

    def open_descriptor(path, flags, mode=0o777, *, dir_fd=None):
        opens.append((path, flags, mode, dir_fd))
        return 11

    monkeypatch.setattr(
        runtime,
        "_open_trusted_absolute_directory",
        lambda path: 10,
    )
    monkeypatch.setattr(runtime.os, "open", open_descriptor)
    monkeypatch.setattr(
        runtime.os,
        "fstat",
        lambda descriptor: (
            _anonymous_stat(directory=True, mode=0o755, nlink=2)
            if descriptor == 10
            else _anonymous_stat(directory=False, mode=0o400)
        ),
    )
    monkeypatch.setattr(
        runtime.os,
        "set_inheritable",
        lambda descriptor, value: inheritable.__setitem__(descriptor, value),
    )
    monkeypatch.setattr(
        runtime.os,
        "get_inheritable",
        lambda descriptor: inheritable.get(descriptor, False),
    )
    monkeypatch.setattr(runtime.os, "fchmod", lambda descriptor, mode: None)
    monkeypatch.setattr(
        runtime.os,
        "write",
        lambda descriptor, value: writes.append((descriptor, bytes(value)))
        or len(value),
    )
    monkeypatch.setattr(runtime.os, "fsync", lambda descriptor: None)
    monkeypatch.setattr(runtime.os, "close", closed.append)

    secret = bytearray(b"A" * 64)
    with runtime._secret_descriptor(secret) as descriptor:
        assert descriptor == 11
        assert closed == [10]

    assert closed == [10, 11]
    assert writes == [(11, bytes(secret))]
    assert opens[0][0] == "."
    assert opens[0][2] == 0o400
    assert opens[0][3] == 10
    assert opens[0][1] & runtime.os.O_TMPFILE
    assert opens[0][1] & runtime.os.O_EXCL
    assert inheritable == {11: False}


def test_anonymous_secret_fallback_rejects_writable_run_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_anonymous_descriptor_flags(monkeypatch)
    monkeypatch.setattr(runtime.os, "memfd_create", None, raising=False)
    monkeypatch.setattr(
        runtime,
        "_open_trusted_absolute_directory",
        lambda path: 10,
    )
    monkeypatch.setattr(
        runtime.os,
        "fstat",
        lambda _descriptor: _anonymous_stat(
            directory=True,
            mode=0o777,
            nlink=2,
        ),
    )
    monkeypatch.setattr(runtime.os, "set_inheritable", lambda *_args: None)
    monkeypatch.setattr(runtime.os, "get_inheritable", lambda _descriptor: False)
    closed: list[int] = []
    monkeypatch.setattr(runtime.os, "close", closed.append)

    with pytest.raises(
        runtime.PhaseBRuntimeError,
        match="phase_b_runtime_anonymous_secret_directory_invalid",
    ):
        runtime._open_anonymous_secret_descriptor()

    assert closed == [10]


def test_anonymous_secret_fallback_rejects_linked_inode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_anonymous_descriptor_flags(monkeypatch)
    monkeypatch.setattr(runtime.os, "memfd_create", None, raising=False)
    monkeypatch.setattr(
        runtime,
        "_open_trusted_absolute_directory",
        lambda path: 10,
    )
    monkeypatch.setattr(
        runtime.os,
        "open",
        lambda _path, _flags, _mode=0o777, *, dir_fd=None: 11,
    )
    monkeypatch.setattr(
        runtime.os,
        "fstat",
        lambda descriptor: (
            _anonymous_stat(directory=True, mode=0o755, nlink=2)
            if descriptor == 10
            else _anonymous_stat(directory=False, mode=0o400, nlink=1)
        ),
    )
    monkeypatch.setattr(runtime.os, "set_inheritable", lambda *_args: None)
    monkeypatch.setattr(runtime.os, "get_inheritable", lambda _descriptor: False)
    monkeypatch.setattr(runtime.os, "fchmod", lambda *_args: None)
    closed: list[int] = []
    monkeypatch.setattr(runtime.os, "close", closed.append)

    with pytest.raises(
        runtime.PhaseBRuntimeError,
        match="phase_b_runtime_anonymous_secret_identity_invalid",
    ):
        runtime._open_anonymous_secret_descriptor()

    assert closed == [10, 11]


def test_memfd_remains_preferred_and_must_be_unlinked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime.os, "memfd_create", lambda *_args: 12, raising=False)
    monkeypatch.setattr(
        runtime.os,
        "fstat",
        lambda _descriptor: _anonymous_stat(directory=False, mode=0o400),
    )
    inheritable: dict[int, bool] = {}
    monkeypatch.setattr(
        runtime.os,
        "set_inheritable",
        lambda descriptor, value: inheritable.__setitem__(descriptor, value),
    )
    monkeypatch.setattr(
        runtime.os,
        "get_inheritable",
        lambda descriptor: inheritable.get(descriptor, False),
    )
    monkeypatch.setattr(
        runtime.os,
        "open",
        lambda *_args, **_kwargs: pytest.fail("O_TMPFILE fallback was used"),
    )
    monkeypatch.setattr(runtime.os, "fchmod", lambda *_args: None)

    assert runtime._open_anonymous_secret_descriptor() == 12
    assert inheritable == {12: False}


@pytest.mark.parametrize(
    ("status", "inheritable"),
    (
        (_anonymous_stat(directory=False, mode=0o400, nlink=1), False),
        (
            SimpleNamespace(
                st_mode=stat.S_IFREG | 0o400,
                st_uid=1,
                st_gid=0,
                st_nlink=0,
            ),
            False,
        ),
        (
            SimpleNamespace(
                st_mode=stat.S_IFREG | 0o400,
                st_uid=0,
                st_gid=1,
                st_nlink=0,
            ),
            False,
        ),
        (_anonymous_stat(directory=False, mode=0o600), False),
        (_anonymous_stat(directory=False, mode=0o400), True),
    ),
)
def test_anonymous_secret_identity_rejects_unsafe_inode(
    monkeypatch: pytest.MonkeyPatch,
    status,
    inheritable: bool,
) -> None:
    monkeypatch.setattr(runtime.os, "set_inheritable", lambda *_args: None)
    monkeypatch.setattr(runtime.os, "fchmod", lambda *_args: None)
    monkeypatch.setattr(runtime.os, "fstat", lambda _descriptor: status)
    monkeypatch.setattr(
        runtime.os,
        "get_inheritable",
        lambda _descriptor: inheritable,
    )

    with pytest.raises(
        runtime.PhaseBRuntimeError,
        match="phase_b_runtime_anonymous_secret_identity_invalid",
    ):
        runtime._validate_anonymous_secret_descriptor(11)


def test_anonymous_secret_fallback_requires_all_linux_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_anonymous_descriptor_flags(monkeypatch)
    monkeypatch.setattr(runtime.os, "memfd_create", None, raising=False)
    monkeypatch.setattr(runtime.os, "O_TMPFILE", 0, raising=False)
    monkeypatch.setattr(
        runtime,
        "_open_trusted_absolute_directory",
        lambda _path: pytest.fail("directory was opened"),
    )

    with pytest.raises(
        runtime.PhaseBRuntimeError,
        match="phase_b_runtime_anonymous_secret_unavailable",
    ):
        runtime._open_anonymous_secret_descriptor()


def test_secret_descriptor_closes_after_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_open_anonymous_secret_descriptor", lambda: 11)
    monkeypatch.setattr(
        runtime.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("write failed")),
    )
    closed: list[int] = []
    monkeypatch.setattr(runtime.os, "close", closed.append)

    with pytest.raises(
        runtime.PhaseBRuntimeError,
        match="phase_b_runtime_secret_transport_failed",
    ):
        with runtime._secret_descriptor(bytearray(b"A" * 64)):
            pytest.fail("descriptor was yielded")

    assert closed == [11]


def test_secret_descriptor_closes_after_consumer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_open_anonymous_secret_descriptor", lambda: 11)
    monkeypatch.setattr(runtime.os, "write", lambda _fd, value: len(value))
    monkeypatch.setattr(runtime.os, "fsync", lambda _fd: None)
    closed: list[int] = []
    monkeypatch.setattr(runtime.os, "close", closed.append)

    with pytest.raises(RuntimeError, match="consumer failed"):
        with runtime._secret_descriptor(bytearray(b"A" * 64)):
            raise RuntimeError("consumer failed")

    assert closed == [11]


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
