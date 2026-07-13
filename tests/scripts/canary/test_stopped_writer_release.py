from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.canary import writer_release


REVISION = "a" * 40
TREE_SHA = "b" * 40


@pytest.fixture(autouse=True)
def _tmp_path_uses_process_primary_group(tmp_path):
    os.chown(tmp_path, -1, os.getgid())


def _allow_local_owner(monkeypatch) -> None:
    monkeypatch.setattr(writer_release, "_BUILD_OWNER_UID", os.geteuid())
    monkeypatch.setattr(writer_release, "_BUILD_OWNER_GID", os.getegid())


def _host() -> dict[str, str]:
    gce = {
        "project_id": "adventico-ai-platform",
        "project_number": "39589465056",
        "zone": "europe-west3-a",
        "instance_name": "muncho-canary-v2-01",
        "instance_id": "9153645328899914617",
        "service_account_email": (
            "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
        ),
    }
    return {
        **gce,
        "gce_identity_sha256": writer_release._sha256_json(gce),
        "machine_id_sha256": "1" * 64,
        "hostname_sha256": "2" * 64,
        "host_identity_sha256": "3" * 64,
        "boot_id_sha256": "4" * 64,
    }


def _service_properties(unit: str, *, loaded: bool = False) -> dict[str, str]:
    if loaded:
        return {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "UnitFileState": "disabled",
            "MainPID": "0",
            "FragmentPath": f"/etc/systemd/system/{unit}",
            "DropInPaths": "",
        }
    return {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "",
        "MainPID": "0",
        "FragmentPath": "",
        "DropInPaths": "",
    }


def _service_stdout(unit: str, *, loaded: bool = False) -> str:
    values = _service_properties(unit, loaded=loaded)
    return "".join(
        f"{name}={values[name]}\n" for name in writer_release._SERVICE_PROPERTIES
    )


class _PlanRunner:
    def __init__(self) -> None:
        self.commands: list[writer_release.BuildCommand] = []

    def __call__(
        self, command: writer_release.BuildCommand
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        argv = command.argv
        if argv[0] == str(writer_release.DEFAULT_SYSTEMCTL_EXECUTABLE):
            stdout = _service_stdout(argv[-1])
        elif argv[-3:] == ("config", "--get", "remote.origin.url"):
            stdout = writer_release.FORK_REPOSITORY + "\n"
        elif argv[-2:] == ("--verify", "HEAD^{tree}"):
            stdout = TREE_SHA + "\n"
        else:  # pragma: no cover - a new command must be explicitly reviewed.
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout, "")


def _patch_plan_filesystem(monkeypatch) -> list[Path]:
    executables: list[Path] = []
    monkeypatch.setattr(writer_release, "_validate_root_parent_chain", lambda _p: None)
    monkeypatch.setattr(
        writer_release,
        "_validate_root_executable",
        lambda path: executables.append(path),
    )
    monkeypatch.setattr(writer_release, "_validate_root_source_tree", lambda _p: None)
    monkeypatch.setattr(
        writer_release,
        "verify_clean_checkout",
        lambda _spec, runner: None,
    )
    return executables


def test_plan_is_deterministic_fixed_and_self_digest_bound(monkeypatch):
    executables = _patch_plan_filesystem(monkeypatch)
    runner = _PlanRunner()

    first = writer_release.plan_stopped_release(
        REVISION,
        runner=runner,
        host_observer=_host,
        path_exists=lambda _path: False,
    )
    second = writer_release.plan_stopped_release(
        REVISION,
        runner=runner,
        host_observer=_host,
        path_exists=lambda _path: False,
    )

    assert first == second
    unsigned = {name: value for name, value in first.items() if name != "plan_sha256"}
    assert first["plan_sha256"] == writer_release._sha256_json(unsigned)
    assert first["source"] == {
        "repository": writer_release.FORK_REPOSITORY,
        "root": f"/opt/muncho-canary-source/{REVISION}",
        "head_sha": REVISION,
        "tree_sha": TREE_SHA,
    }
    assert first["release_root"] == f"/opt/muncho-canary-releases/{REVISION}"
    assert first["host_identity_receipt_path"] == (
        "/etc/muncho/full-canary/host-identity.json"
    )
    assert [item["path"] for item in first["activation_inventory"]] == [
        str(path) for path in writer_release._ACTIVATION_PATHS
    ]
    assert len(first["activation_inventory"]) == 17
    assert {item["state"] for item in first["activation_inventory"]} == {"absent"}
    assert [item["unit"] for item in first["service_states"]] == list(
        writer_release._STOPPED_SERVICE_UNITS
    )
    assert executables.count(writer_release.DEFAULT_SYSTEMCTL_EXECUTABLE) == 2
    assert set(executables) == {
        writer_release.DEFAULT_SYSTEMCTL_EXECUTABLE,
        writer_release.DEFAULT_UV_EXECUTABLE,
        writer_release.DEFAULT_GIT_EXECUTABLE,
    }
    assert "created_at_unix" not in first


def test_plan_rejects_activation_collision_including_dangling_symlink(monkeypatch):
    _patch_plan_filesystem(monkeypatch)
    collision = writer_release._ACTIVATION_PATHS[3]

    with pytest.raises(RuntimeError, match="not fresh"):
        writer_release.plan_stopped_release(
            REVISION,
            runner=_PlanRunner(),
            host_observer=_host,
            path_exists=lambda path: path == collision,
        )


def test_source_identity_rejects_nonexact_remote_framing(monkeypatch):
    _patch_plan_filesystem(monkeypatch)

    class BadRemoteRunner(_PlanRunner):
        def __call__(self, command):
            result = super().__call__(command)
            if command.argv[-3:] == ("config", "--get", "remote.origin.url"):
                return subprocess.CompletedProcess(
                    command.argv,
                    0,
                    writer_release.FORK_REPOSITORY + "\n\n",
                    "",
                )
            return result

    with pytest.raises(RuntimeError, match="fixed fork"):
        writer_release.plan_stopped_release(
            REVISION,
            runner=BadRemoteRunner(),
            host_observer=_host,
            path_exists=lambda _path: False,
        )


def test_fixed_systemctl_observation_is_read_only_and_closed():
    runner = _PlanRunner()

    states = writer_release._collect_service_states(runner=runner)

    service_commands = [
        command
        for command in runner.commands
        if command.argv[0] == str(writer_release.DEFAULT_SYSTEMCTL_EXECUTABLE)
    ]
    assert len(states) == len(service_commands) == 3
    for command, unit in zip(service_commands, writer_release._STOPPED_SERVICE_UNITS):
        assert command.argv == (
            "/usr/bin/systemctl",
            "show",
            "--no-pager",
            *(f"--property={name}" for name in writer_release._SERVICE_PROPERTIES),
            unit,
        )
        assert command.environment() == {
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
        assert not {"start", "stop", "restart", "enable", "disable"} & set(command.argv)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ActiveState", "active"),
        ("SubState", "failed"),
        ("UnitFileState", "enabled"),
        ("MainPID", "12"),
        ("FragmentPath", "/usr/lib/systemd/system/wrong.service"),
        ("DropInPaths", "/etc/systemd/system/x.conf"),
    ],
)
def test_service_parser_rejects_every_unsafe_state(field, value):
    unit = writer_release._STOPPED_SERVICE_UNITS[0]
    properties = _service_properties(unit, loaded=True)
    properties[field] = value
    raw = "".join(
        f"{name}={properties[name]}\n" for name in writer_release._SERVICE_PROPERTIES
    )

    with pytest.raises(RuntimeError, match="not safely stopped"):
        writer_release._parse_service_observation(unit, raw)


def test_service_parser_accepts_only_exact_absent_or_disabled_state():
    unit = writer_release._STOPPED_SERVICE_UNITS[0]

    assert (
        writer_release._parse_service_observation(unit, _service_stdout(unit))["state"]
        == "absent"
    )
    assert (
        writer_release._parse_service_observation(
            unit, _service_stdout(unit, loaded=True)
        )["state"]
        == "disabled_inactive"
    )
    with pytest.raises(RuntimeError, match="unexpected fields"):
        writer_release._parse_service_observation(
            unit,
            _service_stdout(unit) + "LoadState=not-found\n",
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        writer_release._parse_service_observation(
            unit,
            "LoadState=not-found\n",
        )


def test_host_observation_requires_exact_vm_fields_and_digests():
    assert writer_release._validate_host_observation(_host()) == dict(
        sorted(_host().items())
    )
    missing = _host()
    missing.pop("boot_id_sha256")
    with pytest.raises(RuntimeError, match="incomplete"):
        writer_release._validate_host_observation(missing)
    wrong = _host()
    wrong["instance_id"] = "wrong"
    with pytest.raises(RuntimeError, match="fixed VM"):
        writer_release._validate_host_observation(wrong)


def test_cli_parser_is_constructible_and_rejects_extra_or_repeated_inputs():
    parser = writer_release._cli_parser()
    assert parser.parse_args(["plan", "--revision", REVISION]).revision == REVISION
    assert (
        parser.parse_args([
            "apply",
            "--revision",
            REVISION,
            "--approved-plan-sha256",
            "c" * 64,
        ]).approved_plan_sha256
        == "c" * 64
    )
    invalid = (
        ["plan", "--rev", REVISION],
        ["plan", "--revision", REVISION, "--path", "/tmp/x"],
        ["plan", "--revision", REVISION, "--revision", REVISION],
        ["apply", "--revision", REVISION],
        ["plan", "--revision", REVISION, "--approved-plan-sha256", "c" * 64],
    )
    for argv in invalid:
        with pytest.raises(ValueError):
            parser.parse_args(argv)


def test_cli_failure_is_canonical_and_does_not_echo_input(capsys):
    assert writer_release.main(["plan", "--revision", "DO_NOT_ECHO_ME"]) == 2

    raw = capsys.readouterr().out
    value = json.loads(raw)
    assert raw == writer_release._canonical_bytes(value).decode() + "\n"
    assert value == {
        "schema": writer_release.STOPPED_RELEASE_FAILURE_SCHEMA,
        "ok": False,
        "error_code": "stopped_release_failed",
        "error_type": "ValueError",
    }
    assert "DO_NOT_ECHO_ME" not in raw


def _completed_release(tmp_path: Path, monkeypatch):
    _allow_local_owner(monkeypatch)
    spec = writer_release.ReleaseBuildSpec(
        revision=REVISION,
        source_root=tmp_path / "source",
        release_base=tmp_path / "releases",
        uv_executable=tmp_path / "bin/uv",
        git_executable=tmp_path / "bin/git",
        uv_cache_dir=tmp_path / "uv-cache",
    )
    spec.release_root.mkdir(parents=True)
    spec.interpreter.parent.mkdir(parents=True)
    spec.writer_module_origin.parent.mkdir(parents=True)
    spec.wheel_artifact_root.mkdir(parents=True)
    spec.interpreter.write_bytes(b"python-binary")
    spec.writer_module_origin.write_text("writer = True\n", encoding="utf-8")
    spec.gateway_module_origin.write_text("gateway = True\n", encoding="utf-8")
    wheel = spec.wheel_artifact_root / "hermes_agent-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")
    for current, directories, files in os.walk(
        spec.release_root, topdown=False, followlinks=False
    ):
        for name in files:
            path = Path(current) / name
            path.chmod(
                writer_release._SEALED_EXECUTABLE_MODE
                if path == spec.interpreter
                else writer_release._SEALED_FILE_MODE
            )
        for name in directories:
            (Path(current) / name).chmod(writer_release._SEALED_DIRECTORY_MODE)
    manifest = writer_release.create_release_manifest(spec)
    monkeypatch.setattr(writer_release.os, "fchown", lambda *_args: None)
    writer_release._write_release_manifest(spec.release_root, manifest)
    spec.release_root.chmod(writer_release._SEALED_DIRECTORY_MODE)
    return spec, manifest


def test_completed_release_reconstructs_manifest_after_publication(
    tmp_path, monkeypatch
):
    spec, manifest = _completed_release(tmp_path, monkeypatch)

    binding = writer_release._validate_completed_release(spec)

    assert binding["release_artifact_sha256"] == manifest.artifact_sha256
    assert (
        binding["release_manifest_file_sha256"]
        == hashlib.sha256(
            (spec.release_root / writer_release.RELEASE_MANIFEST_NAME).read_bytes()
        ).hexdigest()
    )
    assert (
        binding["retained_wheel_sha256"] == hashlib.sha256(b"wheel-bytes").hexdigest()
    )
    spec.release_root.chmod(0o700)
    extra = spec.release_root / "unexpected"
    extra.write_bytes(b"drift")
    extra.chmod(writer_release._SEALED_FILE_MODE)
    spec.release_root.chmod(writer_release._SEALED_DIRECTORY_MODE)
    with pytest.raises(RuntimeError, match="does not match"):
        writer_release._validate_completed_release(spec)


def test_completed_release_uses_the_shared_planner_manifest_bound(
    tmp_path, monkeypatch
):
    spec, _manifest = _completed_release(tmp_path, monkeypatch)
    observed: list[int] = []
    read_stable_root_file = writer_release._read_stable_root_file

    def capture_manifest_bound(path, *, maximum_bytes, exact_mode):
        if path == spec.release_root / writer_release.RELEASE_MANIFEST_NAME:
            observed.append(maximum_bytes)
        return read_stable_root_file(
            path,
            maximum_bytes=maximum_bytes,
            exact_mode=exact_mode,
        )

    monkeypatch.setattr(
        writer_release,
        "_read_stable_root_file",
        capture_manifest_bound,
    )

    writer_release._validate_completed_release(spec)

    assert observed == [writer_release.MAX_RELEASE_MANIFEST_BYTES]
    assert writer_release.MAX_RELEASE_MANIFEST_BYTES > writer_release._MAX_RECEIPT_BYTES


def test_manifest_bound_accepts_more_than_a_receipt_and_rejects_its_own_overflow(
    tmp_path, monkeypatch
):
    _allow_local_owner(monkeypatch)
    manifest = tmp_path / writer_release.RELEASE_MANIFEST_NAME
    manifest.write_bytes(b"m" * (writer_release._MAX_RECEIPT_BYTES + 1))
    manifest.chmod(writer_release._MANIFEST_MODE)

    raw = writer_release._read_stable_root_file(
        manifest,
        maximum_bytes=writer_release.MAX_RELEASE_MANIFEST_BYTES,
        exact_mode=writer_release._MANIFEST_MODE,
    )

    assert len(raw) == writer_release._MAX_RECEIPT_BYTES + 1
    manifest.chmod(0o600)
    with manifest.open("r+b") as handle:
        handle.truncate(writer_release.MAX_RELEASE_MANIFEST_BYTES + 1)
    manifest.chmod(writer_release._MANIFEST_MODE)
    with pytest.raises(RuntimeError, match="evidence file is not exact"):
        writer_release._read_stable_root_file(
            manifest,
            maximum_bytes=writer_release.MAX_RELEASE_MANIFEST_BYTES,
            exact_mode=writer_release._MANIFEST_MODE,
        )


def _host_receipt(plan: dict[str, object], observed_at_unix: int) -> dict[str, object]:
    unsigned = {
        "schema": "muncho-full-canary-host-identity.v1",
        "collector_authority": "trusted_root_read_only_host_collector",
        **plan["dedicated_host"],
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "receipt_sha256": writer_release._sha256_json(unsigned)}


def test_host_receipt_is_future_traversable_no_replace_and_same_boot_idempotent(
    tmp_path, monkeypatch
):
    _allow_local_owner(monkeypatch)
    path = tmp_path / "etc/muncho/full-canary/host-identity.json"
    path.parent.parent.mkdir(parents=True)
    monkeypatch.setattr(writer_release, "DEFAULT_HOST_RECEIPT_PATH", path)
    monkeypatch.setattr(writer_release, "_validate_root_parent_chain", lambda _p: None)
    plan = {"dedicated_host": _host()}

    binding = writer_release._publish_or_validate_host_receipt(
        plan,
        observed_at_unix=123,
        collector=lambda observed: _host_receipt(plan, observed),
    )

    assert stat.S_IMODE(os.lstat(path.parent).st_mode) == 0o755
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o400
    assert path.read_bytes() == writer_release._canonical_bytes(
        _host_receipt(plan, 123)
    )
    assert (
        binding["host_identity_receipt_file_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    before = os.lstat(path)
    retry = writer_release._publish_or_validate_host_receipt(
        plan,
        observed_at_unix=999,
        collector=lambda _observed: (_ for _ in ()).throw(AssertionError()),
    )
    after = os.lstat(path)
    assert retry == binding
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    drifted = {"dedicated_host": {**_host(), "boot_id_sha256": "9" * 64}}
    with pytest.raises(RuntimeError, match="stale or invalid"):
        writer_release._publish_or_validate_host_receipt(
            drifted,
            observed_at_unix=999,
            collector=lambda _observed: _host_receipt(drifted, 999),
        )


def test_idempotent_evidence_revision_rejects_unexpected_siblings(
    tmp_path, monkeypatch
):
    _allow_local_owner(monkeypatch)
    evidence = tmp_path / "evidence"
    revision_root = evidence / REVISION
    revision_root.mkdir(parents=True)
    evidence.chmod(0o700)
    revision_root.chmod(0o700)
    (revision_root / "stopped-release-publication.json").write_bytes(b"{}\n")
    (revision_root / "unexpected").write_bytes(b"collision")
    monkeypatch.setattr(writer_release, "DEFAULT_EVIDENCE_BASE", evidence)
    monkeypatch.setattr(writer_release, "_validate_root_parent_chain", lambda _p: None)

    with pytest.raises(RuntimeError, match="extra entries"):
        writer_release._validate_evidence_namespace(REVISION, receipt_exists=True)


def _plan_for_apply(revision: str) -> dict[str, object]:
    spec = writer_release._stopped_release_spec(revision)
    unsigned: dict[str, object] = {
        "schema": writer_release.STOPPED_RELEASE_PLAN_SCHEMA,
        "revision": revision,
        "source": {
            "repository": writer_release.FORK_REPOSITORY,
            "root": str(spec.source_root),
            "head_sha": revision,
            "tree_sha": TREE_SHA,
        },
        "release_root": str(spec.release_root),
        "release_manifest_path": str(
            spec.release_root / writer_release.RELEASE_MANIFEST_NAME
        ),
        "evidence_receipt_path": str(
            writer_release.DEFAULT_EVIDENCE_BASE
            / revision
            / "stopped-release-publication.json"
        ),
        "host_identity_receipt_path": str(writer_release.DEFAULT_HOST_RECEIPT_PATH),
        "python_version": spec.python_version,
        "interpreter": str(spec.interpreter),
        "tools": {
            "git": str(spec.git_executable),
            "systemctl": str(writer_release.DEFAULT_SYSTEMCTL_EXECUTABLE),
            "uv": str(spec.uv_executable),
            "uv_cache": str(spec.uv_cache_dir),
        },
        "dedicated_host": _host(),
        "activation_inventory": [
            {"path": str(path), "state": "absent"}
            for path in writer_release._ACTIVATION_PATHS
        ],
        "service_states": [
            {
                "unit": unit,
                "state": "absent",
                "properties": _service_properties(unit),
            }
            for unit in writer_release._STOPPED_SERVICE_UNITS
        ],
    }
    return {**unsigned, "plan_sha256": writer_release._sha256_json(unsigned)}


def _release_binding(spec: writer_release.ReleaseBuildSpec) -> dict[str, str]:
    return {
        "release_root": str(spec.release_root),
        "release_manifest_path": str(spec.release_root / "release-manifest.json"),
        "release_manifest_file_sha256": "5" * 64,
        "release_artifact_sha256": "6" * 64,
        "interpreter": str(spec.interpreter),
        "interpreter_sha256": "7" * 64,
        "python_version": spec.python_version,
        "retained_wheel_path": str(spec.wheel_artifact_root / "release.whl"),
        "retained_wheel_sha256": "8" * 64,
        "build_constraints_sha256": "9" * 64,
    }


def _manifest_with_digest(spec: writer_release.ReleaseBuildSpec, digest: str):
    return writer_release.ReleaseManifest(
        revision=spec.revision,
        artifact_root=str(spec.release_root),
        python_version=spec.python_version,
        interpreter=str(spec.interpreter),
        writer_module_origin=str(spec.writer_module_origin),
        gateway_module_origin=str(spec.gateway_module_origin),
        entries=(),
        artifact_sha256=digest,
    )


def test_apply_requires_exact_approval_before_any_write(monkeypatch):
    plan = _plan_for_apply(REVISION)
    monkeypatch.setattr(writer_release, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        writer_release,
        "plan_stopped_release",
        lambda *_args, **_kwargs: copy.deepcopy(plan),
    )
    built: list[bool] = []

    with pytest.raises(PermissionError, match="does not match"):
        writer_release.apply_stopped_release(
            REVISION,
            "f" * 64,
            release_builder=lambda *_args, **_kwargs: built.append(True),
        )

    assert built == []


def test_host_receipt_collision_is_preflighted_before_release_build(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(writer_release, "DEFAULT_RELEASE_BASE", tmp_path / "releases")
    monkeypatch.setattr(writer_release, "DEFAULT_EVIDENCE_BASE", tmp_path / "evidence")
    monkeypatch.setattr(writer_release, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        writer_release, "_validate_evidence_namespace", lambda *_a, **_k: None
    )
    plan = _plan_for_apply(REVISION)
    monkeypatch.setattr(
        writer_release,
        "plan_stopped_release",
        lambda *_args, **_kwargs: copy.deepcopy(plan),
    )
    monkeypatch.setattr(
        writer_release,
        "_preflight_host_receipt_namespace",
        lambda _plan: (_ for _ in ()).throw(RuntimeError("stale host receipt")),
    )
    built: list[bool] = []

    with pytest.raises(RuntimeError, match="stale host receipt"):
        writer_release.apply_stopped_release(
            REVISION,
            plan["plan_sha256"],
            release_builder=lambda *_args, **_kwargs: built.append(True),
        )

    assert built == []


def test_apply_publishes_exact_receipt_and_retry_is_read_only(tmp_path, monkeypatch):
    _allow_local_owner(monkeypatch)
    monkeypatch.setattr(writer_release, "DEFAULT_SOURCE_BASE", tmp_path / "sources")
    monkeypatch.setattr(writer_release, "DEFAULT_RELEASE_BASE", tmp_path / "releases")
    monkeypatch.setattr(writer_release, "DEFAULT_EVIDENCE_BASE", tmp_path / "evidence")
    host_path = tmp_path / "host/host-identity.json"
    monkeypatch.setattr(writer_release, "DEFAULT_HOST_RECEIPT_PATH", host_path)
    monkeypatch.setattr(writer_release, "_require_root_linux", lambda: None)
    monkeypatch.setattr(writer_release, "_validate_root_parent_chain", lambda _p: None)
    plan = _plan_for_apply(REVISION)
    calls = {"plan": 0, "build": 0}

    def planned(*_args, **_kwargs):
        calls["plan"] += 1
        return copy.deepcopy(plan)

    monkeypatch.setattr(writer_release, "plan_stopped_release", planned)
    spec = writer_release._stopped_release_spec(REVISION)
    release = _release_binding(spec)
    monkeypatch.setattr(
        writer_release,
        "_validate_completed_release",
        lambda _spec: copy.deepcopy(release),
    )
    host_binding = {
        "host_identity_receipt_path": str(host_path),
        "host_identity_receipt_file_sha256": "d" * 64,
        "host_identity_receipt_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        writer_release,
        "_publish_or_validate_host_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(host_binding),
    )
    monkeypatch.setattr(
        writer_release,
        "_read_host_receipt_binding",
        lambda _plan: copy.deepcopy(host_binding),
    )

    def build(current, *, runner):
        del runner
        calls["build"] += 1
        current.release_root.mkdir(parents=True)
        return _manifest_with_digest(current, release["release_artifact_sha256"])

    receipt = writer_release.apply_stopped_release(
        REVISION,
        plan["plan_sha256"],
        release_builder=build,
        clock=lambda: 123.9,
    )

    receipt_path = Path(receipt["receipt_path"])
    raw = receipt_path.read_bytes()
    assert receipt["schema"] == writer_release.STOPPED_RELEASE_RECEIPT_SCHEMA
    assert receipt["services_stopped_and_disabled"] is True
    assert receipt["service_state_before"] == plan["service_states"]
    assert receipt["service_state_after"] == plan["service_states"]
    assert receipt["host_identity_receipt_file_sha256"] == "d" * 64
    assert receipt["receipt_sha256"] == writer_release._sha256_json({
        name: value for name, value in receipt.items() if name != "receipt_sha256"
    })
    assert raw == writer_release._canonical_bytes(receipt) + b"\n"
    assert stat.S_IMODE(os.lstat(receipt_path).st_mode) == 0o400
    before = os.lstat(receipt_path)
    host_path.parent.mkdir(parents=True)
    host_path.write_bytes(b"present")
    retry = writer_release.apply_stopped_release(
        REVISION,
        plan["plan_sha256"],
        release_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("builder reran")
        ),
        clock=lambda: (_ for _ in ()).throw(AssertionError("clock reran")),
    )
    after = os.lstat(receipt_path)
    assert retry == receipt
    assert calls == {"plan": 3, "build": 1}
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_post_build_drift_blocks_final_receipt_after_host_publication(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(writer_release, "DEFAULT_RELEASE_BASE", tmp_path / "releases")
    monkeypatch.setattr(writer_release, "DEFAULT_EVIDENCE_BASE", tmp_path / "evidence")
    monkeypatch.setattr(writer_release, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        writer_release, "_validate_evidence_namespace", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        writer_release, "_preflight_host_receipt_namespace", lambda _plan: None
    )
    first = _plan_for_apply(REVISION)
    second = copy.deepcopy(first)
    second["dedicated_host"]["boot_id_sha256"] = "f" * 64
    unsigned = {name: value for name, value in second.items() if name != "plan_sha256"}
    second["plan_sha256"] = writer_release._sha256_json(unsigned)
    plans = iter((first, second))
    monkeypatch.setattr(
        writer_release,
        "plan_stopped_release",
        lambda *_args, **_kwargs: next(plans),
    )
    spec = writer_release._stopped_release_spec(REVISION)
    release = _release_binding(spec)
    monkeypatch.setattr(
        writer_release,
        "_validate_completed_release",
        lambda _spec: copy.deepcopy(release),
    )
    published: list[bool] = []
    monkeypatch.setattr(
        writer_release,
        "_publish_or_validate_host_receipt",
        lambda *_args, **_kwargs: published.append(True),
    )

    def build(current, *, runner):
        del runner
        current.release_root.mkdir(parents=True)
        return _manifest_with_digest(current, release["release_artifact_sha256"])

    with pytest.raises(RuntimeError, match="drifted during build"):
        writer_release.apply_stopped_release(
            REVISION,
            first["plan_sha256"],
            release_builder=build,
        )

    assert spec.release_root.is_dir()
    assert published == [True]


def test_cli_apply_prints_publication_receipt_not_result_wrapper(monkeypatch, capsys):
    receipt = {
        "schema": writer_release.STOPPED_RELEASE_RECEIPT_SCHEMA,
        "ok": True,
        "release_revision": REVISION,
        "plan_sha256": "c" * 64,
        "services_stopped_and_disabled": True,
        "receipt_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        writer_release,
        "apply_stopped_release",
        lambda revision, digest: receipt,
    )

    assert (
        writer_release.main([
            "apply",
            "--revision",
            REVISION,
            "--approved-plan-sha256",
            "c" * 64,
        ])
        == 0
    )
    assert json.loads(capsys.readouterr().out) == receipt
