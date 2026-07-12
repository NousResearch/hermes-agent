from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.canary import writer_release
from scripts.canary.writer_release import (
    GATEWAY_MODULE,
    RELEASE_SCHEMA,
    WRITER_MODULE,
    ReleaseBuildSpec,
    ReleaseManifest,
    WriterOnlyUnitSpec,
    build_release,
    checkout_commands,
    collect_tree_entries,
    create_release_manifest,
    install_commands,
    python_bootstrap_commands,
    render_systemd_units,
    verify_clean_checkout,
)


REVISION = "a" * 40
UNIT_SPEC = WriterOnlyUnitSpec(database_ip_allow=("10.20.30.40/32",))


def _spec(tmp_path: Path) -> ReleaseBuildSpec:
    return ReleaseBuildSpec(
        revision=REVISION,
        source_root=tmp_path / "source",
        release_base=tmp_path / "releases",
        python_version="3.11.15",
        uv_executable=tmp_path / "bin" / "uv",
        git_executable=tmp_path / "bin" / "git",
        uv_cache_dir=tmp_path / "uv-cache",
    )


def _source(spec: ReleaseBuildSpec) -> None:
    spec.source_root.mkdir(parents=True)
    (spec.source_root / "pyproject.toml").write_text(
        "[project]\nname='test'\nversion='1'\n",
        encoding="utf-8",
    )
    (spec.source_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def _manifest() -> ReleaseManifest:
    root = Path("/opt/muncho-canary-releases") / REVISION
    site = root / "venv/lib/python3.11/site-packages/gateway"
    provisional = ReleaseManifest(
        revision=REVISION,
        artifact_root=str(root),
        python_version="3.11.15",
        interpreter=str(root / "venv/bin/python"),
        writer_module_origin=str(site / "canonical_writer_bootstrap.py"),
        gateway_module_origin=str(site / "canonical_writer_gateway_bootstrap.py"),
        entries=(),
        artifact_sha256="",
    )
    return replace(
        provisional,
        artifact_sha256=provisional.computed_artifact_sha256,
    )


def test_release_commands_pin_managed_copied_frozen_noneditable_install(tmp_path):
    spec = _spec(tmp_path)
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"

    commands = (
        *checkout_commands(spec),
        *python_bootstrap_commands(spec),
        *install_commands(spec, managed),
    )

    assert python_bootstrap_commands(spec)[0].argv == (
        str(spec.uv_executable),
        "python",
        "install",
        "3.11.15",
        "--install-dir",
        str(spec.managed_python_root),
        "--no-bin",
        "--managed-python",
        "--no-config",
    )
    venv_command, lock_command, sync_command = install_commands(spec, managed)
    assert venv_command.argv == (
        str(managed),
        "-I",
        "-m",
        "venv",
        "--copies",
        str(spec.venv_root),
    )
    assert "--check" in lock_command.argv
    assert "--managed-python" in lock_command.argv
    assert "--frozen" in sync_command.argv
    assert "--no-editable" in sync_command.argv
    assert "--no-dev" in sync_command.argv
    assert sync_command.argv[sync_command.argv.index("--link-mode") + 1] == "copy"
    assert sync_command.argv[sync_command.argv.index("--python") + 1] == str(
        spec.interpreter
    )
    assert sync_command.environment()["UV_PROJECT_ENVIRONMENT"] == str(
        spec.venv_root
    )
    assert all(
        command.argv[0] not in {"sh", "bash", "/bin/sh", "/bin/bash"}
        for command in commands
    )
    assert all(
        not any(
            marker in name.casefold()
            for marker in ("token", "secret", "discord", "openai", "api_key")
        )
        for command in commands
        for name in command.environment()
    )


@pytest.mark.parametrize(
    "revision",
    ["A" * 40, "a" * 39, "a" * 41, "main", "0x" + "a" * 38],
)
def test_release_spec_rejects_nonexact_revision(tmp_path, revision):
    with pytest.raises(ValueError, match="revision"):
        replace(_spec(tmp_path), revision=revision).validate()


def test_clean_checkout_is_bound_to_head_and_untracked_inventory(tmp_path):
    spec = _spec(tmp_path)
    _source(spec)
    observed = []

    def clean_runner(command):
        observed.append(command.argv)
        stdout = f"{REVISION}\n" if "rev-parse" in command.argv else ""
        return subprocess.CompletedProcess(command.argv, 0, stdout=stdout, stderr="")

    verify_clean_checkout(spec, runner=clean_runner)

    assert observed == [command.argv for command in checkout_commands(spec)]

    def dirty_runner(command):
        stdout = f"{REVISION}\n" if "rev-parse" in command.argv else "?? rogue.py\n"
        return subprocess.CompletedProcess(command.argv, 0, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match="not clean"):
        verify_clean_checkout(spec, runner=dirty_runner)

    def ignored_runner(command):
        if "rev-parse" in command.argv:
            stdout = f"{REVISION}\n"
        elif "ls-files" in command.argv:
            stdout = "gateway/assets/private.pem\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(command.argv, 0, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match="ignored build inputs"):
        verify_clean_checkout(spec, runner=ignored_runner)


def test_root_gate_precedes_any_build_runner(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    called = []
    monkeypatch.setattr(writer_release, "_effective_uid", lambda: 1000)

    with pytest.raises(PermissionError, match="uid_0"):
        build_release(spec, runner=lambda command: called.append(command))

    assert called == []
    assert not spec.release_root.exists()


def test_root_executable_rejects_writable_parent_chain(monkeypatch):
    executable = Path("/usr/local/bin/uv")

    def fake_lstat(path):
        path = Path(path)
        if path == executable:
            return SimpleNamespace(
                st_mode=0o100755,
                st_uid=0,
                st_gid=0,
            )
        mode = 0o040777 if path == Path("/usr/local") else 0o040755
        return SimpleNamespace(st_mode=mode, st_uid=0, st_gid=0)

    monkeypatch.setattr(writer_release.os, "lstat", fake_lstat)

    with pytest.raises(PermissionError, match="root-controlled"):
        writer_release._validate_root_executable(executable)


def test_tree_manifest_is_canonical_and_binds_installed_module_origins(tmp_path):
    spec = _spec(tmp_path)
    spec.interpreter.parent.mkdir(parents=True)
    spec.writer_module_origin.parent.mkdir(parents=True)
    spec.interpreter.write_bytes(b"copied-python")
    spec.interpreter.chmod(0o555)
    spec.writer_module_origin.write_text("WRITER = True\n", encoding="utf-8")
    spec.gateway_module_origin.write_text("GATEWAY = True\n", encoding="utf-8")
    managed = spec.managed_python_root / "runtime/libpython.so"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"managed-runtime")
    (spec.release_root / "venv/lib64").symlink_to("lib")
    (spec.release_root / writer_release.INCOMPLETE_MARKER_NAME).write_text(
        "incomplete\n",
        encoding="utf-8",
    )

    first = create_release_manifest(spec)
    second = create_release_manifest(spec)

    assert first == second
    assert first.schema == RELEASE_SCHEMA
    assert first.writer_module == WRITER_MODULE
    assert first.gateway_module == GATEWAY_MODULE
    assert first.writer_module_origin == str(spec.writer_module_origin)
    assert first.gateway_module_origin == str(spec.gateway_module_origin)
    assert len(first.artifact_sha256) == 64
    assert [entry.path for entry in first.entries] == sorted(
        entry.path for entry in first.entries
    )
    assert writer_release.INCOMPLETE_MARKER_NAME not in {
        entry.path for entry in first.entries
    }

    spec.gateway_module_origin.write_text("GATEWAY = False\n", encoding="utf-8")
    changed = create_release_manifest(spec)
    assert changed.artifact_sha256 != first.artifact_sha256


def test_tree_manifest_rejects_external_symlink_and_special_file(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    (root / "escape").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes"):
        collect_tree_entries(root)

    (root / "escape").unlink()
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="unsupported"):
        collect_tree_entries(root)


def test_installed_runtime_requires_copied_venv_and_no_dynamic_site_path(tmp_path):
    spec = _spec(tmp_path)
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"managed")
    managed.chmod(0o555)
    spec.interpreter.parent.mkdir(parents=True)
    spec.interpreter.write_bytes(b"copied")
    spec.interpreter.chmod(0o555)
    spec.site_packages.mkdir(parents=True)
    (spec.venv_root / "pyvenv.cfg").write_text(
        "home = " + str(managed.parent) + "\n"
        "include-system-site-packages = false\n"
        "executable = " + str(managed) + "\n",
        encoding="utf-8",
    )

    writer_release._validate_installed_runtime(spec, managed)

    (spec.site_packages / "injected.pth").write_text(
        str(spec.source_root) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="dynamic site path"):
        writer_release._validate_installed_runtime(spec, managed)


def test_hardened_writer_only_units_pin_identity_config_and_readiness():
    manifest = _manifest()

    bundle = render_systemd_units(manifest, UNIT_SPEC)
    repeated = render_systemd_units(manifest, UNIT_SPEC)

    assert bundle == repeated
    assert bundle.writer_service.count("Type=notify") == 1
    assert bundle.writer_service.count("NotifyAccess=main") == 1
    assert bundle.gateway_service.count("Type=notify") == 1
    assert bundle.gateway_service.count("NotifyAccess=main") == 1
    assert (
        "ExecStart=/opt/muncho-canary-releases/"
        f"{REVISION}/venv/bin/python -I -m {WRITER_MODULE} --config "
        "/etc/muncho-canonical-writer/writer.json"
    ) in bundle.writer_service
    assert (
        "ExecStart=/opt/muncho-canary-releases/"
        f"{REVISION}/venv/bin/python -I -m {GATEWAY_MODULE}"
    ) in bundle.gateway_service
    assert "--require-canonical-writer" not in bundle.gateway_service
    assert "--writer-only-canary" not in bundle.gateway_service
    assert "WorkingDirectory=/opt/muncho-canary-releases/" + REVISION in (
        bundle.writer_service
    )
    assert "# PasswdHome=/var/lib/hermes-gateway" in bundle.gateway_service
    assert "# ManagedConfig=/etc/hermes/config.yaml" in bundle.gateway_service
    assert "ReadOnlyPaths=/etc/hermes/config.yaml" in bundle.gateway_service
    assert (
        "BindReadOnlyPaths=/opt/muncho-canary-releases/" + REVISION
        in bundle.writer_service
    )
    assert (
        "BindReadOnlyPaths=/opt/muncho-canary-releases/" + REVISION
        in bundle.gateway_service
    )
    assert "PrivateNetwork=yes" in bundle.gateway_service
    assert "IPAddressDeny=any" in bundle.writer_service
    assert "IPAddressAllow=10.20.30.40/32" in bundle.writer_service
    assert "Environment=HOME=/nonexistent" in bundle.writer_service
    assert "Environment=USER=muncho-canonical-writer" in bundle.writer_service
    assert "Environment=HOME=/var/lib/hermes-gateway" in bundle.gateway_service
    assert "Environment=USER=muncho-gateway" in bundle.gateway_service
    assert "ReadWritePaths=/run/hermes-cloud-gateway" in bundle.gateway_service
    assert "ReadWritePaths=/var/lib/hermes-gateway" not in bundle.gateway_service
    assert "ReadWritePaths=/var/log/hermes-gateway" not in bundle.gateway_service
    for rendered in (bundle.writer_service, bundle.gateway_service):
        assert "Environment=PATH=/usr/bin:/bin" in rendered
        assert "Environment=LANG=C.UTF-8" in rendered
        assert "Environment=LC_ALL=C.UTF-8" in rendered
        assert "Environment=SHELL=/usr/sbin/nologin" in rendered
        assert "Environment=TZ=UTC" in rendered
    assert "RestrictAddressFamilies=AF_UNIX\n" in bundle.gateway_service
    assert "BindsTo=muncho-canonical-writer.service" in bundle.gateway_service
    assert "AssertPathExists=/etc/hermes/config.yaml" in bundle.gateway_service
    assert "AssertPathIsDirectory=/run/muncho-canonical-writer" in (
        bundle.writer_service
    )
    assert "ConditionPathIsDirectory=" not in bundle.writer_service
    assert "RuntimeDirectory=muncho-canonical-writer" not in bundle.writer_service
    assert (
        "d /run/muncho-canonical-writer 2750 muncho-canonical-writer "
        "muncho-writer-client - -"
    ) in bundle.tmpfiles
    assert dict(bundle.contract) == {
        "revision": REVISION,
        "artifact_sha256": manifest.artifact_sha256,
        "working_directory": "/opt/muncho-canary-releases/" + REVISION,
        "writer_user": "muncho-canonical-writer",
        "writer_group": "muncho-canonical-writer",
        "gateway_user": "muncho-gateway",
        "gateway_group": "muncho-gateway",
        "gateway_passwd_home": "/var/lib/hermes-gateway",
        "gateway_config": "/etc/hermes/config.yaml",
        "socket_client_group": "muncho-writer-client",
        "writer_runtime": "/run/muncho-canonical-writer",
        "writer_runtime_mode": "2750",
        "database_ip_allow": "10.20.30.40/32",
    }
    for rendered in (bundle.writer_service, bundle.gateway_service):
        assert "EnvironmentFile=" not in rendered
        assert "PassEnvironment=" not in rendered
        assert "LoadCredential=" not in rendered
        assert "/bin/sh" not in rendered
        assert "discord" not in rendered.casefold()
        assert "openai" not in rendered.casefold()
        assert "api_key" not in rendered.casefold()
        assert "token" not in rendered.casefold()


def test_unit_spec_rejects_config_inside_mutable_gateway_home():
    with pytest.raises(ValueError, match="managed config"):
        replace(
            UNIT_SPEC,
            gateway_config=Path("/var/lib/hermes-gateway/.hermes/config.yaml"),
        ).validate()


def test_unit_spec_requires_one_exact_database_host_route():
    with pytest.raises(ValueError, match="one exact database IP"):
        WriterOnlyUnitSpec().validate()
    with pytest.raises(ValueError, match="exact host"):
        WriterOnlyUnitSpec(database_ip_allow=("10.20.30.0/24",)).validate()


def test_renderer_rejects_module_origin_outside_release():
    bad = replace(
        _manifest(),
        gateway_module_origin="/tmp/gateway/canonical_writer_gateway_bootstrap.py",
        artifact_sha256="",
    )
    bad = replace(bad, artifact_sha256=bad.computed_artifact_sha256)
    with pytest.raises(ValueError, match="module origins"):
        render_systemd_units(bad, UNIT_SPEC)
