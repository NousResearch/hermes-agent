from __future__ import annotations

import hashlib
import os
import shutil
import stat
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
    source_snapshot_command,
    verify_clean_checkout,
    wheel_install_command,
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
    venv_command, lock_command, sync_command, build_command = install_commands(
        spec,
        managed,
    )
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
    assert "--no-install-project" in sync_command.argv
    assert sync_command.argv[sync_command.argv.index("--link-mode") + 1] == "copy"
    assert sync_command.argv[sync_command.argv.index("--python") + 1] == str(
        spec.interpreter
    )
    assert sync_command.environment()["UV_PROJECT_ENVIRONMENT"] == str(spec.venv_root)
    assert sync_command.argv[sync_command.argv.index("--project") + 1] == str(
        spec.build_project_root
    )
    assert str(spec.source_root) not in sync_command.argv
    assert build_command.argv[-1] == str(spec.build_project_root)
    assert "--no-create-gitignore" in build_command.argv
    assert "--force-pep517" in build_command.argv
    assert "--require-hashes" in build_command.argv
    assert build_command.argv[
        build_command.argv.index("--build-constraints") + 1
    ] == str(spec.build_constraints)
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


def _allow_local_materialization_owner(monkeypatch) -> None:
    monkeypatch.setattr(writer_release, "_BUILD_OWNER_UID", os.geteuid())
    monkeypatch.setattr(writer_release, "_BUILD_OWNER_GID", os.getegid())


def test_materializes_exact_managed_python_symlink_as_independent_copy(
    tmp_path,
    monkeypatch,
):
    _allow_local_materialization_owner(monkeypatch)
    spec = _spec(tmp_path)
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"exact-managed-python\n")
    managed.chmod(0o555)
    spec.interpreter.parent.mkdir(parents=True)
    spec.interpreter.symlink_to(managed)

    digest = writer_release._materialize_copied_interpreter(spec, managed)

    source_stat = os.lstat(managed)
    copied_stat = os.lstat(spec.interpreter)
    assert not spec.interpreter.is_symlink()
    assert stat.S_ISREG(copied_stat.st_mode)
    assert copied_stat.st_nlink == 1
    assert (copied_stat.st_dev, copied_stat.st_ino) != (
        source_stat.st_dev,
        source_stat.st_ino,
    )
    assert spec.interpreter.read_bytes() == managed.read_bytes()
    assert digest == hashlib.sha256(managed.read_bytes()).hexdigest()
    assert writer_release._materialize_copied_interpreter(spec, managed) == digest


def test_materializes_managed_python_alias_that_resolves_to_exact_source(
    tmp_path,
    monkeypatch,
):
    _allow_local_materialization_owner(monkeypatch)
    spec = _spec(tmp_path)
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"exact-managed-python\n")
    managed.chmod(0o555)
    managed_alias = managed.with_name("python")
    managed_alias.symlink_to(managed.name)
    spec.interpreter.parent.mkdir(parents=True)
    spec.interpreter.symlink_to(managed_alias)

    digest = writer_release._materialize_copied_interpreter(spec, managed)

    assert not spec.interpreter.is_symlink()
    assert spec.interpreter.read_bytes() == managed.read_bytes()
    assert digest == hashlib.sha256(managed.read_bytes()).hexdigest()


def test_interpreter_materialization_rejects_wrong_target_and_collision(
    tmp_path,
    monkeypatch,
):
    _allow_local_materialization_owner(monkeypatch)
    spec = _spec(tmp_path)
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"exact-managed-python\n")
    managed.chmod(0o555)
    spec.interpreter.parent.mkdir(parents=True)
    wrong = tmp_path / "wrong-python"
    wrong.write_bytes(managed.read_bytes())
    wrong.chmod(0o555)
    spec.interpreter.symlink_to(wrong)

    with pytest.raises(RuntimeError, match="symlink target"):
        writer_release._materialize_copied_interpreter(spec, managed)
    assert spec.interpreter.is_symlink()
    assert spec.interpreter.readlink() == wrong

    spec.interpreter.unlink()
    spec.interpreter.write_bytes(b"collision")
    spec.interpreter.chmod(0o555)
    with pytest.raises(RuntimeError, match="content is not managed Python"):
        writer_release._materialize_copied_interpreter(spec, managed)
    assert spec.interpreter.read_bytes() == b"collision"


def test_interpreter_materialization_rejects_hardlink(tmp_path, monkeypatch):
    _allow_local_materialization_owner(monkeypatch)
    spec = _spec(tmp_path)
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"exact-managed-python\n")
    managed.chmod(0o555)
    spec.interpreter.parent.mkdir(parents=True)
    os.link(managed, spec.interpreter)

    with pytest.raises(RuntimeError, match="independent root-owned executable"):
        writer_release._materialize_copied_interpreter(spec, managed)


def test_snapshot_and_install_paths_cannot_alias_source_release_or_cache(tmp_path):
    spec = _spec(tmp_path)
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"
    snapshot = source_snapshot_command(spec)

    assert snapshot.argv == (
        str(spec.git_executable),
        "-C",
        str(spec.source_root),
        "checkout-index",
        "--all",
        "--force",
        f"--prefix={spec.build_project_root}/",
    )
    assert spec.build_scratch_root.parent == spec.release_root
    assert spec.build_project_root.parent == spec.build_scratch_root
    assert spec.wheel_output_root.parent == spec.build_scratch_root
    assert spec.wheel_artifact_root.parent == spec.release_root
    assert spec.source_root not in spec.build_scratch_root.parents
    assert spec.uv_cache_dir not in spec.build_scratch_root.parents

    scratch_wheel = spec.wheel_output_root / "test-1-py3-none-any.whl"
    with pytest.raises(ValueError, match="artifact directory"):
        wheel_install_command(spec, scratch_wheel)
    with pytest.raises(ValueError, match="uv cache"):
        replace(spec, uv_cache_dir=spec.build_scratch_root).validate()
    with pytest.raises(ValueError, match="disjoint"):
        replace(spec, source_root=spec.build_project_root).validate()

    retained = spec.wheel_artifact_root / "test-1-py3-none-any.whl"
    install = wheel_install_command(spec, retained)
    assert install.argv[-1] == str(retained)
    assert "--no-deps" in install.argv
    assert "--no-build" in install.argv
    assert "--no-index" in install.argv
    assert "--no-cache" in install.argv
    assert install.argv[install.argv.index("--link-mode") + 1] == "copy"
    assert str(managed) not in install.argv


def test_build_constraints_are_exact_and_hash_pinned(tmp_path):
    constraints = (
        Path(writer_release.__file__).resolve().parents[2]
        / writer_release.BUILD_CONSTRAINTS_RELATIVE_PATH
    )
    expected = (
        "setuptools==81.0.0 "
        "--hash=sha256:"
        "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6\n"
    ).encode("ascii")

    assert constraints.read_bytes() == expected

    spec = _spec(tmp_path)
    spec.build_constraints.parent.mkdir(parents=True)
    spec.build_constraints.write_bytes(expected)
    real_lstat = writer_release.os.lstat

    def root_lstat(path, *args, **kwargs):
        item = real_lstat(path, *args, **kwargs)
        return SimpleNamespace(
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_nlink=item.st_nlink,
            st_size=item.st_size,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(writer_release.os, "lstat", root_lstat)
        writer_release._validate_build_constraints(spec)
        spec.build_constraints.write_bytes(expected.replace(b"81.0.0", b"81.0.1"))
        with pytest.raises(PermissionError, match="constraints"):
            writer_release._validate_build_constraints(spec)


def test_build_and_egg_info_writes_are_isolated_from_canonical_source(tmp_path):
    spec = _spec(tmp_path)
    _source(spec)
    before = {
        path.relative_to(spec.source_root): path.read_bytes()
        for path in spec.source_root.rglob("*")
        if path.is_file()
    }
    spec.build_project_root.mkdir(parents=True)
    spec.wheel_output_root.mkdir()
    (spec.build_project_root / "pyproject.toml").write_bytes(
        (spec.source_root / "pyproject.toml").read_bytes()
    )
    (spec.build_project_root / "uv.lock").write_bytes(
        (spec.source_root / "uv.lock").read_bytes()
    )
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"

    for command in install_commands(spec, managed):
        if command.argv[1] not in {"sync", "build"}:
            continue
        project = Path(
            command.argv[command.argv.index("--project") + 1]
            if "--project" in command.argv
            else command.argv[-1]
        )
        (project / "build").mkdir(exist_ok=True)
        (project / "test.egg-info").mkdir(exist_ok=True)
        if command.argv[1] == "build":
            (spec.wheel_output_root / "test-1-py3-none-any.whl").write_bytes(b"wheel")

    after = {
        path.relative_to(spec.source_root): path.read_bytes()
        for path in spec.source_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (spec.source_root / "build").exists()
    assert not (spec.source_root / "test.egg-info").exists()
    assert (spec.build_project_root / "build").is_dir()
    assert (spec.build_project_root / "test.egg-info").is_dir()


def test_real_uv_build_dirties_only_tracked_index_scratch(tmp_path):
    uv_raw = shutil.which("uv")
    git_raw = shutil.which("git")
    if uv_raw is None or git_raw is None:
        pytest.skip("real uv/git executables are unavailable")
    uv = Path(uv_raw).resolve()
    git = Path(git_raw).resolve()
    managed_result = subprocess.run(
        [
            str(uv),
            "python",
            "find",
            "3.11.15",
            "--managed-python",
            "--no-python-downloads",
            "--no-project",
            "--resolve-links",
            "--no-config",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if managed_result.returncode != 0:
        pytest.skip("the exact managed Python is unavailable for real build proof")

    source = tmp_path / "canonical-source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = ['setuptools>=77.0,<83']\n"
        "build-backend = 'setuptools.build_meta'\n"
        "[project]\n"
        "name = 'scratch-proof'\n"
        "version = '1.0.0'\n"
        "[tool.setuptools]\n"
        "py-modules = ['proof']\n",
        encoding="utf-8",
    )
    (source / "proof.py").write_text("PROOF = True\n", encoding="utf-8")
    constraints = source / writer_release.BUILD_CONSTRAINTS_RELATIVE_PATH
    constraints.parent.mkdir(parents=True)
    constraints.write_bytes(writer_release._PINNED_BUILD_CONSTRAINTS)
    subprocess.run([str(git), "init", "--quiet", str(source)], check=True)
    subprocess.run([str(git), "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            str(git),
            "-C",
            str(source),
            "-c",
            "user.name=Writer Release Test",
            "-c",
            "user.email=writer-release@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.run(
        [str(git), "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    spec = ReleaseBuildSpec(
        revision=revision,
        source_root=source,
        release_base=tmp_path / "releases",
        python_version="3.11.15",
        uv_executable=uv,
        git_executable=git,
        uv_cache_dir=tmp_path / "uv-cache",
    )
    spec.build_project_root.mkdir(parents=True)
    spec.wheel_output_root.mkdir()
    managed_real = Path(managed_result.stdout.strip())
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"
    managed.parent.mkdir(parents=True)
    managed.symlink_to(managed_real)

    snapshot = source_snapshot_command(spec)
    snapshot_result = subprocess.run(
        list(snapshot.argv),
        env=snapshot.environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert snapshot_result.returncode == 0, snapshot_result.stderr
    build = install_commands(spec, managed)[-1]
    built = subprocess.run(
        list(build.argv),
        env=build.environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stderr

    status = subprocess.run(
        [str(git), "-C", str(source), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
    assert not (source / "build").exists()
    assert not list(source.glob("*.egg-info"))
    assert (spec.build_project_root / "build").is_dir()
    assert list(spec.build_project_root.glob("*.egg-info"))
    wheels = list(spec.wheel_output_root.glob("*.whl"))
    assert len(wheels) == 1

    venv = subprocess.run(
        [
            str(managed_real),
            "-I",
            "-m",
            "venv",
            "--copies",
            str(spec.venv_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert venv.returncode == 0, venv.stderr
    spec.wheel_artifact_root.mkdir()
    retained = spec.wheel_artifact_root / wheels[0].name
    shutil.copyfile(wheels[0], retained)
    install = wheel_install_command(spec, retained)
    installed = subprocess.run(
        list(install.argv),
        env=install.environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert installed.returncode == 0, installed.stderr
    imported = subprocess.run(
        [str(spec.interpreter), "-I", "-c", "import proof; assert proof.PROOF"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr


def test_built_wheel_selection_rejects_missing_multiple_and_symlink(tmp_path):
    spec = _spec(tmp_path)
    spec.wheel_output_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="one exact wheel"):
        writer_release._select_built_wheel(spec)

    first = spec.wheel_output_root / "test-1-py3-none-any.whl"
    second = spec.wheel_output_root / "test-2-py3-none-any.whl"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    with pytest.raises(RuntimeError, match="one exact wheel"):
        writer_release._select_built_wheel(spec)

    second.unlink()
    first.unlink()
    target = tmp_path / "outside.whl"
    target.write_bytes(b"outside")
    first.symlink_to(target)
    with pytest.raises(RuntimeError, match="root-owned regular file"):
        writer_release._select_built_wheel(spec)


def test_exact_wheel_is_copied_into_manifest_bound_artifacts(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec.wheel_output_root.mkdir(parents=True)
    spec.wheel_artifact_root.mkdir()
    source = spec.wheel_output_root / "test-1-py3-none-any.whl"
    source.write_bytes(b"exact-wheel-content")
    real_lstat = writer_release.os.lstat

    def root_lstat(path, *args, **kwargs):
        item = real_lstat(path, *args, **kwargs)
        return SimpleNamespace(
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_nlink=item.st_nlink,
            st_size=item.st_size,
        )

    monkeypatch.setattr(writer_release.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(writer_release.os, "lstat", root_lstat)

    retained = writer_release._copy_built_wheel(spec, source)

    assert retained == spec.wheel_artifact_root / source.name
    assert retained.read_bytes() == source.read_bytes()
    assert retained.stat().st_ino != source.stat().st_ino
    assert wheel_install_command(spec, retained).argv[-1] == str(retained)


def test_scratch_provenance_mismatch_and_symlink_fail_closed(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec.source_root.mkdir(parents=True)
    spec.release_root.mkdir(parents=True)
    real_lstat = writer_release.os.lstat

    def root_lstat(path, *args, **kwargs):
        item = real_lstat(path, *args, **kwargs)
        return SimpleNamespace(
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_dev=item.st_dev,
            st_ino=item.st_ino,
            st_nlink=item.st_nlink,
            st_size=item.st_size,
        )

    monkeypatch.setattr(writer_release.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(writer_release.os, "lstat", root_lstat)
    device, inode = writer_release._prepare_build_scratch(spec)
    provenance = spec.build_scratch_root / writer_release.SCRATCH_PROVENANCE_NAME
    provenance.chmod(0o600)

    with pytest.raises(RuntimeError, match="provenance drifted"):
        writer_release._remove_build_scratch(
            spec,
            scratch_device=device,
            scratch_inode=inode,
        )
    assert spec.build_scratch_root.is_dir()

    provenance.chmod(0o400)
    outside = tmp_path / "outside"
    outside.mkdir()
    saved = spec.release_root / "saved-scratch"
    spec.build_scratch_root.rename(saved)
    spec.build_scratch_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="scratch identity drifted"):
        writer_release._remove_build_scratch(
            spec,
            scratch_device=device,
            scratch_inode=inode,
        )
    assert outside.is_dir()
    spec.build_scratch_root.unlink()
    saved.rename(spec.build_scratch_root)

    symlink = spec.build_scratch_root / "escape"
    symlink.symlink_to(outside, target_is_directory=True)
    writer_release._remove_build_scratch(
        spec,
        scratch_device=device,
        scratch_inode=inode,
    )
    assert not spec.build_scratch_root.exists()
    assert outside.is_dir()


def test_failed_build_retains_incomplete_release_and_scratch(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    _source(spec)
    spec.release_base.mkdir()
    spec.uv_cache_dir.mkdir()
    managed = spec.managed_python_root / "cpython-3.11.15/bin/python3.11"

    monkeypatch.setattr(writer_release, "_require_root_linux", lambda: None)
    monkeypatch.setattr(writer_release, "_validate_root_parent_chain", lambda _p: None)
    monkeypatch.setattr(writer_release, "_validate_root_executable", lambda _p: None)
    monkeypatch.setattr(writer_release, "_validate_root_source_tree", lambda _p: None)
    monkeypatch.setattr(
        writer_release,
        "_validate_root_directory",
        lambda _p, **_kwargs: None,
    )
    monkeypatch.setattr(
        writer_release,
        "_write_incomplete_marker",
        lambda current: (
            current.release_root / writer_release.INCOMPLETE_MARKER_NAME
        ).write_text("incomplete\n", encoding="utf-8"),
    )

    def prepare(current):
        current.build_project_root.mkdir(parents=True)
        current.wheel_output_root.mkdir()
        current.wheel_artifact_root.mkdir()
        item = os.lstat(current.build_scratch_root)
        return item.st_dev, item.st_ino

    monkeypatch.setattr(writer_release, "_prepare_build_scratch", prepare)
    observed = []

    def runner(command):
        observed.append(command.argv)
        if "rev-parse" in command.argv:
            stdout = f"{REVISION}\n"
        elif command.argv[1:3] == ("python", "find"):
            stdout = f"{managed}\n"
        else:
            stdout = ""
        if command.argv[1:3] == ("python", "install"):
            managed.parent.mkdir(parents=True)
            managed.write_bytes(b"python")
            managed.chmod(0o755)
        if "checkout-index" in command.argv:
            return subprocess.CompletedProcess(command.argv, 23, "", "snapshot failed")
        return subprocess.CompletedProcess(command.argv, 0, stdout, "")

    with pytest.raises(RuntimeError, match="tracked source snapshot failed"):
        build_release(spec, runner=runner)

    assert (spec.release_root / writer_release.INCOMPLETE_MARKER_NAME).is_file()
    assert spec.build_scratch_root.is_dir()
    assert not (spec.release_root / writer_release.RELEASE_MANIFEST_NAME).exists()
    assert sum("rev-parse" in argv for argv in observed) == 2


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
        "exporter_unit": "muncho-canonical-writer-export.service",
        "projection_export_path": (
            "/var/lib/muncho-canonical-writer/projection/canonical-events.json"
        ),
        "projection_export_limit": "200000",
    }
    assert bundle.schema == "muncho-writer-only-systemd-bundle.v2"
    assert "Type=oneshot" in bundle.exporter_service
    assert "[Install]" not in bundle.exporter_service
    assert ".timer" not in bundle.exporter_service
    assert "SupplementaryGroups=muncho-projector" in bundle.exporter_service
    assert "CapabilityBoundingSet=" in bundle.exporter_service
    assert "NoNewPrivileges=yes" in bundle.exporter_service
    assert "IPAddressDeny=any" in bundle.exporter_service
    assert "IPAddressAllow=10.20.30.40/32" in bundle.exporter_service
    assert (
        "ReadWritePaths=/var/lib/muncho-canonical-writer/projection"
        in bundle.exporter_service
    )
    assert "ReadWritePaths=/run/muncho-canonical-writer" not in (
        bundle.exporter_service
    )
    assert "StandardOutput=journal" in bundle.exporter_service
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
