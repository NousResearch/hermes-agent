from __future__ import annotations

import copy
import hashlib
import io
import os
import traceback
import zipfile
from pathlib import Path

import pytest

from scripts.canary import owner_gate_stage0 as stage0


ROOT = Path(__file__).parents[3]
RUNTIME_LOCK = stage0.decode_runtime_lock(
    (ROOT / stage0.RUNTIME_LOCK_RELATIVE).read_bytes()
)
BOOTSTRAP_PIP = {
    key: RUNTIME_LOCK["bootstrap_pip"][key]
    for key in stage0.SIGNED_WHEEL_FIELDS
}


def _release_trust(public_key: bytes) -> dict:
    image_name = "debian-12-bookworm-v20260710"
    return {
        "schema": stage0.TRUST_SCHEMA,
        "approved_for_offline_install": True,
        "fork_repository": stage0.FORK_REPOSITORY,
        "release_revision": "a" * 40,
        "source_tree_oid": "b" * 40,
        "foundation_source_revision": "0" * 40,
        "foundation_source_tree_oid": "8" * 40,
        "package_inventory_sha256": "c" * 64,
        "boot_image_self_link": (
            f"projects/debian-cloud/global/images/{image_name}"
        ),
        "collector_public_key_ids": {
            "network": "1" * 64,
            "cloud": "2" * 64,
            "host": "3" * 64,
        },
        "credential_migration_envelope_sha256": "d" * 64,
        "direct_iam_identity_authority_sha256": "e" * 64,
        "pre_foundation_authority_sha256": "4" * 64,
        "foundation_apply_receipt_sha256": "5" * 64,
        "project_ancestry_evidence_sha256": "6" * 64,
        "project_ancestry_chain_sha256": "7" * 64,
        "resource_ancestor_chain": ["organizations/123456789012"],
        "interpreter_image": {
            "project": "debian-cloud",
            "image_name": image_name,
            "image_numeric_id": "1234567890123456789",
            "image_self_link": (
                "https://www.googleapis.com/compute/v1/projects/"
                f"debian-cloud/global/images/{image_name}"
            ),
            "python_version": stage0.PYTHON_VERSION,
            "interpreter_sha256": "f" * 64,
        },
        "release_attestation": {
            "purpose": stage0.ATTESTATION_PURPOSE,
            "attested_at_unix": 1_800_000_000,
        },
        "signer_key_id": hashlib.sha256(public_key).hexdigest(),
        "signature_ed25519_b64url": "fixture-signature",
    }


def test_stage0_trust_requires_complete_foundation_and_ancestry_chain() -> None:
    public_key = b"p" * 32
    checked = stage0._validate_trust(_release_trust(public_key), public_key)

    assert checked["pre_foundation_authority_sha256"] == "4" * 64
    assert checked["foundation_source_revision"] == "0" * 40
    assert checked["foundation_source_tree_oid"] == "8" * 40
    assert checked["foundation_apply_receipt_sha256"] == "5" * 64
    assert checked["project_ancestry_evidence_sha256"] == "6" * 64
    assert checked["project_ancestry_chain_sha256"] == "7" * 64
    assert checked["resource_ancestor_chain"] == [
        "organizations/123456789012"
    ]
    for field in (
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "project_ancestry_evidence_sha256",
        "project_ancestry_chain_sha256",
        "foundation_source_revision",
        "foundation_source_tree_oid",
    ):
        drifted = _release_trust(public_key)
        drifted[field] = "not-a-digest"
        with pytest.raises(
            stage0.OwnerGateStage0Error,
            match="owner_gate_stage0_trust_invalid",
        ):
            stage0._validate_trust(drifted, public_key)

    drifted = _release_trust(public_key)
    drifted["resource_ancestor_chain"] = ["projects/123456789012"]
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_trust_invalid",
    ):
        stage0._validate_trust(drifted, public_key)

    same_revision = _release_trust(public_key)
    same_revision["foundation_source_revision"] = same_revision[
        "release_revision"
    ]
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_trust_invalid",
    ):
        stage0._validate_trust(same_revision, public_key)


def _python_pair(tmp_path: Path) -> tuple[Path, Path]:
    target = tmp_path / "python3.11"
    target.write_bytes(b"exact Debian CPython interpreter test identity")
    target.chmod(0o755)
    launcher = tmp_path / "python3"
    launcher.symlink_to("python3.11")
    return launcher, target


def test_exact_python_interpreter_accepts_only_pinned_relative_symlink(
    tmp_path: Path,
) -> None:
    launcher, target = _python_pair(tmp_path)

    assert stage0._read_exact_python_interpreter(
        launcher,
        expected_uid=launcher.lstat().st_uid,
        expected_gid=launcher.lstat().st_gid,
        expected_link_mode=launcher.lstat().st_mode & 0o777,
    ) == target.read_bytes()


@pytest.mark.parametrize(
    "case",
    (
        "launcher_regular",
        "absolute_target",
        "wrong_target",
        "target_symlink",
        "target_hardlink",
        "target_mode",
        "wrong_uid",
        "wrong_gid",
    ),
)
def test_exact_python_interpreter_rejects_identity_drift(
    tmp_path: Path,
    case: str,
) -> None:
    launcher, target = _python_pair(tmp_path)
    expected_uid = os.getuid()
    expected_gid = os.getgid()
    if case == "launcher_regular":
        launcher.unlink()
        launcher.write_bytes(target.read_bytes())
        launcher.chmod(0o755)
    elif case == "absolute_target":
        launcher.unlink()
        launcher.symlink_to(target)
    elif case == "wrong_target":
        launcher.unlink()
        launcher.symlink_to("python3.10")
    elif case == "target_symlink":
        real = tmp_path / "python3.11-real"
        target.rename(real)
        target.symlink_to(real.name)
    elif case == "target_hardlink":
        os.link(target, tmp_path / "python3.11-second-link")
    elif case == "target_mode":
        target.chmod(0o555)
    elif case == "wrong_uid":
        expected_uid += 1
    else:
        expected_gid += 1

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_python_identity_invalid",
    ):
        stage0._read_exact_python_interpreter(
            launcher,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            expected_link_mode=launcher.lstat().st_mode & 0o777,
        )


def _manifest() -> dict:
    return {
        "bootstrap_pip": dict(BOOTSTRAP_PIP),
        "wheels": [
            {
                "filename": "demo-1.2.3-py3-none-any.whl",
                "project": "demo",
                "version": "1.2.3",
            },
            {
                "filename": "typing_extensions-4.16.0-py3-none-any.whl",
                "project": "typing-extensions",
                "version": "4.16.0",
            },
        ],
    }


def _wheel(
    path: Path,
    *,
    project: str,
    version: str,
    extra_member: str | None = None,
) -> bytes:
    normalized = project.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {project}\nVersion: {version}\n\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
        archive.writestr(f"{normalized}/__init__.py", "VALUE = 1\n")
        if extra_member is not None:
            archive.writestr(extra_member, "malicious\n")
    path.chmod(0o444)
    return path.read_bytes()


def _inventory(
    venv: Path,
    *,
    distributions: list[dict[str, str]] | None = None,
    sys_path: list[str] | None = None,
) -> bytes:
    value = {
        "base_prefix": "/usr",
        "distributions": distributions or [
            {"project": "demo", "version": "1.2.3"},
            {"project": "pip", "version": BOOTSTRAP_PIP["version"]},
            {"project": "typing-extensions", "version": "4.16.0"},
        ],
        "executable": str(venv / "bin/python"),
        "flags": {
            "ignore_environment": 1,
            "isolated": 1,
            "no_user_site": 1,
            "safe_path": True,
        },
        "prefix": str(venv),
        "sys_path": sys_path or stage0._expected_runtime_sys_path(venv),
    }
    return stage0.canonical_json_bytes(value) + b"\n"


def test_runtime_inventory_accepts_only_exact_wheels_and_bootstrap_pip(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "release/venv"

    value = stage0.validate_runtime_inventory(
        _inventory(venv),
        venv=venv,
        manifest=_manifest(),
    )

    assert value["distributions"] == [
        {"project": "demo", "version": "1.2.3"},
        {"project": "pip", "version": BOOTSTRAP_PIP["version"]},
        {"project": "typing-extensions", "version": "4.16.0"},
    ]


def test_os_release_requires_exact_descriptor_stable_debian_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "os-release"
    path.write_bytes(
        b'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\n'
        b"ID=debian\n"
        b'VERSION_ID="12"\n'
        b"HOME_URL=https://www.debian.org/\n"
    )
    path.chmod(0o644)
    parsed = stage0._read_exact_os_release(
        path,
        expected_uid=path.stat().st_uid,
        expected_gid=path.stat().st_gid,
    )
    assert parsed["ID"] == "debian"
    assert parsed["VERSION_ID"] == "12"

    path.write_bytes(b'ID=debian\nID=debian\nVERSION_ID="12"\n')
    with pytest.raises(stage0.OwnerGateStage0Error, match="stage0_os_invalid"):
        stage0._read_exact_os_release(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )
    path.write_bytes(b'ID=debianized\nVERSION_ID="12"\n')
    with pytest.raises(stage0.OwnerGateStage0Error, match="stage0_os_invalid"):
        stage0._read_exact_os_release(
            path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )


def test_executable_identity_binds_path_target_bytes_and_metadata(
    tmp_path: Path,
) -> None:
    target = tmp_path / "tool-real"
    target.write_bytes(b"#!/bin/sh\nexit 0\n")
    target.chmod(0o755)
    launcher = tmp_path / "tool"
    launcher.symlink_to(target.name)
    first = stage0._capture_executable_identity(
        launcher,
        expected_uid=launcher.lstat().st_uid,
        expected_gid=launcher.lstat().st_gid,
    )
    assert first["target_sha256"] == hashlib.sha256(
        target.read_bytes()
    ).hexdigest()

    target.write_bytes(b"#!/bin/sh\nexit 1\n")
    target.chmod(0o755)
    second = stage0._capture_executable_identity(
        launcher,
        expected_uid=launcher.lstat().st_uid,
        expected_gid=launcher.lstat().st_gid,
    )
    assert second != first

    target.chmod(0o775)
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="stage0_executable_identity_invalid",
    ):
        stage0._capture_executable_identity(
            launcher,
            expected_uid=launcher.lstat().st_uid,
            expected_gid=launcher.lstat().st_gid,
        )


@pytest.mark.parametrize(
    "distributions",
    [
        [
            {"project": "demo", "version": "1.2.3"},
            {"project": "pip", "version": BOOTSTRAP_PIP["version"]},
            {"project": "setuptools", "version": "66.1.1"},
            {"project": "typing-extensions", "version": "4.16.0"},
        ],
        [
            {"project": "demo", "version": "9.9.9"},
            {"project": "pip", "version": BOOTSTRAP_PIP["version"]},
            {"project": "typing-extensions", "version": "4.16.0"},
        ],
        [
            {"project": "demo", "version": "1.2.3"},
            {"project": "demo", "version": "1.2.3"},
            {"project": "pip", "version": BOOTSTRAP_PIP["version"]},
            {"project": "typing-extensions", "version": "4.16.0"},
        ],
        [
            {"project": "demo", "version": "1.2.3"},
            {"project": "typing-extensions", "version": "4.16.0"},
        ],
    ],
)
def test_runtime_inventory_rejects_extra_wrong_duplicate_or_missing_distribution(
    tmp_path: Path,
    distributions: list[dict[str, str]],
) -> None:
    venv = tmp_path / "release/venv"

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_runtime_inventory_invalid",
    ):
        stage0.validate_runtime_inventory(
            _inventory(venv, distributions=distributions),
            venv=venv,
            manifest=_manifest(),
        )


def test_runtime_inventory_rejects_ambient_sys_path(tmp_path: Path) -> None:
    venv = tmp_path / "release/venv"
    paths = stage0._expected_runtime_sys_path(venv)
    paths.insert(0, str(tmp_path / "ambient"))

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_runtime_inventory_invalid",
    ):
        stage0.validate_runtime_inventory(
            _inventory(venv, sys_path=paths),
            venv=venv,
            manifest=_manifest(),
        )


def test_manifest_distribution_inventory_rejects_normalized_duplicate() -> None:
    manifest = _manifest()
    manifest["wheels"].append({
        "filename": "typing.extensions-4.16.0-py3-none-any.whl",
        "project": "typing_extensions",
        "version": "4.16.0",
    })

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_distribution_invalid",
    ):
        stage0._expected_distribution_inventory(manifest)


def test_pip_commands_are_isolated_offline_and_use_closed_environment(
    tmp_path: Path,
) -> None:
    venv = tmp_path / "release/venv"
    bootstrap = tmp_path / BOOTSTRAP_PIP["filename"]
    install = stage0._pip_install_argv(
        venv,
        bootstrap,
        ("/wheelhouse/demo.whl",),
    )
    environment = stage0._pip_command_environment()

    assert install[:5] == (
        str(venv / "bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
    )
    assert str(bootstrap) in install
    assert "--no-index" in install
    assert "--no-deps" in install
    assert "--only-binary=:all:" in install
    assert "uninstall" not in install
    assert environment == {
        "HOME": "/nonexistent",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "PIP_CONFIG_FILE": "/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
    }


def test_runtime_inventory_flags_are_exact_not_subset(tmp_path: Path) -> None:
    venv = tmp_path / "release/venv"
    raw = stage0._load_canonical_json(_inventory(venv).rstrip(b"\n"))
    changed = copy.deepcopy(raw)
    changed["flags"]["ignore_environment"] = 0

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_runtime_inventory_invalid",
    ):
        stage0.validate_runtime_inventory(
            stage0.canonical_json_bytes(changed) + b"\n",
            venv=venv,
            manifest=_manifest(),
        )


@pytest.mark.parametrize(
    "member",
    (
        "demo.pth",
        "demo.data/purelib/nested/evil.pth",
        "demo.data/platlib/sitecustomize.py",
        "usercustomize/__init__.py",
        "sitecustomize.cpython-311-x86_64-linux-gnu.so",
        "demo.data/scripts/launcher",
        "demo.data/data/escape",
        "demo.data/headers/escape.h",
    ),
)
def test_stage0_wheel_parser_rejects_startup_and_external_members(
    tmp_path: Path,
    member: str,
) -> None:
    path = tmp_path / "demo-1.2.3-py3-none-any.whl"
    raw = _wheel(
        path,
        project="demo",
        version="1.2.3",
        extra_member=member,
    )

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_site_packages_invalid",
    ):
        stage0._wheel_expected_site_packages(raw)


def _install_wheel_tree(raw: bytes, site_packages: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            relative = stage0._wheel_site_packages_relative(entry.filename)
            if relative is None:
                continue
            path = site_packages / Path(relative.as_posix())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(entry))


def _installed_site_packages_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict]:
    root = tmp_path / "bundle"
    wheels = root / "wheels"
    wheels.mkdir(parents=True)
    demo = wheels / "demo-1.2.3-py3-none-any.whl"
    demo_raw = _wheel(demo, project="demo", version="1.2.3")
    pip_wheel = root / "bootstrap" / BOOTSTRAP_PIP["filename"]
    pip_wheel.parent.mkdir()
    pip_raw = _wheel(
        pip_wheel,
        project="pip",
        version=BOOTSTRAP_PIP["version"],
    )
    manifest = {
        "bootstrap_pip": {
            "filename": pip_wheel.name,
            "project": "pip",
            "version": BOOTSTRAP_PIP["version"],
            "sha256": hashlib.sha256(pip_raw).hexdigest(),
            "size": len(pip_raw),
        },
        "wheels": [{
            "filename": demo.name,
            "project": "demo",
            "version": "1.2.3",
            "sha256": hashlib.sha256(demo_raw).hexdigest(),
            "size": len(demo_raw),
        }],
    }
    venv = tmp_path / "venv"
    site_packages = venv / "lib/python3.11/site-packages"
    site_packages.mkdir(parents=True)
    site_packages.chmod(0o755)
    _install_wheel_tree(demo_raw, site_packages)
    _install_wheel_tree(pip_raw, site_packages)
    return root, venv, pip_wheel, manifest


def test_stage0_uses_only_signed_bundle_bootstrap_when_host_pip_is_tampered(
    tmp_path: Path,
) -> None:
    root, _venv, signed_pip, manifest = _installed_site_packages_fixture(
        tmp_path
    )
    host_pip = (
        tmp_path
        / "usr/share/python-wheels"
        / manifest["bootstrap_pip"]["filename"]
    )
    host_pip.parent.mkdir(parents=True)
    host_pip.write_bytes(b"tampered host pip wheel")
    host_pip.chmod(0o444)

    stage0.validate_wheel_archives_for_install(
        root,
        manifest=manifest,
        expected_uid=os.getuid(),
    )
    assert host_pip.read_bytes() == b"tampered host pip wheel"

    signed_pip.chmod(0o644)
    signed_pip.write_bytes(b"tampered signed pip wheel")
    signed_pip.chmod(0o444)
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_site_packages_invalid",
    ):
        stage0.validate_wheel_archives_for_install(
            root,
            manifest=manifest,
            expected_uid=os.getuid(),
        )


def test_stage0_scans_exact_installed_tree_without_venv_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, venv, pip_wheel, manifest = _installed_site_packages_fixture(tmp_path)
    invoked = False

    def forbidden(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("venv interpreter must not run during scan")

    monkeypatch.setattr(stage0.subprocess, "run", forbidden)
    site_state = (venv / "lib/python3.11/site-packages").stat()
    stage0.validate_installed_site_packages(
        root,
        venv=venv,
        manifest=manifest,
        expected_uid=site_state.st_uid,
        expected_gid=site_state.st_gid,
    )
    assert invoked is False

    injected = venv / "lib/python3.11/site-packages/injected.pth"
    injected.write_text("import attacker\n", encoding="utf-8")
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_site_packages_invalid",
    ):
        stage0.validate_installed_site_packages(
            root,
            venv=venv,
            manifest=manifest,
            expected_uid=site_state.st_uid,
            expected_gid=site_state.st_gid,
        )
    assert invoked is False


def test_stage0_purges_only_validated_generated_bytecode_without_venv_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "venv"
    site_packages = venv / "lib/python3.11/site-packages"
    cache = site_packages / "pip/_internal/__pycache__"
    cache.mkdir(parents=True)
    cache.chmod(0o755)
    for name in ("__init__.cpython-311.pyc", "main.cpython-311.pyc"):
        path = cache / name
        path.write_bytes(b"generated bytecode")
        path.chmod(0o644)
    invoked = False

    def forbidden(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("venv interpreter must not run during bytecode purge")

    monkeypatch.setattr(stage0.subprocess, "run", forbidden)
    site_state = site_packages.stat()
    stage0.purge_generated_site_packages_bytecode(
        venv,
        expected_uid=site_state.st_uid,
        expected_gid=site_state.st_gid,
    )

    assert not cache.exists()
    assert invoked is False


@pytest.mark.parametrize("malicious_kind", ("non_pyc", "symlink", "writable"))
def test_stage0_bytecode_purge_rejects_untrusted_cache_content(
    tmp_path: Path,
    malicious_kind: str,
) -> None:
    venv = tmp_path / "venv"
    site_packages = venv / "lib/python3.11/site-packages"
    cache = site_packages / "pip/__pycache__"
    cache.mkdir(parents=True)
    cache.chmod(0o755)
    valid = cache / "valid.cpython-311.pyc"
    valid.write_bytes(b"generated bytecode")
    valid.chmod(0o644)
    malicious = cache / "malicious.cpython-311.pyc"
    if malicious_kind == "non_pyc":
        malicious = cache / "unexpected.txt"
        malicious.write_bytes(b"not bytecode")
        malicious.chmod(0o644)
    elif malicious_kind == "symlink":
        target = tmp_path / "outside.pyc"
        target.write_bytes(b"outside")
        malicious.symlink_to(target)
    else:
        malicious.write_bytes(b"writable bytecode")
        malicious.chmod(0o666)
    site_state = site_packages.stat()

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_generated_bytecode_invalid",
    ):
        stage0.purge_generated_site_packages_bytecode(
            venv,
            expected_uid=site_state.st_uid,
            expected_gid=site_state.st_gid,
        )

    assert valid.is_file()
    assert os.path.lexists(malicious)


def test_venv_bin_is_sealed_to_exact_interpreter(tmp_path: Path) -> None:
    venv = tmp_path / "venv"
    bin_root = venv / "bin"
    (venv / "lib/python3.11/site-packages").mkdir(parents=True)
    bin_root.mkdir(parents=True)
    bin_root.chmod(0o755)
    interpreter = b"trusted copied interpreter"
    python = bin_root / "python"
    python.write_bytes(interpreter)
    python.chmod(0o755)
    for name in ("pip", "pip3", "uvicorn", "activate"):
        path = bin_root / name
        path.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        path.chmod(0o755)

    bin_state = bin_root.stat()
    stage0.seal_and_validate_venv_executables(
        venv,
        interpreter_sha256=hashlib.sha256(interpreter).hexdigest(),
        purge_generated_bin=True,
        expected_uid=bin_state.st_uid,
        expected_gid=bin_state.st_gid,
    )
    assert [item.name for item in bin_root.iterdir()] == ["python"]

    unexpected = bin_root / "console-script"
    unexpected.write_text("#!/bin/sh\n", encoding="utf-8")
    unexpected.chmod(0o755)
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_venv_executable_invalid",
    ):
        stage0.seal_and_validate_venv_executables(
            venv,
            interpreter_sha256=hashlib.sha256(interpreter).hexdigest(),
            purge_generated_bin=False,
            expected_uid=bin_state.st_uid,
            expected_gid=bin_state.st_gid,
        )


def test_stage0_runtime_lock_binds_file_digest_closure_and_artifacts(
    tmp_path: Path,
) -> None:
    raw = (
        Path(__file__).parents[3]
        / stage0.RUNTIME_LOCK_RELATIVE
    ).read_bytes()
    runtime_lock = stage0.decode_runtime_lock(raw)
    root = tmp_path / "bundle"
    lock_path = root / "payload" / stage0.RUNTIME_LOCK_RELATIVE
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(raw)
    lock_path.chmod(0o444)
    digest = hashlib.sha256(raw).hexdigest()
    wheels = [
        {
            key: item[key]
            for key in ("filename", "project", "version", "sha256", "size")
        }
        for item in runtime_lock["wheels"]
    ]
    manifest = {
        "bootstrap_pip": {
            key: runtime_lock["bootstrap_pip"][key]
            for key in stage0.SIGNED_WHEEL_FIELDS
        },
        "payloads": [{
            "release_relative": stage0.RUNTIME_LOCK_RELATIVE,
            "sha256": digest,
            "size": len(raw),
        }],
        "runtime_lock_sha256": digest,
        "wheels": wheels,
    }
    assert stage0._verify_runtime_lock_payload(
        root,
        manifest,
        expected_uid=os.getuid(),
    ) == runtime_lock

    changed = copy.deepcopy(manifest)
    changed["wheels"][0]["sha256"] = "0" * 64
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_runtime_lock_invalid",
    ):
        stage0._verify_runtime_lock_payload(
            root,
            changed,
            expected_uid=os.getuid(),
        )

    changed = copy.deepcopy(manifest)
    changed["bootstrap_pip"]["sha256"] = "0" * 64
    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_runtime_lock_invalid",
    ):
        stage0._verify_runtime_lock_payload(
            root,
            changed,
            expected_uid=os.getuid(),
        )


def test_offline_runtime_marker_binds_exact_bootstrap_artifact(
    tmp_path: Path,
) -> None:
    manifest = {
        "package_sha256": "a" * 64,
        "bootstrap_pip": dict(BOOTSTRAP_PIP),
        "wheels": [{
            "filename": "demo-1.2.3-py3-none-any.whl",
            "project": "demo",
            "version": "1.2.3",
            "sha256": "b" * 64,
            "size": 123,
        }],
    }
    venv = tmp_path / "release" / "venv"
    marker = stage0._load_canonical_json(
        stage0._offline_runtime_marker(manifest, venv=venv)
    )

    assert marker["bootstrap_pip"] == BOOTSTRAP_PIP
    assert marker["expected_distributions"] == {
        "demo": "1.2.3",
        "pip": BOOTSTRAP_PIP["version"],
    }

    changed = copy.deepcopy(manifest)
    changed["bootstrap_pip"]["sha256"] = "c" * 64
    assert stage0._offline_runtime_marker(changed, venv=venv) != (
        stage0._offline_runtime_marker(manifest, venv=venv)
    )


def test_stage0_formatted_traceback_suppresses_raw_cause() -> None:
    try:
        stage0.canonical_json_bytes({"unsafe": object()})
    except stage0.OwnerGateStage0Error as exc:
        rendered = "".join(traceback.format_exception(exc))
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected stable Stage0 error")

    assert "TypeError" not in rendered
    assert "not JSON serializable" not in rendered
    assert "The above exception was the direct cause" not in rendered
    assert "During handling of the above exception" not in rendered
    assert rendered.rstrip().endswith("owner_gate_stage0_json_invalid")


def test_verified_bundle_preflight_binds_bundle_without_installing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    manifest = {"release_revision": "a" * 40}
    preflight = {"schema": stage0.PREFLIGHT_SCHEMA}
    calls: list[tuple[object, ...]] = []

    def validate(
        value: object,
        *,
        bundle: Path,
        expected_bundle_uid: int,
    ) -> dict[str, str]:
        calls.append(("validate", value, bundle, expected_bundle_uid))
        return preflight

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("preflight must not install")

    monkeypatch.setattr(stage0, "validate_target_capabilities", validate)
    monkeypatch.setattr(stage0, "prepare_offline_runtime", forbidden)
    monkeypatch.setattr(stage0, "invoke_runtime_installer", forbidden)

    assert stage0.run_verified_bundle_operation(
        "preflight",
        bundle=bundle,
        manifest=manifest,
    ) is preflight
    assert calls == [("validate", manifest, bundle, 0)]


def test_verified_bundle_install_runs_only_after_bound_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    staged = tmp_path / "staged"
    manifest = {"release_revision": "a" * 40}
    receipt = {"schema": "owner-gate-install-receipt.v1"}
    calls: list[tuple[object, ...]] = []

    def validate(
        value: object,
        *,
        bundle: Path,
        expected_bundle_uid: int,
    ) -> dict[str, str]:
        calls.append(("validate", value, bundle, expected_bundle_uid))
        return {"schema": stage0.PREFLIGHT_SCHEMA}

    def prepare(path: Path, value: object) -> Path:
        calls.append(("prepare", path, value))
        return staged

    def invoke(path: Path, incoming: Path) -> dict[str, str]:
        calls.append(("invoke", path, incoming))
        return receipt

    monkeypatch.setattr(stage0, "validate_target_capabilities", validate)
    monkeypatch.setattr(stage0, "prepare_offline_runtime", prepare)
    monkeypatch.setattr(stage0, "invoke_runtime_installer", invoke)

    assert stage0.run_verified_bundle_operation(
        "install",
        bundle=bundle,
        manifest=manifest,
    ) is receipt
    assert calls == [
        ("validate", manifest, bundle, 0),
        ("prepare", bundle, manifest),
        ("invoke", staged, bundle),
    ]
