#!/usr/bin/env python3
"""Build the immutable, release-local production owner Python runtime.

The builder consumes one clean exact Git revision, installs a managed Python
inside the revision-addressed release, creates a copied virtual environment,
installs frozen dependencies plus one non-editable project wheel, removes
dynamic site-path hooks, seals the complete tree, and asks the installed
stdlib-only runtime gate to author its own whole-tree manifest.  A final
``-I -B`` attestation imports cryptography and every production launcher
boundary before this builder reports readiness.

No Cloud, database, Discord, service, or production-host mutation is part of
this local packaging operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway import production_owner_runtime as runtime


PYTHON_VERSION = "3.11.15"
RECEIPT_SCHEMA = "muncho-production-owner-runtime-publication.v1"
DEFAULT_RELEASE_BASE = (
    Path(pwd.getpwuid(os.getuid()).pw_dir)  # windows-footgun: ok — macOS/Linux release-builder boundary
    / ".hermes/trusted/production-owner-runtime"
)
DEFAULT_UV = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".local/bin/uv"  # windows-footgun: ok — macOS/Linux release-builder boundary
DEFAULT_GIT = Path("/usr/bin/git")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SAFE_WHEEL = re.compile(r"^[A-Za-z0-9_.+-]+\.whl$")
_MAX_OUTPUT = 64 * 1024 * 1024
_COMMAND_TIMEOUT = 1800
_PUBLICATION_FIELDS = frozenset({
    "schema",
    "release_revision",
    "release_root",
    "manifest_sha256",
    "attestation_sha256",
    "interpreter_sha256",
    "pyvenv_cfg_sha256",
    "wheel_sha256",
    "runtime_reused",
    "non_editable_install",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})


class ProductionOwnerRuntimePackagingError(RuntimeError):
    """Stable, secret-free owner-runtime packaging failure."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_json_invalid"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_file_unavailable"
        ) from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class OwnerRuntimeBuildSpec:
    revision: str
    source_root: Path
    release_base: Path = DEFAULT_RELEASE_BASE
    uv_executable: Path = DEFAULT_UV
    git_executable: Path = DEFAULT_GIT

    @property
    def release_root(self) -> Path:
        return self.release_base / self.revision

    @property
    def python_root(self) -> Path:
        return self.release_root / "python"

    @property
    def venv_root(self) -> Path:
        return self.release_root / "venv"

    @property
    def interpreter(self) -> Path:
        return self.venv_root / "bin/python"

    @property
    def scratch_root(self) -> Path:
        return self.release_root / ".build-scratch"

    @property
    def snapshot_root(self) -> Path:
        return self.scratch_root / "source"

    @property
    def wheel_root(self) -> Path:
        return self.scratch_root / "wheel"

    @property
    def artifact_root(self) -> Path:
        return self.release_root / "artifacts"

    @property
    def site_packages(self) -> Path:
        return self.venv_root / "lib/python3.11/site-packages"

    @property
    def incomplete_marker(self) -> Path:
        return self.release_root / ".owner-runtime-build-incomplete"

    def validate(self) -> None:
        paths = (
            self.source_root,
            self.release_base,
            self.uv_executable,
            self.git_executable,
        )
        if (
            _REVISION.fullmatch(self.revision or "") is None
            or any(not path.is_absolute() or ".." in path.parts for path in paths)
            or self.release_root.parent != self.release_base
            or self.release_root.name != self.revision
            or self.source_root == self.release_root
            or self.source_root in self.release_root.parents
            or self.release_root in self.source_root.parents
        ):
            raise ProductionOwnerRuntimePackagingError(
                "production_owner_runtime_package_spec_invalid"
            )


def _clean_environment(spec: OwnerRuntimeBuildSpec) -> dict[str, str]:
    home = pwd.getpwuid(os.getuid()).pw_dir  # windows-footgun: ok — macOS/Linux release-builder boundary
    return {
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "UV_CACHE_DIR": str(Path(home) / ".cache/uv"),
        "UV_NO_CONFIG": "1",
    }


def _run(
    argv: Sequence[str],
    *,
    spec: OwnerRuntimeBuildSpec,
    cwd: Path | None = None,
    extra_environment: Mapping[str, str] | None = None,
    timeout: int = _COMMAND_TIMEOUT,
) -> bytes:
    if (
        not argv
        or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_argv_invalid"
        )
    environment = _clean_environment(spec)
    environment.update(dict(extra_environment or {}))
    if any(name.startswith("PYTHON") for name in environment):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_environment_invalid"
        )
    try:
        completed = subprocess.run(
            tuple(argv),
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_command_failed"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(completed.stdout, bytes)
        or not isinstance(completed.stderr, bytes)
        or len(completed.stdout) > _MAX_OUTPUT
        or len(completed.stderr) > _MAX_OUTPUT
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_command_failed"
        )
    return completed.stdout


def _validate_tool(path: Path) -> None:
    try:
        item = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_tool_unavailable"
        ) from exc
    if (
        not stat.S_ISREG(item.st_mode)
        or item.st_uid not in {0, os.getuid()}  # windows-footgun: ok — macOS/Linux release-builder boundary
        or stat.S_IMODE(item.st_mode) & 0o022
        or not stat.S_IMODE(item.st_mode) & 0o111
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_tool_invalid"
        )


def _verify_clean_source(spec: OwnerRuntimeBuildSpec) -> None:
    spec.validate()
    _validate_tool(spec.git_executable)
    _validate_tool(spec.uv_executable)
    try:
        state = os.lstat(spec.source_root)
    except OSError as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_source_unavailable"
        ) from exc
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_source_invalid"
        )
    head = _run(
        (
            str(spec.git_executable),
            "-C",
            str(spec.source_root),
            "rev-parse",
            "--verify",
            "HEAD",
        ),
        spec=spec,
    ).decode("ascii", errors="strict").strip()
    status = _run(
        (
            str(spec.git_executable),
            "-C",
            str(spec.source_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        spec=spec,
    )
    if head != spec.revision or status:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_source_not_exact"
        )


def build_commands(
    spec: OwnerRuntimeBuildSpec,
    *,
    managed_python: Path,
) -> tuple[tuple[str, ...], ...]:
    """Render the fixed dependency/project build argv for review and tests."""

    spec.validate()
    constraints = spec.snapshot_root / "scripts/canary/writer-build-constraints.txt"
    return (
        (
            str(managed_python),
            "-I",
            "-B",
            "-m",
            "venv",
            "--copies",
            str(spec.venv_root),
        ),
        (
            str(spec.uv_executable),
            "sync",
            "--frozen",
            "--no-editable",
            "--no-dev",
            "--no-install-project",
            "--link-mode",
            "copy",
            "--python",
            str(spec.interpreter),
            "--no-python-downloads",
            "--project",
            str(spec.snapshot_root),
            "--no-config",
        ),
        (
            str(spec.uv_executable),
            "build",
            "--wheel",
            "--out-dir",
            str(spec.wheel_root),
            "--no-create-gitignore",
            "--python",
            str(managed_python),
            "--managed-python",
            "--no-python-downloads",
            "--force-pep517",
            "--build-constraints",
            str(constraints),
            "--require-hashes",
            "--no-config",
            str(spec.snapshot_root),
        ),
    )


def _materialize_interpreter(spec: OwnerRuntimeBuildSpec, managed: Path) -> None:
    destination = spec.interpreter
    try:
        source = managed.resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(spec.python_root):
            raise ProductionOwnerRuntimePackagingError(
                "production_owner_runtime_package_python_invalid"
            )
        destination_state = os.lstat(destination)
    except ProductionOwnerRuntimePackagingError:
        raise
    except (OSError, ValueError) as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_python_invalid"
        ) from exc
    if stat.S_ISLNK(destination_state.st_mode):
        destination.unlink()
        shutil.copyfile(source, destination)
        destination.chmod(0o755)
    elif not stat.S_ISREG(destination_state.st_mode):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_python_invalid"
        )
    current = os.lstat(destination)
    source_state = os.lstat(source)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or (current.st_dev, current.st_ino)
        == (source_state.st_dev, source_state.st_ino)
        or _sha256(destination) != _sha256(source)
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_python_invalid"
        )


def _remove_dynamic_site_hooks(spec: OwnerRuntimeBuildSpec) -> None:
    for path in sorted(spec.site_packages.glob("*.pth")):
        try:
            item = os.lstat(path)
            raw = path.read_bytes()
        except OSError as exc:
            raise ProductionOwnerRuntimePackagingError(
                "production_owner_runtime_package_site_hook_invalid"
            ) from exc
        if (
            not stat.S_ISREG(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_nlink != 1
            or path.name != "_virtualenv.pth"
            or raw != b"import _virtualenv"
        ):
            raise ProductionOwnerRuntimePackagingError(
                "production_owner_runtime_package_site_hook_invalid"
            )
        path.unlink()
    if list(spec.site_packages.glob("*.pth")) or list(
        spec.site_packages.glob("*.egg-link")
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_dynamic_site_path"
        )


def _seal_tree(root: Path) -> None:
    for current, directories, files in os.walk(
        root,
        topdown=False,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            item = os.lstat(path)
            if stat.S_ISLNK(item.st_mode):
                continue
            if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
                raise ProductionOwnerRuntimePackagingError(
                    "production_owner_runtime_package_tree_invalid"
                )
            path.chmod(0o555 if stat.S_IMODE(item.st_mode) & 0o111 else 0o444)
        for name in directories:
            path = current_path / name
            item = os.lstat(path)
            if stat.S_ISLNK(item.st_mode):
                continue
            if not stat.S_ISDIR(item.st_mode):
                raise ProductionOwnerRuntimePackagingError(
                    "production_owner_runtime_package_tree_invalid"
                )
            path.chmod(0o555)
    root.chmod(0o555)


def _decode_line(raw: bytes) -> Mapping[str, Any]:
    if not raw.endswith(b"\n") or b"\n" in raw[:-1]:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_output_invalid"
        )
    try:
        value = json.loads(raw[:-1].decode("ascii", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_output_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical(value) + b"\n":
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_output_invalid"
        )
    return value


def validate_publication_receipt(
    value: Mapping[str, Any],
    *,
    spec: OwnerRuntimeBuildSpec,
) -> Mapping[str, Any]:
    unsigned = {
        name: item for name, item in value.items() if name != "receipt_sha256"
    } if isinstance(value, Mapping) else {}
    if (
        not isinstance(value, Mapping)
        or set(value) != _PUBLICATION_FIELDS
        or value.get("schema") != RECEIPT_SCHEMA
        or value.get("release_revision") != spec.revision
        or value.get("release_root") != str(spec.release_root)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value.get(name))) is None
            for name in (
                "manifest_sha256",
                "attestation_sha256",
                "interpreter_sha256",
                "pyvenv_cfg_sha256",
                "wheel_sha256",
                "receipt_sha256",
            )
        )
        or type(value.get("runtime_reused")) is not bool
        or value.get("non_editable_install") is not True
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256")
        != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_receipt_invalid"
        )
    return dict(value)


def _retained_wheel(spec: OwnerRuntimeBuildSpec) -> Path:
    try:
        wheels = sorted(spec.artifact_root.iterdir())
    except OSError as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_wheel_invalid"
        ) from exc
    if (
        len(wheels) != 1
        or _SAFE_WHEEL.fullmatch(wheels[0].name) is None
        or not wheels[0].is_file()
        or wheels[0].is_symlink()
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_wheel_invalid"
        )
    return wheels[0]


def runtime_command(
    spec: OwnerRuntimeBuildSpec,
    action: str,
    *launcher_args: str,
) -> tuple[str, ...]:
    if action not in {"manifest", "attest", "run"}:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_action_invalid"
        )
    command = (
        str(spec.interpreter),
        "-I",
        "-B",
        "-m",
        "gateway.production_owner_runtime",
        action,
        "--revision",
        spec.revision,
    )
    return command + (("--", *launcher_args) if launcher_args else ())


def _write_manifest(spec: OwnerRuntimeBuildSpec, value: Mapping[str, Any]) -> None:
    payload = _canonical(value) + b"\n"
    if len(payload) > runtime.MAX_MANIFEST_BYTES:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_manifest_oversized"
        )
    root = spec.release_root
    path = root / runtime.MANIFEST_NAME
    root.chmod(0o755)
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o444)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short manifest write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except OSError as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_manifest_write_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        root.chmod(0o555)


def _existing_release(spec: OwnerRuntimeBuildSpec) -> Mapping[str, Any]:
    attestation = _decode_line(
        _run(runtime_command(spec, "attest"), spec=spec, timeout=300)
    )
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "release_revision": spec.revision,
        "release_root": str(spec.release_root),
        "manifest_sha256": attestation["manifest_sha256"],
        "attestation_sha256": attestation["attestation_sha256"],
        "interpreter_sha256": attestation["interpreter_sha256"],
        "pyvenv_cfg_sha256": attestation["pyvenv_cfg_sha256"],
        "wheel_sha256": _sha256(_retained_wheel(spec)),
        "runtime_reused": True,
        "non_editable_install": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return validate_publication_receipt(
        {
            **unsigned,
            "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        },
        spec=spec,
    )


def build_owner_runtime(spec: OwnerRuntimeBuildSpec) -> Mapping[str, Any]:
    spec.validate()
    _verify_clean_source(spec)
    if spec.release_root.exists():
        return _existing_release(spec)
    spec.release_base.mkdir(parents=True, exist_ok=True, mode=0o700)
    spec.release_root.mkdir(mode=0o700)
    spec.incomplete_marker.write_text(spec.revision + "\n", encoding="ascii")
    spec.scratch_root.mkdir(mode=0o700)
    spec.snapshot_root.mkdir(mode=0o700)
    spec.wheel_root.mkdir(mode=0o700)
    spec.artifact_root.mkdir(mode=0o700)
    _run(
        (
            str(spec.git_executable),
            "-C",
            str(spec.source_root),
            "checkout-index",
            "--all",
            "--force",
            f"--prefix={spec.snapshot_root}/",
        ),
        spec=spec,
    )
    _run(
        (
            str(spec.uv_executable),
            "python",
            "install",
            PYTHON_VERSION,
            "--install-dir",
            str(spec.python_root),
            "--no-bin",
            "--managed-python",
            "--no-config",
        ),
        spec=spec,
    )
    managed_raw = _run(
        (
            str(spec.uv_executable),
            "python",
            "find",
            PYTHON_VERSION,
            "--managed-python",
            "--no-python-downloads",
            "--no-project",
            "--resolve-links",
            "--no-config",
        ),
        spec=spec,
        extra_environment={"UV_PYTHON_INSTALL_DIR": str(spec.python_root)},
    )
    try:
        managed = Path(managed_raw.decode("utf-8", errors="strict").strip())
    except UnicodeError as exc:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_python_invalid"
        ) from exc
    commands = build_commands(spec, managed_python=managed)
    _run(commands[0], spec=spec)
    _run(
        commands[1],
        spec=spec,
        extra_environment={"UV_PROJECT_ENVIRONMENT": str(spec.venv_root)},
    )
    _run(commands[2], spec=spec)
    wheels = sorted(spec.wheel_root.iterdir())
    if (
        len(wheels) != 1
        or _SAFE_WHEEL.fullmatch(wheels[0].name) is None
        or not wheels[0].is_file()
        or wheels[0].is_symlink()
    ):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_wheel_invalid"
        )
    retained = spec.artifact_root / wheels[0].name
    shutil.copyfile(wheels[0], retained)
    _run(
        (
            str(spec.uv_executable),
            "pip",
            "install",
            "--python",
            str(spec.interpreter),
            "--no-python-downloads",
            "--no-deps",
            "--no-build",
            "--no-index",
            "--no-cache",
            "--link-mode",
            "copy",
            "--no-config",
            str(retained),
        ),
        spec=spec,
    )
    _materialize_interpreter(spec, managed)
    _remove_dynamic_site_hooks(spec)
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_cleanup_unsafe"
        )
    shutil.rmtree(spec.scratch_root)
    spec.incomplete_marker.unlink()
    _seal_tree(spec.release_root)
    manifest = _decode_line(
        _run(runtime_command(spec, "manifest"), spec=spec, timeout=300)
    )
    _write_manifest(spec, manifest)
    attestation = _decode_line(
        _run(runtime_command(spec, "attest"), spec=spec, timeout=300)
    )
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "release_revision": spec.revision,
        "release_root": str(spec.release_root),
        "manifest_sha256": manifest["manifest_sha256"],
        "attestation_sha256": attestation["attestation_sha256"],
        "interpreter_sha256": attestation["interpreter_sha256"],
        "pyvenv_cfg_sha256": attestation["pyvenv_cfg_sha256"],
        "wheel_sha256": _sha256(retained),
        "runtime_reused": False,
        "non_editable_install": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return validate_publication_receipt(
        {
            **unsigned,
            "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        },
        spec=spec,
    )


def verify_owner_runtime(spec: OwnerRuntimeBuildSpec) -> Mapping[str, Any]:
    spec.validate()
    if not spec.release_root.exists() or spec.incomplete_marker.exists():
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_release_unavailable"
        )
    return _existing_release(spec)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the sealed production owner runtime"
    )
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build" and arguments.source_root is None:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_source_required"
        )
    if arguments.command == "verify" and arguments.source_root is not None:
        raise ProductionOwnerRuntimePackagingError(
            "production_owner_runtime_package_source_unexpected"
        )
    spec = OwnerRuntimeBuildSpec(
        revision=arguments.revision,
        source_root=(
            arguments.source_root.resolve(strict=True)
            if arguments.source_root is not None
            else Path("/var/empty/production-owner-runtime-source")
        ),
    )
    result = (
        build_owner_runtime(spec)
        if arguments.command == "build"
        else verify_owner_runtime(spec)
    )
    print(_canonical(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProductionOwnerRuntimePackagingError:
        print(
            '{"error_code":"production_owner_runtime_packaging_failed",'
            '"ok":false}',
            file=sys.stderr,
        )
        raise SystemExit(2) from None


__all__ = [
    "DEFAULT_RELEASE_BASE",
    "OwnerRuntimeBuildSpec",
    "ProductionOwnerRuntimePackagingError",
    "build_commands",
    "build_owner_runtime",
    "runtime_command",
    "validate_publication_receipt",
    "verify_owner_runtime",
]
