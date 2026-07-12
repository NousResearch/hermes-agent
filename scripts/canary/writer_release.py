#!/usr/bin/env python3
"""Build and render one sealed writer-only Muncho canary release.

The builder accepts one clean Git revision and invokes only fixed argv vectors.
It installs a uv-managed Python inside the release, creates a copied virtual
environment from that interpreter, installs frozen dependencies without the
project, and builds the exact project wheel only from a tracked-index scratch
snapshot using a hash-pinned build backend.  The retained wheel and completed
tree are root-owned, read-only, and described by a canonical per-path manifest
whose digest is stable for identical content and modes.

The systemd renderer is deliberately pure.  It emits the Canonical Writer and
credential-free gateway units plus a tmpfiles.d contract for the setgid writer
socket directory.  It never writes units, calls systemctl, starts services, or
accepts environment/secret payloads.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RELEASE_SCHEMA = "muncho-writer-only-release.v1"
UNIT_BUNDLE_SCHEMA = "muncho-writer-only-systemd-bundle.v1"
RELEASE_MANIFEST_NAME = "release-manifest.json"
INCOMPLETE_MARKER_NAME = ".release-build-incomplete"
BUILD_SCRATCH_NAME = ".release-build-scratch"
SCRATCH_PROVENANCE_NAME = "provenance.json"
SCRATCH_PROVENANCE_SCHEMA = "muncho-writer-release-scratch.v1"
BUILD_CONSTRAINTS_RELATIVE_PATH = Path(
    "scripts/canary/writer-build-constraints.txt"
)
WRITER_MODULE = "gateway.canonical_writer_bootstrap"
GATEWAY_MODULE = "gateway.canonical_writer_gateway_bootstrap"
DEFAULT_RELEASE_BASE = Path("/opt/muncho-canary-releases")
DEFAULT_UV_CACHE = Path("/var/cache/muncho-writer-release")
# Exact currently supported 3.11 security release.  The release builder never
# falls back to an older interpreter when this managed runtime is unavailable.
DEFAULT_PYTHON_VERSION = "3.11.15"
DEFAULT_UV_EXECUTABLE = Path("/usr/local/bin/uv")
DEFAULT_GIT_EXECUTABLE = Path("/usr/bin/git")
WRITER_UNIT_NAME = "muncho-canonical-writer.service"
GATEWAY_UNIT_NAME = "hermes-cloud-gateway.service"
TMPFILES_NAME = "muncho-canonical-writer.conf"

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PYTHON_RE = re.compile(r"^3\.(?:11|12|13)\.[0-9]+$")
_IDENTITY_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
_SAFE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MANIFEST_MODE = 0o400
_SEALED_DIRECTORY_MODE = 0o555
_SEALED_FILE_MODE = 0o444
_SEALED_EXECUTABLE_MODE = 0o555
_BUILD_DIRECTORY_MODE = 0o700
_COMMAND_TIMEOUT_SECONDS = 1800
_SETUPTOOLS_BUILD_VERSION = "81.0.0"
_SETUPTOOLS_BUILD_WHEEL_SHA256 = (
    "fdd925d5c5d9f62e4b74b30d6dd7828ce236fd6ed998a08d81de62ce5a6310d6"
)
_PINNED_BUILD_CONSTRAINTS = (
    f"setuptools=={_SETUPTOOLS_BUILD_VERSION} "
    f"--hash=sha256:{_SETUPTOOLS_BUILD_WHEEL_SHA256}\n"
).encode("ascii")
_SAFE_WHEEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+\.whl$")


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("release value is not canonical JSON") from exc
    return encoded.encode("utf-8", errors="strict")


def _absolute_normalized_path(value: str | os.PathLike[str], label: str) -> Path:
    raw = os.fspath(value)
    path = Path(raw)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != raw
        or _CONTROL_RE.search(raw) is not None
        or _SAFE_PATH_RE.fullmatch(raw) is None
    ):
        raise ValueError(f"{label} must be an absolute normalized safe path")
    return path


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _identity(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    return int(getter()) if callable(getter) else -1


def _require_root_linux() -> None:
    if _effective_uid() != 0:
        raise PermissionError("writer_release_builder_requires_uid_0")
    if sys.platform != "linux":
        raise RuntimeError("writer_release_builder_requires_linux")


@dataclass(frozen=True)
class ReleaseBuildSpec:
    revision: str
    source_root: Path
    release_base: Path = DEFAULT_RELEASE_BASE
    python_version: str = DEFAULT_PYTHON_VERSION
    uv_executable: Path = DEFAULT_UV_EXECUTABLE
    git_executable: Path = DEFAULT_GIT_EXECUTABLE
    uv_cache_dir: Path = DEFAULT_UV_CACHE

    @property
    def release_root(self) -> Path:
        return self.release_base / self.revision

    @property
    def managed_python_root(self) -> Path:
        return self.release_root / "python"

    @property
    def venv_root(self) -> Path:
        return self.release_root / "venv"

    @property
    def build_scratch_root(self) -> Path:
        return self.release_root / BUILD_SCRATCH_NAME

    @property
    def build_project_root(self) -> Path:
        return self.build_scratch_root / "project"

    @property
    def wheel_output_root(self) -> Path:
        return self.build_scratch_root / "wheel"

    @property
    def build_constraints(self) -> Path:
        return self.build_project_root / BUILD_CONSTRAINTS_RELATIVE_PATH

    @property
    def wheel_artifact_root(self) -> Path:
        return self.release_root / "artifacts"

    @property
    def interpreter(self) -> Path:
        return self.venv_root / "bin" / "python"

    @property
    def python_minor(self) -> str:
        major, minor, _patch = self.python_version.split(".")
        return f"{major}.{minor}"

    @property
    def site_packages(self) -> Path:
        return self.venv_root / "lib" / f"python{self.python_minor}" / "site-packages"

    @property
    def writer_module_origin(self) -> Path:
        return self.site_packages / "gateway" / "canonical_writer_bootstrap.py"

    @property
    def gateway_module_origin(self) -> Path:
        return (
            self.site_packages
            / "gateway"
            / "canonical_writer_gateway_bootstrap.py"
        )

    def validate(self) -> None:
        if not isinstance(self.revision, str) or _REVISION_RE.fullmatch(
            self.revision
        ) is None:
            raise ValueError("release revision must be exact lowercase 40-char SHA")
        if not isinstance(self.python_version, str) or _PYTHON_RE.fullmatch(
            self.python_version
        ) is None:
            raise ValueError("release Python must be an exact supported patch version")
        source = _absolute_normalized_path(self.source_root, "source root")
        base = _absolute_normalized_path(self.release_base, "release base")
        cache = _absolute_normalized_path(self.uv_cache_dir, "uv cache")
        uv = _absolute_normalized_path(self.uv_executable, "uv executable")
        git = _absolute_normalized_path(self.git_executable, "git executable")
        if source == base or _is_within(source, base) or _is_within(base, source):
            raise ValueError("source and release roots must be disjoint")
        if any(
            _is_within(cache, protected) or _is_within(protected, cache)
            for protected in (source, self.release_root)
        ):
            raise ValueError("uv cache must be outside source and release trees")
        if (
            self.build_scratch_root.parent != self.release_root
            or self.build_project_root.parent != self.build_scratch_root
            or self.wheel_output_root.parent != self.build_scratch_root
            or len(
                {
                    self.build_scratch_root,
                    self.build_project_root,
                    self.wheel_output_root,
                    self.wheel_artifact_root,
                }
            )
            != 4
            or self.wheel_artifact_root.parent != self.release_root
            or _is_within(self.wheel_artifact_root, self.build_scratch_root)
            or _is_within(self.build_scratch_root, self.wheel_artifact_root)
        ):
            raise ValueError("release scratch layout is not exact")
        if uv == git:
            raise ValueError("uv and git executables must be distinct")


@dataclass(frozen=True)
class BuildCommand:
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.argv or any(
            not isinstance(item, str)
            or not item
            or _CONTROL_RE.search(item) is not None
            for item in self.argv
        ):
            raise ValueError("release command argv is invalid")
        if self.argv[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}:
            raise ValueError("release commands cannot invoke a shell")
        if self.cwd is not None:
            _absolute_normalized_path(self.cwd, "command cwd")
        names = [name for name, _value in self.env]
        if len(names) != len(set(names)):
            raise ValueError("release command environment contains duplicates")
        for name, value in self.env:
            if (
                not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
                or _CONTROL_RE.search(value) is not None
            ):
                raise ValueError("release command environment is invalid")

    def environment(self) -> dict[str, str]:
        return dict(self.env)


def _clean_environment(
    spec: ReleaseBuildSpec,
    *,
    project_environment: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    values = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "UV_CACHE_DIR": str(spec.uv_cache_dir),
        "UV_NO_CONFIG": "1",
    }
    if project_environment is not None:
        values["UV_PROJECT_ENVIRONMENT"] = str(project_environment)
    return tuple(sorted(values.items()))


def checkout_commands(spec: ReleaseBuildSpec) -> tuple[BuildCommand, ...]:
    spec.validate()
    git = str(spec.git_executable)
    source = str(spec.source_root)
    environment = _clean_environment(spec)
    return (
        BuildCommand(
            (git, "-C", source, "rev-parse", "--verify", "HEAD"),
            env=environment,
        ),
        BuildCommand(
            (
                git,
                "-C",
                source,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            env=environment,
        ),
        BuildCommand(
            (
                git,
                "-C",
                source,
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
            ),
            env=environment,
        ),
    )


def source_snapshot_command(spec: ReleaseBuildSpec) -> BuildCommand:
    """Render the one-way tracked-index export into the isolated build context."""

    spec.validate()
    return BuildCommand(
        (
            str(spec.git_executable),
            "-C",
            str(spec.source_root),
            "checkout-index",
            "--all",
            "--force",
            f"--prefix={spec.build_project_root}/",
        ),
        env=_clean_environment(spec),
    )


def python_bootstrap_commands(spec: ReleaseBuildSpec) -> tuple[BuildCommand, ...]:
    spec.validate()
    environment = _clean_environment(spec)
    uv = str(spec.uv_executable)
    return (
        BuildCommand(
            (
                uv,
                "python",
                "install",
                spec.python_version,
                "--install-dir",
                str(spec.managed_python_root),
                "--no-bin",
                "--managed-python",
                "--no-config",
            ),
            env=environment,
        ),
        BuildCommand(
            (
                uv,
                "python",
                "find",
                spec.python_version,
                "--managed-python",
                "--no-python-downloads",
                "--no-project",
                "--resolve-links",
                "--no-config",
            ),
            env=tuple(
                sorted(
                    {
                        **dict(environment),
                        "UV_PYTHON_INSTALL_DIR": str(spec.managed_python_root),
                    }.items()
                )
            ),
        ),
    )


def install_commands(
    spec: ReleaseBuildSpec,
    managed_python: str | os.PathLike[str],
) -> tuple[BuildCommand, ...]:
    spec.validate()
    managed = _absolute_normalized_path(managed_python, "managed interpreter")
    if not _is_within(managed, spec.managed_python_root):
        raise ValueError("managed interpreter must be inside the release")
    uv = str(spec.uv_executable)
    project = str(spec.build_project_root)
    clean = _clean_environment(spec)
    sync_environment = _clean_environment(
        spec,
        project_environment=spec.venv_root,
    )
    return (
        BuildCommand(
            (
                str(managed),
                "-I",
                "-m",
                "venv",
                "--copies",
                str(spec.venv_root),
            ),
            env=clean,
        ),
        BuildCommand(
            (
                uv,
                "lock",
                "--check",
                "--python",
                str(managed),
                "--managed-python",
                "--no-python-downloads",
                "--project",
                project,
                "--no-config",
            ),
            env=clean,
        ),
        BuildCommand(
            (
                uv,
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
                project,
                "--no-config",
            ),
            env=sync_environment,
        ),
        BuildCommand(
            (
                uv,
                "build",
                "--wheel",
                "--out-dir",
                str(spec.wheel_output_root),
                "--no-create-gitignore",
                "--python",
                str(managed),
                "--managed-python",
                "--no-python-downloads",
                "--force-pep517",
                "--build-constraints",
                str(spec.build_constraints),
                "--require-hashes",
                "--no-config",
                project,
            ),
            env=clean,
        ),
    )


def wheel_install_command(spec: ReleaseBuildSpec, wheel: Path) -> BuildCommand:
    """Install one already-built local wheel without resolving or building."""

    spec.validate()
    exact_wheel = _absolute_normalized_path(wheel, "release wheel")
    if exact_wheel.parent != spec.wheel_artifact_root or exact_wheel.suffix != ".whl":
        raise ValueError("release wheel must be inside the exact artifact directory")
    return BuildCommand(
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
            str(exact_wheel),
        ),
        env=_clean_environment(spec),
    )


Runner = Callable[[BuildCommand], subprocess.CompletedProcess[str]]


def _runner(command: BuildCommand) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command.argv),
        cwd=command.cwd,
        env=command.environment(),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )


def _run_checked(
    command: BuildCommand,
    *,
    runner: Runner,
    label: str,
) -> subprocess.CompletedProcess[str]:
    completed = runner(command)
    if not isinstance(completed, subprocess.CompletedProcess):
        raise TypeError("release runner returned an invalid result")
    if completed.returncode != 0:
        stdout_sha = hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest()
        stderr_sha = hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest()
        raise RuntimeError(
            f"{label} failed: rc={completed.returncode} "
            f"stdout_sha256={stdout_sha} stderr_sha256={stderr_sha}"
        )
    return completed


def verify_clean_checkout(
    spec: ReleaseBuildSpec,
    *,
    runner: Runner = _runner,
) -> None:
    spec.validate()
    source_stat = os.lstat(spec.source_root)
    if not stat.S_ISDIR(source_stat.st_mode) or stat.S_ISLNK(source_stat.st_mode):
        raise ValueError("release source must be a real directory")
    for required in (spec.source_root / "pyproject.toml", spec.source_root / "uv.lock"):
        item = os.lstat(required)
        if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
            raise ValueError("release source lacks an exact project lock input")
    revision_command, status_command, ignored_command = checkout_commands(spec)
    revision = _run_checked(
        revision_command,
        runner=runner,
        label="git revision verification",
    ).stdout.strip()
    if revision != spec.revision:
        raise RuntimeError("release checkout does not match exact revision")
    status = _run_checked(
        status_command,
        runner=runner,
        label="git cleanliness verification",
    ).stdout
    if status:
        raise RuntimeError("release checkout is not clean")
    ignored = _run_checked(
        ignored_command,
        runner=runner,
        label="git ignored-input verification",
    ).stdout
    if ignored:
        raise RuntimeError("release checkout contains ignored build inputs")


def _validate_root_directory(path: Path, *, exact_mode: int | None = None) -> None:
    item = os.lstat(path)
    mode = stat.S_IMODE(item.st_mode)
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or mode & 0o022
        or (exact_mode is not None and mode != exact_mode)
        or _list_xattrs(path)
    ):
        raise PermissionError("release directory is not root-controlled")


def _validate_root_parent_chain(path: Path) -> None:
    current = path
    while True:
        _validate_root_directory(current)
        if current == current.parent:
            return
        current = current.parent


def _validate_root_executable(path: Path) -> None:
    item = os.lstat(path)
    mode = stat.S_IMODE(item.st_mode)
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or mode & 0o022
        or not mode & 0o111
    ):
        raise PermissionError("release executable is not root-controlled")
    _validate_root_parent_chain(path.parent)


def _validate_root_source_tree(root: Path) -> None:
    _validate_root_parent_chain(root)
    resolved_root = root.resolve(strict=True)
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            path = Path(current) / name
            item = os.lstat(path)
            if item.st_uid != 0 or item.st_gid != 0 or stat.S_IMODE(item.st_mode) & 0o022:
                raise PermissionError("release source tree is not root-controlled")
            if stat.S_ISLNK(item.st_mode):
                if not _is_within(path.resolve(strict=True), resolved_root):
                    raise PermissionError("release source symlink escapes checkout")
            elif not (stat.S_ISDIR(item.st_mode) or stat.S_ISREG(item.st_mode)):
                raise PermissionError("release source contains a special file")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _list_xattrs(path: Path) -> tuple[str, ...]:
    lister = getattr(os, "listxattr", None)
    if not callable(lister):
        if sys.platform == "linux":
            raise RuntimeError("Linux release xattr inspection is unavailable")
        return ()
    try:
        return tuple(lister(path, follow_symlinks=False))
    except OSError as exc:
        raise RuntimeError("release xattrs are unavailable") from exc


@dataclass(frozen=True)
class TreeEntry:
    path: str
    kind: str
    mode: str
    size: int = 0
    sha256: str = ""
    target: str = ""

    def to_mapping(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind,
            "mode": self.mode,
        }
        if self.kind == "file":
            value.update({"size": self.size, "sha256": self.sha256})
        elif self.kind == "symlink":
            value["target"] = self.target
        return value


def collect_tree_entries(root_value: str | os.PathLike[str]) -> tuple[TreeEntry, ...]:
    root = _absolute_normalized_path(root_value, "release root")
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError("release root must be a real directory")
    entries: list[TreeEntry] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in [*directories, *files]:
            if _CONTROL_RE.search(name) is not None:
                raise ValueError("release path contains control characters")
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative in {RELEASE_MANIFEST_NAME, INCOMPLETE_MARKER_NAME}:
                continue
            item = os.lstat(path)
            if _list_xattrs(path):
                raise ValueError("release entries cannot carry extended attributes")
            mode = f"{stat.S_IMODE(item.st_mode):04o}"
            if stat.S_ISLNK(item.st_mode):
                target = os.readlink(path)
                if not target or _CONTROL_RE.search(target) is not None:
                    raise ValueError("release symlink target is invalid")
                resolved = path.resolve(strict=True)
                if not _is_within(resolved, root.resolve(strict=True)):
                    raise ValueError("release symlink escapes immutable artifact")
                entries.append(TreeEntry(relative, "symlink", mode, target=target))
            elif stat.S_ISDIR(item.st_mode):
                entries.append(TreeEntry(relative, "directory", mode))
            elif stat.S_ISREG(item.st_mode):
                if item.st_nlink != 1:
                    raise ValueError("release regular file must not be hard-linked")
                entries.append(
                    TreeEntry(
                        relative,
                        "file",
                        mode,
                        size=item.st_size,
                        sha256=_hash_file(path),
                    )
                )
            else:
                raise ValueError("release contains an unsupported filesystem object")
    entries.sort(key=lambda entry: entry.path)
    if len({entry.path for entry in entries}) != len(entries):
        raise ValueError("release tree contains duplicate paths")
    return tuple(entries)


@dataclass(frozen=True)
class ReleaseManifest:
    revision: str
    artifact_root: str
    python_version: str
    interpreter: str
    writer_module_origin: str
    gateway_module_origin: str
    entries: tuple[TreeEntry, ...]
    artifact_sha256: str
    schema: str = RELEASE_SCHEMA
    writer_module: str = WRITER_MODULE
    gateway_module: str = GATEWAY_MODULE

    def unsigned_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "artifact_root": self.artifact_root,
            "python_version": self.python_version,
            "interpreter": self.interpreter,
            "writer_module": self.writer_module,
            "writer_module_origin": self.writer_module_origin,
            "gateway_module": self.gateway_module,
            "gateway_module_origin": self.gateway_module_origin,
            "entries": [entry.to_mapping() for entry in self.entries],
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "artifact_sha256": self.artifact_sha256}

    @property
    def computed_artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.unsigned_mapping())).hexdigest()


def create_release_manifest(spec: ReleaseBuildSpec) -> ReleaseManifest:
    spec.validate()
    expected = (
        spec.interpreter,
        spec.writer_module_origin,
        spec.gateway_module_origin,
    )
    for path in expected:
        item = os.lstat(path)
        if not stat.S_ISREG(item.st_mode) or stat.S_ISLNK(item.st_mode):
            raise ValueError("release entry point is not an installed regular file")
    entries = collect_tree_entries(spec.release_root)
    provisional = ReleaseManifest(
        revision=spec.revision,
        artifact_root=str(spec.release_root),
        python_version=spec.python_version,
        interpreter=str(spec.interpreter),
        writer_module_origin=str(spec.writer_module_origin),
        gateway_module_origin=str(spec.gateway_module_origin),
        entries=entries,
        artifact_sha256="",
    )
    digest = provisional.computed_artifact_sha256
    return ReleaseManifest(
        revision=provisional.revision,
        artifact_root=provisional.artifact_root,
        python_version=provisional.python_version,
        interpreter=provisional.interpreter,
        writer_module_origin=provisional.writer_module_origin,
        gateway_module_origin=provisional.gateway_module_origin,
        entries=provisional.entries,
        artifact_sha256=digest,
    )


def _seal_release_tree(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for name in sorted(files):
            path = current_path / name
            item = os.lstat(path)
            if stat.S_ISLNK(item.st_mode):
                os.chown(path, 0, 0, follow_symlinks=False)
                continue
            if not stat.S_ISREG(item.st_mode):
                raise ValueError("release contains an unsupported file before sealing")
            executable = bool(stat.S_IMODE(item.st_mode) & 0o111)
            os.chown(path, 0, 0, follow_symlinks=False)
            os.chmod(
                path,
                _SEALED_EXECUTABLE_MODE if executable else _SEALED_FILE_MODE,
                follow_symlinks=False,
            )
        for name in sorted(directories):
            path = current_path / name
            item = os.lstat(path)
            if stat.S_ISLNK(item.st_mode):
                os.chown(path, 0, 0, follow_symlinks=False)
                continue
            if not stat.S_ISDIR(item.st_mode):
                raise ValueError("release contains an unsupported directory before sealing")
            os.chown(path, 0, 0, follow_symlinks=False)
            os.chmod(path, _SEALED_DIRECTORY_MODE, follow_symlinks=False)
    os.chown(root, 0, 0, follow_symlinks=False)


def _write_release_manifest(root: Path, manifest: ReleaseManifest) -> None:
    path = root / RELEASE_MANIFEST_NAME
    raw = _canonical_bytes(manifest.to_mapping()) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchown(descriptor, 0, 0)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("release manifest write made no progress")
            offset += written
        os.fchmod(descriptor, _MANIFEST_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_incomplete_marker(spec: ReleaseBuildSpec) -> None:
    path = spec.release_root / INCOMPLETE_MARKER_NAME
    raw = _canonical_bytes(
        {
            "schema": RELEASE_SCHEMA,
            "revision": spec.revision,
            "state": "incomplete",
        }
    ) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        os.fchown(descriptor, 0, 0)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("release marker write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _scratch_provenance_bytes(
    spec: ReleaseBuildSpec,
    *,
    scratch_device: int,
    scratch_inode: int,
) -> bytes:
    source = os.lstat(spec.source_root)
    return _canonical_bytes(
        {
            "schema": SCRATCH_PROVENANCE_SCHEMA,
            "revision": spec.revision,
            "source_root": str(spec.source_root),
            "source_device": int(source.st_dev),
            "source_inode": int(source.st_ino),
            "scratch_root": str(spec.build_scratch_root),
            "scratch_device": int(scratch_device),
            "scratch_inode": int(scratch_inode),
        }
    ) + b"\n"


def _write_scratch_provenance(
    spec: ReleaseBuildSpec,
    *,
    scratch_device: int,
    scratch_inode: int,
) -> None:
    path = spec.build_scratch_root / SCRATCH_PROVENANCE_NAME
    raw = _scratch_provenance_bytes(
        spec,
        scratch_device=scratch_device,
        scratch_inode=scratch_inode,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        os.fchown(descriptor, 0, 0)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("scratch provenance write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_build_scratch(spec: ReleaseBuildSpec) -> tuple[int, int]:
    """Create one non-reusable scratch context inside the incomplete release."""

    os.mkdir(spec.build_scratch_root, _BUILD_DIRECTORY_MODE)
    scratch = os.lstat(spec.build_scratch_root)
    if (
        not stat.S_ISDIR(scratch.st_mode)
        or stat.S_ISLNK(scratch.st_mode)
        or scratch.st_uid != 0
        or scratch.st_gid != 0
        or stat.S_IMODE(scratch.st_mode) != _BUILD_DIRECTORY_MODE
    ):
        raise PermissionError("release scratch root is not exact")
    identity = (int(scratch.st_dev), int(scratch.st_ino))
    _write_scratch_provenance(
        spec,
        scratch_device=identity[0],
        scratch_inode=identity[1],
    )
    os.mkdir(spec.build_project_root, _BUILD_DIRECTORY_MODE)
    os.mkdir(spec.wheel_output_root, _BUILD_DIRECTORY_MODE)
    os.mkdir(spec.wheel_artifact_root, _BUILD_DIRECTORY_MODE)
    return identity


def _validate_build_constraints(spec: ReleaseBuildSpec) -> None:
    path = spec.build_constraints
    item = os.lstat(path)
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) & 0o022
        or item.st_size != len(_PINNED_BUILD_CONSTRAINTS)
        or path.read_bytes() != _PINNED_BUILD_CONSTRAINTS
    ):
        raise PermissionError("release build constraints are not exact")


def _select_built_wheel(spec: ReleaseBuildSpec) -> Path:
    entries = sorted(spec.wheel_output_root.iterdir(), key=lambda item: item.name)
    if len(entries) != 1 or _SAFE_WHEEL_NAME_RE.fullmatch(entries[0].name) is None:
        raise RuntimeError("release build did not produce one exact wheel")
    wheel = entries[0]
    item = os.lstat(wheel)
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) & 0o022
        or item.st_nlink != 1
        or item.st_size <= 0
    ):
        raise RuntimeError("release build wheel is not a root-owned regular file")
    return wheel


def _copy_built_wheel(spec: ReleaseBuildSpec, source: Path) -> Path:
    """Copy the exact wheel into the manifest-bound release artifact tree."""

    source_path = _absolute_normalized_path(source, "scratch wheel")
    if source_path.parent != spec.wheel_output_root:
        raise ValueError("scratch wheel is outside the exact output directory")
    destination = spec.wheel_artifact_root / source_path.name
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
        os,
        "O_CLOEXEC",
        0,
    )
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
        write_flags |= os.O_NOFOLLOW
    source_fd = os.open(source_path, read_flags)
    try:
        source_stat = os.fstat(source_fd)
        observed = os.lstat(source_path)
        if (
            (source_stat.st_dev, source_stat.st_ino)
            != (observed.st_dev, observed.st_ino)
            or not stat.S_ISREG(source_stat.st_mode)
            or source_stat.st_nlink != 1
        ):
            raise RuntimeError("scratch wheel identity changed before copy")
        destination_fd = os.open(destination, write_flags, 0o400)
        try:
            os.fchown(destination_fd, 0, 0)
            while chunk := os.read(source_fd, 1024 * 1024):
                offset = 0
                while offset < len(chunk):
                    written = os.write(destination_fd, chunk[offset:])
                    if written <= 0:
                        raise OSError("release wheel copy made no progress")
                    offset += written
            os.fchmod(destination_fd, _SEALED_FILE_MODE)
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)
    copied = os.lstat(destination)
    original = os.lstat(source_path)
    if (
        not stat.S_ISREG(copied.st_mode)
        or stat.S_ISLNK(copied.st_mode)
        or copied.st_uid != 0
        or copied.st_gid != 0
        or stat.S_IMODE(copied.st_mode) != _SEALED_FILE_MODE
        or copied.st_nlink != 1
        or copied.st_size != original.st_size
        or (copied.st_dev, copied.st_ino) == (original.st_dev, original.st_ino)
        or _hash_file(destination) != _hash_file(source_path)
    ):
        raise RuntimeError("release wheel copy digest does not match source")
    return destination


def _validate_scratch_provenance(
    spec: ReleaseBuildSpec,
    *,
    scratch_device: int,
    scratch_inode: int,
) -> None:
    root = os.lstat(spec.build_scratch_root)
    if (
        not stat.S_ISDIR(root.st_mode)
        or stat.S_ISLNK(root.st_mode)
        or (root.st_dev, root.st_ino) != (scratch_device, scratch_inode)
        or root.st_uid != 0
        or root.st_gid != 0
        or stat.S_IMODE(root.st_mode) != _BUILD_DIRECTORY_MODE
    ):
        raise RuntimeError("release scratch identity drifted")
    path = spec.build_scratch_root / SCRATCH_PROVENANCE_NAME
    item = os.lstat(path)
    expected = _scratch_provenance_bytes(
        spec,
        scratch_device=scratch_device,
        scratch_inode=scratch_inode,
    )
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or item.st_nlink != 1
        or stat.S_IMODE(item.st_mode) != 0o400
        or item.st_size != len(expected)
    ):
        raise RuntimeError("release scratch provenance drifted")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (item.st_dev, item.st_ino):
            raise RuntimeError("release scratch provenance identity changed")
        raw = bytearray()
        while chunk := os.read(descriptor, 4096):
            raw.extend(chunk)
            if len(raw) > len(expected):
                raise RuntimeError("release scratch provenance is oversized")
    finally:
        os.close(descriptor)
    if bytes(raw) != expected:
        raise RuntimeError("release scratch provenance does not match")


def _remove_build_scratch(
    spec: ReleaseBuildSpec,
    *,
    scratch_device: int,
    scratch_inode: int,
) -> None:
    """Remove only the provenance-bound scratch tree, or leave it untouched."""

    _validate_scratch_provenance(
        spec,
        scratch_device=scratch_device,
        scratch_inode=scratch_inode,
    )
    for current, directories, files in os.walk(
        spec.build_scratch_root,
        topdown=True,
        followlinks=False,
    ):
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            path = Path(current) / name
            item = os.lstat(path)
            if item.st_uid != 0 or item.st_gid != 0:
                raise RuntimeError("release scratch contains a non-root-owned entry")
            if stat.S_ISDIR(item.st_mode):
                if item.st_dev != scratch_device:
                    raise RuntimeError("release scratch crosses a filesystem boundary")
            elif stat.S_ISREG(item.st_mode):
                if item.st_nlink != 1:
                    raise RuntimeError("release scratch contains a hard-linked file")
            elif not stat.S_ISLNK(item.st_mode):
                raise RuntimeError("release scratch contains a special file")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RuntimeError("symlink-safe release scratch cleanup is unavailable")
    shutil.rmtree(spec.build_scratch_root)
    if os.path.lexists(spec.build_scratch_root):
        raise RuntimeError("release scratch cleanup did not complete")


def _validate_installed_runtime(
    spec: ReleaseBuildSpec,
    managed_python: Path,
) -> None:
    interpreter_stat = os.lstat(spec.interpreter)
    if (
        not stat.S_ISREG(interpreter_stat.st_mode)
        or stat.S_ISLNK(interpreter_stat.st_mode)
        or interpreter_stat.st_nlink != 1
        or not stat.S_IMODE(interpreter_stat.st_mode) & 0o111
    ):
        raise RuntimeError("release venv does not contain a copied interpreter")
    config_path = spec.venv_root / "pyvenv.cfg"
    config_stat = os.lstat(config_path)
    if (
        not stat.S_ISREG(config_stat.st_mode)
        or stat.S_ISLNK(config_stat.st_mode)
        or config_stat.st_size > 32 * 1024
    ):
        raise RuntimeError("release pyvenv identity is invalid")
    config: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            continue
        name, value = line.split(" = ", 1)
        if name in config:
            raise RuntimeError("release pyvenv identity contains duplicates")
        config[name] = value
    home = _absolute_normalized_path(config.get("home", ""), "pyvenv home")
    if (
        home != managed_python.parent
        or not _is_within(home, spec.managed_python_root)
        or config.get("include-system-site-packages", "").casefold() != "false"
    ):
        raise RuntimeError("release pyvenv is not bound to managed Python")
    executable_value = config.get("executable")
    if executable_value:
        executable = _absolute_normalized_path(
            executable_value,
            "pyvenv executable",
        )
        if executable != managed_python:
            raise RuntimeError("release pyvenv executable identity drifted")
    site_stat = os.lstat(spec.site_packages)
    if not stat.S_ISDIR(site_stat.st_mode) or stat.S_ISLNK(site_stat.st_mode):
        raise RuntimeError("release site-packages directory is invalid")
    if list(spec.site_packages.glob("*.egg-link")):
        raise RuntimeError("release contains an editable egg-link")
    if list(spec.site_packages.glob("*.pth")):
        raise RuntimeError("release contains a dynamic site path")
    for direct_url in spec.site_packages.glob("*.dist-info/direct_url.json"):
        raw = direct_url.read_bytes()
        if len(raw) > 64 * 1024:
            raise RuntimeError("release direct-url metadata is oversized")
        try:
            value = json.loads(raw.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("release direct-url metadata is invalid") from exc
        if (
            isinstance(value, Mapping)
            and isinstance(value.get("dir_info"), Mapping)
            and value["dir_info"].get("editable") is True
        ):
            raise RuntimeError("release project install is editable")
    forbidden_scripts = (
        spec.site_packages / "scripts/canonical_writer_bootstrap.py",
        spec.site_packages / "scripts/canonical_writer_service.py",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden_scripts):
        raise RuntimeError("release contains a legacy scripts bootstrap")


def build_release(
    spec: ReleaseBuildSpec,
    *,
    runner: Runner = _runner,
) -> ReleaseManifest:
    """Build and seal one exact release; never reuse or replace a path."""

    spec.validate()
    _require_root_linux()
    _validate_root_parent_chain(spec.release_base)
    _validate_root_parent_chain(spec.uv_cache_dir)
    _validate_root_executable(spec.uv_executable)
    _validate_root_executable(spec.git_executable)
    _validate_root_source_tree(spec.source_root)
    verify_clean_checkout(spec, runner=runner)
    try:
        os.mkdir(spec.release_root, _BUILD_DIRECTORY_MODE)
    except FileExistsError as exc:
        raise FileExistsError("exact release path already exists") from exc
    _validate_root_directory(spec.release_root, exact_mode=_BUILD_DIRECTORY_MODE)
    _write_incomplete_marker(spec)

    try:
        install_python, find_python = python_bootstrap_commands(spec)
        _run_checked(install_python, runner=runner, label="uv managed Python install")
        managed_raw = _run_checked(
            find_python,
            runner=runner,
            label="uv managed Python discovery",
        ).stdout.strip()
        managed_python = _absolute_normalized_path(
            managed_raw,
            "discovered managed interpreter",
        )
        if not _is_within(managed_python, spec.managed_python_root):
            raise RuntimeError("uv discovered Python outside the exact release")
        managed_stat = os.lstat(managed_python)
        if (
            not stat.S_ISREG(managed_stat.st_mode)
            or stat.S_ISLNK(managed_stat.st_mode)
            or not stat.S_IMODE(managed_stat.st_mode) & 0o111
        ):
            raise RuntimeError("uv managed Python is not a copied executable")

        scratch_device, scratch_inode = _prepare_build_scratch(spec)
        _run_checked(
            source_snapshot_command(spec),
            runner=runner,
            label="tracked source snapshot",
        )
        _validate_root_source_tree(spec.build_project_root)
        _validate_build_constraints(spec)
        for index, command in enumerate(
            install_commands(spec, managed_python),
            start=1,
        ):
            _run_checked(command, runner=runner, label=f"release install step {index}")
        scratch_wheel = _select_built_wheel(spec)
        artifact_wheel = _copy_built_wheel(spec, scratch_wheel)
        if any(
            os.path.lexists(path)
            for path in (spec.writer_module_origin, spec.gateway_module_origin)
        ):
            raise RuntimeError("release project was installed before exact wheel gate")
        _run_checked(
            wheel_install_command(spec, artifact_wheel),
            runner=runner,
            label="exact release wheel install",
        )
        _validate_installed_runtime(spec, managed_python)
    except Exception as build_error:
        try:
            verify_clean_checkout(spec, runner=runner)
        except Exception as source_error:
            raise ExceptionGroup(
                "release build failed and canonical source re-attestation failed",
                [build_error, source_error],
            ) from None
        raise
    else:
        # Every post-marker success and ordinary failure path re-attests that
        # build tooling left the canonical checkout byte/SCM clean.  This code
        # deliberately never attempts to clean or repair that source.
        verify_clean_checkout(spec, runner=runner)
    _remove_build_scratch(
        spec,
        scratch_device=scratch_device,
        scratch_inode=scratch_inode,
    )
    _seal_release_tree(spec.release_root)
    manifest = create_release_manifest(spec)
    _write_release_manifest(spec.release_root, manifest)
    (spec.release_root / INCOMPLETE_MARKER_NAME).unlink()
    os.chmod(spec.release_root, _SEALED_DIRECTORY_MODE, follow_symlinks=False)
    directory_fd = os.open(
        spec.release_root,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _validate_root_directory(
        spec.release_root,
        exact_mode=_SEALED_DIRECTORY_MODE,
    )
    manifest_stat = os.lstat(spec.release_root / RELEASE_MANIFEST_NAME)
    if (
        not stat.S_ISREG(manifest_stat.st_mode)
        or manifest_stat.st_uid != 0
        or manifest_stat.st_gid != 0
        or stat.S_IMODE(manifest_stat.st_mode) != _MANIFEST_MODE
    ):
        raise RuntimeError("release manifest was not sealed")
    return manifest


@dataclass(frozen=True)
class WriterOnlyUnitSpec:
    writer_user: str = "muncho-canonical-writer"
    writer_group: str = "muncho-canonical-writer"
    gateway_user: str = "muncho-gateway"
    gateway_group: str = "muncho-gateway"
    socket_client_group: str = "muncho-writer-client"
    projector_group: str = "muncho-projector"
    writer_config: Path = Path("/etc/muncho-canonical-writer/writer.json")
    gateway_config: Path = Path("/etc/hermes/config.yaml")
    gateway_home: Path = Path("/var/lib/hermes-gateway")
    writer_runtime: Path = Path("/run/muncho-canonical-writer")
    projection_directory: Path = Path(
        "/var/lib/muncho-canonical-writer/projection"
    )
    gateway_runtime: Path = Path("/run/hermes-cloud-gateway")
    gateway_state: Path = Path("/var/lib/hermes-gateway")
    gateway_logs: Path = Path("/var/log/hermes-gateway")
    database_ip_allow: tuple[str, ...] = ()

    def validate(self) -> None:
        identities = (
            _identity(self.writer_user, "writer user"),
            _identity(self.writer_group, "writer group"),
            _identity(self.gateway_user, "gateway user"),
            _identity(self.gateway_group, "gateway group"),
            _identity(self.socket_client_group, "socket client group"),
            _identity(self.projector_group, "projector group"),
        )
        if (
            self.writer_user == self.gateway_user
            or len(
                {
                    self.writer_group,
                    self.gateway_group,
                    self.socket_client_group,
                    self.projector_group,
                }
            )
            != 4
        ):
            raise ValueError("writer-only runtime identities must be distinct")
        paths = (
            self.writer_config,
            self.gateway_config,
            self.writer_runtime,
            self.projection_directory,
            self.gateway_runtime,
            self.gateway_state,
            self.gateway_logs,
        )
        normalized = tuple(
            _absolute_normalized_path(path, "unit path") for path in paths
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("writer-only unit paths must be distinct")
        if self.writer_runtime != Path("/run/muncho-canonical-writer"):
            raise ValueError("writer runtime path is protocol-pinned")
        if (
            self.gateway_config != Path("/etc/hermes/config.yaml")
            or self.gateway_home != Path("/var/lib/hermes-gateway")
            or self.gateway_state != self.gateway_home
        ):
            raise ValueError("gateway managed config and passwd home are pinned")
        if len(self.database_ip_allow) != 1:
            raise ValueError("writer-only release requires one exact database IP")
        try:
            network = ipaddress.ip_network(self.database_ip_allow[0], strict=True)
        except ValueError as exc:
            raise ValueError("writer database IP allow-list is invalid") from exc
        if network.num_addresses != 1 or str(network) != self.database_ip_allow[0]:
            raise ValueError("writer database IP allow-list must be an exact host")


@dataclass(frozen=True)
class SystemdUnitBundle:
    writer_service: str
    gateway_service: str
    tmpfiles: str
    contract: tuple[tuple[str, str], ...]
    sha256: str
    schema: str = UNIT_BUNDLE_SCHEMA

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "writer_service": self.writer_service,
            "gateway_service": self.gateway_service,
            "tmpfiles": self.tmpfiles,
            "contract": dict(self.contract),
            "sha256": self.sha256,
        }


def _common_hardening(*, address_families: str) -> list[str]:
    return [
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "MemoryDenyWriteExecute=yes",
        "PrivateDevices=yes",
        "PrivateTmp=yes",
        "ProtectClock=yes",
        "ProtectControlGroups=yes",
        "ProtectHome=yes",
        "ProtectHostname=yes",
        "ProtectKernelLogs=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "ProtectProc=invisible",
        "ProtectSystem=strict",
        "ProcSubset=pid",
        "RemoveIPC=yes",
        f"RestrictAddressFamilies={address_families}",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "UMask=0077",
    ]


def _fixed_service_environment(*, user: str, home: str) -> list[str]:
    """Return the complete non-secret deterministic service environment."""

    return [
        f"Environment=HOME={home}",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        f"Environment=LOGNAME={user}",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=SHELL=/usr/sbin/nologin",
        "Environment=TZ=UTC",
        f"Environment=USER={user}",
    ]


def render_systemd_units(
    manifest: ReleaseManifest,
    spec: WriterOnlyUnitSpec,
) -> SystemdUnitBundle:
    spec.validate()
    if manifest.schema != RELEASE_SCHEMA:
        raise ValueError("release manifest schema is invalid")
    if _REVISION_RE.fullmatch(manifest.revision) is None:
        raise ValueError("release manifest revision is invalid")
    if _SHA256_RE.fullmatch(manifest.artifact_sha256) is None:
        raise ValueError("release artifact digest is invalid")
    if manifest.artifact_sha256 != manifest.computed_artifact_sha256:
        raise ValueError("release artifact digest does not match manifest")
    if _PYTHON_RE.fullmatch(manifest.python_version) is None:
        raise ValueError("release Python identity is invalid")
    release_root = _absolute_normalized_path(
        manifest.artifact_root,
        "manifest artifact root",
    )
    if (
        release_root.parent != DEFAULT_RELEASE_BASE
        or release_root.name != manifest.revision
    ):
        raise ValueError("release path is not revision-addressed")
    interpreter = _absolute_normalized_path(
        manifest.interpreter,
        "manifest interpreter",
    )
    writer_origin = _absolute_normalized_path(
        manifest.writer_module_origin,
        "writer module origin",
    )
    gateway_origin = _absolute_normalized_path(
        manifest.gateway_module_origin,
        "gateway module origin",
    )
    python_parts = manifest.python_version.split(".")
    expected_site_packages = (
        release_root
        / "venv"
        / "lib"
        / f"python{python_parts[0]}.{python_parts[1]}"
        / "site-packages"
    )
    if (
        manifest.writer_module != WRITER_MODULE
        or manifest.gateway_module != GATEWAY_MODULE
        or interpreter != release_root / "venv/bin/python"
        or writer_origin
        != expected_site_packages / "gateway/canonical_writer_bootstrap.py"
        or gateway_origin
        != expected_site_packages / "gateway/canonical_writer_gateway_bootstrap.py"
    ):
        raise ValueError("release module origins are not exact")

    writer_lines = [
        "# Generated from a digest-bound writer-only release; do not edit.",
        f"# ArtifactSHA256={manifest.artifact_sha256}",
        f"# ModuleOrigin={writer_origin}",
        "[Unit]",
        "Description=Muncho privileged Canonical Writer (isolated canary)",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT_NAME}",
        f"AssertPathIsDirectory={spec.writer_runtime}",
        f"AssertPathIsDirectory={spec.projection_directory}",
        f"AssertPathExists={spec.writer_config}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={spec.writer_user}",
        f"Group={spec.writer_group}",
        f"SupplementaryGroups={spec.projector_group}",
        f"WorkingDirectory={release_root}",
        (
            f"ExecStart={interpreter} -I -m {WRITER_MODULE} "
            f"--config {spec.writer_config}"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=60s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_service_environment(
            user=spec.writer_user,
            home="/nonexistent",
        ),
        *_common_hardening(address_families="AF_UNIX AF_INET AF_INET6"),
        "IPAddressDeny=any",
        f"IPAddressAllow={spec.database_ip_allow[0]}",
        f"BindReadOnlyPaths={release_root}",
        f"ReadOnlyPaths={spec.writer_config}",
        f"ReadWritePaths={spec.writer_runtime}",
        f"ReadWritePaths={spec.projection_directory}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    gateway_lines = [
        "# Generated from a digest-bound writer-only release; do not edit.",
        f"# ArtifactSHA256={manifest.artifact_sha256}",
        f"# ModuleOrigin={gateway_origin}",
        f"# PasswdHome={spec.gateway_home}",
        f"# ManagedConfig={spec.gateway_config}",
        "[Unit]",
        "Description=Muncho credential-free gateway (writer-only canary)",
        f"BindsTo={WRITER_UNIT_NAME}",
        f"After={WRITER_UNIT_NAME}",
        f"AssertPathIsDirectory={spec.gateway_home}",
        f"AssertPathExists={spec.gateway_config}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={spec.gateway_user}",
        f"Group={spec.gateway_group}",
        f"SupplementaryGroups={spec.socket_client_group}",
        f"WorkingDirectory={release_root}",
        (
            f"ExecStart={interpreter} -I -m {GATEWAY_MODULE}"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=60s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_service_environment(
            user=spec.gateway_user,
            home=str(spec.gateway_home),
        ),
        "PrivateNetwork=yes",
        *_common_hardening(address_families="AF_UNIX"),
        f"BindReadOnlyPaths={release_root}",
        f"ReadOnlyPaths={spec.gateway_config}",
        f"ReadWritePaths={spec.gateway_runtime}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    tmpfiles_lines = [
        "# type path mode user group age argument",
        (
            f"d {spec.writer_runtime} 2750 {spec.writer_user} "
            f"{spec.socket_client_group} - -"
        ),
        (
            f"d {spec.projection_directory} 0750 {spec.writer_user} "
            f"{spec.projector_group} - -"
        ),
        (
            f"d {spec.gateway_runtime} 0700 {spec.gateway_user} "
            f"{spec.gateway_group} - -"
        ),
        (
            f"d {spec.gateway_state} 0700 {spec.gateway_user} "
            f"{spec.gateway_group} - -"
        ),
        (
            f"d {spec.gateway_logs} 0700 {spec.gateway_user} "
            f"{spec.gateway_group} - -"
        ),
    ]
    writer = "\n".join(writer_lines) + "\n"
    gateway = "\n".join(gateway_lines) + "\n"
    tmpfiles = "\n".join(tmpfiles_lines) + "\n"
    forbidden = re.compile(
        r"(?im)^(?:EnvironmentFile|PassEnvironment|LoadCredential)="
    )
    if forbidden.search(writer) or forbidden.search(gateway):
        raise RuntimeError("writer-only units cannot inject environment or credentials")
    payload = {
        "schema": UNIT_BUNDLE_SCHEMA,
        "writer_service": writer,
        "gateway_service": gateway,
        "tmpfiles": tmpfiles,
        "contract": {
            "revision": manifest.revision,
            "artifact_sha256": manifest.artifact_sha256,
            "working_directory": str(release_root),
            "writer_user": spec.writer_user,
            "writer_group": spec.writer_group,
            "gateway_user": spec.gateway_user,
            "gateway_group": spec.gateway_group,
            "gateway_passwd_home": str(spec.gateway_home),
            "gateway_config": str(spec.gateway_config),
            "socket_client_group": spec.socket_client_group,
            "writer_runtime": str(spec.writer_runtime),
            "writer_runtime_mode": "2750",
            "database_ip_allow": spec.database_ip_allow[0],
        },
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return SystemdUnitBundle(
        writer_service=writer,
        gateway_service=gateway,
        tmpfiles=tmpfiles,
        contract=tuple(sorted(payload["contract"].items())),
        sha256=digest,
    )


__all__ = [
    "BUILD_CONSTRAINTS_RELATIVE_PATH",
    "BUILD_SCRATCH_NAME",
    "BuildCommand",
    "GATEWAY_MODULE",
    "GATEWAY_UNIT_NAME",
    "INCOMPLETE_MARKER_NAME",
    "RELEASE_MANIFEST_NAME",
    "RELEASE_SCHEMA",
    "SCRATCH_PROVENANCE_NAME",
    "SCRATCH_PROVENANCE_SCHEMA",
    "ReleaseBuildSpec",
    "ReleaseManifest",
    "SystemdUnitBundle",
    "TMPFILES_NAME",
    "TreeEntry",
    "UNIT_BUNDLE_SCHEMA",
    "WRITER_MODULE",
    "WRITER_UNIT_NAME",
    "WriterOnlyUnitSpec",
    "build_release",
    "checkout_commands",
    "collect_tree_entries",
    "create_release_manifest",
    "install_commands",
    "python_bootstrap_commands",
    "render_systemd_units",
    "source_snapshot_command",
    "verify_clean_checkout",
    "wheel_install_command",
]
