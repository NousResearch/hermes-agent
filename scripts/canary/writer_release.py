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
socket directory.  The source-side stopped-release CLI separately performs
fixed, read-only ``systemctl show`` observations; this module never writes
units, changes service state, or accepts environment/secret payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.canonical_writer_release_contract import (
    DEFAULT_EXPORT_LIMIT,
    DEFAULT_RELEASE_BASE,
    EXPORTER_UNIT_NAME,
    GATEWAY_MODULE,
    GATEWAY_UNIT_NAME,
    INCOMPLETE_MARKER_NAME,
    RELEASE_MANIFEST_NAME,
    RELEASE_SCHEMA,
    TMPFILES_NAME,
    UNIT_BUNDLE_SCHEMA,
    WRITER_MODULE,
    WRITER_UNIT_NAME,
    ReleaseManifest,
    SystemdUnitBundle,
    TreeEntry,
    WriterOnlyUnitSpec,
    _absolute_normalized_path,
    _canonical_bytes,
    _CONTROL_RE,
    _is_within,
    _PYTHON_RE,
    _REVISION_RE,
    render_systemd_units,
)

BUILD_SCRATCH_NAME = ".release-build-scratch"
SCRATCH_PROVENANCE_NAME = "provenance.json"
SCRATCH_PROVENANCE_SCHEMA = "muncho-writer-release-scratch.v1"
BUILD_CONSTRAINTS_RELATIVE_PATH = Path("scripts/canary/writer-build-constraints.txt")
DEFAULT_UV_CACHE = Path("/var/cache/muncho-writer-release")
# Exact currently supported 3.11 security release.  The release builder never
# falls back to an older interpreter when this managed runtime is unavailable.
DEFAULT_PYTHON_VERSION = "3.11.15"
DEFAULT_UV_EXECUTABLE = Path("/usr/local/bin/uv")
DEFAULT_GIT_EXECUTABLE = Path("/usr/bin/git")
_MANIFEST_MODE = 0o400
_SEALED_DIRECTORY_MODE = 0o555
_SEALED_FILE_MODE = 0o444
_SEALED_EXECUTABLE_MODE = 0o555
_BUILD_DIRECTORY_MODE = 0o700
_BUILD_OWNER_UID = 0
_BUILD_OWNER_GID = 0
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
_VIRTUALENV_SITE_HOOK_NAME = "_virtualenv.pth"
_VIRTUALENV_SITE_HOOK_BYTES = b"import _virtualenv"
_PACKAGED_DISCORD_EDGE_MODULES = (
    Path("gateway/discord_edge_bootstrap.py"),
    Path("gateway/discord_edge_service.py"),
)
CANARY_BOOTSTRAP_SQL_RELATIVE_PATH = Path(
    "scripts/sql/canonical_writer_canary_bootstrap_v1.sql"
)
CANARY_BOOTSTRAP_RETIRE_SQL_RELATIVE_PATH = Path(
    "scripts/sql/canonical_writer_canary_bootstrap_retire_v1.sql"
)
_TRACKED_RELEASE_ARTIFACTS = (
    CANARY_BOOTSTRAP_SQL_RELATIVE_PATH,
    CANARY_BOOTSTRAP_RETIRE_SQL_RELATIVE_PATH,
)
_MAX_TRACKED_RELEASE_ARTIFACT_BYTES = 1024 * 1024

# The stopped publication boundary is intentionally narrower than the release
# builder API above.  Its CLI derives every path and tool from one exact commit
# SHA; operators cannot redirect it to a different checkout, repository,
# evidence directory, executable, service, host, or environment.
STOPPED_RELEASE_PLAN_SCHEMA = "muncho-canary-stopped-release-plan.v1"
STOPPED_RELEASE_RECEIPT_SCHEMA = "muncho-canary-stopped-release-publication.v1"
STOPPED_RELEASE_FAILURE_SCHEMA = "muncho-canary-stopped-release-failure.v1"
FORK_REPOSITORY = "https://github.com/lomliev/hermes-agent.git"
DEFAULT_SOURCE_BASE = Path("/opt/muncho-canary-source")
DEFAULT_EVIDENCE_BASE = Path("/var/lib/muncho-canary-release-evidence")
DEFAULT_HOST_RECEIPT_PATH = Path("/etc/muncho/full-canary/host-identity.json")
DEFAULT_SYSTEMCTL_EXECUTABLE = Path("/usr/bin/systemctl")
_EVIDENCE_DIRECTORY_MODE = 0o700
_HOST_RECEIPT_DIRECTORY_MODE = 0o755
_RECEIPT_MODE = 0o400
_MAX_RECEIPT_BYTES = 1024 * 1024
_MAX_SERVICE_OUTPUT_BYTES = 64 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ACTIVATION_PATHS = (
    Path("/etc/muncho/writer-activation/staged/writer.json"),
    Path("/etc/muncho/writer-activation/staged/gateway.yaml"),
    Path("/etc/muncho/writer-activation/staged/native-observation-plan.json"),
    Path("/etc/muncho/writer-activation/staged/activation-plan.json"),
    Path("/etc/muncho/writer-activation/staged/owner-approval.json"),
    Path("/etc/muncho/writer-activation/staged/external-iam-receipt.json"),
    Path("/etc/muncho/writer-activation/staged/muncho-canonical-writer.service"),
    Path("/etc/muncho/writer-activation/staged/hermes-cloud-gateway.service"),
    Path("/etc/muncho/writer-activation/native-observation-plan.json"),
    Path("/etc/muncho/writer-activation/activation-plan.json"),
    Path("/etc/muncho/writer-activation/deployment-manifest.json"),
    Path("/etc/systemd/system/muncho-canonical-writer.service"),
    Path("/etc/systemd/system/hermes-cloud-gateway.service"),
    Path("/etc/systemd/system/muncho-canonical-writer-export.service"),
    Path("/etc/tmpfiles.d/muncho-canonical-writer.conf"),
    Path("/etc/muncho-canonical-writer/writer.json"),
    Path("/etc/hermes/config.yaml"),
)
_STOPPED_SERVICE_UNITS = (
    "muncho-discord-egress.service",
    "muncho-canonical-writer.service",
    "hermes-cloud-gateway.service",
)
_SERVICE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
)
_HOST_OBSERVATION_FIELDS = frozenset({
    "project_id",
    "project_number",
    "zone",
    "instance_name",
    "instance_id",
    "service_account_email",
    "gce_identity_sha256",
    "machine_id_sha256",
    "hostname_sha256",
    "host_identity_sha256",
    "boot_id_sha256",
})


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
        return self.site_packages / "gateway" / "canonical_writer_gateway_bootstrap.py"

    def validate(self) -> None:
        if (
            not isinstance(self.revision, str)
            or _REVISION_RE.fullmatch(self.revision) is None
        ):
            raise ValueError("release revision must be exact lowercase 40-char SHA")
        if (
            not isinstance(self.python_version, str)
            or _PYTHON_RE.fullmatch(self.python_version) is None
        ):
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
            or len({
                self.build_scratch_root,
                self.build_project_root,
                self.wheel_output_root,
                self.wheel_artifact_root,
            })
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
                "-B",
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
        or item.st_uid != _BUILD_OWNER_UID
        or item.st_gid != _BUILD_OWNER_GID
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
            if (
                item.st_uid != 0
                or item.st_gid != 0
                or stat.S_IMODE(item.st_mode) & 0o022
            ):
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
                raise ValueError(
                    "release contains an unsupported directory before sealing"
                )
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
    raw = (
        _canonical_bytes({
            "schema": RELEASE_SCHEMA,
            "revision": spec.revision,
            "state": "incomplete",
        })
        + b"\n"
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
    return (
        _canonical_bytes({
            "schema": SCRATCH_PROVENANCE_SCHEMA,
            "revision": spec.revision,
            "source_root": str(spec.source_root),
            "source_device": int(source.st_dev),
            "source_inode": int(source.st_ino),
            "scratch_root": str(spec.build_scratch_root),
            "scratch_device": int(scratch_device),
            "scratch_inode": int(scratch_inode),
        })
        + b"\n"
    )


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
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(
            os,
            "O_CLOEXEC",
            0,
        )
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


def _copy_tracked_release_artifact(
    spec: ReleaseBuildSpec,
    relative_path: Path,
) -> Path:
    """Copy one revision-snapshot file into the manifest-bound release tree."""

    if (
        relative_path.is_absolute()
        or relative_path.as_posix() != str(relative_path)
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise ValueError("tracked release artifact path is invalid")
    source = spec.build_project_root / relative_path
    destination = spec.release_root / relative_path
    source_before = os.lstat(source)
    if (
        not stat.S_ISREG(source_before.st_mode)
        or stat.S_ISLNK(source_before.st_mode)
        or source_before.st_uid != _BUILD_OWNER_UID
        or source_before.st_gid != _BUILD_OWNER_GID
        or source_before.st_nlink != 1
        or stat.S_IMODE(source_before.st_mode) & 0o022
        or not 0 < source_before.st_size <= _MAX_TRACKED_RELEASE_ARTIFACT_BYTES
        or _list_xattrs(source)
    ):
        raise PermissionError("tracked release artifact source is not exact")

    current = spec.release_root
    for part in relative_path.parent.parts:
        current = current / part
        try:
            os.mkdir(current, _BUILD_DIRECTORY_MODE)
        except FileExistsError:
            _validate_root_directory(current, exact_mode=_BUILD_DIRECTORY_MODE)

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
        write_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, read_flags)
    destination_fd: int | None = None
    copied_digest = hashlib.sha256()
    copied_size = 0
    try:
        source_opened = os.fstat(source_fd)
        if (
            source_opened.st_dev,
            source_opened.st_ino,
            source_opened.st_mode,
            source_opened.st_nlink,
            source_opened.st_uid,
            source_opened.st_gid,
            source_opened.st_size,
            source_opened.st_mtime_ns,
            source_opened.st_ctime_ns,
        ) != (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_mode,
            source_before.st_nlink,
            source_before.st_uid,
            source_before.st_gid,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_ctime_ns,
        ):
            raise RuntimeError("tracked release artifact changed during open")
        destination_fd = os.open(destination, write_flags, 0o400)
        os.fchown(destination_fd, _BUILD_OWNER_UID, _BUILD_OWNER_GID)
        while chunk := os.read(source_fd, 64 * 1024):
            copied_size += len(chunk)
            if copied_size > _MAX_TRACKED_RELEASE_ARTIFACT_BYTES:
                raise RuntimeError("tracked release artifact became oversized")
            copied_digest.update(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(destination_fd, chunk[offset:])
                if written <= 0:
                    raise OSError("tracked release artifact copy made no progress")
                offset += written
        os.fchmod(destination_fd, _SEALED_FILE_MODE)
        os.fsync(destination_fd)
        source_after = os.fstat(source_fd)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)

    source_reachable = os.lstat(source)
    source_identity = (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_mode,
        source_before.st_nlink,
        source_before.st_uid,
        source_before.st_gid,
        source_before.st_size,
        source_before.st_mtime_ns,
        source_before.st_ctime_ns,
    )
    if source_identity != (
        source_after.st_dev,
        source_after.st_ino,
        source_after.st_mode,
        source_after.st_nlink,
        source_after.st_uid,
        source_after.st_gid,
        source_after.st_size,
        source_after.st_mtime_ns,
        source_after.st_ctime_ns,
    ) or source_identity != (
        source_reachable.st_dev,
        source_reachable.st_ino,
        source_reachable.st_mode,
        source_reachable.st_nlink,
        source_reachable.st_uid,
        source_reachable.st_gid,
        source_reachable.st_size,
        source_reachable.st_mtime_ns,
        source_reachable.st_ctime_ns,
    ):
        raise RuntimeError("tracked release artifact changed during copy")

    copied = os.lstat(destination)
    if (
        not stat.S_ISREG(copied.st_mode)
        or stat.S_ISLNK(copied.st_mode)
        or copied.st_uid != _BUILD_OWNER_UID
        or copied.st_gid != _BUILD_OWNER_GID
        or copied.st_nlink != 1
        or stat.S_IMODE(copied.st_mode) != _SEALED_FILE_MODE
        or copied.st_size != source_before.st_size
        or (copied.st_dev, copied.st_ino)
        == (source_before.st_dev, source_before.st_ino)
        or _hash_file(destination) != copied_digest.hexdigest()
    ):
        raise RuntimeError("tracked release artifact copy does not match source")
    return destination


def _copy_tracked_release_artifacts(spec: ReleaseBuildSpec) -> tuple[Path, ...]:
    return tuple(
        _copy_tracked_release_artifact(spec, relative_path)
        for relative_path in _TRACKED_RELEASE_ARTIFACTS
    )


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
    for relative_path in _PACKAGED_DISCORD_EDGE_MODULES:
        module_path = spec.site_packages / relative_path
        try:
            module_stat = os.lstat(module_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "release is missing a packaged Discord edge module"
            ) from exc
        if (
            not stat.S_ISREG(module_stat.st_mode)
            or stat.S_ISLNK(module_stat.st_mode)
            or module_stat.st_nlink != 1
            or module_stat.st_uid != site_stat.st_uid
            or module_stat.st_gid != site_stat.st_gid
            or stat.S_IMODE(module_stat.st_mode) != 0o644
            or module_stat.st_size == 0
        ):
            raise RuntimeError("release packaged Discord edge module is invalid")
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
        spec.site_packages / "scripts/discord_edge_bootstrap.py",
        spec.site_packages / "scripts/discord_edge_service.py",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden_scripts):
        raise RuntimeError("release contains a legacy scripts bootstrap")


def _remove_exact_virtualenv_site_hook(spec: ReleaseBuildSpec) -> bool:
    """Remove uv's exact build-time hook before sealing the runtime."""

    _validate_root_directory(spec.site_packages)
    hook = spec.site_packages / _VIRTUALENV_SITE_HOOK_NAME
    if not os.path.lexists(hook):
        return False
    before = os.lstat(hook)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != _BUILD_OWNER_UID
        or before.st_gid != _BUILD_OWNER_GID
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o644
        or before.st_size != len(_VIRTUALENV_SITE_HOOK_BYTES)
        or _list_xattrs(hook)
    ):
        raise RuntimeError("release virtualenv site hook identity is not exact")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(hook, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_gid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise RuntimeError("release virtualenv site hook changed during open")
        raw = os.read(descriptor, len(_VIRTUALENV_SITE_HOOK_BYTES) + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if raw != _VIRTUALENV_SITE_HOOK_BYTES or (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ):
        raise RuntimeError("release virtualenv site hook content drifted")
    current = os.lstat(hook)
    if (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ):
        raise RuntimeError("release virtualenv site hook changed before removal")
    hook.unlink()
    directory_fd = os.open(
        spec.site_packages,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if os.path.lexists(hook):
        raise RuntimeError("release virtualenv site hook removal did not persist")
    return True


def _require_root_owned_regular_executable(
    item: os.stat_result,
    *,
    label: str,
) -> None:
    if (
        not stat.S_ISREG(item.st_mode)
        or item.st_uid != _BUILD_OWNER_UID
        or item.st_gid != _BUILD_OWNER_GID
        or item.st_nlink != 1
        or not stat.S_IMODE(item.st_mode) & 0o111
        or stat.S_IMODE(item.st_mode) & 0o022
    ):
        raise RuntimeError(f"{label} is not an independent root-owned executable")


def _materialize_copied_interpreter(
    spec: ReleaseBuildSpec,
    managed_python: Path,
) -> str:
    """Atomically replace uv's exact managed-Python symlink with a real copy."""

    managed_python = _absolute_normalized_path(
        managed_python,
        "managed interpreter copy source",
    )
    if not _is_within(managed_python, spec.managed_python_root):
        raise RuntimeError("interpreter copy source is outside managed Python")
    if managed_python.resolve(strict=True) != managed_python:
        raise RuntimeError("managed interpreter copy source is not fully resolved")
    parent_stat = os.lstat(spec.interpreter.parent)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or stat.S_ISLNK(parent_stat.st_mode)
        or parent_stat.st_uid != _BUILD_OWNER_UID
        or parent_stat.st_gid != _BUILD_OWNER_GID
        or stat.S_IMODE(parent_stat.st_mode) & 0o022
    ):
        raise RuntimeError("release interpreter directory is not root-controlled")

    source_stat = os.lstat(managed_python)
    _require_root_owned_regular_executable(
        source_stat,
        label="managed interpreter copy source",
    )
    source_digest = _hash_file(managed_python)
    temp_path = spec.interpreter.parent / ".python.materialize"
    temp_fd: int | None = None
    try:
        destination_stat = os.lstat(spec.interpreter)
        if stat.S_ISREG(destination_stat.st_mode):
            _require_root_owned_regular_executable(
                destination_stat,
                label="release interpreter",
            )
            if (destination_stat.st_dev, destination_stat.st_ino) == (
                source_stat.st_dev,
                source_stat.st_ino,
            ):
                raise RuntimeError("release interpreter is hard-linked to managed Python")
            if _hash_file(spec.interpreter) != source_digest:
                raise RuntimeError("release interpreter content is not managed Python")
            return source_digest
        if not stat.S_ISLNK(destination_stat.st_mode):
            raise RuntimeError("release interpreter path contains an unexpected collision")
        destination_target_raw = os.readlink(spec.interpreter)
        destination_target = _absolute_normalized_path(
            destination_target_raw,
            "release interpreter symlink target",
        )
        if (
            not _is_within(destination_target, spec.managed_python_root)
            or destination_target.resolve(strict=True) != managed_python
        ):
            raise RuntimeError("release interpreter symlink target is not managed Python")

        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temp_fd = os.open(temp_path, create_flags, 0o500)
        copied_digest = hashlib.sha256()
        with managed_python.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                copied_digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(temp_fd, view)
                    if written <= 0:
                        raise RuntimeError("release interpreter copy made no progress")
                    view = view[written:]
        os.fchmod(temp_fd, _SEALED_EXECUTABLE_MODE)
        os.fchown(temp_fd, _BUILD_OWNER_UID, _BUILD_OWNER_GID)
        os.fsync(temp_fd)
        copied_stat = os.fstat(temp_fd)
        _require_root_owned_regular_executable(
            copied_stat,
            label="materialized release interpreter",
        )
        if (
            (copied_stat.st_dev, copied_stat.st_ino)
            == (source_stat.st_dev, source_stat.st_ino)
            or copied_digest.hexdigest() != source_digest
        ):
            raise RuntimeError("materialized release interpreter provenance is invalid")
        source_after = os.lstat(managed_python)
        if (
            source_after.st_dev,
            source_after.st_ino,
            source_after.st_size,
            source_after.st_mtime_ns,
            source_after.st_ctime_ns,
        ) != (
            source_stat.st_dev,
            source_stat.st_ino,
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
        ):
            raise RuntimeError("managed interpreter changed during materialization")
        current_destination = os.lstat(spec.interpreter)
        if (
            not stat.S_ISLNK(current_destination.st_mode)
            or (current_destination.st_dev, current_destination.st_ino)
            != (destination_stat.st_dev, destination_stat.st_ino)
            or os.readlink(spec.interpreter) != destination_target_raw
            or destination_target.resolve(strict=True) != managed_python
        ):
            raise RuntimeError("release interpreter symlink changed during materialization")
        os.close(temp_fd)
        temp_fd = None
        os.replace(temp_path, spec.interpreter)
        parent_fd = os.open(
            spec.interpreter.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

        final_stat = os.lstat(spec.interpreter)
        _require_root_owned_regular_executable(
            final_stat,
            label="release interpreter",
        )
        if (
            (final_stat.st_dev, final_stat.st_ino)
            == (source_stat.st_dev, source_stat.st_ino)
            or final_stat.st_nlink != 1
        ):
            raise RuntimeError("release interpreter is not an independent copy")
        if _hash_file(spec.interpreter) != source_digest:
            raise RuntimeError("release interpreter digest changed after activation")
        return source_digest
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if os.path.lexists(temp_path):
            os.unlink(temp_path)


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
        _copy_tracked_release_artifacts(spec)
        install_steps = install_commands(spec, managed_python)
        _run_checked(
            install_steps[0],
            runner=runner,
            label="release install step 1",
        )
        _materialize_copied_interpreter(spec, managed_python)
        for index, command in enumerate(install_steps[1:], start=2):
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
        _materialize_copied_interpreter(spec, managed_python)
        _remove_exact_virtualenv_site_hook(spec)
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


HostObserver = Callable[[], Mapping[str, str]]
HostReceiptCollector = Callable[[int], Mapping[str, Any]]
PathExists = Callable[[Path], bool]
ReleaseBuilder = Callable[..., ReleaseManifest]
Clock = Callable[[], float]


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stopped_release_spec(revision: str) -> ReleaseBuildSpec:
    """Derive the only source, release, interpreter, and tool paths allowed."""

    return ReleaseBuildSpec(
        revision=revision,
        source_root=DEFAULT_SOURCE_BASE / revision,
        release_base=DEFAULT_RELEASE_BASE,
        python_version=DEFAULT_PYTHON_VERSION,
        uv_executable=DEFAULT_UV_EXECUTABLE,
        git_executable=DEFAULT_GIT_EXECUTABLE,
        uv_cache_dir=DEFAULT_UV_CACHE,
    )


def _stopped_observation_environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "HOME": "/nonexistent",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": (
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ),
            }.items()
        )
    )


def _source_identity_commands(spec: ReleaseBuildSpec) -> tuple[BuildCommand, ...]:
    environment = _clean_environment(spec)
    git = str(spec.git_executable)
    source = str(spec.source_root)
    return (
        BuildCommand(
            (git, "-C", source, "config", "--get", "remote.origin.url"),
            env=environment,
        ),
        BuildCommand(
            (git, "-C", source, "rev-parse", "--verify", "HEAD^{tree}"),
            env=environment,
        ),
    )


def _bounded_command_stdout(
    command: BuildCommand,
    *,
    runner: Runner,
    label: str,
    maximum_bytes: int,
) -> str:
    completed = runner(command)
    if (
        not isinstance(completed, subprocess.CompletedProcess)
        or not isinstance(completed.stdout, str)
        or not isinstance(completed.stderr, str)
    ):
        raise TypeError(f"{label} returned an invalid result")
    try:
        stdout_bytes = completed.stdout.encode("utf-8", errors="strict")
        stderr_bytes = completed.stderr.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"{label} output is not UTF-8") from exc
    if len(stdout_bytes) > maximum_bytes or len(stderr_bytes) > maximum_bytes:
        raise RuntimeError(f"{label} output exceeds its bound")
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed")
    if completed.stderr:
        raise RuntimeError(f"{label} emitted unexpected stderr")
    return completed.stdout


def _collect_source_identity(
    spec: ReleaseBuildSpec,
    *,
    runner: Runner,
) -> dict[str, str]:
    spec.validate()
    expected_source = DEFAULT_SOURCE_BASE / spec.revision
    if spec.source_root != expected_source:
        raise RuntimeError("stopped-release source path is not exact")
    _validate_root_parent_chain(spec.release_base)
    _validate_root_parent_chain(spec.uv_cache_dir)
    _validate_root_executable(spec.uv_executable)
    _validate_root_executable(spec.git_executable)
    _validate_root_source_tree(spec.source_root)
    verify_clean_checkout(spec, runner=runner)
    origin_command, tree_command = _source_identity_commands(spec)
    origin_raw = _bounded_command_stdout(
        origin_command,
        runner=runner,
        label="source repository verification",
        maximum_bytes=4096,
    )
    if origin_raw != FORK_REPOSITORY + "\n":
        raise RuntimeError("stopped-release source repository is not the fixed fork")
    tree_raw = _bounded_command_stdout(
        tree_command,
        runner=runner,
        label="source tree verification",
        maximum_bytes=4096,
    )
    if not tree_raw.endswith("\n") or tree_raw.count("\n") != 1:
        raise RuntimeError("stopped-release source tree identity is invalid")
    tree_sha = tree_raw[:-1]
    if _REVISION_RE.fullmatch(tree_sha) is None:
        raise RuntimeError("stopped-release source tree identity is invalid")
    return {
        "repository": FORK_REPOSITORY,
        "root": str(spec.source_root),
        "head_sha": spec.revision,
        "tree_sha": tree_sha,
    }


def _default_host_observer() -> Mapping[str, str]:
    # Import lazily: the release builder remains usable without importing the
    # broader full-canary runtime until this dedicated-host gate is requested.
    from gateway.canonical_canary_host_identity import (
        _observe_dedicated_canary_host,
    )

    return _observe_dedicated_canary_host()


def _validate_host_observation(value: Mapping[str, str]) -> dict[str, str]:
    from gateway.canonical_canary_host_identity import (
        _dedicated_canary_gce_identity,
    )

    if not isinstance(value, Mapping) or set(value) != _HOST_OBSERVATION_FIELDS:
        raise RuntimeError("dedicated canary host observation is incomplete")
    observed: dict[str, str] = {}
    for name in sorted(_HOST_OBSERVATION_FIELDS):
        item = value[name]
        if not isinstance(item, str) or not item or _CONTROL_RE.search(item):
            raise RuntimeError("dedicated canary host observation is invalid")
        observed[name] = item
    expected_gce = _dedicated_canary_gce_identity()
    if any(observed[name] != expected for name, expected in expected_gce.items()):
        raise RuntimeError("dedicated canary host observation is not the fixed VM")
    if observed["gce_identity_sha256"] != _sha256_json(expected_gce):
        raise RuntimeError("dedicated canary GCE digest is invalid")
    for name in _HOST_OBSERVATION_FIELDS:
        if name.endswith("_sha256") and _SHA256_RE.fullmatch(observed[name]) is None:
            raise RuntimeError("dedicated canary host digest is invalid")
    return observed


def _collect_activation_inventory(
    *,
    path_exists: PathExists = os.path.lexists,
) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for path in _ACTIVATION_PATHS:
        if path_exists(path):
            raise RuntimeError("stopped-release activation path is not fresh")
        inventory.append({"path": str(path), "state": "absent"})
    return inventory


def _service_observation_command(unit: str) -> BuildCommand:
    if unit not in _STOPPED_SERVICE_UNITS:
        raise ValueError("stopped-release service is not allow-listed")
    return BuildCommand(
        (
            str(DEFAULT_SYSTEMCTL_EXECUTABLE),
            "show",
            "--no-pager",
            *(f"--property={name}" for name in _SERVICE_PROPERTIES),
            unit,
        ),
        env=_stopped_observation_environment(),
    )


def _parse_service_observation(unit: str, raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or any(
        character in raw for character in ("\x00", "\r")
    ):
        raise RuntimeError("stopped-release service output is invalid")
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or "=" not in line:
            raise RuntimeError("stopped-release service output is malformed")
        name, item = line.split("=", 1)
        if name not in _SERVICE_PROPERTIES or name in values:
            raise RuntimeError("stopped-release service output has unexpected fields")
        if _CONTROL_RE.search(item) is not None:
            raise RuntimeError("stopped-release service value is invalid")
        values[name] = item
    if set(values) != set(_SERVICE_PROPERTIES):
        raise RuntimeError("stopped-release service output is incomplete")
    if not re.fullmatch(r"0|[1-9][0-9]*", values["MainPID"]):
        raise RuntimeError("stopped-release service PID is invalid")

    absent = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "",
        "MainPID": "0",
        "FragmentPath": "",
        "DropInPaths": "",
    }
    disabled = {
        "LoadState": "loaded",
        "ActiveState": "inactive",
        "SubState": "dead",
        "UnitFileState": "disabled",
        "MainPID": "0",
        "FragmentPath": f"/etc/systemd/system/{unit}",
        "DropInPaths": "",
    }
    if values == absent:
        state = "absent"
    elif values == disabled:
        state = "disabled_inactive"
    else:
        raise RuntimeError("stopped-release service is not safely stopped")
    return {
        "unit": unit,
        "state": state,
        "properties": {name: values[name] for name in _SERVICE_PROPERTIES},
    }


def _collect_service_states(*, runner: Runner = _runner) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for unit in _STOPPED_SERVICE_UNITS:
        command = _service_observation_command(unit)
        raw = _bounded_command_stdout(
            command,
            runner=runner,
            label="stopped-release service observation",
            maximum_bytes=_MAX_SERVICE_OUTPUT_BYTES,
        )
        states.append(_parse_service_observation(unit, raw))
    return states


def plan_stopped_release(
    revision: str,
    *,
    runner: Runner = _runner,
    host_observer: HostObserver = _default_host_observer,
    path_exists: PathExists = os.path.lexists,
) -> dict[str, Any]:
    """Create a deterministic, read-only plan for one stopped release."""

    spec = _stopped_release_spec(revision)
    spec.validate()
    _validate_root_executable(DEFAULT_SYSTEMCTL_EXECUTABLE)
    source = _collect_source_identity(spec, runner=runner)
    host = _validate_host_observation(host_observer())
    inventory = _collect_activation_inventory(path_exists=path_exists)
    service_states = _collect_service_states(runner=runner)
    unsigned: dict[str, Any] = {
        "schema": STOPPED_RELEASE_PLAN_SCHEMA,
        "revision": spec.revision,
        "source": source,
        "release_root": str(spec.release_root),
        "release_manifest_path": str(spec.release_root / RELEASE_MANIFEST_NAME),
        "evidence_receipt_path": str(
            DEFAULT_EVIDENCE_BASE / spec.revision / "stopped-release-publication.json"
        ),
        "host_identity_receipt_path": str(DEFAULT_HOST_RECEIPT_PATH),
        "python_version": spec.python_version,
        "interpreter": str(spec.interpreter),
        "tools": {
            "git": str(spec.git_executable),
            "systemctl": str(DEFAULT_SYSTEMCTL_EXECUTABLE),
            "uv": str(spec.uv_executable),
            "uv_cache": str(spec.uv_cache_dir),
        },
        "dedicated_host": host,
        "activation_inventory": inventory,
        "service_states": service_states,
    }
    return {**unsigned, "plan_sha256": _sha256_json(unsigned)}


def _stat_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _read_stable_root_file(
    path: Path,
    *,
    maximum_bytes: int,
    exact_mode: int,
) -> bytes:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != _BUILD_OWNER_UID
        or before.st_gid != _BUILD_OWNER_GID
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != exact_mode
        or not 0 <= before.st_size <= maximum_bytes
        or _list_xattrs(path)
    ):
        raise RuntimeError("stopped-release evidence file is not exact")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise RuntimeError("stopped-release evidence changed during open")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise RuntimeError("stopped-release evidence exceeds its bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.lstat(path)
    if (
        size != before.st_size
        or _stat_identity(after) != _stat_identity(before)
        or _stat_identity(reachable) != _stat_identity(before)
    ):
        raise RuntimeError("stopped-release evidence changed during read")
    return b"".join(chunks)


def _hash_stable_root_file(
    path: Path,
    *,
    maximum_bytes: int,
    allowed_modes: frozenset[int],
) -> str:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != _BUILD_OWNER_UID
        or before.st_gid != _BUILD_OWNER_GID
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) not in allowed_modes
        or not 0 < before.st_size <= maximum_bytes
        or _list_xattrs(path)
    ):
        raise RuntimeError("stopped-release artifact file is not exact")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise RuntimeError("stopped-release artifact changed during open")
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > maximum_bytes:
                raise RuntimeError("stopped-release artifact exceeds its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.lstat(path)
    if (
        size != before.st_size
        or _stat_identity(after) != _stat_identity(before)
        or _stat_identity(reachable) != _stat_identity(before)
    ):
        raise RuntimeError("stopped-release artifact changed during hashing")
    return digest.hexdigest()


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("stopped-release JSON contains duplicate fields")
        value[name] = item
    return value


def _decode_canonical_mapping(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise RuntimeError(f"{label} does not have canonical framing")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def _decode_unframed_canonical_mapping(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def _validate_sealed_release_tree(spec: ReleaseBuildSpec) -> None:
    _validate_root_directory(spec.release_root, exact_mode=_SEALED_DIRECTORY_MODE)
    if os.path.lexists(spec.release_root / INCOMPLETE_MARKER_NAME):
        raise RuntimeError("stopped-release artifact is incomplete")
    if os.path.lexists(spec.release_root / BUILD_SCRATCH_NAME):
        raise RuntimeError("stopped-release artifact retains build scratch")
    for current, directories, files in os.walk(
        spec.release_root,
        topdown=True,
        followlinks=False,
    ):
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            path = Path(current) / name
            item = os.lstat(path)
            if (
                item.st_uid != _BUILD_OWNER_UID
                or item.st_gid != _BUILD_OWNER_GID
                or _list_xattrs(path)
            ):
                raise RuntimeError("stopped-release tree is not root-controlled")
            if stat.S_ISLNK(item.st_mode):
                if not _is_within(
                    path.resolve(strict=True),
                    spec.release_root.resolve(strict=True),
                ):
                    raise RuntimeError("stopped-release symlink escapes artifact")
            elif stat.S_ISDIR(item.st_mode):
                if stat.S_IMODE(item.st_mode) != _SEALED_DIRECTORY_MODE:
                    raise RuntimeError("stopped-release directory is not sealed")
            elif stat.S_ISREG(item.st_mode):
                expected_modes = (
                    frozenset({_MANIFEST_MODE})
                    if path == spec.release_root / RELEASE_MANIFEST_NAME
                    else frozenset({_SEALED_FILE_MODE, _SEALED_EXECUTABLE_MODE})
                )
                if (
                    item.st_nlink != 1
                    or stat.S_IMODE(item.st_mode) not in expected_modes
                ):
                    raise RuntimeError("stopped-release file is not sealed")
            else:
                raise RuntimeError("stopped-release tree contains a special file")


def _validate_completed_release(spec: ReleaseBuildSpec) -> dict[str, Any]:
    _validate_sealed_release_tree(spec)
    reconstructed = create_release_manifest(spec)
    manifest_path = spec.release_root / RELEASE_MANIFEST_NAME
    manifest_raw = _read_stable_root_file(
        manifest_path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        exact_mode=_MANIFEST_MODE,
    )
    manifest_value = _decode_canonical_mapping(
        manifest_raw,
        label="stopped-release manifest",
    )
    if manifest_value != reconstructed.to_mapping():
        raise RuntimeError("stopped-release manifest does not match the sealed tree")

    artifact_stat = os.lstat(spec.wheel_artifact_root)
    if (
        not stat.S_ISDIR(artifact_stat.st_mode)
        or stat.S_ISLNK(artifact_stat.st_mode)
        or stat.S_IMODE(artifact_stat.st_mode) != _SEALED_DIRECTORY_MODE
    ):
        raise RuntimeError("stopped-release wheel directory is invalid")
    artifact_names = sorted(os.listdir(spec.wheel_artifact_root))
    if (
        len(artifact_names) != 1
        or _SAFE_WHEEL_NAME_RE.fullmatch(artifact_names[0]) is None
    ):
        raise RuntimeError("stopped-release must retain exactly one safe wheel")
    wheel_path = spec.wheel_artifact_root / artifact_names[0]
    wheel_sha256 = _hash_stable_root_file(
        wheel_path,
        maximum_bytes=256 * 1024 * 1024,
        allowed_modes=frozenset({_SEALED_FILE_MODE}),
    )
    interpreter_sha256 = _hash_stable_root_file(
        spec.interpreter,
        maximum_bytes=256 * 1024 * 1024,
        allowed_modes=frozenset({_SEALED_EXECUTABLE_MODE}),
    )
    return {
        "release_root": str(spec.release_root),
        "release_manifest_path": str(manifest_path),
        "release_manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "release_artifact_sha256": reconstructed.artifact_sha256,
        "interpreter": str(spec.interpreter),
        "interpreter_sha256": interpreter_sha256,
        "python_version": spec.python_version,
        "retained_wheel_path": str(wheel_path),
        "retained_wheel_sha256": wheel_sha256,
        "build_constraints_sha256": hashlib.sha256(
            _PINNED_BUILD_CONSTRAINTS
        ).hexdigest(),
    }


def _evidence_receipt_path(revision: str) -> Path:
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("stopped-release revision is invalid")
    return DEFAULT_EVIDENCE_BASE / revision / "stopped-release-publication.json"


def _validate_evidence_directory(path: Path) -> None:
    _validate_root_directory(path, exact_mode=_EVIDENCE_DIRECTORY_MODE)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_evidence_namespace(revision: str, *, receipt_exists: bool) -> None:
    receipt_path = _evidence_receipt_path(revision)
    revision_root = receipt_path.parent
    _validate_root_parent_chain(DEFAULT_EVIDENCE_BASE.parent)
    if not os.path.lexists(DEFAULT_EVIDENCE_BASE):
        if receipt_exists or os.path.lexists(revision_root):
            raise RuntimeError("stopped-release evidence namespace is inconsistent")
        return
    _validate_evidence_directory(DEFAULT_EVIDENCE_BASE)
    if receipt_exists:
        _validate_evidence_directory(revision_root)
        if sorted(os.listdir(revision_root)) != [receipt_path.name]:
            raise RuntimeError("stopped-release evidence revision has extra entries")
    elif os.path.lexists(revision_root):
        raise RuntimeError("stopped-release evidence revision collision exists")


def _create_root_evidence_directory(path: Path) -> None:
    try:
        os.mkdir(path, _EVIDENCE_DIRECTORY_MODE)
    except FileExistsError as exc:
        raise RuntimeError(
            "stopped-release evidence directory collision exists"
        ) from exc
    os.chown(
        path,
        _BUILD_OWNER_UID,
        _BUILD_OWNER_GID,
        follow_symlinks=False,
    )
    os.chmod(path, _EVIDENCE_DIRECTORY_MODE, follow_symlinks=False)
    _validate_evidence_directory(path)
    _fsync_directory(path.parent)


def _create_evidence_namespace(revision: str) -> Path:
    receipt_path = _evidence_receipt_path(revision)
    _validate_root_parent_chain(DEFAULT_EVIDENCE_BASE.parent)
    if os.path.lexists(DEFAULT_EVIDENCE_BASE):
        _validate_evidence_directory(DEFAULT_EVIDENCE_BASE)
    else:
        _create_root_evidence_directory(DEFAULT_EVIDENCE_BASE)
    _create_root_evidence_directory(receipt_path.parent)
    return receipt_path


def _receipt_unsigned(
    plan: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    service_state_after: Sequence[Mapping[str, Any]],
    receipt_path: Path,
    created_at_unix: int,
) -> dict[str, Any]:
    if type(created_at_unix) is not int or created_at_unix < 0:
        raise ValueError("stopped-release receipt time is invalid")
    return {
        "schema": STOPPED_RELEASE_RECEIPT_SCHEMA,
        "ok": True,
        "state": "published_services_stopped",
        "release_revision": plan["revision"],
        "plan_sha256": plan["plan_sha256"],
        "source": plan["source"],
        "dedicated_host": plan["dedicated_host"],
        "activation_inventory": plan["activation_inventory"],
        "service_state_before": plan["service_states"],
        "service_state_after": list(service_state_after),
        "services_stopped_and_disabled": True,
        "tools": plan["tools"],
        **dict(release),
        "receipt_path": str(receipt_path),
        "created_at_unix": created_at_unix,
    }


def _create_receipt(
    plan: Mapping[str, Any],
    release: Mapping[str, Any],
    *,
    service_state_after: Sequence[Mapping[str, Any]],
    receipt_path: Path,
    created_at_unix: int,
) -> dict[str, Any]:
    unsigned = _receipt_unsigned(
        plan,
        release,
        service_state_after=service_state_after,
        receipt_path=receipt_path,
        created_at_unix=created_at_unix,
    )
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _write_receipt_no_replace(path: Path, receipt: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(receipt) + b"\n"
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise RuntimeError("stopped-release receipt exceeds its bound")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchown(descriptor, _BUILD_OWNER_UID, _BUILD_OWNER_GID)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("stopped-release receipt write made no progress")
            offset += written
        os.fchmod(descriptor, _RECEIPT_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_host_receipt_no_replace(path: Path, receipt: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(receipt)
    if len(raw) > 16 * 1024:
        raise RuntimeError("full-canary host identity receipt exceeds its bound")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchown(descriptor, _BUILD_OWNER_UID, _BUILD_OWNER_GID)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("full-canary host identity write made no progress")
            offset += written
        os.fchmod(descriptor, _RECEIPT_MODE)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _default_host_receipt_collector(observed_at_unix: int) -> Mapping[str, Any]:
    from gateway.canonical_canary_host_identity import (
        collect_dedicated_canary_host_identity_receipt,
    )

    return collect_dedicated_canary_host_identity_receipt(
        observed_at_unix=observed_at_unix
    )


def _validate_host_receipt_mapping(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    from gateway.canonical_canary_host_identity import (
        FULL_CANARY_HOST_IDENTITY_SCHEMA,
    )

    expected_fields = _HOST_OBSERVATION_FIELDS | {
        "schema",
        "collector_authority",
        "observed_at_unix",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RuntimeError("full-canary host identity receipt is incomplete")
    receipt = dict(value)
    observed_at_unix = receipt["observed_at_unix"]
    if (
        receipt["schema"] != FULL_CANARY_HOST_IDENTITY_SCHEMA
        or receipt["collector_authority"] != "trusted_root_read_only_host_collector"
        or type(observed_at_unix) is not int
        or observed_at_unix < 0
        or any(
            receipt[name] != plan["dedicated_host"][name]
            for name in _HOST_OBSERVATION_FIELDS
        )
    ):
        raise RuntimeError("full-canary host identity receipt is stale or invalid")
    unsigned = {
        name: item for name, item in receipt.items() if name != "receipt_sha256"
    }
    if (
        not isinstance(receipt["receipt_sha256"], str)
        or _SHA256_RE.fullmatch(receipt["receipt_sha256"]) is None
        or receipt["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise RuntimeError("full-canary host identity receipt digest is invalid")
    return receipt


def _validate_host_receipt_parent() -> None:
    _validate_root_parent_chain(DEFAULT_HOST_RECEIPT_PATH.parent.parent)
    _validate_root_directory(
        DEFAULT_HOST_RECEIPT_PATH.parent,
        exact_mode=_HOST_RECEIPT_DIRECTORY_MODE,
    )


def _create_host_receipt_parent() -> None:
    parent = DEFAULT_HOST_RECEIPT_PATH.parent
    _validate_root_parent_chain(parent.parent)
    if os.path.lexists(parent):
        _validate_host_receipt_parent()
        return
    try:
        os.mkdir(parent, _HOST_RECEIPT_DIRECTORY_MODE)
    except FileExistsError as exc:
        raise RuntimeError(
            "full-canary host receipt directory collision exists"
        ) from exc
    os.chown(
        parent,
        _BUILD_OWNER_UID,
        _BUILD_OWNER_GID,
        follow_symlinks=False,
    )
    os.chmod(parent, _HOST_RECEIPT_DIRECTORY_MODE, follow_symlinks=False)
    _validate_host_receipt_parent()
    _fsync_directory(parent.parent)


def _read_host_receipt_binding(plan: Mapping[str, Any]) -> dict[str, str]:
    _validate_host_receipt_parent()
    raw = _read_stable_root_file(
        DEFAULT_HOST_RECEIPT_PATH,
        maximum_bytes=16 * 1024,
        exact_mode=_RECEIPT_MODE,
    )
    value = _decode_unframed_canonical_mapping(
        raw,
        label="full-canary host identity receipt",
    )
    receipt = _validate_host_receipt_mapping(value, plan=plan)
    return {
        "host_identity_receipt_path": str(DEFAULT_HOST_RECEIPT_PATH),
        "host_identity_receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "host_identity_receipt_sha256": receipt["receipt_sha256"],
    }


def _publish_or_validate_host_receipt(
    plan: Mapping[str, Any],
    *,
    observed_at_unix: int,
    collector: HostReceiptCollector,
) -> dict[str, str]:
    if os.path.lexists(DEFAULT_HOST_RECEIPT_PATH):
        return _read_host_receipt_binding(plan)
    collected = _validate_host_receipt_mapping(
        collector(observed_at_unix),
        plan=plan,
    )
    _create_host_receipt_parent()
    _write_host_receipt_no_replace(DEFAULT_HOST_RECEIPT_PATH, collected)
    return _read_host_receipt_binding(plan)


def _preflight_host_receipt_namespace(plan: Mapping[str, Any]) -> None:
    """Reject deterministic host-receipt collisions before release creation."""

    if os.path.lexists(DEFAULT_HOST_RECEIPT_PATH):
        _read_host_receipt_binding(plan)
    elif os.path.lexists(DEFAULT_HOST_RECEIPT_PATH.parent):
        _validate_host_receipt_parent()
    else:
        _validate_root_parent_chain(DEFAULT_HOST_RECEIPT_PATH.parent.parent)


def _validate_existing_receipt(
    path: Path,
    *,
    plan: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _read_stable_root_file(
        path,
        maximum_bytes=_MAX_RECEIPT_BYTES,
        exact_mode=_RECEIPT_MODE,
    )
    receipt = _decode_canonical_mapping(raw, label="stopped-release receipt")
    created_at_unix = receipt.get("created_at_unix")
    if type(created_at_unix) is not int or created_at_unix < 0:
        raise RuntimeError("stopped-release receipt time is invalid")
    expected = _create_receipt(
        plan,
        release,
        service_state_after=plan["service_states"],
        receipt_path=path,
        created_at_unix=created_at_unix,
    )
    if receipt != expected:
        raise RuntimeError("stopped-release receipt binding is invalid")
    return receipt


def apply_stopped_release(
    revision: str,
    approved_plan_sha256: str,
    *,
    runner: Runner = _runner,
    host_observer: HostObserver = _default_host_observer,
    path_exists: PathExists = os.path.lexists,
    release_builder: ReleaseBuilder = build_release,
    host_receipt_collector: HostReceiptCollector = _default_host_receipt_collector,
    clock: Clock = time.time,
) -> dict[str, Any]:
    """Publish one exact sealed release while all canary services stay stopped."""

    _require_root_linux()
    if (
        not isinstance(approved_plan_sha256, str)
        or _SHA256_RE.fullmatch(approved_plan_sha256) is None
    ):
        raise ValueError("stopped-release approved plan digest is invalid")
    plan = plan_stopped_release(
        revision,
        runner=runner,
        host_observer=host_observer,
        path_exists=path_exists,
    )
    if plan["plan_sha256"] != approved_plan_sha256:
        raise PermissionError("stopped-release approved plan digest does not match")

    spec = _stopped_release_spec(revision)
    receipt_path = _evidence_receipt_path(revision)
    receipt_exists = os.path.lexists(receipt_path)
    release_exists = os.path.lexists(spec.release_root)
    _validate_evidence_namespace(revision, receipt_exists=receipt_exists)

    if receipt_exists:
        if not release_exists:
            raise RuntimeError("stopped-release receipt exists without its release")
        release = _validate_completed_release(spec)
        if not os.path.lexists(DEFAULT_HOST_RECEIPT_PATH):
            raise RuntimeError("stopped-release receipt lacks its host receipt")
        host_receipt = _read_host_receipt_binding(plan)
        return _validate_existing_receipt(
            receipt_path,
            plan=plan,
            release={**release, **host_receipt},
        )
    if release_exists:
        raise RuntimeError("stopped-release release collision has no receipt")
    _preflight_host_receipt_namespace(plan)

    built = release_builder(spec, runner=runner)
    if not isinstance(built, ReleaseManifest):
        raise TypeError("stopped-release builder returned an invalid manifest")
    release = _validate_completed_release(spec)
    if built.artifact_sha256 != release["release_artifact_sha256"]:
        raise RuntimeError("stopped-release builder result does not match artifact")
    created_at_unix = int(clock())
    if created_at_unix < 0:
        raise ValueError("stopped-release receipt time is invalid")
    host_receipt = _publish_or_validate_host_receipt(
        plan,
        observed_at_unix=created_at_unix,
        collector=host_receipt_collector,
    )
    post_build_plan = plan_stopped_release(
        revision,
        runner=runner,
        host_observer=host_observer,
        path_exists=path_exists,
    )
    if post_build_plan != plan:
        raise RuntimeError("stopped-release state drifted during build")
    release_with_host = {**release, **host_receipt}

    receipt_path = _create_evidence_namespace(revision)
    receipt = _create_receipt(
        plan,
        release_with_host,
        service_state_after=post_build_plan["service_states"],
        receipt_path=receipt_path,
        created_at_unix=created_at_unix,
    )
    _write_receipt_no_replace(receipt_path, receipt)
    return _validate_existing_receipt(
        receipt_path,
        plan=post_build_plan,
        release=release_with_host,
    )


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ValueError("invalid stopped-release CLI arguments")


class _StoreOnce(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        if getattr(namespace, self.dest, None) is not None:
            raise ValueError("stopped-release CLI option was repeated")
        setattr(namespace, self.dest, values)


def _exact_revision(value: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid revision")
    return value


def _exact_sha256(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid digest")
    return value


def _cli_parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(
        description="Publish one fixed stopped Muncho canary release",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_CanonicalArgumentParser,
    )
    plan = subparsers.add_parser(
        "plan",
        allow_abbrev=False,
    )
    plan.add_argument(
        "--revision",
        required=True,
        default=None,
        type=_exact_revision,
        action=_StoreOnce,
    )
    apply = subparsers.add_parser(
        "apply",
        allow_abbrev=False,
    )
    apply.add_argument(
        "--revision",
        required=True,
        default=None,
        type=_exact_revision,
        action=_StoreOnce,
    )
    apply.add_argument(
        "--approved-plan-sha256",
        required=True,
        default=None,
        type=_exact_sha256,
        action=_StoreOnce,
    )
    return parser


def _emit_canonical(value: Mapping[str, Any]) -> None:
    print(_canonical_bytes(value).decode("utf-8", errors="strict"))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _cli_parser().parse_args(argv)
        if arguments.command == "plan":
            result = plan_stopped_release(arguments.revision)
        elif arguments.command == "apply":
            result = apply_stopped_release(
                arguments.revision,
                arguments.approved_plan_sha256,
            )
        else:  # pragma: no cover - argparse enforces the command set.
            raise RuntimeError("unsupported stopped-release command")
        _emit_canonical(result)
        return 0
    except Exception as exc:
        _emit_canonical({
            "schema": STOPPED_RELEASE_FAILURE_SCHEMA,
            "ok": False,
            "error_code": "stopped_release_failed",
            "error_type": type(exc).__name__,
        })
        return 2


__all__ = [
    "BUILD_CONSTRAINTS_RELATIVE_PATH",
    "BUILD_SCRATCH_NAME",
    "CANARY_BOOTSTRAP_RETIRE_SQL_RELATIVE_PATH",
    "CANARY_BOOTSTRAP_SQL_RELATIVE_PATH",
    "BuildCommand",
    "GATEWAY_MODULE",
    "GATEWAY_UNIT_NAME",
    "EXPORTER_UNIT_NAME",
    "DEFAULT_EXPORT_LIMIT",
    "DEFAULT_EVIDENCE_BASE",
    "DEFAULT_HOST_RECEIPT_PATH",
    "DEFAULT_SOURCE_BASE",
    "DEFAULT_SYSTEMCTL_EXECUTABLE",
    "FORK_REPOSITORY",
    "INCOMPLETE_MARKER_NAME",
    "RELEASE_MANIFEST_NAME",
    "RELEASE_SCHEMA",
    "SCRATCH_PROVENANCE_NAME",
    "SCRATCH_PROVENANCE_SCHEMA",
    "STOPPED_RELEASE_FAILURE_SCHEMA",
    "STOPPED_RELEASE_PLAN_SCHEMA",
    "STOPPED_RELEASE_RECEIPT_SCHEMA",
    "ReleaseBuildSpec",
    "ReleaseManifest",
    "SystemdUnitBundle",
    "TMPFILES_NAME",
    "TreeEntry",
    "UNIT_BUNDLE_SCHEMA",
    "WRITER_MODULE",
    "WRITER_UNIT_NAME",
    "WriterOnlyUnitSpec",
    "apply_stopped_release",
    "build_release",
    "checkout_commands",
    "collect_tree_entries",
    "create_release_manifest",
    "install_commands",
    "main",
    "plan_stopped_release",
    "python_bootstrap_commands",
    "render_systemd_units",
    "source_snapshot_command",
    "verify_clean_checkout",
    "wheel_install_command",
]


if __name__ == "__main__":
    raise SystemExit(main())
