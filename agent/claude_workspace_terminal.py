"""Worktree-confined command wrapper for Claude's Hermes terminal MCP."""

from __future__ import annotations

import json
import hashlib
import platform
import shlex
import sys
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


def _seatbelt_string(path: Path) -> str:
    return json.dumps(str(path), ensure_ascii=False)


def _reject_linked_workspace_files(root: Path) -> None:
    """Fail closed if a regular file could alias a path outside the workspace."""

    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise RuntimeError(f"Could not inspect worker workspace: {directory}") from exc
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(f"Could not inspect workspace path: {entry.path}") from exc
            if stat.S_ISDIR(info.st_mode):
                pending.append(Path(entry.path))
            elif stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise RuntimeError(
                    "Workspace terminal rejects hard-linked regular file: "
                    f"{entry.path}. Recreate dependency files in copy mode "
                    "(for uv, set UV_LINK_MODE=copy)."
                )


def _metadata_ancestors(path: Path) -> list[Path]:
    return [path, *path.parents]


def _write_terminal_profile(profile: str) -> Path:
    """Persist a stable, owner-only Seatbelt profile outside the workspace."""

    directory = get_hermes_home() / "cache" / "claude-agent-sdk" / "terminal-profiles"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    digest = hashlib.sha256(profile.encode("utf-8")).hexdigest()
    path = directory / f"{digest}.sb"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        fd = -1
    if fd >= 0:
        try:
            data = profile.encode("utf-8")
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
        or path.read_text(encoding="utf-8") != profile
    ):
        raise RuntimeError("Claude terminal Seatbelt profile failed integrity checks")
    return path


def build_workspace_seatbelt_profile(
    *,
    workspace: str | Path,
    host_home: str | Path,
    allow_network: bool,
    readable_roots: list[str | Path] | None = None,
    readable_paths: list[str | Path] | None = None,
    restrict_reads: bool = True,
    control_write_paths: list[str | Path] | None = None,
    control_write_roots: list[str | Path] | None = None,
) -> str:
    """Build a deny-by-default write boundary with workspace-only effects."""

    root = Path(workspace).expanduser().resolve()
    host = Path(host_home).expanduser().resolve()
    # The control-plane Claude CLI still uses the legacy allow-default branch
    # with narrow write denials. The model-callable terminal always takes the
    # restrict_reads branch below, which is a stable default-deny policy.
    lines = ["(version 1)"]
    if restrict_reads:
        lines.extend(
            [
                "(deny default)",
                '(import "system.sb")',
                "(allow process-exec)",
                "(allow process-fork)",
                "(allow process-info* (target self))",
                "(allow signal (target children))",
            ]
        )
        if not allow_network:
            lines.append("(deny network*)")
    else:
        lines.append("(allow default)")
        if not allow_network:
            lines.append("(deny network*)")
    lines.extend(
        [
            "(deny file-write*)",
            f"(allow file-write* (subpath {_seatbelt_string(root)}))",
            '(allow file-write* (literal "/dev/null"))',
        ]
    )
    if restrict_reads:
        lines.append(f"(allow file-read* (subpath {_seatbelt_string(root)}))")
        for ancestor in _metadata_ancestors(root):
            lines.append(
                f"(allow file-read-metadata (literal {_seatbelt_string(ancestor)}))"
            )
    del host  # retained in the interface to make the protected boundary explicit
    for system_root in (
        "/System",
        "/usr",
        "/bin",
        "/sbin",
        "/dev",
    ):
        if restrict_reads:
            lines.append(f'(allow file-read* (subpath "{system_root}"))')
    if restrict_reads:
        # `/usr` is required for macOS runtime files, but these locally
        # managed descendants can contain service credentials/package state.
        for private_local_root in (
            "/usr/local/etc",
            "/usr/local/var",
            "/opt/homebrew/etc",
            "/opt/homebrew/var",
        ):
            lines.append(f'(deny file-read* (subpath "{private_local_root}"))')
    for readable in readable_roots or []:
        lexical = Path(readable).expanduser().absolute()
        if restrict_reads:
            for path in dict.fromkeys((lexical, lexical.resolve())):
                lines.append(f"(allow file-read* (subpath {_seatbelt_string(path)}))")
                lines.append(
                    f"(allow file-read-metadata (literal {_seatbelt_string(path)}))"
                )
                for parent in path.parents:
                    lines.append(
                        f"(allow file-read-metadata (literal {_seatbelt_string(parent)}))"
                    )
    for readable_path in readable_paths or []:
        lexical = Path(readable_path).expanduser().absolute()
        for path in dict.fromkeys((lexical, lexical.resolve())):
            lines.append(f"(allow file-read* (literal {_seatbelt_string(path)}))")
            for parent in path.parents:
                lines.append(
                    f"(allow file-read-metadata (literal {_seatbelt_string(parent)}))"
                )
    for writable in control_write_paths or []:
        path = Path(writable).expanduser().resolve(strict=False)
        lines.append(f"(allow file-write* (literal {_seatbelt_string(path)}))")
    for writable_root in control_write_roots or []:
        path = Path(writable_root).expanduser().resolve(strict=False)
        lines.append(f"(allow file-write* (subpath {_seatbelt_string(path)}))")
    return "\n".join(lines)


def _git_common_dir(root: Path) -> Path | None:
    git = shutil.which("git")
    if not git:
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        path = Path(result.stdout.strip()).expanduser().resolve()
        return path if path.exists() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _mach_o_dependencies(paths: list[Path]) -> list[Path]:
    """Return exact existing dylib paths needed by the selected executables."""

    otool = Path("/usr/bin/otool")
    if not otool.exists():
        return []
    dependencies: list[Path] = []
    pending = list(paths)
    inspected: set[Path] = set()
    while pending and len(inspected) < 128:
        candidate = pending.pop()
        canonical = candidate.resolve(strict=False)
        if canonical in inspected or not canonical.is_file():
            continue
        inspected.add(canonical)
        try:
            result = subprocess.run(
                [str(otool), "-L", str(canonical)],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in result.stdout.splitlines()[1:]:
            raw = line.strip().split(" ", 1)[0]
            if not raw.startswith("/"):
                continue
            dependency = Path(raw)
            if dependency.exists() and dependency not in dependencies:
                dependencies.append(dependency)
                pending.append(dependency)
    return dependencies


def _homebrew_formula_roots(paths: list[Path]) -> list[Path]:
    """Return immutable versioned Cellar roots for selected Homebrew tools."""

    roots: list[Path] = []
    for path in paths:
        parts = path.resolve(strict=False).parts
        try:
            cellar_index = parts.index("Cellar")
        except ValueError:
            continue
        if len(parts) < cellar_index + 3:
            continue
        root = Path(*parts[: cellar_index + 3])
        if root.is_dir() and root not in roots:
            roots.append(root)
    return roots


def build_workspace_terminal_args(
    arguments: Mapping[str, Any],
    *,
    workspace: str | Path,
    host_home: str | Path,
    exact_env: Mapping[str, str],
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Wrap a Hermes terminal call in exact-env macOS Seatbelt isolation."""

    if (platform_name or platform.system()) != "Darwin":
        raise RuntimeError("Workspace terminal sandbox is unsupported on this OS")
    root = Path(workspace).expanduser().resolve()
    host = Path(host_home).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Worker workspace does not exist: {root}")
    command = str(arguments.get("command") or "").strip()
    if not command:
        raise RuntimeError("Workspace terminal requires a command")
    _reject_linked_workspace_files(root)
    executable_paths = [
        Path(path)
        for path in (
            sys.executable,
            shutil.which("uv"),
            shutil.which("rg"),
            shutil.which("git"),
            str(root / ".venv" / "bin" / "python"),
            str(root / ".venv" / "bin" / "python3"),
        )
        if path and Path(path).exists()
    ]
    for executable in list(executable_paths):
        if executable.is_symlink():
            target = Path(os.readlink(executable))
            if not target.is_absolute():
                target = executable.parent / target
            executable_paths.append(target.absolute())
    executable_paths.extend(_mach_o_dependencies(executable_paths))

    git_common_dir = _git_common_dir(root)
    toolchain_roots = [
        Path(path)
        for path in (
            str(Path(sys.executable).resolve().parents[1]),
        )
        if Path(path).exists()
    ]
    toolchain_roots.extend(_homebrew_formula_roots(executable_paths))
    if git_common_dir is not None and not git_common_dir.is_relative_to(root):
        toolchain_roots.append(git_common_dir)
    profile = build_workspace_seatbelt_profile(
        workspace=root,
        host_home=host,
        allow_network=False,
        readable_roots=toolchain_roots,
        readable_paths=executable_paths,
    )
    profile_path = _write_terminal_profile(profile)
    allowed_env_keys = {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
    }
    terminal_env = {
        key: str(value)
        for key, value in exact_env.items()
        if key in allowed_env_keys
    }
    terminal_env["HOME"] = str(root)
    terminal_tmp = root / ".hermes-claude-runtime" / "tmp"
    terminal_tmp.mkdir(mode=0o700, parents=True, exist_ok=True)
    terminal_tmp.chmod(0o700)
    terminal_env["TMPDIR"] = str(terminal_tmp)
    terminal_env["GIT_CONFIG_NOSYSTEM"] = "1"
    terminal_env["GIT_OPTIONAL_LOCKS"] = "0"
    terminal_env["UV_LINK_MODE"] = "copy"
    env_argv = [f"{key}={value}" for key, value in sorted(terminal_env.items())]
    wrapped_argv = [
        "/usr/bin/env",
        "-i",
        *env_argv,
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile_path),
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        command,
    ]
    transformed = dict(arguments)
    workdir = Path(str(arguments.get("workdir") or root)).expanduser()
    if not workdir.is_absolute():
        workdir = root / workdir
    resolved_workdir = workdir.resolve(strict=False)
    if resolved_workdir != root and not resolved_workdir.is_relative_to(root):
        raise RuntimeError("Workspace terminal workdir is outside the worker workspace")
    transformed["workdir"] = str(resolved_workdir)
    transformed["command"] = shlex.join(wrapped_argv)
    return transformed


__all__ = ["build_workspace_seatbelt_profile", "build_workspace_terminal_args"]
