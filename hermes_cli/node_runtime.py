"""Helpers for Hermes-managed Node.js runtime discovery.

Hermes may use its private Node inside Hermes subprocesses, but it must not
expose that Node by installing user-global ``node``/``npm``/``npx`` shims.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def hermes_node_bin_dir(hermes_home: Path | None = None) -> Path:
    """Return the platform-preferred directory for Hermes-managed Node tools."""
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    node_root = Path(hermes_home) / "node"
    return node_root if sys.platform == "win32" else node_root / "bin"


def _node_command_name() -> str:
    return "node.exe" if sys.platform == "win32" else "node"


def _npm_command_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("npm.cmd", "npm.exe", "npm")
    return ("npm",)


def _parse_node_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"v?(\d+)\.(\d+)(?:\.\d+)?", text.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _version_satisfies_floor(version: tuple[int, int]) -> bool:
    major, minor = version
    if major == 20:
        return minor >= 19
    if major == 22:
        return minor >= 12
    return major > 22


def node_satisfies_hermes_floor(node_bin: str | Path) -> bool:
    """Return whether ``node --version`` satisfies ``^20.19 || >=22.12``.

    Keep this floor in sync with ``scripts/install.sh::node_satisfies_build``
    and ``scripts/install.ps1::Test-NodeVersionOk``.
    """
    try:
        result = subprocess.run(
            [str(node_bin), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    version = _parse_node_version(result.stdout or result.stderr or "")
    return bool(version and _version_satisfies_floor(version))


def _path_parts() -> list[str]:
    return [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]


def prepend_node_bin_dir(node_bin: str | Path) -> bool:
    """Prepend the directory containing *node_bin* to process-local ``PATH``."""
    path = Path(node_bin)
    bin_dir = path.parent if path.name in {"node", "node.exe"} or path.is_file() else path
    if not bin_dir.is_dir():
        return False

    bin_dir_s = str(bin_dir)
    parts = _path_parts()
    if parts and parts[0] == bin_dir_s:
        return False
    parts = [part for part in parts if part != bin_dir_s]
    os.environ["PATH"] = os.pathsep.join([bin_dir_s, *parts])
    return True


def _private_node_path(hermes_home: Path | None = None) -> Path:
    return hermes_node_bin_dir(hermes_home) / _node_command_name()


def _private_npm_path(hermes_home: Path | None = None) -> Path | None:
    node_bin = hermes_node_bin_dir(hermes_home)
    for name in _npm_command_names():
        candidate = node_bin / name
        if candidate.is_file() and (sys.platform == "win32" or os.access(candidate, os.X_OK)):
            return candidate
    return None


def has_valid_private_hermes_node(hermes_home: Path | None = None) -> bool:
    """Return whether Hermes has a complete private Node/npm pair."""
    node = _private_node_path(hermes_home)
    if not node.is_file() or (sys.platform != "win32" and not os.access(node, os.X_OK)):
        return False
    if _private_npm_path(hermes_home) is None:
        return False
    return node_satisfies_hermes_floor(node)


def private_hermes_npm_path(hermes_home: Path | None = None) -> Path | None:
    """Return private Hermes npm when the private Node runtime is complete."""
    if not has_valid_private_hermes_node(hermes_home):
        return None
    return _private_npm_path(hermes_home)


def augment_path_with_hermes_node() -> bool:
    """Prepend Hermes-managed Node to this process when current Node is unusable.

    Missing ``npm`` alone is not a reason to shadow a modern user-managed Node.
    """
    current_node = shutil.which("node")
    if current_node and node_satisfies_hermes_floor(current_node):
        return False
    if not has_valid_private_hermes_node():
        return False
    return prepend_node_bin_dir(_private_node_path())


def legacy_node_symlink_candidate_dirs() -> list[Path]:
    """Return directories where old POSIX installers created node/npm/npx links."""
    dirs: list[Path] = [Path.home() / ".local" / "bin"]
    prefix = os.environ.get("PREFIX")
    if prefix:
        dirs.append(Path(prefix) / "bin")
    if sys.platform != "win32":
        dirs.append(Path("/usr/local/bin"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            deduped.append(directory)
            seen.add(key)
    return deduped


def _link_target_points_into(link: Path, node_dir: Path) -> bool:
    if not link.is_symlink():
        return False
    try:
        target = Path(os.readlink(link))
    except OSError:
        return False
    if not target.is_absolute():
        target = link.parent / target
    try:
        target_resolved = target.resolve(strict=False)
        node_resolved = node_dir.resolve(strict=False)
    except OSError:
        return False
    return target_resolved == node_resolved or node_resolved in target_resolved.parents


def remove_legacy_node_symlinks(
    hermes_home: Path,
    *,
    candidate_dirs: Iterable[Path] | None = None,
) -> list[Path]:
    """Remove legacy Hermes-owned node/npm/npx symlinks from command dirs."""
    node_dir = Path(hermes_home) / "node"
    removed: list[Path] = []
    for bin_dir in candidate_dirs or legacy_node_symlink_candidate_dirs():
        if not bin_dir.is_dir():
            continue
        for name in ("node", "npm", "npx"):
            link = bin_dir / name
            if not _link_target_points_into(link, node_dir):
                continue
            try:
                link.unlink()
            except OSError:
                continue
            removed.append(link)
    return removed
