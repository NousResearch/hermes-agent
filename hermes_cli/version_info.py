"""Truthful derived build-version metadata for user-facing Hermes displays.

``__version__`` remains the package/API version. This module adds a display
suffix only when it can prove the number of commits since that release.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hermes_cli import __release_date__, __version__


@dataclass(frozen=True)
class VersionInfo:
    base_version: str
    derived_version: str
    distance: int | None
    commit: str | None
    branch: str | None
    source: Literal["git", "nix", "build", "unknown"]
    dirty: bool = False


def format_display_version(info: VersionInfo | None = None) -> str:
    """Return ``0.x.y`` or ``0.x.y+N`` without exposing unknown distance."""
    info = info or get_version_info()
    return info.derived_version


def _derived_version(base_version: str, distance: int | None) -> str:
    return f"{base_version}+{distance}" if distance and distance > 0 else base_version


def _run_git(repo_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=3, cwd=str(repo_dir)
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    return value if result.returncode == 0 and value else None


def _resolve_repo_dir() -> Path | None:
    """Use the executing checkout before a profile's optional clone."""
    repo_dir = Path(__file__).parent.parent.resolve()
    if (repo_dir / ".git").exists():
        return repo_dir
    try:
        from hermes_constants import get_hermes_home

        candidate = get_hermes_home() / "hermes-agent"
        if (candidate / ".git").exists():
            return candidate
    except Exception:
        pass
    return None


def _parse_nonnegative(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _nix_version_info() -> VersionInfo | None:
    commit = os.environ.get("HERMES_REVISION") or None
    current_count = _parse_nonnegative(os.environ.get("HERMES_REVISION_COUNT"))
    release_count = _parse_nonnegative(os.environ.get("HERMES_RELEASE_REV_COUNT"))
    if not commit:
        return None
    distance = (
        max(0, current_count - release_count)
        if current_count is not None and release_count is not None
        else None
    )
    return VersionInfo(
        __version__,
        _derived_version(__version__, distance),
        distance,
        commit,
        os.environ.get("HERMES_REVISION_BRANCH") or None,
        "nix",
        os.environ.get("HERMES_REVISION_DIRTY") == "1",
    )


def _git_version_info(repo_dir: Path) -> VersionInfo:
    commit = _run_git(repo_dir, "rev-parse", "HEAD")
    branch = _run_git(repo_dir, "branch", "--show-current")
    if not branch and commit:
        branch = commit[:8]
    try:
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(repo_dir),
        )
        dirty = dirty_result.returncode == 0 and bool((dirty_result.stdout or "").strip())
    except (OSError, subprocess.SubprocessError):
        dirty = False

    # New releases are SemVer tags. The release-date fallback lets existing
    # CalVer-tagged releases display a correct distance during the transition.
    distance = None
    for tag in (f"v{__version__}", f"v{__release_date__}"):
        raw_distance = _run_git(repo_dir, "rev-list", "--count", f"{tag}..HEAD")
        parsed_distance = _parse_nonnegative(raw_distance)
        if parsed_distance is not None:
            distance = parsed_distance
            break

    return VersionInfo(
        __version__, _derived_version(__version__, distance), distance, commit, branch, "git", dirty
    )


_cached_version_info: VersionInfo | None = None


def _reset_version_info_cache() -> None:
    """Test-only cache reset."""
    global _cached_version_info
    _cached_version_info = None


def get_version_info() -> VersionInfo:
    """Return cached provenance from Nix metadata, git, or a baked SHA."""
    global _cached_version_info
    if _cached_version_info is not None:
        return _cached_version_info

    info = _nix_version_info()
    if info is None:
        repo_dir = _resolve_repo_dir()
        if repo_dir is not None:
            info = _git_version_info(repo_dir)
        else:
            try:
                from hermes_cli.build_info import get_build_sha

                commit = get_build_sha(short=0)
            except Exception:
                commit = None
            info = VersionInfo(__version__, __version__, None, commit, None, "build" if commit else "unknown")

    _cached_version_info = info
    return info


def format_version_details(info: VersionInfo | None = None) -> str:
    """Format verbose, support-friendly provenance without pretending certainty."""
    info = info or get_version_info()
    values = [f"version {info.derived_version}"]
    if info.branch:
        values.append(f"branch {info.branch}")
    if info.commit:
        values.append(f"commit {info.commit}")
    values.append(f"source {info.source}")
    return " · ".join(values)
