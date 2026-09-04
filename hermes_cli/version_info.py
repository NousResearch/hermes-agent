"""Truthful derived build-version metadata for user-facing Hermes displays.

``__version__`` remains the package/API version. This module adds a display
suffix only when it can prove the number of commits since that release.

Resolution order:
1. Install stamp (``install-stamp.json``) — written at build time by
   ``scripts/write_install_stamp.py`` for every packager (Docker, Nix, and
   the desktop app). The stamp is authoritative
   for packaged builds.
2. Live git — for source/dev installs with a ``.git`` directory.
3. Unknown — no stamp and no git. The provenance is unknown.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hermes_cli import __release_date__, __version__


@dataclass(frozen=True)
class VersionInfo:
    base_version: str
    derived_version: str
    distance: int | None
    commit: str | None
    branch: str | None
    source: Literal["build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"]
    dirty: bool = False
    commit_date: int | None = None
    distribution: Literal["docker", "nix", "desktop-app"] | None = None


def _derived_version(base_version: str, distance: int | None, dirty: bool = False) -> str:
    if distance and distance > 0:
        return f"{base_version}+{distance}"
    if dirty and distance is None:
        return f"{base_version}+?"
    return base_version


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


# --- Install stamp reader ---------------------------------------------------

# The stamp file lives at the install root: beside the code in source
# checkouts and Docker (which writes it to the project root), and in the
# artifact's resources dir for sealed installs — whose processes carry
# HERMES_INSTALL_ROOT (the Nix wrapper points it at the store path's
# share/hermes-agent, where the stamp is baked). One resolution path for
# every steward; no stamp-specific env override.
_CODE_ROOT = Path(__file__).parent.parent


def _resolve_stamp_file() -> Path | None:
    root = os.environ.get("HERMES_INSTALL_ROOT")
    base = Path(root) if root else _CODE_ROOT
    p = base / "install-stamp.json"
    return p if p.is_file() else None


def _stamp_version_info() -> VersionInfo | None:
    """Read provenance from a build-time install stamp."""
    stamp_file = _resolve_stamp_file()
    if stamp_file is None:
        return None
    try:
        raw = stamp_file.read_text(encoding="utf-8-sig")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict) or "commit" not in data:
        return None

    # updateMechanism is required in every stamp. A stamp without it means
    # the writing build lane must be fixed, not tolerated.
    if data.get("updateMechanism") not in ("self", "electron-updater", "external"):
        raise RuntimeError(
            f"install-stamp.json at {stamp_file} is missing a valid "
            "'updateMechanism' (one of self, electron-updater, external). The "
            "build lane that wrote this stamp must pass --update-mechanism to "
            "scripts/write_install_stamp.py (or bake the field directly)."
        )
    # A light artifact ships no Python runtime. Reading a light stamp from
    # a Python process means the artifact was mispackaged.
    if data.get("payload") == "light":
        raise RuntimeError(
            f"install-stamp.json at {stamp_file} marks this artifact as 'light' "
            "(no agent runtime). No Python process can legitimately run from a "
            "light artifact — this build is mispackaged."
        )

    commit = data.get("commit") or None
    if not commit or set(commit) == {"0"}:
        # All-zero placeholder = fallback stamp, not real provenance.
        return None

    base_version = data.get("baseVersion") or __version__
    display_version = data.get("displayVersion") or base_version
    distance = data.get("distance")
    if isinstance(distance, str):
        distance = _parse_nonnegative(distance)

    # ``source`` describes build provenance, while ``distribution`` identifies
    # the package form users installed. Keep both facts intact for support.
    stamp_source = str(data.get("source") or "")
    source = (
        cast(Literal["build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"], stamp_source)
        if stamp_source in {"ci", "docker", "fallback", "local", "nix"}
        else "build"
    )
    distribution = data.get("distribution")
    if distribution not in {"docker", "nix", "desktop-app"}:
        distribution = None

    commit_date = data.get("commitDate")
    if not isinstance(commit_date, int):
        commit_date = None

    return VersionInfo(
        base_version,
        display_version,
        distance if isinstance(distance, int) else None,
        commit,
        data.get("branch") or None,
        source,
        bool(data.get("dirty")),
        commit_date,
        distribution,
    )


# --- Git provenance (source/dev installs) -----------------------------------


def _git_version_info(repo_dir: Path) -> VersionInfo:
    commit = _run_git(repo_dir, "rev-parse", "HEAD")
    # A detached HEAD has no branch. Leave the field None: every formatter
    # already prints the commit separately and handles a missing branch.
    branch = _run_git(repo_dir, "branch", "--show-current")
    commit_date_raw = _run_git(repo_dir, "log", "-1", "--format=%ct", "HEAD")
    commit_date: int | None = None
    if commit_date_raw and commit_date_raw.isdigit():
        commit_date = int(commit_date_raw)
    try:
        # -uno: skip the untracked-file scan. This runs on the startup-banner
        # path, and a full working-tree walk costs real time on large or cold
        # checkouts. Same semantics as write_install_stamp.py.
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "-uno"],
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
        __version__, _derived_version(__version__, distance, dirty), distance, commit, branch, "git", dirty, commit_date
    )


# --- Cache + public API -----------------------------------------------------

_cached_version_info: VersionInfo | None = None


def _reset_version_info_cache() -> None:
    """Test-only cache reset."""
    global _cached_version_info
    _cached_version_info = None


def get_version_info() -> VersionInfo:
    """Return cached provenance from install stamp, git, or unknown."""
    global _cached_version_info
    if _cached_version_info is not None:
        return _cached_version_info

    # 1. Install stamp (packaged builds: Docker, Nix)
    # A malformed stamp (missing/illegal updateMechanism, mispackaged "light"
    # payload) raises RuntimeError — the build lane that wrote it must be
    # fixed, but a crashing CLI is the wrong failure mode for every uncached
    # caller. Degrade to "unknown" and let the authoring lane's own tests
    # surface the malformed stamp.
    try:
        info = _stamp_version_info()
    except RuntimeError:
        info = None

    # 2. Live git (source/dev installs)
    if info is None:
        repo_dir = _resolve_repo_dir()
        if repo_dir is not None:
            info = _git_version_info(repo_dir)

    # 3. Unknown — no stamp, no git
    if info is None:
        info = VersionInfo(__version__, __version__, None, None, None, "unknown")

    _cached_version_info = info
    return info


def get_code_identity(refresh: bool = False) -> dict:
    """Return the running install's code identity as a flat dict.

    Shape: ``{"sha": full sha | None, "short_sha": str | None, "version":
    base package version | None, "source": str}`` — what the update
    receipt, runtime inventory, and gateway status stamping consume.
    Backed by :func:`get_version_info` (install stamp first, live git
    second, unknown third), so every consumer shares one resolution
    policy. Returned dicts are fresh copies, never the shared cache.

    ``refresh=True`` drops the per-process cache first — the updater uses
    it to re-read identity after swapping the checkout out from under the
    running process.
    """
    global _cached_version_info
    if refresh:
        _cached_version_info = None
    info = get_version_info()
    return {
        "sha": info.commit,
        "short_sha": info.commit[:8] if info.commit else None,
        "version": info.base_version,
        "source": info.source,
    }
