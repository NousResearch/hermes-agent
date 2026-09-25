"""Canonical runtime identity for Hermes.

Resolution order:
1. Install stamp (``install-stamp.json``) — written at build time by
   ``scripts/write_install_stamp.py`` for every packager (Docker, Nix, and
   the desktop app). The stamp is authoritative
   for packaged builds.
2. Live git — for unstamped source/dev installs with a ``.git`` directory.
3. Unknown — no stamp and no git. The provenance is unknown.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from hermes_cli.steward import UPDATE_MECHANISMS
from hermes_cli.update_channel import STABLE_TAG_RE


@dataclass(frozen=True)
class VersionInfo:
    base_version: str
    derived_version: str
    distance: int | None
    commit: str | None
    branch: str | None
    source: Literal["build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"]
    dirty: bool = False
    commit_date: int | None = None
    distribution: Literal["docker", "nix", "desktop-app"] | None = None

    @property
    def display_version(self) -> str:
        """``<base>+<distance>``: the short form surfaces label a version by.
        The commit is shown beside it where there is room, never inside it."""
        return _derived_version(self.base_version, self.distance)


def _derived_version(
    base_version: str,
    distance: int | None,
    dirty: bool = False,
    short_commit: str | None = None,
) -> str:
    if distance and distance > 0:
        suffix = f"{distance}.g{short_commit}" if short_commit else str(distance)
        return f"{base_version}+{suffix}{'.dirty' if dirty and short_commit else ''}"
    if dirty and distance is None:
        return f"{base_version}+?"
    return base_version


def _run_git(repo_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=3, cwd=str(repo_dir),
            encoding="utf-8", errors="replace",
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
    # The PROCESS home: this is the running code's identity and is cached
    # process-wide, so a profile's context-local override must not pick it.
    from hermes_constants import get_process_hermes_home

    candidate = get_process_hermes_home() / "hermes-agent"
    if (candidate / ".git").exists():
        return candidate
    return None


def _parse_nonnegative(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _calver_release_version(repo_dir: Path) -> tuple[str, int] | None:
    """The version the nearest CalVer release shipped, and the commits since it.

    Releases before semver tags existed are tagged ``vYYYY.M.D`` only; the
    version users actually run is in that tag's pyproject. Without this, a
    checkout past such a release would compare as "unknown" against plugins'
    ``requires_hermes``.
    """
    described = _run_git(repo_dir, "describe", "--tags", "--long", "--match", "v2[0-9][0-9][0-9].*", "HEAD")
    if not described:
        return None
    tag, count, _ = described.rsplit("-", 2)
    distance = _parse_nonnegative(count)
    try:
        project = tomllib.loads(_run_git(repo_dir, "show", f"{tag}:pyproject.toml") or "").get("project", {})
    except tomllib.TOMLDecodeError:
        return None
    version = project.get("version")
    if distance is None or not isinstance(version, str) or not STABLE_TAG_RE.fullmatch(f"v{version}"):
        return None
    return version, distance


# --- Install stamp reader ---------------------------------------------------

def _resolve_stamp_file() -> Path | None:
    """The executing tree's stamp (pm.paths.install_stamp_path owns the location)."""
    from pm.paths import install_stamp_path, repo_root

    p = install_stamp_path(repo_root())
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
    if data.get("updateMechanism") not in UPDATE_MECHANISMS:
        raise RuntimeError(
            f"install-stamp.json at {stamp_file} is missing a valid "
            f"'updateMechanism' (one of {', '.join(UPDATE_MECHANISMS)}). The "
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
    stamp_source = str(data.get("source") or "")
    if stamp_source == "git" and (stamp_file.parent / ".git").exists():
        live_commit = _run_git(stamp_file.parent, "rev-parse", "HEAD")
        if live_commit and live_commit != commit:
            return None

    base_version = data.get("baseVersion") or "unknown"
    display_version = data.get("displayVersion") or base_version
    distance = data.get("distance")
    if isinstance(distance, str):
        distance = _parse_nonnegative(distance)

    # ``source`` describes build provenance, while ``distribution`` identifies
    # the package form users installed. Keep both facts intact for support.
    source = (
        cast(Literal["build", "commit-build", "ci", "docker", "fallback", "git", "local", "nix", "unknown"], stamp_source)
        if stamp_source in {"commit-build", "ci", "docker", "fallback", "git", "local", "nix"}
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


def _git_version_info(repo_dir: Path, *, include_untracked: bool = False) -> VersionInfo:
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
        status_command = ["git", "status", "--porcelain"]
        if not include_untracked:
            status_command.append("-uno")
        dirty_result = subprocess.run(
            status_command,
            capture_output=True,
            text=True,
            timeout=3,
            cwd=str(repo_dir),
            encoding="utf-8",
            errors="replace",
        )
        dirty = dirty_result.returncode == 0 and bool((dirty_result.stdout or "").strip())
    except (OSError, subprocess.SubprocessError):
        dirty = False

    tags = _run_git(repo_dir, "tag", "--merged", "HEAD", "--list", "v[0-9]*")
    releases = [
        tag[1:]
        for tag in (tags or "").splitlines()
        if STABLE_TAG_RE.fullmatch(tag)
    ]
    base_version = (
        max(releases, key=lambda value: tuple(int(part) for part in value.split(".")))
        if releases else "unknown"
    )
    distance = _parse_nonnegative(
        _run_git(repo_dir, "rev-list", "--count", f"v{base_version}..HEAD")
    ) if releases else None
    if not releases:
        base_version, distance = _calver_release_version(repo_dir) or ("unknown", None)
    short_commit = _run_git(repo_dir, "rev-parse", "--short=7", "HEAD")
    if base_version == "unknown" and short_commit:
        display_version = f"git.{short_commit}{'.dirty' if dirty else ''}"
    else:
        display_version = _derived_version(base_version, distance, dirty, short_commit)

    return VersionInfo(
        base_version, display_version, distance, commit, branch, "git", dirty, commit_date
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
        info = VersionInfo("unknown", "unknown", None, None, None, "unknown")

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
