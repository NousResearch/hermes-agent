"""Stable-tag update checks for Hermes source checkouts.

By default Hermes compares a checkout against moving ``origin/main`` and
offers an update for *every* commit pushed upstream, which is noisy for a
tracking install. When ``updates.check_strategy: stable-tags`` is set, the
update check instead resolves the newest stable release *tag* from the
remote and reports an update only when a newer stable release exists.

This module only reads git state; it never mutates the checkout.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

DEFAULT_STABLE_TAG_PATTERN = "v[0-9]*"
DEFAULT_STABLE_TAG_REMOTE = "origin"
_STABLE_STRATEGIES = {"stable-tags", "stable_tags", "stable-tag", "tags"}

# vMAJOR.MINOR(.PATCH)? — pre-release suffixes (-rc1, -beta) are excluded so
# only stable releases surface.
_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)(?:\.(\d+))?$")


def stable_updates_enabled(config: dict[str, Any] | None) -> bool:
    """Return True when config opts update checks into stable git tags."""
    updates = config.get("updates", {}) if isinstance(config, dict) else {}
    if not isinstance(updates, dict):
        return False
    strategy = str(updates.get("check_strategy") or "").strip().lower()
    return bool(updates.get("stable_tags")) or strategy in _STABLE_STRATEGIES


def stable_tag_settings(config: dict[str, Any] | None) -> dict[str, str]:
    """Extract stable-tag settings (pattern, remote) with safe defaults."""
    updates = config.get("updates", {}) if isinstance(config, dict) else {}
    if not isinstance(updates, dict):
        updates = {}
    pattern = str(updates.get("stable_tag_pattern") or "").strip()
    remote = str(updates.get("stable_tag_remote") or "").strip()
    return {
        "pattern": pattern or DEFAULT_STABLE_TAG_PATTERN,
        "remote": remote or DEFAULT_STABLE_TAG_REMOTE,
    }


def _parse_version(tag: str) -> Optional[tuple[int, int, int]]:
    """Parse ``vX.Y[.Z]`` into a sortable tuple, or None if not stable."""
    match = _VERSION_RE.match(tag.strip())
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch or 0))


def _run_git(
    repo_dir: Path, args: list[str], *, timeout: float = 10.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def latest_stable_tag(
    repo_dir: Path,
    *,
    remote: str = DEFAULT_STABLE_TAG_REMOTE,
    pattern: str = DEFAULT_STABLE_TAG_PATTERN,
) -> Optional[str]:
    """Return the newest stable release tag on *remote*, or None.

    Uses ``git ls-remote --tags`` so no fetch/checkout is needed and network
    failures fail soft. Only tags shaped like ``vX.Y[.Z]`` count as stable.
    """
    try:
        result = _run_git(
            repo_dir, ["ls-remote", "--tags", remote, pattern], timeout=15
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    best: Optional[tuple[tuple[int, int, int], str]] = None
    for line in (result.stdout or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        ref = parts[1].strip()
        # Skip peeled dereference lines (refs/tags/v1.2.3^{}).
        if ref.endswith("^{}"):
            continue
        tag = ref.rsplit("/", 1)[-1]
        version = _parse_version(tag)
        if version is None:
            continue
        if best is None or version > best[0]:
            best = (version, tag)
    return best[1] if best else None


def current_stable_tag(repo_dir: Path) -> Optional[str]:
    """Return the highest stable tag reachable from HEAD, or None."""
    try:
        result = _run_git(
            repo_dir,
            ["tag", "--merged", "HEAD", "--list", "v[0-9]*"],
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    stable_tags = (
        (version, tag)
        for tag in (result.stdout or "").splitlines()
        if (version := _parse_version(tag)) is not None
    )
    best = max(stable_tags, default=None)
    return best[1] if best else None


def stable_update_status(
    repo_dir: Path,
    *,
    remote: str = DEFAULT_STABLE_TAG_REMOTE,
    pattern: str = DEFAULT_STABLE_TAG_PATTERN,
) -> dict[str, Any]:
    """Compare the local checkout to the newest remote stable tag.

    Returns a JSON-serializable dict. ``update_available`` is True only when
    the remote has a stable release strictly newer than the one reachable
    from HEAD. ``error`` is set (and ``update_available`` False) on any
    failure so the caller can fail soft.
    """
    repo_dir = Path(repo_dir)
    status: dict[str, Any] = {
        "mode": "stable-tags",
        "current_tag": None,
        "latest_tag": None,
        "update_available": False,
        "error": None,
    }
    if not (repo_dir / ".git").exists():
        status["error"] = "not-a-git-checkout"
        return status

    latest = latest_stable_tag(repo_dir, remote=remote, pattern=pattern)
    if latest is None:
        status["error"] = "no-remote-stable-tags"
        return status
    status["latest_tag"] = latest

    latest_v = _parse_version(latest)
    if latest_v is None:
        status["error"] = "no-remote-stable-tags"
        status["latest_tag"] = None
        return status

    current = current_stable_tag(repo_dir)
    status["current_tag"] = current
    current_v = _parse_version(current) if current else None
    if current_v is None:
        # No stable tag reachable from HEAD: a release exists we aren't on.
        status["update_available"] = True
    else:
        status["update_available"] = latest_v > current_v
    return status
