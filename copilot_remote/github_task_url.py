"""Helpers for safely building GitHub task URLs for Copilot remotes."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlsplit


# === ROSENBLATT PATCH START: harden GitHub task URL derivation ===
# Reason: Amazon Inspector flagged the previous helper for unsafe path and URL handling.
# Upstream: internal-only
_GITHUB_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def _is_safe_github_segment(value: str) -> bool:
    return bool(value) and bool(_GITHUB_SEGMENT_RE.fullmatch(value))


def _resolve_repo_dir(repo_path: str, repo_slug: str) -> Optional[Path]:
    if not repo_path or not _is_safe_github_segment(repo_slug):
        return None
    try:
        repo_dir = Path(repo_path).expanduser().resolve(strict=True)
    except OSError:
        return None
    if not repo_dir.is_dir() or repo_dir.name != repo_slug:
        return None
    return repo_dir


def _git_origin_url(repo_dir: Path) -> Optional[str]:
    git_bin = shutil.which("git")
    if not git_bin or not os.path.isabs(git_bin):
        return None
    try:
        inside_worktree = subprocess.run(
            [git_bin, "rev-parse", "--is-inside-work-tree"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if inside_worktree.returncode != 0 or inside_worktree.stdout.strip() != "true":
            return None
        result = subprocess.run(
            [git_bin, "config", "--get", "remote.origin.url"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    origin_url = result.stdout.strip()
    return origin_url if result.returncode == 0 and origin_url else None


def _github_owner_from_origin_url(origin_url: str) -> Optional[str]:
    normalized = (origin_url or "").strip()
    if not normalized:
        return None

    if normalized.startswith("git@"):
        host, separator, remote_path = normalized.partition(":")
        if separator != ":" or host.split("@", 1)[-1].lower() != "github.com":
            return None
        parts = [part for part in remote_path.split("/") if part]
    else:
        parsed = urlsplit(normalized)
        if (parsed.hostname or "").lower() != "github.com":
            return None
        parts = [part for part in parsed.path.split("/") if part]

    if len(parts) < 2:
        return None
    owner = parts[0]
    return owner if _is_safe_github_segment(owner) else None


def build_github_task_web_url(
    repo_path: str,
    repo_slug: str,
    connect_handle: Optional[str],
) -> Optional[str]:
    handle = str(connect_handle or "").strip()
    if not handle or not _is_safe_github_segment(repo_slug):
        return None

    repo_dir = _resolve_repo_dir(repo_path, repo_slug)
    if repo_dir is None:
        return None

    owner = _github_owner_from_origin_url(_git_origin_url(repo_dir) or "")
    if not owner:
        return None

    return (
        f"https://github.com/{quote(owner, safe='-._~')}"
        f"/{quote(repo_slug, safe='-._~')}"
        f"/tasks/{quote(handle, safe='-._~')}"
    )


# === ROSENBLATT PATCH END ===