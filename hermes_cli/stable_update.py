"""Stable-tag update helpers for Hermes source checkouts.

This module intentionally does *not* mutate the checkout.  It only fetches
remote tags and resolves refs so the CLI banner/update-check path can report
stable release availability without comparing against origin/main commit
counts.  Operational switching/downgrading is handled by the workspace overlay
script on installations that opt into the stable-tag channel.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

DEFAULT_STABLE_TAG_PATTERN = "v20*"
DEFAULT_STABLE_TAG_REMOTE = "origin"
STABLE_TAG_STRATEGIES = {"stable-tags", "stable_tags", "stable-tag", "tags"}


def stable_updates_enabled(config: dict[str, Any] | None) -> bool:
    """Return whether config opts update checks into stable git tags."""
    updates = config.get("updates", {}) if isinstance(config, dict) else {}
    if not isinstance(updates, dict):
        return False
    strategy = str(
        updates.get("check_strategy")
        or updates.get("strategy")
        or updates.get("channel")
        or ""
    ).strip().lower()
    return bool(updates.get("stable_tags")) or strategy in STABLE_TAG_STRATEGIES


def stable_update_config(config: dict[str, Any] | None) -> dict[str, str]:
    """Extract stable-tag update settings with safe defaults."""
    updates = config.get("updates", {}) if isinstance(config, dict) else {}
    if not isinstance(updates, dict):
        updates = {}
    return {
        "pattern": str(updates.get("stable_tag_pattern") or DEFAULT_STABLE_TAG_PATTERN),
        "remote": str(updates.get("stable_tag_remote") or DEFAULT_STABLE_TAG_REMOTE),
        "command": str(updates.get("stable_update_command") or ""),
    }


def _run_git(repo_dir: Path, args: list[str], *, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def fetch_tags(repo_dir: Path, remote: str = DEFAULT_STABLE_TAG_REMOTE) -> tuple[bool, str]:
    """Fetch tags from *remote*. Returns ``(ok, stderr)`` and never raises."""
    try:
        result = _run_git(repo_dir, ["fetch", remote, "--tags", "--quiet"], timeout=15)
    except Exception as exc:  # network failures/timeouts should not break startup
        return False, str(exc)
    return result.returncode == 0, (result.stderr or "").strip()


def list_stable_tags(repo_dir: Path, pattern: str = DEFAULT_STABLE_TAG_PATTERN) -> list[str]:
    """List local stable-looking tags newest-first using git's version sort."""
    try:
        result = _run_git(
            repo_dir,
            ["tag", "--list", pattern, "--sort=-v:refname"],
            timeout=5,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def resolve_commit(repo_dir: Path, ref: str) -> Optional[str]:
    """Resolve a git ref/tag to a full commit hash, or None on failure."""
    try:
        result = _run_git(repo_dir, ["rev-parse", f"{ref}^{{commit}}"], timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def exact_head_tag(repo_dir: Path) -> Optional[str]:
    """Return the exact tag pointing at HEAD, if any."""
    try:
        result = _run_git(repo_dir, ["describe", "--tags", "--exact-match", "HEAD"], timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def stable_update_status(
    repo_dir: Path,
    *,
    pattern: str = DEFAULT_STABLE_TAG_PATTERN,
    remote: str = DEFAULT_STABLE_TAG_REMOTE,
    target_tag: str | None = None,
    fetch: bool = True,
) -> dict[str, Any]:
    """Return stable-tag update status for *repo_dir*.

    The returned dict is JSON-serializable and includes enough context for the
    banner to say which stable tag is available.  It only compares HEAD to a
    tag's commit; it never compares against ``origin/main``/``origin/master``.
    """
    repo_dir = Path(repo_dir)
    status: dict[str, Any] = {
        "mode": "stable-tags",
        "pattern": pattern,
        "remote": remote,
        "current_tag": None,
        "head": None,
        "latest_tag": None,
        "target_tag": target_tag,
        "target_commit": None,
        "up_to_date": None,
        "update_available": False,
        "fetch_ok": None,
        "fetch_error": None,
        "error": None,
    }

    if not (repo_dir / ".git").exists():
        status["error"] = "not-a-git-checkout"
        return status

    if fetch:
        ok, err = fetch_tags(repo_dir, remote=remote)
        status["fetch_ok"] = ok
        status["fetch_error"] = err or None

    tags = list_stable_tags(repo_dir, pattern=pattern)
    if not tags and not target_tag:
        status["error"] = "no-stable-tags"
        return status

    latest_tag = tags[0] if tags else target_tag
    target = target_tag or latest_tag
    status["latest_tag"] = latest_tag
    status["target_tag"] = target
    status["current_tag"] = exact_head_tag(repo_dir)
    status["head"] = resolve_commit(repo_dir, "HEAD")
    status["target_commit"] = resolve_commit(repo_dir, target) if target else None

    if not status["head"] or not status["target_commit"]:
        status["error"] = "could-not-resolve-ref"
        return status

    status["up_to_date"] = status["head"] == status["target_commit"]
    status["update_available"] = not status["up_to_date"]
    return status
