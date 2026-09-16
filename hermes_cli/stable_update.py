"""Stable-tag update helpers for Hermes source checkouts.

This module intentionally does *not* mutate the checkout.  It only fetches
remote tags and resolves refs so the CLI banner/update-check path can report
stable release availability without comparing against origin/main commit
counts.  Operational switching/downgrading is handled by the workspace overlay
script on installations that opt into the stable-tag channel.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

DEFAULT_STABLE_TAG_PATTERN = "v20*"
OFFICIAL_RELEASE_REPOSITORY = "https://github.com/NousResearch/hermes-agent.git"
OFFICIAL_RELEASES_API = "https://api.github.com/repos/NousResearch/hermes-agent/releases"
DEFAULT_STABLE_TAG_REMOTE = OFFICIAL_RELEASE_REPOSITORY
OFFICIAL_RELEASE_TAG_RE = re.compile(r"^v(20\d{2})\.(\d{1,2})\.(\d{1,2})(?:\.(\d+))?$")
STABLE_TAG_STRATEGIES = {"stable-tags", "stable_tags", "stable-tag", "tags"}
UPDATE_CHANNELS = {"main", "release"}


def normalize_update_channel(value: Any, *, default: str = "release") -> str:
    """Normalize stable/beta (and historical release/main) to one internal channel."""
    normalized_default = default if default in UPDATE_CHANNELS else "release"
    raw = str(value or "").strip().lower()
    if raw in {"release", "stable", *STABLE_TAG_STRATEGIES}:
        return "release"
    if raw in {"main", "beta", "branch", "fast", "fast-track", "fast_track"}:
        return "main"
    return normalized_default


def configured_update_channel(config: dict[str, Any] | None, *, default: str = "release") -> str:
    """Return the canonical update channel from an effective config mapping."""
    updates = config.get("updates", {}) if isinstance(config, dict) else {}
    if not isinstance(updates, dict):
        return normalize_update_channel(None, default=default)
    return normalize_update_channel(
        updates.get("channel") or updates.get("check_strategy") or updates.get("strategy"),
        default=default,
    )


def stable_updates_enabled(config: dict[str, Any] | None) -> bool:
    """Return whether config opts update checks into stable git tags."""
    updates = config.get("updates", {}) if isinstance(config, dict) else {}
    if not isinstance(updates, dict):
        return False
    return bool(updates.get("stable_tags")) or configured_update_channel(config) == "release"


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
    # Embedded/source builds may have no checkout. Git ref probes can still use
    # the process cwd for the canonical remote, but never fail over to main.
    cwd = str(repo_dir) if (Path(repo_dir) / ".git").exists() else None
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
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
    """Resolve a git ref to its peeled commit SHA."""
    try:
        result = _run_git(repo_dir, ["rev-parse", f"{ref}^{{commit}}"], timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def parse_official_release_refs(output: str) -> dict[str, str]:
    """Return strict official release tags mapped to their peeled commit SHA."""
    direct: dict[str, str] = {}
    peeled: dict[str, str] = {}

    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        match = re.fullmatch(
            r"refs/tags/(v20\d{2}\.\d{1,2}\.\d{1,2}(?:\.\d+)?)(\^\{\})?",
            ref,
        )
        if not match or len(sha) != 40 or any(c not in "0123456789abcdefABCDEF" for c in sha):
            continue
        tag = match.group(1)
        (peeled if match.group(2) else direct)[tag] = sha.lower()

    return {tag: peeled.get(tag, sha) for tag, sha in direct.items()}


def resolve_published_release_tag(
    requested_tag: str | None = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve a published GitHub Release to its strict date tag."""
    if requested_tag is not None and not OFFICIAL_RELEASE_TAG_RE.fullmatch(requested_tag):
        return None, f"Official release tag '{requested_tag}' not found."

    endpoint = (
        f"{OFFICIAL_RELEASES_API}/tags/{urllib.parse.quote(requested_tag, safe='')}"
        if requested_tag
        else f"{OFFICIAL_RELEASES_API}/latest"
    )
    request = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "hermes-agent-updater",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and requested_tag:
            return None, f"Official release tag '{requested_tag}' not found."
        return None, f"Could not read official GitHub releases (HTTP {exc.code})."
    except Exception as exc:
        return None, str(exc)

    tag = str(payload.get("tag_name") or "") if isinstance(payload, dict) else ""
    is_draft = bool(payload.get("draft")) if isinstance(payload, dict) else True
    if (
        not OFFICIAL_RELEASE_TAG_RE.fullmatch(tag)
        or is_draft
        or (requested_tag is not None and tag != requested_tag)
    ):
        label = requested_tag or tag or "latest"
        return None, f"Official release tag '{label}' not found."
    return tag, None


def _official_tag_commit(repo_dir: Path, tag: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve one published release tag from the official repository."""
    try:
        result = _run_git(
            repo_dir,
            [
                "ls-remote",
                "--tags",
                OFFICIAL_RELEASE_REPOSITORY,
                f"refs/tags/{tag}",
                f"refs/tags/{tag}^{{}}",
            ],
            timeout=30,
        )
    except Exception as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, (result.stderr or "Could not read the official release tag.").strip()

    refs = parse_official_release_refs(result.stdout or "")
    commit = refs.get(tag)
    return commit, None if commit else f"Official release tag '{tag}' was not found in the official repository."


def official_release_status(repo_dir: Path, *, current_sha: str | None = None) -> dict[str, Any]:
    """Return exact-pin status against the latest published GitHub Release."""
    latest_tag, error = resolve_published_release_tag()
    target_commit = None
    if not error and latest_tag:
        target_commit, error = _official_tag_commit(Path(repo_dir), latest_tag)
    status: dict[str, Any] = {
        "mode": "official-releases",
        "current_release_tag": None,
        "head": (str(current_sha).strip().lower() if current_sha else resolve_commit(Path(repo_dir), "HEAD")),
        "latest_tag": None,
        "target_tag": None,
        "target_commit": None,
        "up_to_date": None,
        "update_available": False,
        "releases_behind": None,
        "error": error,
    }
    if error or not latest_tag or not target_commit or not status["head"]:
        status["error"] = error or "could-not-resolve-head"
        return status

    status["latest_tag"] = latest_tag
    status["target_tag"] = latest_tag
    status["target_commit"] = target_commit
    if status["head"] == target_commit:
        status["current_release_tag"] = latest_tag
        status["releases_behind"] = 0
    status["up_to_date"] = status["head"] == target_commit
    status["update_available"] = not status["up_to_date"]
    return status


def resolve_official_release(repo_dir: Path, requested_tag: str | None = None) -> dict[str, Any]:
    """Fetch and verify one published GitHub Release from the official repo."""
    tag, error = resolve_published_release_tag(requested_tag)
    if error or not tag:
        return {"tag": requested_tag, "commit": None, "error": error or "No official release found."}

    expected_commit, error = _official_tag_commit(Path(repo_dir), tag)
    if error or not expected_commit:
        return {"tag": tag, "commit": None, "error": error}
    try:
        fetched = _run_git(
            Path(repo_dir),
            ["fetch", "--quiet", "--no-tags", OFFICIAL_RELEASE_REPOSITORY, f"refs/tags/{tag}"],
            timeout=30,
        )
    except Exception as exc:
        return {"tag": tag, "commit": None, "error": str(exc)}
    if fetched.returncode != 0:
        return {
            "tag": tag,
            "commit": None,
            "error": (fetched.stderr or f"Could not fetch official release {tag}.").strip(),
        }

    fetched_commit = resolve_commit(Path(repo_dir), "FETCH_HEAD")
    if fetched_commit != expected_commit:
        return {
            "tag": tag,
            "commit": fetched_commit,
            "error": f"Official release {tag} did not resolve to its advertised commit.",
        }
    return {"tag": tag, "commit": expected_commit, "error": None}


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


def current_branch(repo_dir: Path) -> Optional[str]:
    """Return the checked-out branch, or ``None`` for detached HEAD."""
    try:
        result = _run_git(repo_dir, ["symbolic-ref", "--quiet", "--short", "HEAD"], timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = (result.stdout or "").strip()
    return value or None


def is_ancestor(repo_dir: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    """Return whether *ancestor* is reachable from *descendant*."""
    try:
        result = _run_git(
            repo_dir,
            ["merge-base", "--is-ancestor", ancestor, descendant],
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0


def latest_reachable_tag(repo_dir: Path, tags: list[str]) -> Optional[str]:
    """Return the newest candidate tag reachable from HEAD."""
    for tag in tags:
        if is_ancestor(repo_dir, tag):
            return tag
    return None


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
        "current_release_tag": None,
        "current_branch": None,
        "head": None,
        "latest_tag": None,
        "target_tag": target_tag,
        "target_commit": None,
        "up_to_date": None,
        "update_available": False,
        "releases_behind": None,
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
    status["current_branch"] = current_branch(repo_dir)
    status["head"] = resolve_commit(repo_dir, "HEAD")
    status["target_commit"] = resolve_commit(repo_dir, target) if target else None

    if not status["head"] or not status["target_commit"]:
        status["error"] = "could-not-resolve-ref"
        return status

    current_release_tag = latest_reachable_tag(repo_dir, tags)
    status["current_release_tag"] = current_release_tag
    if current_release_tag in tags:
        status["releases_behind"] = tags.index(current_release_tag)

    # Release track means an exact, reproducible tag pin. Descendants of the tag
    # include unreleased main and local overlays; neither is the selected build.
    status["up_to_date"] = status["head"] == status["target_commit"]
    status["update_available"] = not status["up_to_date"]
    return status
