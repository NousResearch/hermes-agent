"""Legacy install detection for the adoption funnel.

Detects whether the current install is a legacy git-checkout that can adopt
managed slots (phase 2, task 2.3 of the updater rework), and whether it is
pristine or dirty.

The detector is crash-proof: any git error is caught and results in
``pristine=False`` with a reason explaining what failed. It must be cheap
(<5ms in the common no-op case) because it runs on every launch.

See ``docs/plans/updater-rework/03-phase2-compat-and-adoption.md`` task 2.3
and ``docs/updater-world.md`` §2.13.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hermes_cli.config import detect_install_method


# Canonical GitHub remote, lowercased, no .git suffix — matches
# ``OFFICIAL_REPO_CANONICAL`` in ``apps/desktop/electron/update-remote.ts``.
OFFICIAL_REPO_CANONICAL = "github.com/nousresearch/hermes-agent"


@dataclass
class LegacyInfo:
    """Result of detecting a legacy git-checkout install.

    ``pristine`` is True only when the checkout is clean, on the official
    origin, on a known branch, and has no local commits ahead of origin.
    ``reasons`` explains why pristine is False (empty when pristine is True).
    """

    pristine: bool
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Remote URL canonicalization (ported from update-remote.ts)
# ---------------------------------------------------------------------------

def _canonical_github_remote(url: str) -> str:
    """Normalize a GitHub remote URL to ``host/owner/repo`` (lowercased).

    Ported from ``canonicalGitHubRemote`` in
    ``apps/desktop/electron/update-remote.ts``. Handles SSH
    (``git@github.com:owner/repo.git``), SSH URL scheme
    (``ssh://git@github.com/owner/repo.git``), and HTTPS URLs.
    """

    if not url:
        return ""

    value = url.strip()

    # git@github.com:owner/repo[.git]
    if value.startswith("git@github.com:"):
        value = "github.com/" + value[len("git@github.com:"):]

    # ssh://git@github.com/owner/repo[.git]
    elif value.startswith("ssh://git@github.com/"):
        value = "github.com/" + value[len("ssh://git@github.com/"):]

    else:
        # Try URL parsing for HTTPS forms (and anything URL-shaped).
        from urllib.parse import urlparse

        try:
            parsed = urlparse(value)
            if parsed.hostname and parsed.path:
                value = parsed.hostname + parsed.path
        except Exception:
            pass  # leave non-URL forms unchanged

    # Strip trailing slashes.
    value = value.strip().rstrip("/")

    # Strip .git suffix.
    if value.endswith(".git"):
        value = value[:-4]

    return value.lower()


def _is_official_remote(url: str) -> bool:
    """Return True if the remote URL points at the official repo."""
    return _canonical_github_remote(url) == OFFICIAL_REPO_CANONICAL


# ---------------------------------------------------------------------------
# Git helpers (all crash-proof)
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path) -> Optional[str]:
    """Run a git command, returning stdout (stripped) or None on failure."""

    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def _is_under_versions(path: Path) -> bool:
    """Check if ``path`` is under a ``versions/`` directory (a managed slot)."""

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    for parent in resolved.parents:
        if parent.name == "versions":
            return True
    return False


# Branches considered "known" / safe for adoption. The checkout should be
# on ``main`` (the default release branch) to be pristine.
_KNOWN_BRANCHES = frozenset({"main"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_legacy_install(
    project_root: Path, hermes_home: Path
) -> Optional[LegacyInfo]:
    """Detect whether this is a legacy git-checkout install that can adopt
    managed slots.

    Returns:
        None — running from a managed slot, or docker/nixos/homebrew/pip.
        LegacyInfo — for git checkouts, with ``pristine`` indicating whether
            the checkout is clean / official / on main / up-to-date.
    """

    # 1. Running from a managed slot → no adoption.
    if _is_under_versions(project_root):
        return None

    # 2. Non-git install methods → no adoption.
    try:
        method = detect_install_method(project_root)
    except Exception:
        return None

    if method != "git":
        return None

    # 3. We're a git checkout — determine pristine vs dirty.
    reasons: list[str] = []

    # 3a. Clean tree?
    porcelain = _git(["status", "--porcelain"], project_root)
    if porcelain is None:
        reasons.append("git status failed; cannot determine tree cleanliness")
    elif porcelain:
        reasons.append("dirty working tree")

    # 3b. Official origin?
    origin_url = _git(["remote", "get-url", "origin"], project_root)
    if origin_url is None:
        reasons.append("no origin remote found (or git failed)")
    elif not _is_official_remote(origin_url):
        reasons.append("fork remote (origin does not point to NousResearch/hermes-agent)")

    # 3c. On a known branch?
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
    if branch is None:
        reasons.append("cannot determine current branch")
    elif branch not in _KNOWN_BRANCHES:
        reasons.append(f"on branch '{branch}', not on main")

    # 3d. No local commits ahead of origin/main?
    #     ``origin/main..HEAD`` counts commits in HEAD not in origin/main
    #     (i.e. local commits not yet pushed).
    ahead = _git(["rev-list", "origin/main..HEAD", "--count"], project_root)
    if ahead is None:
        reasons.append("cannot determine commits ahead of origin/main (remote may not exist)")
    else:
        try:
            count = int(ahead)
        except ValueError:
            reasons.append("unexpected output from rev-list for commits-ahead check")
            count = 0
        if count > 0:
            reasons.append(f"{count} local commit(s) ahead of origin/main")

    pristine = len(reasons) == 0
    return LegacyInfo(pristine=pristine, reasons=reasons)
