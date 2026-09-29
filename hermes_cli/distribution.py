"""Where this build of the agent comes from — the ONE place that names its official repository.

Agent Czesiek is a fork of Hermes. Update checks, "am I on a fork?" detection, the ZIP-fallback
download and release links all have to point at the repository this build is *published from*,
not at the project it was forked from. Change the slug below (or set ``HERMES_DISTRIBUTION_REPO``
for a private mirror) and every consumer follows; never spell the repository out anywhere else.
"""

from __future__ import annotations

import os

_DEFAULT_SLUG = "aievolutionpl/AGENT_CZESIEK"


def repo_slug() -> str:
    """``owner/name`` of the official repository (env override for mirrors and tests)."""
    raw = os.environ.get("HERMES_DISTRIBUTION_REPO", "").strip().strip("/")
    return raw if raw.count("/") == 1 and " " not in raw else _DEFAULT_SLUG


def https_url() -> str:
    return f"https://github.com/{repo_slug()}.git"


def canonical() -> str:
    """``host/owner/name`` lowercased, the form remote URLs are normalised to for comparison."""
    return f"github.com/{repo_slug()}".lower()


def official_urls() -> set[str]:
    slug = repo_slug()
    return {
        f"https://github.com/{slug}.git",
        f"git@github.com:{slug}.git",
        f"https://github.com/{slug}",
        f"git@github.com:{slug}",
    }


def compare_api_url(base: str, head: str) -> str:
    return f"https://api.github.com/repos/{repo_slug()}/compare/{base}...{head}"


def release_tag_url_base() -> str:
    return f"https://github.com/{repo_slug()}/releases/tag"


def branch_zip_url(branch: str) -> str:
    return f"https://github.com/{repo_slug()}/archive/refs/heads/{branch}.zip"


def issues_url() -> str:
    return f"https://github.com/{repo_slug()}/issues"
