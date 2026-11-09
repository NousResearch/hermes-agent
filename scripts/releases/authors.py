"""Map commit authors to GitHub logins for release notes and attribution checks.

``AUTHOR_MAP`` merges the frozen ``LEGACY_AUTHOR_MAP`` (authors_legacy.py) with
the one-file-per-email entries under ``contributors/emails/``; the directory
wins on duplicates.
"""
from __future__ import annotations

import re
from pathlib import Path

from scripts.releases.authors_legacy import LEGACY_AUTHOR_MAP

REPO_ROOT = Path(__file__).resolve().parents[2]

# ──────────────────────────────────────────────────────────────────────
# Directory-based mappings: contributors/emails/<email> → login
# ──────────────────────────────────────────────────────────────────────
CONTRIBUTORS_EMAILS_DIR = REPO_ROOT / "contributors" / "emails"


def _load_contributor_dir(directory: Path | None = None) -> dict:
    """Load one-file-per-email mappings from contributors/emails/.

    Filename = commit-author email, first non-comment line = GitHub login.
    Additions never merge-conflict (each mapping is a distinct file), which
    is why new entries go here instead of the frozen LEGACY_AUTHOR_MAP.
    """
    directory = directory or CONTRIBUTORS_EMAILS_DIR
    mapping = {}
    if not directory.is_dir():
        return mapping
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    mapping[path.name] = line.lstrip("@")
                    break
        except OSError:
            continue
    return mapping


# Effective map: frozen legacy dict + directory entries (directory wins).
AUTHOR_MAP = {**LEGACY_AUTHOR_MAP, **_load_contributor_dir()}


def login_for_email(email: str, author_map: dict | None = None) -> str | None:
    """The mapped GitHub login for a RAW commit-author email (``git log %ae``).

    Email addresses are matched case-insensitively (``casefold``, like
    ``add_contributor.py``'s collision check): the same person commits as
    ``J.Doe@x`` and ``j.doe@x``, and one mapping file must credit both. The
    release collector and the PR attribution gate both resolve through here,
    so the gate can never certify an email the release notes would drop.
    Conflicting casefold-equivalent owners raise ``ValueError`` even for an
    exact key; lookup spelling or insertion order cannot select authority.
    """
    mapping = AUTHOR_MAP if author_map is None else author_map
    folded = email.casefold()
    logins = {login for known, login in mapping.items()
              if known.casefold() == folded and login}
    if len(logins) > 1:
        raise ValueError(f"ambiguous contributor identity for {email!r}")
    return next(iter(logins), None)


def resolve_author(name: str, email: str) -> str:
    """Resolve a git author to a GitHub @mention."""
    # Try email lookup first
    gh_user = login_for_email(email)
    if gh_user:
        return f"@{gh_user}"

    # Try noreply pattern
    noreply_match = re.match(r"(\d+)\+(.+)@users\.noreply\.github\.com", email)
    if noreply_match:
        return f"@{noreply_match.group(2)}"

    # Try username@users.noreply.github.com
    noreply_match2 = re.match(r"(.+)@users\.noreply\.github\.com", email)
    if noreply_match2:
        return f"@{noreply_match2.group(1)}"

    # Fallback to git name
    return name
