#!/usr/bin/env python3
"""Add a contributor email → GitHub login mapping.

Writes one file per email under contributors/emails/ (filename = email,
content = login). File additions never merge-conflict, unlike the legacy
AUTHOR_MAP dict in scripts/release.py, which is frozen — do not append to it.

Usage (from the repo root):
    python3 scripts/add_contributor.py <email> <github-login> [comment...]

    # e.g.
    python3 scripts/add_contributor.py jane@example.com janedoe "PR #12345 salvage"

Idempotent: if the mapping already exists with the same login, prints
"present" and exits 0. If the email maps to a DIFFERENT login (here or in the
legacy AUTHOR_MAP), refuses with exit 1 so a typo can't silently reassign
someone's commits.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EMAILS_DIR = REPO_ROOT / "contributors" / "emails"

_EMAIL_RE = re.compile(r"^[^/\\\s]+@[^/\\\s]+$")
_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")


def read_mapping_file(path: Path) -> str | None:
    """Return the login from a mapping file (first non-comment line)."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    except OSError:
        pass
    return None


def _legacy_login(email: str) -> str | None:
    """Look the email up in the frozen legacy AUTHOR_MAP in release.py."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from release import LEGACY_AUTHOR_MAP  # noqa: PLC0415

        return LEGACY_AUTHOR_MAP.get(email)
    except Exception:
        return None


def add_contributor(email: str, login: str, comment: str = "") -> int:
    email = email.strip()
    login = login.strip().lstrip("@")

    if not _EMAIL_RE.match(email):
        print(f"error: {email!r} does not look like a commit-author email", file=sys.stderr)
        return 2
    if not _LOGIN_RE.match(login):
        print(f"error: {login!r} is not a valid GitHub login", file=sys.stderr)
        return 2

    path = EMAILS_DIR / email
    existing = read_mapping_file(path) if path.is_file() else None
    if existing is None:
        existing = _legacy_login(email)
    if existing is not None:
        if existing == login:
            print("present")
            return 0
        print(
            f"error: {email} already maps to {existing!r} (asked for {login!r}) — "
            "resolve manually",
            file=sys.stderr,
        )
        return 1

    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    body = login + "\n"
    if comment:
        body += f"# {comment}\n"
    path.write_text(body, encoding="utf-8")
    print(f"added: contributors/emails/{email} -> {login}")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    email, login = sys.argv[1], sys.argv[2]
    comment = " ".join(sys.argv[3:])
    return add_contributor(email, login, comment)


if __name__ == "__main__":
    sys.exit(main())
