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
# GitHub's *current* signup rules forbid consecutive hyphens, but legacy
# accounts with them exist and are valid (e.g. Roger--Han, verified via the
# users API July 2026). Accept any alphanumeric/hyphen login that doesn't
# start or end with a hyphen, max 39 chars.
_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")


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


# Emails whose filename would differ from another entry only by letter case
# cannot live in contributors/emails/ (same file on macOS/Windows): they go
# in this sidecar instead. See contributors/README.md and #88257.
CASELESS_SIDECAR = REPO_ROOT / "contributors" / "emails.caseless.json"


def _dir_names(emails_dir: Path | None = None) -> set[str]:
    """Actual on-disk mapping filenames (case as stored by the filesystem)."""
    emails_dir = emails_dir or EMAILS_DIR
    if not emails_dir.is_dir():
        return set()
    return {p.name for p in emails_dir.iterdir()}


def _caseless_collision(email: str, emails_dir: Path | None = None) -> bool:
    """True when another mapping file differs from ``email`` only by case."""
    names = _dir_names(emails_dir)
    return any(
        name != email and name.casefold() == email.casefold() for name in names
    )


def _sidecar_login(email: str) -> str | None:
    """Look the email up in contributors/emails.caseless.json."""
    try:
        import json

        data = json.loads(CASELESS_SIDECAR.read_text(encoding="utf-8"))
        login = data.get(email) if isinstance(data, dict) else None
        return str(login).lstrip("@") if login else None
    except (OSError, ValueError):
        return None


def _write_sidecar(email: str, login: str) -> None:
    import json

    try:
        data = json.loads(CASELESS_SIDECAR.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[email] = login
    CASELESS_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    CASELESS_SIDECAR.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def add_contributor(email: str, login: str, comment: str = "") -> int:
    email = email.strip()
    login = login.strip().lstrip("@")

    if not _EMAIL_RE.match(email):
        print(f"error: {email!r} does not look like a commit-author email", file=sys.stderr)
        return 2
    if not _LOGIN_RE.match(login):
        print(f"error: {login!r} is not a valid GitHub login", file=sys.stderr)
        return 2

    dir_names = _dir_names()
    colliding = any(
        name != email and name.casefold() == email.casefold()
        for name in dir_names
    )
    existing = None
    # Existence is checked against the on-disk NAME LIST, not path.is_file():
    # on case-insensitive filesystems the path of a case-variant email
    # resolves to the variant's file and would read the WRONG login. On
    # case-sensitive filesystems where both variants exist as files, the
    # exact-name mapping must still be honored (refuse-on-different-login).
    if email in dir_names:
        existing = read_mapping_file(EMAILS_DIR / email)
    if existing is None:
        existing = _sidecar_login(email)
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

    if colliding:
        # Writing contributors/emails/<email> would either clobber the
        # case-variant mapping on case-insensitive filesystems or recreate
        # the collision the sidecar exists to avoid.
        _write_sidecar(email, login)
        print(f"added: contributors/emails.caseless.json[{email}] -> {login} "
              "(case-collides with an existing mapping)")
        return 0

    EMAILS_DIR.mkdir(parents=True, exist_ok=True)
    path = EMAILS_DIR / email
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
