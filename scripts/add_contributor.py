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

Also refuses with exit 1 when the email differs only by CASE from an existing
mapping file. Email hostnames are case-insensitive per DNS, so the two are one
address — and on a case-insensitive filesystem (macOS/Windows) the two files
cannot coexist, leaving the checkout permanently dirty and blocking
``git rebase``. Refusal is deliberate even when the login matches: a variant
spelling means two commit authors share one machine-default email, and only a
human can decide who owns it.
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


def find_mapping_file(email: str, emails_dir: Path | None = None) -> str | None:
    """Return the mapping filename matching ``email`` case-insensitively.

    A mapping filename IS an email address, and the hostname part of an email
    is case-insensitive per DNS — so two files differing only by case are two
    spellings of one address.

    This compares names from ``iterdir()`` rather than probing with
    ``Path.is_file()``, on purpose: ``is_file()`` answers True for a variant
    spelling on a case-insensitive filesystem (macOS/Windows) and False on a
    case-sensitive one (Linux), so any check built on it behaves differently
    per platform. Comparing real directory entry names is a pure string
    operation and gives the same answer everywhere — which is what makes this
    testable on any host.

    Returns the exact stored spelling (which may equal ``email``), or ``None``.
    """
    directory = EMAILS_DIR if emails_dir is None else emails_dir
    # casefold(), not lower(): it folds pairs lower() misses (German ß/ss), and
    # a case-insensitive filesystem's own folding is at least as aggressive, so
    # this must not be narrower than the filesystem's.
    target = email.casefold()
    try:
        entries = list(directory.iterdir())
    except OSError:
        return None
    exact: str | None = None
    variant: str | None = None
    for entry in entries:
        if entry.name.casefold() != target or not entry.is_file():
            continue
        if entry.name == email:
            exact = entry.name
        elif variant is None:
            variant = entry.name
    # Prefer the exact spelling when both are present (a case-sensitive
    # filesystem can hold both, which is the state this guard exists to stop
    # from spreading).
    return exact if exact is not None else variant


def find_case_variant(email: str, emails_dir: Path | None = None) -> str | None:
    """Return an existing mapping filename differing from ``email`` only by case.

    ``None`` for an exact match (that is the normal update path) or no match.
    Two such files cannot both exist on a case-insensitive checkout: one shows
    as permanently modified and ``git rebase`` refuses to run at all.
    """
    found = find_mapping_file(email, emails_dir)
    return None if found is None or found == email else found


def _legacy_login(email: str) -> str | None:
    """Look the email up in the frozen legacy AUTHOR_MAP in release.py."""
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from release import LEGACY_AUTHOR_MAP  # noqa: PLC0415

        return LEGACY_AUTHOR_MAP.get(email)
    except Exception:
        return None


def _case_collision(email: str) -> str | None:
    """An existing mapping whose filename differs from `email` only in case.

    Returns the colliding filename, or None. Exact matches are not collisions --
    that is the ordinary "already mapped" path handled by the caller.
    """
    if not EMAILS_DIR.is_dir():
        return None

    # casefold (not lower) matches how macOS/Windows fold non-ASCII text —
    # same key scripts/check-case-collisions.py uses repo-wide.
    folded = email.casefold()
    for entry in EMAILS_DIR.iterdir():
        if entry.name != email and entry.name.casefold() == folded:
            return entry.name
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

    # Refuse before touching the directory: adding a second spelling would
    # produce a pair of files that cannot coexist on a case-insensitive
    # checkout. Refusing (rather than silently reusing the existing file) is
    # deliberate — a variant spelling means two commit authors share one
    # machine-default email, and only a human can decide who owns it.
    variant = find_case_variant(email)
    if variant is not None:
        variant_login = read_mapping_file(EMAILS_DIR / variant)
        print(
            f"error: {email} differs only by case from the existing mapping "
            f"{variant} -> {variant_login!r} (asked for {login!r}).\n"
            f"  Email hostnames are case-insensitive, so these are one address, "
            f"and both files cannot coexist on a macOS/Windows checkout.\n"
            f"  Resolve manually: decide which login owns this address, then "
            f"either reuse {variant} as-is or rename it — do not add a second "
            f"spelling.",
            file=sys.stderr,
        )
        return 1

    path = EMAILS_DIR / email

    # One file per email means the FILENAME is the key, and on a
    # case-insensitive filesystem (Windows, default macOS) two emails differing
    # only in case are the same file. Creating both makes the repo impossible to
    # check out cleanly there -- `git status` reports a phantom modification
    # forever, because whichever file git wrote second wins on disk. Refuse for
    # the same reason a conflicting login is refused: resolve it deliberately.
    collision = _case_collision(email)
    if collision is not None:
        print(
            f"error: {email} collides with existing mapping {collision} on "
            "case-insensitive filesystems (Windows/macOS) — the two are the same "
            "file there. Reuse that mapping, or resolve manually.",
            file=sys.stderr,
        )
        return 1

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
