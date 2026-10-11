#!/usr/bin/env python3
"""Audit (and auto-fix) contributor email mappings for a PR branch.

Mirrors the CI gate in .github/workflows/contributor-check.yml so salvage
branches never bounce off the check-attribution job. Run it from the branch
you are about to push:

    python3 scripts/audit_pr_attribution.py            # report only
    python3 scripts/audit_pr_attribution.py --fix      # create mapping files

Logic (kept in sync with contributor-check.yml):
  - scans ``git log $(git merge-base origin/main HEAD)..HEAD --format=%ae``
  - skips teknium/bot emails and ``<id>+<login>@users.noreply.github.com``
    (CI auto-resolves those)
  - everything else must have ``contributors/emails/<email>`` or a legacy
    AUTHOR_MAP entry in scripts/releases/authors_legacy.py

``--fix`` resolution order for an unmapped email:
  1. bare ``<login>@users.noreply.github.com`` → ``<login>``, verified via
     ``gh api users/<login>``. A warning is printed: the local part is
     *usually* the GitHub login but is user-controlled (the historical
     ``bryan@…`` → ``hydraxman`` case) — eyeball it against the PR author.
  2. ``gh api 'search/users?q=<email>+in:email'``
  3. otherwise: prints the manual ``add_contributor.py`` command and exits 1.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_SUBSTRINGS = (
    "teknium",
    "noreply@github.com",
    "dependabot",
    "github-actions",
    "anthropic.com",
    "cursor.com",
)
ID_NOREPLY_RE = re.compile(r"\d+\+.+@users\.noreply\.github\.com$")
BARE_NOREPLY_RE = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})@users\.noreply\.github\.com$")


def run(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        list(args), capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(REPO_ROOT),
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def new_emails() -> list[str]:
    base = run("git", "merge-base", "origin/main", "HEAD")
    log = run("git", "log", f"{base}..HEAD", "--format=%ae", "--no-merges", check=False)
    return sorted({e for e in log.splitlines() if e.strip()})


def is_mapped(email: str) -> bool:
    if any(s in email for s in SKIP_SUBSTRINGS):
        return True
    if ID_NOREPLY_RE.search(email):
        return True
    emails_dir = REPO_ROOT / "contributors" / "emails"
    if (emails_dir / email).is_file():
        return True
    folded = email.casefold()
    if emails_dir.is_dir() and any(path.name.casefold() == folded for path in emails_dir.iterdir()):
        return True
    authors_py = REPO_ROOT / "scripts" / "releases" / "authors_legacy.py"
    try:
        if f'"{email}"' in authors_py.read_text(encoding="utf-8-sig", errors="replace"):
            return True
    except OSError:
        pass
    return False


def gh_json(*args: str):
    try:
        out = run("gh", "api", *args, check=False)
        return json.loads(out) if out else None
    except (RuntimeError, json.JSONDecodeError, FileNotFoundError):
        return None


def resolve_login(email: str) -> tuple[str, str] | None:
    """Return (login, how) or None."""
    m = BARE_NOREPLY_RE.match(email)
    if m:
        login = m.group(1)
        user = gh_json(f"users/{login}")
        if user and user.get("login"):
            return user["login"], "bare-noreply local part (verified user exists)"
    found = gh_json(f"search/users?q={email}+in:email")
    if found and found.get("items"):
        return found["items"][0]["login"], "GitHub email search"
    return None


def verified_unlinked_author(email: str, sha: str, *, repo: Path = REPO_ROOT) -> bool:
    """Accept only an explicitly declared immutable source author, never an email wildcard.

    An upstream author without a linked GitHub account retains their raw git
    name in release notes. This declaration is not added to AUTHOR_MAP.
    """
    if not re.fullmatch(r"[^/\\\s]+@[^/\\\s]+", email) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return False
    path = repo / "contributors" / "unlinked" / f"{email}.json"
    try:
        if path.is_symlink():
            return False
        declaration = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(declaration, dict) or declaration.get("email") != email:
            return False
        records = declaration.get("commits")
        if not isinstance(records, list) or not records:
            return False
        if any(not isinstance(item, dict) for item in records):
            return False
        declared = [item.get("sha") for item in records]
        if any(not isinstance(value, str) for value in declared) or len(set(declared)) != len(declared):
            return False
        matches = [item for item in records if item.get("sha") == sha]
        if len(matches) != 1:
            return False
        record = matches[0]
        if record.get("source_url") != f"https://github.com/NousResearch/hermes-agent/commit/{sha}":
            return False
        if not isinstance(record.get("reason"), str) or not record["reason"].strip():
            return False
        identity = subprocess.run(
            ["git", "show", "-s", "--format=%ae%x00%an", sha],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.rstrip("\n").split("\0")
        return len(identity) == 2 and identity == [email, record.get("name")]
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def unlinked_email_is_verified(email: str) -> bool:
    base = run("git", "merge-base", "origin/main", "HEAD")
    authors = run("git", "log", f"{base}..HEAD", "--format=%H%x09%ae", "--no-merges")
    commits = [line.split("\t", 1)[0] for line in authors.splitlines()
               if "\t" in line and line.split("\t", 1)[1] == email]
    return bool(commits) and all(verified_unlinked_author(email, sha) for sha in commits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true",
                        help="auto-create contributors/emails/ mapping files")
    parser.add_argument("--verify-unlinked-email",
                        help="CI seam: every commit for this exact email must have a verified declaration")
    args = parser.parse_args()
    if args.verify_unlinked_email is not None:
        return 0 if unlinked_email_is_verified(args.verify_unlinked_email) else 1

    unmapped = [e for e in new_emails() if not is_mapped(e) and not unlinked_email_is_verified(e)]
    if not unmapped:
        print("✅ All contributor emails are mapped or verified unlinked source identities.")
        return 0

    failed = []
    for email in unmapped:
        author = run("git", "log", f"--author={email}", "--format=%an", "-1", check=False)
        if not args.fix:
            print(f"⚠️  unmapped: {email} ({author})")
            continue
        resolved = resolve_login(email)
        if resolved:
            login, how = resolved
            run("python3", "scripts/add_contributor.py", email, login)
            print(f"✔ mapped {email} -> {login}  [{how}]")
            if BARE_NOREPLY_RE.match(email):
                print(f"  ⚠ local part is user-controlled — confirm @{login} really is "
                      f"the contributor (git name: {author!r}) before pushing.")
        else:
            failed.append((email, author))

    if not args.fix:
        print("\nRun with --fix to auto-create mapping files, or manually:")
        for email in unmapped:
            print(f"    python3 scripts/add_contributor.py {email} <github-username>")
        return 1

    if failed:
        print("\nCould not auto-resolve; map manually:")
        for email, author in failed:
            print(f"    python3 scripts/add_contributor.py {email} <github-username>  # {author}")
        return 1

    print("\nDone — remember to `git add contributors && git commit`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
