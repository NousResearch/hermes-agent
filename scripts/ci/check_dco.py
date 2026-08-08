#!/usr/bin/env python3
"""Verify every non-merge, non-bot commit in a range carries a DCO sign-off.

A commit satisfies the Developer Certificate of Origin
(https://developercertificate.org) when it has a ``Signed-off-by:`` *trailer*
whose ``Name <email>`` identity matches **either the commit's author or its
committer**.

Two deliberate choices, both prompted by review of the first draft of this
check:

* **Author *or* committer.** ``git commit -s`` appends the sign-off for the
  *committer*, which is not always the author (e.g. ``git commit --author=... -s``
  or an applied patch). Requiring the trailer to match only the author would
  reject those documented, valid workflows, so a sign-off from either identity
  is accepted — the same rule the reference DCO GitHub App uses.
* **Parsed trailer, not a substring.** Sign-offs are extracted with
  ``git interpret-trailers --parse`` so only a real trailer in the message's
  trailer block counts. A ``Signed-off-by:`` line sitting in ordinary prose
  does not satisfy the check.

Bot commits (Dependabot, GitHub Actions, GitHub noreply) are exempt, mirroring
``contributor-check.yml``.

Usage::

    python3 scripts/ci/check_dco.py [<base-ref>] [<head-ref>]

Defaults to ``origin/main`` and ``HEAD``. Exits non-zero (and prints the exact
remediation commands) when any commit is missing a matching sign-off trailer.
"""

from __future__ import annotations

import subprocess
import sys

# Author/committer emails we never require a sign-off from, matched as
# substrings the same way contributor-check.yml exempts them.
_BOT_EMAIL_MARKERS = ("noreply@github.com", "dependabot", "github-actions")


def format_identity(name: str, email: str) -> str:
    """Render a git identity the way a ``Signed-off-by`` trailer does."""
    return f"{name} <{email}>"


def is_bot_email(email: str) -> bool:
    """True for automation identities that are exempt from sign-off."""
    lowered = email.lower()
    return any(marker in lowered for marker in _BOT_EMAIL_MARKERS)


def parse_signoff_trailers(parsed_trailers: str) -> list[str]:
    """Pull ``Signed-off-by`` values out of ``git interpret-trailers --parse`` output.

    The input is the newline-separated ``Key: Value`` block that
    ``git interpret-trailers --parse`` emits — only real trailers appear there,
    so a ``Signed-off-by:`` line buried in prose never reaches this function.
    """
    signoffs: list[str] = []
    for line in parsed_trailers.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "signed-off-by":
            signoffs.append(value.strip())
    return signoffs


def is_signed_off(author_identity: str, committer_identity: str, signoffs: list[str]) -> bool:
    """True when a sign-off matches the author or committer (case-insensitively)."""
    accepted = {author_identity.strip().lower(), committer_identity.strip().lower()}
    return any(s.strip().lower() in accepted for s in signoffs)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def _signoffs_for(sha: str) -> list[str]:
    body = _git("show", "-s", "--format=%B", sha)
    parsed = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        input=body,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_signoff_trailers(parsed)


def load_commit(sha: str) -> dict:
    """Collect the identity, subject, and sign-off trailers for one commit."""
    fields = _git(
        "show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%s", sha
    ).rstrip("\n")
    an, ae, cn, ce, subject = fields.split("\x00", 4)
    return {
        "sha": sha,
        "subject": subject,
        "author_identity": format_identity(an, ae),
        "committer_identity": format_identity(cn, ce),
        "author_email": ae,
        "committer_email": ce,
        "signoffs": _signoffs_for(sha),
    }


def commit_range(base_ref: str, head_ref: str) -> list[str]:
    merge_base = _git("merge-base", base_ref, head_ref).strip()
    revs = _git("rev-list", "--no-merges", f"{merge_base}..{head_ref}").split()
    return revs


def main(argv: list[str]) -> int:
    base_ref = argv[1] if len(argv) > 1 else "origin/main"
    head_ref = argv[2] if len(argv) > 2 else "HEAD"

    shas = commit_range(base_ref, head_ref)
    if not shas:
        print("No commits to check.")
        return 0

    missing: list[dict] = []
    for sha in shas:
        commit = load_commit(sha)
        is_bot = is_bot_email(commit["author_email"]) or is_bot_email(commit["committer_email"])
        signed_off = is_signed_off(
            commit["author_identity"], commit["committer_identity"], commit["signoffs"]
        )
        if is_bot:
            print(f"skip (bot)  {sha[:9]}  {commit['author_email']}")
        elif signed_off:
            print(f"ok          {sha[:9]}  {commit['subject']}")
        else:
            missing.append(commit)

    if missing:
        print("")
        print("Commit(s) missing a matching Signed-off-by trailer:")
        for commit in missing:
            print(f"  {commit['sha'][:9]}  {commit['subject']}")
            print(f"      expected a trailer matching: {commit['author_identity']}")
            print(f"      or the committer:            {commit['committer_identity']}")
        print("")
        print("Sign off your commits to certify the Developer Certificate of Origin")
        print("(https://developercertificate.org):")
        print("")
        print("  # sign the most recent commit")
        print("  git commit --amend --no-edit -s")
        print("")
        print("  # sign every commit on this branch, then force-push")
        print("  git rebase --exec 'git commit --amend --no-edit -s' origin/main")
        print("  git push --force-with-lease")
        print("")
        print("Add -s to future commits (git commit -s) to sign off automatically.")
        return 1

    print("")
    print("All commits are signed off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
