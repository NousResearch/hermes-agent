#!/usr/bin/env python3
"""Contributor Audit Script

Cross-references git authors, Co-authored-by trailers, and salvaged PR
descriptions to find any contributors missing from the release notes.

Usage:
    # Basic audit since a tag
    python scripts/contributor_audit.py --since-tag v2026.4.8

    # Audit with a custom endpoint
    python scripts/contributor_audit.py --since-tag v2026.4.8 --until v2026.4.13

    # Compare against a release notes file
    python scripts/contributor_audit.py --since-tag v2026.4.8 --release-file RELEASE_v0.9.0.md
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Import AUTHOR_MAP and resolve_author from the sibling release.py module
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from release import AUTHOR_MAP, resolve_author  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# AI assistants, bots, and machine accounts to exclude from contributor lists
# ---------------------------------------------------------------------------
IGNORED_PATTERNS = [
    re.compile(r"^Claude", re.IGNORECASE),
    re.compile(r"^Copilot$", re.IGNORECASE),
    re.compile(r"^Cursor\s+Agent$", re.IGNORECASE),
    re.compile(r"^GitHub\s*Actions?$", re.IGNORECASE),
    re.compile(r"^dependabot", re.IGNORECASE),
    re.compile(r"^renovate", re.IGNORECASE),
    re.compile(r"^Hermes\s+(Agent|Audit)$", re.IGNORECASE),
    re.compile(r"^Ubuntu$", re.IGNORECASE),
]

IGNORED_EMAILS = {
    "noreply@anthropic.com",
    "noreply@github.com",
    "cursoragent@cursor.com",
    "hermes@nousresearch.com",
    "hermes-audit@example.com",
    "hermes@habibilabs.dev",
}


def is_ignored(handle: str, email: str = "") -> bool:
    """Return True if this contributor is a bot/AI/machine account."""
    if email in IGNORED_EMAILS:
        return True
    for pattern in IGNORED_PATTERNS:
        if pattern.search(handle):
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args, cwd=None):
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=cwd or str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"  [warn] git {' '.join(args)} failed: {result.stderr.strip()}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def gh_pr_list():
    """Fetch merged PRs from GitHub using the gh CLI.

    Returns a list of dicts with keys: number, title, body, author.
    Returns an empty list if gh is not available or the call fails.
    """
    try:
        result = subprocess.run(
            [
                "gh", "pr", "list",
                "--repo", "NousResearch/hermes-agent",
                "--state", "merged",
                "--json", "number,title,body,author,mergedAt",
                "--limit", "300",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  [warn] gh pr list failed: {result.stderr.strip()}", file=sys.stderr)
            return []
        return json.loads(result.stdout)
    except FileNotFoundError:
        print("  [warn] 'gh' CLI not found — skipping salvaged PR scan.", file=sys.stderr)
        return []
    except subprocess.TimeoutExpired:
        print("  [warn] gh pr list timed out — skipping salvaged PR scan.", file=sys.stderr)
        return []
    except json.JSONDecodeError:
        print("  [warn] gh pr list returned invalid JSON — skipping salvaged PR scan.", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Contributor collection
# ---------------------------------------------------------------------------

# Patterns that indicate salvaged/cherry-picked/co-authored work in PR bodies
SALVAGE_PATTERNS = [
    # "Salvaged from @username" or "Salvaged from #123"
    re.compile(r"[Ss]alvaged\s+from\s+@(\w[\w-]*)"),
    re.compile(r"[Ss]alvaged\s+from\s+#(\d+)"),
    # "Cherry-picked from @username"
    re.compile(r"[Cc]herry[- ]?picked\s+from\s+@(\w[\w-]*)"),
    # "Based on work by @username"
    re.compile(r"[Bb]ased\s+on\s+work\s+by\s+@(\w[\w-]*)"),
    # "Original PR by @username"
    re.compile(r"[Oo]riginal\s+PR\s+by\s+@(\w[\w-]*)"),
    # "Co-authored with @username"
    re.compile(r"[Cc]o[- ]?authored\s+with\s+@(\w[\w-]*)"),
]

# Pattern for Co-authored-by trailers in commit messages
CO_AUTHORED_RE = re.compile(
    r"Co-authored-by:\s*(.+?)\s*<([^>]+)>",
    re.IGNORECASE,
)


def collect_commit_authors(since_tag, until="HEAD"):
    """Collect contributors from git commit authors.

    Returns:
        contributors: dict mapping github_handle -> set of source labels
        unknown_emails: dict mapping email -> git name (for emails not in AUTHOR_MAP)
    """
    range_spec = f"{since_tag}..{until}"
    log = git(
        "log", range_spec,
        "--format=%H|%an|%ae|%s",
        "--no-merges",
    )

    contributors = defaultdict(set)
    unknown_emails = {}

    if not log:
        return contributors, unknown_emails

    for line in log.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        _sha, name, email, _subject = parts

        handle = resolve_author(name, email)
        # resolve_author returns "@handle" or plain name
        if handle.startswith("@"):
            contributors[handle.lstrip("@")].add("commit")
        else:
            # Could not resolve — record as unknown
            contributors[handle].add("commit")
            unknown_emails[email] = name

    return contributors, unknown_emails


def collect_co_authors(since_tag, until="HEAD"):
    """Collect contributors from Co-authored-by trailers in commit messages.

    Returns:
        contributors: dict mapping github_handle -> set of source labels
        unknown_emails: dict mapping email -> git name
    """
    range_spec = f"{since_tag}..{until}"
    # Get full commit messages to scan for trailers
    log = git(
        "log", range_spec,
        "--format=__COMMIT__%H%n%b",
        "--no-merges",
    )

    contributors = defaultdict(set)
    unknown_emails = {}

    if not log:
        return contributors, unknown_emails

    for line in log.split("\n"):
        match = CO_AUTHORED_RE.search(line)
        if match:
            name = match.group(1).strip()
            email = match.group(2).strip()
            handle = resolve_author(name, email)
            if handle.startswith("@"):
                contributors[handle.lstrip("@")].add("co-author")
            else:
                contributors[handle].add("co-author")
                unknown_emails[email] = name

    return contributors, unknown_emails


def collect_salvaged_contributors(since_tag, until="HEAD"):
    """Scan merged PR bodies for salvage/cherry-pick/co-author attribution.

    Uses the gh CLI to fetch PRs, then filters to the date range defined
    by since_tag..until and scans bodies for salvage patterns.

    Returns:
        contributors: dict mapping github_handle -> set of source labels
        pr_refs: dict mapping github_handle -> list of PR numbers where found
    """
    contributors = defaultdict(set)
    pr_refs = defaultdict(list)

    # Determine the date range from git tags/refs
    since_date = git("log", "-1", "--format=%aI", since_tag)
    if until == "HEAD":
        until_date = git("log", "-1", "--format=%aI", "HEAD")
    else:
        until_date = git("log", "-1", "--format=%aI", until)

    if not since_date:
        print(f"  [warn] Could not resolve date for {since_tag}", file=sys.stderr)
        return contributors, pr_refs

    prs = gh_pr_list()
    if not prs:
        return contributors, pr_refs

    for pr in prs:
        # Filter by merge date if available
        merged_at = pr.get("mergedAt", "")
        if merged_at and since_date:
            if merged_at < since_date:
                continue
            if until_date and merged_at > until_date:
                continue

        body = pr.get("body") or ""
        pr_number = pr.get("number", "?")

        # Also credit the PR author
        pr_author = pr.get("author", {})
        pr_author_login = pr_author.get("login", "") if isinstance(pr_author, dict) else ""

        for pattern in SALVAGE_PATTERNS:
            for match in pattern.finditer(body):
                value = match.group(1)
                # If it's a number, it's a PR reference — skip for now
                # (would need another API call to resolve PR author)
                if value.isdigit():
                    continue
                contributors[value].add("salvage")
                pr_refs[value].append(pr_number)

    return contributors, pr_refs


# ---------------------------------------------------------------------------
# Release file comparison
# ---------------------------------------------------------------------------

def check_release_file(release_file, all_contributors):
    """Check which contributors are mentioned in the release file.

    Returns:
        mentioned: set of handles found in the file
        missing: set of handles NOT found in the file
    """
    try:
        content = Path(release_file).read_text()
    except FileNotFoundError:
        print(f"  [error] Release file not found: {release_file}", file=sys.stderr)
        return set(), set(all_contributors)

    mentioned = set()
    missing = set()
    content_lower = content.lower()

    for handle in all_contributors:
        # Check for @handle or just handle (case-insensitive)
        if f"@{handle.lower()}" in content_lower or handle.lower() in content_lower:
            mentioned.add(handle)
        else:
            missing.add(handle)

    return mentioned, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit contributors across git history, co-author trailers, and salvaged PRs.",
    )
    parser.add_argument(
        "--since-tag",
        required=True,
        help="Git tag to start from (e.g., v2026.4.8)",
    )
    parser.add_argument(
        "--until",
        default="HEAD",
        help="Git ref to end at (default: HEAD)",
    )
    parser.add_argument(
        "--release-file",
        default=None,
        help="Path to a release notes file to check for missing contributors",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if new unmapped emails are found (for CI)",
    )
    parser.add_argument(
        "--diff-base",
        default=None,
        help="Git ref to diff against (only flag emails from commits after this ref)",
    )
    args = parser.parse_args()

    print(f"=== Contributor Audit: {args.since_tag}..{args.until} ===")
    print()

    # ---- 1. Git commit authors ----
    print("[1/3] Scanning git commit authors...")
    commit_contribs, commit_unknowns = collect_commit_authors(args.since_tag, args.until)
    print(f"      Found {len(commit_contribs)} contributor(s) from commits.")

    # ---- 2. Co-authored-by trailers ----
    print("[2/3] Scanning Co-authored-by trailers...")
    coauthor_contribs, coauthor_unknowns = collect_co_authors(args.since_tag, args.until)
    print(f"      Found {len(coauthor_contribs)} contributor(s) from co-author trailers.")

    # ---- 3. Salvaged PRs ----
    print("[3/3] Scanning salvaged/cherry-picked PR descriptions...")
    salvage_contribs, salvage_pr_refs = collect_salvaged_contributors(args.since_tag, args.until)
    print(f"      Found {len(salvage_contribs)} contributor(s) from salvaged PRs.")

    # ---- Merge all contributors ----
    all_contributors = defaultdict(set)
    for handle, sources in commit_contribs.items():
        all_contributors[handle].update(sources)
    for handle, sources in coauthor_contribs.items():
        all_contributors[handle].update(sources)
    for handle, sources in salvage_contribs.items():
        all_contributors[handle].update(sources)

    # Merge unknown emails
    all_unknowns = {}
    all_unknowns.update(commit_unknowns)
    all_unknowns.update(coauthor_unknowns)

    # Filter out AI assistants, bots, and machine accounts
    ignored = {h for h in all_contributors if is_ignored(h)}
    for h in ignored:
        del all_contributors[h]
    # Also filter unknowns by email
    all_unknowns = {e: n for e, n in all_unknowns.items() if not is_ignored(n, e)}

    # ---- Output ----
    print()
    print(f"=== All Contributors ({len(all_contributors)}) ===")
    print()

    # Sort by handle, case-insensitive
    for handle in sorted(all_contributors.keys(), key=str.lower):
        sources = sorted(all_contributors[handle])
        source_str = ", ".join(sources)
        extra = ""
        if handle in salvage_pr_refs:
            pr_nums = salvage_pr_refs[handle]
            extra = f"  (PRs: {', '.join(f'#{n}' for n in pr_nums)})"
        print(f"  @{handle}  [{source_str}]{extra}")

    # ---- Unknown emails ----
    if all_unknowns:
        print()
        print(f"=== Unknown Emails ({len(all_unknowns)}) ===")
        print("These emails are not in AUTHOR_MAP and should be added:")
        print()
        for email, name in sorted(all_unknowns.items()):
            print(f'  "{email}": "{name}",')

    # ---- Strict mode: fail CI if new unmapped emails are introduced ----
    if args.strict and all_unknowns:
        # In strict mode, check if ANY unknown emails come from commits in this
        # PR's diff range (new unmapped emails that weren't there before).
        # This is the CI gate: existing unknowns are grandfathered, but new
        # commits must have their author email in AUTHOR_MAP.
        new_unknowns = {}
        if args.diff_base:
            # Only flag emails from commits after diff_base
            new_commits_output = git(
                "log", f"{args.diff_base}..HEAD",
                "--format=%ae", "--no-merges",
            )
            new_emails = set(new_commits_output.splitlines()) if new_commits_output else set()
            for email, name in all_unknowns.items():
                if email in new_emails:
                    new_unknowns[email] = name
        else:
            new_unknowns = all_unknowns

        if new_unknowns:
            print()
            print(f"=== STRICT MODE FAILURE: {len(new_unknowns)} new unmapped email(s) ===")
            print("Add these to AUTHOR_MAP in scripts/release.py before merging:")
            print()
            for email, name in sorted(new_unknowns.items()):
                print(f'    "{email}": "<github-username>",')
            print()
            print("To find the GitHub username:")
            print("  gh api 'search/users?q=EMAIL+in:email' --jq '.items[0].login'")
            strict_failed = True
        else:
            strict_failed = False
    else:
        strict_failed = False

    # ---- Release file comparison ----
    if args.release_file:
        print()
        print(f"=== Release File Check: {args.release_file} ===")
        print()
        mentioned, missing = check_release_file(args.release_file, all_contributors.keys())
        print(f"  Mentioned in release notes: {len(mentioned)}")
        print(f"  Missing from release notes: {len(missing)}")
        if missing:
            print()
            print("  Contributors NOT mentioned in the release file:")
            for handle in sorted(missing, key=str.lower):
                sources = sorted(all_contributors[handle])
                print(f"    @{handle}  [{', '.join(sources)}]")
        else:
            print()
            print("  All contributors are mentioned in the release file!")

    print()
    print("Done.")

    if strict_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
