#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/_hermes-python" "$0" "$@"'
"""Hermes Agent release entrypoint.

Stable releases use the ``release``, ``publish``, and ``abandon`` subcommands.
Canary, commit, and dynamic-channel operations retain their top-level flags.
See ``website/docs/developer-guide/stable-releases.md`` for the operator flow.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Bootstrap the repo root onto sys.path so this script can import the
# canary-tag authority from hermes_cli.update_channel (hermes_cli/__init__.py
# is import-light: only os/sys + version constants).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_cli.update_channel import (  # noqa: E402
    _CANARY_TAG_RE, STABLE_TAG_RE, canary_tag_for_date, canary_timestamp,
    is_canary_tag,
)
from scripts.releases.authors import resolve_author  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def git(*args, cwd=None):
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=cwd or str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed: {result.stderr}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def git_result(*args, cwd=None):
    """Run a git command and return the full CompletedProcess."""
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=cwd or str(REPO_ROOT),
    )


def list_remotes() -> list[str]:
    """The configured git remote names, in git's order."""
    result = git_result("remote")
    if result.returncode != 0:
        return []
    return [name for name in result.stdout.split() if name]


def dispatch_desktop_build(tag: str, gh_repo: str | None) -> bool:
    """Dispatch the release pipeline after its draft exists.

    Stable workflows run on the tag so all reusable checks see the same
    commit. Their gate owns artifact publication and channel promotion.
    Canary builds keep default-branch workflow/cache scope and tagged inputs.
    Explicit dispatch also works for tags created by GITHUB_TOKEN.
    """
    canary = _CANARY_TAG_RE.fullmatch(tag) is not None
    if not canary and not STABLE_TAG_RE.fullmatch(tag):
        raise ValueError("Expected an exact stable or canary release tag")
    workflow = "desktop-bundled-release.yml" if canary else "stable-release.yml"
    cmd = ["gh", "workflow", "run", workflow, "--ref", "main" if canary else tag,
           "-f", f"tag={tag}"]
    if canary:
        cmd += ["-f", "upload_release=true"]
    if gh_repo:
        cmd += ["--repo", gh_repo]

    if not shutil.which("gh"):
        print("  ✗ Cannot start the release pipeline: `gh` CLI not found.")
        print(f"    Start it manually: {' '.join(cmd)}")
        return False

    dispatch_ref = (_default_branch(gh_repo) or "main") if canary else tag
    cmd[cmd.index("--ref") + 1] = dispatch_ref

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print(f"  ✗ Could not start the release pipeline: {result.stderr.strip()}")
        print(f"    Start it manually: {' '.join(cmd)}")
        return False

    print(f"  ✓ {workflow} started for {tag} (workflow from {dispatch_ref})")
    return True


def _default_branch(gh_repo: str | None) -> str | None:
    """The repo's default branch, resolved via gh. None on any failure."""
    cmd = ["gh", "repo", "view", "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"]
    if gh_repo:
        cmd += [gh_repo]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def resolve_push_remote(requested: str | None) -> str:
    """Pick the remote that receives the release push (and the GitHub
    release). One configured remote: use it. More than one: the release
    lands on whichever repo the tag is pushed to, so an implicit default
    is a foot-gun — require an explicit --remote.
    """
    remotes = list_remotes()
    if not remotes:
        raise SystemExit("release: no git remotes configured — nothing to push to")
    if requested:
        if requested not in remotes:
            raise SystemExit(
                f"release: remote {requested!r} is not configured "
                f"(available: {', '.join(remotes)})"
            )
        return requested
    if len(remotes) == 1:
        return remotes[0]
    raise SystemExit(
        "release: multiple remotes are configured "
        f"({', '.join(remotes)}) — pass --remote <name> to say which one "
        "receives the release push"
    )


def remote_github_repo(remote: str) -> str | None:
    """Read the GitHub repository that receives the remote's pushes."""
    result = git_result("remote", "get-url", "--push", remote)
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    match = re.fullmatch(
        r"(?:https?://(?:[^/@\s]+@)?github\.com/|"
        r"ssh://(?:[^/@\s]+@)?github\.com(?::\d+)?/|"
        r"git://github\.com/|[^/@:\s]+@github\.com:)"
        r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", url,
    )
    return match.group(1) if match else None


# Stable tags are matched with STABLE_TAG_RE and canary tags
# with _CANARY_TAG_RE, both imported from hermes_cli.update_channel — the
# single authority for both tag shapes (the stable major is capped at three
# digits so historical CalVer tags like v2026.7.20 never match; the canary
# identity is the exact stable core plus full UTC build metadata.
# Second precision allows several canaries per day; the fixed-width UTC stamp
# sorts chronologically even though SemVer ignores it for precedence.


def get_last_canary_tag():
    """The newest canonical canary receipt, by embedded build time."""
    raw = git("tag", "--list", "v*+canary.*")
    tags = [tag for tag in (raw.split("\n") if raw else []) if is_canary_tag(tag)]
    return max(tags, key=lambda tag: canary_timestamp(tag) or "", default=None)


def categorize_commit(subject: str) -> str:
    """Categorize a commit by its conventional commit prefix."""
    subject_lower = subject.lower()

    # Match conventional commit patterns
    patterns = {
        "breaking": [r"^breaking[\s:(]", r"^!:", r"BREAKING CHANGE"],
        "features": [r"^feat[\s:(]", r"^feature[\s:(]", r"^add[\s:(]"],
        "fixes": [r"^fix[\s:(]", r"^bugfix[\s:(]", r"^bug[\s:(]", r"^hotfix[\s:(]"],
        "improvements": [r"^improve[\s:(]", r"^perf[\s:(]", r"^enhance[\s:(]",
                         r"^refactor[\s:(]", r"^cleanup[\s:(]", r"^clean[\s:(]",
                         r"^update[\s:(]", r"^optimize[\s:(]"],
        "docs": [r"^doc[\s:(]", r"^docs[\s:(]"],
        "tests": [r"^test[\s:(]", r"^tests[\s:(]"],
        "chore": [r"^chore[\s:(]", r"^ci[\s:(]", r"^build[\s:(]",
                  r"^deps[\s:(]", r"^bump[\s:(]"],
    }

    for category, regexes in patterns.items():
        for regex in regexes:
            if re.match(regex, subject_lower):
                return category

    # Heuristic fallbacks
    if any(w in subject_lower for w in ["add ", "new ", "implement", "support "]):
        return "features"
    if any(w in subject_lower for w in ["fix ", "fixed ", "resolve", "patch "]):
        return "fixes"
    if any(w in subject_lower for w in ["refactor", "cleanup", "improve", "update "]):
        return "improvements"

    return "other"


def clean_subject(subject: str) -> str:
    """Clean up a commit subject for display."""
    # Remove conventional commit prefix
    cleaned = re.sub(r"^(feat|fix|docs|chore|refactor|test|perf|ci|build|improve|add|update|cleanup|hotfix|breaking|enhance|optimize|bugfix|bug|feature|tests|deps|bump)[\s:(!]+\s*", "", subject, flags=re.IGNORECASE)
    # Remove trailing issue refs that are redundant with PR links
    cleaned = cleaned.strip()
    # Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def parse_coauthors(body: str) -> list:
    """Extract Co-authored-by trailers from a commit message body.

    Returns a list of {'name': ..., 'email': ...} dicts.
    Filters out AI assistants and bots (Claude, Copilot, Cursor, etc.).
    """
    if not body:
        return []
    # AI/bot emails to ignore in co-author trailers
    _ignored_emails = {"noreply@anthropic.com", "noreply@github.com",
                       "cursoragent@cursor.com", "hermes@nousresearch.com"}
    _ignored_names = re.compile(r"^(Claude|Copilot|Cursor Agent|GitHub Actions?|dependabot|renovate)", re.IGNORECASE)
    pattern = re.compile(r"Co-authored-by:\s*(.+?)\s*<([^>]+)>", re.IGNORECASE)
    results = []
    for m in pattern.finditer(body):
        name, email = m.group(1).strip(), m.group(2).strip()
        if email in _ignored_emails or _ignored_names.match(name):
            continue
        results.append({"name": name, "email": email})
    return results


def get_commits(since_tag=None, until="HEAD", cwd=None):
    """Get commits in ``since_tag..until`` (or all of ``until`` if since_tag is None)."""
    if since_tag:
        range_spec = f"{since_tag}..{until}"
    else:
        range_spec = until

    # Format: hash<US>author_name<US>author_email<US>subject\0body
    # Using %x1f (unit separator) to avoid conflict with | in author names
    log = git(
        "log", range_spec,
        "--format=%H%x1f%an%x1f%ae%x1f%s%x00%b%x00",
        "--no-merges",
        cwd=cwd,
    )

    if not log:
        return []

    commits = []
    # Split on double-null to get each commit entry, since body ends with \0
    # and format ends with \0, each record ends with \0\0 between entries
    for entry in log.split("\0\0"):
        entry = entry.strip()
        if not entry:
            continue
        # Split on first null to separate "hash<US>name<US>email<US>subject" from "body"
        if "\0" in entry:
            header, body = entry.split("\0", 1)
            body = body.strip()
        else:
            header = entry
            body = ""
        parts = header.split("\x1f", 3)
        if len(parts) != 4:
            continue
        sha, name, email, subject = parts
        coauthor_info = parse_coauthors(body)
        coauthors = [resolve_author(ca["name"], ca["email"]) for ca in coauthor_info]
        commits.append({
            "sha": sha,
            "short_sha": sha[:8],
            "author_name": name,
            "author_email": email,
            "subject": subject,
            "category": categorize_commit(subject),
            "github_author": resolve_author(name, email),
            "coauthors": coauthors,
        })

    return commits


def get_pr_number(subject: str) -> str | None:
    """Extract PR number from commit subject if present."""
    match = re.search(r"#(\d+)", subject)
    if match:
        return match.group(1)
    return None


def generate_changelog(commits, tag_name, semver, repo_url="https://github.com/NousResearch/hermes-agent",
                       prev_tag=None, first_release=False, no_changelog=False):
    """Generate markdown changelog from categorized commits."""
    lines = []

    # Header
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    lines.append(f"# Hermes Agent v{semver} ({tag_name})")
    lines.append("")
    lines.append(f"**Release Date:** {date_str}")
    lines.append("")
    # The builds-pending job in desktop-bundled-release.yml replaces this
    # marker with a "builds in progress" link to the run as soon as the
    # workflow starts. The builds-table job then replaces the link with
    # the download tables once every matrix leg has uploaded its
    # artifacts to the R2 bucket (real object names, never predicted
    # ones; the release body links point at the R2 public URL). A release
    # whose matrix never finishes keeps the link — visibly unfinished,
    # and it points at the run that stopped.
    lines.append("<!-- HERMES_BUILDS_TABLE -->")
    lines.append("")

    if first_release:
        lines.append("> 🎉 **First official release!** This marks the beginning of regular weekly releases")
        lines.append("> for Hermes Agent. See below for everything included in this initial release.")
        lines.append("")

    all_authors = set()
    teknium_aliases = {"@teknium1"}
    if not no_changelog:
        # Group commits by category
        categories = defaultdict(list)

        for commit in commits:
            categories[commit["category"]].append(commit)
            author = commit["github_author"]
            if author not in teknium_aliases:
                all_authors.add(author)
            for coauthor in commit.get("coauthors", []):
                if coauthor not in teknium_aliases:
                    all_authors.add(coauthor)

        # Category display order and emoji
        category_order = [
            ("breaking", "⚠️ Breaking Changes"),
            ("features", "✨ Features"),
            ("improvements", "🔧 Improvements"),
            ("fixes", "🐛 Bug Fixes"),
            ("docs", "📚 Documentation"),
            ("tests", "🧪 Tests"),
            ("chore", "🏗️ Infrastructure"),
            ("other", "📦 Other Changes"),
        ]

        for cat_key, cat_title in category_order:
            cat_commits = categories.get(cat_key, [])
            if not cat_commits:
                continue

            lines.append(f"## {cat_title}")
            lines.append("")

            for commit in cat_commits:
                subject = clean_subject(commit["subject"])
                pr_num = get_pr_number(commit["subject"])
                author = commit["github_author"]

                # Build the line
                parts = [f"- {subject}"]
                if pr_num:
                    parts.append(f"([#{pr_num}]({repo_url}/pull/{pr_num}))")
                else:
                    parts.append(f"([`{commit['short_sha']}`]({repo_url}/commit/{commit['sha']}))")

                if author not in teknium_aliases:
                    parts.append(f"— {author}")

                lines.append(" ".join(parts))

            lines.append("")

    # Contributors section
    if all_authors:
        # Sort contributors by commit count
        author_counts = defaultdict(int)
        for commit in commits:
            author = commit["github_author"]
            if author not in teknium_aliases:
                author_counts[author] += 1
            for coauthor in commit.get("coauthors", []):
                if coauthor not in teknium_aliases:
                    author_counts[coauthor] += 1

        sorted_authors = sorted(author_counts.items(), key=lambda x: -x[1])

        lines.append("## 👥 Contributors")
        lines.append("")
        lines.append("Thank you to everyone who contributed to this release!")
        lines.append("")
        for author, count in sorted_authors:
            commit_word = "commit" if count == 1 else "commits"
            lines.append(f"- {author} ({count} {commit_word})")
        lines.append("")

    # Full changelog link
    if prev_tag:
        lines.append(f"**Full Changelog**: [{prev_tag}...{tag_name}]({repo_url}/compare/{prev_tag}...{tag_name})")
    else:
        lines.append(f"**Full Changelog**: [{tag_name}]({repo_url}/commits/{tag_name})")
    lines.append("")

    return "\n".join(lines)


def _resume_canary(tag: str, remote: str, repository: str, *, notes_file: Path | None = None) -> None:
    """Converge a tag-pushed canary through draft, dispatch, and protected head."""
    ref = f"refs/tags/{tag}"
    local_object = git("rev-parse", ref)
    commit = git("rev-parse", f"{ref}^{{commit}}")
    if git("cat-file", "-t", local_object) != "tag":
        raise ValueError("Canary receipt must be an annotated tag")
    remote_refs = dict(line.split()[::-1] for line in git(
        "ls-remote", remote, ref, f"{ref}^{{}}",
    ).splitlines())
    if remote_refs.get(ref) != local_object or remote_refs.get(f"{ref}^{{}}") != commit:
        raise ValueError("Canary receipt differs from its exact remote tag object")

    view = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repository, "--json", "tagName,isDraft,isPrerelease,url"],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    # A draft is served at an untagged-* URL, never releases/tag/<tag>; gh's
    # answer is the only working link to it.
    if view.returncode != 0:
        create = [
            "gh", "release", "create", tag, "--repo", repository,
            "--verify-tag", "--draft", "--prerelease",
            "--title", f"Hermes Agent canary {tag}",
        ]
        create.extend(["--notes-file", str(notes_file)] if notes_file else ["--generate-notes"])
        created = subprocess.run(
            create, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        if created.returncode != 0:
            raise ValueError(created.stderr.strip() or "Canary draft could not be recovered")
        draft_url = created.stdout.strip()
    else:
        release = json.loads(view.stdout)
        if (release.get("tagName") != tag or not isinstance(release.get("isDraft"), bool)
                or release.get("isPrerelease") is not True):
            raise ValueError("Canary release state differs from its receipt")
        draft_url = release.get("url") or ""

    from scripts.releases.versioning import published_channel_identity
    published = published_channel_identity(repository, "canary")
    if published is not None and published == (tag[1:], commit):
        print(f"✓ {tag} is already published.")
        return
    if not dispatch_desktop_build(tag, repository):
        raise SystemExit(1)
    print(f"Resumed canary publication for {tag}.")
    print(f"Workflow: https://github.com/{repository}/actions/workflows/desktop-bundled-release.yml")
    print("Wait for that workflow to finish. It builds this canary and publishes the draft when the build is green.")
    print(f"The draft is at {draft_url}.")


def cmd_canary(args) -> None:
    """--canary: tag + draft a canary source identity.

    The source identity is the newest stable version plus a full UTC timestamp
    in SemVer build metadata. It therefore compares equal to that stable; only
    the R2 canary head moves subscribers. No version-file bump or commit is
    created — the receipt tag points at HEAD as-is.

    Created as a DRAFT prerelease, for the same reason the stable path
    drafts: a published release with no installers attached is a release
    users can reach and cannot use. The desktop matrix attaches the
    installers to this draft by tag, and the canary workflow's publish
    job flips it to published only after that matrix is green. A failed
    bundle therefore leaves an inspectable draft rather than a broken
    canary.

    Exits 0 with "nothing to do" when HEAD is already tagged by the last
    canary — the skip-if-no-new-commits gate lives HERE, not in workflow
    YAML.
    """
    date_utc = args.date or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    push_remote = resolve_push_remote(args.remote)
    gh_repo = remote_github_repo(push_remote)
    if not gh_repo:
        raise ValueError("Canary publication requires a GitHub repository remote")
    prev_canary = get_last_canary_tag()
    if prev_canary:
        head = git("rev-parse", "HEAD")
        if head and head == git("rev-parse", f"{prev_canary}^{{commit}}"):
            if args.publish:
                _resume_canary(prev_canary, push_remote, gh_repo)
            else:
                print(f"✓ No new commits since {prev_canary} — nothing to do.")
            return
    from scripts.releases.versioning import published_stable_identity
    stable_version, stable_commit = published_stable_identity(gh_repo)
    tag_name = canary_tag_for_date(stable_version, date_utc)

    since = prev_canary or stable_commit

    if git_result("rev-parse", "--verify", "--quiet", f"refs/tags/{tag_name}").returncode == 0:
        if git("rev-parse", f"{tag_name}^{{commit}}") != git("rev-parse", "HEAD"):
            raise ValueError(f"{tag_name} already receipts a different commit")
        if args.publish:
            _resume_canary(tag_name, push_remote, gh_repo)
        else:
            print(f"✓ {tag_name} already exists — nothing to do.")
        return


    commits = get_commits(since_tag=since)
    if not commits:
        print(f"✓ No new commits since {since} — nothing to do.")
        return

    version = tag_name.lstrip("v")
    print(f"Canary: {tag_name} ({len(commits)} commits since {since})")
    changelog = generate_changelog(
        commits, tag_name, version, prev_tag=since, first_release=False, no_changelog=args.no_changelog
    )

    if not args.publish:
        print(changelog)
        print("\nDry run complete. To publish, add --publish")
        return

    tag_result = git_result(
        "tag", "-a", tag_name, "-m", f"Hermes Agent canary {date_utc}"
    )
    if tag_result.returncode != 0:
        print(f"✗ Failed to create tag {tag_name}: {tag_result.stderr.strip()}")
        sys.exit(1)
    push_result = git_result("push", push_remote, f"refs/tags/{tag_name}")
    if push_result.returncode != 0:
        print(f"✗ Failed to push {tag_name}: {push_result.stderr.strip()}")
        sys.exit(1)
    print(f"✓ Pushed {tag_name} to {push_remote}")

    changelog_file = REPO_ROOT / ".release_notes.md"
    changelog_file.write_text(changelog, encoding="utf-8")
    _resume_canary(tag_name, push_remote, gh_repo, notes_file=changelog_file)
    changelog_file.unlink(missing_ok=True)
    # Record the tag for any workflow step that wants it. release.py
    # starts the build itself, so nothing consumes this today; it stays
    # because a step output is the cheap, conventional handle for "which
    # tag did this run cut".
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"tag={tag_name}\n")


def _canary_date(tag: str) -> str | None:
    """The YYYYMMDD receipt date for a canonical canary tag."""
    if not is_canary_tag(tag):
        return None
    return tag.split("+canary.", 1)[1][:8]


def prune_old_canaries(args) -> None:
    """--prune-canaries: delete canary releases+tags older than 14 days.

    Keep-on-doubt: any parse failure keeps the release. The keep window is
    dated by the tag's own UTC suffix, not the release timestamp, so a
    re-published old tag never resets its clock.
    """
    push_remote = resolve_push_remote(args.remote)
    gh_repo = remote_github_repo(push_remote)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y%m%d")

    tags = git("tag", "--list", "v*+canary.*", "--sort=-creatordate")
    doomed = []
    for tag in (tags.split("\n") if tags else []):
        date = _canary_date(tag)
        if date is not None and date < cutoff:
            doomed.append(tag)
    if not doomed:
        print("✓ No canaries older than 14 days.")
        return
    for tag in doomed:
        if not args.publish:
            print(f"Would delete {tag}")
            continue
        gh_cmd = ["gh", "release", "delete", tag, "--yes", "--cleanup-tag"]
        if gh_repo:
            gh_cmd += ["--repo", gh_repo]
        result = subprocess.run(
            gh_cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            print(f"✓ Deleted {tag}")
        else:
            print(f"⚠ Could not delete {tag}: {result.stderr.strip()}")
    if not args.publish:
        print("Dry run complete. No release was deleted.")
        print("Run the same command with --publish to delete the tags above.")


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent Release Tool")
    parser.add_argument("--canary", action="store_true",
                        help="Tag + publish a stable-core canary "
                             "(v<stable>+canary.<YYYYMMDDTHHMMSSZ>); no-op when "
                             "HEAD has no new commits since the last canary")
    parser.add_argument("--build-commit", type=str, metavar="REV",
                        help="Preview an exact-commit build into releases/commit/<sha>/ on R2. "
                             "Add --publish to dispatch without a tag or release. The same "
                             "direct dispatch runs from any GitHub remote; add --channel NAME "
                             "to publish into an updatable R2 channel instead of a one-off.")
    parser.add_argument("--bundle-env", action="append", default=[], metavar="NAME=VALUE",
                        help="Bake a non-secret environment default into a commit desktop bundle. "
                             "Repeat for multiple variables. Runtime environment values win.")
    parser.add_argument("--bundle-unset", action="append", default=[], metavar="NAME",
                        help="Clear an inherited variable at desktop launch, even when set. "
                             "Uses an explicit empty value; repeat for multiple variables.")
    parser.add_argument("--prune-canaries", action="store_true",
                        help="Delete canary releases+tags older than 14 days")
    parser.add_argument("--publish", action="store_true",
                        help="Actually create the tag and GitHub release (otherwise dry run)")
    parser.add_argument("--remote", type=str,
                        help="Git remote that receives the release push and the GitHub "
                             "release. Required with --publish when more than one remote "
                             "is configured; the single remote is used when only one exists.")
    parser.add_argument("--date", type=str,
                        help="Override release date metadata (format: YYYY.M.D)")

    parser.add_argument("--no-changelog", action="store_true",
                        help="Skip changelog")
    subcommands = parser.add_subparsers(dest="command")
    release_cmd = subcommands.add_parser(
        "release", help="Claim a version, cut a draft, and dispatch the gate")
    release_cmd.add_argument("--commit", required=True, metavar="SHA",
                             help="The main commit to release")
    release_cmd.add_argument("--bump", choices=["major", "minor", "patch"], default="patch")
    release_cmd.add_argument("--autopublish", action="store_true",
                             help="Publish on green instead of leaving a draft")
    release_cmd.add_argument("--skip-bundles", action="store_true",
                             help="Release only the tag and the Docker image: no desktop, Termux or "
                                  "PM bundle builds, smokes, feeds or Store check. The desktop update "
                                  "channel stays on the previous bundle release.")
    release_cmd.add_argument("--skip-tests", action="store_true",
                             help="Emergency release: skip CI, Nix, PM bundle, install/update E2E, "
                                  "Termux, Windows, native smoke and upgrade acceptance jobs. "
                                  "Artifacts still build and publish.")
    # SUPPRESS keeps a --no-changelog given before the subcommand from being
    # reset by this parser's default.
    release_cmd.add_argument("--no-changelog", action="store_true", default=argparse.SUPPRESS,
                             help="Leave the commit list out of the draft body")
    release_cmd.add_argument("--remote", type=str)
    publish_cmd = subcommands.add_parser(
        "publish", help="Publish a green stable release through the ordered sequencer")
    publish_cmd.add_argument("--version", required=True)
    publish_cmd.add_argument("--remote", type=str)
    abandon_cmd = subcommands.add_parser(
        "abandon", help="Clear the outstanding attempt, keeping its attempt ref and writing an abandon marker")
    abandon_cmd.add_argument("--version", required=True)
    abandon_cmd.add_argument("--remote", type=str)
    from scripts.releases.channel_build import add_arguments, validate_arguments, cmd_channel

    add_arguments(parser)
    args = parser.parse_args()

    from scripts.releases.entrypoint import cmd_abandon, cmd_publish, cmd_release

    stable_commands = {"release": cmd_release, "publish": cmd_publish, "abandon": cmd_abandon}
    if args.command:
        stable_commands[args.command](args)
        return
    if validate_arguments(parser, args):
        cmd_channel(args)
        return
    if (args.bundle_env or args.bundle_unset) and args.build_commit is None:
        parser.error("--bundle-env and --bundle-unset require --build-commit")
    if args.build_commit is not None:
        conflicting = [name for name, supplied in (
            ("--canary", args.canary), ("--prune-canaries", args.prune_canaries),
            ("--date", args.date), ("--no-changelog", args.no_changelog),
        ) if supplied]
        if conflicting:
            parser.error("--build-commit cannot be combined with " + ", ".join(conflicting))
    from scripts.releases.commit_build import cmd_build_commit

    modes = (
        (args.build_commit is not None, cmd_build_commit),
        (args.canary, cmd_canary),
        (args.prune_canaries, prune_old_canaries),
    )
    for selected, handler in modes:
        if selected:
            handler(args)
            return
    parser.error("select release, publish, abandon, --canary, --build-commit, or a channel operation")


if __name__ == "__main__":
    main()
