#!/usr/bin/env python3
"""Hermes Agent Release Script

Generates changelogs and creates GitHub releases with CalVer tags.

Usage:
    # Preview changelog (dry run)
    python scripts/release.py

    # Preview with semver bump
    python scripts/release.py --bump minor

    # Create the release
    python scripts/release.py --bump minor --publish

    # First release (no previous tag)
    python scripts/release.py --bump minor --publish --first-release

    # Override CalVer date (e.g. for a belated release)
    python scripts/release.py --bump minor --publish --date 2026.3.15
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "hermes_cli" / "__init__.py"
PYPROJECT_FILE = REPO_ROOT / "pyproject.toml"

# ──────────────────────────────────────────────────────────────────────
# Contributor identity resolution (git email → GitHub login) lives in
# scripts/release_contributors.py (Part of #79962, #78647). It is re-exported
# here so ``release.<name>`` and ``from release import <name>`` keep resolving
# to the identical objects.
# ──────────────────────────────────────────────────────────────────────

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_contributors import (  # noqa: E402, F401 — re-exported for `release.<name>`
    AUTHOR_MAP,
    CONTRIBUTORS_EMAILS_DIR,
    LEGACY_AUTHOR_MAP,
    _load_contributor_dir,
    resolve_author,
)


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


def get_last_tag():
    """Get the most recent CalVer tag."""
    tags = git("tag", "--list", "v20*", "--sort=-v:refname")
    if tags:
        return tags.split("\n")[0]
    return None


def next_available_tag(base_tag: str) -> tuple[str, str]:
    """Return a tag/calver pair, suffixing same-day releases when needed."""
    if not git("tag", "--list", base_tag):
        return base_tag, base_tag.removeprefix("v")

    suffix = 2
    while git("tag", "--list", f"{base_tag}.{suffix}"):
        suffix += 1
    tag_name = f"{base_tag}.{suffix}"
    return tag_name, tag_name.removeprefix("v")


def get_current_version():
    """Read current semver from __init__.py."""
    content = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    return match.group(1) if match else "0.0.0"


def bump_version(current: str, part: str) -> str:
    """Bump a semver version string."""
    parts = current.split(".")
    if len(parts) != 3:
        parts = ["0", "0", "0"]
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump part: {part}")

    return f"{major}.{minor}.{patch}"


def update_version_files(semver: str, calver_date: str):
    """Update version strings in source files."""
    # Update __init__.py
    content = VERSION_FILE.read_text(encoding="utf-8")
    content = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{semver}"',
        content,
    )
    content = re.sub(
        r'__release_date__\s*=\s*"[^"]+"',
        f'__release_date__ = "{calver_date}"',
        content,
    )
    VERSION_FILE.write_text(content, encoding="utf-8")

    # Update pyproject.toml
    pyproject = PYPROJECT_FILE.read_text(encoding="utf-8")
    pyproject = re.sub(
        r'^version\s*=\s*"[^"]+"',
        f'version = "{semver}"',
        pyproject,
        # Turn-end file-mutation verifier footer appended by run_agent.py
        # (``_format_file_mutation_failure_footer``). It's a UI affordance — reading "warning file mutation
        # verifier, 2 files were NOT modified..." aloud is noise (#40772). The footer is a ``⚠️
        # File-mutation verifier:`` header line followed by indented ``•`` bullet lines; strip the whole
        # block.
        flags=re.MULTILINE,
    )
    PYPROJECT_FILE.write_text(pyproject, encoding="utf-8")

    # Keep the desktop Electron app's package.json version in lockstep with the
    # Python package version. The desktop About panel reads the live Hermes
    # version at runtime, but app.getVersion()/packaging metadata still come
    # from this field, so it must track pyproject to avoid drift.
    desktop_pkg = REPO_ROOT / "apps" / "desktop" / "package.json"
    if desktop_pkg.exists():
        pkg_text = desktop_pkg.read_text(encoding="utf-8")
        pkg_text = re.sub(
            r'("version"\s*:\s*)"[^"]+"',
            rf'\g<1>"{semver}"',
            pkg_text,
            count=1,
        )
        desktop_pkg.write_text(pkg_text, encoding="utf-8")

    # Keep the bootstrap installer (Hermes-Setup.dmg CFBundleShortVersionString)
    # in lockstep with the Python package version. Tauri reads `version` from
    # package.json + tauri.conf.json; a hardcoded 0.0.1 ships in the DMG.
    installer_pkg = REPO_ROOT / "apps" / "bootstrap-installer" / "package.json"
    if installer_pkg.exists():
        pkg_text = installer_pkg.read_text(encoding="utf-8")
        pkg_text = re.sub(
            r'("version"\s*:\s*)"[^"]+"',
            rf'\g<1>"{semver}"',
            pkg_text,
            count=1,
        )
        installer_pkg.write_text(pkg_text, encoding="utf-8")

    installer_tauri = (
        REPO_ROOT / "apps" / "bootstrap-installer" / "src-tauri" / "tauri.conf.json"
    )
    if installer_tauri.exists():
        pkg_text = installer_tauri.read_text(encoding="utf-8")
        pkg_text = re.sub(
            r'("version"\s*:\s*)"[^"]+"',
            rf'\g<1>"{semver}"',
            pkg_text,
            count=1,
        )
        installer_tauri.write_text(pkg_text, encoding="utf-8")

    installer_cargo = (
        REPO_ROOT / "apps" / "bootstrap-installer" / "src-tauri" / "Cargo.toml"
    )
    if installer_cargo.exists():
        cargo_text = installer_cargo.read_text(encoding="utf-8")
        cargo_text = re.sub(
            r'^version\s*=\s*"[^"]+"',
            f'version = "{semver}"',
            cargo_text,
            count=1,
            flags=re.MULTILINE,
        )
        installer_cargo.write_text(cargo_text, encoding="utf-8")


def version_files_to_stage() -> list[str]:
    """Return version-bearing files that exist and should be `git add`ed after a bump."""
    candidates = [
        VERSION_FILE,
        PYPROJECT_FILE,
        REPO_ROOT / "apps" / "desktop" / "package.json",
        REPO_ROOT / "apps" / "bootstrap-installer" / "package.json",
        REPO_ROOT / "apps" / "bootstrap-installer" / "src-tauri" / "tauri.conf.json",
        REPO_ROOT / "apps" / "bootstrap-installer" / "src-tauri" / "Cargo.toml",
    ]
    return [str(path) for path in candidates if path.exists()]


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


def get_commits(since_tag=None):
    """Get commits since a tag (or all commits if None)."""
    if since_tag:
        range_spec = f"{since_tag}..HEAD"
    else:
        range_spec = "HEAD"

    # Format: hash<US>author_name<US>author_email<US>subject\0body
    # Using %x1f (unit separator) to avoid conflict with | in author names
    log = git(
        "log", range_spec,
        "--format=%H%x1f%an%x1f%ae%x1f%s%x00%b%x00",
        "--no-merges",
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
                       prev_tag=None, first_release=False):
    """Generate markdown changelog from categorized commits."""
    lines = []

    # Header
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    lines.append(f"# Hermes Agent v{semver} ({tag_name})")
    lines.append("")
    lines.append(f"**Release Date:** {date_str}")
    lines.append("")

    if first_release:
        lines.append("> 🎉 **First official release!** This marks the beginning of regular weekly releases")
        lines.append("> for Hermes Agent. See below for everything included in this initial release.")
        lines.append("")

    # Group commits by category
    categories = defaultdict(list)
    all_authors = set()
    teknium_aliases = {"@teknium1"}

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


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent Release Tool")
    parser.add_argument("--bump", choices=["major", "minor", "patch"],
                        help="Which semver component to bump")
    parser.add_argument("--publish", action="store_true",
                        help="Actually create the tag and GitHub release (otherwise dry run)")
    parser.add_argument("--date", type=str,
                        help="Override CalVer date (format: YYYY.M.D)")
    parser.add_argument("--first-release", action="store_true",
                        help="Mark as first release (no previous tag expected)")
    parser.add_argument("--output", type=str,
                        help="Write changelog to file instead of stdout")
    args = parser.parse_args()

    # Determine CalVer date
    if args.date:
        calver_date = args.date
    else:
        now = datetime.now()
        calver_date = f"{now.year}.{now.month}.{now.day}"

    base_tag = f"v{calver_date}"
    tag_name, calver_date = next_available_tag(base_tag)
    if tag_name != base_tag:
        print(f"Note: Tag {base_tag} already exists, using {tag_name}")

    # Determine semver
    current_version = get_current_version()
    if args.bump:
        new_version = bump_version(current_version, args.bump)
    else:
        new_version = current_version

    # Get previous tag
    prev_tag = get_last_tag()
    if not prev_tag and not args.first_release:
        print("No previous tags found. Use --first-release for the initial release.")
        print(f"Would create tag: {tag_name}")
        print(f"Would set version: {new_version}")
        return

    # Get commits
    commits = get_commits(since_tag=prev_tag)
    if not commits:
        print("No new commits since last tag.")
        if not args.first_release:
            return

    print(f"{'='*60}")
    print("  Hermes Agent Release Preview")
    print(f"{'='*60}")
    print(f"  CalVer tag:      {tag_name}")
    print(f"  SemVer:          v{current_version} → v{new_version}")
    print(f"  Previous tag:    {prev_tag or '(none — first release)'}")
    print(f"  Commits:         {len(commits)}")
    print(f"  Unique authors:  {len({c['github_author'] for c in commits})}")
    print(f"  Mode:            {'PUBLISH' if args.publish else 'DRY RUN'}")
    print(f"{'='*60}")
    print()

    # Generate changelog
    changelog = generate_changelog(
        commits, tag_name, new_version,
        prev_tag=prev_tag,
        first_release=args.first_release,
    )

    if args.output:
        Path(args.output).write_text(changelog, encoding="utf-8")
        print(f"Changelog written to {args.output}")
    else:
        print(changelog)

    if args.publish:
        print(f"\n{'='*60}")
        print("  Publishing release...")
        print(f"{'='*60}")

        # Update version files
        if args.bump:
            update_version_files(new_version, calver_date)
            print(f"  ✓ Updated version files to v{new_version} ({calver_date})")

            # Commit version bump
            add_files = version_files_to_stage()
            add_result = git_result("add", *add_files)
            if add_result.returncode != 0:
                print(f"  ✗ Failed to stage version files: {add_result.stderr.strip()}")
                return

            commit_result = git_result(
                "commit", "-m", f"chore: bump version to v{new_version} ({calver_date})"
            )
            if commit_result.returncode != 0:
                print(f"  ✗ Failed to commit version bump: {commit_result.stderr.strip()}")
                return
            print("  ✓ Committed version bump")

        # Create annotated tag
        tag_result = git_result(
            "tag", "-a", tag_name, "-m",
            f"Hermes Agent v{new_version} ({calver_date})\n\nWeekly release"
        )
        if tag_result.returncode != 0:
            print(f"  ✗ Failed to create tag {tag_name}: {tag_result.stderr.strip()}")
            return
        print(f"  ✓ Created tag {tag_name}")

        # Push
        push_result = git_result("push", "origin", "HEAD", "--tags")
        if push_result.returncode == 0:
            print("  ✓ Pushed to origin")
        else:
            print(f"  ✗ Failed to push to origin: {push_result.stderr.strip()}")
            print("    Continue manually after fixing access:")
            print("    git push origin HEAD --tags")

        # Create GitHub release
        changelog_file = REPO_ROOT / ".release_notes.md"
        changelog_file.write_text(changelog, encoding="utf-8")

        gh_cmd = [
            "gh", "release", "create", tag_name,
            "--title", f"Hermes Agent v{new_version} ({calver_date})",
            "--notes-file", str(changelog_file),
        ]

        gh_bin = shutil.which("gh")
        if gh_bin:
            result = subprocess.run(
                gh_cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(REPO_ROOT),
            )
        else:
            result = None

        if result and result.returncode == 0:
            changelog_file.unlink(missing_ok=True)
            print(f"  ✓ GitHub release created: {result.stdout.strip()}")
            print(f"\n  🎉 Release v{new_version} ({tag_name}) published!")
        else:
            if result is None:
                print("  ✗ GitHub release skipped: `gh` CLI not found.")
            else:
                print(f"  ✗ GitHub release failed: {result.stderr.strip()}")
            print(f"    Release notes kept at: {changelog_file}")
            print("    Tag was created locally. Create the release manually:")
            print(
                f"    gh release create {tag_name} --title 'Hermes Agent v{new_version} ({calver_date})' "
                f"--notes-file .release_notes.md"
            )
            print(f"\n  ✓ Release v{new_version} ({tag_name}) prepared for manual publish.")
    else:
        print(f"\n{'='*60}")
        print("  Dry run complete. To publish, add --publish")
        print("  Example: python scripts/release.py --bump minor --publish")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
