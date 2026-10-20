#!/usr/bin/env python3
"""
Changelog Generator — generate changelogs from git commit history.

Usage:
    python3 changelog_gen.py
    python3 changelog_gen.py --path /path/to/repo
    python3 changelog_gen.py --all --output CHANGELOG.md
    python3 changelog_gen.py --json
"""

import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

# Field separator inside a record and record separator between records.
# Git's --pretty=format emits one record per commit using these literals.
FIELD_SEP = b"\x00"
REC_SEP = b"\x1e"

COMMIT_FORMAT = "%H%x00%s%x00%b%x1e"

COMMIT_PATTERN = re.compile(
    r"^(?P<type>\w+)(?:\((?P<scope>[^)]*)\))?"
    r"(?P<breaking>!)?\s*:\s*(?P<description>.+)$"
    r"(?:\n\n(?P<body>.*?))?"
    r"(?:\n\n(?P<footer>.*))?$",
    re.DOTALL,
)

TYPE_ORDER = [
    "breaking",
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "style",
    "test",
    "chore",
    "ci",
    "build",
    "other",
]

TYPE_LABELS = {
    "breaking": "Breaking Changes",
    "feat": "Features",
    "fix": "Bug Fixes",
    "perf": "Performance Improvements",
    "refactor": "Code Refactoring",
    "docs": "Documentation",
    "style": "Styling",
    "test": "Tests",
    "chore": "Chores",
    "ci": "CI/CD",
    "build": "Build System",
    "other": "Other",
}


def parse_commit_records(raw: bytes) -> list:
    """Parse a NUL/RS stream into (sha, subject, body) tuples.

    Each record is `hash<SEP>subject<SEP>body<REC_SEP>`. Records are delimited
    by REC_SEP (\x1e); fields within a record by FIELD_SEP (\x00). Using a
    distinct record delimiter keeps multi-line bodies intact and lets us split
    each record into exactly three fields instead of flattening everything.
    """
    commits = []
    for rec in raw.split(REC_SEP):
        rec = rec.strip(FIELD_SEP)
        if not rec.strip():
            continue
        parts = rec.split(FIELD_SEP, 2)
        if len(parts) < 2:
            continue
        sha = parts[0].decode("utf-8", errors="replace").strip()
        subject = parts[1].decode("utf-8", errors="replace").strip()
        body = (
            parts[2].decode("utf-8", errors="replace").strip()
            if len(parts) > 2
            else ""
        )
        commits.append(
            {
                "sha": sha,
                "subject": subject,
                "body": body,
            }
        )
    return commits


def get_commits(repo_path: str = ".", all_commits: bool = False) -> list:
    cmd = ["git", "log", f"--pretty=format:{COMMIT_FORMAT}"]
    if not all_commits:
        cmd.append("--max-count=100")
    try:
        result = subprocess.run(
            cmd, capture_output=True, check=True, cwd=repo_path
        )
        return parse_commit_records(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e.stderr.decode('utf-8', errors='replace')}",
              file=sys.stderr)
        sys.exit(1)


def get_tags(repo_path: str = ".") -> list:
    try:
        result = subprocess.run(
            ["git", "tag", "--sort=-creatordate"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_path,
        )
        return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
    except subprocess.CalledProcessError:
        return []


def _shas_in_range(repo_path: str, rev_range: str) -> list:
    try:
        result = subprocess.run(
            ["git", "log", "--pretty=format:%H", rev_range],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_path,
        )
        return [s.strip() for s in result.stdout.splitlines() if s.strip()]
    except subprocess.CalledProcessError:
        return []


def get_tag_groups(
    repo_path: str, all_commits: bool, commits: list
) -> list:
    """Group commits by tag range: [(label, [commit, ...]), ...].

    The newest tag starts an "Unreleased" bucket (commits after it up to HEAD);
    each subsequent pair of tags bounds the releases between them; if
    ``all_commits`` is set, commits before the oldest tag form a final bucket.
    Order is newest-first so the changelog reads top-down.
    """
    tags = get_tags(repo_path)
    if not tags:
        return []

    by_sha = {c["sha"]: c for c in commits}
    groups = []

    # Unreleased: commits newer than the latest tag.
    groups.append(
        ("Unreleased", [by_sha[s] for s in _shas_in_range(repo_path, f"{tags[0]}..HEAD") if s in by_sha])
    )

    for i in range(len(tags) - 1):
        older, newer = tags[i + 1], tags[i]
        shas = _shas_in_range(repo_path, f"{older}..{newer}")
        groups.append((newer, [by_sha[s] for s in shas if s in by_sha]))

    if all_commits:
        oldest = tags[-1]
        shas = _shas_in_range(repo_path, f"{oldest}")
        groups.append(
            (f"Before {oldest}", [by_sha[s] for s in shas if s in by_sha])
        )

    return groups


def parse_commit(subject: str, body: str) -> dict:
    m = COMMIT_PATTERN.match(subject)
    if not m:
        return {
            "type": "other",
            "scope": "",
            "description": subject,
            "breaking": False,
            "body": body,
        }

    d = m.groupdict()
    commit_type = d["type"].lower()
    is_breaking = d["breaking"] == "!" or "BREAKING CHANGE" in (body or "").upper()

    if is_breaking:
        commit_type = "breaking"

    return {
        "type": commit_type,
        "scope": d.get("scope", "") or "",
        "description": d["description"].strip(),
        "breaking": is_breaking,
        "body": (d.get("body") or "").strip(),
    }


def categorize_commits(commits: list) -> OrderedDict:
    categorized = OrderedDict()
    for t in TYPE_ORDER:
        categorized[t] = []

    for c in commits:
        parsed = parse_commit(c["subject"], c["body"])
        t = parsed["type"]
        if t not in categorized:
            t = "other"
        entry = {
            "sha": c["sha"][:8],
            "description": parsed["description"],
            "scope": parsed["scope"],
            "breaking": parsed["breaking"],
        }
        categorized[t].append(entry)

    return categorized


def _render_type_section(t: str, entries: list) -> list:
    lines = []
    label = TYPE_LABELS.get(t, t.capitalize())
    lines.append(f"\n### {label}\n")
    for e in entries:
        scope = f"**{e['scope']}:** " if e["scope"] else ""
        breaking = "⚠️ " if e["breaking"] else ""
        lines.append(f"- {breaking}{scope}{e['description']} ({e['sha']})")
    return lines


def format_changelog(
    categorized: OrderedDict,
    repo_name: str = "",
    tags: list | None = None,
    tag_groups: list | None = None,
) -> str:
    lines = []
    if repo_name:
        lines.append(f"# Changelog for {repo_name}\n")
    else:
        lines.append("# Changelog\n")

    if tags:
        lines.append(f"\n*Tags: {', '.join(tags[:5])}*\n")

    if tag_groups:
        any_content = False
        for label, commit_list in tag_groups:
            if not commit_list:
                continue
            any_content = True
            lines.append(f"\n## {label}\n")
            sub = categorize_commits(commit_list)
            for t in TYPE_ORDER:
                if sub[t]:
                    lines.extend(_render_type_section(t, sub[t]))
        if not any_content:
            lines.append("\n*No conventional commits found.*\n")
        return "\n".join(lines)

    has_content = any(v for v in categorized.values())
    if not has_content:
        lines.append("\n*No conventional commits found. Showing raw history:*\n")

    for t in TYPE_ORDER:
        entries = categorized[t]
        if not entries:
            continue
        lines.extend(_render_type_section(t, entries))

    return "\n".join(lines)


def _build_stats(repo_name: str, commits: list, categorized: OrderedDict, tags: list) -> dict:
    return {
        "repo": repo_name,
        "total_commits": len(commits),
        "tags": tags or [],
        "by_type": {
            TYPE_LABELS.get(t, t): len(entries)
            for t, entries in categorized.items()
            if len(entries)
        },
    }


def cmd_generate(args):
    repo_path = args.path or "."
    commits = get_commits(repo_path, args.all)
    tags = get_tags(repo_path)

    if not commits:
        print("No commits found.", file=sys.stderr)
        sys.exit(1)

    categorized = categorize_commits(commits)
    repo_name = os.path.basename(os.path.abspath(repo_path))

    if args.json:
        # Exclusive JSON output mode: emit only the JSON object on stdout.
        print(json.dumps(_build_stats(repo_name, commits, categorized, tags), indent=2))
        return

    tag_groups = get_tag_groups(repo_path, args.all, commits) if tags else None
    changelog = format_changelog(categorized, repo_name, tags, tag_groups)

    if args.output:
        with open(args.output, "w") as f:
            f.write(changelog)
        print(f"Changelog written to {args.output}")
    else:
        print(changelog)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Generate changelog from git history")
    p.add_argument("--path", help="Path to git repository (default: current dir)")
    p.add_argument(
        "--all", action="store_true", help="Include all commits (default: last 100)"
    )
    p.add_argument("--output", "-o", help="Output file path")
    p.add_argument("--json", action="store_true", help="Print JSON stats to stdout (exclusive)")
    args = p.parse_args()

    cmd_generate(args)


if __name__ == "__main__":
    main()
