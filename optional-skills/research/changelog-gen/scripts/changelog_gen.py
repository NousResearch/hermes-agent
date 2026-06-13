#!/usr/bin/env python3
"""
Changelog Generator — generate changelogs from git commit history.

Usage:
    python3 changelog_gen.py
    python3 changelog_gen.py --path /path/to/repo
    python3 changelog_gen.py --all --output CHANGELOG.md
"""

import os
import re
import subprocess
import sys
from collections import OrderedDict


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


def get_commits(repo_path: str = ".", all_commits: bool = False) -> list:
    cmd = ["git", "log", "--pretty=format:%H%x00%s%x00%b%x00"]
    if not all_commits:
        cmd.append("--max-count=100")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=False, check=True, cwd=repo_path
        )
        commits = []
        raw = result.stdout
        records = raw.split(b"\x00")
        for rec in records:
            if not rec.strip():
                continue
            parts = rec.split(b"\x00", 2)
            if len(parts) >= 2:
                sha = parts[0].decode("utf-8", errors="replace")[:8]
                subject = parts[1].decode("utf-8", errors="replace").strip()
                body = (
                    parts[2].decode("utf-8", errors="replace").strip()
                    if len(parts) > 2
                    else ""
                )
                commits.append({"sha": sha[:8], "subject": subject, "body": body})
        return commits
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e.stderr}", file=sys.stderr)
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
            "sha": c["sha"],
            "description": parsed["description"],
            "scope": parsed["scope"],
            "breaking": parsed["breaking"],
        }
        categorized[t].append(entry)

    return categorized


def format_changelog(
    categorized: OrderedDict, repo_name: str = "", tags: list | None = None
) -> str:
    lines = []
    if repo_name:
        lines.append(f"# Changelog for {repo_name}\n")
    else:
        lines.append("# Changelog\n")

    if tags:
        lines.append(f"\n*Tags: {', '.join(tags[:5])}*\n")

    has_content = any(v for v in categorized.values())

    if not has_content:
        lines.append("\n*No conventional commits found. Showing raw history:*\n")

    for t in TYPE_ORDER:
        entries = categorized[t]
        if not entries:
            continue
        label = TYPE_LABELS.get(t, t.capitalize())
        lines.append(f"\n## {label}\n")
        for e in entries:
            scope = f"**{e['scope']}:** " if e["scope"] else ""
            breaking = "⚠️ " if e["breaking"] else ""
            lines.append(f"- {breaking}{scope}{e['description']} ({e['sha']})")

    return "\n".join(lines)


def cmd_generate(args):
    repo_path = args.path or "."
    commits = get_commits(repo_path, args.all)
    tags = get_tags(repo_path)

    if not commits:
        print("No commits found.", file=sys.stderr)
        sys.exit(1)

    categorized = categorize_commits(commits)
    repo_name = os.path.basename(os.path.abspath(repo_path))
    changelog = format_changelog(categorized, repo_name, tags)

    if args.output:
        with open(args.output, "w") as f:
            f.write(changelog)
        print(f"Changelog written to {args.output}")
    else:
        print(changelog)

    if args.json:
        stats = {t: len(entries) for t, entries in categorized.items()}
        import json as _json_mod

        print(
            _json_mod.dumps(
                {
                    "repo": repo_name,
                    "total_commits": len(commits),
                    "tags": tags or [],
                    "by_type": {
                        TYPE_LABELS.get(t, t): count
                        for t, count in stats.items()
                        if count
                    },
                },
                indent=2,
            )
        )


def main():
    import argparse

    p = argparse.ArgumentParser(description="Generate changelog from git history")
    p.add_argument("--path", help="Path to git repository (default: current dir)")
    p.add_argument(
        "--all", action="store_true", help="Include all commits (default: last 100)"
    )
    p.add_argument("--output", "-o", help="Output file path")
    p.add_argument("--json", action="store_true", help="Print JSON stats to stdout")
    args = p.parse_args()

    cmd_generate(args)


if __name__ == "__main__":
    main()
