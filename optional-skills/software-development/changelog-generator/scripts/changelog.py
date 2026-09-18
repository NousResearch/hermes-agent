#!/usr/bin/env python3
"""Categorize git commits and write a Keep a Changelog section.

The public helpers are pure and side-effect free so they can be unit
tested without a live repo or network:

  parse_log(text)                -> list of commit dicts
  categorize(subject)            -> (category, cleaned_subject)
  render_section(version, date, commits, include_chores=False) -> str
  prepend_section(existing, new) -> str   # preserves ALL prior history

`prepend_section` is the reason this ships as a script instead of an
inline `write_file`: it reads the WHOLE existing changelog and splices
the new release above the previous one, so historical entries are never
truncated. The CLI wires these together and rewrites the file in full.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date as _date
from pathlib import Path

# Canonical Keep a Changelog section order. "Chores" and "Tests" are
# omitted from output by default (opt in with include_chores=True).
SECTION_ORDER = [
    "Breaking Changes",
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Documentation",
    "Tests",
    "Chores",
    "Other",
]

_NOISE = {"Tests", "Chores"}

# Conventional Commit type -> changelog category.
_PREFIX_MAP = {
    "feat": "Added", "add": "Added", "new": "Added",
    "fix": "Fixed", "bugfix": "Fixed", "patch": "Fixed",
    "refactor": "Changed", "change": "Changed", "update": "Changed",
    "improve": "Changed", "perf": "Changed",
    "remove": "Removed", "delete": "Removed", "drop": "Removed",
    "security": "Security", "sec": "Security",
    "deprecate": "Deprecated", "deprecated": "Deprecated",
    "docs": "Documentation", "doc": "Documentation",
    "test": "Tests", "tests": "Tests",
    "chore": "Chores", "ci": "Chores", "build": "Chores", "release": "Chores",
}

_PREFIX_RE = re.compile(r"^(\w+)(\([^)]*\))?(!)?:\s*(.*)$")


def _cap(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def categorize(subject: str):
    """Return (category, cleaned_subject) for one commit subject line."""
    subject = subject.strip()
    m = _PREFIX_RE.match(subject)
    if m:
        typ, _scope, bang, rest = m.group(1).lower(), m.group(2), m.group(3), m.group(4)
        rest = rest or subject
        if bang:  # feat!:, fix(scope)!: -> breaking regardless of type
            return "Breaking Changes", _cap(rest)
        if typ in _PREFIX_MAP:
            return _PREFIX_MAP[typ], _cap(rest)
    low = subject.lower()
    # Keyword fallback for repos that don't use Conventional Commits.
    if re.search(r"\b(break|breaking|incompatible)\b", low):
        return "Breaking Changes", _cap(subject)
    if re.search(r"\b(fix|bug|error|crash|patch)\b", low):
        return "Fixed", _cap(subject)
    if re.search(r"\b(add|new|feature|implement|introduce)\b", low):
        return "Added", _cap(subject)
    if re.search(r"\b(remove|delete|drop)\b", low):
        return "Removed", _cap(subject)
    if re.search(r"\b(update|refactor|improve|perf|optimi[sz]e)\b", low):
        return "Changed", _cap(subject)
    if re.search(r"\b(doc|docs|readme|comment)\b", low):
        return "Documentation", _cap(subject)
    return "Other", _cap(subject)


def parse_log(text: str):
    """Parse `git log --pretty=format:%H|%s|%an|%ae|%ad` output."""
    commits = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        commits.append({
            "hash": parts[0] if len(parts) > 0 else "",
            "subject": parts[1] if len(parts) > 1 else "",
            "author": parts[2] if len(parts) > 2 else "",
            "email": parts[3] if len(parts) > 3 else "",
            "date": parts[4] if len(parts) > 4 else "",
        })
    return commits


def _pr_number(subject: str):
    m = re.search(r"#(\d+)", subject)
    return f" (#{m.group(1)})" if m else ""


def _strip_pr(text: str) -> str:
    """Remove existing PR/issue refs so we don't append a duplicate."""
    text = re.sub(r"\s*\(#\d+\)", "", text)
    text = re.sub(r"\s+#\d+\b", "", text)
    return text.rstrip()


def render_section(version, date, commits, include_chores=False) -> str:
    """Render one `## [version] - date` block, omitting empty categories."""
    buckets = {name: [] for name in SECTION_ORDER}
    for c in commits:
        category, cleaned = categorize(c["subject"])
        buckets[category].append(_strip_pr(cleaned) + _pr_number(c["subject"]))

    out = [f"## [{version}] - {date}", ""]
    for name in SECTION_ORDER:
        if name in _NOISE and not include_chores:
            continue
        entries = buckets[name]
        if not entries:
            continue
        out.append(f"### {name}")
        out.extend(f"- {e}" for e in entries)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


_HEADER = (
    "# Changelog\n\n"
    "All notable changes to this project will be documented in this file.\n\n"
    "This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)\n"
    "and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n"
    "## [Unreleased]\n"
)


def prepend_section(existing: str, new_section: str) -> str:
    """Splice new_section above the newest existing release.

    Preserves the file header, an `## [Unreleased]` block, and every
    prior `## [x.y.z]` section. Never truncates history.
    """
    new_block = new_section.strip("\n")
    if not existing.strip():
        return _HEADER + "\n" + new_block + "\n"

    lines = existing.splitlines()
    insert_at = None
    for i, ln in enumerate(lines):
        if ln.startswith("## [") and not ln.startswith("## [Unreleased]"):
            insert_at = i
            break

    if insert_at is None:
        # No released section yet — keep all content, append below it.
        return existing.rstrip("\n") + "\n\n" + new_block + "\n"

    before = "\n".join(lines[:insert_at]).rstrip("\n")
    after = "\n".join(lines[insert_at:]).rstrip("\n")
    return before + "\n\n" + new_block + "\n\n" + after + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build a changelog section from git log output.")
    p.add_argument("--version", required=True, help="Version label, e.g. v1.2.0")
    p.add_argument("--date", default=_date.today().isoformat(), help="Release date (YYYY-MM-DD)")
    p.add_argument("--log-file", help="File with `%%H|%%s|%%an|%%ae|%%ad` lines; omit to read stdin")
    p.add_argument("--changelog", help="Existing CHANGELOG.md to update in place (preserves history)")
    p.add_argument("--all", action="store_true", help="Include chore/test commits")
    p.add_argument("--json", action="store_true", help="Emit a JSON summary instead of the section")
    args = p.parse_args(argv)

    raw = Path(args.log_file).read_text(encoding="utf-8") if args.log_file else sys.stdin.read()
    commits = parse_log(raw)
    section = render_section(args.version, args.date, commits, include_chores=args.all)

    if args.changelog:
        path = Path(args.changelog)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(prepend_section(existing, section), encoding="utf-8")

    if args.json:
        counts = {}
        for c in commits:
            counts[categorize(c["subject"])[0]] = counts.get(categorize(c["subject"])[0], 0) + 1
        print(json.dumps({"version": args.version, "commits": len(commits), "counts": counts}, indent=2))
    else:
        sys.stdout.write(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
