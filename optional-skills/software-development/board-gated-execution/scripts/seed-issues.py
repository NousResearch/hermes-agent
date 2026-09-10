#!/usr/bin/env python3
"""
Seed an issue board from work that currently exists only in conversation.

WHY

A repository with zero issues keeps every open thread in chat history: the
half-built feature, the deferred decision, the missing doc, the thing waiting on
a human. Work that is not on a board drifts, and the drift is invisible until
someone asks about it. An issue is the smallest artifact that survives a context
window.

HOW TO USE

Edit ISSUES below to describe the real backlog, then run it. This is a template
on purpose — the value is in writing acceptance criteria that can be *settled by
evidence*, which no script can do for you. "Feature done" is not a criterion;
"the user can hold a button, speak, and hear a reply on the live site" is.

IDEMPOTENT

Each issue carries a `<!-- seed:<key> -->` marker in its body. The script reads
existing issues and skips any key already present, so re-running creates nothing
and duplicates nothing.

    python3 seed-issues.py                # repo inferred from the git remote
    python3 seed-issues.py owner/repo     # explicit
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=180)


def detect_repo() -> str:
    p = run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    if p.returncode != 0 or not p.stdout.strip():
        print("Cannot determine the repository. Pass it explicitly: "
              "python3 seed-issues.py owner/repo", file=sys.stderr)
        sys.exit(2)
    return p.stdout.strip()


def existing_markers(repo: str) -> set[str]:
    p = run(["gh", "issue", "list", "--repo", repo, "--state", "all",
             "--limit", "200", "--json", "body"])
    if p.returncode != 0:
        return set()
    found: set[str] = set()
    for item in json.loads(p.stdout or "[]"):
        for line in (item.get("body") or "").splitlines():
            if line.startswith("<!-- seed:") and line.endswith("-->"):
                found.add(line[len("<!-- seed:"):-len(" -->")].strip())
    return found


# ---------------------------------------------------------------------------
# EDIT THIS. Two worked examples showing the shape; replace with the real
# backlog. Keep acceptance criteria checkable.
# ---------------------------------------------------------------------------
ISSUES: list[dict] = [
    {
        "key": "example-half-built",
        "title": "Example: finish the half-built feature",
        "labels": ["area:backend", "P1", "in-progress"],
        "body": """Describe the state honestly: what exists, what does not, what is untested.

## Acceptance criteria
- [ ] A criterion a script or a human can settle by looking at evidence
- [ ] Tests covering the behaviour, not merely the code paths
- [ ] Works at the smallest viewport or platform the project actually supports
""",
    },
    {
        "key": "example-blocked",
        "title": "Example: something only the human can unblock",
        "labels": ["area:infra", "blocked:human", "P1"],
        "body": """State exactly what is needed and why the agent cannot do it — a credential,
a billing decision, a judgment call.

## Acceptance criteria
- [ ] The observable result once the human has acted
""",
    },
]


def main(argv: list[str]) -> int:
    repo = argv[1] if len(argv) > 1 else detect_repo()
    have = existing_markers(repo)
    print(f"{repo}: {len(have)} seeded issue(s) already present\n")

    created = 0
    tmp = Path(tempfile.gettempdir())
    for spec in ISSUES:
        key = spec["key"]
        if key in have:
            print(f"  skip     {key} (already seeded)")
            continue
        body_file = tmp / f"seed-issue-{key}.md"
        body_file.write_text(spec["body"] + f"\n<!-- seed:{key} -->\n")
        p = run(["gh", "issue", "create", "--repo", repo,
                 "--title", spec["title"],
                 "--body-file", str(body_file),
                 "--label", ",".join(spec["labels"])])
        if p.returncode != 0:
            print(f"  FAILED   {key}: {p.stderr.strip()[:160]}")
            continue
        num = p.stdout.strip().rstrip("/").split("/")[-1]
        created += 1
        print(f"  created  #{num:<4} {spec['title'][:58]}")

    print(f"\n{created} issue(s) created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
