#!/usr/bin/env python3
"""Block private/business deliverables from being PR'd to public repositories.

This is intentionally lightweight: it can run as a pre-PR sanity check from an
agent prompt, a shell script, or CI without importing Hermes internals.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\brevenue\s+lab\b",
        r"\baligned\s+insights\b",
        r"\blife\s+church\s+(?:private\s+)?strategy\b",
        r"\bsample[-\s_]*deliverables?\b",
        r"\bprivate\s+(?:strategy|business|client|customer)\b",
        r"\blinear\s+(?:business\s+)?tickets?\b",
        r"\bRAN-\d+\b",
    )
)

TEXT_EXTENSIONS = {
    ".adoc",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".mdx",
    ".py",
    ".rst",
    ".text",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

# Files whose purpose is to define or test this guardrail. These may mention the
# blocked phrases without being private/business deliverables themselves.
ALLOWLISTED_PATH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"^scripts/check_private_deliverable_public_repo\.py$",
        r"^tests/test_check_private_deliverable_public_repo\.py$",
        r"^skills/github/github-pr-workflow/SKILL\.md$",
        r"^skills/productivity/linear/SKILL\.md$",
        r"^skills/autonomous-ai-agents/codex/SKILL\.md$",
    )
)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def git_root() -> Path:
    result = run(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    if result.returncode != 0:
        raise SystemExit("error: must be run inside a git repository")
    return Path(result.stdout.strip())


def owner_repo_from_remote(root: Path) -> str | None:
    result = run(["git", "remote", "get-url", "origin"], root)
    if result.returncode != 0:
        return None
    remote = result.stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?$", remote)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def detect_visibility(root: Path, repo: str | None) -> str | None:
    if not repo:
        return None
    result = run(["gh", "repo", "view", repo, "--json", "visibility", "-q", ".visibility"], root)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().lower()
    return None


def changed_files(root: Path, base: str) -> list[Path]:
    result = run(["git", "diff", "--name-only", f"{base}...HEAD"], root)
    if result.returncode != 0:
        result = run(["git", "diff", "--name-only", "HEAD"], root)
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "error: failed to inspect changed files")
    return [root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def matches_sensitive(text: str) -> str | None:
    for pattern in SENSITIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def file_text(path: Path) -> str:
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:500_000]
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("PR_BASE", "origin/main"), help="base ref for git diff, default: origin/main")
    parser.add_argument("--repo", help="GitHub owner/repo; inferred from origin when omitted")
    parser.add_argument("--visibility", choices=("public", "private", "internal"), help="repo visibility; detected with gh when omitted")
    parser.add_argument("--text", action="append", default=[], help="additional PR title/body/task text to scan")
    parser.add_argument("--path", action="append", dest="paths", help="specific file/path to scan instead of git diff")
    args = parser.parse_args(argv)

    root = git_root()
    repo = args.repo or owner_repo_from_remote(root)
    visibility = args.visibility or detect_visibility(root, repo)

    if visibility != "public":
        print(f"ok: repo {repo or '<unknown>'} visibility is {visibility or 'unknown'}; public-repo private-deliverable guardrail not triggered")
        return 0

    paths = [root / p for p in args.paths] if args.paths else changed_files(root, args.base)
    violations: list[str] = []

    for raw in args.text:
        token = matches_sensitive(raw)
        if token:
            violations.append(f"PR text contains private/business marker: {token!r}")

    for path in paths:
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        if any(pattern.search(rel) for pattern in ALLOWLISTED_PATH_PATTERNS):
            continue
        token = matches_sensitive(rel)
        if token:
            violations.append(f"path {rel!r} contains private/business marker: {token!r}")
            continue
        token = matches_sensitive(file_text(path)) if path.exists() else None
        if token:
            violations.append(f"file {rel!r} contains private/business marker: {token!r}")

    if violations:
        print("BLOCKED: private/business deliverables must not be opened against a public repository.", file=sys.stderr)
        print(f"Repository: {repo or '<unknown>'} ({visibility})", file=sys.stderr)
        print("Route Revenue Lab / Aligned Insights / private strategy work to Ryan's private workspace, local vault, or a dedicated private repo.", file=sys.stderr)
        for item in violations:
            print(f"- {item}", file=sys.stderr)
        return 2

    print(f"ok: no private/business deliverable markers detected for public repo {repo or '<unknown>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
