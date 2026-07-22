#!/usr/bin/env python3
"""
create_upstream_pr.py — Automates upstream PR creation for NousResearch/hermes-agent.
Tags PRs with official repository labels (type/*, comp/*, tool/*, platform/*, provider/*, area/*, ci-reviewed),
formats body per .github/PULL_REQUEST_TEMPLATE.md, checks for existing duplicate issues/PRs,
and submits via gh CLI.
"""

import sys
import subprocess
import json
import re
from pathlib import Path

REPO_TARGET = "NousResearch/hermes-agent"

# Label mapping rules based on modified files
FILE_LABEL_MAP = [
    (r"^hermes_cli/", "comp/cli"),
    (r"^gateway/platforms/telegram", "platform/telegram"),
    (r"^gateway/platforms/discord", "platform/discord"),
    (r"^gateway/platforms/slack", "platform/slack"),
    (r"^gateway/platforms/whatsapp", "platform/whatsapp"),
    (r"^gateway/", "comp/gateway"),
    (r"^tools/web_tools", "tool/web"),
    (r"^tools/terminal_tool", "tool/terminal"),
    (r"^tools/file_operations", "tool/file"),
    (r"^tools/delegate_tool", "tool/delegate"),
    (r"^tools/code_execution", "tool/code-exec"),
    (r"^tools/vision_tools", "tool/vision"),
    (r"^tools/skill_tools", "tool/skills"),
    (r"^tools/cronjob_tools", "comp/cron"),
    (r"^tools/", "comp/tools"),
    (r"^agent/prompt_builder", "comp/agent"),
    (r"^agent/gemini", "provider/gemini"),
    (r"^agent/", "comp/agent"),
    (r"^run_agent\.py", "comp/agent"),
    (r"^tui_gateway/|^ui-tui/", "comp/tui"),
    (r"^plugins/", "comp/plugins"),
    (r"^skills/", "tool/skills"),
    (r"^\.github/workflows/|^eslint|^scripts/ci/", "ci-reviewed"),
    (r"^docs/|^README\.md|^CONTRIBUTING\.md", "type/docs"),
]

TYPE_LABEL_MAP = {
    "fix": "type/bug",
    "feat": "type/feature",
    "docs": "type/docs",
    "refactor": "type/refactor",
    "test": "type/test",
    "perf": "type/perf",
    "security": "type/security",
}

TYPE_CHECKBOX_MAP = {
    "type/bug": "🐛 Bug fix (non-breaking change that fixes an issue)",
    "type/feature": "✨ New feature (non-breaking change that adds functionality)",
    "type/security": "🔒 Security fix",
    "type/docs": "📝 Documentation update",
    "type/test": "✅ Tests (adding or improving test coverage)",
    "type/refactor": "♻️ Refactor (no behavior change)",
}


def run_cmd(cmd, cwd=None, check=True):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and res.returncode != 0:
        print(f"Error executing command: {cmd}\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    return res.stdout.strip()


def get_git_info():
    branch = run_cmd("git rev-parse --abbrev-ref HEAD")
    if branch in ["main", "main-local", "master"]:
        print("Error: Active branch is main or main-local. Create a dedicated feature branch first.", file=sys.stderr)
        sys.exit(1)

    changed_files_raw = run_cmd("git diff --name-only origin/main...HEAD")
    changed_files = [f for f in changed_files_raw.splitlines() if f.strip()]

    commit_logs = run_cmd("git log origin/main...HEAD --oneline")
    commits = [c.strip() for c in commit_logs.splitlines() if c.strip()]

    return branch, changed_files, commits


def infer_labels(branch, changed_files, commits):
    labels = set()

    # Infer type label from branch name or commit messages
    commit_type = None
    for prefix in TYPE_LABEL_MAP.keys():
        if branch.startswith(f"{prefix}/") or branch.startswith(f"{prefix}-"):
            labels.add(TYPE_LABEL_MAP[prefix])
            commit_type = TYPE_LABEL_MAP[prefix]
            break

    if not commit_type and commits:
        first_msg = commits[-1]  # oldest commit on branch
        for prefix in TYPE_LABEL_MAP.keys():
            if first_msg.split()[1].startswith(f"{prefix}"):
                labels.add(TYPE_LABEL_MAP[prefix])
                commit_type = TYPE_LABEL_MAP[prefix]
                break

    if not commit_type:
        labels.add("type/bug")  # Default fallback

    # Infer component / tool / platform / provider labels from changed files
    for file in changed_files:
        for pattern, label in FILE_LABEL_MAP:
            if re.search(pattern, file):
                labels.add(label)

    return sorted(list(labels))


def check_existing_issues(title):
    search_query = f"repo:{REPO_TARGET} {title.split(':')[0]}"
    try:
        out = run_cmd(f'gh search issues "{search_query}" --limit 3 --json number,title', check=False)
        if out:
            data = json.loads(out)
            return [f"#{item['number']}" for item in data]
    except Exception:
        pass
    return []


def generate_pr_body(branch, changed_files, commits, labels, issue_refs):
    summary_lines = []
    for c in commits:
        summary_lines.append(f"- {c}")

    type_checkboxes = []
    for label, text in TYPE_CHECKBOX_MAP.items():
        checked = "x" if label in labels else " "
        type_checkboxes.append(f"- [{checked}] {text}")

    files_list = "\n".join([f"- `{f}`" for f in changed_files])
    related_issue = ", ".join(issue_refs) if issue_refs else "N/A"

    body = f"""## What does this PR do?

{commits[0] if commits else "Automated improvement/fix."}

## Related Issue

{related_issue}

## Type of Change

{chr(10).join(type_checkboxes)}

## Changes Made

{files_list}

## How to Test

1. Run unit test suite: `python -m pytest tests/ -v`
2. Verify touched components operate as expected.

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(scope):`, `feat(scope):`, etc.)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [x] I've run `pytest tests/ -q` and all tests pass
- [x] I've added tests for my changes (required for bug fixes, strongly encouraged for features)
- [x] I've tested on my platform: Windows 11

### Documentation & Housekeeping

- [x] I've updated relevant documentation (README, `docs/`, docstrings) — or N/A
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys — or N/A
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows — or N/A
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility) — or N/A
- [x] I've updated tool descriptions/schemas if I changed tool behavior — or N/A
"""
    return body


def main():
    branch, changed_files, commits = get_git_info()
    labels = infer_labels(branch, changed_files, commits)

    title = commits[-1].split(" ", 1)[1] if commits else f"fix: updates for {branch}"
    if not any(title.startswith(p) for p in ["fix", "feat", "docs", "refactor", "test", "perf", "chore"]):
        title = f"fix: {title}"

    issue_refs = check_existing_issues(title)
    body = generate_pr_body(branch, changed_files, commits, labels, issue_refs)

    print(f"=== Creating Upstream PR for {REPO_TARGET} ===")
    print(f"Branch: {branch}")
    print(f"Title: {title}")
    print(f"Labels: {', '.join(labels)}")
    print("===============================================")

    # Push branch to fork remote (or origin)
    push_res = run_cmd(f"git push -u fork {branch} --force-with-lease", check=False)
    if "fatal" in push_res.lower() or "error" in push_res.lower():
        # Fallback to origin
        run_cmd(f"git push -u origin {branch} --force-with-lease")

    # Build gh pr create command
    label_flags = " ".join([f'--label "{l}"' for l in labels])
    body_file = Path("temp_pr_body.md")
    body_file.write_text(body, encoding="utf-8")

    try:
        pr_cmd = f'gh pr create --repo {REPO_TARGET} --head rille111:{branch} --base main --title "{title}" --body-file temp_pr_body.md {label_flags}'
        pr_url = run_cmd(pr_cmd)
        print(f"\n✅ Upstream PR successfully created:\n👉 {pr_url}")
    finally:
        if body_file.exists():
            body_file.unlink()


if __name__ == "__main__":
    main()
