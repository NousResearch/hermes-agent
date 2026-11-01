#!/usr/bin/env python3
"""Generate a concise Traditional Chinese morning PR report for nightly work.

This helper is intentionally standalone: future cron jobs can call it from a
feature branch after committing/pushing work, and the output can be delivered
as the final scheduled-task response without touching Hermes runtime code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


_STATUS_LABELS = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "U": "unmerged",
    "?": "untracked",
    "!": "ignored",
}


@dataclass(frozen=True)
class ChangedFile:
    status: str
    path: str
    old_path: str | None = None


@dataclass(frozen=True)
class GitSnapshot:
    branch: str
    base_branch: str = "main"
    commit_subjects: list[str] = field(default_factory=list)
    changed_files: list[ChangedFile] = field(default_factory=list)
    pr_url: str | None = None


def _status_label(code: str) -> str:
    for char in code:
        if char in _STATUS_LABELS and char != " ":
            return _STATUS_LABELS[char]
    return "changed"


def parse_porcelain_status(output: str) -> list[ChangedFile]:
    """Parse `git status --porcelain=v1` output into stable file entries."""

    changed: list[ChangedFile] = []
    for raw_line in output.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue
        code = line[:2]
        path_part = line[3:] if len(line) > 3 else ""
        label = _status_label(code)
        if " -> " in path_part and ("R" in code or "C" in code):
            old_path, new_path = path_part.split(" -> ", 1)
            changed.append(ChangedFile(status=label, path=new_path, old_path=old_path))
        else:
            changed.append(ChangedFile(status=label, path=path_part))
    return changed


def should_silence(snapshot: GitSnapshot) -> bool:
    """Return True when there is genuinely no branch/PR work to report."""

    return not snapshot.pr_url and not snapshot.commit_subjects and not snapshot.changed_files


def _bullet_lines(items: Sequence[str], empty: str = "無。") -> list[str]:
    if not items:
        return [f"- {empty}"]
    return [f"- {item}" for item in items]


def _changed_file_lines(files: Sequence[ChangedFile]) -> list[str]:
    if not files:
        return ["- 無未提交檔案。"]
    lines = []
    for item in files:
        if item.old_path:
            lines.append(f"- {item.status}: `{item.old_path}` → `{item.path}`")
        else:
            lines.append(f"- {item.status}: `{item.path}`")
    return lines


def format_report(
    snapshot: GitSnapshot,
    *,
    title: str,
    why: str,
    verification: Sequence[str] = (),
    blockers: Sequence[str] = (),
) -> str:
    """Format a Joe-style morning report in Traditional Chinese."""

    pr_line = snapshot.pr_url or "尚未建立 PR（請看 blockers / commands）。"
    commits = _bullet_lines(snapshot.commit_subjects, empty="尚無本分支 commit。")
    changed_files = _changed_file_lines(snapshot.changed_files)
    verification_lines = _bullet_lines(list(verification))
    blocker_lines = _bullet_lines(list(blockers))

    return "\n".join(
        [
            "## TL;DR",
            f"- 已完成：{title}",
            f"- PR：{pr_line}",
            f"- 分支：`{snapshot.branch} → {snapshot.base_branch}`",
            f"- 為什麼幫到你：{why}",
            "",
            "## 事實 / 已驗證",
            "- 本次 commit：",
            *commits,
            "- 變更檔案：",
            *changed_files,
            "- 驗證：",
            *verification_lines,
            "",
            "## Blockers / 需要你看",
            *blocker_lines,
        ]
    )


def _run_git(args: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _run_optional(args: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def collect_snapshot(cwd: Path, base_branch: str) -> GitSnapshot:
    branch = _run_git(["branch", "--show-current"], cwd) or "HEAD"
    status = _run_git(["status", "--porcelain=v1"], cwd)
    commits_output = _run_optional(["git", "log", f"origin/{base_branch}..HEAD", "--pretty=%s"], cwd)
    pr_url = _run_optional(["gh", "pr", "view", "--json", "url", "--jq", ".url"], cwd)
    return GitSnapshot(
        branch=branch,
        base_branch=base_branch,
        commit_subjects=commits_output.splitlines() if commits_output else [],
        changed_files=parse_porcelain_status(status),
        pr_url=pr_url,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Git repository path")
    parser.add_argument("--base", default="main", help="Base branch name")
    parser.add_argument("--title", required=True, help="One-line description of what was built")
    parser.add_argument("--why", required=True, help="Why this helps Joe")
    parser.add_argument("--verify", action="append", default=[], help="Verification command/result; repeatable")
    parser.add_argument("--blocker", action="append", default=[], help="Blocker or question; repeatable")
    parser.add_argument("--silent-if-empty", action="store_true", help="Emit [SILENT] if no work/PR is detected")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    snapshot = collect_snapshot(args.cwd.resolve(), args.base)
    if args.silent_if_empty and should_silence(snapshot):
        print("[SILENT]")
        return 0
    print(
        format_report(
            snapshot,
            title=args.title,
            why=args.why,
            verification=args.verify,
            blockers=args.blocker,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
