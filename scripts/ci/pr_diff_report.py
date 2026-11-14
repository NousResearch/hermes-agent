#!/usr/bin/env python3
"""Write a PR's advisory diff report and publish it to the step summary.

Run by two steps of ``.github/workflows/repo-guards.yml`` on pull requests, with the target
branch in ``BASE_REF``. Each report diffs from the merge base with ``origin/$BASE_REF``, not the
base tip, so changes that landed on main after the branch point are not this PR's doing.

    python3 scripts/ci/pr_diff_report.py prompt-surface   # $RUNNER_TEMP/prompt-surface.md
    python3 scripts/ci/pr_diff_report.py lockfile         # $RUNNER_TEMP/lockfile-diff.md

A non-empty report appends ``changed=true`` to ``$GITHUB_OUTPUT`` and the report, under its
heading, to ``$GITHUB_STEP_SUMMARY``; an empty one appends ``changed=false``.
``scripts/ci/review_status.py emit`` then publishes it for the live PR comment.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROMPT_SURFACE_SNAPSHOT = "tests/fixtures/prompt_surface"
HEADINGS = {"prompt-surface": "## Prompt surface", "lockfile": "## JS lockfile semantic diff"}
GIT_TIMEOUT = 120
REPORT_TIMEOUT = 600
REPORTS = {"prompt-surface": "prompt-surface.md", "lockfile": "lockfile-diff.md"}


def merge_base() -> str:
    return subprocess.run(("git", "merge-base", f"origin/{os.environ['BASE_REF']}", "HEAD"),
                          stdout=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL,
                          timeout=GIT_TIMEOUT, check=True).stdout.strip()


def extract_base_snapshot(base_sha: str, destination: Path) -> None:
    """Extract the snapshot as committed at the merge base; a base without one leaves it absent."""
    destination.mkdir(parents=True, exist_ok=True)
    present = subprocess.run(("git", "cat-file", "-e", f"{base_sha}:{PROMPT_SURFACE_SNAPSHOT}"),
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             timeout=GIT_TIMEOUT, check=False)
    if present.returncode != 0:
        return
    archive = subprocess.run(("git", "archive", base_sha, PROMPT_SURFACE_SNAPSHOT),
                             stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                             timeout=GIT_TIMEOUT, check=True).stdout
    subprocess.run(("tar", "-x", "-C", str(destination)), input=archive, timeout=GIT_TIMEOUT,
                   check=True)


def write_report(kind: str, base_sha: str, report: Path) -> None:
    if kind == "prompt-surface":
        base_dir = report.parent / "base"
        extract_base_snapshot(base_sha, base_dir)
        command = [str(SCRIPTS / "prompt_surface.py"), "diff",
                   str(base_dir / PROMPT_SURFACE_SNAPSHOT), PROMPT_SURFACE_SNAPSHOT]
    else:
        print(f"Merge base: {base_sha}", flush=True)
        command = [str(SCRIPTS / "lockfile_diff.py"), "--base", base_sha, "--head", "HEAD"]
    subprocess.run([sys.executable, *command, "--output", str(report)],
                   stdin=subprocess.DEVNULL, timeout=REPORT_TIMEOUT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a PR's advisory diff report.")
    parser.add_argument("kind", choices=sorted(REPORTS))
    args = parser.parse_args(argv)
    report = Path(os.environ["RUNNER_TEMP"], REPORTS[args.kind])
    try:
        write_report(args.kind, merge_base(), report)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    changed = report.is_file() and report.stat().st_size > 0
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
        fh.write(f"changed={'true' if changed else 'false'}\n")
    if changed:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "ab") as fh:
            fh.write(f"{HEADINGS[args.kind]}\n\n".encode() + report.read_bytes())
    elif args.kind == "lockfile":
        report.write_bytes(b"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
