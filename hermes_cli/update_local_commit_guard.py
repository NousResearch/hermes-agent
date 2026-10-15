"""Fail-closed preflight for source updates that could discard local main commits."""

from __future__ import annotations

import subprocess
from pathlib import Path


class LocalUpdateGuardError(RuntimeError):
    """The updater cannot prove that a local-only main commit is safe to replace."""


def local_main_commits(repo_root: Path, *, run=None) -> list[str]:
    """Return commits on local ``main`` absent from the known ``origin/main`` ref.

    This is deliberately read-only: no fetch, stash, branch movement, rebase, or push.
    An unavailable ref or failed Git inspection blocks the update rather than guessing.
    Other branches retain the updater's existing branch-specific behavior.
    """
    runner = subprocess.run if run is None else run

    def git(*args: str) -> str:
        try:
            result = runner(
                ["git", *args],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LocalUpdateGuardError(f"Git preflight failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise LocalUpdateGuardError(detail or f"Git preflight failed: git {' '.join(args)}")
        return result.stdout.strip()

    branch = git("branch", "--show-current")
    if branch != "main":
        return []
    git("rev-parse", "--verify", "origin/main^{commit}")
    result = git("rev-list", "--oneline", "origin/main..HEAD")
    return result.splitlines() if result else []


def enforce_local_main_commit_guard(repo_root: Path) -> None:
    """Stop ``hermes update`` before snapshots or checkout mutations when main is ahead."""
    try:
        commits = local_main_commits(repo_root)
    except LocalUpdateGuardError as exc:
        print(f"✗ Update blocked: cannot verify local main commits safely ({exc}).")
        print(f"  Inspect: git -C {repo_root} status --short && git -C {repo_root} log origin/main..HEAD --oneline")
        raise SystemExit(1) from exc
    if not commits:
        return
    print("✗ Update blocked: local main contains commits not present in origin/main.")
    for commit in commits:
        print(f"  {commit}")
    print("  No stash, rebase, reset, or push was attempted.")
    print(f"  Review and preserve these commits first: git -C {repo_root} log origin/main..HEAD --oneline")
    raise SystemExit(1)
