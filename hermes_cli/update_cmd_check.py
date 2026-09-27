"""Steps of ``hermes update --check``: debris cleanup, channel target, scoped fetch, verdict.

``update_cmd._cmd_update_check`` (a frozen updater surface, see ``tests/compat``) orchestrates
these. Facade helpers are read through ``_uc()`` at call time and origin helpers are imported
per function, so test patches on ``hermes_cli.update_cmd`` and the origin modules stay effective.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(git_cmd: list[str], root: Path, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        git_cmd + args, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs,
    )


def _uc():
    from hermes_cli import update_cmd

    return update_cmd


def clear_git_debris(root: Path) -> None:
    """Remove abandoned git locks and aborted-fetch pack temps before fetching.

    A crashed fetch can leave ``.git/shallow.lock`` (or another lock) behind, and every later
    fetch then fails with "File exists". Aborted fetches on flaky lines also strand
    ``tmp_pack_*`` debris: unchecked it reached 6 GB and corrupted the pack dir (#93732).
    """
    from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

    for lock_path in clear_stale_git_locks(root):
        print(f"  (removed stale git lock: {lock_path})")
    swept = clear_stale_tmp_packs(root)
    if swept:
        print(f"  (removed {len(swept)} aborted-fetch pack temp file(s))")


def channel_compare_branch(selected_channel: str, git_cmd: list[str], root: Path) -> str | None:
    """Report a release-pinned channel's verdict, or return the branch to compare against.

    ``None`` means the verdict is printed (the channel pins a commit); exits 1 when the channel
    cannot be resolved.
    """
    from hermes_cli.source_releases import resolve_source_target

    print(f"→ Update channel: {selected_channel}")
    try:
        target = resolve_source_target(selected_channel, git_cmd, root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"✗ Could not resolve the {selected_channel} source channel: {exc}")
        sys.exit(1)
    if not target.commit:
        return target.branch
    if target.retired:
        print(f"→ {selected_channel} retired; source destination: {target.channel}")
    if _uc()._capture_head_sha(git_cmd, root) == target.commit:
        print(f"✓ Up to date with the latest release ({target.label}).")
    else:
        print(f"→ Selected release available: {target.label}")
        print("  Run `hermes update` to install it.")
    return None


def is_shallow_repository(git_cmd: list[str], root: Path) -> bool:
    return _git(git_cmd, root, ["rev-parse", "--is-shallow-repository"]).stdout.strip() == "true"


def _fetch(git_cmd: list[str], root: Path, depth_args: list[str], remote: str, branch: str):
    print(f"→ Fetching from {remote}...")
    return _git(git_cmd, root, ["fetch", *depth_args, remote, branch], **_uc()._no_prompt_git_kwargs())


def fetch_compare_branch(git_cmd: list[str], root: Path, branch: str, depth_args: list[str]):
    """Fetch only ``branch`` and return ``(fetch_result, compare_ref)``.

    A bare ``git fetch <remote>`` pulls every ref, and this repo has thousands of auto-generated
    branches. ``main`` prefers upstream as the canonical reference; other branches go straight
    to origin, because a fork's branch usually has no upstream counterpart.
    """
    if branch == "main":
        # A local probe (~6 ms) spares non-fork installs a failed network fetch (~0.3-1 s).
        if _git(git_cmd, root, ["remote", "get-url", "upstream"]).returncode == 0:
            fetch_result = _fetch(git_cmd, root, depth_args, "upstream", branch)
            if fetch_result.returncode == 0:
                return fetch_result, f"upstream/{branch}"
    return _fetch(git_cmd, root, depth_args, "origin", branch), f"origin/{branch}"


def repair_shallow_grafts(root: Path) -> None:
    """Drop the stale ``.git/shallow`` grafts a depth-1 fetch leaves behind.

    Git never removes old grafts; unpruned, the file keeps growing and merge-base / the
    orphan-divergence heuristic stop working (#105951).
    """
    from hermes_cli.gitlock import prune_stale_shallow_grafts, repair_broken_shallow_boundaries

    repaired = repair_broken_shallow_boundaries(root)
    if repaired:
        print(f"  (restored {repaired} broken shallow boundary(ies))")
    pruned = prune_stale_shallow_grafts(root)
    if pruned:
        print(f"  (pruned {pruned} stale shallow graft(s) left by past depth-1 checks)")


def compare_ref_exists(git_cmd: list[str], root: Path, compare_branch: str) -> bool:
    # rev-list on a missing ref exits 128 and would surface a traceback; report it instead.
    return _git(git_cmd, root, ["rev-parse", "--verify", "--quiet", compare_branch]).returncode == 0


def report_shallow_verdict(git_cmd: list[str], root: Path, compare_branch: str) -> None:
    """Report behind-ness without local history across the shallow boundary.

    Compares tip SHAs (mirrors the banner's ``_check_via_local_git``), then asks the GitHub
    compare API for the exact count: the remote graph is complete even when the local one is not.
    """
    head_sha = _git(git_cmd, root, ["rev-parse", "HEAD"]).stdout.strip()
    target_sha = _git(git_cmd, root, ["rev-parse", compare_branch]).stdout.strip()
    if head_sha and target_sha and head_sha == target_sha:
        print("✓ Already up to date.")
        return
    from hermes_cli.config import recommended_update_command
    from hermes_cli.source_check import _github_compare_behind

    counted = _github_compare_behind(head_sha, target_sha)
    if counted == 0:
        # Local commits on top of the remote tip — not behind.
        print("✓ Already up to date.")
        return
    if counted is not None:
        commits_word = "commit" if counted == 1 else "commits"
        print(f"⚕ Update available: {counted} {commits_word} behind {compare_branch}.")
    else:
        print(f"⚕ Update available (behind {compare_branch}).")
    print(f"  Run '{recommended_update_command()}' to install.")


def report_rev_list_verdict(git_cmd: list[str], root: Path, compare_branch: str) -> None:
    rev_result = _git(git_cmd, root, ["rev-list", f"HEAD..{compare_branch}", "--count"], check=True)
    behind = int(rev_result.stdout.strip())
    if behind == 0:
        print("✓ Already up to date.")
        return
    commits_word = "commit" if behind == 1 else "commits"
    print(f"⚕ Update available: {behind} {commits_word} behind {compare_branch}.")
    from hermes_cli.config import recommended_update_command

    print(f"  Run '{recommended_update_command()}' to install.")
