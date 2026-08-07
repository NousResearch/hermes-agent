"""Shared worktree base-ref resolution.

Extracted from ``cli.py`` so both the interactive ``hermes -w`` path
(``cli.py``) and the kanban dispatch path (``hermes_cli/kanban_db.py``) can
branch a new worktree from the freshly-fetched remote tip. ``kanban_db`` must
not import ``cli`` (``cli`` imports ``kanban_db``, not vice versa), so this
helper lives in a module both can import.

The body is ``cli._resolve_worktree_base`` verbatim; ``cli`` now imports the
symbol from here, so the two paths cannot drift apart again.
"""
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_worktree_base(
    repo_root: str,
    fetch_timeout: float = 5,
    freshness_window: float = 300,
) -> tuple:
    """Resolve the freshest base ref to branch a new worktree from.

    The standalone clone's ``HEAD`` can lag the remote by hundreds of commits
    (the ``~/.hermes/hermes-agent`` clone is updated only by ``hermes update``,
    not on every session). Branching a worktree from that stale ``HEAD`` roots
    every new branch on an old base — so the PR diff GitHub computes against
    current ``main`` balloons with unrelated changes, and the agent has to
    discover the staleness via the pre-push gate and rebase. Branching from the
    freshly-fetched remote tip instead means the worktree starts current.

    Strategy (each step falls back to the next on failure):
      1. If the current branch tracks an upstream, refresh and use that
         upstream ref — so a deliberate feature-branch worktree tracks its own
         remote, not the default branch.
      2. Else refresh the remote's default branch (``origin/HEAD`` → e.g.
         ``origin/main``) and use it.
      3. Else fall back to ``HEAD`` (offline, no remote, or detached) — the
         old behavior, never worse than before.

    "Refresh" is deliberately cheap on the startup path (the fetch here used
    to stall ``hermes -w`` launches for 30-60s on flaky smart-HTTP
    connections):

    - The fetch is SKIPPED entirely when the repo's ``FETCH_HEAD`` is younger
      than *freshness_window* seconds — a base fetched moments ago cannot have
      meaningfully moved, so repeated launches don't re-pay a network round
      trip.
    - The fetch is capped at *fetch_timeout* seconds. On timeout or failure we
      fall back to the locally-known remote-tracking ref (labelled "cached")
      instead of cascading into a second fetch attempt. Genuine staleness is
      backstopped by the pre-push stale-base gate.

    Returns ``(base_ref, label)`` where *base_ref* is a git revision suitable
    for ``git worktree add ... <base_ref>`` and *label* is a short
    human-readable description for the session banner.
    """
    import subprocess

    from hermes_cli._subprocess_compat import noninteractive_git_env

    def _git(args, timeout: float = 20):
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
            stdin=subprocess.DEVNULL,
            env=noninteractive_git_env(),
        )

    def _ref_exists(ref: str) -> bool:
        try:
            return _git(["rev-parse", "--verify", "--quiet", ref + "^{commit}"]).returncode == 0
        except Exception:
            return False

    def _fetch_head_age() -> Optional[float]:
        """Seconds since the last fetch in this repo, or None if unknown."""
        try:
            gd = _git(["rev-parse", "--git-dir"])
            if gd.returncode != 0:
                return None
            git_dir = Path(gd.stdout.strip())
            if not git_dir.is_absolute():
                git_dir = Path(repo_root) / git_dir
            fetch_head = git_dir / "FETCH_HEAD"
            if not fetch_head.exists():
                return None
            return max(0.0, time.time() - fetch_head.stat().st_mtime)
        except Exception:
            return None

    def _refresh(remote: str, branch: str, ref: str) -> tuple:
        """Return (ref, label) after a cheap best-effort refresh of *ref*.

        Never raises, never fetches twice, never blocks longer than
        *fetch_timeout*.
        """
        age = _fetch_head_age()
        if age is not None and age < freshness_window and _ref_exists(ref):
            return ref, f"{ref} (fetched {int(age)}s ago)"
        try:
            fetched = _git(["fetch", remote, branch], timeout=fetch_timeout)
            if fetched.returncode == 0:
                return ref, f"{ref} (fetched)"
            reason = "fetch failed"
        except subprocess.TimeoutExpired:
            reason = f"fetch timed out after {fetch_timeout:g}s"
        except Exception as e:
            reason = f"fetch error: {e}"
        if _ref_exists(ref):
            logger.debug("worktree base: %s — using cached %s", reason, ref)
            return ref, f"{ref} (cached — {reason})"
        return "HEAD", f"HEAD (local — {reason}, no cached {ref})"

    # 1. Current branch's upstream, if it tracks one.
    try:
        up = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
        if up.returncode == 0:
            upstream = up.stdout.strip()  # e.g. "origin/main"
            if upstream and "/" in upstream:
                remote, branch = upstream.split("/", 1)
                return _refresh(remote, branch, upstream)
    except Exception as e:
        logger.debug("worktree base: upstream resolution failed: %s", e)

    # 2. Remote default branch (origin/HEAD).
    try:
        # Resolve the remote's default branch symref.
        head_ref = _git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
        default_ref = ""
        if head_ref.returncode == 0:
            default_ref = head_ref.stdout.strip().replace("refs/remotes/", "", 1)
        if not default_ref:
            # origin/HEAD not set locally; ask the remote (network — capped
            # like the fetch so a stalled connection can't hang startup).
            show = _git(["remote", "show", "origin"], timeout=max(fetch_timeout, 5))
            for line in show.stdout.splitlines():
                line = line.strip()
                if line.startswith("HEAD branch:"):
                    _branch = line.split(":", 1)[1].strip()
                    # A remote with no default branch reports "(unknown)";
                    # don't construct a bogus "origin/(unknown)" ref from it.
                    if _branch and _branch != "(unknown)":
                        default_ref = "origin/" + _branch
                    break
        if default_ref and "/" in default_ref:
            remote, branch = default_ref.split("/", 1)
            return _refresh(remote, branch, default_ref)
    except Exception as e:
        logger.debug("worktree base: default-branch resolution failed: %s", e)

    # 3. Fall back to local HEAD (offline / no remote / detached).
    return "HEAD", "HEAD (local — could not reach remote)"
