"""Shared worktree base-ref resolution.

Extracted from ``cli.py`` so both the interactive ``hermes -w`` path
(``cli.py``) and the kanban dispatch path (``hermes_cli/kanban_db.py``) can
branch a new worktree from the freshly-fetched remote tip. ``kanban_db`` must
not import ``cli`` (``cli`` imports ``kanban_db``, not vice versa), so this
helper lives in a module both can import.
"""
import logging

logger = logging.getLogger(__name__)


def _resolve_worktree_base(repo_root: str) -> tuple:
    """Resolve the freshest base ref to branch a new worktree from.

    The standalone clone's ``HEAD`` can lag the remote by hundreds of commits
    (the ``~/.hermes/hermes-agent`` clone is updated only by ``hermes update``,
    not on every session). Branching a worktree from that stale ``HEAD`` roots
    every new branch on an old base — so the PR diff GitHub computes against
    current ``main`` balloons with unrelated changes, and the agent has to
    discover the staleness via the pre-push gate and rebase. Branching from the
    freshly-fetched remote tip instead means the worktree starts current.

    Strategy (each step falls back to the next on failure):
      1. If the current branch tracks an upstream, fetch and use that upstream
         ref — so a deliberate feature-branch worktree tracks its own remote,
         not the default branch.
      2. Else fetch the remote's default branch (``origin/HEAD`` → e.g.
         ``origin/main``) and use it.
      3. Else fall back to ``HEAD`` (offline, no remote, or detached) — the
         old behavior, never worse than before.

    Returns ``(base_ref, label)`` where *base_ref* is a git revision suitable
    for ``git worktree add ... <base_ref>`` and *label* is a short
    human-readable description for the session banner.
    """
    import subprocess

    def _git(args, timeout=20):
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=timeout, cwd=repo_root,
        )

    # 1. Current branch's upstream, if it tracks one.
    try:
        up = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
        if up.returncode == 0:
            upstream = up.stdout.strip()  # e.g. "origin/main"
            if upstream and "/" in upstream:
                remote = upstream.split("/", 1)[0]
                # Fetch just that branch; fail-soft if offline.
                _git(["fetch", remote, upstream.split("/", 1)[1]], timeout=30)
                return upstream, f"{upstream} (fetched)"
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
            # origin/HEAD not set locally; ask the remote.
            show = _git(["remote", "show", "origin"], timeout=30)
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
            _git(["fetch", remote, branch], timeout=30)
            return default_ref, f"{default_ref} (fetched)"
    except Exception as e:
        logger.debug("worktree base: default-branch resolution failed: %s", e)

    # 3. Fall back to local HEAD (offline / no remote / detached).
    return "HEAD", "HEAD (local — could not reach remote)"
