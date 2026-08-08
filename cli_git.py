#!/usr/bin/env python3
"""
cli_git.py — Git worktree isolation utilities for the Hermes CLI.

Extracted from cli.py for improved modularity and security auditability.
Names are re-exported from cli.py, so ``from cli import _setup_worktree``
continues to work.

Manages per-session isolated git worktrees (synchronized-base setup),
worktree lock classification, stale-worktree pruning, and orphaned
branch cleanup.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# =============================================================================
# Git Worktree Isolation (#652)
# =============================================================================

# Tracks the active worktree for cleanup on exit
_active_worktree: Optional[Dict[str, str]] = None


def set_active_worktree(wt_info: Optional[Dict[str, str]]) -> None:
    """Set the active worktree. Called by cli.py main() instead of
    ``global _active_worktree; _active_worktree = wt_info`` so the
    rebinding happens in cli_git's namespace, not a stale cli.py copy."""
    global _active_worktree
    _active_worktree = wt_info


def _normalize_git_bash_path(p: Optional[str]) -> Optional[str]:
    """Translate a Git Bash-style path (``/c/Users/...``) to the native
    Windows form (``C:\\Users\\...``) that Python's ``subprocess.Popen``
    and ``pathlib.Path`` accept.

    No-op on non-Windows and for paths that already look native.  Git on
    native Windows normally emits forward-slash Windows paths
    (``C:/Users/...``) which both bash and Python handle, but certain
    configurations (Git Bash shells, MSYS2, WSL-mounted repos) surface
    ``/c/...`` or ``/cygdrive/c/...`` variants.
    """
    if not p:
        return p
    if sys.platform != "win32":
        return p
    import re as _re
    # /c/Users/... or /C/Users/...
    m = _re.match(r"^/([a-zA-Z])/(.*)$", p)
    if m:
        drive, rest = m.group(1), m.group(2)
        return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"
    # /cygdrive/c/... or /mnt/c/...
    m = _re.match(r"^/(?:cygdrive|mnt)/([a-zA-Z])/(.*)$", p)
    if m:
        drive, rest = m.group(1), m.group(2)
        return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"
    return p


def _git_repo_root() -> Optional[str]:
    """Return the git repo root for CWD, or None if not in a repo.

    Runs through :func:`_normalize_git_bash_path` so callers can pass
    the result directly to ``Path``/``subprocess.Popen(cwd=...)`` on
    Windows without hitting ``C:\\c\\Users\\...`` style resolution
    mistakes.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if result.returncode == 0:
            return _normalize_git_bash_path(result.stdout.strip())
    except Exception:
        pass
    return None


def _path_is_within_root(path: Path, root: Path) -> bool:
    """Return True when a resolved path stays within the expected root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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


def _setup_worktree(repo_root: str = None, sync_base: bool = True) -> Optional[Dict[str, str]]:
    """Create an isolated git worktree for this CLI session.

    Returns a dict with worktree metadata on success, None on failure.
    The dict contains: path, branch, repo_root.

    When *sync_base* is True (default), the worktree branches from the
    freshly-fetched remote tip rather than the (possibly stale) local ``HEAD``
    — see ``_resolve_worktree_base``. Set ``worktree_sync: false`` in config to
    branch from local ``HEAD`` (the pre-#10760-followup behavior).
    """
    import subprocess

    repo_root = repo_root or _git_repo_root()
    if not repo_root:
        print("\033[31m✗ --worktree requires being inside a git repository.\033[0m")
        print("  cd into your project repo first, then run hermes -w")
        return None

    short_id = uuid.uuid4().hex[:8]
    wt_name = f"hermes-{short_id}"
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    wt_path = worktrees_dir / wt_name

    # Ensure .worktrees/ is in .gitignore
    gitignore = Path(repo_root) / ".gitignore"
    _ignore_entry = ".worktrees/"
    try:
        # utf-8-sig: git files are UTF-8 and Notepad prepends a BOM, which
        # would glue to the first line and defeat the membership check below
        # (duplicating the entry); the locale default also breaks non-ASCII
        # patterns on Windows. The append below already writes UTF-8.
        existing = (
            gitignore.read_text(encoding="utf-8-sig", errors="replace")
            if gitignore.exists()
            else ""
        )
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")
    except Exception as e:
        logger.debug("Could not update .gitignore: %s", e)

    # Resolve the base ref. By default branch from the freshly-fetched remote
    # tip so the worktree starts current with the project, not from the
    # (possibly stale) local HEAD of the standalone clone (#10760 follow-up).
    if sync_base:
        base_ref, base_label = _resolve_worktree_base(repo_root)
    else:
        base_ref, base_label = "HEAD", "HEAD (local — worktree_sync disabled)"

    # Create the worktree
    try:
        result = subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch_name, base_ref],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=repo_root,
        )
        if result.returncode != 0:
            # If branching from the resolved remote ref failed for any reason
            # (e.g. a partial fetch left the ref unusable), retry from local
            # HEAD so worktree creation never hard-fails on a sync hiccup.
            if base_ref != "HEAD":
                logger.warning(
                    "worktree add from %s failed (%s); retrying from local HEAD",
                    base_ref, result.stderr.strip(),
                )
                base_ref, base_label = "HEAD", "HEAD (fallback — remote base failed)"
                result = subprocess.run(
                    ["git", "worktree", "add", str(wt_path), "-b", branch_name, base_ref],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=repo_root,
                )
            if result.returncode != 0:
                print(f"\033[31m✗ Failed to create worktree: {result.stderr.strip()}\033[0m")
                return None
    except Exception as e:
        print(f"\033[31m✗ Failed to create worktree: {e}\033[0m")
        return None

    # Copy files listed in .worktreeinclude (gitignored files the agent needs)
    include_file = Path(repo_root) / ".worktreeinclude"
    if include_file.exists():
        try:
            repo_root_resolved = Path(repo_root).resolve()
            wt_path_resolved = wt_path.resolve()
            # utf-8-sig, not the locale default: on a cp1251/GBK Windows
            # machine a UTF-8 include list either decodes to mojibake paths
            # (entries silently not copied) or raises UnicodeDecodeError,
            # which the enclosing handler swallows at DEBUG — no include is
            # copied at all. A Notepad BOM likewise glued to the first entry.
            for line in include_file.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines():
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                src = Path(repo_root) / entry
                dst = wt_path / entry
                # Prevent path traversal and symlink escapes: both the resolved
                # source and the resolved destination must stay inside their
                # expected roots before any file or symlink operation happens.
                try:
                    src_resolved = src.resolve(strict=False)
                    dst_resolved = dst.resolve(strict=False)
                except (OSError, ValueError):
                    logger.debug("Skipping invalid .worktreeinclude entry: %s", entry)
                    continue
                if not _path_is_within_root(src_resolved, repo_root_resolved):
                    logger.warning("Skipping .worktreeinclude entry outside repo root: %s", entry)
                    continue
                if not _path_is_within_root(dst_resolved, wt_path_resolved):
                    logger.warning("Skipping .worktreeinclude entry that escapes worktree: %s", entry)
                    continue
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                elif src.is_dir():
                    # Symlink directories (faster, saves disk).  On Windows,
                    # symlink creation requires Developer Mode or elevation,
                    # and fails with OSError otherwise — fall back to a
                    # recursive copy so the worktree is still usable.  The
                    # copy is slower and uses disk, but it doesn't require
                    # admin and matches the Linux/macOS symlink outcome
                    # functionally.
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.symlink(str(src_resolved), str(dst))
                        except (OSError, NotImplementedError) as _sym_err:
                            if sys.platform == "win32":
                                logger.info(
                                    ".worktreeinclude: symlink failed (%s) — "
                                    "falling back to copytree on Windows.",
                                    _sym_err,
                                )
                                try:
                                    shutil.copytree(
                                        str(src_resolved),
                                        str(dst),
                                        symlinks=True,
                                        dirs_exist_ok=False,
                                    )
                                except Exception as _copy_err:
                                    logger.warning(
                                        ".worktreeinclude: copy fallback "
                                        "also failed for %s -> %s: %s",
                                        src, dst, _copy_err,
                                    )
                            else:
                                raise
        except Exception as e:
            logger.debug("Error copying .worktreeinclude entries: %s", e)

    # Lock the worktree so other processes (and `git worktree remove`) can see
    # it is actively in use.  Fail-soft: a lock failure never blocks the session.
    try:
        subprocess.run(
            ["git", "worktree", "lock", "--reason", f"hermes pid={os.getpid()}", str(wt_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        logger.debug("Worktree locked: %s (pid=%s)", wt_path, os.getpid())
    except Exception as e:
        logger.debug("git worktree lock failed (non-fatal): %s", e)

    info = {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
        "base": base_ref,
    }

    print(f"\033[32m✓ Worktree created:\033[0m {wt_path}")
    print(f"  Branch: {branch_name}")
    print(f"  Base:   {base_label}")

    return info


def _worktree_has_unpushed_commits(worktree_path: str, timeout: int = 10) -> bool:
    """Return whether a worktree has commits not reachable from any remote branch.

    ``git log HEAD --not --remotes`` compares against remote-tracking refs under
    ``refs/remotes/*``. If a repo has no remote-tracking refs yet, there is no
    usable remote baseline to compare against, so treat it as having no
    "unpushed" commits.
    """
    import subprocess

    try:
        remote_refs = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", "refs/remotes"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if remote_refs.returncode != 0:
            return True
        if not remote_refs.stdout.strip():
            return False

        result = subprocess.run(
            ["git", "log", "--oneline", "HEAD", "--not", "--remotes"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _worktree_is_dirty(worktree_path: str, timeout: int = 10) -> bool:
    """Return whether a worktree has uncommitted changes (staged, unstaged, or
    untracked).

    Fails SAFE: on any error returns True so callers do not delete a worktree
    whose state they cannot determine.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


# Upper bound on retained `git cherry` verdict entries (see
# _save_worktree_merge_cache). Each entry is ~90 bytes, so this caps the cache
# near 90 KB even on a repo that churns thousands of worktree branches.
_WORKTREE_MERGE_CACHE_MAX = 1000


def _worktree_merge_cache_path() -> Path:
    """Path of the patch-equivalence verdict cache (profile-aware)."""
    return get_hermes_home() / "cache" / "worktree_merge_verdicts.json"


def _load_worktree_merge_cache() -> Dict[str, bool]:
    """Load the ``git cherry`` verdict cache. Missing/corrupt cache = empty."""
    try:
        raw = json.loads(
            _worktree_merge_cache_path().read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("verdicts")
    if not isinstance(entries, dict):
        return {}
    # Only keep well-formed bool verdicts — a hand-edited or partially written
    # cache must never inject a non-bool into the prune decision.
    return {k: v for k, v in entries.items() if isinstance(v, bool)}


def _save_worktree_merge_cache(verdicts: Dict[str, bool]) -> None:
    """Persist the verdict cache atomically. Best-effort — never raises.

    Bounded to the most recent ``_WORKTREE_MERGE_CACHE_MAX`` entries so the
    file can't grow without limit across thousands of sessions.
    """
    path = _worktree_merge_cache_path()
    tmp = None
    try:
        items = list(verdicts.items())
        if len(items) > _WORKTREE_MERGE_CACHE_MAX:
            items = items[-_WORKTREE_MERGE_CACHE_MAX:]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"version": 1, "verdicts": dict(items)}),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    except Exception as e:
        logger.debug("Could not persist worktree merge cache: %s", e)
        if tmp is not None:
            try:
                tmp.unlink()
            except Exception:
                pass


def _worktree_commits_all_merged_upstream(
    worktree_path: str,
    timeout: int = 30,
    max_ahead: int = 20,
    cache: Optional[Dict[str, bool]] = None,
) -> bool:
    """Return whether every local-only commit is patch-equivalent to a commit
    already on the default upstream branch.

    The dominant ``.worktrees/`` leak: a branch is pushed, its PR is
    squash-merged (or cherry-picked), and the remote branch is deleted. The
    local commits are then unreachable from ``refs/remotes/*`` forever, so the
    unpushed-commits guard preserves the worktree indefinitely even though its
    content is fully merged. ``git cherry`` detects patch-equivalence, letting
    the pruner reap these.

    Bounded: skips (returns False) when the branch is more than ``max_ahead``
    commits ahead — a stale-base tree, too expensive to diff-hash and unlikely
    to be a merged scratch branch. Fails SAFE toward False (preserve).

    ``git cherry`` diff-hashes every commit in the range, which on a large repo
    costs ~0.2-1.0s per worktree — and a tree preserved for unpushed work is
    re-tested on *every* startup, forever, always reaching the same answer. When
    *cache* is provided, the verdict is memoized against
    ``(base_sha, head_sha, max_ahead)``: the exact inputs ``git cherry``
    consumes. A cache hit is therefore identical to recomputation by
    construction — if either ref moves the key changes and the real git call
    runs again.
    """
    import subprocess

    base = None
    for candidate in ("origin/HEAD", "origin/main", "origin/master"):
        try:
            probe = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", candidate],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                base = candidate
                break
        except Exception:
            return False
    if base is None:
        return False

    try:
        # Resolve both endpoints to shas up front. These are the complete
        # inputs to the range below, so they form an exact cache key. Cheap
        # (~1ms) relative to the diff-hashing `git cherry` they guard.
        cache_key = None
        if cache is not None:
            revs = subprocess.run(
                ["git", "rev-parse", f"{base}^{{commit}}", "HEAD^{commit}"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            )
            if revs.returncode == 0:
                shas = revs.stdout.split()
                if len(shas) == 2:
                    cache_key = f"{shas[0]}..{shas[1]}:{max_ahead}"
                    if cache_key in cache:
                        return cache[cache_key]

        def _memo(verdict: bool) -> bool:
            if cache is not None and cache_key is not None:
                cache[cache_key] = verdict
            return verdict

        ahead = subprocess.run(
            ["git", "rev-list", "--count", f"{base}..HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if ahead.returncode != 0:
            return False
        count = int(ahead.stdout.strip() or "0")
        if count == 0:
            return _memo(True)
        if count > max_ahead:
            return _memo(False)

        cherry = subprocess.run(
            ["git", "cherry", base, "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
        )
        if cherry.returncode != 0:
            return False
        lines = [ln for ln in cherry.stdout.splitlines() if ln.strip()]
        # "-" = patch-equivalent commit exists upstream; "+" = unique local work
        return _memo(bool(lines) and all(ln.startswith("-") for ln in lines))
    except Exception:
        return False


def _worktree_lock_is_live(repo_root: str, worktree_path: str, timeout: int = 10):
    """Classify a worktree's git lock as live, dead, or absent.

    ``hermes -w`` locks each worktree with reason ``hermes pid=<pid>`` so a
    concurrent hermes process' startup prune leaves an in-use worktree alone.
    But a *crashed* session leaves the lock behind forever, and
    ``git worktree remove --force`` (single ``-f``) refuses to remove a locked
    worktree — so dead-locked worktrees accumulate indefinitely. This lets the
    pruner tell the two apart:

    - ``"live"``  — locked and the owning pid is still running (skip it).
    - ``"dead"``  — locked but the owning pid is gone, or the reason isn't a
                    parseable hermes lock (safe to unlock + reap).
    - ``None``    — not locked at all.

    Fails SAFE toward ``"live"``: if git can't be queried at all we cannot
    prove the worktree is safe to touch, so we report it as live.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=repo_root,
        )
        if result.returncode != 0:
            return "live"
    except Exception:
        return "live"

    target = Path(worktree_path).resolve()
    current: Optional[Path] = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            try:
                current = Path(line[len("worktree "):].strip()).resolve()
            except Exception:
                current = None
        elif line == "locked" or line.startswith("locked "):
            if current != target:
                continue
            reason = line[len("locked"):].strip()
            m = re.search(r"hermes pid=(\d+)", reason)
            if not m:
                # Locked by something we don't recognize as a hermes session
                # (or lock reason unavailable). Treat as dead — a foreign lock
                # on a hermes -w worktree is almost certainly a leftover, and
                # the age/dirty/unpushed gates already ran before we got here.
                return "dead"
            pid = int(m.group(1))
            if pid == os.getpid():
                return "live"
            try:
                from gateway.status import _pid_exists
                return "live" if _pid_exists(pid) else "dead"
            except Exception:
                # Can't determine liveness — fail safe toward keeping it.
                return "live"
    return None


def _cleanup_worktree(info: Dict[str, str] = None) -> None:
    """Remove a worktree and its branch on exit.

    Preserves the worktree only if it has unpushed commits (real work
    that hasn't been pushed to any remote).  Uncommitted changes alone
    (untracked files, test artifacts) are not enough to keep it — agent
    work lives in commits/PRs, not the working tree.
    """
    global _active_worktree
    info = info or _active_worktree
    if not info:
        return

    import subprocess

    wt_path = info["path"]
    branch = info["branch"]
    repo_root = info["repo_root"]

    if not Path(wt_path).exists():
        return

    has_unpushed = _worktree_has_unpushed_commits(wt_path, timeout=10)

    if has_unpushed:
        print(f"\n\033[33m⚠ Worktree has unpushed commits, keeping: {wt_path}\033[0m")
        print(f"  To clean up manually: git worktree remove --force {wt_path}")
        _active_worktree = None
        return

    # Remove worktree (even if working tree is dirty — uncommitted
    # changes without unpushed commits are just artifacts)
    # Unlock first so `git worktree remove` isn't blocked by the lock we
    # placed at creation time.  Fail-soft — never block cleanup.
    try:
        subprocess.run(
            ["git", "worktree", "unlock", wt_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("git worktree unlock failed (non-fatal): %s", e)

    try:
        subprocess.run(
            ["git", "worktree", "remove", wt_path, "--force"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to remove worktree: %s", e)

    # Delete the branch
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to delete branch %s: %s", branch, e)

    _active_worktree = None
    print(f"\033[32m✓ Worktree cleaned up: {wt_path}\033[0m")


def _prune_stale_worktrees(repo_root: str, max_age_hours: int = 24) -> None:
    """Remove stale worktrees and orphaned branches on startup.

    Covers EVERY directory under ``.worktrees/`` except kanban task trees
    (``t_<hex>`` — owned by the kanban dispatcher's own gc). Scratch trees
    created by ``hermes -w`` (``hermes-*``) age out fast; named trees created
    manually for salvage/review lanes age out on a slower schedule:

    - ``hermes-*``: skip under 24h; reap 24h+ when clean and merged/pushed;
      72h+ is the aggressive tier (still never deletes real work).
    - named trees: same logic at 3x the timeline (72h soft / 9d hard).

    Work-preservation guards (all tiers, any age):
    - uncommitted changes (dirty) — never removed;
    - unpushed commits — never removed, UNLESS every local-only commit is
      patch-equivalent to a commit already on upstream (``git cherry``): the
      squash-merged-PR case, which is the dominant ``.worktrees/`` leak since
      those commits stay unreachable from ``refs/remotes/*`` forever.

    Lock handling (orthogonal to age): ``hermes -w`` locks each worktree with
    reason ``hermes pid=<pid>`` so a concurrent hermes process leaves an in-use
    worktree alone. A *live*-locked worktree is skipped at any age; a
    *dead*-locked one (owning pid gone — a crashed session) is unlocked first
    so ``git worktree remove --force`` can actually reap it, otherwise those
    leftovers accumulate forever (``remove --force`` refuses a locked tree).

    Branch deletion is gated on ``git worktree remove`` succeeding, so a failed
    removal never orphans the branch (which would drop easy reachability of any
    commits still in the worktree).

    Preserved-work visibility: trees skipped for unpushed/dirty reasons that
    are older than 7 days are listed in a single WARNING so real in-flight
    work can't rot silently.

    Also prunes orphaned ``hermes/*`` and ``pr-*`` local branches that
    have no corresponding worktree.

    Performance: this runs on the startup path of every ``hermes -w`` session,
    and each candidate tree costs several git subprocesses (the ``git cherry``
    patch-equivalence probe dominates at ~0.2-1.0s on a large repo). With
    dozens of accumulated worktrees the serial version added ~11-18s of latency
    before the banner. Two changes keep the decisions byte-identical while
    removing nearly all of that:

    1. The read-only classification of each tree (dirty / unpushed / merged /
       lock state) is independent per tree, so it runs on a thread pool. Only
       the mutating phase (unlock, remove, branch -D) stays serial and ordered.
    2. ``git cherry`` verdicts are memoized on disk keyed by the exact
       ``(base_sha, head_sha)`` range they were computed from, so a tree
       preserved for unpushed work is not re-diff-hashed on every subsequent
       startup.
    """
    import re
    import subprocess
    import time

    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        _prune_orphaned_branches(repo_root)
        return

    now = time.time()
    stale_work_cutoff = now - (7 * 24 * 3600)
    preserved_stale: list = []
    # Kanban task worktrees (<repo>/.worktrees/t_<hex>) have their own
    # dispatcher-driven lifecycle (hermes kanban gc) — never touch them here.
    kanban_re = re.compile(r"^t_[0-9a-f]+$")

    # ── Phase 1: age filter (no subprocesses) ───────────────────────────────
    # Cheap stat-only pass so the thread pool below is sized to the trees that
    # actually need git work, not to everything on disk.
    candidates: list = []
    for entry in sorted(worktrees_dir.iterdir()):
        if not entry.is_dir() or kanban_re.match(entry.name):
            continue

        # Scratch trees (hermes-*) age out on the default schedule; named
        # trees (salvage/review lanes someone created deliberately) get 3x.
        scratch = entry.name.startswith("hermes-")
        tier_hours = max_age_hours if scratch else max_age_hours * 3
        soft_cutoff = now - (tier_hours * 3600)
        hard_cutoff = now - (tier_hours * 3 * 3600)

        try:
            mtime = entry.stat().st_mtime
            if mtime > soft_cutoff:
                continue  # Too recent — skip
        except Exception:
            continue

        candidates.append((entry, mtime, mtime <= hard_cutoff))

    if not candidates:
        _prune_orphaned_branches(repo_root)
        return

    # ── Phase 2: classify in parallel (read-only git queries) ───────────────
    # Every check here is a read-only git query against a distinct worktree, so
    # they are safe to run concurrently (git takes no repo-wide lock for these,
    # and each has its own index). Verdicts are collected and applied serially
    # below so removal order and log output stay deterministic.
    merge_cache = _load_worktree_merge_cache()
    cache_size_before = len(merge_cache)
    cache_lock = threading.Lock()

    def _classify(item):
        entry, mtime, force = item
        # Never delete real work, regardless of age or tier. Uncommitted
        # changes and unpushed commits may be a crashed session's in-flight
        # work; only clean, fully-merged/pushed trees (the scratch trees that
        # actually cause .worktrees/ bloat) are ever reaped.
        if _worktree_is_dirty(str(entry), timeout=5):
            return (entry, mtime, force, "dirty", None)
        if _worktree_has_unpushed_commits(str(entry), timeout=5):
            # Squash-merge escape hatch: commits unreachable from any remote
            # ref but patch-equivalent to upstream commits are merged work,
            # not unpushed work.
            with cache_lock:
                snapshot = dict(merge_cache)
            merged = _worktree_commits_all_merged_upstream(
                str(entry), timeout=30, cache=snapshot
            )
            with cache_lock:
                merge_cache.update(snapshot)
            if not merged:
                return (entry, mtime, force, "unpushed", None)

        # Respect git-native session locks. A lock owned by a still-running
        # hermes process means the worktree is actively in use — never touch
        # it. A lock whose owning pid is gone is a crashed session's leftover:
        # unlock it so `git worktree remove --force` (single -f) can reap it,
        # otherwise dead-locked worktrees pile up indefinitely.
        lock_state = _worktree_lock_is_live(repo_root, str(entry), timeout=5)
        if lock_state == "live":
            return (entry, mtime, force, "locked-live", None)
        return (entry, mtime, force, "reap", lock_state)

    # Bounded pool: enough to hide git's per-process startup latency without
    # spawning dozens of concurrent git processes on a small machine.
    workers = max(1, min(8, (os.cpu_count() or 4), len(candidates)))
    try:
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="hermes-wt-prune"
            ) as pool:
                verdicts = list(pool.map(_classify, candidates))
        else:
            verdicts = [_classify(c) for c in candidates]
    except Exception as e:
        # Never let a pool failure block startup — fall back to serial.
        logger.debug("Parallel worktree classification failed (%s); serial", e)
        verdicts = [_classify(c) for c in candidates]

    if len(merge_cache) != cache_size_before:
        _save_worktree_merge_cache(merge_cache)

    # ── Phase 3: mutate serially (unlock / remove / branch -D) ──────────────
    for entry, mtime, force, verdict, lock_state in verdicts:
        if verdict == "dirty":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (uncommitted changes)")
            continue
        if verdict == "unpushed":
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} (unpushed commits)")
            continue
        if verdict == "locked-live":
            logger.debug("Skipping live-locked worktree: %s", entry.name)
            continue

        if lock_state == "dead":
            try:
                subprocess.run(
                    ["git", "worktree", "unlock", str(entry)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
                )
            except Exception as e:
                logger.debug("Failed to unlock dead worktree %s: %s", entry.name, e)

        # Safe to remove
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()

            remove_result = subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, cwd=repo_root,
            )
            if remove_result.returncode != 0:
                # Removal failed — keep the branch so any commits stay
                # reachable rather than orphaning it.
                logger.debug(
                    "Failed to remove worktree %s: %s",
                    entry.name, remove_result.stderr.strip(),
                )
                continue
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
                )
            logger.debug("Pruned stale worktree: %s (force=%s)", entry.name, force)
        except Exception as e:
            logger.debug("Failed to prune worktree %s: %s", entry.name, e)

    if preserved_stale:
        logger.warning(
            "Preserving %d worktree(s) older than 7 days with unmerged work "
            "(push or remove them to reclaim disk): %s",
            len(preserved_stale), ", ".join(sorted(preserved_stale)),
        )

    _prune_orphaned_branches(repo_root)


def _prune_orphaned_branches(repo_root: str) -> None:
    """Delete local ``hermes/hermes-*`` and ``pr-*`` branches with no worktree.

    These are auto-generated by ``hermes -w`` sessions and PR review
    workflows respectively.  Once their worktree is gone they serve no
    purpose and just accumulate.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        if result.returncode != 0:
            return
        all_branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
    except Exception:
        return

    # Collect branches that are actively checked out in a worktree
    active_branches: set = set()
    try:
        wt_result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10, cwd=repo_root,
        )
        for line in wt_result.stdout.split("\n"):
            if line.startswith("branch refs/heads/"):
                active_branches.add(line.split("branch refs/heads/", 1)[-1].strip())
    except Exception:
        return  # Can't determine active branches — bail

    # Also protect the currently checked-out branch and main
    try:
        head_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, cwd=repo_root,
        )
        current = head_result.stdout.strip()
        if current:
            active_branches.add(current)
    except Exception:
        pass
    active_branches.add("main")

    orphaned = [
        b for b in all_branches
        if b not in active_branches
        and (b.startswith("hermes/hermes-") or b.startswith("pr-"))
    ]

    if not orphaned:
        return

    # Delete in batches
    for i in range(0, len(orphaned), 50):
        batch = orphaned[i:i + 50]
        try:
            subprocess.run(
                ["git", "branch", "-D"] + batch,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=repo_root,
            )
        except Exception as e:
            logger.debug("Failed to prune orphaned branches: %s", e)

    logger.debug("Pruned %d orphaned branches", len(orphaned))
