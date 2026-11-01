"""Git worktree isolation for ``hermes -w`` sessions: create, classify, prune.

Every git call goes through ``_git``/``_git_out``/``_git_quiet`` (UTF-8 text, captured,
bounded timeout). Classification helpers fail SAFE toward "preserve". ``cli`` re-exports
these names; ``_cprint`` is imported lazily from ``cli`` to avoid a cycle.
"""
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from hermes_cli._subprocess_compat import noninteractive_git_env
from hermes_cli.git_commit_graph import (
    claim_commit_graph_refresh,
    claim_maintenance_slot,
    common_git_dir,
    refresh_commit_graph,
    run_bounded_git,
)
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger("cli")


def _cprint(text: str) -> None:
    from cli import _cprint as _impl
    _impl(text)


def _git(args, cwd, timeout: float = 10, **kwargs):
    """Run ``git *args`` in *cwd* capturing UTF-8 text; raises like ``subprocess.run``."""
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",  # noqa: PLW1510 -- forwarding wrapper: callers pass subprocess kwargs via **kwargs (check may arrive through it); an explicit check=False would raise TypeError
                          errors="replace", timeout=timeout, cwd=cwd, **kwargs)


def _git_out(args, cwd, timeout: float = 10, **kwargs) -> Optional[str]:
    """``_git`` returning stripped stdout, or None on a non-zero exit. Raises like ``_git``."""
    result = _git(args, cwd, timeout=timeout, **kwargs)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_quiet(args, cwd, timeout: float = 10, log: str | None = None, **kwargs) -> None:
    """Fail-soft ``_git``: swallow every error, optionally logging it at DEBUG with *log* as prefix."""
    try:
        _git(args, cwd, timeout=timeout, **kwargs)
    except Exception as e:
        if log:
            logger.debug("%s: %s", log, e)


def _normalize_git_bash_path(p: Optional[str]) -> Optional[str]:
    """Translate a Git Bash path (``/c/..``, ``/cygdrive/c/..``, ``/mnt/c/..``) to ``C:\\..`` on Windows."""
    if not p or sys.platform != "win32":
        return p
    m = re.match(r"^/(?:(?:cygdrive|mnt)/)?([a-zA-Z])/(.*)$", p)
    if m:
        return f"{m.group(1).upper()}:\\{m.group(2).replace('/', chr(92))}"
    return p


def _git_repo_root() -> Optional[str]:
    """Return the git repo root for CWD (Git-Bash-normalized), or None if not in a repo."""
    try:
        return _normalize_git_bash_path(_git_out(["rev-parse", "--show-toplevel"], None, timeout=5))
    except Exception:
        return None


def _path_is_within_root(path: Path, root: Path) -> bool:
    """Return True when a resolved path stays within the expected root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def release_lsp_clients(wt_path: str) -> None:
    """Shut down this process's language servers for ``wt_path`` before ``git worktree remove``.

    A gateway outlives the sessions it runs, so without this the ``(server, root)`` client for the
    removed tree stays registered (tsserver heaps of several GiB pointed at a deleted worktree).
    Best-effort: LSP trouble must never block worktree removal.
    """
    try:
        from agent.lsp import release_workspace
        release_workspace(wt_path)
    except Exception as e:
        logger.debug("LSP release for worktree %s failed: %s", wt_path, e)


def _cleanup_failed_worktree_add(repo_root: str, wt_path: Path, branch_name: str) -> None:
    """Sweep the leftovers of a failed/timed-out ``git worktree add`` (fail-soft).

    ``worktree add`` is not transactional: killed mid-checkout it leaves the partial dir, a
    LOCKED admin entry naming the *live* pid (immune to the pruner's dead-pid unlock) and
    sometimes the branch, so any retry of the same name fails.
    """
    try:
        # Unlock first: `worktree remove --force` refuses a locked tree.
        _git_quiet(["worktree", "unlock", str(wt_path)], repo_root, timeout=15)
        _git_quiet(["worktree", "remove", "--force", str(wt_path)], repo_root, timeout=15)
        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        # `remove` needs the dir; `prune` drops the admin entry when it is already gone.
        _git_quiet(["worktree", "prune"], repo_root, timeout=15,
                   stdin=subprocess.DEVNULL, env=noninteractive_git_env())
        # A ref update runs the repository's reference-transaction hook (core.hooksPath).
        _git_quiet(["branch", "-D", branch_name], repo_root, timeout=15,
                   stdin=subprocess.DEVNULL, env=noninteractive_git_env())
    except Exception as e:
        logger.debug("cleanup after failed worktree add: %s", e)


_PACK_SPRAWL_THRESHOLD = 15
_REPACK_TIMEOUT = 1800
# One repack attempt per clone per interval, box-wide. Every ``hermes -w`` launch on a shared clone
# used to start its own ``git repack -a`` of the whole store; on a multi-agent box that stacked 50+
# concurrent multi-GB repacks (each too slow under the others to ever finish inside the timeout).
_REPACK_MIN_INTERVAL = 6 * 3600
_REPACK_LOCK = "hermes-repack.lock"


def _claim_repack_slot(git_dir: Path) -> bool:
    """Exactly one process per clone gets to repack per ``_REPACK_MIN_INTERVAL``."""
    return claim_maintenance_slot(git_dir, _REPACK_LOCK, _REPACK_MIN_INTERVAL)


def _run_bounded_repack(repo_root: str) -> None:
    """Incremental geometric repack whose whole process tree dies with the timeout or with us."""
    run_bounded_git(repo_root, ["repack", "-d", "--geometric=2", "--write-midx", "--quiet"], _REPACK_TIMEOUT)


def _maintain_pack_health(repo_root: str) -> None:
    """Repack on pack sprawl and keep the commit-graph fresh (background thread, fail-soft).

    ``gc --auto`` only fires at 50 packs; past a few dozen, every object lookup scans every
    pack index and worktree creation can blow its timeout under concurrent load.
    """
    try:
        # A linked worktree's or submodule's ``.git`` is a file; its objects live in the common dir.
        git_dir = common_git_dir(repo_root)
        pack_dir = git_dir / "objects" / "pack" if git_dir else None
        if pack_dir is None or not pack_dir.is_dir():
            return
        packs = len(list(pack_dir.glob("*.pack")))
        if packs >= _PACK_SPRAWL_THRESHOLD and _claim_repack_slot(git_dir):
            from hermes_cli.gitlock import clear_stale_tmp_packs
            clear_stale_tmp_packs(git_dir)
            logger.info("git pack sprawl (%d packs) — repacking in background", packs)
            _run_bounded_repack(repo_root)
            # Repacking can strand now-duplicated admin files; prune on the same pass.
            _git(["worktree", "prune"], repo_root, timeout=60, check=False,
                 stdin=subprocess.DEVNULL, env=noninteractive_git_env())
        if claim_commit_graph_refresh(git_dir):
            refresh_commit_graph(repo_root, git_dir)
    except Exception as e:
        logger.debug("pack maintenance skipped: %s", e)


def _resolve_worktree_base(repo_root: str, fetch_timeout: float = 5,
                           freshness_window: float = 300) -> tuple:
    """Resolve the freshest base ref to branch a new worktree from -> ``(base_ref, banner_label)``.

    Local ``HEAD`` can lag the remote by hundreds of commits, so try in order: (1) the current
    branch's upstream, refreshed; (2) the remote default branch (``origin/HEAD``), refreshed;
    (3) local ``HEAD``. The fetch is skipped when ``FETCH_HEAD`` is younger than
    *freshness_window* s, capped at *fetch_timeout*, and never retried: on failure the cached
    remote-tracking ref is used (the pre-push stale-base gate backstops genuine staleness).
    """
    from hermes_cli._subprocess_compat import noninteractive_git_env
    from hermes_cli.update_cmd_check import tracking_refspec

    def _run(args, timeout: float = 20):
        return _git(args, repo_root, timeout=timeout, stdin=subprocess.DEVNULL, env=noninteractive_git_env())

    def _ref_exists(ref: str) -> bool:
        try:
            return _run(["rev-parse", "--verify", "--quiet", ref + "^{commit}"]).returncode == 0
        except Exception:
            return False

    def _fetch_head_age() -> Optional[float]:
        try:
            gd = _run(["rev-parse", "--git-dir"])
            if gd.returncode != 0:
                return None
            fetch_head = Path(repo_root) / gd.stdout.strip() / "FETCH_HEAD"
            if not fetch_head.exists():
                return None
            return max(0.0, time.time() - fetch_head.stat().st_mtime)
        except Exception:
            return None

    def _refresh(remote: str, branch: str, ref: str) -> tuple:
        """(ref, label) after one best-effort fetch; never raises."""
        age = _fetch_head_age()
        if age is not None and age < freshness_window and _ref_exists(ref):
            return ref, f"{ref} (fetched {int(age)}s ago)"
        try:
            fetched = _run(["fetch", remote, tracking_refspec(remote, branch)], timeout=fetch_timeout)
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
        up = _run(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])
        if up.returncode == 0:
            upstream = up.stdout.strip()
            if upstream and "/" in upstream:
                remote, branch = upstream.split("/", 1)
                return _refresh(remote, branch, upstream)
    except Exception as e:
        logger.debug("worktree base: upstream resolution failed: %s", e)

    # 2. Remote default branch (origin/HEAD).
    try:
        head_ref = _run(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"])
        default_ref = (head_ref.stdout.strip().replace("refs/remotes/", "", 1)
                       if head_ref.returncode == 0 else "")
        if not default_ref:
            # origin/HEAD not set locally; ask the remote (network, capped like the fetch).
            show = _run(["remote", "show", "origin"], timeout=max(fetch_timeout, 5))
            for line in show.stdout.splitlines():
                line = line.strip()
                if line.startswith("HEAD branch:"):
                    _branch = line.split(":", 1)[1].strip()
                    if _branch and _branch != "(unknown)":
                        default_ref = "origin/" + _branch
                    break
        if default_ref and "/" in default_ref:
            remote, branch = default_ref.split("/", 1)
            return _refresh(remote, branch, default_ref)
    except Exception as e:
        logger.debug("worktree base: default-branch resolution failed: %s", e)

    # 3. Local HEAD (offline / no remote / detached).
    return "HEAD", "HEAD (local — could not reach remote)"


def _ensure_worktrees_gitignored(repo_root: str) -> None:
    """Append ``.worktrees/`` to the repo's .gitignore when missing (fail-soft)."""
    gitignore = Path(repo_root) / ".gitignore"
    try:
        # utf-8-sig: a Notepad BOM would glue to the first line and defeat the membership check.
        existing = gitignore.read_text(encoding="utf-8-sig", errors="replace") if gitignore.exists() else ""
        if ".worktrees/" not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(".worktrees/\n")
    except Exception as e:
        logger.debug("Could not update .gitignore: %s", e)


def _worktreeinclude_entries(repo_root) -> list:
    """Non-blank, non-comment entries of *repo_root*/``.worktreeinclude`` (``[]`` when absent).

    utf-8-sig, not the locale default: a cp1251/GBK locale would mojibake or raise on a UTF-8
    list, and a Notepad BOM would glue to the first entry.
    """
    include_file = Path(repo_root) / ".worktreeinclude"
    if not include_file.is_file():
        return []
    entries = []
    for line in include_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            entries.append(entry)
    return entries


def _copy_worktree_includes(repo_root: str, wt_path: Path) -> None:
    """Copy/symlink the entries listed in ``.worktreeinclude`` (gitignored files the agent needs)."""
    try:
        entries = _worktreeinclude_entries(repo_root)
        if not entries:
            return
        repo_root_resolved = Path(repo_root).resolve()
        wt_path_resolved = wt_path.resolve()
        for entry in entries:
            src, dst = Path(repo_root) / entry, wt_path / entry
            # Traversal/symlink-escape guard: both resolved endpoints must stay inside their roots.
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
            elif src.is_dir() and not dst.exists():
                # Symlink directories (no disk). Windows needs Developer Mode for symlinks: copy there.
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(str(src_resolved), str(dst))
                except (OSError, NotImplementedError) as _sym_err:
                    if sys.platform != "win32":
                        raise
                    logger.info(".worktreeinclude: symlink failed (%s) — falling back to copytree on Windows.",
                                _sym_err)
                    try:
                        shutil.copytree(str(src_resolved), str(dst), symlinks=True, dirs_exist_ok=False)
                    except Exception as _copy_err:
                        logger.warning(".worktreeinclude: copy fallback also failed for %s -> %s: %s",
                                       src, dst, _copy_err)
    except Exception as e:
        logger.debug("Error copying .worktreeinclude entries: %s", e)


def _worktree_add(repo_root: str, wt_path: Path, branch_name: str, base_ref: str, base_label: str):
    """``git worktree add`` with a local-HEAD retry -> ``(base_ref, base_label)``, or None on failure.

    Every failed attempt is swept with ``_cleanup_failed_worktree_add`` so the retry is not poisoned.
    """
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env

    # The checkout runs repo-named smudge filters too; refuse when they cannot be neutralized.
    env = noninteractive_repo_git_env(repo_root)
    if env is None:
        _cprint("\033[31m✗ Failed to create worktree: could not read the repository's filter config\033[0m")
        return None

    def _add(cfg):
        # 120s: on a multi-agent box the ~10k-file checkout contends for disk (113s measured under load).
        return _git([*cfg, "worktree", "add", str(wt_path), "-b", branch_name, base_ref], repo_root,
                    timeout=120, stdin=subprocess.DEVNULL, env=env)

    # checkout.workers parallelizes materialization; older git ignores unknown -c keys.
    try:
        result = _add(["-c", "checkout.workers=8", "-c", "checkout.thresholdForParallelism=100"])
        if result.returncode != 0:
            if base_ref != "HEAD":
                # A partial fetch can leave the remote ref unusable; never hard-fail on a sync hiccup.
                logger.warning("worktree add from %s failed (%s); retrying from local HEAD",
                               base_ref, result.stderr.strip())
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                base_ref, base_label = "HEAD", "HEAD (fallback — remote base failed)"
                result = _add([])
            if result.returncode != 0:
                _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
                _cprint(f"\033[31m✗ Failed to create worktree: {result.stderr.strip()}\033[0m")
                return None
    except Exception as e:
        _cleanup_failed_worktree_add(repo_root, wt_path, branch_name)
        _cprint(f"\033[31m✗ Failed to create worktree: {e}\033[0m")
        return None
    return base_ref, base_label


def _setup_worktree(repo_root: str | None = None, sync_base: bool = True,
                    name: Optional[str] = None, retain_db=None) -> Optional[dict[str, str]]:
    """Create an isolated git worktree -> ``{path, branch, repo_root, base}``, or None on failure.

    *sync_base* branches from the fetched remote tip (``_resolve_worktree_base``), else local
    HEAD. *name* replaces the random ``hermes-<id>``; named trees lack the ``hermes-`` prefix so
    the pruner ages them on its slower schedule. *retain_db* (``-w``, ``--tui -w``) records the tree
    for resume and names the profile store in the lock. A ``--tui -w`` tree outlives its launcher
    while a session there can resume in it; a classic ``-w`` row's cwd is the repo root, so its
    reaped clean tree is recreated on resume instead.

    Set ``worktree_sync: false`` in config to branch from local ``HEAD`` (the pre-#10760-followup behavior).
    """
    repo_root = repo_root or _git_repo_root()
    if not repo_root:
        _cprint("\033[31m✗ --worktree requires being inside a git repository.\033[0m")
        print("  cd into your project repo first, then run hermes -w")
        return None

    wt_name = ((name and re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")[:40])
               or f"hermes-{uuid.uuid4().hex[:8]}")
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt_path = worktrees_dir / wt_name
    if name and wt_path.exists():
        _cprint(f"\033[31m✗ Worktree already exists: {wt_path}\033[0m")
        print(f"  Pick a different name, or remove it with: git worktree remove {wt_path}")
        return None

    _ensure_worktrees_gitignored(repo_root)

    # Resolve the base ref. By default branch from the freshly-fetched remote tip so the worktree starts
    # current with the project, not from the (possibly stale) local HEAD of the standalone clone (#10760
    # follow-up).
    base_ref, base_label = (_resolve_worktree_base(repo_root) if sync_base
                            else ("HEAD", "HEAD (local — worktree_sync disabled)"))

    added = _worktree_add(repo_root, wt_path, branch_name, base_ref, base_label)
    if added is None:
        return None
    base_ref, base_label = added
    _copy_worktree_includes(repo_root, wt_path)

    # Lock so other processes (and `git worktree remove`) see it is in use; fail-soft.
    try:
        reason = f"hermes pid={os.getpid()}" + (f" db={retain_db}" if retain_db is not None else "")
        _git(["worktree", "lock", "--reason", reason, str(wt_path)], repo_root)
        logger.debug("Worktree locked: %s (pid=%s)", wt_path, os.getpid())
    except Exception as e:
        logger.debug("git worktree lock failed (non-fatal): %s", e)

    from hermes_cli.worktree_registry import record_worktree_repo
    record_worktree_repo(repo_root)
    if retain_db is not None:  # a -w tree: a resume may recreate it once quit removed it (N8)
        from hermes_cli.tui_worktree_resume import record_tui_worktree
        record_tui_worktree({"path": str(wt_path), "branch": branch_name, "repo_root": repo_root})
    _cprint(f"\033[32m✓ Worktree created:\033[0m {wt_path}")
    print(f"  Branch: {branch_name}")
    print(f"  Base:   {base_label}")
    return {"path": str(wt_path), "branch": branch_name, "repo_root": repo_root, "base": base_ref}


_REMOTE_TRUNK_CANDIDATES = ("origin/HEAD", "origin/main", "origin/master")
_LOCAL_TRUNK_CANDIDATES = ("main", "master")


def _worktree_local_trunk(path: str, timeout: float = 5) -> Optional[str]:
    """Local trunk of a repo with NO remote-tracking refs: ``main``/``master``, else the branch
    checked out in the main worktree. None = no baseline at all; callers must preserve.

    Nothing in such a repo was ever pushed, so "merged into the local trunk" is the only fact
    that can tell redundant scratch work from unique commits. May raise like ``_git``.
    """
    for name in _LOCAL_TRUNK_CANDIDATES:
        if _git_out(["rev-parse", "--verify", "--quiet", f"refs/heads/{name}"], path, timeout=timeout):
            return name
    porcelain = _git_out(["worktree", "list", "--porcelain"], path, timeout=timeout) or ""
    for line in porcelain.split("\n\n", 1)[0].splitlines():  # first block = the main worktree
        if line.startswith("branch refs/heads/"):
            return line[len("branch refs/heads/"):].strip() or None
    return None


def _worktree_merge_base_ref(path: str, timeout: float = 5) -> Optional[str]:
    """Ref merged work is judged against: ``origin/HEAD``/``origin/main``/``origin/master``, or the
    local trunk when the repo has no remote-tracking refs at all. None = nothing to compare
    against -> every consumer must preserve. May raise like ``_git``.
    """
    for cand in _REMOTE_TRUNK_CANDIDATES:
        if _git_out(["rev-parse", "--verify", "--quiet", cand], path, timeout=timeout):
            return cand
    if _git_out(["for-each-ref", "--format=%(refname)", "refs/remotes"], path, timeout=timeout) == "":
        return _worktree_local_trunk(path, timeout=timeout)
    return None


def _worktree_has_unpushed_commits(worktree_path: str, timeout: int = 10) -> bool:
    """Whether a worktree has commits unreachable from any remote branch. Fails SAFE toward True.

    No remote-tracking refs = nothing was ever pushed, so the tree is compared against the local
    trunk instead (``_worktree_local_trunk``); no trunk either -> True. A shallow boundary can
    disconnect an older HEAD from origin/* so public commits look unpushed;
    ``_deepen_shallow_repo`` first if affordable.
    """
    try:
        remote_refs = _git_out(["for-each-ref", "--format=%(refname)", "refs/remotes"], worktree_path,
                               timeout=timeout)
        if remote_refs is None:
            return True
        baseline = ["--remotes"]
        if not remote_refs:
            trunk = _worktree_local_trunk(worktree_path, timeout=timeout)
            if trunk is None:
                return True
            baseline = [trunk]
        unpushed = _git_out(["log", "--oneline", "HEAD", "--not", *baseline], worktree_path,
                            timeout=timeout)
        return unpushed is None or bool(unpushed)
    except Exception:
        return True


def _include_symlink_paths(worktree_path: str, repo_root) -> set:
    """Relative paths in *worktree_path* that are ``.worktreeinclude`` directory symlinks.

    ``_copy_worktree_includes`` symlinks included directories back to the main checkout
    (*repo_root*). A trailing-slash gitignore pattern
    (``node_modules/``) never matches a symlink, so git reports each one as untracked. Only a
    symlink at a listed entry that resolves to that same entry in the main repo qualifies —
    anything else is real user state.
    """
    root = Path(repo_root)
    wt = Path(worktree_path)
    paths = set()
    for entry in _worktreeinclude_entries(root):
        dst = wt / entry
        if dst.is_symlink() and dst.resolve() == (root / entry).resolve():
            paths.add(Path(entry).as_posix().rstrip("/"))  # git status paths use "/" on Windows too
    return paths


def _worktree_is_dirty(worktree_path: str, repo_root, timeout: int = 10) -> bool:
    """Whether a worktree has staged/unstaged/untracked changes. Fails SAFE toward True.

    Untracked ``.worktreeinclude`` directory symlinks back to the main checkout *repo_root* are
    ignored: they are our own scaffolding, and counting them would keep every worktree of such a
    repo forever. ``status.showUntrackedFiles=no`` is overridden (it would hide new files), and
    skip-worktree / assume-unchanged files whose bytes differ from the index count as changes
    (``git status`` trusts both flags).
    """
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env
    try:
        # Reclaimers run this unattended; status reads the index (core.fsmonitor, clean filters).
        env = noninteractive_repo_git_env(worktree_path)
        if env is None:
            return True
        result = _git(["status", "--porcelain", "-z", "--untracked-files=normal"], worktree_path,
                      timeout=timeout, stdin=subprocess.DEVNULL, env=env)
        if result.returncode != 0:
            return True
        entries = [e for e in result.stdout.split("\0") if e]
        if not entries:
            from hermes_cli.worktree_retention import hidden_index_edits
            return bool(hidden_index_edits(worktree_path))
        if any(not e.startswith("?? ") for e in entries):
            return True
        include_links = _include_symlink_paths(worktree_path, repo_root)
        return any(e[3:].rstrip("/") not in include_links for e in entries)
    except Exception:
        return True


def _repo_is_shallow(repo_path: str, timeout: int = 5) -> bool:
    """Whether *repo_path* is a shallow clone (installer default). Fails toward False on unknown state.

    Shallowness poisons connectivity verdicts: an old worktree HEAD misreports as unpushed forever.
    """
    try:
        return _git_out(["rev-parse", "--is-shallow-repository"], repo_path, timeout=timeout) == "true"
    except Exception:
        return False


def _deepen_shallow_repo(repo_root: str, timeout: int = 600) -> bool:
    """Blobless unshallow so history verdicts are correct -> whether the repo is non-shallow afterwards.

    Falls back to a plain ``--unshallow`` if the server rejects filters. Background paths only.
    """
    if not _repo_is_shallow(repo_root):
        return True
    try:
        remotes = _git_out(["remote"], repo_root)
        if not remotes:
            return False
        names = [r.strip() for r in remotes.splitlines() if r.strip()]
        remote = "origin" if "origin" in names else names[0]

        try:
            for extra in (["--filter=blob:none"], []):
                try:
                    # Unattended fetch: a repo-level core.sshCommand / credential.helper must not run.
                    result = _git(["fetch", remote, "--unshallow", *extra], repo_root, timeout=timeout,
                                  stdin=subprocess.DEVNULL, env=noninteractive_git_env())
                except subprocess.TimeoutExpired:
                    return False
                if result.returncode == 0:
                    break
                logger.debug("git fetch --unshallow%s failed: %s", " " + " ".join(extra) if extra else "",
                             result.stderr.strip()[-500:])
        finally:
            # The filtered attempt makes the clone partial (git writes the config before fetching,
            # so even when it fails); its old packs need the marker or git 2.53+ crashes every
            # later fetch (#124272). Markers are inert in a clone without a promisor remote.
            from hermes_cli.gitlock import mark_unmarked_packs_promisor
            mark_unmarked_packs_promisor(Path(repo_root))
    except Exception as e:
        logger.debug("Deepening shallow repo failed (non-fatal): %s", e)
        return False

    deepened = not _repo_is_shallow(repo_root)
    if deepened:
        logger.info("Deepened shallow clone at %s so worktree cleanup can verify push state", repo_root)
    return deepened


# Retained `git cherry` verdict entries (~90 bytes each).
_WORKTREE_MERGE_CACHE_MAX = 1000


def _worktree_merge_cache_path() -> Path:
    """Path of the patch-equivalence verdict cache (profile-aware)."""
    return get_hermes_home() / "cache" / "worktree_merge_verdicts.json"


def _load_worktree_merge_cache() -> dict[str, bool]:
    """Load the ``git cherry`` verdict cache. Missing/corrupt cache = empty."""
    try:
        entries = json.loads(_worktree_merge_cache_path().read_text(encoding="utf-8-sig")).get("verdicts")
    except Exception:
        return {}
    # A hand-edited or partially written cache must never inject a non-bool verdict.
    return {k: v for k, v in entries.items() if isinstance(v, bool)} if isinstance(entries, dict) else {}


def _save_worktree_merge_cache(verdicts: dict[str, bool]) -> None:
    """Atomically persist the newest ``_WORKTREE_MERGE_CACHE_MAX`` verdicts. Never raises."""
    try:
        items = list(verdicts.items())[-_WORKTREE_MERGE_CACHE_MAX:]
        atomic_json_write(_worktree_merge_cache_path(), {"version": 1, "verdicts": dict(items)}, indent=None)
    except Exception as e:
        logger.debug("Could not persist worktree merge cache: %s", e)


def _worktree_commits_all_merged_upstream(
    worktree_path: str, timeout: int = 30, max_ahead: int = 20, cache: Optional[dict[str, bool]] = None,
) -> bool:
    """Whether every local-only commit is patch-equivalent (``git cherry``) to upstream. Fails SAFE -> False.

    Catches squash-merged/cherry-picked PRs whose remote branch was deleted (commits unreachable
    from ``refs/remotes/*`` forever). Upstream is ``_worktree_merge_base_ref`` (the local trunk in
    a repo without remotes). More than *max_ahead* ahead = stale-base tree -> False.
    *cache* memoizes on ``(base_sha, head_sha, max_ahead)``, exactly what ``git cherry`` consumes.
    """
    try:
        base = _worktree_merge_base_ref(worktree_path, timeout=timeout)
        if base is None:
            return False

        cache_key = None
        if cache is not None:
            revs = _git_out(["rev-parse", f"{base}^{{commit}}", "HEAD^{commit}"], worktree_path,
                            timeout=timeout)
            shas = (revs or "").split()
            if len(shas) == 2:
                cache_key = f"{shas[0]}..{shas[1]}:{max_ahead}"
                if cache_key in cache:
                    return cache[cache_key]

        def _memo(verdict: bool) -> bool:
            if cache is not None and cache_key is not None:
                cache[cache_key] = verdict
            return verdict

        ahead = _git_out(["rev-list", "--count", f"{base}..HEAD"], worktree_path, timeout=timeout)
        if ahead is None:
            return False
        count = int(ahead or "0")
        if count == 0:
            return _memo(True)
        if count > max_ahead:
            return _memo(False)

        cherry = _git(["cherry", base, "HEAD"], worktree_path, timeout=timeout)
        if cherry.returncode != 0:
            return False
        lines = [ln for ln in cherry.stdout.splitlines() if ln.strip()]
        # "-" = patch-equivalent upstream; "+" = unique local work
        return _memo(bool(lines) and all(ln.startswith("-") for ln in lines))
    except Exception:
        return False


def _worktree_current_branch(worktree_path: str, timeout: int) -> Optional[str]:
    """Checked-out branch name, or None when detached/git fails. May raise on subprocess errors."""
    branch = _git_out(["rev-parse", "--abbrev-ref", "HEAD"], worktree_path, timeout=timeout)
    return branch if branch and branch != "HEAD" else None  # "HEAD" = detached


def _worktree_branch_pr_merged(
    worktree_path: str, timeout: int = 15, cache: Optional[dict[str, bool]] = None,
) -> bool:
    """Whether the branch's PR is MERGED on GitHub (``gh pr list``). Fails SAFE toward False.

    Catches rebase-merges whose altered diff defeats ``git cherry``. Memoized on
    ``(branch, head_sha)``; only True is cached since the PR may merge later without new commits.
    """
    try:
        branch = _worktree_current_branch(worktree_path, timeout)
        if branch is None:
            return False

        cache_key = None
        if cache is not None:
            sha = _git_out(["rev-parse", "HEAD"], worktree_path, timeout=timeout)
            if sha:
                cache_key = f"pr-merged:{branch}:{sha}"
                if cache.get(cache_key) is True:
                    return True

        result = subprocess.run(
            ["gh", "pr", "list", "--head", branch, "--state", "merged", "--json", "number", "--limit", "1"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=worktree_path,
            check=False,
        )
        if result.returncode != 0:
            return False
        prs = json.loads(result.stdout or "[]")
        merged = isinstance(prs, list) and bool(prs)
        if merged and cache is not None and cache_key is not None:
            cache[cache_key] = True
        return merged
    except Exception:
        return False


def _fetch_remote_branch_heads(repo_root: str, timeout: int = 20) -> Optional[dict[str, str]]:
    """``{branch: sha}`` for every branch on origin (one ``ls-remote``), or None = cannot verify, preserve.

    Managed installs fetch a single-branch refspec, so pushed PR branches have no
    remote-tracking ref and would read as unpushed forever.
    """
    try:
        # Runs from the reclaim sweep, unattended: same transport pins as the base fetch.
        result = _git(["ls-remote", "--heads", "origin"], repo_root, timeout=timeout,
                      stdin=subprocess.DEVNULL, env=noninteractive_git_env())
        if result.returncode != 0:
            return None
        pairs = (line.split("\t", 1) for line in result.stdout.splitlines())
        return {p[1][len("refs/heads/"):].strip(): p[0].strip()
                for p in pairs if len(p) == 2 and p[1].startswith("refs/heads/")}
    except Exception:
        return None


def _worktree_branch_pushed_exact(
    worktree_path: str, remote_heads: Optional[dict[str, str]], timeout: int = 10,
) -> bool:
    """Whether the branch head is EXACTLY what origin holds (tree redundant; reap it, keep the branch).

    Equality is deliberately the only True case: ahead/diverged heads have commits origin lacks
    and ancestry can't be proven cheaply without remote-tracking refs -> fail SAFE toward preserve.
    """
    if not remote_heads:
        return False
    try:
        branch = _worktree_current_branch(worktree_path, timeout)
        remote_sha = remote_heads.get(branch) if branch is not None else None
        return bool(remote_sha) and (
            _git_out(["rev-parse", "HEAD"], worktree_path, timeout=timeout) == remote_sha)
    except Exception:
        return False


# ``hermes pid=<pid>`` (``/worktree new``), ``hermes pid=<pid> db=<state.db>`` (``-w`` / ``--tui -w`` launch)
# and ``hermes session=<id> db=<state.db>`` (re-lock when the TUI exits). Read raw from the admin
# dir: plain ``list --porcelain`` C-quotes a reason holding a backslash, a quote, a control or a
# non-ASCII byte (every Windows path), and the ``db=`` path would never parse.
_HERMES_LOCK_RE = re.compile(r"hermes (?:pid=(\d+)|session=\S+)(?: db=(.+))?", re.DOTALL)


def _retain_worktree_for_session(repo_root: str, worktree_path: str, session_id: str, db_path) -> bool:
    """Re-lock a ``--tui -w`` tree its exiting launcher retains, naming the session that uses it.

    The tree stays live while any resumable session of that store works inside it
    (``_retained_tree_is_live``), not only the named one: ``/new`` leaves the older session's
    history in the same checkout.
    """
    _git_quiet(["worktree", "unlock", worktree_path], repo_root, log="worktree unlock before retain failed")
    try:
        result = _git(["worktree", "lock", "--reason", f"hermes session={session_id} db={db_path}",
                       worktree_path], repo_root)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("worktree retain lock failed for %s: %s", worktree_path, exc)
        return False
    return result.returncode == 0


def _inside_tree(worktree_path: str):
    """``inside(path) -> 0/1``: *path* is the tree or under it (native path identity: Windows is
    case-insensitive, either separator; the tree's own spelling and its resolved one)."""
    def key(value) -> str:
        return os.path.normcase(str(value)).replace("\\", "/").rstrip("/")

    roots = {key(Path(worktree_path)), key(Path(worktree_path).resolve())}

    def inside(cwd) -> int:
        return int(bool(cwd) and any(key(cwd) == r or key(cwd).startswith(r + "/") for r in roots))
    return inside


# The gateway's own records that it will still act in a session after its last viewer quits:
# an active /loop, /heartbeat or /goal (its continuation) row (``hermes_cli.loops`` / ``heartbeat`` /
# ``goals``, keyed by the stored session id, status ``active`` when the field is absent), and a row
# no viewer close ended: Ink's exit detaches with ``tui_shutdown`` and the owner ends the idle row
# only once no other viewer holds it (``gateway/session_viewer_close.py``).
_SERVED_TREE_SQL = (
    "SELECT 1 FROM sessions s WHERE hermes_inside_tree(s.cwd) AND (s.ended_at IS NULL OR EXISTS ("
    "SELECT 1 FROM state_meta m WHERE m.key IN ('loop:'||s.id, 'heartbeat:'||s.id, 'goal:'||s.id) "
    "AND (json_valid(m.value)=0 OR COALESCE(json_extract(m.value,'$.status'),'active')='active'))) LIMIT 1")


def _tree_still_served(db_path: str, worktree_path: str) -> Optional[str]:
    """Why the gateway still needs a ``--tui -w`` tree the launcher is about to remove, or None.

    Beyond pending admissions (``_retained_tree_is_live(pending_only=True)``): scheduled work for a
    session inside it (an active loop, heartbeat or goal continuation), another viewer still holding
    such a session (its row was not ended by the quit), or an enabled cron job that runs in the tree
    (``workdir``) or is bound to such a session's local chat. Reads only; unreadable fails SAFE."""
    import sqlite3
    from urllib.parse import quote

    path = Path(db_path)
    chats: set = set()
    if path.is_file():
        try:
            conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=2)
            try:
                conn.create_function("hermes_inside_tree", 1, _inside_tree(worktree_path), deterministic=True)
                if conn.execute(_SERVED_TREE_SQL).fetchone():
                    return "session"
                chats = {row[0] for row in conn.execute(
                    "SELECT chat_id FROM sessions WHERE hermes_inside_tree(cwd) AND chat_id IS NOT NULL")}
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.debug("retained worktree serve check failed (%s); keeping tree", exc)
            return "unreadable"
    return "cron" if _cron_bound_to_tree(worktree_path, chats) else None


def _cron_bound_to_tree(worktree_path: str, chats: set) -> bool:
    """An enabled, non-terminal cron job whose ``workdir`` is inside the tree, or whose origin is
    the local chat of a session there. Reads ``jobs.json`` directly (``load_jobs`` may repair-write);
    an unreadable store fails SAFE."""
    try:
        from cron.jobs import _current_cron_store, _parse_jobs_file, is_job_runnable, is_terminal_job
        jobs_file = _current_cron_store().jobs_file
        if not jobs_file.exists():
            return False
        data, _ = _parse_jobs_file(jobs_file)
    except Exception as exc:  # health: allow BLE001 -- an unreadable cron store keeps the tree (fail safe)
        logger.debug("cron store unreadable (%s); keeping tree", exc)
        return True
    jobs = data.get("jobs") if isinstance(data, dict) else data
    inside = _inside_tree(worktree_path)
    for job in jobs if isinstance(jobs, list) else ():
        if not isinstance(job, dict) or not is_job_runnable(job) or is_terminal_job(job):
            continue
        origin = job.get("origin") if isinstance(job.get("origin"), dict) else {}
        if inside(job.get("workdir")) or (
                str(origin.get("platform") or "").lower() == "local" and str(origin.get("chat_id") or "") in chats):
            return True
    return False


def _retained_tree_is_live(db_path: str, worktree_path: str, *, pending_only: bool = False) -> bool:
    """Whether a session that can still be resumed has its frozen cwd inside the retained tree:
    a row with history or pending gateway work (*pending_only*: pending work alone, the exiting
    ``--tui -w`` launcher's "is the gateway still using it" check). A missing store means the
    profile is gone; unreadable fails SAFE (live)."""
    import sqlite3
    from urllib.parse import quote

    path = Path(db_path)
    if not path.is_file():
        return False
    inside = _inside_tree(worktree_path)
    try:
        conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=2)
        try:
            conn.create_function("hermes_inside_tree", 1, inside, deterministic=True)
            row = conn.execute(
                "SELECT 1 FROM sessions s WHERE hermes_inside_tree(s.cwd) AND ("
                + ("" if pending_only else "s.message_count>0 OR ")
                + "EXISTS (SELECT 1 FROM session_admissions a WHERE a.target_session_id=s.id "
                "AND a.status!='terminal')) LIMIT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("retained worktree session check failed (%s); keeping tree", exc)
        return True
    return row is not None


def _worktree_lock_reason(repo_root: str, worktree_path: str, timeout: int):
    """``(found, reason)`` from the tree's admin dir (``gitdir:`` of its ``.git`` file, under
    ``<common-dir>/worktrees/``); reason is ``None`` when unlocked and ``""`` for a bare lock.
    Read raw and trimmed, as git does: ``list --porcelain -z`` needs git 2.36. Raises when git
    or the admin dir cannot be read."""
    common = _git_out(["rev-parse", "--git-common-dir"], repo_root, timeout=timeout)
    if not common:
        raise RuntimeError("not a git repository")
    tree = Path(worktree_path)
    if not (tree / ".git").is_file():
        return False, None  # no tree there, or the main checkout
    link = (tree / ".git").read_text(encoding="utf-8-sig")
    if not link.startswith("gitdir:"):
        raise RuntimeError(f"unreadable .git file in {tree}")
    admin = (tree / link[len("gitdir:"):].strip()).resolve()
    if admin.parent != (Path(repo_root) / common / "worktrees").resolve():
        return False, None  # a checkout of another repository
    try:
        return True, (admin / "locked").read_text(encoding="utf-8-sig", errors="replace").strip(" \t\n\v\f\r")
    except FileNotFoundError:
        return True, None


def _worktree_lock_is_live(repo_root: str, worktree_path: str, timeout: int = 10):
    """Lock state: ``"live"`` (owning pid runs / a resumable session works in the tree),
    ``"dead"`` (pid and sessions gone / non-hermes reason), None (unlocked).

    A Hermes tree locks with reason ``hermes pid=<pid>``; ``worktree remove --force`` refuses
    locked trees, so a crashed session's lock would keep its tree forever. A ``-w`` tree
    also names its profile's ``state.db`` from creation, so a ``--tui -w`` launcher killed with
    its terminal (SIGHUP skips the exit re-lock) still keeps the tree while a resumable session's
    row cwd is inside it. A classic ``-w`` row's cwd is the repo root: its tree is only recorded,
    and a reaped clean one is recreated on resume. Fails SAFE toward "live", including a
    ``hermes`` reason this version cannot parse.
    """
    try:
        found, reason = _worktree_lock_reason(repo_root, worktree_path, timeout)
    except Exception:
        return "live"
    if not found or reason is None:
        return None
    if not reason.startswith("hermes "):
        # A foreign lock here is a leftover; the age/dirty/unpushed gates already passed.
        return "dead"
    lock = _HERMES_LOCK_RE.fullmatch(reason)
    if lock is None:
        return "live"
    pid, db_path = lock.groups()
    if pid is not None:
        if int(pid) == os.getpid():
            return "live"
        try:
            from gateway.status import _pid_exists
            if _pid_exists(int(pid)):
                return "live"
        except Exception:
            return "live"
    if db_path is not None:
        return "live" if _retained_tree_is_live(db_path, worktree_path) else "dead"
    return "dead" if pid is not None else "live"  # a session lock without its store: unverifiable


# Trees Hermes names itself (``hermes -w`` sessions, subagent isolation) age on the base tier.
_AUTO_TREE_PREFIXES = ("hermes-", "subagent-")


def prune_tier_hours(name: str, max_age_hours: float) -> float:
    """Idle hours before a tree may be reclaimed: *max_age_hours* for Hermes-made trees
    (``hermes-*``, ``subagent-*``), 3x for deliberately named ones."""
    return max_age_hours if name.startswith(_AUTO_TREE_PREFIXES) else max_age_hours * 3


def _prune_candidates(worktrees_dir: Path, max_age_hours: int, now: float) -> list:
    """Phase 1, stat-only age filter -> ``[(entry, mtime, force)]`` for trees past their soft cutoff.

    Kanban trees (``t_<hex>``) belong to the kanban gc. Hermes-made trees age on *max_age_hours*,
    deliberately named trees at 3x (``prune_tier_hours``); *force* marks the hard (3x) tier.
    """
    kanban_re = re.compile(r"^t_[0-9a-f]+$")
    candidates: list = []
    for entry in sorted(worktrees_dir.iterdir()):
        if not entry.is_dir() or kanban_re.match(entry.name):
            continue
        tier_hours = prune_tier_hours(entry.name, max_age_hours)
        try:
            mtime = entry.stat().st_mtime
            if mtime > now - (tier_hours * 3600):
                continue  # Too recent — skip
        except Exception:
            continue
        candidates.append((entry, mtime, mtime <= now - (tier_hours * 3 * 3600)))
    return candidates


def _classify_prune_candidates(repo_root: str, candidates: list) -> list:
    """Phase 2, parallel read-only classification -> ``[(entry, mtime, force, verdict, lock_state)]``.

    verdict in ``dirty`` / ``unpushed`` / ``locked-live`` / ``reap`` / ``reap-keep-branch``. Each
    check is a read-only query on a distinct worktree (no repo-wide lock), so a bounded pool is
    safe; mutation stays serial. ``git cherry`` verdicts are memoized on disk.
    """
    merge_cache = _load_worktree_merge_cache()
    cache_size_before = len(merge_cache)
    cache_lock = threading.Lock()

    # Lazy once-per-sweep ls-remote: only paid when a tree reaches the pushed tier (TUI runs this sync).
    _remote_heads_memo: dict = {}
    _remote_heads_lock = threading.Lock()

    def _get_remote_heads():
        with _remote_heads_lock:
            if "heads" not in _remote_heads_memo:
                _remote_heads_memo["heads"] = _fetch_remote_branch_heads(repo_root, timeout=10)
            return _remote_heads_memo["heads"]

    def _classify(item):
        entry, mtime, force = item
        # Never delete real work regardless of age: only clean, merged/pushed trees are reaped.
        if _worktree_is_dirty(str(entry), repo_root, timeout=5):
            return (entry, mtime, force, "dirty", None)
        keep_branch = False
        if _worktree_has_unpushed_commits(str(entry), timeout=5):
            # Squash-merge escape hatch: patch-equivalent commits are merged, not unpushed.
            with cache_lock:
                snapshot = dict(merge_cache)
            merged = _worktree_commits_all_merged_upstream(str(entry), timeout=30, cache=snapshot)
            if not merged:
                # Rebase-merge escape hatch: cherry misses changed patch-ids, GitHub knows.
                merged = _worktree_branch_pr_merged(str(entry), timeout=15, cache=snapshot)
            with cache_lock:
                merge_cache.update(snapshot)
            # Pushed tier: head EXACTLY matches origin -> reap the tree, keep the branch (open-PR anchor).
            if not merged and not _worktree_branch_pushed_exact(str(entry), _get_remote_heads(),
                                                                 timeout=10):
                return (entry, mtime, force, "unpushed", None)
            keep_branch = not merged

        # Live lock = running hermes; a dead lock is unlocked in phase 3.
        lock_state = _worktree_lock_is_live(repo_root, str(entry), timeout=5)
        if lock_state == "live":
            return (entry, mtime, force, "locked-live", None)
        return (entry, mtime, force, "reap-keep-branch" if keep_branch else "reap", lock_state)

    # Enough workers to hide git's per-process startup latency without dozens of gits.
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
        logger.debug("Parallel worktree classification failed (%s); serial", e)
        verdicts = [_classify(c) for c in candidates]

    if len(merge_cache) != cache_size_before:
        _save_worktree_merge_cache(merge_cache)
    return verdicts


# Preserving verdicts -> reason reported for trees past the stale-work cutoff.
_PRESERVE_REASONS = {"dirty": "uncommitted changes", "unpushed": "unpushed commits"}


def removal_blocked(path: str) -> Optional[tuple[str, str]]:
    """``(keep-action, reason)`` when deleting the clean tree at *path* would lose something no
    branch holds (``worktree_retention.removal_blocker``: a checked-out submodule, an embedded
    repository, a per-worktree ref); unverifiable counts as blocked."""
    from hermes_cli.worktree_retention import removal_blocker
    return removal_blocker(path)


def _reap_prune_verdicts(repo_root: str, verdicts: list, stale_work_cutoff: float) -> tuple[list, set]:
    """Phase 3, serial unlock / remove / branch -D -> ``(preserved_stale, kept_branches)``.

    *kept_branches* must survive the orphaned-branch pass. Branch deletion is gated on
    ``worktree remove`` succeeding so a failed removal never orphans reachable commits.
    """
    preserved_stale: list = []
    kept_branches: set = set()
    for entry, mtime, force, verdict, lock_state in verdicts:
        reason = _PRESERVE_REASONS.get(verdict)
        if reason:
            if mtime <= stale_work_cutoff:
                preserved_stale.append(f"{entry.name} ({reason})")
            continue
        if verdict == "locked-live":
            logger.debug("Skipping live-locked worktree: %s", entry.name)
            continue
        blocked = removal_blocked(str(entry))
        if blocked:
            preserved_stale.append(f"{entry.name} ({blocked[1]})")
            continue

        if lock_state == "dead":
            _git_quiet(["worktree", "unlock", str(entry)], repo_root,
                       log=f"Failed to unlock dead worktree {entry.name}")

        try:
            branch = _git(["branch", "--show-current"], str(entry), timeout=5).stdout.strip()
            remove_result = _git(["worktree", "remove", str(entry), "--force"], repo_root, timeout=15)
            if remove_result.returncode != 0:
                logger.debug("Failed to remove worktree %s: %s", entry.name, remove_result.stderr.strip())
                continue
            if branch and verdict == "reap-keep-branch":
                kept_branches.add(branch)
            elif branch:
                # Unattended sweep: the repository's reference-transaction hook must not run.
                _git(["branch", "-D", branch], repo_root,
                     stdin=subprocess.DEVNULL, env=noninteractive_git_env())
            logger.debug("Pruned stale worktree: %s (force=%s)", entry.name, force)
        except Exception as e:
            logger.debug("Failed to prune worktree %s: %s", entry.name, e)
    return preserved_stale, kept_branches


def _prune_stale_worktrees(repo_root: str, max_age_hours: int = 24) -> None:
    """Remove stale worktrees and orphaned branches on startup.

    Default (``worktrees.archive_dirty: true``): ``worktree_sweep.run_sweep`` — an idle dirty or
    unpushed tree past its tier is archived to ``refs/hermes/archive/<tree>`` and then removed;
    live trees (lock owner or any process inside) and Kanban trees are never touched.
    ``archive_dirty: false`` (no size budget) runs the classic pass below: dirty trees are never
    removed; unpushed commits are never removed UNLESS patch-equivalent to upstream, the PR is
    MERGED on GitHub, or the head EXACTLY matches origin (tree reaped, branch kept). Live-locked
    trees are skipped; dead locks are unlocked first. Phases: ``_prune_candidates`` ->
    ``_classify_prune_candidates`` -> ``_reap_prune_verdicts`` -> ``_prune_orphaned_branches``.
    """
    try:
        _prune_stale_worktree_trees(repo_root, max_age_hours)
    finally:
        _reclaim_orphan_install_states()


def _reclaim_orphan_install_states() -> None:
    """Trees removed by any path (the prune pass, `git worktree remove` by hand, `rm -rf` of a
    scratch clone) leave their ~200 MB dependency state under <home>/installs/; reclaim it on
    every pass, including the early returns where no tree is old enough to prune. Startup
    maintenance must never block a launch; the traceback goes to the debug log."""
    try:
        from pm.environments import installs_root
        from pm.install_states import collect_orphan_install_states

        reclaimed = collect_orphan_install_states(installs_root())
        if reclaimed:
            logger.info("Reclaimed %d dependency state dir(s) of deleted checkouts", len(reclaimed))
    except Exception:
        logger.debug("Orphan install-state reclaim failed", exc_info=True)


def _prune_stale_worktree_trees(repo_root: str, max_age_hours: int) -> None:
    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        _prune_orphaned_branches(repo_root)
        return

    # Shallow clones make every aged tree read as unpushed forever; deepen once (fail-soft).
    if _repo_is_shallow(repo_root):
        _deepen_shallow_repo(repo_root)

    from hermes_cli import worktree_sweep
    settings = worktree_sweep.sweep_settings()
    if not worktree_sweep.classic_mode(settings):
        worktree_sweep.run_sweep(repo_root, max_age_hours, settings)
        _warn_worktree_footprint(repo_root)
        return

    now = time.time()
    candidates = _prune_candidates(worktrees_dir, max_age_hours, now)
    if not candidates:
        _prune_orphaned_branches(repo_root)
        return

    verdicts = _classify_prune_candidates(repo_root, candidates)
    preserved_stale, kept_branches = _reap_prune_verdicts(repo_root, verdicts, now - (7 * 24 * 3600))

    if preserved_stale:
        logger.warning("Preserving %d worktree(s) older than 7 days with unmerged work "
                       "(run `hermes worktree prune` to review and reclaim): %s",
                       len(preserved_stale), ", ".join(sorted(preserved_stale)))

    _prune_orphaned_branches(repo_root, protect=kept_branches)
    _warn_worktree_footprint(repo_root)


def _warn_worktree_footprint(repo_root: str) -> None:
    """The startup pass can leave trees it may not reclaim; say so once per launch."""
    try:
        from hermes_cli.worktree_gc import worktrees_summary

        count, size_mb = worktrees_summary(repo_root, size_below_count=10)
        if count >= 10 or (size_mb or 0) >= 5120:
            size_txt = f" ({size_mb / 1024:.1f}GB)" if size_mb else ""
            logger.warning(".worktrees/ holds %d tree(s)%s — run `hermes worktree list` "
                           "to audit and `hermes worktree prune` to reclaim safely.", count, size_txt)
    except Exception:
        logger.debug("worktree summary failed", exc_info=True)


def _prune_orphaned_branches(repo_root: str, protect: Optional[set] = None) -> None:
    """Delete local ``hermes/hermes-*`` and ``pr-*`` branches with no worktree, except *protect*."""
    try:
        listing = _git_out(["branch", "--format=%(refname:short)"], repo_root)
        if listing is None:
            return
        all_branches = [b.strip() for b in listing.split("\n") if b.strip()]
    except Exception:
        return

    active_branches: set = set()
    try:
        wt_result = _git(["worktree", "list", "--porcelain"], repo_root)
        for line in wt_result.stdout.split("\n"):
            if line.startswith("branch refs/heads/"):
                active_branches.add(line.split("branch refs/heads/", 1)[-1].strip())
    except Exception:
        return  # can't determine active branches: bail

    # Also protect the checked-out branch and main.
    try:
        current = _git(["branch", "--show-current"], repo_root, timeout=5).stdout.strip()
        if current:
            active_branches.add(current)
    except Exception:
        pass
    active_branches.add("main")

    orphaned = [b for b in all_branches if b not in active_branches and b not in (protect or ())
                and (b.startswith("hermes/hermes-") or b.startswith("pr-"))]

    if not orphaned:
        return
    for i in range(0, len(orphaned), 50):
        _git_quiet(["branch", "-D"] + orphaned[i:i + 50], repo_root, timeout=30,
                   log="Failed to prune orphaned branches",
                   stdin=subprocess.DEVNULL, env=noninteractive_git_env())
    logger.debug("Pruned %d orphaned branches", len(orphaned))
