"""``hermes worktree merge`` — fold a child branch into the current worktree, all-or-nothing.

A coordinator that fans work out to worktree-isolated subagents gets a branch back from each one.
Merging them by hand trusts every child's summary. This command checks the branch against what the
coordinator asked for before the merge lands, and keeps it only when the check passes:

* ``--allow PATH``: every file the branch changes (both sides of a rename) must equal a listed path
  or sit under a listed directory. Any file outside them refuses the merge.
* ``--tested SHA``: the commit the child says it verified. It must be on the branch and be its tip;
  commits after it were never run through that verification, so they refuse the merge.
* ``--gate CMD``: runs in the worktree after the merge commit. A failure or timeout (which kills
  the gate's whole process tree) puts the branch back at the commit it had before the merge.

Before merging it also refuses: tracked files with uncommitted changes (including ones hidden by
assume-unchanged/skip-worktree), and any path the branch writes that exists here untracked, since
git silently overwrites ignored files. So the rollback only touches what the merge brought in; files
the gate created or staged stay on disk. A gate that moves HEAD itself (a commit) is reported and
nothing is reset. A conflict aborts the merge and lists the conflicting files.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

_GIT_TIMEOUT = 60
_OUTPUT_TAIL_CHARS = 4000


class MergeRefused(Exception):
    """A precondition failed; nothing was changed."""


@dataclass
class GateResult:
    command: str
    exit_code: Optional[int]
    timed_out: bool
    duration_s: float
    output_tail: str


@dataclass
class MergeResult:
    branch: str
    tip: str = ""
    merged: bool = False
    merge_commit: str = ""
    already_merged: bool = False
    files: list[str] = field(default_factory=list)
    outside_allowed: list[str] = field(default_factory=list)
    untested_commits: list[str] = field(default_factory=list)
    untracked_collisions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    gate: Optional[GateResult] = None
    reset_to: str = ""
    error: str = ""


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    from hermes_cli._subprocess_compat import harden_git_argv
    from hermes_cli.worktree_ops import _git as run_git
    result = run_git(harden_git_argv(args), repo, timeout=_GIT_TIMEOUT, stdin=subprocess.DEVNULL)
    if check and result.returncode != 0:
        raise MergeRefused(f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()[:300]}")
    return result


def _out(repo: str, *args: str) -> str:
    return _git(repo, *args).stdout.strip()


def _paths(repo: str, *args: str) -> list[str]:
    """NUL-separated path output, read as bytes: text mode would turn a ``\\r`` in a name into ``\\n``."""
    from hermes_cli._subprocess_compat import harden_git_argv
    result = subprocess.run(["git", *harden_git_argv(args)], cwd=repo, capture_output=True,
                            stdin=subprocess.DEVNULL, timeout=_GIT_TIMEOUT, check=False)
    if result.returncode != 0:
        raise MergeRefused(f"git {args[0]} failed: {result.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return sorted({os.fsdecode(name) for name in result.stdout.split(b"\0") if name})


def _resolve(repo: str, rev: str, what: str) -> str:
    result = _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False)
    if result.returncode != 0:
        raise MergeRefused(f"{what} {rev!r} does not name a commit in this repository")
    return result.stdout.strip()


def _normalize_allowed(entries: list[str]) -> list[str]:
    normalized = []
    for raw in entries:
        entry = raw.strip().replace("\\", "/")
        while entry.startswith("./"):
            entry = entry[2:]
        if not entry or entry.startswith("/") or ".." in entry.split("/"):
            raise MergeRefused(f"--allow {raw!r} must be a path relative to the repository root")
        normalized.append(entry.rstrip("/"))
    return normalized


def _is_allowed(path: str, allowed: list[str]) -> bool:
    return any(path == entry or path.startswith(entry + "/") for entry in allowed)


def _require_clean(repo: str) -> None:
    git_dir = _out(repo, "rev-parse", "--git-dir")
    in_progress = _git(repo, "rev-parse", "--verify", "--quiet", "MERGE_HEAD", check=False)
    if in_progress.returncode == 0:
        raise MergeRefused(f"a merge is already in progress in {repo} ({git_dir}/MERGE_HEAD); finish or abort it first")
    dirty = _out(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise MergeRefused("the worktree has uncommitted changes to tracked files; commit or stash them first:\n"
                           + dirty[:1000])
    from hermes_cli.worktree_archive import ArchiveError
    from hermes_cli.worktree_retention import hidden_index_edits
    try:
        hidden = hidden_index_edits(repo)
    except (ArchiveError, OSError) as exc:
        raise MergeRefused(f"could not check assume-unchanged/skip-worktree files: {exc}") from exc
    if hidden:
        raise MergeRefused("files marked assume-unchanged or skip-worktree have local edits that git status "
                           "does not show; commit them or clear the flag first: " + ", ".join(hidden[:20]))


def _changed_files(repo: str, base: str, tip: str) -> list[str]:
    # Plumbing diff-tree ignores diff.* porcelain config (renames, ignoreSubmodules, relative).
    return _paths(repo, "diff-tree", "-r", "-z", "--name-only", "--no-renames", base, tip)


def _untracked_collisions(repo: str, files: list[str]) -> list[str]:
    """Paths the branch writes that exist here untracked; git overwrites ignored ones without asking."""
    tracked = set(_paths(repo, "ls-files", "-z"))
    return [f for f in files if f not in tracked and os.path.lexists(os.path.join(repo, f))]


def _untested_commits(repo: str, tested: str, tip: str) -> list[str]:
    if _git(repo, "merge-base", "--is-ancestor", tested, tip, check=False).returncode != 0:
        raise MergeRefused(f"the tested commit {tested[:12]} is not on the branch being merged")
    log = _out(repo, "log", "--format=%h %s", f"{tested}..{tip}")
    return [line for line in log.splitlines() if line]


def _run_gate(repo: str, command: str, timeout: float) -> GateResult:
    """Run the gate in its own owned process group; output goes to a file, so a descendant that
    keeps a handle open cannot stall the wait, and a timeout kills the whole group."""
    from hermes_cli._subprocess_compat import kill_process_tree
    from hermes_cli.process_identity import release_process_group_once_empty, spawn_owned
    group_kw = {} if os.name == "posix" else {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    started = time.monotonic()
    with tempfile.TemporaryFile() as out:
        proc = spawn_owned(command, purpose="worktree merge gate", shell=True, cwd=repo,
                           stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **group_kw)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(proc)  # SIGKILL the live tree at once: a TERM grace lets a setsid'd child slip out
        finally:
            release_process_group_once_empty(proc.pid, proc=proc)
        out.seek(0, os.SEEK_END)
        out.seek(max(0, out.tell() - _OUTPUT_TAIL_CHARS))
        tail = out.read().decode("utf-8", "replace")
    return GateResult(command=command, exit_code=None if timed_out else proc.returncode, timed_out=timed_out,
                      duration_s=round(time.monotonic() - started, 2), output_tail=tail)


def _check_branch(repo: str, result: MergeResult, allowed: list[str], tested: Optional[str]) -> None:
    head = _resolve(repo, "HEAD", "HEAD")
    base = _out(repo, "merge-base", head, result.tip)
    result.files = _changed_files(repo, base, result.tip)
    if allowed:
        result.outside_allowed = [f for f in result.files if not _is_allowed(f, allowed)]
        if result.outside_allowed:
            raise MergeRefused(f"{len(result.outside_allowed)} changed file(s) are outside --allow")
    if tested:
        result.untested_commits = _untested_commits(repo, _resolve(repo, tested, "--tested"), result.tip)
        if result.untested_commits:
            raise MergeRefused(f"{len(result.untested_commits)} commit(s) landed after the tested commit")
    result.untracked_collisions = _untracked_collisions(repo, result.files)
    if result.untracked_collisions:
        raise MergeRefused(f"{len(result.untracked_collisions)} file(s) the branch writes exist here untracked "
                           "(possibly git-ignored); move them aside first")


def _merge(repo: str, result: MergeResult) -> bool:
    # Merge the commit that was checked, not the branch name, which can move in the meantime.
    merged = _git(repo, "merge", "--no-ff", "--no-edit", "-m", f"Merge branch '{result.branch}'", result.tip,
                  check=False)
    if merged.returncode == 0:
        result.merge_commit = _out(repo, "rev-parse", "HEAD")
        return True
    unmerged = _git(repo, "diff", "--name-only", "--diff-filter=U", check=False).stdout
    result.conflicts = [line for line in unmerged.splitlines() if line]
    _git(repo, "merge", "--abort", check=False)
    result.error = ("merge conflict; the merge was aborted and the worktree is unchanged. Resolve it by running "
                    f"`git merge {result.branch}` yourself" if result.conflicts
                    else f"git merge failed: {(merged.stderr or merged.stdout).strip()[:300]}")
    return False


def merge_branch(repo: str, branch: str, *, allowed: Optional[list[str]] = None, tested: Optional[str] = None,
                 gate: Optional[str] = None, gate_timeout: float = 1800) -> MergeResult:
    """Check and merge *branch* into *repo*'s HEAD; on any failure HEAD and the tree are as before."""
    result = MergeResult(branch=branch)
    try:
        allowed = _normalize_allowed(allowed or [])
        _require_clean(repo)
        result.tip = _resolve(repo, branch, "branch")
        if _git(repo, "merge-base", "--is-ancestor", result.tip, "HEAD", check=False).returncode == 0:
            result.already_merged = True
            return result
        _check_branch(repo, result, allowed, tested)
    except MergeRefused as exc:
        result.error = str(exc)
        return result
    before = _out(repo, "rev-parse", "HEAD")
    head_ref = _git(repo, "symbolic-ref", "-q", "HEAD", check=False).stdout.strip()
    if not _merge(repo, result):
        return result
    if gate:
        result.gate = _run_gate(repo, gate, gate_timeout)
        if result.gate.timed_out or result.gate.exit_code != 0:
            result.error = "the gate timed out" if result.gate.timed_out else "the gate failed"
            _undo_merge(repo, result, before, head_ref)
            return result
    result.merged = True
    return result


def _undo_merge(repo: str, result: MergeResult, before: str, head_ref: str) -> None:
    """Put the merged-into branch back at *before* without deleting anything the merge did not bring.

    Same checkout, HEAD still the merge commit: unstage whatever the gate staged (it stays on disk,
    untracked), then reset the tracked files; the merge brought no path that existed untracked
    (checked before merging). The gate checked out another branch: leave that checkout alone and move
    only the original branch ref, and only while it still holds the merge. Anything else (the gate
    committed) is reported and left as it is."""
    now_ref = _git(repo, "symbolic-ref", "-q", "HEAD", check=False).stdout.strip()
    if now_ref == head_ref and _out(repo, "rev-parse", "HEAD") == result.merge_commit:
        _git(repo, "reset", "-q")
        _git(repo, "reset", "-q", "--hard", before)
    elif now_ref == head_ref or not head_ref or \
            _git(repo, "update-ref", head_ref, before, result.merge_commit, check=False).returncode:
        result.error += (f"; the gate moved HEAD itself, so nothing was reset: merge commit "
                         f"{result.merge_commit[:12]} is still on {head_ref or 'the detached HEAD it was made on'}")
        return
    result.reset_to, result.merge_commit = before, ""


def _print_human(result: MergeResult) -> None:
    if result.already_merged:
        print(f"✓ {result.branch} is already merged; nothing to do.")
        return
    if result.merged:
        print(f"✓ Merged {result.branch} ({result.tip[:12]}) as {result.merge_commit[:12]}: {len(result.files)} file(s)")
    else:
        print(f"✗ Not merged: {result.error}")
    for label, items in (("Outside --allow", result.outside_allowed),
                         ("Commits after the tested commit", result.untested_commits),
                         ("Untracked files the branch would overwrite", result.untracked_collisions),
                         ("Conflicting files", result.conflicts)):
        if items:
            print(f"  {label}:")
            for item in items[:50]:
                print(f"    {item}")
    if result.gate:
        state = "timed out" if result.gate.timed_out else f"exit {result.gate.exit_code}"
        print(f"  Gate `{result.gate.command}`: {state} after {result.gate.duration_s}s")
        if not result.merged and result.gate.output_tail:
            print("  Gate output (tail):\n" + result.gate.output_tail)
    if result.reset_to:
        print(f"  Worktree reset to {result.reset_to[:12]}.")


def cmd_merge(_worktree_gc, repo_root: str, args) -> int:
    """``hermes worktree merge``: exit 0 when merged or already merged, 1 otherwise."""
    branch = getattr(args, "branch", None)
    if not branch:
        print("Usage: hermes worktree merge <branch> [--allow PATH ...] [--tested SHA] [--gate CMD]")
        return 1
    timeout = getattr(args, "gate_timeout", None) or 1800
    if timeout <= 0:
        print("--gate-timeout must be a positive number of seconds.")
        return 1
    result = merge_branch(repo_root, branch, allowed=getattr(args, "allow", None) or [],
                          tested=getattr(args, "tested", None), gate=getattr(args, "gate", None),
                          gate_timeout=timeout)
    if getattr(args, "json", False):
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)
    return 0 if (result.merged or result.already_merged) else 1
