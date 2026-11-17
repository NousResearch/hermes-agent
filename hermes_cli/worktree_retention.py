"""What a worktree holds that neither git nor an archive can bring back.

Every path that deletes a tree (the sweep, the classic clean-tree pruner, exit-time Archive,
subagent finalization, the attended ``hermes worktree prune``) asks :func:`removal_blocker`
first; anything it names keeps the tree. The rule: a tree may only be removed when everything
that is not reproducible is in git or in the archive. Not reproducible:

* an embedded repository anywhere below the tree, ignored or not (``build/tool/.git``, a nested
  worktree): its private commits live only on disk (``keep-unarchivable``);
* a commit held only by a per-worktree ref (``refs/worktree/*``, ``refs/bisect/*``,
  ``refs/rewritten/*``), which dies with the tree's admin dir (``keep-worktree-refs``);
* an untracked file the snapshot skipped for size (``keep-large-untracked``).

Ignored files (``.env`` copies, caches, build and test output) are disposable: they are not
archived and are deleted with the tree. Only an embedded repository inside an ignored directory
keeps it, because its commits may exist nowhere else.

:func:`hidden_index_edits` finds skip-worktree / assume-unchanged files whose bytes differ from
the index: ``git status`` never reports them, so dirty checks must add them.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Dependency stores and caches: never descended, not even to look for an embedded repository.
_OPAQUE_DIRS = frozenset({
    "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".next", ".turbo", ".cache", ".gradle",
})
_PER_WORKTREE_REFS = ("refs/worktree/", "refs/bisect/", "refs/rewritten/")
_MAX_ENTRIES = 200_000   # per tree; past this an embedded repository cannot be ruled out -> keep
_NAMED = 5               # offenders named in a reason

KEEP_UNARCHIVABLE = "keep-unarchivable"
KEEP_WORKTREE_REFS = "keep-worktree-refs"
KEEP_LARGE_UNTRACKED = "keep-large-untracked"
KEEP_SUBMODULE_WORK = "keep-submodule-work"


def _env(wt: Path) -> dict:
    from hermes_cli.worktree_archive import _env as archive_env
    return {**archive_env(wt), "GIT_OPTIONAL_LOCKS": "0"}


def _git(args, wt: Path, env, timeout: float = 120) -> str:
    from hermes_cli.worktree_archive import _ok
    return _ok(args, wt, env, timeout=timeout)


@dataclass
class _Scan:
    embedded: list = field(default_factory=list)
    entries: int = 0
    truncated: bool = False


def _walk(scan: _Scan, wt: Path, rel_dir: str) -> None:
    """Below ignored *rel_dir*: record nested ``.git`` entries. Symlinks are not followed."""
    stack = [rel_dir]
    while stack and not scan.truncated:
        rel = stack.pop()
        try:
            entries = list(os.scandir(wt / rel))
        except OSError:
            continue
        for entry in entries:
            scan.entries += 1
            if scan.entries > _MAX_ENTRIES:
                scan.truncated = True
                return
            child = f"{rel}/{entry.name}"
            if entry.name == ".git":
                scan.embedded.append(rel)
                continue
            if entry.is_dir(follow_symlinks=False) and entry.name not in _OPAQUE_DIRS:
                stack.append(child)


def _ignored_entries(wt: Path, env) -> list[str]:
    """Ignored paths, whole ignored directories collapsed to ``dir/``. ``ls-files --directory``
    hits an internal error on some trees (git 2.53: "directory entry not superset of prefix");
    ``status --ignored`` reports the same set."""
    from hermes_cli.worktree_archive import ArchiveError
    try:
        out = _git(["ls-files", "--others", "--ignored", "--exclude-standard", "--directory",
                    "--no-empty-directory", "-z"], wt, env)
        return [p for p in out.split("\0") if p]
    except ArchiveError:
        out = _git(["status", "--porcelain", "-z", "--ignored=traditional", "--untracked-files=normal",
                    "--no-renames"], wt, env)
        return [e[3:] for e in out.split("\0") if e.startswith("!! ")]


def scan_tree(wt_path) -> _Scan:
    """Embedded repositories of *wt_path*, ignored or not. Read-only.

    Untracked non-ignored repositories are reported by ``ls-files`` itself (``dir/``); ignored
    ones hide inside collapsed ignored directories, so those are walked (bounded; dependency
    stores are never descended). Raises ``ArchiveError`` when git fails.
    """
    wt = Path(wt_path)
    env = _env(wt)
    scan = _Scan()
    # Without --directory git descends untracked dirs and lists each embedded repository as "dir/".
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z"], wt, env)
    scan.embedded += [p.rstrip("/") for p in untracked.split("\0") if p.endswith("/")]
    for rel in _ignored_entries(wt, env):
        rel_clean = rel.rstrip("/")
        path = wt / rel_clean
        if not rel.endswith("/") or path.is_symlink() or rel_clean.rsplit("/", 1)[-1] in _OPAQUE_DIRS:
            continue
        if (path / ".git").exists():
            scan.embedded.append(rel_clean)
            continue
        _walk(scan, wt, rel_clean)
        if scan.truncated:
            break
    return scan


def hidden_index_edits(wt_path, env=None) -> list[str]:
    """Skip-worktree / assume-unchanged paths whose disk state differs from the index.

    ``git status`` trusts both flags and reports nothing. A skip-worktree file missing from disk
    is a sparse checkout, not an edit; a missing assume-unchanged file is a deletion. Raises
    ``ArchiveError`` when git fails.
    """
    wt = Path(wt_path)
    env = env or _env(wt)
    flagged = []
    for record in _git(["ls-files", "-v", "-s", "-z"], wt, env).split("\0"):
        meta, _, rel = record.partition("\t")
        parts = meta.split()
        if not rel or len(parts) != 4 or not (parts[0] == "S" or parts[0].islower()):
            continue
        path = wt / rel
        if not os.path.lexists(path):
            if parts[0] != "S":
                flagged.append(rel)  # assume-unchanged file deleted
            continue
        flagged.append((rel, parts[1], parts[2]))
    edits = [f for f in flagged if isinstance(f, str)]
    for rel, mode, sha in (f for f in flagged if not isinstance(f, str)):
        path = wt / rel
        if mode == "120000":
            data = os.readlink(path).encode("utf-8", "surrogateescape") if path.is_symlink() else None
        else:
            data = None if path.is_symlink() or not path.is_file() else path.read_bytes()
        if data is None:
            edits.append(rel)  # type changed
            continue
        if _hash(wt, env, data, rel, raw=True) != sha and _hash(wt, env, data, rel, raw=False) != sha:
            edits.append(rel)
    return edits


def _hash(wt: Path, env, data: bytes, rel: str, *, raw: bool) -> str:
    args = ["hash-object", "--stdin"] + (["--no-filters"] if raw else [f"--path={rel}"])
    result = subprocess.run(["git", *args], cwd=str(wt), env=env, input=data, capture_output=True,
                            timeout=60, check=False)
    return result.stdout.decode("ascii", "replace").strip() if result.returncode == 0 else ""


def private_worktree_refs(wt_path, env=None) -> list[str]:
    """Per-worktree refs of *wt_path* holding commits no branch, tag, remote or archive reaches.

    They live in the tree's admin dir and vanish with it. Raises ``ArchiveError`` when git fails.
    """
    wt = Path(wt_path)
    env = env or _env(wt)
    out = _git(["for-each-ref", "--format=%(refname)%00%(objectname)", *_PER_WORKTREE_REFS], wt, env)
    private = []
    for line in out.splitlines():
        name, _, sha = line.partition("\0")
        if sha and _git(["rev-list", "--max-count=1", sha, "--not", "--branches", "--tags", "--remotes",
                         "--glob=refs/hermes/archive/*"], wt, env).strip():
            private.append(name)
    return private


def _named(paths: list, more: bool = False) -> str:
    extra = len(paths) - _NAMED
    tail = f" (+{extra} more)" if extra > 0 else (" (+more)" if more else "")
    return ", ".join(paths[:_NAMED]) + tail


def removal_blocker(wt_path, *, skipped: Optional[list] = None, gitlinks: bool = False
                    ) -> Optional[tuple[str, str]]:
    """``(keep-action, reason)`` when removing *wt_path* would lose something, else None.

    *skipped* is the snapshot's ``skipped_untracked`` list (or the oversize files a preflight
    would skip). *gitlinks* refuses any submodule (archive paths); without it only a checked-out
    or retained module repository blocks. Any failure to verify blocks. Read-only.
    """
    from hermes_cli.worktree_archive import ArchiveError, submodule_work
    try:
        why = submodule_work(wt_path, gitlinks=gitlinks)
        if why:
            return KEEP_SUBMODULE_WORK, why
        scan = scan_tree(wt_path)
        if scan.embedded:
            return KEEP_UNARCHIVABLE, f"embedded repository {scan.embedded[0]} cannot be archived"
        if scan.truncated:
            return KEEP_UNARCHIVABLE, f"over {_MAX_ENTRIES} ignored entries; cannot rule out an embedded repository"
        refs = private_worktree_refs(wt_path)
        if refs:
            return KEEP_WORKTREE_REFS, f"per-worktree ref {_named(refs)} holds commits no branch has"
        if skipped:
            return KEEP_LARGE_UNTRACKED, f"untracked file(s) over the archive size cap: {_named([s['path'] for s in skipped])}"
    except (ArchiveError, OSError, subprocess.SubprocessError) as exc:
        return KEEP_UNARCHIVABLE, f"could not verify the tree is safe to remove: {exc}"
    return None
