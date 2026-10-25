"""Restorable snapshots of git worktrees: ``refs/hermes/archive/<tree>``.

A snapshot is two commits built with a TEMPORARY index (the tree's real index and working files
are only read): ``I`` = the staged state (tree of a copy of the real index, parent ``HEAD``) and
``W`` = the whole working state (tracked + non-ignored untracked files, parents ``HEAD`` and
``I``), the same shape ``git stash -u`` produces. ``W`` carries the metadata (original path,
branch, HEAD, reason, skipped files) as JSON in its message, and the ref keeps every object
reachable, so the tree and even its branch can be deleted without losing work.

Bytes are stored raw: every tracked/untracked regular file is re-hashed with
``hash-object --no-filters`` and restored by writing the blob bytes directly, so no attribute
source (``.gitattributes``, ``$GIT_COMMON_DIR/info/attributes``, ``core.attributesFile``) and no
clean/smudge/eol conversion can change them; a restore writes back exactly the archived bytes. Untracked files over a size cap are skipped and listed
(so ``.git`` never balloons with build artefacts the ignore rules missed), and a tree with a
skipped file is never removed; an untracked embedded repository cannot be captured at all and
fails the snapshot, which keeps the tree. Ignored files are not captured and are deleted with the
tree; an embedded repository inside an ignored directory keeps it (``worktree_retention``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ARCHIVE_PREFIX = "refs/hermes/archive/"
_SUBJECT = "hermes worktree archive: "
_IDENTITY = {
    "GIT_AUTHOR_NAME": "Hermes worktree archive", "GIT_AUTHOR_EMAIL": "hermes-archive@localhost",
    "GIT_COMMITTER_NAME": "Hermes worktree archive", "GIT_COMMITTER_EMAIL": "hermes-archive@localhost",
}
# Raw bytes in, raw bytes out: no line-ending conversion on either side of the round trip.
_RAW_CONFIG = (("core.autocrlf", "false"), ("core.safecrlf", "false"), ("core.eol", "lf"))


class ArchiveError(RuntimeError):
    """The snapshot (or restore) could not be completed; callers must keep the tree."""


@dataclass
class Archive:
    ref: str
    name: str
    sha: str
    created: float
    meta: dict = field(default_factory=dict)
    staged: str = ""  # staged_state() of exactly the index this archive captured


def _env(cwd, index: Optional[str] = None) -> dict:
    from hermes_cli._subprocess_compat import noninteractive_repo_git_env
    env = noninteractive_repo_git_env(cwd)
    if env is None:
        raise ArchiveError("git filter discovery failed")
    env.update(_IDENTITY)
    start = int(env["GIT_CONFIG_COUNT"])
    for offset, (key, value) in enumerate(_RAW_CONFIG):
        env[f"GIT_CONFIG_KEY_{start + offset}"] = key
        env[f"GIT_CONFIG_VALUE_{start + offset}"] = value
    env["GIT_CONFIG_COUNT"] = str(start + len(_RAW_CONFIG))
    if index:
        env["GIT_INDEX_FILE"] = index
    return env


def _run(args, cwd, env, timeout: float = 300, stdin_text: Optional[str] = None):
    return subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True,  # windows-footgun: ok -- encoding passed on the next line
                          encoding="utf-8", errors="surrogateescape", timeout=timeout,
                          input=stdin_text, stdin=None if stdin_text is not None else subprocess.DEVNULL,
                          check=False)


def _ok(args, cwd, env, **kw) -> str:
    result = _run(args, cwd, env, **kw)
    if result.returncode != 0:
        raise ArchiveError(f"git {args[0]} failed: {result.stderr.strip()[-300:]}")
    return result.stdout


def _raw_attrs(args, cwd, env, empty_tree: str, **kw) -> str:
    """Run a content-converting command with attributes read from the empty tree (git >= 2.40);
    older git lacks ``--attr-source`` and falls back to the checkout's own attributes."""
    result = _run(["--attr-source", empty_tree, *args], cwd, env, **kw)
    if result.returncode != 0 and "attr-source" in result.stderr:
        result = _run(args, cwd, env, **kw)
    if result.returncode != 0:
        raise ArchiveError(f"git {args[0]} failed: {result.stderr.strip()[-300:]}")
    return result.stdout


def _run_bytes(args, cwd, env, data: bytes, timeout: float = 300) -> bytes:
    result = subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, input=data,
                            timeout=timeout, check=False)
    if result.returncode != 0:
        raise ArchiveError(f"git {args[0]} failed: {result.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return result.stdout


def _regular_entries(cwd, env) -> list[tuple[str, str, str]]:
    """``(mode, sha, path)`` for every regular-file entry of the index ``env`` points at."""
    out = _ok(["ls-files", "-s", "-z"], cwd, env)
    entries = []
    for record in out.split("\0"):
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if path and len(parts) == 3 and parts[0] in ("100644", "100755"):
            entries.append((parts[0], parts[1], path))
    return entries


def _rehash_raw(wt: Path, tenv) -> None:
    """Replace every regular-file blob in the temporary index with the file's exact bytes.

    ``add`` runs content conversion from every attribute source, and ``--attr-source`` only
    replaces ``.gitattributes``: ``info/attributes`` and ``core.attributesFile`` still apply.
    ``hash-object --no-filters`` skips all of it.
    """
    entries = [e for e in _regular_entries(wt, tenv) if os.path.isfile(wt / e[2]) and not os.path.islink(wt / e[2])]
    if not entries:
        return
    shas: list[str] = []
    # --stdin-paths reads one path per line and C-unquotes a leading '"': hash those one by one.
    batch = [p for _m, _s, p in entries if "\n" not in p and not p.startswith('"')]
    if batch:
        data = "".join(p + "\n" for p in batch).encode("utf-8", "surrogateescape")
        shas = _run_bytes(["hash-object", "-w", "--no-filters", "--stdin-paths"], wt, tenv, data).decode().split()
    by_path = dict(zip(batch, shas))
    for _m, _s, path in entries:
        if path not in by_path:
            raw = (wt / path).read_bytes()
            by_path[path] = _run_bytes(["hash-object", "-w", "--no-filters", "--stdin"], wt, tenv, raw).decode().strip()
    info = "".join(f"{mode} {by_path[path]}\t{path}\0" for mode, _s, path in entries)
    _run_bytes(["update-index", "-z", "--index-info"], wt, tenv, info.encode("utf-8", "surrogateescape"))


def _write_raw(dest: Path, env) -> None:
    """Overwrite every regular file of the index with its blob's exact bytes (no smudge/eol)."""
    entries = _regular_entries(dest, env)
    if not entries:
        return
    out = _run_bytes(["cat-file", "--batch"], dest, env, "".join(f"{s}\n" for _m, s, _p in entries).encode())
    pos = 0
    for _mode, _sha, path in entries:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        body = out[header_end + 1:header_end + 1 + size]
        pos = header_end + 1 + size + 1
        target = dest / path
        if target.is_symlink():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = target.stat().st_mode if target.exists() else None
        target.write_bytes(body)
        if mode is not None:
            os.chmod(target, mode)


def ref_safe(name: str) -> str:
    """A tree name as a single valid ref component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "tree"
    return safe.replace("..", "-").removesuffix(".lock")


def _untracked(wt: Path, env) -> list[str]:
    out = _ok(["ls-files", "--others", "--exclude-standard", "-z"], wt, env)
    return [p for p in out.split("\0") if p]


SUBMODULE_REASON = "submodule "


def submodule_work(wt_path, *, gitlinks: bool = True) -> Optional[str]:
    """Why a submodule keeps *wt_path*, or None. Read-only.

    An archive records a submodule only as its ``160000`` gitlink, and removing the tree deletes
    its per-worktree ``modules/`` repositories (kept by ``git submodule deinit``, private branches
    included). Until submodules are archived recursively: no tree with a gitlink is archived
    (*gitlinks*), and no tree whose ``modules/`` holds a repository or whose gitlink is checked out
    is removed at all.
    """
    wt = Path(wt_path)
    env = {**_env(wt), "GIT_OPTIONAL_LOCKS": "0"}
    modules = Path(_ok(["rev-parse", "--absolute-git-dir"], wt, env).strip()) / "modules"
    if modules.is_dir() and any(modules.iterdir()):
        return f"{SUBMODULE_REASON}repository in {modules} would be deleted with the tree"
    staged = _ok(["ls-files", "--stage", "-z"], wt, env)
    links = [e.split("\t", 1)[1] for e in staged.split("\0") if e.startswith("160000 ") and "\t" in e]
    kept = [p for p in links if gitlinks or (wt / p / ".git").exists()]
    return f"{SUBMODULE_REASON}{kept[0]} cannot be archived (gitlink only)" if kept else None


def staged_state(wt_path) -> str:
    """Digest of the semantic staged entries (mode, blob id, stage, path) of *wt_path*.

    A stat-only index refresh (``git status``) leaves it unchanged; any ``git add``/``apply
    --cached``/``rm --cached`` changes it. Raises :class:`ArchiveError` when git fails.
    """
    wt = Path(wt_path)
    return _digest_index(wt, _env(wt))


def _digest_index(wt: Path, env) -> str:
    import hashlib
    env = {**env, "GIT_OPTIONAL_LOCKS": "0"}
    return hashlib.sha256(_ok(["ls-files", "--stage", "-z"], wt, env).encode("utf-8", "surrogateescape")).hexdigest()


def archive_preflight(wt_path, max_untracked_mb: float = 50) -> Optional[tuple[str, str]]:
    """``(keep-action, reason)`` when archiving and removing *wt_path* would lose something
    (``worktree_retention.removal_blocker``: submodule, embedded repository, per-worktree ref,
    an untracked file over the size cap), else None. Read-only."""
    from hermes_cli.worktree_retention import KEEP_UNARCHIVABLE, removal_blocker
    try:
        wt = Path(wt_path)
        skipped = _split_oversized(wt, _untracked(wt, _env(wt)), max_untracked_mb)
    except (ArchiveError, OSError, subprocess.SubprocessError) as exc:
        return KEEP_UNARCHIVABLE, str(exc)
    return removal_blocker(wt, skipped=skipped, gitlinks=True)


def _split_oversized(wt: Path, untracked: list[str], max_mb: float) -> list[dict]:
    if max_mb <= 0:
        return []
    cap = max_mb * 1024 * 1024
    skipped = []
    for rel in untracked:
        try:
            size = os.lstat(wt / rel).st_size
        except OSError:
            continue
        if size > cap:
            skipped.append({"path": rel, "bytes": size})
    return skipped


def _create_ref(repo_root: str, name: str, sha: str, env) -> str:
    base = ARCHIVE_PREFIX + ref_safe(name)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for ref in (base, f"{base}--{stamp}", *(f"{base}--{stamp}-{i}" for i in range(2, 50))):
        # Empty old value = create only: two racing sweeps never overwrite each other's archive.
        if _run(["update-ref", "-m", "hermes worktree archive", ref, sha, ""], repo_root, env).returncode == 0:
            return ref
    raise ArchiveError(f"could not create an archive ref for {name}")


def snapshot_worktree(repo_root: str, wt_path, *, reason: str, max_untracked_mb: float = 50) -> Archive:
    """Archive the full working state of *wt_path*; raises :class:`ArchiveError` on any failure.

    Never writes the tree's real index or working files (temporary ``GIT_INDEX_FILE`` only).
    """
    wt = Path(wt_path)
    env = _env(wt)
    # Shared refusal point (sweep, exit-time Archive, subagent finalization).
    unsafe = submodule_work(wt)
    if unsafe:
        raise ArchiveError(unsafe)
    head = _ok(["rev-parse", "--verify", "HEAD^{commit}"], wt, env).strip()
    branch = _run(["symbolic-ref", "--short", "-q", "HEAD"], wt, env).stdout.strip() or None
    git_dir = Path(_ok(["rev-parse", "--absolute-git-dir"], wt, env).strip())
    untracked = _untracked(wt, env)
    embedded = [p for p in untracked if p.endswith("/")]
    if embedded:
        raise ArchiveError(f"untracked embedded repository {embedded[0].rstrip('/')} cannot be archived")
    skipped = _split_oversized(wt, untracked, max_untracked_mb)

    token = f"hermes-archive-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp_index, pathspec = git_dir / f"{token}.index", git_dir / f"{token}.pathspec"
    try:
        tenv = _env(wt, index=str(tmp_index))
        if (git_dir / "index").is_file():
            shutil.copyfile(git_dir / "index", tmp_index)
        else:
            _ok(["read-tree", head], wt, tenv)
        staged = _digest_index(wt, tenv)
        # Staged state first (fails on an unmerged index: a conflicted tree is never archived).
        index_tree = _ok(["write-tree"], wt, tenv).strip()
        empty_tree = _ok(["mktree"], wt, env, stdin_text="").strip()
        specs = [":(top)"] + [f":(top,exclude,literal){s['path']}" for s in skipped]
        pathspec.write_bytes(("\0".join(specs) + "\0").encode("utf-8", "surrogateescape"))
        _raw_attrs(["add", "-A", "--pathspec-from-file", str(pathspec), "--pathspec-file-nul"],
                   wt, tenv, empty_tree)
        # Every blob from the exact bytes on disk (add converted them per attributes).
        _rehash_raw(wt, tenv)
        work_tree = _ok(["write-tree"], wt, tenv).strip()
        index_commit = _ok(["commit-tree", index_tree, "-p", head, "-m", f"index of {wt.name}"], wt, env).strip()
        meta = {"version": 1, "name": wt.name, "path": str(wt), "repo_root": str(repo_root),
                "branch": branch, "head": head, "index_commit": index_commit,
                "created": int(time.time()), "reason": reason, "skipped_untracked": skipped}
        message = f"{_SUBJECT}{wt.name}\n\n{json.dumps(meta, indent=1, ensure_ascii=False)}\n"
        sha = _ok(["commit-tree", work_tree, "-p", head, "-p", index_commit], wt, env,
                  stdin_text=message).strip()
    finally:
        for leftover in (tmp_index, tmp_index.with_name(tmp_index.name + ".lock"), pathspec):
            leftover.unlink(missing_ok=True)
    ref = _create_ref(repo_root, wt.name, sha, env)
    logger.info("Archived worktree %s -> %s (%s)", wt, ref, reason)
    return Archive(ref=ref, name=ref[len(ARCHIVE_PREFIX):], sha=sha, created=meta["created"], meta=meta,
                   staged=staged)


def _parse_meta(body: str) -> dict:
    start = body.find("{")
    try:
        meta = json.loads(body[start:]) if start >= 0 else {}
    except ValueError:
        meta = {}
    return meta if isinstance(meta, dict) else {}


def list_archives(repo_root: str) -> list[Archive]:
    """Every archive in *repo_root*, newest first. Read-only; [] when git fails."""
    try:
        env = _env(repo_root)
        out = _ok(["for-each-ref", "--sort=-creatordate",
                   "--format=%(refname)%00%(objectname)%00%(creatordate:unix)%00%(contents:body)%00",
                   ARCHIVE_PREFIX], repo_root, env, timeout=60)
    except (ArchiveError, OSError, subprocess.SubprocessError):
        return []
    archives = []
    for record in out.split("\0\n"):
        parts = record.split("\0")
        if len(parts) < 4 or not parts[0].startswith(ARCHIVE_PREFIX):
            continue
        meta = _parse_meta(parts[3])
        created = float(meta.get("created") or parts[2] or 0)
        archives.append(Archive(ref=parts[0], name=parts[0][len(ARCHIVE_PREFIX):], sha=parts[1],
                                created=created, meta=meta))
    return archives


def find_archive(repo_root: str, name: str) -> Optional[Archive]:
    """Exact archive name/ref first, else the newest archive of the tree called *name*."""
    archives = list_archives(repo_root)
    wanted = name.removeprefix(ARCHIVE_PREFIX)
    for arc in archives:
        if arc.name == wanted:
            return arc
    return next((a for a in archives if a.meta.get("name") == name or a.name == ref_safe(name)), None)


def prune_expired_archives(repo_root: str, retention_days: float, *, now: Optional[float] = None,
                           dry_run: bool = False) -> list[str]:
    """Delete archive refs older than *retention_days* (0 keeps them forever)."""
    if not retention_days or retention_days <= 0:
        return []
    cutoff = (now or time.time()) - retention_days * 86400
    expired = [a for a in list_archives(repo_root) if a.created and a.created < cutoff]
    if dry_run or not expired:
        return [a.ref for a in expired]
    env = _env(repo_root)
    deleted = []
    for arc in expired:
        if _run(["update-ref", "-d", arc.ref, arc.sha], repo_root, env).returncode == 0:
            deleted.append(arc.ref)
    return deleted


def _branch_tip(repo_root: str, branch: str, env) -> Optional[str]:
    result = _run(["rev-parse", "--verify", "-q", f"refs/heads/{branch}^{{commit}}"], repo_root, env)
    return result.stdout.strip() if result.returncode == 0 else None


def _checked_out_branches(repo_root: str, env) -> set:
    out = _run(["worktree", "list", "--porcelain"], repo_root, env).stdout
    return {line[len("branch refs/heads/"):] for line in out.splitlines() if line.startswith("branch refs/heads/")}


def _add_args(repo_root: str, dest: Path, branch: Optional[str], head: str, env) -> tuple[list, bool]:
    """``worktree add`` argv restoring *branch* when that is safe -> (argv, detached)."""
    base = ["worktree", "add", "--no-checkout"]
    if branch:
        tip = _branch_tip(repo_root, branch, env)
        if tip is None:
            return [*base, "-b", branch, str(dest), head], False
        if tip == head and branch not in _checked_out_branches(repo_root, env):
            return [*base, str(dest), branch], False
    return [*base, "--detach", str(dest), head], True


def restore_archive(repo_root: str, name: str, target=None) -> dict:
    """Recreate the archived tree at its original path (or *target*); raises ArchiveError."""
    arc = find_archive(repo_root, name)
    if arc is None:
        raise ArchiveError(f"no archive named {name}")
    meta = arc.meta
    head = meta.get("head") or f"{arc.sha}^1"
    index_commit = meta.get("index_commit") or f"{arc.sha}^2"
    dest = Path(target or meta.get("path") or Path(repo_root) / ".worktrees" / arc.name).absolute()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ArchiveError(f"{dest} already exists; pass a different path")
    env = _env(repo_root)
    args, detached = _add_args(repo_root, dest, meta.get("branch"), head, env)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ok(args, repo_root, env)
    wenv = _env(dest)
    empty_tree = _ok(["mktree"], dest, wenv, stdin_text="").strip()
    _ok(["read-tree", f"{arc.sha}^{{tree}}"], dest, wenv)
    _raw_attrs(["checkout-index", "-a", "-f"], dest, wenv, empty_tree)
    _write_raw(dest, wenv)  # checkout-index still converted per info/attributes
    _ok(["read-tree", f"{index_commit}^{{tree}}"], dest, wenv)
    _run(["update-index", "-q", "--refresh"], dest, wenv)
    from hermes_cli.worktree_registry import record_worktree_repo
    record_worktree_repo(repo_root)
    return {"path": str(dest), "branch": meta.get("branch"), "detached": detached, "head": head,
            "ref": arc.ref, "skipped_untracked": meta.get("skipped_untracked") or []}
