"""Shared-tree (skills-shared/) curator safety contract.

The git-backed shared skills tree is mutated by the curator ONLY under the
hard gate implemented here (see the curator-scope-shared-skills spec,
Invariant 3):

- an fcntl ``.curator.lock`` held for the whole shared pass. The lock is
  ADVISORY and serializes CURATOR-vs-CURATOR only — an arbitrary sibling
  writer (another agent's ``skill_manage``, a human, the sync habit) never
  acquires it. The anti-clobber primitive against a non-cooperating
  concurrent writer is the EXPLICIT PATHSPEC commit + pre-commit porcelain
  drift-abort in :func:`commit_shared`.
- a clean ``git status --porcelain -- skills-shared/`` precheck; a dirty
  tree skips the shared pass (with crash recovery for self-inflicted dirt).
- a pre-mutation snapshot: baseline git rev + a separate
  ``shared-<ts>.tar.gz`` of the in-scope dirs + a manifest recording the
  INTENDED-WRITE file set (written BEFORE mutation — crash recovery keys
  on it).
- every mutation lands as one ``curator:`` commit staged from an explicit
  pathspec — NEVER ``git add skills-shared/`` (a wildcard stage would
  absorb a concurrent sibling file into the commit and entangle its
  revert).

fcntl semantics honesty: the kernel auto-releases an fcntl lock when its
holder dies, so a dead owner never leaves a HELD lock. "Stale" therefore
means: the lock ACQUIRES cleanly yet the shared tree is dirty. The PID
recorded in the lockfile is diagnostic only.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tarfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None

logger = logging.getLogger(__name__)

LOCK_NAME = ".curator.lock"
SHARED_MANIFEST_NAME = "shared-manifest.json"


def _shared_root() -> Path:
    from agent.skill_utils import get_shared_skills_root

    return get_shared_skills_root()


def _git_toplevel(shared_root: Path) -> Optional[Path]:
    """The git working tree that tracks skills-shared/, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(shared_root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    top = out.stdout.strip()
    return Path(top) if top else None


def _git(repo: Path, *args: str, timeout: int = 30) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.SubprocessError) as e:
        return 1, "", str(e)


def _porcelain_shared(repo: Path, shared_root: Path) -> Optional[List[str]]:
    """Dirty paths under skills-shared/ (repo-relative), or None on error."""
    try:
        rel = shared_root.resolve().relative_to(repo.resolve())
    except ValueError:
        return None
    code, out, _err = _git(
        repo, "status", "--porcelain", "--untracked-files=all", "--", str(rel)
    )
    if code != 0:
        return None
    lines = [ln for ln in out.splitlines() if ln.strip()]
    paths = []
    for ln in lines:
        # porcelain v1: XY <path> (or XY <old> -> <new> for renames)
        payload = ln[3:]
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        payload = payload.strip().strip('"')
        # The curator's own lockfile lives inside the shared tree; it is
        # infrastructure, not content — never treated as dirt (and never
        # committed: commit_shared only stages explicit content paths).
        if payload.rsplit("/", 1)[-1] == LOCK_NAME:
            continue
        paths.append(payload)
    return paths


@contextmanager
def shared_pass_lock(shared_root: Optional[Path] = None):
    """Non-blocking fcntl lock over the shared tree for the whole pass.

    Yields True when acquired, False on contention (another curator holds
    it — the shared pass must be skipped, never blocked behind it). Records
    owner PID + start time in the lockfile (diagnostic only).
    
    P2 FIX: Fail CLOSED when fcntl unavailable (Windows, some BSD). Shared-tree
    curation requires advisory locking to serialize curator-vs-curator; platforms
    without fcntl simply skip the shared pass.
    """
    root = shared_root or _shared_root()
    lock_path = root / LOCK_NAME
    if fcntl is None:
        logger.warning(
            "fcntl unavailable (platform does not support advisory locking); "
            "shared-tree curation skipped — cannot serialize curator-vs-curator"
        )
        yield False
        return
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = open(lock_path, "a+", encoding="utf-8")
    except OSError as e:
        logger.debug("shared lockfile open failed: %s", e)
        yield False
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(json.dumps({
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }))
            fd.flush()
        except OSError:
            pass
        yield True
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        fd.close()


def git_precheck_shared(
    shared_root: Optional[Path] = None,
) -> Tuple[bool, str, List[str]]:
    """Clean-tree gate. Returns (ok, reason, dirty_paths)."""
    root = shared_root or _shared_root()
    if not root.is_dir():
        return False, "no skills-shared/ tree", []
    repo = _git_toplevel(root)
    if repo is None:
        return False, "skills-shared/ is not inside a git repo", []
    dirty = _porcelain_shared(repo, root)
    if dirty is None:
        return False, "git status failed", []
    if dirty:
        return False, "dirty working tree", dirty
    return True, "clean", []


def snapshot_shared(
    dirs: Iterable[Path],
    intended_writes: Iterable[str],
    *,
    shared_root: Optional[Path] = None,
    dest_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Pre-mutation shared snapshot. Returns the snapshot dir or None.

    Writes ``shared-<ts>.tar.gz`` of the given in-scope dirs (a SEPARATE
    tarball — never appended into the agent snapshot's gzip stream) plus a
    ``shared-manifest.json`` recording the baseline git rev, the tar path,
    and the INTENDED-WRITE file set (repo-relative). The manifest is written
    BEFORE any mutation; crash recovery keys its exact-set match on it.

    Failure returns None — the caller MUST hard-gate: no shared mutation
    without a successful snapshot (unlike the agent tree's log-and-continue).
    """
    root = shared_root or _shared_root()
    repo = _git_toplevel(root)
    if repo is None:
        return None
    code, head, _ = _git(repo, "rev-parse", "HEAD")
    if code != 0:
        return None
    if dest_dir is None:
        from agent.curator_backup import _backups_dir, _utc_id

        dest_dir = _backups_dir() / f"shared-{_utc_id()}"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        tar_path = dest_dir / f"shared-{ts}.tar.gz"
        with tarfile.open(tar_path, "w:gz", compresslevel=6) as tf:
            for d in dirs:
                d = Path(d)
                if d.exists():
                    tf.add(str(d), arcname=d.name, recursive=True)
        manifest = {
            "baseline_rev": head.strip(),
            "tar": tar_path.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "intended_writes": sorted(set(str(p) for p in intended_writes)),
            "pid": os.getpid(),
        }
        (dest_dir / SHARED_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return dest_dir
    except (OSError, tarfile.TarError) as e:
        logger.debug("shared snapshot failed: %s", e, exc_info=True)
        return None


def latest_shared_snapshot() -> Optional[Path]:
    """Most recent shared snapshot dir, or None."""
    from agent.curator_backup import _backups_dir

    base = _backups_dir()
    if not base.is_dir():
        return None
    candidates = sorted(
        (p for p in base.iterdir()
         if p.is_dir() and p.name.startswith("shared-")),
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def commit_shared(
    summary: str,
    written_files: List[Path],
    precheck_dirty: Optional[List[str]] = None,
    *,
    shared_root: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Commit exactly *written_files* with curator provenance.

    Safety order:
      1. re-run the porcelain check; if any dirty path is NOT one of the
         files this run wrote (tracked set drifted since the precheck —
         e.g. a non-cooperating sibling wrote during the pass), ABORT with
         a drift report and stage nothing.
      2. ``git add -- <exact file list>`` (explicit pathspec; NEVER a
         directory wildcard).
      3. ``git commit -m "curator: <summary>"``.

    Returns (ok, message). ``message`` carries the commit sha or the reason.
    """
    root = shared_root or _shared_root()
    repo = _git_toplevel(root)
    if repo is None:
        return False, "skills-shared/ is not inside a git repo"
    if not written_files:
        return True, "nothing to commit"

    repo_resolved = repo.resolve()
    rel_files: List[str] = []
    for f in written_files:
        try:
            rel_files.append(str(Path(f).resolve().relative_to(repo_resolved)))
        except ValueError:
            return False, f"refusing to commit a path outside the repo: {f}"

    dirty = _porcelain_shared(repo, root)
    if dirty is None:
        return False, "pre-commit git status failed"
    expected = set(rel_files) | set(precheck_dirty or [])
    drifted = [p for p in dirty if p not in expected]
    if drifted:
        return False, (
            "aborted: tracked set drifted since precheck "
            f"(unexpected paths: {', '.join(sorted(drifted)[:10])})"
        )

    code, _, err = _git(repo, "add", "--", *rel_files)
    if code != 0:
        return False, f"git add failed: {err.strip()}"
    code, out, err = _git(
        repo, "commit", "-m", f"curator: {summary}", "--", *rel_files,
    )
    if code != 0:
        return False, f"git commit failed: {err.strip() or out.strip()}"
    code, sha, _ = _git(repo, "rev-parse", "--short", "HEAD")
    return True, sha.strip() if code == 0 else "committed"


def attempt_crash_recovery(
    shared_root: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Recover a dirty shared tree left by a KILLED prior curator run.

    Fires ONLY under the exact-set rule (spec 5.3.3a): every dirty path must
    be in the last shared snapshot's manifested intended-write set and
    byte-restorable from that snapshot's tar, and NO un-manifested path may
    be dirty. Any superset / unknown path / missing snapshot →
    skip-and-report (NEVER auto-clobber a sibling edit).

    The caller must hold the shared pass lock (a cleanly-acquired lock over
    a dirty tree IS the staleness signal — fcntl auto-releases on holder
    death, so a live holder would have made acquisition fail).

    P1 FIX: Manifest paths are validated for traversal and cross-checked against
    the baseline commit to prevent arbitrary file deletion via planted manifests.

    Returns (recovered, reason).
    """
    root = shared_root or _shared_root()
    repo = _git_toplevel(root)
    if repo is None:
        return False, "no git repo"
    dirty = _porcelain_shared(repo, root)
    if not dirty:
        return False, "tree is clean"

    snap = latest_shared_snapshot()
    if snap is None:
        return False, "no shared snapshot to recover from"
    manifest_path = snap / SHARED_MANIFEST_NAME
    if not manifest_path.exists():
        return False, "snapshot has no manifest"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "unreadable snapshot manifest"
    
    # P1 FIX: Validate manifest paths before trusting them for deletion
    intended_raw = manifest.get("intended_writes") or []
    intended = set()
    repo_resolved = repo.resolve()
    root_resolved = root.resolve()
    
    for path_str in intended_raw:
        # Reject absolute paths
        p = Path(path_str)
        if p.is_absolute():
            return False, f"manifest contains absolute path (rejected): {path_str}"
        
        # Require the path to resolve inside the shared tree
        try:
            full_path = (repo / path_str).resolve(strict=False)
            full_path.relative_to(root_resolved)
        except ValueError:
            return False, f"manifest path escapes shared tree (rejected): {path_str}"
        
        intended.add(path_str)
    
    # P1 FIX: Cross-check intended deletions against baseline commit
    # Only delete files that existed in the baseline (curator couldn't have created
    # a file that never existed in git)
    baseline_rev = manifest.get("baseline_rev")
    if baseline_rev:
        for path_str in intended:
            # Check if this path existed in the baseline commit
            code, _, _ = _git(repo, "cat-file", "-e", f"{baseline_rev}:{path_str}")
            if code != 0:
                # Path didn't exist in baseline — curator couldn't have written it
                # during this run, so it's either a new sibling file or a manifest lie
                logger.debug(
                    "recovery: path in manifest but not in baseline %s: %s",
                    baseline_rev, path_str
                )
                # Don't fail recovery for untracked new files (carves), but DO
                # fail if a tracked file in dirty set wasn't in baseline
                if path_str in dirty:
                    code2, _, _ = _git(repo, "ls-files", "--", path_str)
                    if code2 == 0:  # It's tracked now but wasn't in baseline
                        return False, (
                            f"manifest claims path not in baseline (rejected): {path_str}"
                        )
    
    unmanifested = [p for p in dirty if p not in intended]
    if unmanifested:
        return False, (
            "sibling edit present — skip "
            f"(un-manifested dirty paths: {', '.join(sorted(unmanifested)[:10])})"
        )

    # Every dirty path is a manifested in-flight file → self-inflicted dirt.
    # Restore via git (the files were tracked and clean at the precheck).
    code, _, err = _git(repo, "checkout", "--", *sorted(dirty))
    if code != 0:
        # untracked leftovers (new carve files) — remove them, they were
        # curator-written per the manifest.
        removed = []
        for p in dirty:
            if p not in intended:
                continue  # Defense in depth: only delete validated manifest paths
            fp = repo / p
            try:
                if fp.exists() and not fp.is_dir():
                    # Re-validate containment before unlink
                    fp_resolved = fp.resolve()
                    fp_resolved.relative_to(root_resolved)
                    fp.unlink()
                    removed.append(p)
            except (OSError, ValueError):
                pass
        still = _porcelain_shared(repo, root)
        if still:
            return False, f"recovery incomplete: {err.strip()}"
    else:
        # git checkout restores modified tracked files but leaves untracked
        # curator-written leftovers; clear those too (manifested only).
        leftover = _porcelain_shared(repo, root) or []
        for p in leftover:
            if p in intended:
                fp = repo / p
                try:
                    if fp.exists() and not fp.is_dir():
                        # Re-validate containment before unlink
                        fp_resolved = fp.resolve()
                        fp_resolved.relative_to(root_resolved)
                        fp.unlink()
                except (OSError, ValueError):
                    pass
        still = _porcelain_shared(repo, root)
        if still:
            return False, "recovery incomplete: tree still dirty"
    return True, "restored self-inflicted dirt from git/manifest"


def archive_shared_skill(
    skill_dir: Path,
    *,
    shared_root: Optional[Path] = None,
) -> Tuple[bool, str, List[Path]]:
    """Archive a shared skill in-tree: ``skills-shared/<group>/.archive/<name>/``.

    A plain rename INSIDE the shared tree so the run-commit's explicit
    pathspec captures both the deletion and the .archive addition — a
    ``git revert`` of the run-commit restores the skill for the whole fleet.
    Never moves the skill to the local ``skills/.archive/`` (other agents
    would lose it). Returns (ok, message, touched_paths).
    
    P1 FIX: This function now goes through the FULL safety contract (lock,
    precheck, snapshot, commit) to prevent dirty-tree corruption. The raw
    rename logic is now in _archive_shared_skill_impl; this wrapper enforces
    the git safety gates.
    """
    root = shared_root or _shared_root()
    skill_dir = Path(skill_dir)
    try:
        rel = skill_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return False, f"{skill_dir} is not under the shared tree", []
    if not rel.parts or len(rel.parts) < 2:
        return False, f"{skill_dir} is not a <group>/<skill> dir", []
    
    # P1 FIX: Acquire lock, precheck, snapshot, commit
    with shared_pass_lock(root) as acquired:
        if not acquired:
            return False, "lock contention (another curator holds shared lock)", []
        
        ok, reason, dirty = git_precheck_shared(root)
        if not ok and reason == "dirty working tree":
            recovered, why = attempt_crash_recovery(root)
            if recovered:
                ok, reason, dirty = git_precheck_shared(root)
        if not ok:
            return False, f"precheck failed ({reason})", []
        
        # Compute intended paths BEFORE the rename (for drift detection)
        repo = _git_toplevel(root)
        if repo is None:
            return False, "no git repo", []
        
        repo_resolved = repo.resolve()
        group = rel.parts[0]
        archive_root_path = root / group / ".archive"
        dest_name = skill_dir.name
        # Check if dest already exists and would get timestamped
        dest_path = archive_root_path / dest_name
        if dest_path.exists():
            from datetime import datetime, timezone
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            dest_path = archive_root_path / f"{dest_name}-{stamp}"
        
        # Pre-compute ALL files that will be dirty after rename (recursively)
        intended_pre = []
        # All files in source dir
        for item in skill_dir.rglob("*"):
            if item.is_file():
                try:
                    intended_pre.append(str(item.resolve().relative_to(repo_resolved)))
                except ValueError:
                    pass
        # All files in dest dir (mirror structure)
        for item in skill_dir.rglob("*"):
            if item.is_file():
                try:
                    rel_in_skill = item.relative_to(skill_dir)
                    dest_file = dest_path / rel_in_skill
                    intended_pre.append(str(dest_file.resolve(strict=False).relative_to(repo_resolved)))
                except ValueError:
                    pass
        
        precheck_dirty = (dirty or []) + intended_pre
        
        # P1 FIX (Greptile): Take snapshot BEFORE rename (invariant: manifest before mutation)
        snap = snapshot_shared([skill_dir.parent], intended_pre, shared_root=root)
        if snap is None:
            return False, "shared snapshot failed (hard gate)", []
        
        # Perform the rename (AFTER snapshot is safe)
        ok_rename, msg_rename, touched = _archive_shared_skill_impl(
            skill_dir, shared_root=root
        )
        if not ok_rename:
            return False, msg_rename, []
        
        # Build intended list from actual touched paths
        intended = []
        for p in touched:
            try:
                intended.append(str(p.resolve().relative_to(repo_resolved)))
            except ValueError:
                pass
        
        # Commit the archive operation (pass precheck_dirty for drift detection)
        ok_commit, msg_commit = commit_shared(
            f"archive shared skill {skill_dir.name}",
            touched,
            precheck_dirty=precheck_dirty,
            shared_root=root,
        )
        if not ok_commit:
            # Rollback
            try:
                dest = touched[1] if len(touched) > 1 else None
                if dest and dest.exists():
                    dest.rename(skill_dir)
            except OSError:
                pass
            return False, f"commit failed: {msg_commit}", []
        
        return True, f"archived to {touched[1] if len(touched) > 1 else '?'}", touched


def _archive_shared_skill_impl(
    skill_dir: Path,
    *,
    shared_root: Optional[Path] = None,
) -> Tuple[bool, str, List[Path]]:
    """Implementation of shared skill archival (rename only, no git ops).
    
    Used internally by archive_shared_skill after the safety contract is acquired.
    """
    root = shared_root or _shared_root()
    skill_dir = Path(skill_dir)
    try:
        rel = skill_dir.resolve().relative_to(root.resolve())
    except ValueError:
        return False, f"{skill_dir} is not under the shared tree", []
    if not rel.parts or len(rel.parts) < 2:
        return False, f"{skill_dir} is not a <group>/<skill> dir", []
    group = rel.parts[0]
    archive_root = root / group / ".archive"
    try:
        archive_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"failed to create shared archive dir: {e}", []
    dest = archive_root / skill_dir.name
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        dest = archive_root / f"{skill_dir.name}-{stamp}"
    try:
        skill_dir.rename(dest)
    except OSError:
        import shutil

        try:
            shutil.move(str(skill_dir), str(dest))
        except Exception as e:
            return False, f"failed to archive shared skill: {e}", []
    return True, f"archived to {dest}", [skill_dir, dest]
