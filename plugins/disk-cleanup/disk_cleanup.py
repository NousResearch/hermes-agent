"""disk_cleanup — ephemeral file cleanup for Hermes Agent.

Library module wrapping the deterministic cleanup rules written by
@LVT382009 in PR #12212. The plugin ``__init__.py`` wires these
functions into ``post_tool_call`` and ``on_session_end`` hooks so
tracking and cleanup happen automatically — the agent never needs to
call a tool or remember a skill.

Rules:
  - disposable test files outside Git-protected paths → delete at task end
  - temp files    → delete after 7 days
  - cron-output   → delete after 14 days
  - unprotected empty dirs → always delete (under HERMES_HOME)
  - research      → keep 10 newest, prompt for older (deep only)
  - chrome-profile→ prompt after 14 days (deep only)
  - >500 MB files → prompt always (deep only)

Scope: strictly HERMES_HOME and /tmp/hermes-*
Automatic cleanup never touches Git worktree/source paths, ~/.hermes/logs/,
or system directories. Explicit manual tracking may override protection for the
selected path, but recursive deletion still refuses nested or bare Git repositories.
Resolved scope checks and parent-directory descriptors prevent symlink escapes
or non-canonical path components from redirecting destructive operations.
Durable Hermes state and filesystem mount boundaries always fail closed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover — plugin may load before constants resolves
    import os

    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


logger = logging.getLogger(__name__)

# Keep the capability-tested primitives stable even when tests or host
# instrumentation wrap functions on the shared ``os`` module.
_OS_OPEN = os.open
_OS_STAT = os.stat
_OS_UNLINK = os.unlink
_OS_RMDIR = os.rmdir
_SHUTIL_RMTREE = shutil.rmtree
_RMTREE_SYMLINK_SAFE = getattr(_SHUTIL_RMTREE, "avoids_symlink_attacks", False)
_DIR_FD_DELETE_SUPPORTED = (
    {_OS_OPEN, _OS_STAT, _OS_UNLINK}.issubset(os.supports_dir_fd)
    and _OS_STAT in os.supports_follow_symlinks
    and hasattr(os, "O_DIRECTORY")
)
_DIR_FD_RMDIR_SUPPORTED = (
    _DIR_FD_DELETE_SUPPORTED and _OS_RMDIR in os.supports_dir_fd
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_state_dir() -> Path:
    """State dir — separate from ``$HERMES_HOME/logs/``."""
    return get_hermes_home() / "disk-cleanup"


def get_tracked_file() -> Path:
    return get_state_dir() / "tracked.json"


def get_log_file() -> Path:
    """Audit log — intentionally NOT under ``$HERMES_HOME/logs/``."""
    return get_state_dir() / "cleanup.log"


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def is_safe_path(path: Path) -> bool:
    """Accept only resolved paths under HERMES_HOME or ``/tmp/hermes-*``.

    Rejects Windows mounts (``/mnt/c`` etc.), symlink escapes, and system
    directories.
    """
    raw_path = Path(path).expanduser()
    raw_parts = raw_path.parts
    if (
        len(raw_parts) >= 3
        and raw_parts[1] == "mnt"
        and len(raw_parts[2]) == 1
    ):
        return False

    try:
        resolved = raw_path.resolve()
        hermes_home = get_hermes_home().resolve()
    except OSError:
        return False

    try:
        resolved.relative_to(hermes_home)
        return True
    except ValueError:
        pass

    # Resolve both sides before applying the /tmp/hermes-* allowance.  A
    # lexical prefix is insufficient because an interior symlink can escape
    # the cleanup root (for example /tmp/hermes-x/outside -> /etc).
    try:
        tmp_relative = resolved.relative_to(Path("/tmp").resolve())
    except (ValueError, OSError):
        return False
    return bool(
        tmp_relative.parts and tmp_relative.parts[0].startswith("hermes-")
    )


def _is_canonical_candidate_path(path: Path) -> bool:
    """Reject relative, special-component, symlinked, or unresolved paths."""
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        return False
    try:
        return path == path.resolve(strict=True)
    except OSError:
        return False


def _is_same_cleanup_device(path: Path) -> bool:
    """Reject nested mounts so cleanup cannot cross filesystem boundaries."""
    try:
        resolved = path.resolve(strict=True)
        hermes_home = get_hermes_home().resolve(strict=True)
        try:
            resolved.relative_to(hermes_home)
            scope_root = hermes_home
        except ValueError:
            tmp_root = Path("/tmp").resolve(strict=True)
            relative = resolved.relative_to(tmp_root)
            if not relative.parts or not relative.parts[0].startswith("hermes-"):
                return False
            scope_root = tmp_root / relative.parts[0]
        return resolved.stat().st_dev == scope_root.stat().st_dev
    except (OSError, ValueError):
        return False


def _looks_like_bare_git_repo(path: Path) -> bool:
    """Return True when *path* has the structural markers of a bare repo."""
    return (
        (path / "HEAD").is_file()
        and (path / "objects").is_dir()
        and (path / "refs").is_dir()
    )


def _find_git_root(path: Path) -> Optional[Path]:
    """Return the nearest normal, linked-worktree, or bare Git root.

    Filesystem inspection errors propagate so callers can fail closed rather
    than treating an unreadable repository marker as proof of no repository.
    """
    try:
        resolved = path.resolve()
        probe = resolved if resolved.is_dir() else resolved.parent
    except OSError as exc:
        raise OSError(f"cannot resolve Git probe path: {exc}") from exc
    for candidate in (probe, *probe.parents):
        try:
            marker = candidate / ".git"
            if marker.is_dir() or marker.is_file():
                return candidate
            if _looks_like_bare_git_repo(candidate):
                return candidate
        except OSError as exc:
            raise OSError(f"cannot inspect Git marker at {candidate}: {exc}") from exc
    return None


def _is_git_protected_path(path: Path) -> bool:
    """Return True when automatic cleanup must preserve *path*.

    Dedicated/nested Git worktrees are entirely protected.  ``HERMES_HOME``
    may itself be a Git repository, so paths in that root are protected when
    tracked or non-ignored; declared cache/cron-output roots and explicitly
    ignored ephemeral paths remain eligible for cleanup.  Git command errors
    fail closed and preserve the path.
    """
    try:
        resolved = path.resolve()
        hermes_home = get_hermes_home().resolve()
    except OSError:
        return True

    for dirname in ("worktrees", ".worktrees"):
        try:
            resolved.relative_to(hermes_home / dirname)
            return True
        except ValueError:
            pass

    try:
        git_root = _find_git_root(path)
    except OSError:
        return True
    if git_root is None:
        return False
    if git_root.resolve() != hermes_home:
        return True

    try:
        relative = resolved.relative_to(git_root.resolve())
    except ValueError:
        return True

    try:
        tracked = subprocess.run(
            [
                "git", "-C", str(git_root), "ls-files",
                "--error-unmatch", "--", str(relative),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if tracked.returncode == 0:
        return True
    if tracked.returncode != 1:
        return True

    parts = relative.parts
    if parts and parts[0] == "cache":
        return False
    if (
        len(parts) >= 2
        and parts[0] in {"cron", "cronjobs"}
        and parts[1] == "output"
    ):
        return False

    try:
        ignored = subprocess.run(
            ["git", "-C", str(git_root), "check-ignore", "-q", "--", str(relative)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True

    if ignored.returncode == 0:
        return False
    if ignored.returncode == 1:
        return True
    return True


def _directory_contains_git_marker(path: Path) -> bool:
    """Return True for nested Git repositories or filesystem boundaries.

    Recursive tracked-directory deletion is never allowed to cross a nested
    repository or mount boundary.  Inspection errors fail closed because an
    unreadable subtree cannot be proven disposable.
    """
    try:
        root_device = path.lstat().st_dev
    except OSError:
        return True
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            if os.path.ismount(current) or current.lstat().st_dev != root_device:
                return True
            # Bare repositories have no enclosing .git marker.
            if _looks_like_bare_git_repo(current):
                return True
        except OSError:
            return True
        try:
            children = list(current.iterdir())
        except OSError:
            return True
        for child in children:
            if child.name.casefold() == ".git":
                return True
            try:
                if child.is_dir() and not child.is_symlink():
                    stack.append(child)
            except OSError:
                return True
    return False


def _open_bound_parent(path: Path) -> Tuple[Optional[int], str]:
    """Open and identity-bind *path*'s parent directory.

    Destructive operations use the returned descriptor plus the candidate's
    basename, so replacing an ancestor with a symlink cannot redirect unlink
    or recursive deletion to a different tree.
    """
    if not _DIR_FD_DELETE_SUPPORTED:
        return None, "platform lacks required dir_fd safety"

    try:
        parent = path.parent.resolve(strict=True)
        parent_before = parent.stat()
        flags = os.O_RDONLY | os.O_DIRECTORY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_fd = _OS_OPEN(parent, flags)
    except OSError as exc:
        raise OSError(f"cannot bind candidate parent: {exc}") from exc

    try:
        parent_bound = os.fstat(parent_fd)
        if (
            parent_before.st_dev,
            parent_before.st_ino,
        ) != (
            parent_bound.st_dev,
            parent_bound.st_ino,
        ):
            os.close(parent_fd)
            return None, "parent identity changed during validation"
    except OSError as exc:
        os.close(parent_fd)
        raise OSError(f"cannot validate candidate parent: {exc}") from exc
    return parent_fd, ""


def _bound_candidate_stat(parent_fd: int, path: Path):
    """Stat *path* without following its final symlink via a bound parent."""
    return _OS_STAT(path.name, dir_fd=parent_fd, follow_symlinks=False)


def _same_identity(left, right) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _delete_candidate(path: Path, *, explicit: bool) -> Tuple[bool, str]:
    """Revalidate and delete one tracked path at the destructive boundary.

    Explicit manual tracking may override protection for the selected file or
    directory itself, but recursive deletion still refuses directories that
    contain nested Git repositories/worktrees.
    """
    if not _is_canonical_candidate_path(path):
        return False, "non-canonical candidate path"
    if _is_durable_protected_path(path):
        return False, "durable Hermes state is protected"
    if not _is_same_cleanup_device(path) or os.path.ismount(path):
        return False, "filesystem mount boundary is protected"
    if not path.exists() and not path.is_symlink():
        return False, "missing"
    if not is_safe_path(path):
        return False, "outside allowed cleanup scope"
    if _is_git_protected_path(path) and not explicit:
        return False, "Git-protected"

    try:
        before = path.lstat()
    except OSError as exc:
        raise OSError(f"cannot inspect candidate: {exc}") from exc

    is_directory = path.is_dir() and not path.is_symlink()
    if is_directory and _directory_contains_git_marker(path):
        return False, "contains nested Git repository/worktree"

    # Detect pathname replacement during validation.  shutil.rmtree also uses
    # fd-based symlink-attack resistance where the platform supports it.
    try:
        after = path.lstat()
    except OSError as exc:
        raise OSError(f"candidate changed during validation: {exc}") from exc
    if not _same_identity(before, after):
        return False, "candidate identity changed during validation"

    if not is_safe_path(path):
        return False, "outside allowed cleanup scope at delete boundary"
    if _is_git_protected_path(path) and not explicit:
        return False, "became Git-protected at delete boundary"

    if path.is_symlink() or path.is_file():
        parent_fd, reason = _open_bound_parent(path)
        if parent_fd is None:
            return False, reason
        try:
            # Re-check pathname policy after binding the parent.  If an
            # ancestor changed while the descriptor was opened, either scope
            # validation or the bound inode comparison below fails closed.
            if not is_safe_path(path):
                return False, "outside allowed scope after parent bind"
            if not _is_same_cleanup_device(path) or os.path.ismount(path):
                return False, "filesystem boundary changed after parent bind"
            if _is_git_protected_path(path) and not explicit:
                return False, "became Git-protected after parent bind"
            bound = _bound_candidate_stat(parent_fd, path)
            if not _same_identity(before, bound):
                return False, "candidate identity changed after parent bind"
            _OS_UNLINK(path.name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    elif path.is_dir():
        if not _RMTREE_SYMLINK_SAFE:
            return False, "platform lacks symlink-safe recursive deletion"
        # Re-scan immediately before recursive deletion so a repository added
        # during the earlier checks is still preserved.
        if _directory_contains_git_marker(path):
            return False, "contains nested Git repository/worktree at delete boundary"
        if not is_safe_path(path):
            return False, "outside allowed cleanup scope after directory scan"
        if _is_git_protected_path(path) and not explicit:
            return False, "became Git-protected after directory scan"
        try:
            final = path.lstat()
        except OSError as exc:
            raise OSError(f"candidate changed before recursive delete: {exc}") from exc
        if not _same_identity(before, final):
            return False, "candidate identity changed before recursive delete"

        parent_fd, reason = _open_bound_parent(path)
        if parent_fd is None:
            return False, reason
        try:
            if not is_safe_path(path):
                return False, "outside allowed scope after directory parent bind"
            if not _is_same_cleanup_device(path) or os.path.ismount(path):
                return False, "filesystem boundary changed after directory parent bind"
            if _is_git_protected_path(path) and not explicit:
                return False, "became Git-protected after directory parent bind"
            bound = _bound_candidate_stat(parent_fd, path)
            if not _same_identity(before, bound):
                return False, "directory identity changed after parent bind"
            _SHUTIL_RMTREE(path.name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    else:
        return False, "unsupported filesystem object"
    return True, ""


def _remove_empty_directory(path: Path) -> bool:
    """Remove one empty directory through a validated parent descriptor."""
    if not _DIR_FD_RMDIR_SUPPORTED:
        return False
    if not _is_canonical_candidate_path(path):
        return False
    if _is_durable_protected_path(path):
        return False
    if not _is_same_cleanup_device(path) or os.path.ismount(path):
        return False
    if not is_safe_path(path) or _is_git_protected_path(path):
        return False
    try:
        before = path.lstat()
        if path.is_symlink() or not path.is_dir():
            return False
        parent_fd, _reason = _open_bound_parent(path)
        if parent_fd is None:
            return False
        try:
            if not is_safe_path(path) or _is_git_protected_path(path):
                return False
            if not _is_same_cleanup_device(path) or os.path.ismount(path):
                return False
            bound = _bound_candidate_stat(parent_fd, path)
            if not _same_identity(before, bound):
                return False
            _OS_RMDIR(path.name, dir_fd=parent_fd)
            return True
        finally:
            os.close(parent_fd)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    try:
        log_file = get_log_file()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except OSError:
        # Never let the audit log break the agent loop.
        pass


# ---------------------------------------------------------------------------
# tracked.json — atomic read/write, backup scoped to tracked.json only
# ---------------------------------------------------------------------------

def load_tracked() -> List[Dict[str, Any]]:
    """Load tracked.json.  Restores from ``.bak`` on corruption."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)

    if not tf.exists():
        return []

    try:
        return json.loads(tf.read_text())
    except (json.JSONDecodeError, ValueError):
        bak = tf.with_suffix(".json.bak")
        if bak.exists():
            try:
                data = json.loads(bak.read_text())
                _log("WARN: tracked.json corrupted — restored from .bak")
                return data
            except Exception:
                pass
        _log("WARN: tracked.json corrupted, no backup — starting fresh")
        return []


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    """Atomic write: ``.tmp`` → backup old → rename."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp = tf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tracked, indent=2))
    if tf.exists():
        shutil.copy2(tf, tf.with_suffix(".json.bak"))
    tmp.replace(tf)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES = {
    "temp", "test", "research", "download",
    "chrome-profile", "cron-output", "other",
}

_EMPTY_DIR_PROTECTED_TOP_LEVEL = frozenset({
    "logs", "memories", "sessions", "cron", "cronjobs",
    "cache", "skills", "plugins", "disk-cleanup", "optional-skills",
    "hermes-agent", "backups", "profiles", ".worktrees", "worktrees",
})

_DURABLE_PROTECTED_TOP_LEVEL = frozenset({
    "logs", "memories", "sessions", "skills", "plugins", "disk-cleanup",
    "optional-skills", "backups", "profiles", "config.yaml", "config.yml",
    ".env", "USER.md", "MEMORY.md", "SOUL.md", "auth.json",
})

_EMPTY_DIR_SWEEP_PRUNE_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv",
    "site-packages", "__pycache__",
})


# Paths under $HERMES_HOME that must NEVER be deleted by quick(),
# regardless of what the stored category says.  This is a defense-in-depth
# guard against stale or corrupted tracked.json entries.
_PROTECTED_CRON_PATHS: set[str] = set()


def _is_durable_protected_path(p: Path) -> bool:
    """Protect Hermes control-plane and durable user state from all cleanup.

    Manual provenance may override Git protection for one selected path, but
    it never overrides these durable-state boundaries.  ``cron/output`` is
    intentionally excluded because it is the cron tree's disposable subtree.
    """
    try:
        resolved = p.resolve()
        hermes_home = get_hermes_home().resolve()
    except OSError:
        return True
    try:
        relative = resolved.relative_to(hermes_home)
    except ValueError:
        return False
    if not relative.parts:
        return True
    top = relative.parts[0]
    if top in _DURABLE_PROTECTED_TOP_LEVEL:
        return True
    if top in {"cron", "cronjobs"}:
        return not (len(relative.parts) >= 2 and relative.parts[1] == "output")
    return False


def _is_protected_cron_path(p: Path) -> bool:
    """Return True if *p* is a cron control-plane file/directory that must
    never be deleted.

    This matches, by EXACT path only, the ``cron/`` directory itself, known
    control-plane files (``jobs.json``, ``.tick.lock``), and the ``output/``
    root directory. It does NOT (and must not be "simplified" to) blanket-match
    everything under ``cron/output/`` — those run artifacts are disposable and
    are cleaned by retention policy; only the ``output/`` root itself is
    protected, because deleting it wholesale erases every job's retained run
    history at once.
    """
    # Lazily build the set once per process so HERMES_HOME is resolved
    # exactly once.
    if not _PROTECTED_CRON_PATHS:
        hermes_home = get_hermes_home()
        for parent in ("cron", "cronjobs"):
            base = hermes_home / parent
            _PROTECTED_CRON_PATHS.add(str(base))
            _PROTECTED_CRON_PATHS.add(str(base / "output"))
            _PROTECTED_CRON_PATHS.add(str(base / "jobs.json"))
            _PROTECTED_CRON_PATHS.add(str(base / ".tick.lock"))
    resolved = str(p.resolve())
    return resolved in _PROTECTED_CRON_PATHS


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Track / forget
# ---------------------------------------------------------------------------

def track(
    path_str: str,
    category: str,
    silent: bool = False,
    explicit: bool = True,
) -> bool:
    """Register a file for tracking. Returns True if newly tracked or upgraded.

    ``explicit`` records whether the user deliberately invoked ``track``.
    Automatic hooks pass ``False``; legacy entries without the field are
    therefore treated as non-explicit and fail closed around Git source paths.
    """
    if category not in ALLOWED_CATEGORIES:
        _log(f"WARN: unknown category '{category}', using 'other'")
        category = "other"

    path = Path(path_str).resolve()

    if not path.exists():
        _log(f"SKIP: {path} (does not exist)")
        return False

    if not is_safe_path(path):
        _log(f"REJECT: {path} (outside HERMES_HOME)")
        return False

    if _is_durable_protected_path(path):
        _log(f"REJECT: {path} (durable Hermes state)")
        return False

    if not _is_same_cleanup_device(path) or os.path.ismount(path):
        _log(f"REJECT: {path} (filesystem mount boundary)")
        return False

    size = path.stat().st_size if path.is_file() else 0
    tracked = load_tracked()

    # Deduplicate. An explicit manual command may safely upgrade a legacy or
    # auto-hook entry so the user's deliberate cleanup request still works.
    for item in tracked:
        if item["path"] != str(path):
            continue
        if explicit and not item.get("explicit", False):
            item.update({
                "explicit": True,
                "category": category,
                "size": size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            save_tracked(tracked)
            _log(f"TRACKED explicit upgrade: {path} ({category})")
            return True
        return False

    tracked.append({
        "path": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "size": size,
        "explicit": explicit,
    })
    save_tracked(tracked)
    _log(f"TRACKED: {path} ({category}, {fmt_size(size)})")
    if not silent:
        print(f"Tracked: {path} ({category}, {fmt_size(size)})")
    return True


def forget(path_str: str) -> int:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    tracked = load_tracked()
    before = len(tracked)
    tracked = [i for i in tracked if Path(i["path"]).resolve() != p]
    removed = before - len(tracked)
    if removed:
        save_tracked(tracked)
        _log(f"FORGOT: {p} ({removed} entries)")
    return removed


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run() -> Tuple[List[Dict], List[Dict]]:
    """Return (auto_delete_list, needs_prompt_list) without touching files."""
    tracked = load_tracked()
    now = datetime.now(timezone.utc)

    auto: List[Dict] = []
    prompt: List[Dict] = []

    for item in tracked:
        p = Path(item["path"])
        if not _is_canonical_candidate_path(p):
            continue
        if _is_durable_protected_path(p):
            continue
        if not p.exists():
            continue
        if not is_safe_path(p):
            continue
        if _is_git_protected_path(p) and not item.get("explicit", False):
            continue
        if (
            p.is_dir()
            and not p.is_symlink()
            and _directory_contains_git_marker(p)
        ):
            continue
        age = (now - datetime.fromisoformat(item["timestamp"])).days
        cat = item["category"]
        size = item["size"]

        # Re-validate stale "cron-output" entries (fixes #37721).
        if cat == "cron-output" and not item.get("explicit", False):
            re_cat = guess_category(p)
            if re_cat != "cron-output":
                # Stale entry — would be skipped by quick(); omit from
                # dry-run output too.
                continue

        if cat == "test":
            auto.append(item)
        elif cat == "temp" and age > 7:
            auto.append(item)
        elif cat == "cron-output" and age > 14:
            auto.append(item)
        elif cat == "research" and age > 30:
            prompt.append(item)
        elif cat == "chrome-profile" and age > 14:
            prompt.append(item)
        elif size > 500 * 1024 * 1024:
            prompt.append(item)

    return auto, prompt


# ---------------------------------------------------------------------------
# Quick cleanup
# ---------------------------------------------------------------------------

def quick() -> Dict[str, Any]:
    """Safe deterministic cleanup — no prompts.

    Returns: ``{"deleted": N, "empty_dirs": N, "freed": bytes,
               "errors": [str, ...]}``.
    """
    tracked = load_tracked()
    now = datetime.now(timezone.utc)
    deleted = 0
    freed = 0
    new_tracked: List[Dict] = []
    errors: List[str] = []

    for item in tracked:
        p = Path(item["path"])
        cat = item["category"]

        if not _is_canonical_candidate_path(p):
            _log(f"SKIP non-canonical stale path: {p} (removed from tracking)")
            continue
        if _is_durable_protected_path(p):
            _log(f"SKIP durable stale path: {p} (removed from tracking)")
            continue
        if not p.exists():
            _log(f"STALE: {p} (removed from tracking)")
            continue

        if not is_safe_path(p):
            _log(f"SKIP unsafe stale path: {p} (removed from tracking)")
            continue

        if _is_git_protected_path(p) and not item.get("explicit", False):
            _log(f"SKIP Git-protected path: {p} (removed from tracking)")
            continue

        age = (now - datetime.fromisoformat(item["timestamp"])).days

        # ---- stale-state migration (fixes #37721) ----
        # Old tracked.json entries may carry a "cron-output" category for
        # paths that are NOT under cron/output/ (e.g. cron/jobs.json).
        # guess_category() was fixed in #34840, but existing entries are
        # never re-validated.  Re-classify here so stale entries for cron
        # control-plane state are not deleted.
        if cat == "cron-output" and not item.get("explicit", False):
            re_cat = guess_category(p)
            if re_cat != "cron-output":
                _log(
                    f"SKIP stale cron-output entry: {p} "
                    f"(re-classified as {re_cat!r})"
                )
                # Drop the stale entry — it was misclassified.
                continue

        # Hard safety net: never delete cron control-plane state even if
        # the category somehow slipped through re-validation above.
        if _is_protected_cron_path(p):
            _log(f"SKIP protected cron path: {p}")
            continue

        should_delete = (
            cat == "test"
            or (cat == "temp" and age > 7)
            or (cat == "cron-output" and age > 14)
        )

        if should_delete:
            try:
                removed, reason = _delete_candidate(
                    p, explicit=item.get("explicit", False)
                )
                if removed:
                    freed += item["size"]
                    deleted += 1
                    _log(f"DELETED: {p} ({cat}, {fmt_size(item['size'])})")
                else:
                    _log(f"SKIP delete-boundary check: {p} ({reason})")
                    if item.get("explicit", False):
                        new_tracked.append(item)
            except OSError as e:
                _log(f"ERROR deleting {p}: {e}")
                errors.append(f"{p}: {e}")
                new_tracked.append(item)
        else:
            new_tracked.append(item)

    # Remove empty dirs under HERMES_HOME, but never recurse into known
    # durable state trees.  Some installs place the Hermes checkout, venv,
    # and desktop build under HERMES_HOME; a full rglob over that tree can
    # stall the gateway event loop for minutes.
    hermes_home = get_hermes_home()
    empty_removed = 0
    sweep_stack: List[Tuple[Path, bool]] = []
    try:
        for top in hermes_home.iterdir():
            if (
                top.is_dir()
                and not top.is_symlink()
                and not os.path.ismount(top)
                and _is_same_cleanup_device(top)
                and not _is_git_protected_path(top)
                and top.name not in _EMPTY_DIR_PROTECTED_TOP_LEVEL
                and top.name not in _EMPTY_DIR_SWEEP_PRUNE_DIRS
            ):
                sweep_stack.append((top, False))
    except OSError:
        sweep_stack = []

    while sweep_stack:
        dirpath, visited = sweep_stack.pop()
        if visited:
            try:
                if not any(dirpath.iterdir()) and _remove_empty_directory(dirpath):
                    empty_removed += 1
                    _log(f"DELETED: {dirpath} (empty dir)")
            except OSError:
                pass
            continue

        sweep_stack.append((dirpath, True))
        try:
            for child in dirpath.iterdir():
                if (
                    child.is_dir()
                    and not child.is_symlink()
                    and not os.path.ismount(child)
                    and _is_same_cleanup_device(child)
                    and not _is_git_protected_path(child)
                    and child.name not in _EMPTY_DIR_SWEEP_PRUNE_DIRS
                ):
                    sweep_stack.append((child, False))
        except OSError:
            pass

    save_tracked(new_tracked)
    _log(
        f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, "
        f"{fmt_size(freed)}"
    )
    return {
        "deleted": deleted,
        "empty_dirs": empty_removed,
        "freed": freed,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Deep cleanup (interactive — not called from plugin hooks)
# ---------------------------------------------------------------------------

def deep(
    confirm: Optional[callable] = None,
) -> Dict[str, Any]:
    """Deep cleanup.

    Runs :func:`quick` first, then asks the *confirm* callable for each
    risky item (research > 30d beyond 10 newest, chrome-profile > 14d,
    any file > 500 MB).  *confirm(item)* must return True to delete.

    Returns: ``{"quick": {...}, "deep_deleted": N, "deep_freed": bytes}``.
    """
    quick_result = quick()

    if confirm is None:
        # No interactive confirmer — deep stops after the quick pass.
        return {"quick": quick_result, "deep_deleted": 0, "deep_freed": 0}

    tracked = load_tracked()
    now = datetime.now(timezone.utc)
    research, chrome, large = [], [], []

    for item in tracked:
        p = Path(item["path"])
        if not _is_canonical_candidate_path(p):
            continue
        if _is_durable_protected_path(p):
            continue
        if not p.exists():
            continue
        if not is_safe_path(p):
            continue
        if _is_git_protected_path(p) and not item.get("explicit", False):
            continue
        age = (now - datetime.fromisoformat(item["timestamp"])).days
        cat = item["category"]

        if cat == "research" and age > 30:
            research.append(item)
        elif cat == "chrome-profile" and age > 14:
            chrome.append(item)
        elif item["size"] > 500 * 1024 * 1024:
            large.append(item)

    research.sort(key=lambda x: x["timestamp"], reverse=True)
    old_research = research[10:]

    freed, count = 0, 0
    to_remove: List[Dict] = []

    for group in (old_research, chrome, large):
        for item in group:
            if confirm(item):
                try:
                    p = Path(item["path"])
                    removed, reason = _delete_candidate(
                        p, explicit=item.get("explicit", False)
                    )
                    if not removed:
                        _log(f"SKIP delete-boundary check: {p} ({reason})")
                        continue
                    to_remove.append(item)
                    freed += item["size"]
                    count += 1
                    _log(
                        f"DELETED: {p} ({item['category']}, "
                        f"{fmt_size(item['size'])})"
                    )
                except OSError as e:
                    _log(f"ERROR deleting {item['path']}: {e}")

    if to_remove:
        remove_paths = {i["path"] for i in to_remove}
        save_tracked([i for i in tracked if i["path"] not in remove_paths])

    return {"quick": quick_result, "deep_deleted": count, "deep_freed": freed}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def status() -> Dict[str, Any]:
    """Return per-category breakdown and top 10 largest tracked files."""
    tracked = load_tracked()
    cats: Dict[str, Dict] = {}
    for item in tracked:
        c = item["category"]
        cats.setdefault(c, {"count": 0, "size": 0})
        cats[c]["count"] += 1
        cats[c]["size"] += item["size"]

    existing = [
        (i["path"], i["size"], i["category"])
        for i in tracked if Path(i["path"]).exists()
    ]
    existing.sort(key=lambda x: x[1], reverse=True)

    return {
        "categories": cats,
        "top10": existing[:10],
        "total_tracked": len(tracked),
    }


def format_status(s: Dict[str, Any]) -> str:
    """Human-readable status string (for slash command output)."""
    lines = [f"{'Category':<20} {'Files':>6}  {'Size':>10}", "-" * 40]
    cats = s["categories"]
    for cat, d in sorted(cats.items(), key=lambda x: x[1]["size"], reverse=True):
        lines.append(f"{cat:<20} {d['count']:>6}  {fmt_size(d['size']):>10}")

    if not cats:
        lines.append("(nothing tracked yet)")

    lines.append("")
    lines.append("Top 10 largest tracked files:")
    if not s["top10"]:
        lines.append("  (none)")
    else:
        for rank, (path, size, cat) in enumerate(s["top10"], 1):
            lines.append(f"  {rank:>2}. {fmt_size(size):>8}  [{cat}]  {path}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-categorisation from tool-call inspection
# ---------------------------------------------------------------------------

_TEST_PATTERNS = ("test_", "tmp_")
_TEST_SUFFIXES = (".test.py", ".test.js", ".test.ts", ".test.md")


def guess_category(path: Path) -> Optional[str]:
    """Return a category label for *path*, or None if we shouldn't track it.

    Used by the ``post_tool_call`` hook to auto-track ephemeral files.
    """
    if not is_safe_path(path):
        return None
    if _is_durable_protected_path(path):
        return None
    if _is_git_protected_path(path):
        return None

    # Skip the state dir itself, logs, memory files, sessions, config.
    hermes_home = get_hermes_home()
    try:
        rel = path.resolve().relative_to(hermes_home)
        top = rel.parts[0] if rel.parts else ""
        if top in {
            "disk-cleanup", "logs", "memories", "sessions", "config.yaml",
            "skills", "plugins", ".env", "USER.md", "MEMORY.md", "SOUL.md",
            "auth.json", "hermes-agent",
        }:
            return None
        if top == "cron" or top == "cronjobs":
            # Only files under the disposable ``output/`` subtree are
            # cleanup candidates. Top-level cron control-plane state
            # (e.g. ``jobs.json``, ``.tick.lock``) must never be
            # auto-tracked — deleting it wipes the live scheduler
            # registry. See issue #32164.
            if len(rel.parts) >= 3 and rel.parts[1] == "output":
                return "cron-output"
            return None
        if top == "cache":
            return "temp"
    except ValueError:
        # Path isn't under HERMES_HOME (e.g. /tmp/hermes-*) — fall through.
        pass

    name = path.name
    if name.startswith(_TEST_PATTERNS):
        return "test"
    if any(name.endswith(sfx) for sfx in _TEST_SUFFIXES):
        return "test"
    return None
