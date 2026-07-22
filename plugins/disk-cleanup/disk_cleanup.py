"""disk_cleanup — ephemeral file cleanup for Hermes Agent.

Library module wrapping the deterministic cleanup rules written by
@LVT382009 in PR #12212. The plugin ``__init__.py`` wires these
functions into ``post_tool_call`` and ``on_session_end`` hooks so
tracking and cleanup happen automatically — the agent never needs to
call a tool or remember a skill.

Rules:
  - test files    → delete immediately at task end (age >= 0)
  - temp files    → delete after 7 days
  - cron-output   → delete after 14 days
  - empty dirs    → delete only inside explicitly marked Hermes-owned roots
  - research      → keep 10 newest, prompt for older (deep only)
  - chrome-profile→ prompt after 14 days (deep only)
  - >500 MB files → prompt always (deep only)

Scope: strictly HERMES_HOME and /tmp/hermes-*
Never touches: ~/.hermes/logs/ or any system directory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover — plugin may load before constants resolves
    import os

    def get_hermes_home() -> Path:  # type: ignore[no-redef]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val).resolve() if val else (Path.home() / ".hermes").resolve()


logger = logging.getLogger(__name__)
_TRACKED_THREAD_LOCK = threading.RLock()


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
    """Accept only paths under HERMES_HOME or ``/tmp/hermes-*``.

    Rejects Windows mounts (``/mnt/c`` etc.) and any system directory.
    """
    try:
        candidate = Path(path)
        hermes_home = Path(get_hermes_home()).resolve()
        resolved = candidate.resolve()
    except (TypeError, ValueError, OSError):
        return False

    try:
        resolved.relative_to(hermes_home)
        return True
    except (ValueError, OSError):
        pass

    # Allow the explicitly temporary POSIX-style ``/tmp/hermes-*`` scope.
    # On Windows, ``Path('/tmp/...').resolve()`` becomes ``C:\\tmp\\...``;
    # retain that compatibility while rejecting an explicit drive path such
    # as ``C:\\tmp\\hermes-user``.  The latter is not the approved temporary
    # namespace and could otherwise be mistaken for it by ``parts`` alone.
    if os.name == "nt" and candidate.anchor not in {"", "\\", "/"}:
        return False
    parts = resolved.parts
    if (
        len(parts) >= 3
        and (
            parts[0] in {"/", "\\"}
            or (os.name == "nt" and candidate.anchor in {"\\", "/"})
        )
        and parts[1].casefold() == "tmp"
        and parts[2].casefold().startswith("hermes-")
    ):
        return True
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

@contextmanager
def _tracked_registry_lock():
    """Serialize tracked.json read-modify-write cycles across processes."""
    lock_path = get_state_dir() / "tracked.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _TRACKED_THREAD_LOCK, lock_path.open("a+b") as lock_file:
        if lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            acquired = True
            yield
        finally:
            if acquired:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_tracked_unlocked() -> List[Dict[str, Any]]:
    """Load tracked.json.  Restores from ``.bak`` on corruption."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)

    if not tf.exists():
        return []

    try:
        data = json.loads(tf.read_text())
        if not isinstance(data, list):
            raise ValueError("tracked.json must contain a list")
        return data
    except (json.JSONDecodeError, ValueError):
        bak = tf.with_suffix(".json.bak")
        if bak.exists():
            try:
                data = json.loads(bak.read_text())
                if not isinstance(data, list):
                    raise ValueError("tracked.json backup must contain a list")
                _log("WARN: tracked.json corrupted — restored from .bak")
                return data
            except Exception:
                pass
        _log("WARN: tracked.json corrupted, no backup — starting fresh")
        return []


def load_tracked() -> List[Dict[str, Any]]:
    with _tracked_registry_lock():
        return _load_tracked_unlocked()


def _save_tracked_unlocked(tracked: List[Dict[str, Any]]) -> None:
    """Atomic write: ``.tmp`` → backup old → rename."""
    tf = get_tracked_file()
    tf.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=tf.parent,
            prefix="tracked-",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(tracked, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        if tf.exists():
            shutil.copy2(tf, tf.with_suffix(".json.bak"))
        tmp_path.replace(tf)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def save_tracked(tracked: List[Dict[str, Any]]) -> None:
    with _tracked_registry_lock():
        _save_tracked_unlocked(tracked)


def _commit_tracked_snapshot(
    original: List[Dict[str, Any]], desired: List[Dict[str, Any]]
) -> None:
    """Apply removals from one snapshot without losing concurrent additions."""
    desired_ids = {id(item) for item in desired}
    removed = [item for item in original if id(item) not in desired_ids]
    with _tracked_registry_lock():
        current = _load_tracked_unlocked()
        for old_item in removed:
            try:
                current.remove(old_item)
            except ValueError:
                pass
        _save_tracked_unlocked(current)


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
    "hermes-agent", "backups", "profiles", ".worktrees",
})

_EMPTY_DIR_SWEEP_PRUNE_DIRS = frozenset({
    ".git", "node_modules", "venv", ".venv",
    "site-packages", "__pycache__",
})

# Empty-directory pruning is limited to roots that this plugin explicitly
# owns.  An arbitrary empty directory under HERMES_HOME may be a user's
# project output or a control-plane staging area; emptiness is not ownership
# evidence.  Keep persistent Singularity/cache subtrees out of the sweep even
# when they live below the opt-in scratch root.
_EMPTY_DIR_OWNED_TOP_LEVEL = frozenset({
    "scratch", "tmp", "temp", "artifacts", "downloads",
})
_EMPTY_DIR_SWEEP_PROTECTED_NAMES = frozenset({"hermes-overlays", ".apptainer"})
_EMPTY_DIR_OWNERSHIP_MARKER = ".hermes-managed"

_MANAGED_ARTIFACT_RETENTION_DAYS = {
    "hook_outputs": 14,
    "spawn-trees": 30,
}
_MANAGED_ARTIFACT_SKIP_NAMES = frozenset({
    ".hermes-managed", ".in-progress", ".active", ".lock",
})


# Paths under $HERMES_HOME that must NEVER be deleted by quick(),
# regardless of what the stored category says.  This is a defense-in-depth
# guard against stale tracked.json entries from before #34840.
_PROTECTED_CRON_PATHS: set[str] = set()

_PROTECTED_TRACKED_TOP_LEVEL = frozenset({
    ".worktrees",
    "backups",
    "disk-cleanup",
    "gateway",
    "hermes-agent",
    "kanban",
    "logs",
    "memories",
    "optional-skills",
    "pairing",
    "platforms",
    "plugins",
    "profiles",
    "sessions",
    "skills",
})
_PROTECTED_TRACKED_TOP_LEVEL_FILES = frozenset({
    ".env",
    "MEMORY.md",
    "SOUL.md",
    "USER.md",
    "auth.json",
    "channel_aliases.json",
    "channel_directory.json",
    "config.yaml",
    "feishu_comment_pairing.json",
    "gateway_state.json",
    "kanban.db",
    "memory_store.db",
    "processes.json",
    "projects.db",
    "response_store.db",
    "state.db",
    "verification_evidence.db",
})
_PROTECTED_TRACKED_TOP_LEVEL_CASEFOLD = frozenset(
    name.casefold() for name in _PROTECTED_TRACKED_TOP_LEVEL
)
_PROTECTED_TRACKED_TOP_LEVEL_FILES_CASEFOLD = frozenset(
    name.casefold() for name in _PROTECTED_TRACKED_TOP_LEVEL_FILES
)
_DISPOSABLE_TOP_LEVEL_PREFIXES = ("test_", "tmp_", "temp_")


def _is_narrow_top_level_artifact(name: str) -> bool:
    """Allow only unmistakably disposable names at HERMES_HOME's root."""
    folded = name.casefold()
    return (
        folded.startswith(_DISPOSABLE_TOP_LEVEL_PREFIXES)
        or ".test." in folded
        or folded.endswith(".tmp")
    )


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
            for protected in (base, base / "output", base / "jobs.json", base / ".tick.lock"):
                _PROTECTED_CRON_PATHS.add(
                    os.path.normcase(os.path.normpath(str(protected.resolve())))
                )
    resolved = os.path.normcase(os.path.normpath(str(p.resolve())))
    return resolved in _PROTECTED_CRON_PATHS


def _is_protected_tracked_path(path: Path) -> bool:
    """Protect durable Hermes state from manual or stale tracking records."""
    try:
        rel = path.resolve().relative_to(get_hermes_home().resolve())
    except (OSError, ValueError):
        return False
    if not rel.parts:
        return True
    top = rel.parts[0].casefold()
    return (
        top in _PROTECTED_TRACKED_TOP_LEVEL_CASEFOLD
        or (len(rel.parts) == 1 and not _is_narrow_top_level_artifact(top))
        or (
            len(rel.parts) == 1
            and top in _PROTECTED_TRACKED_TOP_LEVEL_FILES_CASEFOLD
        )
    )


def _is_owned_empty_dir_root(path: Path) -> bool:
    """Return whether *path* carries explicit ownership evidence.

    Names such as ``scratch`` and ``tmp`` are common in user projects and
    cannot authorize deletion by themselves.  The creator of a disposable
    Hermes root must leave a regular ``.hermes-managed`` marker containing
    the expected root name.  Missing, symlinked, or malformed markers fail
    closed and keep the entire subtree untouched.
    """
    if path.name not in _EMPTY_DIR_OWNED_TOP_LEVEL:
        return False
    marker = path / _EMPTY_DIR_OWNERSHIP_MARKER
    try:
        owned = (
            marker.is_file()
            and not marker.is_symlink()
            and marker.read_text(encoding="utf-8").strip() == path.name
        )
        if not owned:
            return False
        # A lock/active marker is evidence that this namespace may still be
        # live.  Treat malformed or inaccessible markers as active too.
        return not _has_active_maintenance_marker(path)
    except (OSError, UnicodeError):
        return False


def _has_active_maintenance_marker(path: Path) -> bool:
    """Return whether *path* has a live or unverifiable maintenance marker."""
    for name in (".in-progress", ".active", ".lock"):
        marker = path / name
        try:
            marker.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return True
        # A symlink or any existing marker is intentionally conservative:
        # stale ownership cannot be inferred from its name or contents.
        return True
    return False


def _capture_identity(path: Path) -> Optional[Dict[str, int]]:
    """Capture the filesystem object identity required for later deletion."""
    try:
        stat_result = path.lstat()
    except (OSError, ValueError):
        return None
    return {
        "version": 1,
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "mode": int(stat_result.st_mode),
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "ctime_ns": int(stat_result.st_ctime_ns),
    }


def _identity_matches(path: Path, expected: Dict[str, int]) -> bool:
    """Return whether *path* is still the exact object that was tracked."""
    current = _capture_identity(path)
    return current is not None and current == expected


def _validated_tracked_item(item: Any):
    """Return validated tracking fields, or ``None`` for untrusted evidence."""
    if not isinstance(item, dict):
        return None
    path_value = item.get("path")
    category = item.get("category")
    timestamp = item.get("timestamp")
    size = item.get("size")
    identity = item.get("identity")
    if (
        not isinstance(path_value, str)
        or not path_value
        or category not in ALLOWED_CATEGORIES
        or not isinstance(timestamp, str)
        or not timestamp
        or not isinstance(size, (int, float))
        or isinstance(size, bool)
        or (isinstance(size, float) and not size.is_integer())
        or not isinstance(identity, dict)
    ):
        return None
    try:
        path = Path(path_value)
        parsed_timestamp = datetime.fromisoformat(timestamp)
        if parsed_timestamp.tzinfo is None:
            parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
        parsed_size = int(size)
        parsed_identity = {
            key: int(identity[key])
            for key in (
                "version",
                "device",
                "inode",
                "mode",
                "size",
                "mtime_ns",
                "ctime_ns",
            )
        }
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return None
    if parsed_size < 0 or parsed_identity["version"] != 1:
        return None
    return path, category, parsed_timestamp, parsed_size, parsed_identity


def _managed_marker_matches(path: Path, expected: str) -> bool:
    """Return true only for a regular, readable ownership marker."""
    marker = path / _EMPTY_DIR_OWNERSHIP_MARKER
    try:
        return (
            marker.is_file()
            and not marker.is_symlink()
            and marker.read_text(encoding="utf-8").strip() == expected
        )
    except (OSError, UnicodeError):
        return False


def _remove_path(path: Path) -> Optional[str]:
    """Remove one regular file or directory, returning an error if unsure."""
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return "not a regular file or directory"
    except OSError as exc:
        return str(exc)
    return None


def _remove_tracked_path(path: Path, identity: Dict[str, int]) -> Optional[str]:
    """Remove only the unchanged filesystem object authorized by tracking."""
    if _is_protected_tracked_path(path) or _is_protected_cron_path(path):
        return "protected durable path"
    if not stat.S_ISREG(identity.get("mode", 0)):
        return "tracked path is not a regular file"
    if not _identity_matches(path, identity):
        return "tracked filesystem identity changed"
    return _remove_path(path)


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
    expected_identity: Optional[Dict[str, int]] = None,
) -> bool:
    """Register a file for tracking. Returns True if newly tracked."""
    if category not in ALLOWED_CATEGORIES:
        _log(f"WARN: unknown category '{category}', using 'other'")
        category = "other"

    source_path = Path(path_str).expanduser()

    try:
        if source_path.is_symlink() or not source_path.is_file():
            _log(f"REJECT: {source_path} (not a regular file)")
            return False
        path = source_path.resolve()
    except (OSError, ValueError):
        _log(f"REJECT: {source_path} (filesystem identity unavailable)")
        return False

    if not path.exists():
        _log(f"SKIP: {path} (does not exist)")
        return False

    if not is_safe_path(path):
        _log(f"REJECT: {path} (outside HERMES_HOME)")
        return False

    if _is_protected_tracked_path(path) or _is_protected_cron_path(path):
        _log(f"REJECT: {path} (protected durable path)")
        return False

    identity = _capture_identity(path)
    if identity is None:
        _log(f"SKIP: {path} (filesystem identity unavailable)")
        return False
    if expected_identity is not None and identity != expected_identity:
        _log(f"REJECT: {path} (creation identity changed before tracking)")
        return False
    size = identity["size"] if path.is_file() else 0
    with _tracked_registry_lock():
        tracked = _load_tracked_unlocked()
        if any(
            isinstance(item, dict) and item.get("path") == str(path)
            for item in tracked
        ):
            return False
        tracked.append({
            "path": str(path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "size": size,
            "identity": identity,
        })
        _save_tracked_unlocked(tracked)
    _log(f"TRACKED: {path} ({category}, {fmt_size(size)})")
    if not silent:
        print(f"Tracked: {path} ({category}, {fmt_size(size)})")
    return True


def forget(path_str: str) -> int:
    """Remove a path from tracking without deleting the file."""
    p = Path(path_str).resolve()
    with _tracked_registry_lock():
        tracked = _load_tracked_unlocked()
        before = len(tracked)
        tracked = [i for i in tracked if Path(i["path"]).resolve() != p]
        removed = before - len(tracked)
        if removed:
            _save_tracked_unlocked(tracked)
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
        validated = _validated_tracked_item(item)
        if validated is None:
            # A malformed record is not deletion authorization.  Omit it
            # from the preview rather than aborting the whole report.
            continue
        p, cat, timestamp, size, identity = validated
        if (
            not p.exists()
            or not is_safe_path(p)
            or _is_protected_tracked_path(p)
            or _is_protected_cron_path(p)
            or not stat.S_ISREG(identity.get("mode", 0))
            or not _identity_matches(p, identity)
        ):
            continue
        age = (now - timestamp).days

        # Re-validate stale "cron-output" entries (fixes #37721).
        if cat == "cron-output":
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

def quick(paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Safe deterministic cleanup — no prompts.

    Returns: ``{"deleted": N, "artifacts": N, "empty_dirs": N,
               "freed": bytes, "errors": [str, ...]}``.
    """
    tracked = load_tracked()
    scoped_paths = None
    if paths is not None:
        scoped_paths = set()
        for path in paths:
            try:
                scoped_paths.add(str(Path(path).resolve()))
            except (OSError, TypeError, ValueError):
                continue
    now = datetime.now(timezone.utc)
    deleted = 0
    freed = 0
    new_tracked: List[Any] = []
    errors: List[str] = []
    artifacts = 0

    for item in tracked:
        validated = _validated_tracked_item(item)
        if validated is None:
            # Preserve malformed records and continue.  Hand-edited or
            # partially-written tracking state is never proof of ownership.
            errors.append(f"invalid tracked record: {item!r}")
            new_tracked.append(item)
            continue
        p, cat, timestamp, size, identity = validated

        if scoped_paths is not None and str(p.resolve()) not in scoped_paths:
            new_tracked.append(item)
            continue

        try:
            p.stat()
        except FileNotFoundError:
            _log(f"STALE: {p} (removed from tracking)")
            continue
        except (OSError, ValueError) as exc:
            _log(f"ERROR inspecting {p}: {exc}")
            errors.append(f"{p}: {exc}")
            new_tracked.append(item)
            continue

        # tracked.json is durable state, not proof that the current target is
        # still owned by Hermes.  A stale or hand-edited record must never make
        # quick() delete a path outside the active HERMES_HOME (or approved
        # /tmp/hermes-* scope).
        if not is_safe_path(p):
            _log(f"SKIP unsafe tracked path: {p}")
            errors.append(f"{p}: unsafe path")
            new_tracked.append(item)
            continue

        if _is_protected_tracked_path(p) or _is_protected_cron_path(p):
            _log(f"SKIP protected tracked path: {p}")
            errors.append(f"{p}: protected durable path")
            new_tracked.append(item)
            continue

        if not stat.S_ISREG(identity.get("mode", 0)):
            _log(f"SKIP non-file tracked path: {p}")
            errors.append(f"{p}: tracked path is not a regular file")
            new_tracked.append(item)
            continue

        if not _identity_matches(p, identity):
            _log(f"SKIP changed tracked identity: {p}")
            errors.append(f"{p}: tracked filesystem identity changed")
            new_tracked.append(item)
            continue

        age = (now - timestamp).days

        # ---- stale-state migration (fixes #37721) ----
        # Old tracked.json entries may carry a "cron-output" category for
        # paths that are NOT under cron/output/ (e.g. cron/jobs.json).
        # guess_category() was fixed in #34840, but existing entries are
        # never re-validated.  Re-classify here so stale entries for cron
        # control-plane state are not deleted.
        if cat == "cron-output":
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
            error = _remove_tracked_path(p, identity)
            if error is None:
                freed += size
                deleted += 1
                _log(f"DELETED: {p} ({cat}, {fmt_size(size)})")
            else:
                _log(f"ERROR deleting {p}: {error}")
                errors.append(f"{p}: {error}")
                new_tracked.append(item)
        else:
            new_tracked.append(item)

    # Scoped session cleanup is limited to the exact newly tracked paths. The
    # manual/global quick command additionally performs retention and empty-dir
    # sweeps under their separately marked ownership roots.
    if scoped_paths is not None:
        _commit_tracked_snapshot(tracked, new_tracked)
        _log(
            f"QUICK_SUMMARY: {deleted} files, 0 dirs, "
            f"{fmt_size(freed)}"
        )
        return {
            "deleted": deleted,
            "artifacts": 0,
            "empty_dirs": 0,
            "freed": freed,
            "errors": errors,
        }

    # Prune artifact files only inside explicitly marked session directories.
    # The marker is the durable ownership proof; directory names alone are not.
    hermes_home = get_hermes_home()
    for root_name, retention_days in _MANAGED_ARTIFACT_RETENTION_DAYS.items():
        artifact_root = hermes_home / root_name
        try:
            if artifact_root.is_symlink() or not artifact_root.is_dir():
                continue
            session_dirs = [
                p for p in artifact_root.iterdir()
                if p.is_dir() and not p.is_symlink()
            ]
        except OSError:
            continue
        for session_dir in session_dirs:
            try:
                if (
                    not _managed_marker_matches(session_dir, root_name)
                    or _has_active_maintenance_marker(session_dir)
                ):
                    continue
                for artifact in session_dir.iterdir():
                    if (
                        artifact.name in _MANAGED_ARTIFACT_SKIP_NAMES
                        or artifact.is_symlink()
                        or not artifact.is_file()
                    ):
                        continue
                    try:
                        age_seconds = now.timestamp() - artifact.stat().st_mtime
                        if age_seconds <= retention_days * 24 * 60 * 60:
                            continue
                        size = artifact.stat().st_size
                        artifact.unlink()
                        artifacts += 1
                        freed += size
                        _log(f"DELETED: {artifact} (managed artifact retention)")
                    except OSError as exc:
                        errors.append(f"{artifact}: {exc}")
            except OSError as exc:
                errors.append(f"{session_dir}: {exc}")

    # Remove empty dirs under HERMES_HOME, but never recurse into known
    # durable state trees.  Some installs place the Hermes checkout, venv,
    # and desktop build under HERMES_HOME; a full rglob over that tree can
    # stall the gateway event loop for minutes.
    empty_removed = 0
    sweep_stack: List[Tuple[Path, bool]] = []
    try:
        for top in hermes_home.iterdir():
            if (
                top.is_dir()
                and not top.is_symlink()
                and _is_owned_empty_dir_root(top)
            ):
                sweep_stack.append((top, False))
    except OSError:
        sweep_stack = []

    while sweep_stack:
        dirpath, visited = sweep_stack.pop()
        if visited:
            try:
                if not any(dirpath.iterdir()):
                    dirpath.rmdir()
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
                    and child.name not in _EMPTY_DIR_SWEEP_PRUNE_DIRS
                    and child.name not in _EMPTY_DIR_SWEEP_PROTECTED_NAMES
                ):
                    sweep_stack.append((child, False))
        except OSError:
            pass

    _commit_tracked_snapshot(tracked, new_tracked)
    _log(
        f"QUICK_SUMMARY: {deleted} files, {empty_removed} dirs, "
        f"{fmt_size(freed)}"
    )
    return {
        "deleted": deleted,
        "artifacts": artifacts,
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
        validated = _validated_tracked_item(item)
        if validated is None:
            continue
        p, cat, timestamp, size, identity = validated
        try:
            p.stat()
        except (FileNotFoundError, OSError, ValueError):
            continue
        if (
            not is_safe_path(p)
            or _is_protected_cron_path(p)
            or _is_protected_tracked_path(p)
            or not stat.S_ISREG(identity.get("mode", 0))
        ):
            continue
        age = (now - timestamp).days

        if cat == "research" and age > 30:
            research.append(item)
        elif cat == "chrome-profile" and age > 14:
            chrome.append(item)
        elif size > 500 * 1024 * 1024:
            large.append(item)

    research.sort(key=lambda x: x["timestamp"], reverse=True)
    old_research = research[10:]

    freed, count = 0, 0
    to_remove: List[Dict] = []

    for group in (old_research, chrome, large):
        for item in group:
            if confirm(item):
                validated = _validated_tracked_item(item)
                if validated is None:
                    continue
                p, _cat, _timestamp, size, identity = validated
                if (
                    not is_safe_path(p)
                    or _is_protected_cron_path(p)
                    or _is_protected_tracked_path(p)
                    or not stat.S_ISREG(identity.get("mode", 0))
                ):
                    continue
                error = _remove_tracked_path(p, identity)
                if error is None:
                    to_remove.append(item)
                    freed += size
                    count += 1
                    _log(
                        f"DELETED: {p} ({item['category']}, "
                        f"{fmt_size(size)})"
                    )
                else:
                    _log(f"ERROR deleting {p}: {error}")

    if to_remove:
        remove_paths = {i["path"] for i in to_remove}
        _commit_tracked_snapshot(
            tracked, [i for i in tracked if i["path"] not in remove_paths]
        )

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
