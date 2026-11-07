"""Task workspace lifecycle: scratch/dir/worktree resolution (incl. git worktree creation), post-completion cleanup with containment guards, worker tmux teardown and the first-use scratch-workspace tip.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import os
import json
import re
import shutil
import sqlite3
import stat
import subprocess
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING
import contextlib

from hermes_cli.worktree_ops import release_lsp_clients

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_REMOVABLE_KINDS = ("scratch", "worktree")


def _path_key(path: Path | str | None) -> str:
    """Unicode-form-insensitive identity for a filesystem path.

    macOS hands back DECOMPOSED path strings (NFD: ``o`` + U+0308) for names the
    user typed in composed form (NFC: ``ö``) — a OneDrive/FileProvider path like
    ``OneDrive-Persönlich`` round-trips through ``git rev-parse --show-toplevel``
    as NFD while the DB row holds NFC. Raw ``Path`` equality then reports a real
    repo root as "not a repo" purely on Unicode form, so every path identity
    check here goes through this key.
    """
    return unicodedata.normalize("NFC", str(path)) if path is not None else ""

# Statuses after which a task no longer needs a workspace: a child stops
# needing its parent's handoff artifacts, and a sharer stops holding a dir.
_TERMINAL_STATUSES_SQL = "('done', 'archived', 'failed', 'cancelled')"

_ACTIVE_CHILDREN_SQL = (
    "SELECT 1 FROM task_links l "
    "JOIN tasks t ON t.id = l.child_id "
    f"WHERE l.parent_id = ? AND t.status NOT IN {_TERMINAL_STATUSES_SQL} "
    "LIMIT 1"
)

_WORKSPACE_ROW_SQL = "SELECT workspace_kind, workspace_path, branch_name FROM tasks WHERE id = ?"


def _git(repo_root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
    """``git -C repo_root args``; never raises on a non-zero exit.

    :func:`noninteractive_repo_git_env` (GHSA-7x36-8jrh-v4pw): the dispatcher runs ``worktree add``
    unattended, which executes the repo's hooks, ``core.fsmonitor`` and smudge filters.
    """
    from hermes_cli._subprocess_compat import FILTER_DISCOVERY_FAILED, noninteractive_repo_git_env
    env = noninteractive_repo_git_env(repo_root)
    if env is None:
        return subprocess.CompletedProcess(["git", "-C", str(repo_root), *args], 1, "", FILTER_DISCOVERY_FAILED)
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def _has_active_children(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(_ACTIVE_CHILDREN_SQL, (task_id,)).fetchone() is not None


_TERMINAL_TASK_SQL = (
    f"SELECT 1 FROM tasks WHERE id = ? AND status IN {_TERMINAL_STATUSES_SQL}"
)


def _is_terminal_task(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(_TERMINAL_TASK_SQL, (task_id,)).fetchone() is not None


_OTHER_LIVE_PATHS_SQL = (
    "SELECT workspace_path FROM tasks "
    "WHERE id != ? AND workspace_path IS NOT NULL "
    f"AND status NOT IN {_TERMINAL_STATUSES_SQL}"
)
# Sibling boards mint their own ids, so the id being cleaned up here can
# name a different live task there. Do not exclude it.
_ANY_LIVE_PATHS_SQL = (
    "SELECT workspace_path FROM tasks "
    "WHERE workspace_path IS NOT NULL "
    f"AND status NOT IN {_TERMINAL_STATUSES_SQL}"
)


def _conn_uses_path(
    conn: sqlite3.Connection, task_id: str, key: str, *, exclude_task_id: bool = True
) -> bool:
    if exclude_task_id:
        rows = conn.execute(_OTHER_LIVE_PATHS_SQL, (task_id,)).fetchall()
    else:
        rows = conn.execute(_ANY_LIVE_PATHS_SQL).fetchall()
    for row in rows:
        other = row["workspace_path"]
        if not other:
            continue
        try:
            other_key = _path_key(Path(other).expanduser().resolve(strict=False))
        except OSError:
            continue
        if other_key == key:
            return True
    return False


def _sibling_board_db_files(conn: sqlite3.Connection) -> list[Path]:
    """Every other board's ``kanban.db``. Raises ``OSError`` when the set is unknown.

    ``kanban_db_path`` follows ``HERMES_KANBAN_DB`` and would collapse every
    slug onto the pinned file, so the scan uses the on-disk layout: the
    default board at ``<home>/kanban.db`` and named boards at
    ``<home>/kanban/boards/<slug>/kanban.db``.
    """
    from hermes_cli.kanban_db_connect import _main_db_file  # late: import cycle

    current_file = _main_db_file(conn)
    current = Path(current_file).resolve() if current_file else None
    candidates = [_kb.kanban_home() / "kanban.db"]
    root = _kb.boards_root()
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                candidates.append(child / "kanban.db")
    found: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if current is not None and resolved == current:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def _other_board_uses_path(db_file: Path, task_id: str, key: str) -> bool:
    uri = db_file.as_uri() + "?mode=ro"
    other = sqlite3.connect(uri, uri=True, timeout=1.0)
    try:
        other.row_factory = sqlite3.Row
        return _conn_uses_path(other, task_id, key, exclude_task_id=False)
    finally:
        other.close()


def _workspace_in_use_by_other(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> Optional[str]:
    """Why *path* must be kept: ``"shared"``, ``"unknown"``, or None when free.

    ``gc``, completion and deferred parent cleanup used to ``rmtree`` a shared
    ``workspace_path`` as soon as one of its tasks went terminal. Compare the
    resolved path: a row may store ``~`` or a symlinked spelling.

    The connection covers one board. A ready task on a named board can point
    at the same directory. If that set of databases cannot be read, refuse
    the delete (``"unknown"``) and log the board DB that failed, so one broken
    sibling DB is not reported as a live sharer.
    """
    try:
        key = _path_key(Path(path).expanduser().resolve(strict=False))
    except OSError as exc:
        _kb._log.warning("Cannot resolve workspace %s for task %s: %s", path, task_id, exc)
        return "unknown"
    if not key:
        return None
    try:
        if _conn_uses_path(conn, task_id, key):
            return "shared"
    except sqlite3.Error as exc:
        _kb._log.warning("Cannot read live workspaces for task %s: %s", task_id, exc)
        return "unknown"
    try:
        siblings = _sibling_board_db_files(conn)
    except OSError as exc:
        _kb._log.warning("Cannot list kanban boards for task %s: %s", task_id, exc)
        return "unknown"
    for db_file in siblings:
        try:
            if _other_board_uses_path(db_file, task_id, key):
                return "shared"
        except (OSError, sqlite3.Error) as exc:
            _kb._log.warning(
                "Cannot read board db %s for task %s: %s", db_file, task_id, exc,
            )
            return "unknown"
    return None


def _defer_shared_workspace_cleanup(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> bool:
    """Skip removal and record why, when *path* may still be used by a live task."""
    reason = _workspace_in_use_by_other(conn, task_id, path)
    if reason is None:
        return False
    if reason == "shared":
        _kb._log.warning(
            "Deferring workspace cleanup for task %s: %s is still used by "
            "another non-terminal task",
            task_id, path,
        )
    else:
        _kb._log.warning(
            "Deferring workspace cleanup for task %s: cannot tell whether "
            "another task still uses %s",
            task_id, path,
        )
    try:
        _kb._append_event(
            conn, task_id, "workspace_cleanup_deferred_shared",
            {"path": str(path), "reason": reason},
        )
    except sqlite3.Error:
        pass
    return True


def _defer_shared_worktree_cleanup(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> bool:
    """Defer removing a linked worktree another live task may still use.

    Only a real linked worktree is ever removed, so the sharing scan runs just
    for those.
    """
    wt = Path(path).expanduser()
    return (
        wt.is_dir()
        and _is_linked_worktree_checkout(wt)
        and _defer_shared_workspace_cleanup(conn, task_id, path)
    )


def _lexical_path(path: Path | str) -> Path:
    """Absolute, ``..``-collapsed, NFC form of *path* WITHOUT following symlinks."""
    return Path(_path_key(os.path.abspath(path)))


def _managed_scratch_path_info(p: Path) -> tuple[bool, Optional[str]]:
    """Return whether *p* is managed scratch storage and the matching board.

    *p* must be strictly below a managed root both after resolving symlinks
    AND lexically (as spelled, without resolving). Resolved containment alone
    is not enough: when a root is itself a symlink to a broad directory
    (relocated storage, or a planted link), every path inside the link target
    would resolve "under" the root, so a scratch task naming such a path
    directly would get it rmtree'd. Tasks created through the root are spelled
    through it, so the lexical check keeps them managed. A root's lexical form
    is accepted both as configured and with its anchor (kanban home, or the
    override's parent) resolved, so a process spelling a symlinked home by its
    real path still matches; the managed ``kanban/.../workspaces`` components
    themselves are never resolved for the lexical check.
    """
    try:
        p_abs = p.resolve(strict=False)
    except OSError:
        return False, None
    p_lex = _lexical_path(p)
    # (resolved root, lexical spellings of the root, board)
    roots: list[tuple[Path, tuple[Path, ...], Optional[str]]] = []

    def _add_root(
        anchor: Path, anchor_real: Path, parts: tuple[str, ...], board: Optional[str]
    ) -> None:
        root = anchor.joinpath(*parts)
        with contextlib.suppress(OSError):
            roots.append((
                root.resolve(strict=False),
                (_lexical_path(root), _lexical_path(anchor_real.joinpath(*parts))),
                board,
            ))

    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        override_root = Path(override).expanduser()
        with contextlib.suppress(OSError):
            override_parent = override_root.parent
            _add_root(
                override_parent,
                override_parent.resolve(strict=False),
                (override_root.name,),
                None,
            )
    try:
        home = _kb.kanban_home()
        # Resolve the shared anchor once, not once per board root.
        home_real = home.resolve(strict=False)
    except OSError:
        home = None
    if home is not None:
        _add_root(home, home_real, ("kanban", "workspaces"), _kb.DEFAULT_BOARD)
        entries: list[Path] = []
        with contextlib.suppress(OSError):
            entries = list((home / "kanban" / "boards").resolve(strict=False).iterdir())
        for entry in entries:
            with contextlib.suppress(OSError):
                if entry.is_dir():
                    _add_root(
                        home, home_real, ("kanban", "boards", entry.name, "workspaces"), entry.name
                    )
    for root, lexical_roots, board in roots:
        if p_abs == root:
            continue
        try:
            if p_abs.is_relative_to(root) and any(
                p_lex != lex and p_lex.is_relative_to(lex) for lex in lexical_roots
            ):
                return True, board
        except ValueError:
            continue
    return False, None


def _scratch_workspace(conn: sqlite3.Connection, task_id: str) -> Optional[Path]:
    """Expanded ``workspace_path`` when the task uses a scratch workspace, else ``None``."""
    row = conn.execute(
        "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
        return None
    return Path(row["workspace_path"]).expanduser()


def _is_managed_scratch_path(p: Path) -> bool:
    """True iff *p* is a STRICT descendant of a kanban-managed ``workspaces/``
    root (``HERMES_KANBAN_WORKSPACES_ROOT``, ``<kanban_home>/kanban/workspaces``,
    or ``<kanban_home>/kanban/boards/<slug>/workspaces``). A path equal to a
    root is not managed (deleting it would wipe every task's scratch dir);
    ``<kanban_home>/kanban``, ``.../logs`` and ``.../boards/<slug>`` hold
    Hermes' own DB and metadata. :func:`_cleanup_workspace` refuses
    ``rmtree`` outside managed storage — a board ``default_workdir`` on a real
    source tree paired with ``workspace_kind='scratch'`` would otherwise make
    task completion delete user data.

    See #28818.
    """
    return _managed_scratch_path_info(p)[0]


def _cleanup_workspace(conn: sqlite3.Connection, task_id: str) -> None:
    """Remove a task's scratch workspace dir and kill its stale tmux session.
    Called from :func:`complete_task` after the transaction commits; best-effort
    so cleanup never blocks completion. ``scratch`` is removed; ``worktree``
    only when provably free of work (clean tree, every commit reachable from a
    remote-tracking ref); ``dir`` is intentionally preserved."""
    try:
        row = conn.execute(_WORKSPACE_ROW_SQL, (task_id,)).fetchone()
        if not row:
            return
        kind: Optional[str] = row["workspace_kind"]
        path: Optional[str] = row["workspace_path"]
        if kind not in _REMOVABLE_KINDS or not path:
            # Not removable itself, but completing may still unblock a deferred
            # parent scratch cleanup (e.g. a 'dir' child of a scratch parent).
            # See #33774.
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        # Defer while any child is not yet terminal so it can still read
        # handoff artifacts from this workspace.
        if _has_active_children(conn, task_id):
            _kb._log.debug(
                "Deferring %s workspace cleanup for task %s: "
                "active children still need workspace at %s",
                kind, task_id, path,
            )
            return
        # Kill the (dead) tmux worker session BEFORE removing a worktree so a
        # lingering worker never has its cwd deleted from under it.
        if kind == "worktree":
            _cleanup_worker_tmux(conn, task_id)
            if not _defer_shared_worktree_cleanup(conn, task_id, path):
                _cleanup_worktree_workspace(task_id, path, row["branch_name"])
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        wp = Path(path)
        if wp.is_dir():
            # Containment guard: a board's ``default_workdir`` can pair
            # ``workspace_kind='scratch'`` with a user path pointing at a real
            # source tree; without this, completion would rmtree the user's data.
            # See #28818. Containment runs first so an unmanaged dir that is
            # never removed does not pay for (or log) the shared-use scan.
            if _is_managed_scratch_path(wp):
                if not _defer_shared_workspace_cleanup(conn, task_id, path):
                    release_lsp_clients(str(wp))
                    shutil.rmtree(wp, ignore_errors=True)
                    _kb._log.debug("Removed scratch workspace: %s", wp)
            else:
                _kb._log.warning(
                    "Refusing to remove out-of-scratch workspace for task %s: %s "
                    "(workspace_kind='scratch' but path is outside any "
                    "kanban-managed workspaces root)",
                    task_id, wp,
                )
        # Kill the owning worker's tmux session if it is now dead, then let any
        # parent whose children are all done run its deferred cleanup.
        _cleanup_worker_tmux(conn, task_id)
        # After cleaning up this task's workspace, check if any parent tasks now have all children done —
        # their deferred cleanup can proceed (#33774).
        _try_cleanup_parent_workspaces(conn, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _ref_lock_owner_alive(payload: dict) -> bool:
    pid, started_at = payload.get("pid"), payload.get("started_at")
    if not isinstance(pid, int) or not isinstance(started_at, int):
        return True
    try:
        from gateway.status import _pid_exists, get_process_start_time

        current = get_process_start_time(pid)
    except (ImportError, OSError, ValueError):
        return True
    if current is not None:
        return current == started_at
    return _pid_exists(pid)


_REF_RECOVERY_GUARD = object()
_REF_RECOVERY_LOCAL = threading.local()


@contextlib.contextmanager
def _ref_recovery_guard(common: Path):
    from hermes_cli.active_sessions import _FileLock

    depth = int(getattr(_REF_RECOVERY_LOCAL, "depth", 0))
    if depth:
        _REF_RECOVERY_LOCAL.depth = depth + 1
        try:
            yield _REF_RECOVERY_GUARD
        finally:
            _REF_RECOVERY_LOCAL.depth = depth
        return
    guard_path = common.resolve(strict=True) / "hermes-kanban-ref-recovery.lock"
    with _FileLock(guard_path):
        _REF_RECOVERY_LOCAL.depth = 1
        try:
            yield _REF_RECOVERY_GUARD
        finally:
            _REF_RECOVERY_LOCAL.depth = 0


def _remove_abandoned_ref_lock_under_guard(lock_path: Path, guard: object) -> bool:
    """Remove a dead Hermes lock while the persistent recovery guard is held."""
    if guard is not _REF_RECOVERY_GUARD or not getattr(_REF_RECOVERY_LOCAL, "depth", 0):
        return False
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags)
    except (FileNotFoundError, OSError):
        return False
    try:
        identity = os.fstat(fd)
        if not stat.S_ISREG(identity.st_mode) or identity.st_size > 4096:
            return False
        raw = os.read(fd, 4097)
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("owner") != "hermes-kanban-pr-cleanup":
            return False
        if _ref_lock_owner_alive(payload):
            return False
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    finally:
        os.close(fd)
    try:
        current = lock_path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            return False
        lock_path.unlink()
        return True
    except OSError:
        return False


def _unlink_matching_inode(path: Path, identity: os.stat_result) -> None:
    try:
        current = path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            path.unlink()
    except FileNotFoundError:
        pass


def _validated_branch_ref_paths(
    common: Path, branch: str
) -> tuple[Path, str, Path, Path]:
    repo_root = common.parent
    if _git(repo_root, "check-ref-format", "--branch", branch, timeout=10).returncode != 0:
        raise RuntimeError("invalid branch for cleanup ref lock")
    ref = f"refs/heads/{branch}"
    ref_path = common.joinpath(*ref.split("/"))
    lock_path = ref_path.with_name(ref_path.name + ".lock")
    common_real = common.resolve(strict=True)
    parent = lock_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.resolve(strict=True).is_relative_to(common_real):
        raise RuntimeError("branch ref lock escapes git common directory")
    candidates = (common / "refs", common / "refs" / "heads", parent, *parent.parents)
    for candidate in candidates:
        if candidate == common:
            break
        if candidate.is_symlink():
            raise RuntimeError("branch ref lock traverses a symlink")
    return repo_root, ref, lock_path, parent


def _prepare_ref_lock_temp(parent: Path, lock_path: Path, payload: bytes):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    temp_path = parent / f".{lock_path.name}.hermes-{uuid.uuid4().hex}.tmp"
    fd = os.open(temp_path, flags, 0o600)
    identity = os.fstat(fd)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short write while preparing branch ref lock")
            remaining = remaining[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        _unlink_matching_inode(temp_path, identity)
        raise
    os.close(fd)
    return temp_path, identity


def _publish_prepared_ref_lock(
    temp_path: Path,
    lock_path: Path,
    identity: os.stat_result,
    publication: list[bool],
) -> None:
    for attempt in range(2):
        try:
            os.link(temp_path, lock_path, follow_symlinks=False)
            publication[0] = True
            break
        except FileExistsError:
            if attempt or not _remove_abandoned_ref_lock_under_guard(
                lock_path, _REF_RECOVERY_GUARD
            ):
                raise RuntimeError(
                    "branch ref is locked by Git or a live cleanup owner"
                ) from None
    else:
        raise RuntimeError("could not acquire branch ref lock")
    current = lock_path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        raise RuntimeError("branch ref lock inode changed during publication")


@contextlib.contextmanager
def _branch_ref_write_lock_guarded(common: Path, branch: str, expected_head: str):
    """Inner ref lock protocol; caller holds the persistent recovery guard."""
    repo_root, ref, lock_path, parent = _validated_branch_ref_paths(common, branch)
    from gateway.status import get_process_start_time

    started_at = get_process_start_time(os.getpid())
    if started_at is None:
        raise RuntimeError("process identity unavailable for crash-recoverable ref lock")
    payload = json.dumps(
        {
            "owner": "hermes-kanban-pr-cleanup",
            "pid": os.getpid(),
            "started_at": started_at,
            "branch": branch,
            "expected_head": expected_head,
            "created_at": int(time.time()),
        },
        sort_keys=True,
    ).encode("utf-8")
    temp_path, identity = _prepare_ref_lock_temp(parent, lock_path, payload)
    publication = [False]
    try:
        _publish_prepared_ref_lock(temp_path, lock_path, identity, publication)
        _unlink_matching_inode(temp_path, identity)
        actual = (_kb._git_out(repo_root, "rev-parse", ref) or "").strip()
        if actual != expected_head:
            raise RuntimeError("branch advanced before cleanup ref lock")
        yield
    finally:
        if publication[0]:
            _unlink_matching_inode(lock_path, identity)
        _unlink_matching_inode(temp_path, identity)


@contextlib.contextmanager
def _branch_ref_write_lock(common: Path, branch: str, expected_head: str):
    """Serialize stale recovery/publication, then hold Git's ref lock."""
    common_real = common.resolve(strict=True)
    with _ref_recovery_guard(common_real):
        with _branch_ref_write_lock_guarded(common_real, branch, expected_head):
            yield


def _cleanup_worktree_workspace_locked(
    task_id: str,
    path: str,
    branch_name: Optional[str] = None,
    *,
    merged_pr_head_oid: Optional[str] = None,
) -> bool:
    """Remove a finished task's linked git worktree when it holds no work.

    Normally removal requires a clean tree and every commit reachable from a
    remote-tracking ref. An exact GitHub merged-PR head OID may substitute for
    reachability after GitHub deleted the head branch or squash-merged it. Any
    doubt preserves the tree. Returns True only after Git removed the worktree.
    """
    from hermes_cli.worktree_ops import _worktree_has_unpushed_commits, _worktree_is_dirty

    try:
        wp = Path(path).expanduser()
        if not wp.is_dir():
            return False
        common = _git_common_dir(wp)
        if common is None or common.name != ".git":
            return False
        repo_root = common.parent
        if _path_key(wp.resolve(strict=False)) == _path_key(repo_root.resolve(strict=False)):
            return False
        expected_head = (merged_pr_head_oid or "").strip()
        current_head = (_kb._git_out(wp, "rev-parse", "HEAD") or "").strip()
        branch = (branch_name or "").strip() or f"wt/{task_id}"
        if expected_head and (
            current_head != expected_head or _git_current_branch(wp) != branch
        ):
            return False
        merged_pr_proves_head = bool(expected_head)
        if _worktree_is_dirty(str(wp), str(repo_root)) or (
            _worktree_has_unpushed_commits(str(wp)) and not merged_pr_proves_head
        ):
            _kb._log.info(
                "Preserving worktree for task %s: dirty or unpushed work at %s",
                task_id, wp,
            )
            return False
        worktree_path = wp.resolve(strict=False)
        try:
            cwd = Path.cwd().resolve(strict=False)
        except OSError:
            cwd = None
        if cwd is None or cwd == worktree_path or cwd.is_relative_to(worktree_path):
            try:
                os.chdir(repo_root)
            except OSError as exc:
                _kb._log.warning(
                    "Preserving worktree for task %s: cannot leave %s for %s: %s",
                    task_id, cwd or "<deleted cwd>", repo_root, exc,
                )
                return False
        # No --force: Git re-verifies dirtiness at removal time. Re-read the
        # proof-bound identity immediately before mutation as well. A standard
        # Git ref lock then prevents a clean concurrent commit from advancing
        # the branch between this check and ``worktree remove``.
        if expected_head and (
            (_kb._git_out(wp, "rev-parse", "HEAD") or "").strip() != expected_head
            or _git_current_branch(wp) != branch
        ):
            return False
        ref_lock = (
            _branch_ref_write_lock(common, branch, expected_head)
            if expected_head else contextlib.nullcontext()
        )
        with ref_lock:
            release_lsp_clients(str(worktree_path))
            result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
            if result.returncode != 0:
                time.sleep(0.1)
                result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
        if result.returncode != 0:
            _kb._log.warning(
                "git worktree remove failed for task %s at %s: %s",
                task_id, wp, (result.stderr or result.stdout or "").strip(),
            )
            return False
        _kb._log.debug("Removed worktree workspace: %s", wp)
        branch = (branch_name or "").strip() or f"wt/{task_id}"
        if branch.startswith("wt/"):
            if expected_head:
                # Compare-and-delete is atomic. A branch advanced after the PR
                # proof survives even though its obsolete checkout was removed.
                _git(
                    repo_root,
                    "update-ref",
                    "-d",
                    f"refs/heads/{branch}",
                    expected_head,
                    timeout=30,
                )
            else:
                _git(repo_root, "branch", "-D", branch, timeout=30)
        return True
    except Exception:  # health: allow BLE001 -- destructive cleanup must preserve the tree on any Git/filesystem fault
        _kb._log.exception("Preserving worktree for task %s after cleanup fault", task_id)
        return False


def _cleanup_worktree_workspace(
    task_id: str,
    path: str,
    branch_name: Optional[str] = None,
    *,
    merged_pr_head_oid: Optional[str] = None,
) -> bool:
    """Lock the repository lifecycle before removing a linked worktree."""
    try:
        common = _git_common_dir(Path(path).expanduser())
        if common is None or common.name != ".git":
            return False
        with _worktree_lifecycle_lock(common):
            return _cleanup_worktree_workspace_locked(
                task_id,
                path,
                branch_name,
                merged_pr_head_oid=merged_pr_head_oid,
            )
    except Exception:  # health: allow BLE001 -- lifecycle lock uncertainty must preserve the worktree
        _kb._log.exception("Preserving worktree for task %s after lifecycle-lock fault", task_id)
        return False


_MERGED_PR_PROBE_EVENT = "workspace_cleanup_pr_probe"
_MERGED_PR_CLEANUP_EVENT = "workspace_cleanup_pr_merged"


def _view_pr(pr_url: str, assignee: Optional[str]) -> Optional[dict]:
    """Exact durable-contract PR identity, using the assignee's GitHub login."""
    try:
        from hermes_cli.kanban_pr_acceptance import _PR, _api, _assignee_profile_home

        match = _PR.fullmatch(pr_url)
        if match is None:
            return None
        repo, number = match[1], int(match[2])
        owner, name = repo.split("/", 1)
        query = (
            f'{{repository(owner:{json.dumps(owner)},name:{json.dumps(name)})'
            f'{{pullRequest(number:{number}){{number url state mergedAt mergeable mergeStateStatus '
            'headRefName headRefOid headRepository{nameWithOwner}}}}}}'
        )
        repository = _api(
            "graphql",
            query=query,
            profile_home=_assignee_profile_home(assignee),
            timeout=5,
        )["data"]["repository"]
        pr = repository["pullRequest"] if repository else None
        return pr if isinstance(pr, dict) else None
    except Exception as exc:  # health: allow BLE001 -- remote/API uncertainty is a fail-closed preservation boundary
        _kb._log.warning("Cannot query PR state for task contract %s: %s", pr_url, exc)
        return None


_PR_RECONCILE_CLAIM_TTL = 90
_PR_WORKER_GRACE_SECONDS = 120
_PR_INTENT_EVENT = "workspace_cleanup_pr_intent"
_PR_ALIGNMENT_EVENT = "pr_alignment_scheduled"


def _terminal_worker_settled(conn: sqlite3.Connection, task_id: str, now: int) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n, "
        "SUM(CASE WHEN ended_at IS NULL OR worker_pid IS NOT NULL THEN 1 ELSE 0 END) AS unsettled, "
        "MAX(ended_at) AS last_ended FROM task_runs WHERE task_id=?",
        (task_id,),
    ).fetchone()
    return bool(
        row
        and int(row["n"] or 0) > 0
        and int(row["unsettled"] or 0) == 0
        and row["last_ended"] is not None
        and int(row["last_ended"]) <= now - _PR_WORKER_GRACE_SECONDS
    )


def _conn_path_workers_settled(
    conn: sqlite3.Connection, path_key: str, now: int
) -> bool:
    rows = conn.execute(
        "SELECT id, status, workspace_path FROM tasks WHERE workspace_path IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            other_key = _path_key(Path(row["workspace_path"]).expanduser().resolve(strict=False))
        except OSError:
            return False
        if other_key != path_key:
            continue
        if row["status"] not in {"done", "archived", "failed", "cancelled"}:
            return False
        if not _terminal_worker_settled(conn, row["id"], now):
            return False
    return True


def _all_path_workers_settled(
    conn: sqlite3.Connection, path: str, now: int
) -> bool:
    """Every same- or sibling-board owner is terminal and its worker is gone."""
    try:
        key = _path_key(Path(path).expanduser().resolve(strict=False))
        if not key or not _conn_path_workers_settled(conn, key, now):
            return False
        for db_file in _sibling_board_db_files(conn):
            uri = db_file.as_uri() + "?mode=ro"
            sibling = sqlite3.connect(uri, uri=True, timeout=1.0)
            try:
                sibling.row_factory = sqlite3.Row
                if not _conn_path_workers_settled(sibling, key, now):
                    return False
            finally:
                sibling.close()
        return True
    except (OSError, sqlite3.Error):
        return False


@contextlib.contextmanager
def _all_board_write_fences(conn: sqlite3.Connection):
    """Freeze topology, then fence every board in canonical path order."""
    from hermes_cli.kanban_board_topology import board_topology_lock
    from hermes_cli.kanban_db_connect import _main_db_file

    with board_topology_lock(_kb.kanban_home(), purpose="snapshot"):
        current_file = _main_db_file(conn)
        if current_file is None:
            raise sqlite3.OperationalError("current board database path unavailable")
        current_file = Path(current_file).resolve(strict=True)
        db_files = {current_file}
        db_files.update(path.resolve(strict=True) for path in _sibling_board_db_files(conn))
        fenced: list[tuple[Path, sqlite3.Connection]] = []
        try:
            for db_file in sorted(db_files, key=lambda item: str(item)):
                handle = sqlite3.connect(db_file, timeout=1.0, isolation_level=None)
                handle.row_factory = sqlite3.Row
                handle.execute("PRAGMA busy_timeout=1000")
                handle.execute("BEGIN IMMEDIATE")
                fenced.append((db_file, handle))
            current = next(handle for path, handle in fenced if path == current_file)
            siblings = [handle for path, handle in fenced if path != current_file]
            yield current, siblings
        finally:
            for _path, handle in reversed(fenced):
                with contextlib.suppress(sqlite3.Error):
                    handle.execute("ROLLBACK")
                handle.close()


def _fenced_path_workers_settled(
    conn: sqlite3.Connection,
    siblings: list[sqlite3.Connection],
    path: str,
    now: int,
) -> bool:
    try:
        key = _path_key(Path(path).expanduser().resolve(strict=False))
        return bool(
            key
            and _conn_path_workers_settled(conn, key, now)
            and all(_conn_path_workers_settled(sibling, key, now) for sibling in siblings)
        )
    except (OSError, sqlite3.Error):
        return False


def _origin_github_repo(path: Path) -> Optional[str]:
    remote = (_kb._git_out(path, "remote", "get-url", "origin") or "").strip()
    patterns = (
        r"https?://github\.com/([^/]+/[^/]+?)(?:\.git)?$",
        r"git@github\.com:([^/]+/[^/]+?)(?:\.git)?$",
        r"ssh://git@github\.com/([^/]+/[^/]+?)(?:\.git)?$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, remote, re.IGNORECASE)
        if match:
            return match[1]
    return None


def _claim_pr_reconciliation(
    conn: sqlite3.Connection, task_id: str, now: int
) -> Optional[str]:
    token = f"pr-reconcile:{uuid.uuid4().hex}"
    try:
        with _kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET claim_lock=?, claim_expires=? "
                f"WHERE id=? AND status IN {_TERMINAL_STATUSES_SQL} "
                "AND (claim_lock IS NULL OR claim_expires IS NULL OR claim_expires < ?)",
                (token, now + _PR_RECONCILE_CLAIM_TTL, task_id, now),
            )
            return token if cur.rowcount == 1 else None
    except sqlite3.Error:
        return None


def _finish_pr_probe_txn(
    conn: sqlite3.Connection,
    task_id: str,
    token: str,
    result: str,
    extra: Optional[dict] = None,
) -> None:
    _kb._append_event(
        conn, task_id, _MERGED_PR_PROBE_EVENT, {"result": result, **(extra or {})}
    )
    conn.execute(
        "UPDATE tasks SET claim_lock=NULL, claim_expires=NULL "
        "WHERE id=? AND claim_lock=?",
        (task_id, token),
    )


def _finish_pr_probe(
    conn: sqlite3.Connection,
    task_id: str,
    token: str,
    result: str,
    extra: Optional[dict] = None,
) -> None:
    try:
        with _kb.write_txn(conn):
            _finish_pr_probe_txn(conn, task_id, token, result, extra)
    except sqlite3.Error:
        pass


def _recover_finished_pr_cleanup(
    conn: sqlite3.Connection, task_id: str, path: str
) -> bool:
    """Settle DB state only after proving Git no longer registers the worktree."""
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, _PR_INTENT_EVENT),
    ).fetchone()
    if not row:
        return False
    try:
        payload = json.loads(row["payload"] or "{}")
        repo_root = Path(payload["repo_root"]).expanduser().resolve(strict=True)
        recorded_common = Path(payload["common_dir"]).expanduser().resolve(strict=True)
        recorded_head = payload["head"]
        branch = payload["branch"]
    except (KeyError, TypeError, ValueError, OSError):
        return False
    if payload.get("path") != path:
        return False
    actual_common = _git_common_dir(repo_root)
    if actual_common is None:
        return False
    try:
        if actual_common.resolve(strict=True) != recorded_common:
            return False
        listing = _git(repo_root, "worktree", "list", "--porcelain", "-z", timeout=30)
        if listing.returncode != 0:
            return False
        registered = {
            _path_key(Path(field[len("worktree "):]).resolve(strict=False))
            for field in (listing.stdout or "").split("\0")
            if field.startswith("worktree ")
        }
        if _path_key(Path(path).expanduser().resolve(strict=False)) in registered:
            return False
        branch_head = (_kb._git_out(repo_root, "rev-parse", f"refs/heads/{branch}") or "").strip()
        if branch_head and branch_head != recorded_head:
            return False
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    try:
        with _kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET workspace_path=NULL, claim_lock=NULL, claim_expires=NULL "
                f"WHERE id=? AND workspace_path=? AND status IN {_TERMINAL_STATUSES_SQL}",
                (task_id, path),
            )
            if cur.rowcount != 1:
                return False
            _kb._append_event(
                conn, task_id, _MERGED_PR_CLEANUP_EVENT,
                {**payload, "recovered": True},
            )
        return True
    except sqlite3.Error:
        return False


def _existing_alignment_task(
    conn: sqlite3.Connection, idempotency_key: str
) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key=? LIMIT 1",
        (idempotency_key,),
    ).fetchone()
    if row:
        return row["id"]
    for db_file in _sibling_board_db_files(conn):
        uri = db_file.as_uri() + "?mode=ro"
        sibling = sqlite3.connect(uri, uri=True, timeout=1.0)
        try:
            sibling.row_factory = sqlite3.Row
            row = sibling.execute(
                "SELECT id FROM tasks WHERE idempotency_key=? LIMIT 1",
                (idempotency_key,),
            ).fetchone()
            if row:
                return row["id"]
        finally:
            sibling.close()
    return None


def _schedule_pr_alignment(
    conn: sqlite3.Connection, row: sqlite3.Row, pr: dict, head: str
) -> Optional[str]:
    pr_url = row["completion_contract"]
    number = pr.get("number")
    idempotency_key = f"pr-alignment:{pr_url}:{head}"
    try:
        common = _git_common_dir(Path(row["workspace_path"]).expanduser())
        if common is None or common.name != ".git":
            return None
        # Every controller that can create this special task participates in
        # the repository lifecycle lock. The cross-board scan + insert is thus
        # atomic for PR-alignment writers even though ordinary task keys are not
        # globally unique in SQLite.
        with _worktree_lifecycle_lock(common):
            existing = _existing_alignment_task(conn, idempotency_key)
            if existing:
                return existing
            child_id = _kb.create_task(
                conn,
                title=f"Align PR #{number} with its base branch",
                body=(
                    f"Resolve merge conflicts for {pr_url}. Work only in the existing branch "
                    f"{row['branch_name']!r} at {row['workspace_path']}. Fetch the base branch, "
                    "align the PR branch, preserve intended changes, run relevant gates, push the "
                    "updated branch, and complete only with current PR acceptance evidence."
                ),
                assignee=row["assignee"],
                created_by="kanban-pr-controller",
                workspace_kind="worktree",
                workspace_path=row["workspace_path"],
                branch_name=row["branch_name"],
                priority=int(row["priority"] or 0),
                idempotency_key=idempotency_key,
                creator_task_id=row["id"],
                skills=("github-pr-workflow",),
                project_id=row["project_id"],
                project_source_task_id=row["id"],
                completion_contract=pr_url,
            )
            with _kb.write_txn(conn):
                _kb._append_event(
                    conn,
                    row["id"],
                    _PR_ALIGNMENT_EVENT,
                    {"alignment_task_id": child_id, "pr_url": pr_url, "head": head},
                )
            return child_id
    except Exception as exc:  # health: allow BLE001 -- alignment creation fails closed and is retried
        _kb._log.exception("Cannot schedule PR alignment for task %s: %s", row["id"], exc)
        return None


def _record_pr_probe(
    conn: sqlite3.Connection, task_id: str, result: str
) -> None:
    try:
        with _kb.write_txn(conn):
            _kb._append_event(conn, task_id, _MERGED_PR_PROBE_EVENT, {"result": result})
    except sqlite3.Error:
        pass


def _finalize_merged_pr_cleanup(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    wp: Path,
    common: Path,
    branch: str,
    head: str,
    token: str,
    intent: dict,
) -> bool:
    task_id, path = row["id"], row["workspace_path"]
    removed = False
    _cleanup_worker_tmux(conn, task_id)
    try:
        with _all_board_write_fences(conn) as (current_board, siblings):
            fresh = current_board.execute(
                "SELECT status, workspace_path, branch_name, completion_contract, claim_lock "
                "FROM tasks WHERE id=?",
                (task_id,),
            ).fetchone()
            valid = bool(
                fresh
                and fresh["status"] in {"done", "archived", "failed", "cancelled"}
                and fresh["workspace_path"] == path
                and fresh["branch_name"] == branch
                and fresh["completion_contract"] == row["completion_contract"]
                and fresh["claim_lock"] == token
                and _fenced_path_workers_settled(
                    current_board, siblings, path, int(time.time())
                )
                and not _has_active_children(current_board, task_id)
                and _git_current_branch(wp) == branch
                and (_kb._git_out(wp, "rev-parse", "HEAD") or "").strip() == head
            )
            if not valid:
                _finish_pr_probe_txn(
                    current_board, task_id, token, "final_revalidation_failed"
                )
            else:
                removed = _cleanup_worktree_workspace_locked(
                    task_id, path, branch, merged_pr_head_oid=head,
                )
                if removed:
                    cur = current_board.execute(
                        "UPDATE tasks SET workspace_path=NULL, claim_lock=NULL, "
                        "claim_expires=NULL WHERE id=? AND workspace_path=? "
                        "AND claim_lock=?",
                        (task_id, path, token),
                    )
                    if cur.rowcount != 1:
                        raise sqlite3.IntegrityError(
                            "cleanup ownership changed after Git removal"
                        )
                    _kb._append_event(
                        current_board, task_id, _MERGED_PR_CLEANUP_EVENT, intent,
                    )
                else:
                    _finish_pr_probe_txn(
                        current_board, task_id, token, "cleanup_deferred"
                    )
            current_board.execute("COMMIT")
    except (OSError, sqlite3.Error):
        if not removed:
            _finish_pr_probe(conn, task_id, token, "finalization_failed")
        return False
    return removed


def _cleanup_merged_pr_candidate(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    pr: dict,
    wp: Path,
    common: Path,
    branch: str,
    head: str,
    token: str,
) -> bool:
    task_id, path = row["id"], row["workspace_path"]
    intent = {
        "path": path,
        "repo_root": str(common.parent.resolve(strict=True)),
        "common_dir": str(common.resolve(strict=True)),
        "branch": branch,
        "head": head,
        "pr_number": pr.get("number"),
        "pr_url": pr.get("url"),
        "merged_at": pr.get("mergedAt"),
        "claim": token,
    }
    with _worktree_lifecycle_lock(common):
        current = conn.execute(
            "SELECT status, workspace_path, branch_name, completion_contract, claim_lock "
            "FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        valid = bool(
            current
            and current["status"] in {"done", "archived", "failed", "cancelled"}
            and current["workspace_path"] == path
            and current["branch_name"] == branch
            and current["completion_contract"] == row["completion_contract"]
            and current["claim_lock"] == token
            and _all_path_workers_settled(conn, path, int(time.time()))
            and not _has_active_children(conn, task_id)
            and _git_current_branch(wp) == branch
            and (_kb._git_out(wp, "rev-parse", "HEAD") or "").strip() == head
        )
        if not valid:
            _finish_pr_probe(conn, task_id, token, "revalidation_failed")
            return False
        try:
            with _kb.write_txn(conn):
                fresh = conn.execute(
                    "SELECT status, workspace_path, branch_name, claim_lock "
                    "FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if not fresh or (
                    fresh["status"] not in {"done", "archived", "failed", "cancelled"}
                    or fresh["workspace_path"] != path
                    or fresh["branch_name"] != branch
                    or fresh["claim_lock"] != token
                ):
                    raise sqlite3.IntegrityError("cleanup intent ownership changed")
                _kb._append_event(conn, task_id, _PR_INTENT_EVENT, intent)
        except sqlite3.Error:
            _finish_pr_probe(conn, task_id, token, "intent_failed")
            return False
        return _finalize_merged_pr_cleanup(
            conn, row, wp, common, branch, head, token, intent
        )


def _reconcile_pr_candidate(
    conn: sqlite3.Connection, row: sqlite3.Row, now: int
) -> bool:
    task_id, path = row["id"], row["workspace_path"]
    wp = Path(path).expanduser()
    if not wp.is_dir():
        recovered = _recover_finished_pr_cleanup(conn, task_id, path)
        if not recovered:
            _record_pr_probe(conn, task_id, "missing_path_unproven")
        return recovered
    if not _is_linked_worktree_checkout(wp) or not _terminal_worker_settled(
        conn, task_id, now
    ):
        _record_pr_probe(conn, task_id, "precheck_deferred")
        return False
    token = _claim_pr_reconciliation(conn, task_id, now)
    if token is None:
        return False
    branch = (row["branch_name"] or "").strip()
    common = _git_common_dir(wp)
    head = (_kb._git_out(wp, "rev-parse", "HEAD") or "").strip()
    origin_repo = _origin_github_repo(wp)
    if (
        common is None
        or common.name != ".git"
        or not head
        or not origin_repo
        or _git_current_branch(wp) != branch
    ):
        _finish_pr_probe(conn, task_id, token, "git_identity_unproven")
        return False
    pr = _view_pr(row["completion_contract"], row["assignee"])
    head_repo = ((pr or {}).get("headRepository") or {}).get("nameWithOwner")
    if (
        not pr
        or not head_repo
        or head_repo.casefold() != origin_repo.casefold()
        or pr.get("headRefName") != branch
        or pr.get("headRefOid") != head
    ):
        _finish_pr_probe(conn, task_id, token, "pr_identity_unproven")
        return False
    if pr.get("state") == "OPEN" and (
        pr.get("mergeable") == "CONFLICTING" or pr.get("mergeStateStatus") == "DIRTY"
    ):
        alignment_id = _schedule_pr_alignment(conn, row, pr, head)
        _finish_pr_probe(
            conn,
            task_id,
            token,
            "conflicting" if alignment_id else "alignment_failed",
            {"pr_url": pr.get("url"), "alignment_task_id": alignment_id},
        )
        return False
    if pr.get("state") != "MERGED" or not pr.get("mergedAt"):
        _finish_pr_probe(
            conn, task_id, token, "not_merged", {"pr_url": pr.get("url")}
        )
        return False
    return _cleanup_merged_pr_candidate(
        conn, row, pr, wp, common, branch, head, token
    )


def reap_merged_pr_worktrees(
    conn: sqlite3.Connection,
    *,
    max_checks: int = 1,
    probe_interval_seconds: int = 900,
) -> list[str]:
    """Reconcile exact-contract PRs: align conflicts, reap only after merge.

    The specific persisted PR URL is authoritative. A merged PR must still
    match the local origin repository, branch and exact head. Conflict state
    schedules one idempotent alignment task per PR head. All destructive work
    requires a settled terminal worker, a claim, a pre-delete intent record,
    repeated DB/Git identity checks, and the repository lifecycle lock.
    """
    now = int(time.time())
    cutoff = now - max(0, int(probe_interval_seconds))
    try:
        rows = conn.execute(
            "SELECT t.id, t.workspace_path, t.branch_name, t.completion_contract, "
            "t.assignee, t.project_id, t.priority "
            "FROM tasks t "
            "WHERE t.workspace_kind='worktree' "
            "AND t.workspace_path IS NOT NULL AND t.branch_name IS NOT NULL "
            "AND t.completion_contract GLOB 'https://github.com/*/*/pull/*' "
            f"AND t.status IN {_TERMINAL_STATUSES_SQL} "
            "AND NOT EXISTS (SELECT 1 FROM task_events done "
            "  WHERE done.task_id=t.id AND done.kind=?) "
            "AND COALESCE((SELECT MAX(probe.created_at) FROM task_events probe "
            "  WHERE probe.task_id=t.id AND probe.kind=?), 0) <= ? "
            "ORDER BY COALESCE((SELECT MAX(probe.created_at) FROM task_events probe "
            "  WHERE probe.task_id=t.id AND probe.kind=?), 0), "
            "COALESCE(t.completed_at, t.created_at), t.id",
            (
                _MERGED_PR_CLEANUP_EVENT,
                _MERGED_PR_PROBE_EVENT,
                cutoff,
                _MERGED_PR_PROBE_EVENT,
            ),
        ).fetchall()
    except sqlite3.Error:
        return []

    cleaned: list[str] = []
    for row in rows[: max(0, int(max_checks))]:
        if _reconcile_pr_candidate(conn, row, now):
            cleaned.append(row["id"])
    return cleaned


def _try_cleanup_parent_workspaces(conn: sqlite3.Connection, task_id: str) -> None:
    """Run the deferred cleanup of any parent scratch/worktree workspace whose
    children are now all done/archived/failed/cancelled (called after each
    child completes).

    See #33774.
    """
    try:
        parents = conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        for (parent_id,) in parents:
            row = conn.execute(_WORKSPACE_ROW_SQL, (parent_id,)).fetchone()
            if (
                not row
                or row["workspace_kind"] not in _REMOVABLE_KINDS
                or not row["workspace_path"]
                or not _is_terminal_task(conn, parent_id)
                or _has_active_children(conn, parent_id)
            ):
                continue
            ws_path = row["workspace_path"]
            if row["workspace_kind"] == "worktree":
                if _defer_shared_worktree_cleanup(conn, parent_id, ws_path):
                    continue
                _cleanup_worktree_workspace(parent_id, ws_path, row["branch_name"])
                continue
            wp = Path(ws_path)
            if not wp.is_dir():
                continue
            if _is_managed_scratch_path(wp):
                if _defer_shared_workspace_cleanup(conn, parent_id, ws_path):
                    continue
                release_lsp_clients(str(wp))
                shutil.rmtree(wp, ignore_errors=True)
                _kb._log.debug("Deferred cleanup: removed parent %s scratch workspace: %s", parent_id, wp)
    except Exception:
        pass  # best-effort


def _cleanup_worker_tmux(conn: sqlite3.Connection, task_id: str) -> None:
    """Kill the tmux session associated with a task's assignee, if dead."""
    try:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row or not row["assignee"]:
            return
        # Workers named swarm1-12 use tmux sessions named swarm-swarm1 etc.
        session = f"swarm-{row['assignee']}"
        out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
            check=False,
        )
        if out.stdout.strip() == "1":
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=5, check=False)
            _kb._log.debug("Killed stale tmux session: %s", session)
    except Exception:
        pass  # best-effort — never block completion


_SCRATCH_TIP_SENTINEL_NAME = ".scratch_tip_shown"


_SCRATCH_TIP_MESSAGE = (
    "scratch workspaces are ephemeral — they're deleted when the task "
    "completes. Use --workspace worktree: (git worktree) or "
    "--workspace dir:/abs/path (existing dir) to preserve worker output."
)


def _scratch_tip_sentinel_path() -> Path:
    """Path to the per-install scratch-workspace-tip sentinel file."""
    return _kb.kanban_home() / _SCRATCH_TIP_SENTINEL_NAME


def _scratch_tip_shown() -> bool:
    """True iff the scratch-workspace tip was already emitted on this install.
    Best-effort — any error re-emits, the safer failure mode for a help message."""
    try:
        return _scratch_tip_sentinel_path().exists()
    except OSError:
        return False


def _mark_scratch_tip_shown() -> None:
    """Touch the sentinel so future scratch workspaces stay silent. Best-effort:
    a failure means the tip may appear once more, preferable to crashing dispatch."""
    try:
        path = _scratch_tip_sentinel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def _maybe_emit_scratch_tip(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_kind: Optional[str],
) -> None:
    """Emit the first-use scratch-workspace tip once per install, right after a
    scratch workspace is materialized. No-op for ``worktree``/``dir`` (preserved
    by design) and once the sentinel exists."""
    if (workspace_kind or "scratch") != "scratch" or _scratch_tip_shown():
        return
    try:
        _kb._log.warning("kanban: %s (task %s)", _SCRATCH_TIP_MESSAGE, task_id)
        with _kb.write_txn(conn):
            _kb._append_event(
                conn, task_id, "tip_scratch_workspace",
                {"message": _SCRATCH_TIP_MESSAGE},
            )
    except Exception:
        # Best-effort — never block the spawn loop over a help message.
        pass
    finally:
        _mark_scratch_tip_shown()


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _git_toplevel(path: Path) -> Optional[Path]:
    """Return the git toplevel containing ``path``, or ``None`` if not in a repo."""
    out = _kb._git_out(path, "rev-parse", "--show-toplevel")
    if out is None:
        return None
    try:
        return Path(out).expanduser().resolve()
    except Exception:
        return Path(out).expanduser()


def _git_branch_exists(repo_root: Path, branch_name: str) -> bool:
    try:
        result = _git(repo_root, "show-ref", "--verify", f"refs/heads/{branch_name}", timeout=30)
    except Exception:
        return False
    return result.returncode == 0


def _git_abs_path(path: Path, flag: str) -> Optional[Path]:
    out = _kb._git_out(path, "rev-parse", "--path-format=absolute", flag)
    return Path(out).expanduser().resolve(strict=False) if out else None


def _git_common_dir(path: Path) -> Optional[Path]:
    return _git_abs_path(path, "--git-common-dir")


def _git_current_branch(path: Path) -> Optional[str]:
    return _kb._git_out(path, "branch", "--show-current")


def _is_linked_worktree_checkout(path: Path) -> bool:
    """True when *path* is a linked worktree (git-dir differs from common-dir).

    One ``rev-parse`` call answers both directories.
    """
    out = _kb._git_out(
        path, "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"
    )
    lines = out.splitlines() if out else []
    return len(lines) == 2 and (
        Path(lines[0]).resolve(strict=False) != Path(lines[1]).resolve(strict=False)
    )


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _repo_root_for_worktree_target(path: Path) -> Optional[Path]:
    current = _nearest_existing_path(path).resolve(strict=False)
    while True:
        repo_root = _git_toplevel(current)
        if repo_root is not None:
            return repo_root
        if current == current.parent:
            return None
        current = current.parent


def _worktree_lifecycle_lock(common_dir: Path):
    """Cross-process lock shared by Kanban worktree creation and removal."""
    from hermes_cli.active_sessions import _FileLock

    return _FileLock(common_dir / "hermes-kanban-worktrees.lock")


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str) -> None:
    """Materialize ``target`` as a linked git worktree under ``repo_root``."""
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if repo_common is None:
        raise RuntimeError(f"cannot resolve git common directory for {repo_root}")
    with _worktree_lifecycle_lock(repo_common):
        if target.exists() and _path_key(_git_common_dir(target)) == _path_key(repo_common):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if _git_branch_exists(repo_root, branch_name):
            args = ["worktree", "add", str(target), branch_name]
        else:
            args = ["worktree", "add", "-b", branch_name, str(target), "HEAD"]
        result = _git(repo_root, *args, timeout=60)
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
            )


def _anchored_worktree(repo_root: Path, task_id: str, branch_name: str) -> tuple[Path, str]:
    """Materialize the canonical ``<repo>/.worktrees/<task-id>`` worktree."""
    target = repo_root / ".worktrees" / task_id
    _ensure_git_worktree(repo_root, target, branch_name)
    return target, branch_name


def _resolve_worktree_workspace(task: Task, *, board: Optional[str] = None) -> tuple[Path, str]:
    """Resolve + materialize a linked git worktree for ``task``. With no
    ``task.workspace_path`` the anchor is the board's ``default_workdir`` so
    every worktree lands under a board-owned repo (``<repo>/.worktrees/<id>``)
    instead of the dispatcher's incidental CWD (whatever dir the gateway was
    launched from); with no anchor configured we fail loudly rather than guess."""
    branch_name = (task.branch_name or "").strip() or f"wt/{task.id}"
    if not task.workspace_path:
        board_slug = board if board else _kb.get_current_board()
        board_default = (_kb.read_board_metadata(board_slug).get("default_workdir") or "").strip()
        if not board_default:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but no workspace_path, "
                f"and board {board_slug!r} has no default_workdir set. Set a board "
                "default workdir (a git repo) or create the task with "
                "--workspace worktree:<absolute-repo-path>."
            )
        anchor = Path(board_default).expanduser()
        if not anchor.is_absolute():
            raise ValueError(
                f"board {board_slug!r} default_workdir {board_default!r} is not "
                "absolute; use an absolute path to a git repo"
            )
        repo_root = _git_toplevel(anchor)
        if repo_root is None:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but board "
                f"{board_slug!r} default_workdir {board_default!r} is not inside a git repo"
            )
        return _anchored_worktree(repo_root, task.id, branch_name)

    requested = Path(task.workspace_path).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            f"task {task.id} has non-absolute worktree path "
            f"{task.workspace_path!r}; use an absolute path"
        )
    requested_resolved = requested.resolve(strict=False)

    if requested.exists() and _is_linked_worktree_checkout(requested):
        actual_branch = _git_current_branch(requested)
        if actual_branch == branch_name:
            return requested_resolved, actual_branch
        # The requested path is an existing checkout of a DIFFERENT task's
        # branch (decompose children inherit the root's workspace_path
        # verbatim, so siblings all point here). Reusing it would run this task
        # on the other task's branch — silent cross-task provenance corruption,
        # unsafe under concurrency — so fall back to our own worktree.
        fallback_root = _repo_root_for_worktree_target(requested.parent)
        if fallback_root is not None:
            fallback = fallback_root / ".worktrees" / task.id
            if _path_key(fallback.resolve(strict=False)) != _path_key(requested_resolved):
                _ensure_git_worktree(fallback_root, fallback, branch_name)
                return fallback.resolve(strict=False), branch_name
        # No repo to anchor a fallback on (or the occupied path IS this task's
        # own canonical worktree): keep the legacy reuse rather than fail dispatch.
        return requested_resolved, actual_branch or branch_name

    repo_root = _git_toplevel(requested)
    if repo_root is not None and _path_key(requested_resolved) == _path_key(repo_root):
        return _anchored_worktree(repo_root, task.id, branch_name)

    repo_root = _repo_root_for_worktree_target(requested.parent)
    if repo_root is None:
        raise ValueError(
            f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo "
            "and does not point at a git repo root"
        )
    _ensure_git_worktree(repo_root, requested, branch_name)
    return requested, branch_name


def resolve_workspace(task: Task, *, board: Optional[str] = None) -> Path:
    """Resolve (and create if needed) the workspace for a task.

    ``scratch``: ``<board-root>/workspaces/<id>/`` — path-stable across the
    dispatcher and every profile worker. ``dir``: ``workspace_path``, created
    if missing; MUST be absolute (relative paths would resolve against the
    dispatcher's CWD — confused-deputy traversal). ``worktree``: a linked git
    worktree; a repo-root ``workspace_path`` anchors ``<repo>/.worktrees/<id>``,
    a concrete path is created/reused, none -> the board's ``default_workdir``
    (raises if unset rather than guessing). Persist via ``set_workspace_path``.
    """
    kind = task.workspace_kind or "scratch"
    if kind == "worktree":
        return _resolve_worktree_workspace(task, board=board)[0]
    if kind == "scratch" and not task.workspace_path:
        p = _kb.workspaces_root(board=board) / task.id
    elif kind == "scratch":
        # Legacy explicit-path scratch tasks get the same absolute-path guard
        # as dir: — same threat model.
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; workspace paths must be absolute"
            )
    elif kind == "dir":
        if not task.workspace_path:
            raise ValueError(f"task {task.id} has workspace_kind=dir but no workspace_path")
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; use an absolute path "
                f"(relative paths are ambiguous against the dispatcher's CWD)"
            )
    else:
        raise ValueError(f"unknown workspace_kind: {kind}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_task_column(conn: sqlite3.Connection, task_id: str, column: str, value: str) -> None:
    with _kb.write_txn(conn):
        conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, task_id))


def set_workspace_path(conn: sqlite3.Connection, task_id: str, path: Path | str) -> None:
    _set_task_column(conn, task_id, "workspace_path", str(path))


def set_branch_name(conn: sqlite3.Connection, task_id: str, branch_name: str) -> None:
    _set_task_column(conn, task_id, "branch_name", str(branch_name))


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
