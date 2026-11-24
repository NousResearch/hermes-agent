"""Task workspace lifecycle: scratch/dir/worktree resolution (incl. git worktree creation), post-completion cleanup with containment guards, worker tmux teardown and the first-use scratch-workspace tip.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING
import contextlib

from hermes_cli.worktree_ops import release_lsp_clients

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_REMOVABLE_KINDS = ("scratch", "worktree")
_REMOVABLE_KINDS_SQL = "('scratch', 'worktree')"


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


_SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"


def _sibling_db_state(db_file: Path) -> str:
    """Classify a sibling board's ``kanban.db`` WITHOUT opening it as a database.

    ``empty`` (missing or 0 bytes) and ``not-sqlite`` (wrong header) files are
    leftovers that never hold tasks — e.g. a bare ``sqlite3.connect()`` probe on
    a guessed path creates a 0-byte file on Windows; ``archived`` names a board
    whose ``board.json`` tombstone says it is archived (nothing dispatches
    there); ``ready`` is a file that looks like a database and must be read.
    The header is read directly because opening through SQLite — even read-only
    — can create ``-wal``/``-shm`` sidecars.
    """
    try:
        if db_file.stat().st_size == 0:
            return "empty"
    except OSError:
        return "empty"
    try:
        with open(db_file, "rb") as fh:
            head = fh.read(len(_SQLITE_HEADER_MAGIC))
    except OSError:
        return "ready"  # header unreadable: let the query decide
    if not head.startswith(_SQLITE_HEADER_MAGIC):
        return "not-sqlite"
    try:
        meta = json.loads((db_file.parent / "board.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return "ready"
    if isinstance(meta, dict) and meta.get("archived") is True:
        return "archived"
    return "ready"


def _is_missing_tasks_table(exc: sqlite3.Error) -> bool:
    """``no such table: tasks``: the file opened fine and was never initialized as a board."""
    return isinstance(exc, sqlite3.OperationalError) and "no such table: tasks" in str(exc)


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

    A sibling file that is not a live board cannot hold a live task, so it is
    skipped instead of deferring every cleanup on the board: 0 bytes, not a
    SQLite file, an archived board, or a database with no ``tasks`` table (never
    initialized). Only a board DB that has a schema and still cannot be read
    (locked, corrupt) defers the delete.
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
        state = _sibling_db_state(db_file)
        if state != "ready":
            _kb._log.info(
                "Skipping sibling board db %s for task %s: %s (not a live board)",
                db_file, task_id, state,
            )
            continue
        try:
            if _other_board_uses_path(db_file, task_id, key):
                return "shared"
        except sqlite3.Error as exc:
            if _is_missing_tasks_table(exc):
                _kb._log.info(
                    "Skipping sibling board db %s for task %s: no tasks table "
                    "(not an initialized board)",
                    db_file, task_id,
                )
                continue
            _kb._log.warning(
                "Cannot read board db %s for task %s: %s", db_file, task_id, exc,
            )
            return "unknown"
        except OSError as exc:
            _kb._log.warning(
                "Cannot read board db %s for task %s: %s", db_file, task_id, exc,
            )
            return "unknown"
    return None


def _stray_board_db_stubs(conn: sqlite3.Connection) -> list[tuple[Path, str]]:
    """``(path, why)`` for board DB files that are not live boards — read-only.

    Every ``<boards>/<slug>/kanban.db`` whose :func:`_sibling_db_state` is not
    ``ready`` or which has no ``tasks`` table, plus a zero-byte
    ``<kanban_home>/kanban/kanban.db`` (a guessed-path leftover, not a layout
    Hermes ever writes). The board in use is never listed. The shared-workspace
    scan skips these; ``kanban gc --workspaces`` lists them so they get deleted
    instead of lingering.
    """
    from hermes_cli.kanban_db_connect import _main_db_file  # late: import cycle

    current_file = _main_db_file(conn)
    current = Path(current_file).resolve() if current_file else None
    candidates: list[Path] = [_kb.kanban_home() / "kanban" / "kanban.db"]
    root = _kb.boards_root()
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                candidates.append(child / "kanban.db")
    stubs: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not path.is_file() or resolved in seen:
            continue
        if current is not None and resolved == current:
            continue
        seen.add(resolved)
        state = _sibling_db_state(resolved)
        if state != "ready":
            stubs.append((resolved, state))
            continue
        try:
            other = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True, timeout=1.0)
            try:
                tables = {r[0] for r in other.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
            finally:
                other.close()
        except (OSError, sqlite3.Error) as exc:
            stubs.append((resolved, f"unreadable: {exc}"))
            continue
        if "tasks" not in tables:
            stubs.append((resolved, "no tasks table"))
    return stubs


def _record_deferred_shared_cleanup(
    conn: sqlite3.Connection, task_id: str, path: Path | str, reason: str
) -> None:
    """Log why *path* is kept and record ``workspace_cleanup_deferred_shared``."""
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


def _defer_shared_workspace_cleanup(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> bool:
    """Skip removal and record why, when *path* may still be used by a live task."""
    reason = _workspace_in_use_by_other(conn, task_id, path)
    if reason is None:
        return False
    _record_deferred_shared_cleanup(conn, task_id, path, reason)
    return True


def _defer_shared_worktree_cleanup(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> bool:
    """Defer removing a linked worktree another live task may still use.

    Only a real linked worktree is ever removed, so the sharing scan runs just
    for those.
    """
    wt = Path(path).expanduser()
    if not (wt.is_dir() and _is_linked_worktree_checkout(wt)):
        return False
    reason = _workspace_in_use_by_other(conn, task_id, path)
    if reason is None:
        return False
    _record_deferred_shared_cleanup(conn, task_id, path, reason)
    return True


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


def _clear_readonly_bits(root: Path) -> None:
    """Best-effort: drop the read-only attribute from *root* and everything below it."""
    targets = [root]
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=lambda _exc: None):
        targets.extend(Path(dirpath, name) for name in dirnames)
        targets.extend(Path(dirpath, name) for name in filenames)
    for target in targets:
        with contextlib.suppress(OSError):
            os.chmod(target, stat.S_IWRITE)


def _leave_tree(path: Path) -> bool:
    """Move this process's cwd out of *path* when it sits inside it.

    Windows keeps a handle on the directory a process is sitting in, so an
    ``rmtree`` of that tree fails on the directory itself (``PermissionError``
    / WinError 32) after emptying it — one silent empty shell per finished
    scratch card. A cwd the process reached via ``os.chdir`` is released by
    moving out. POSIX lets the removal through but strands the process on a
    deleted cwd; moving out prevents that too. One handle this cannot release:
    the one for the directory the process was *started* in survives until the
    process exits (that residue is the dispatcher's second pass, see
    :func:`_retry_workspace_cleanup_after_exit`). True when moved.
    """
    target = path.resolve(strict=False)
    try:
        cwd = Path.cwd().resolve(strict=False)
    except OSError:
        return False  # the cwd is already gone: it cannot be holding *path* open
    if cwd != target and not cwd.is_relative_to(target):
        return False
    try:
        os.chdir(target.parent)
    except OSError as exc:
        _kb._log.warning("Could not leave %s before removing it: %s", path, exc)
        return False
    return True


def _force_rmtree(path: Path) -> Optional[str]:
    """Remove the tree at *path*; None once it is gone, else why it could not be.

    ``shutil.rmtree(path, ignore_errors=True)`` gives up silently on the first
    read-only entry — on Windows ``unlink``/``rmdir`` of a read-only file or
    directory raises ``PermissionError`` — leaving a half-deleted tree with no
    event and no log line (git checkouts and fixtures copied out of a
    repository keep the read-only bit). After a failed first pass the bit is
    cleared on what remains and the removal is retried once; the caller logs
    any residue. A cwd inside *path* is moved out first (:func:`_leave_tree`):
    on Windows the directory the caller is sitting in can never be removed
    otherwise, whatever the retry.
    """
    if _leave_tree(path):
        _kb._log.debug("Left %s to remove it (the cwd was inside)", path)
    try:
        shutil.rmtree(path)
    except OSError:
        pass
    else:
        return None
    _clear_readonly_bits(path)
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return str(exc) or exc.__class__.__name__
    return None


@dataclass(frozen=True)
class _CleanupPlan:
    """What :func:`_plan_workspace_cleanup` decided about one task's workspace.

    ``reason`` is ``""`` when the workspace can go now; any other value names
    why it is kept (``not_removable``, ``active_children``, ``missing``,
    ``unmanaged``, ``shared``, ``unknown``, or a worktree preserve reason).
    """

    kind: Optional[str]
    path: Optional[Path]
    reason: str
    branch_name: Optional[str] = None
    repo_root: Optional[Path] = None

    @property
    def removable(self) -> bool:
        return self.reason == ""


def _plan_workspace_cleanup(conn: sqlite3.Connection, task_id: str) -> _CleanupPlan:
    """Decide whether *task_id*'s workspace may be removed now — the one predicate
    set behind completion cleanup, deferred parent cleanup (#33774) and
    ``kanban gc --workspaces``.

    ``scratch`` goes only when it sits under a kanban-managed workspaces root
    and no live task — here, or on a readable sibling board — still points at
    it; ``worktree`` only when it is a linked worktree that is clean with every
    commit pushed; ``dir`` is never removed. Active children keep a parent's
    workspace for its handoff artifacts. The task's own status is the caller's
    to check (completion/archive paths, or the terminal filter of ``gc``).
    """
    row = conn.execute(_WORKSPACE_ROW_SQL, (task_id,)).fetchone()
    if not row:
        return _CleanupPlan(None, None, "not_removable")
    kind: Optional[str] = row["workspace_kind"]
    path: Optional[str] = row["workspace_path"]
    branch: Optional[str] = row["branch_name"]
    if kind not in _REMOVABLE_KINDS or not path:
        return _CleanupPlan(kind, None, "not_removable", branch)
    wp = Path(path).expanduser()
    # Defer while any child is not yet terminal so it can still read handoff
    # artifacts from this workspace.
    if _has_active_children(conn, task_id):
        return _CleanupPlan(kind, wp, "active_children", branch)
    if kind == "worktree":
        if not wp.is_dir():
            return _CleanupPlan(kind, wp, "missing", branch)
        if not _is_linked_worktree_checkout(wp):
            return _CleanupPlan(kind, wp, "not_linked_worktree", branch)
    elif not wp.is_dir():
        return _CleanupPlan(kind, wp, "missing", branch)
    elif not _is_managed_scratch_path(wp):
        # Containment guard: a board's ``default_workdir`` can pair
        # ``workspace_kind='scratch'`` with a user path pointing at a real
        # source tree; without this, completion would rmtree the user's data.
        # See #28818. Containment runs first so an unmanaged dir that is never
        # removed does not pay for (or log) the shared-use scan.
        return _CleanupPlan(kind, wp, "unmanaged", branch)
    shared = _workspace_in_use_by_other(conn, task_id, path)
    if shared:
        return _CleanupPlan(kind, wp, shared, branch)
    if kind == "worktree":
        repo_root, preserve = _worktree_preserve_reason(wp)
        return _CleanupPlan(kind, wp, preserve or "", branch, repo_root)
    return _CleanupPlan(kind, wp, "", branch)


def _apply_cleanup_plan(
    conn: sqlite3.Connection, task_id: str, plan: _CleanupPlan
) -> bool:
    """Remove per an actionable plan (``plan.removable``); True when the path is gone."""
    wp = plan.path
    if wp is None:
        return False
    if plan.kind == "worktree":
        _remove_linked_worktree(task_id, wp, plan.repo_root, plan.branch_name)
        return not wp.exists()
    release_lsp_clients(str(wp))
    residue = _force_rmtree(wp)
    if residue:
        _kb._log.warning(
            "Scratch workspace for task %s could not be fully removed (%s): %s",
            task_id, residue, wp,
        )
        return False
    _kb._log.debug("Removed scratch workspace: %s", wp)
    return True


def _cleanup_workspace(conn: sqlite3.Connection, task_id: str) -> None:
    """Remove a task's scratch workspace dir and kill its stale tmux session.
    Called from :func:`complete_task` after the transaction commits; best-effort
    so cleanup never blocks completion. ``scratch`` is removed; ``worktree``
    only when provably free of work (clean tree, every commit reachable from a
    remote-tracking ref); ``dir`` is intentionally preserved. Decisions come
    from :func:`_plan_workspace_cleanup`, shared with ``kanban gc --workspaces``.
    """
    try:
        plan = _plan_workspace_cleanup(conn, task_id)
        if plan.reason == "not_removable":
            # Not removable itself, but completing may still unblock a deferred
            # parent scratch cleanup (e.g. a 'dir' child of a scratch parent).
            # See #33774.
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        if plan.reason == "active_children":
            _kb._log.debug(
                "Deferring %s workspace cleanup for task %s: "
                "active children still need workspace at %s",
                plan.kind, task_id, plan.path,
            )
            return
        wp = plan.path
        if wp is None:
            return
        # Kill the (dead) tmux worker session BEFORE removing a worktree so a
        # lingering worker never has its cwd deleted from under it.
        if plan.kind == "worktree":
            _cleanup_worker_tmux(conn, task_id)
        if plan.removable:
            _apply_cleanup_plan(conn, task_id, plan)
        elif plan.reason in ("shared", "unknown"):
            _record_deferred_shared_cleanup(conn, task_id, wp, plan.reason)
        elif plan.reason == "unmanaged":
            _kb._log.warning(
                "Refusing to remove out-of-scratch workspace for task %s: %s "
                "(workspace_kind='scratch' but path is outside any "
                "kanban-managed workspaces root)",
                task_id, wp,
            )
        elif plan.reason == "dirty_or_unpushed":
            _kb._log.info(
                "Preserving worktree for task %s: dirty or unpushed work at %s",
                task_id, wp,
            )
        if plan.kind == "worktree":
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        # Kill the owning worker's tmux session if it is now dead, then let any
        # parent whose children are all done run its deferred cleanup.
        _cleanup_worker_tmux(conn, task_id)
        # After cleaning up this task's workspace, check if any parent tasks now have all children done —
        # their deferred cleanup can proceed (#33774).
        _try_cleanup_parent_workspaces(conn, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _retry_workspace_cleanup_after_exit(conn: sqlite3.Connection, task_id: str) -> bool:
    """One more cleanup pass for a terminal task whose worker is known to be gone.

    One more pass matters because completion cleanup runs inside the worker,
    which the dispatcher started with the workspace as its cwd; on Windows the
    directory a process was started in keeps a handle until that process exits,
    so the worker's own deletion always failed on the directory itself and left
    an empty shell. The dispatcher calls this once it has confirmed the run's
    worker exited and cleared its pid evidence; True when the directory went.
    A residue is logged here rather than left for a user to find, and
    ``kanban gc --workspaces`` still reports it.
    """
    if not _is_terminal_task(conn, task_id):
        return False  # requeued: the next run reuses this workspace
    plan = _plan_workspace_cleanup(conn, task_id)
    if not plan.removable:
        return False  # gone already, or kept for a reason completion recorded
    _cleanup_workspace(conn, task_id)
    if plan.path is not None and plan.path.exists():
        _kb._log.warning(
            "Workspace for task %s still present after its worker exited: %s",
            task_id, plan.path,
        )
        return False
    _kb._log.info(
        "Removed leftover workspace for task %s after its worker exited: %s",
        task_id, plan.path,
    )
    return True


def _worktree_preserve_reason(wp: Path) -> tuple[Optional[Path], Optional[str]]:
    """Why the checkout at *wp* must be kept, together with its repo root.

    ``(repo_root, None)`` means provably free of work: a real linked worktree
    with a clean tree and every commit reachable from a remote-tracking ref.
    Mirrors the CLI startup pruner (``cli._prune_stale_worktrees``); any doubt
    (missing, not a linked worktree, the main checkout, dirty, unpushed,
    predicates unavailable) preserves it.
    """
    try:
        from hermes_cli.worktree_ops import _worktree_has_unpushed_commits, _worktree_is_dirty
    except ImportError:
        return None, "predicates_unavailable"  # CLI safety predicates unavailable — preserve
    if not wp.is_dir():
        return None, "missing"
    common = _git_common_dir(wp)
    if common is None or common.name != ".git":
        # Only ever remove a linked worktree of a normal repo. `_git_common_dir`
        # returns None for a bare repo, a submodule, or anything without a
        # worktree-scoped `.git`; a submodule's gitdir is `<repo>/.git/modules/<name>`
        # and a plain dir's `.git` is a file, not a dir, so neither can name a
        # `.git` dir. Refuse to guess in those cases: never `rm -rf` an unknown
        # checkout type.
        return None, "not_linked_worktree"  # not a linked worktree of a normal repo — never guess
    repo_root = common.parent
    if _path_key(wp.resolve(strict=False)) == _path_key(repo_root.resolve(strict=False)):
        return repo_root, "main_checkout"  # never remove the main checkout
    if _worktree_is_dirty(str(wp), str(repo_root)) or _worktree_has_unpushed_commits(str(wp)):
        return repo_root, "dirty_or_unpushed"
    return repo_root, None


def _remove_linked_worktree(
    task_id: str, wp: Path, repo_root: Optional[Path], branch_name: Optional[str]
) -> None:
    """``git worktree remove`` a linked worktree already proven free of work.
    The auto-generated ``wt/<task-id>`` branch is deleted with it; custom
    branches are kept. Best-effort."""
    try:
        if repo_root is None:
            return
        # Windows cannot delete a directory while this process has its current
        # directory inside it. Completed workers normally run from their own
        # linked worktree, so move this process back to the main checkout
        # before asking Git to remove the worktree.
        worktree_path = wp.resolve(strict=False)
        try:
            cwd = Path.cwd().resolve(strict=False)
        except OSError:
            # cwd was already deleted (a scratch-kind child's own workspace is
            # rmtree'd before this deferred parent cleanup runs, #33774). A
            # dead cwd cannot hold the worktree open, so leaving it is safe.
            cwd = None
        if cwd is None or cwd == worktree_path or cwd.is_relative_to(worktree_path):
            try:
                os.chdir(repo_root)
            except OSError as exc:
                _kb._log.warning(
                    "Preserving worktree for task %s: cannot leave %s for %s: %s",
                    task_id, cwd or "<deleted cwd>", repo_root, exc,
                )
                return
        # No --force: git's own dirty guard re-verifies at removal time, so if
        # the tree became dirty since our check (TOCTOU) removal fails safe.
        release_lsp_clients(str(worktree_path))
        result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
        if result.returncode != 0:
            # Windows can retain a directory handle briefly after cwd changes.
            # Retry once without --force; Git still enforces its dirty guard.
            time.sleep(0.1)
            result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
        if result.returncode != 0:
            _kb._log.warning(
                "git worktree remove failed for task %s at %s: %s",
                task_id, wp, (result.stderr or result.stdout or "").strip(),
            )
            return
        _kb._log.debug("Removed worktree workspace: %s", wp)
        branch = (branch_name or "").strip() or f"wt/{task_id}"
        if branch.startswith("wt/"):
            _git(repo_root, "branch", "-D", branch, timeout=30)
    except Exception:
        # Best-effort by contract: completion must never be blocked by a
        # worktree removal, but the failure is no longer silent.
        logger.debug("worktree removal skipped for task %s at %s", task_id, wp, exc_info=True)


def _cleanup_worktree_workspace(
    task_id: str, path: str, branch_name: Optional[str] = None
) -> None:
    """Remove a finished task's linked git worktree when it holds no work.
    Mirrors the CLI startup pruner (``cli._prune_stale_worktrees``): removal
    requires a clean tree AND every commit reachable from a remote-tracking
    ref; any doubt (dirty, unpushed, unresolvable repo, failing git) preserves
    it. The auto-generated ``wt/<task-id>`` branch is deleted with it; custom
    branches are kept. Best-effort."""
    try:
        wp = Path(path).expanduser()
        repo_root, preserve = _worktree_preserve_reason(wp)
        if preserve == "dirty_or_unpushed":
            _kb._log.info(
                "Preserving worktree for task %s: dirty or unpushed work at %s",
                task_id, wp,
            )
        if preserve is not None or repo_root is None:
            return
        _remove_linked_worktree(task_id, wp, repo_root, branch_name)
    except Exception:
        # Best-effort by contract: completion must never be blocked by a
        # worktree check, but the failure is no longer silent.
        logger.debug("worktree cleanup skipped for task %s at %s", task_id, path, exc_info=True)


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
            if not _is_terminal_task(conn, parent_id):
                continue
            plan = _plan_workspace_cleanup(conn, parent_id)
            if plan.path is None:
                continue
            if plan.removable:
                if _apply_cleanup_plan(conn, parent_id, plan):
                    _kb._log.debug("Deferred cleanup: removed parent %s workspace: %s", parent_id, plan.path)
            elif plan.reason in ("shared", "unknown"):
                _record_deferred_shared_cleanup(conn, parent_id, plan.path, plan.reason)
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


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str) -> None:
    """Materialize ``target`` as a linked git worktree under ``repo_root``."""
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if target.exists() and repo_common is not None and _path_key(_git_common_dir(target)) == _path_key(repo_common):
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

# Same logger as the origin module (one kanban log stream). Module-level name
# so ruff's blind-except rule sees the handler logs the traceback.
logger = _kb._log
