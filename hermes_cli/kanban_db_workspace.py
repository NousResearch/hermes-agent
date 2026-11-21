"""Task workspace lifecycle: scratch/dir/worktree resolution (incl. git worktree creation), post-completion cleanup with containment guards, worker tmux teardown and the first-use scratch-workspace tip.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import time
import unicodedata
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
        from hermes_cli.worktree_ops import _worktree_has_unpushed_commits, _worktree_is_dirty
    except Exception:
        return  # CLI safety predicates unavailable — preserve
    try:
        wp = Path(path).expanduser()
        if not wp.is_dir():
            return
        common = _git_common_dir(wp)
        if common is None or common.name != ".git":
            return  # not a linked worktree of a normal repo — never guess
        repo_root = common.parent
        if _path_key(wp.resolve(strict=False)) == _path_key(repo_root.resolve(strict=False)):
            return  # never remove the main checkout
        if _worktree_is_dirty(str(wp), str(repo_root)) or _worktree_has_unpushed_commits(str(wp)):
            _kb._log.info(
                "Preserving worktree for task %s: dirty or unpushed work at %s",
                task_id, wp,
            )
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
        pass  # best-effort — never block completion


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


_BASE_REF_FETCH_TIMEOUT = 20
_BASE_REF_FALLBACKS = ("origin/main", "origin/master")


def _fetch_origin(repo_root: Path) -> None:
    """Best-effort ``git fetch origin``: a missing remote or dead network must never fail the task."""
    try:
        fetch = _git(repo_root, "fetch", "origin", "--quiet", timeout=_BASE_REF_FETCH_TIMEOUT)
        if fetch.returncode != 0:
            _kb._log.debug("git fetch origin failed in %s: %s", repo_root, (fetch.stderr or "").strip())
    except (OSError, subprocess.SubprocessError) as exc:
        _kb._log.debug("git fetch origin failed in %s: %s", repo_root, exc)


def _remote_default_ref(repo_root: Path) -> Optional[str]:
    """The remote default branch as a local ref (``origin/HEAD`` target, else ``origin/main``,
    else ``origin/master``), or ``None`` when none verifies as a commit. Never fetches."""
    candidates: list[str] = []
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        head = _git(repo_root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD", timeout=10)
        if head.returncode == 0 and head.stdout.strip():
            candidates.append(head.stdout.strip())
    candidates.extend(ref for ref in _BASE_REF_FALLBACKS if ref not in candidates)
    for ref in candidates:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            verify = _git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", timeout=10)
            if verify.returncode != 0 or not verify.stdout.strip():
                continue
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                ahead = _git(repo_root, "rev-list", "--count", f"{ref}..HEAD", timeout=10)
                count = ahead.stdout.strip()
                if ahead.returncode == 0 and count not in ("", "0"):
                    _kb._log.debug(
                        "HEAD carries %s commits not on %s in %s; the base is still %s",
                        count, ref, repo_root, ref,
                    )
            return ref
    return None


def _worktree_base_ref(repo_root: Path) -> str:
    """Start point for a new task branch: the fetched remote default branch, else ``HEAD``.

    Forking from the anchor's local ``HEAD`` made every task worktree inherit a stale
    anchor's blindness: parked on an old branch, files that exist on ``origin/main``
    read as nonexistent and CI compiles code the change never touches. The rule is the
    fetched remote default branch always (``origin/HEAD`` target, else ``origin/main``,
    else ``origin/master``); ``HEAD`` only when none resolves (today's behaviour). An
    anchor parked on a local-only commit is not stranded: its branch ref keeps that
    commit, and the task worktree simply starts from the remote default branch. The
    fetch is best-effort: a missing remote or dead network must never fail the task.
    """
    _fetch_origin(repo_root)
    return _remote_default_ref(repo_root) or "HEAD"


def _commits_behind(tree: Path, ref: str) -> Optional[int]:
    out = _kb._git_out(tree, "rev-list", "--count", f"HEAD..{ref}")
    return int(out) if out and out.isdigit() else None


def _op_in_progress(tree: Path) -> bool:
    """True while a rebase or merge is mid-flight in *tree* (rebasing over it would corrupt it)."""
    for name in ("rebase-merge", "rebase-apply", "MERGE_HEAD"):
        out = _kb._git_out(tree, "rev-parse", "--path-format=absolute", "--git-path", name)
        if out and Path(out).exists():
            return True
    return False


def _tree_is_clean(tree: Path) -> bool:
    """Unknown counts as dirty: a status we cannot read must not be rebased over."""
    try:
        status = _git(tree, "status", "--porcelain", "--untracked-files=normal", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return status.returncode == 0 and not status.stdout.strip()


def _stale_reuse_error(tree: Path, branch_name: str, behind: int, ref: str, reason: str) -> RuntimeError:
    return RuntimeError(
        f"refusing to dispatch into {tree}: {reason} and the branch {branch_name} is "
        f"{behind} commit(s) behind {ref}. Resolve by hand: "
        f"git -C {tree} status ; git -C {tree} rebase {ref}"
    )


def _rebase_onto(tree: Path, branch_name: str, behind: int, ref: str) -> None:
    """Rebase *tree* onto *ref*; on conflict abort (branch left byte-identical) and raise."""
    try:
        rebase = _git(tree, "rebase", ref, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            _git(tree, "rebase", "--abort", timeout=60)
        raise _stale_reuse_error(tree, branch_name, behind, ref, f"the rebase onto {ref} failed ({exc})") from exc
    if rebase.returncode == 0:
        return
    conflicted = ""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        conflicted = _git(tree, "diff", "--name-only", "--diff-filter=U", timeout=30).stdout.strip()
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        _git(tree, "rebase", "--abort", timeout=60)
    paths = ", ".join(conflicted.splitlines()) or "see git status"
    raise _stale_reuse_error(
        tree, branch_name, behind, ref, f"the rebase onto {ref} conflicts ({paths})"
    )


def _push_rebased_branch(tree: Path, branch_name: str) -> None:
    """Publish the rebase only for a branch that already exists on origin; never fails dispatch."""
    tracked = _kb._git_out(tree, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch_name}")
    if not tracked:
        return
    try:
        push = _git(tree, "push", "--force-with-lease", "origin", branch_name, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        _kb._log.warning("kanban: push of rebased %s from %s failed: %s", branch_name, tree, exc)
        return
    if push.returncode != 0:
        _kb._log.warning(
            "kanban: push of rebased %s from %s failed: %s",
            branch_name, tree, (push.stderr or push.stdout or "").strip(),
        )


def _reuse_gate(repo_root: Path, tree: Path, branch_name: str) -> None:
    """Bring a REUSED task branch up to the fetched remote default branch, or refuse.

    A new branch forks from the remote default branch (:func:`_worktree_base_ref`) but a
    reused branch/tree is handed back wherever it was parked, so the worker ran on stale
    code. A clean tree behind the base is rebased (work is kept, never reset or stashed);
    a dirty tree, an operation already in progress, or a conflict refuses the dispatch so
    the card blocks for a human instead of running stale. With no remote default ref
    (offline, no ``origin``) the reuse is unchanged.
    """
    _fetch_origin(repo_root)
    ref = _remote_default_ref(repo_root)
    if ref is None:
        return
    behind = _commits_behind(tree, ref)
    if not behind:
        return
    if _op_in_progress(tree):
        raise _stale_reuse_error(tree, branch_name, behind, ref, "a rebase/merge is already in progress")
    if not _tree_is_clean(tree):
        raise _stale_reuse_error(
            tree, branch_name, behind, ref,
            "working tree is dirty (uncommitted or untracked files)",
        )
    _rebase_onto(tree, branch_name, behind, ref)
    _kb._log.info(
        "kanban: rebased reused worktree %s (%s) onto %s: behind %s -> %s",
        tree, branch_name, ref, behind, _commits_behind(tree, ref),
    )
    _push_rebased_branch(tree, branch_name)


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str) -> None:
    """Materialize ``target`` as a linked git worktree under ``repo_root``."""
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if target.exists() and repo_common is not None and _path_key(_git_common_dir(target)) == _path_key(repo_common):
        if _git_current_branch(target) == branch_name:
            _reuse_gate(repo_root, target, branch_name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    reused_branch = _git_branch_exists(repo_root, branch_name)
    if reused_branch:
        args = ["worktree", "add", str(target), branch_name]
    else:
        # A new branch starts at the fetched default branch so it cannot
        # inherit a stale anchor checkout's blindness.
        args = ["worktree", "add", "-b", branch_name, str(target), _worktree_base_ref(repo_root)]
    result = _git(repo_root, *args, timeout=60)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
        )
    if reused_branch:
        _reuse_gate(repo_root, target, branch_name)


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
            _reuse_gate(_git_toplevel(requested) or requested, requested, branch_name)
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
