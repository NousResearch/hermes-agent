"""Workspace and project-root resolution for LSP.

1. **Workspace gate** — LSP only runs when the cwd (or the edited file) sits inside a git
   worktree, so gateway users on user-home cwd's never spawn daemons.
2. **Workspace trust** — whether a server may load code the project itself ships.
3. **nearest_root** — the per-server project-root walk: up from a start path looking for marker
   files (``pyproject.toml``, ``Cargo.toml``, ...), optionally bailing if an exclude marker
   shows up first.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import AbstractSet, Iterable, Iterator, Optional, Set, Tuple

logger = logging.getLogger("agent.lsp.workspace")

# Cache: start dir → (worktree_root, is_git) so repeated calls don't re-stat.  Cleared on shutdown; capped
# because every distinct file dir a long gateway session touches lands here (#62950).
_workspace_cache: dict = {}
_WORKSPACE_CACHE_CAP = 512

# Walk cap: the deepest reasonable monorepo is well under 64 levels; bounds a
# pathological cwd or symlink cycle even though parent-equality normally stops us.
_MAX_WALK = 64


def normalize_path(path: str) -> str:
    """Expand ``~``, make absolute, collapse ``.``/``..``.  Symlinks are deliberately NOT resolved —
    some servers (rust-analyzer's Cargo workspace identity) care, and we want the path the user typed."""
    return os.path.abspath(os.path.expanduser(path))


def _start_dir(start: str) -> Optional[Path]:
    """Normalized start directory (a file's parent), or ``None`` on pathological input."""
    try:
        start_path = Path(normalize_path(start))
        if start_path.is_file():
            start_path = start_path.parent
    except (OSError, RuntimeError, ValueError):
        # Symlink loop, encoding error, etc. — bail rather than crash the lint hook.
        return None
    return start_path


def _walk_up(start: Path) -> Iterator[Path]:
    """Yield ``start`` and its ancestors up to the filesystem root, bounded by ``_MAX_WALK``."""
    cur = start
    for _ in range(_MAX_WALK):
        yield cur
        parent = cur.parent
        if parent == cur:
            return
        cur = parent


def find_git_worktree(start: str) -> Optional[str]:
    """Return the nearest ancestor dir containing ``.git`` (file or dir — worktrees count), else ``None``."""
    start_path = _start_dir(start)
    if start_path is None:
        return None
    cached = _workspace_cache.get(str(start_path))
    if cached is not None:
        return cached[0]
    resolved = None
    for cur in _walk_up(start_path):
        try:
            if (cur / ".git").exists():
                resolved = str(cur)
                break
        except OSError:
            break  # permission error on a parent dir — bail out cleanly
    _workspace_cache[str(start_path)] = (resolved, resolved is not None)
    if len(_workspace_cache) > _WORKSPACE_CACHE_CAP:
        _workspace_cache.clear()  # a stat cache: resetting is a few re-stats, and one atomic op is thread-safe
    return resolved


def is_inside_workspace(path: str, workspace_root: str) -> bool:
    """True iff ``path`` is inside (or equal to) ``workspace_root``.  Symlinks are not resolved: a
    symlink pointing outside still counts as outside, matching servers that reject didOpen for
    unrelated files."""
    p = normalize_path(path)
    root = normalize_path(workspace_root)
    if p == root:
        return True
    # commonpath handles case-insensitive filesystems on macOS/Windows.
    try:
        return os.path.commonpath([p, root]) == root
    except ValueError:
        return False  # different drives on Windows


def nearest_root(
    start: str,
    markers: Iterable[str],
    *,
    excludes: Optional[Iterable[str]] = None,
    ceiling: Optional[str] = None,
) -> Optional[str]:
    """Walk up from ``start`` for the directory containing the first matched marker.

    Returns ``None`` past ``ceiling`` (or the filesystem root), or when an exclude marker is found
    first — the server is gated off for that file (e.g. typescript skips deno projects when
    ``deno.json`` precedes ``package.json``).  Marker names are exact filenames — no globs.
    """
    start_path = _start_dir(start)
    if start_path is None:
        return None
    ceiling_path = Path(normalize_path(ceiling)) if ceiling else None
    markers_list = list(markers)
    excludes_list = list(excludes) if excludes else []

    def present(cur: Path, names: list) -> bool:
        for name in names:
            try:
                if (cur / name).exists():
                    return True
            except OSError:
                continue
        return False

    for cur in _walk_up(start_path):
        # Excludes are checked before markers at each level.
        if present(cur, excludes_list):
            return None
        # A directory holding __init__.py is a Python package, never a project root (hermes_cli/setup.py
        # matched the python marker list and gave every package dir its own pyright).
        if not present(cur, ["__init__.py"]) and present(cur, markers_list):
            return str(cur)
        if ceiling_path is not None and cur == ceiling_path:
            return None
    return None


def resolve_workspace_for_file(file_path: str, *, cwd: Optional[str] = None) -> tuple[Optional[str], bool]:
    """Return ``(workspace_root, gated_in)`` for a file.  The cwd's worktree wins when the file is
    inside it; otherwise the file's own worktree is the fallback anchor (monorepos / unrelated
    checkouts).  ``(None, False)`` when neither is in a git worktree."""
    try:
        cwd_anchor = cwd or os.getcwd()
    except OSError:
        # The process cwd was removed underneath us (a scratch workspace cleaned up at
        # card completion); getcwd keeps raising even after the path is recreated, so
        # there is simply no cwd anchor — fall through to the file's own worktree.
        cwd_anchor = None
    cwd_root = find_git_worktree(cwd_anchor) if cwd_anchor else None
    if cwd_root is not None and is_inside_workspace(file_path, cwd_root):
        return cwd_root, True
    file_root = find_git_worktree(file_path)
    if file_root is not None:
        return file_root, True
    return None, False


def operator_workspace_roots() -> set[str]:
    """Git worktrees the operator pointed Hermes at: the launch dir and the surface-set workspace
    (``resolve_agent_cwd``: the Desktop/TUI session cwd, ``hermes -w``'s worktree, a gateway's
    ``terminal.cwd``).  The agent's ``cd`` moves neither (it only moves the terminal's cwd).  A repo at
    or above ``$HOME`` never counts: a dotfiles repo would trust every directory below it."""
    from agent.runtime_cwd import resolve_agent_cwd
    from gateway.session_context import get_session_env
    from tools.terminal_scope import TerminalPolicyUnavailable
    from utils import is_truthy_value
    # Work the model can schedule has no operator anchor: a kanban worker is launched in (with
    # TERMINAL_CWD =) the task's workspace and a cron run's session cwd is the job's workdir, and the
    # kanban_create / cronjob tools let the model pick both.
    if os.environ.get("HERMES_KANBAN_TASK"):
        return set()
    anchors = [os.getcwd]
    if not is_truthy_value(get_session_env("HERMES_CRON_SESSION", "")):
        anchors.append(resolve_agent_cwd)
    home = normalize_path("~")
    roots: set[str] = set()
    for anchor in anchors:
        try:
            root = find_git_worktree(str(anchor()))
        except (OSError, TerminalPolicyUnavailable):  # a deleted cwd; a profile whose terminal policy failed
            continue
        if root is not None and not is_inside_workspace(home, root):
            roots.add(root)
    return roots


def is_trusted_workspace(root: str, trusted_roots: Iterable[str], operator_roots: AbstractSet[str]) -> bool:
    """True iff a language server may load code the project at ``root`` ships (its own interpreter,
    TypeScript SDK, config files, build scripts): ``root`` is inside an ``lsp.trusted_workspaces``
    entry, or belongs to one of the ``operator_workspace_roots`` worktrees.  A nested clone inside
    such a worktree has its own ``.git`` and is not trusted: the agent may have fetched it."""
    if any(is_inside_workspace(root, t) for t in trusted_roots):
        return True
    return find_git_worktree(root) in operator_roots


def clear_cache() -> None:
    """Clear the workspace-resolution cache (on service shutdown, so re-init doesn't see stale results)."""
    _workspace_cache.clear()


__all__ = [
    "find_git_worktree", "is_inside_workspace", "is_trusted_workspace", "nearest_root", "normalize_path",
    "operator_workspace_roots", "resolve_workspace_for_file", "clear_cache",
]
