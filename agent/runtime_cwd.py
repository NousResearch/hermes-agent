"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the configured working directory (`terminal.cwd`
is bridged to it once at gateway/cron startup; the local CLI leaves it unset and relies on
the launch dir). Reading it in one place keeps the system prompt, tool surfaces, and
context-file discovery agreeing on where the agent lives. Multi-session gateways can pin a
logical cwd via `_SESSION_CWD`.
"""

import contextlib
import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)

# The package/source root (<root>/agent/runtime_cwd.py). A backend launched from or
# self-spawned into this tree (desktop default) must never let an os.getcwd() fallback
# inject this repo's contributor AGENTS.md as project context.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _is_install_tree(p: Path) -> bool:
    """True only when ``p`` IS the package root or sits inside it — ancestors
    (a home dir containing the checkout) are legitimate workspaces."""
    try:
        p = p.resolve()
    except Exception:
        return False
    return p == _PACKAGE_ROOT or _PACKAGE_ROOT in p.parents


def set_session_cwd(cwd: str | None) -> Token:
    """Pin the logical cwd for the current context."""
    return _SESSION_CWD.set((cwd or "").strip())


def clear_session_cwd() -> None:
    _SESSION_CWD.set("")


def reset_session_cwd(token: Token) -> None:
    """Restore the logical cwd that was active before the matching ``set_session_cwd``."""
    _SESSION_CWD.reset(token)


def scope_terminal_cwd() -> str:
    """Scope-aware TERMINAL_CWD value (may be empty) — every cwd consumer reads through this.

    Under gateway multiplexing the per-turn terminal scope carries the active profile's cwd;
    the process-global env var may hold another profile's. Only an ImportError falls back: an
    active refusal scope must raise, not silently resolve the launch profile's cwd.
    """
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def _existing_dir(raw: str, label: str) -> Path | None:
    p = Path(raw).expanduser()
    if p.is_dir():
        return p
    logger.warning("%s does not exist: %s", label, raw)
    return None


def _resolve_configured_cwd(*, override_is_final: bool) -> Path | None:
    """Session override, then TERMINAL_CWD; each validated as a real directory.

    ``override_is_final``: a set-but-missing session override yields None
    instead of falling through to TERMINAL_CWD.
    """
    override = _SESSION_CWD.get()
    override = "" if override is _UNSET else str(override).strip()
    if override:
        p = _existing_dir(override, "configured working directory")
        if p is not None or override_is_final:
            return p
    raw = scope_terminal_cwd().strip()
    return _existing_dir(raw, "TERMINAL_CWD") if raw else None


def resolve_agent_cwd() -> Path:
    """Configured cwd, else the launch dir (os.getcwd()'s OSError on a deleted cwd deliberately propagates)."""
    return _resolve_configured_cwd(override_is_final=False) or Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    """Configured cwd for context-file discovery, or None (build_context_files_prompt then falls back to the
    launch dir). An existing configured path is honored verbatim — including the Hermes source tree, a
    legitimate workspace when developing Hermes; fallback-directory policy lives in the caller."""
    return _resolve_configured_cwd(override_is_final=True)


# --- Workspace identity -------------------------------------------------------

def session_cwd_override() -> str | None:
    """Per-session cwd override, shape-preserving so callers can tell an explicit
    empty session cwd from no session context at all:

    - ``""`` when the bound session explicitly has no cwd (gateway/API turns) —
      callers must NOT fall back to the ambient TERMINAL_CWD in that case;
    - ``None`` when no session context is installed on this task (local CLI) —
      the ambient surface cwd may still be the user's real working directory.
    """
    value = _SESSION_CWD.get()
    if value is _UNSET:
        return None
    return str(value).strip()


def _hermes_home_key() -> str:
    """normalized-comparison key for ``$HERMES_HOME`` ("" when unresolvable)."""
    with contextlib.suppress(Exception):
        from hermes_constants import get_hermes_home
        return os.path.normcase(os.path.realpath(str(get_hermes_home())))
    return ""


def _non_workspace_dirs() -> set[str]:
    """Directories that are never a workspace identity: the filesystem root, the
    user's home, the dir homes live in, plus both POSIX spellings on every host —
    mirrors the Desktop's ``tui_gateway.methods_projects._non_workspace_dirs``
    (keep the two in sync) — plus ``$HERMES_HOME`` itself."""
    home = os.path.realpath(os.path.expanduser("~"))
    candidates = (os.sep, home, os.path.dirname(home), "/home", "/Users")
    dirs = {os.path.normcase(os.path.realpath(p)) for p in candidates if p}
    if key := _hermes_home_key():
        dirs.add(key)
    return dirs


def _is_non_workspace_path(path: str) -> bool:
    """True for the never-a-workspace dirs and anything inside ``$HERMES_HOME``
    (an install/data tree, not a project — unless a declared project owns it,
    which callers check first)."""
    try:
        key = os.path.normcase(os.path.realpath(str(path)))
    except Exception:
        return False
    if not key:
        return False
    if key in _non_workspace_dirs():
        return True
    home_key = _hermes_home_key()
    return bool(home_key) and key.startswith(home_key + os.sep)


def _workspace_name(path: str) -> str:
    """Basename of *path* as a workspace identity ("" for degenerate names)."""
    name = os.path.basename(str(path).rstrip("/\\"))
    return "" if name in ("", ".", "..") else name


def resolve_workspace_identity(cwd: str, *, repo_root: str = "") -> str:
    """Workspace identity for workspace-scoped features (memory-provider
    ``bank_id_template`` placeholders, per-workspace tagging).

    Deterministic, generic cascade — the same order the Desktop's project tree
    resolves a session workspace:

    1. the declared Hermes project owning *cwd* (deepest owning folder wins);
    2. the git repository root's name (the session-stamped *repo_root* first —
       the Desktop folds linked worktrees into the common root; otherwise a
       bounded ``.git`` walk-up, never a git subprocess);
    3. the working directory's basename;
    4. ``""`` when nothing usable is found (never-a-workspace dirs, Hermes home
       internals, empty input).

    Never raises: a missing/corrupt projects DB, a vanished path or an import
    failure all degrade to the next arm of the cascade.
    """
    raw = str(cwd or "").strip()
    if not raw:
        return ""
    try:
        path = os.path.abspath(os.path.expanduser(raw))
    except Exception:
        return ""

    # 1. Declared Hermes project (per-profile projects.db). Best-effort and
    #    read-only: never creates the DB, never raises.
    with contextlib.suppress(Exception):
        from hermes_cli.projects_db import connect_closing, project_for_path, projects_db_path
        if projects_db_path().exists():
            with connect_closing() as conn:
                project = project_for_path(conn, path)
            if project is not None and (slug := str(project.slug or "").strip()):
                return slug

    if _is_non_workspace_path(path):
        return ""

    # 2. Git repository root (session-stamped root first).
    root = str(repo_root or "").strip()
    if root and os.path.isdir(root) and not _is_non_workspace_path(root):
        return _workspace_name(root)
    with contextlib.suppress(Exception):
        from agent.skill_utils import find_project_root
        if (found := find_project_root(Path(path))) is not None and not _is_non_workspace_path(str(found)):
            return _workspace_name(str(found))

    # 3. Working-directory basename.
    return _workspace_name(path)
