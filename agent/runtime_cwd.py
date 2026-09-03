"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the configured working directory
(design #19214/#19242: `terminal.cwd` is bridged once to `TERMINAL_CWD` at
gateway/cron startup). The local-CLI backend deliberately leaves it unset and
relies on the launch dir. Reading it in one place keeps the system prompt, the
tool surfaces, and context-file discovery agreeing on where the agent lives.

Multi-session gateways can pin a logical cwd via the `_SESSION_CWD`
contextvar; CLI/cron fall through to `TERMINAL_CWD`/launch cwd.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)

# The Python package/source root (this file lives at <root>/agent/runtime_cwd.py).
# When a backend is launched from, or self-spawns into, this tree (the desktop
# app default), an os.getcwd() fallback would inject this repo's contributor
# AGENTS.md as authoritative project context. Context discovery must never
# resolve here.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _is_install_tree(p: Path) -> bool:
    # True only when p IS the package root or sits inside it. Ancestors of the
    # package root (a user home that happens to contain the checkout, a --user
    # site-packages parent) are legitimate workspaces and must not be blocked.
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


def _session_cwd_override() -> str:
    value = _SESSION_CWD.get()
    if value is _UNSET:
        return ""
    return str(value).strip()


def _terminal_cwd_env() -> str:
    """Scope-aware TERMINAL_CWD read (tools.terminal_scope.terminal_env).

    Under gateway multiplexing the per-turn terminal scope carries the active
    profile's cwd; the process-global env var may hold another profile's
    value. Only an import failure falls back: an active refusal scope must
    raise, not silently resolve the launch profile's cwd.
    """
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def scope_terminal_cwd() -> str:
    """Public wrapper — the scope-aware TERMINAL_CWD value (may be empty).

    Shared by agent_init / skill_utils / code_execution_tool so every cwd
    consumer reads through the per-turn terminal scope under gateway
    multiplexing instead of the process-global env var.
    """
    return _terminal_cwd_env()


def resolve_agent_cwd() -> Path:
    override = _session_cwd_override()
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            return p
        logger.warning("configured working directory does not exist: %s", override)
    raw = _terminal_cwd_env().strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
        logger.warning("TERMINAL_CWD does not exist: %s", raw)
    return Path(os.getcwd())


# Project-root markers for resolve_project_scope (issue #33638).
# Mirrors _marker_root / _PROJECT_MARKERS / _CONTEXT_FILES in coding_context.py
# but narrower: only the markers the memory project-scoping feature cares about.
_PROJECT_SCOPE_MARKERS = (
    ".git",
    "AGENTS.md",
    ".hermes-memory.md",
)


def resolve_project_scope() -> str:
    """Map the agent's working directory to a project scope identifier.

    Returns the basename of the nearest ancestor project root, or ``""`` when
    the cwd is not inside a recognisable project (or is in the install tree,
    ``$HOME``, or a shared temp dir — none of those are projects).

    This feeds the ``memory.project_scoping`` feature (issue #33638) and must
    agree with :func:`build_context_files_prompt`'s cwd semantics.
    """
    import tempfile

    cwd = resolve_context_cwd()
    if cwd is None:
        cwd = Path(os.getcwd())

    # Never scope to the Hermes repo itself.
    if _is_install_tree(cwd):
        return ""

    current = cwd.resolve()

    # Resolve $HOME and shared temp root once, matching _marker_root behaviour.
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
    except Exception:
        temp_root = None

    for depth, parent in enumerate([current, *current.parents]):
        if depth > 6:
            break
        # Stop the walk at $HOME or the shared temp root — their parents are
        # never project roots, so continuing past them is wasted work and can
        # produce false positives from unrelated markers in parent directories.
        if parent == home or (temp_root is not None and parent == temp_root):
            break
        for marker in _PROJECT_SCOPE_MARKERS:
            try:
                if (parent / marker).exists():
                    return parent.name
            except PermissionError:
                # Permission error on any ancestor stops marker checks at
                # this level but does not abort the walk — continue upward.
                continue
    return ""


def resolve_context_cwd() -> Path | None:
    # None means "no configured cwd": build_context_files_prompt then falls back
    # to the launch dir (os.getcwd()), correct for a local CLI launched inside a
    # real project. A configured path is validated here (previously it was passed
    # through unchecked, diverging from resolve_agent_cwd). An explicitly
    # configured path is otherwise honored verbatim — including the Hermes
    # source tree itself, which is a legitimate workspace when the user is
    # developing Hermes (per-surface policy for fallback-picked directories
    # lives in build_context_files_prompt; see #64590).
    override = _session_cwd_override()
    if override:
        p = Path(override).expanduser()
        if not p.is_dir():
            logger.warning("configured working directory does not exist: %s", override)
        else:
            return p
        return None
    raw = _terminal_cwd_env().strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_dir():
            logger.warning("TERMINAL_CWD does not exist: %s", raw)
        else:
            return p
    return None
