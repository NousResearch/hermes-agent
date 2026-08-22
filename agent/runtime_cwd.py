"""Single source of truth for agent execution and context-file directories.

`TERMINAL_CWD` is the runtime carrier for the configured working directory
(design #19214/#19242: `terminal.cwd` is bridged once to `TERMINAL_CWD` at
gateway/cron startup). The local-CLI backend deliberately leaves it unset and
relies on the launch dir. Reading it in one place keeps the system prompt and
tool surfaces agreeing on where the agent executes.

Multi-session gateways can pin a logical execution cwd via `_SESSION_CWD`.
Multiplex gateways independently pin `_CONTEXT_FILE_CWD` to the routed
profile home so profile instructions never follow a shared process cwd.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)
_CONTEXT_FILE_CWD: ContextVar = ContextVar("HERMES_CONTEXT_FILE_CWD", default=_UNSET)

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


def set_context_file_cwd(cwd: str | None) -> Token:
    """Pin context-file discovery without changing the agent execution cwd."""
    return _CONTEXT_FILE_CWD.set((cwd or "").strip())


def reset_context_file_cwd(token: Token) -> None:
    """Restore the context-file discovery override for a nested runtime scope."""
    _CONTEXT_FILE_CWD.reset(token)


def _session_cwd_override() -> str:
    value = _SESSION_CWD.get()
    if value is _UNSET:
        return ""
    return str(value).strip()


def _context_file_cwd_override() -> str:
    value = _CONTEXT_FILE_CWD.get()
    if value is _UNSET:
        return ""
    return str(value).strip()


def is_context_file_cwd_scoped() -> bool:
    """Return whether instruction discovery has an independent context scope."""
    return bool(_context_file_cwd_override())


def resolve_agent_cwd() -> Path:
    override = _session_cwd_override()
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            return p
        logger.warning("configured working directory does not exist: %s", override)
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
        logger.warning("TERMINAL_CWD does not exist: %s", raw)
    return Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    # None means "no configured cwd": build_context_files_prompt then falls back
    # to the launch dir (os.getcwd()), correct for a local CLI launched inside a
    # real project. A configured path is validated here (previously it was passed
    # through unchecked, diverging from resolve_agent_cwd). An explicitly
    # configured path is otherwise honored verbatim — including the Hermes
    # source tree itself, which is a legitimate workspace when the user is
    # developing Hermes (per-surface policy for fallback-picked directories
    # lives in build_context_files_prompt; see #64590).
    context_override = _context_file_cwd_override()
    if context_override:
        p = Path(context_override).expanduser()
        if not p.is_dir():
            # Stay authoritative anyway. None here does NOT mean "no context
            # files" — build_context_files_prompt reads it as "fall back to
            # os.getcwd()", which under multiplex is the shared gateway launch
            # directory, silently restoring the cross-profile leak this scope
            # exists to prevent. Honouring the missing path instead makes
            # discovery find nothing, which is the correct outcome, and the
            # prompt still records the scope. Logged at error level because the
            # same directory is simultaneously the HERMES_HOME override, so a
            # misconfigured route must be visible rather than merely degraded.
            logger.error(
                "configured context-file directory does not exist: %s — "
                "instruction discovery will find nothing (not falling back to "
                "the execution cwd)",
                context_override,
            )
        return p
    override = _session_cwd_override()
    if override:
        p = Path(override).expanduser()
        if not p.is_dir():
            logger.warning("configured working directory does not exist: %s", override)
        else:
            return p
        return None
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_dir():
            logger.warning("TERMINAL_CWD does not exist: %s", raw)
        else:
            return p
    return None
