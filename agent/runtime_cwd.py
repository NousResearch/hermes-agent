"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the configured working directory
(design #19214/#19242: `terminal.cwd` is bridged once to `TERMINAL_CWD` at
gateway/cron startup). The local-CLI backend deliberately leaves it unset and
relies on the launch dir. Reading it in one place keeps the system prompt, the
tool surfaces, and context-file discovery agreeing on where the agent lives.

Gateway sessions and cron jobs can pin a logical cwd via the `_SESSION_CWD`
contextvar; local CLI calls fall through to `TERMINAL_CWD`/launch cwd.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)
_SESSION_CWD_AUTHORITATIVE: ContextVar = ContextVar(
    "HERMES_SESSION_CWD_AUTHORITATIVE", default=False
)
_SESSION_CWD_AUTHORITY_SCOPE: ContextVar = ContextVar(
    "HERMES_SESSION_CWD_AUTHORITY_SCOPE", default=None
)

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


def set_authoritative_session_cwd(cwd: str | None) -> tuple[Token, Token, Token]:
    """Pin a cwd that command dispatch must prefer over cached session state."""
    cwd_token = set_session_cwd(cwd)
    authority_token = _SESSION_CWD_AUTHORITATIVE.set(True)
    scope_token = _SESSION_CWD_AUTHORITY_SCOPE.set(object())
    return cwd_token, authority_token, scope_token


def reset_authoritative_session_cwd(tokens: tuple[Token, Token, Token]) -> None:
    """Restore state captured by :func:`set_authoritative_session_cwd`."""
    cwd_token, authority_token, scope_token = tokens
    _SESSION_CWD_AUTHORITY_SCOPE.reset(scope_token)
    _SESSION_CWD_AUTHORITATIVE.reset(authority_token)
    _SESSION_CWD.reset(cwd_token)


def clear_session_cwd() -> None:
    _SESSION_CWD.set("")
    _SESSION_CWD_AUTHORITATIVE.set(False)
    _SESSION_CWD_AUTHORITY_SCOPE.set(None)


def _session_cwd_override() -> str:
    value = _SESSION_CWD.get()
    if value is _UNSET:
        return ""
    return str(value).strip()


def resolve_authoritative_tool_cwd() -> str:
    """Return the context cwd only when command dispatch must treat it as fixed."""
    if not _SESSION_CWD_AUTHORITATIVE.get():
        return ""
    return _session_cwd_override()


def resolve_authoritative_cwd_scope() -> object | None:
    """Return the active authority scope used to validate live cwd records."""
    if not _SESSION_CWD_AUTHORITATIVE.get():
        return None
    return _SESSION_CWD_AUTHORITY_SCOPE.get()


def resolve_tool_cwd() -> str:
    """Return the task-local cwd, falling back to the legacy process carrier.

    Tool backends may use remote or container paths that do not exist on the
    Hermes host, so this accessor deliberately does not validate the path.
    """
    override = _session_cwd_override()
    if override:
        return override
    return os.environ.get("TERMINAL_CWD", "").strip()


def resolve_session_cwd() -> Path | None:
    """Return the validated task-local cwd, without falling back to process env."""
    override = _session_cwd_override()
    if not override:
        return None
    p = Path(override).expanduser()
    return p if p.is_dir() else None


def resolve_agent_cwd() -> Path:
    authoritative_cwd = resolve_authoritative_tool_cwd()
    if authoritative_cwd:
        # Preserve the intended task path even if it disappears mid-run. Users
        # should get a cwd error, never silent execution in another workspace.
        return Path(authoritative_cwd).expanduser()
    session_cwd = resolve_session_cwd()
    if session_cwd is not None:
        return session_cwd
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
        logger.warning("TERMINAL_CWD does not exist: %s", raw)
    return Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    # An authoritative task path must remain authoritative even if it disappears
    # after job admission. Returning the missing path makes discovery yield no
    # project files; returning None would fall back to the scheduler process cwd
    # and inject unrelated AGENTS.md instructions.
    authoritative_cwd = resolve_authoritative_tool_cwd()
    if authoritative_cwd:
        return Path(authoritative_cwd).expanduser()

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
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_dir():
            logger.warning("TERMINAL_CWD does not exist: %s", raw)
        else:
            return p
    return None
