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

# Terminal backends whose TERMINAL_CWD refers to a path on a REMOTE machine /
# container (ssh, docker, ...), NOT on the host where Hermes runs. For these,
# TERMINAL_CWD must never be stat'd from the host process: e.g. a root@ssh
# profile sets TERMINAL_CWD=/root, and an unprivileged local user cannot even
# stat /root/.git — prompting a PermissionError that aborts system-prompt
# construction. Single source of truth: prompt_builder and env_probe import
# this set instead of redefining it, so a backend added here is automatically
# honored by all three consumers.
_REMOTE_TERMINAL_BACKENDS = frozenset({
    "docker", "singularity", "modal", "daytona", "ssh",
    "vercel_sandbox", "managed_modal",
})


def _terminal_env_is_remote() -> bool:
    return (os.getenv("TERMINAL_ENV") or "local").strip().lower() in _REMOTE_TERMINAL_BACKENDS

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


def resolve_agent_cwd() -> Path:
    override = _session_cwd_override()
    if override:
        p = Path(override).expanduser()
        if p.is_dir():
            return p
        logger.warning("configured working directory does not exist: %s", override)
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    # For remote terminal backends TERMINAL_CWD is a path on the remote
    # machine/container — never stat it from the host process (it may be
    # unreadable, e.g. root@ssh → /root). Fall back to the launch dir.
    if raw and not _terminal_env_is_remote():
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
    override = _session_cwd_override()
    if override:
        p = Path(override).expanduser()
        if not p.is_dir():
            logger.warning("configured working directory does not exist: %s", override)
        else:
            return p
        return None
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    # Remote backends: TERMINAL_CWD is a remote path; do not validate/return
    # it from the host process. None lets build_context_files_prompt fall back
    # to the launch dir.
    if raw and not _terminal_env_is_remote():
        p = Path(raw).expanduser()
        if not p.is_dir():
            logger.warning("TERMINAL_CWD does not exist: %s", raw)
        else:
            return p
    return None
