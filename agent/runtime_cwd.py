"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the working directory: gateway and cron bridge
`terminal.cwd` into it, while a local CLI carries its launch dir (see `claim_local_cli_cwd`).
Reading it in one place keeps the system prompt, tool surfaces, and context-file discovery
agreeing on where the agent lives. Multi-session gateways can pin a logical cwd via
`_SESSION_CWD`.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)

# A local CLI runs where it was launched (#19214): the shell cwd, ``--in DIR`` or a resumed
# session's directory, later moved by ``-w``, ``/worktree`` and ``/resume``. Each of those writes
# TERMINAL_CWD, and a later config bridge (dotenv reload, the terminal tool's fallback) must not
# re-pin it to ``terminal.cwd``, which stays the gateway/cron default (#86411). Process-local:
# gateways, cron, ACP and the dashboard never claim it.
_cli_owns_cwd = False

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


def scoped_session_cwd() -> str:
    """Return the current session's declared cwd without local path validation.

    Remote and container paths may not exist on the Hermes host. Callers that only need
    logical workspace identity should preserve the declared value instead of resolving it.
    """
    value = _SESSION_CWD.get()
    return "" if value is _UNSET else str(value).strip()


def reset_session_cwd(token: Token) -> None:
    """Restore the logical cwd that was active before the matching ``set_session_cwd``."""
    _SESSION_CWD.reset(token)


def claim_local_cli_cwd() -> None:
    """Mark this process as a local CLI that owns its working directory (``_cli_owns_cwd``).

    Called once the launch dir is final. Every ``hermes`` process bridges config at import, before
    it knows it is a CLI, so a ``terminal.cwd`` already written to TERMINAL_CWD yields to it here.
    """
    global _cli_owns_cwd
    _cli_owns_cwd = True
    if (os.environ.get("TERMINAL_ENV") or "local") == "local" and os.environ.get("TERMINAL_CWD", "").strip():
        os.environ["TERMINAL_CWD"] = os.getcwd()


def config_owns_terminal_cwd(value: Any, backend: str, *, process_global: bool) -> bool:
    """Whether a ``terminal.cwd`` value may set TERMINAL_CWD: never a launch-dir placeholder, nor a
    process-wide bridge in a local CLI that claimed its working directory."""
    if str(value or "").strip() in {".", "auto", "cwd"}:
        return False
    return not (process_global and _cli_owns_cwd and backend == "local")


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


def _resolve_configured_cwd(
    *, override_is_final: bool, include_session_override: bool = True,
) -> Path | None:
    """Session override, then TERMINAL_CWD; each validated as a real directory.

    ``override_is_final``: a set-but-missing session override yields None
    instead of falling through to TERMINAL_CWD.
    ``include_session_override``: skip a session cwd known to be a launch artifact
    while still consulting the active profile's TERMINAL_CWD.
    """
    if include_session_override:
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


def resolve_context_cwd(*, include_session_override: bool = True) -> Path | None:
    """Configured cwd for context-file discovery, or None (build_context_files_prompt then falls back to the
    launch dir). An existing configured path is honored verbatim — including the Hermes source tree, a
    legitimate workspace when developing Hermes; fallback-directory policy lives in the caller.

    Launch-artifact callers can skip the session override while still honoring the active profile's
    TERMINAL_CWD.
    """
    return _resolve_configured_cwd(
        override_is_final=True, include_session_override=include_session_override,
    )
