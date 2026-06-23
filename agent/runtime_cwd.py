"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the configured working directory
(design #19214/#19242: `terminal.cwd` is bridged once to `TERMINAL_CWD` at
gateway/cron startup). The local-CLI backend deliberately leaves it unset and
relies on the launch dir. Reading it in one place keeps the system prompt, the
tool surfaces, and context-file discovery agreeing on where the agent lives.

Multi-session gateways can pin a logical cwd via the `_SESSION_CWD`
contextvar; CLI/cron fall through to `TERMINAL_CWD`/launch cwd.
"""

import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)
_SESSION_WORKTREE_MAP: ContextVar = ContextVar("HERMES_SESSION_WORKTREE_MAP", default=_UNSET)


def set_session_cwd(cwd: str | None) -> Token:
    """Pin the logical cwd for the current context."""
    return _SESSION_CWD.set((cwd or "").strip())


def clear_session_cwd() -> None:
    _SESSION_CWD.set("")


def set_session_worktree_map(mapping: dict[str, str] | None) -> Token:
    """Pin repo-root → session-worktree routing for the current context."""
    normalized: dict[str, str] = {}
    for root, worktree in (mapping or {}).items():
        try:
            root_path = Path(str(root)).expanduser().resolve()
            worktree_path = Path(str(worktree)).expanduser().resolve()
        except Exception:
            continue
        if str(root_path) and str(worktree_path):
            normalized[str(root_path)] = str(worktree_path)
    return _SESSION_WORKTREE_MAP.set(normalized)


def clear_session_worktree_map() -> None:
    _SESSION_WORKTREE_MAP.set({})


def _session_worktree_map() -> dict[str, str]:
    value = _SESSION_WORKTREE_MAP.get()
    if value is _UNSET or not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if str(k) and str(v)}


def map_session_path_to_worktree(path: str | Path) -> Path:
    """Route absolute repo paths into this session's auto-worktree.

    Gateway sessions can start outside a configured repo and later pass an
    absolute ``workdir`` or file path inside that repo.  When a repo→worktree
    map is active, translate ``<repo>/sub/path`` to
    ``<session-worktree>/sub/path`` so late moves stay isolated.  Relative paths
    are left untouched and resolved by the caller against the active cwd.
    """
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        return raw
    try:
        resolved = raw.resolve()
    except Exception:
        resolved = raw

    for root_s, worktree_s in _session_worktree_map().items():
        root = Path(root_s)
        worktree = Path(worktree_s)
        try:
            resolved.relative_to(worktree)
            return resolved
        except ValueError:
            pass
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        return (worktree / rel).resolve()
    return resolved


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
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
    return Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    # None means "no configured cwd": build_context_files_prompt then falls back
    # to the launch dir (os.getcwd()) — correct for the local CLI. The gateway
    # avoids slurping its install dir by setting TERMINAL_CWD (see system_prompt.py)
    # or, per session, the _SESSION_CWD contextvar above.
    override = _session_cwd_override()
    if override:
        return Path(override).expanduser()
    raw = os.environ.get("TERMINAL_CWD", "").strip()
    return Path(raw).expanduser() if raw else None
