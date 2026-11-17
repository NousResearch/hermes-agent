"""A messaging chat's bound workspace: ``/workspace DIR`` (and, opt-in, its per-thread fork).

The binding lives on the chat's routing entry (``metadata['workspace']``), next to the terminal cwd
record ``gateway.session_terminal_cwd`` keeps there, so it survives restarts, compression splits and
``/new``, and one place answers "where does this chat work". ``ensure_binding`` resolves it once per
turn, before the first reader (the inbound ``@`` expansion, else the turn itself); ``workspace_turn``
then pins the session's logical cwd (system prompt, context files) and registers the root as the
terminal and file tools' cwd override for the turn's task id. The root is fixed for the conversation,
so the cached system prompt stays valid; only ``/workspace`` moves it, and that evicts the cached agent
like any other deliberate settings change. ``/workspace clear`` stores an explicit "global directory"
choice, so a configured ``gateway.workspaces.dir`` does not come back on the next turn.

With ``gateway.workspaces.per_thread: worktree`` the root is the thread's fork of the bound
directory (``gateway.session_workspace_fork``), made on the first turn that finds none and rebuilt
when the stale-worktree sweep removed a tree of it. Its trees are locked only while a turn runs.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from gateway.session_terminal_cwd import _routing_entry
from gateway.session_workspace_fork import (
    Fork, WorkspaceSettings, fork_workspace, local_repos, lock_trees, unlock_trees,
)

logger = logging.getLogger(__name__)

METADATA_KEY = "workspace"
_CLEARED = {"cleared": True}  # /workspace clear: terminal.cwd, even when gateway.workspaces.dir is set


def binding(runner, session_key: str | None) -> dict[str, Any]:
    """The chat's stored binding: ``{"dir": <bound by /workspace>, "root": <where it works>, ...}``."""
    entry = _routing_entry(runner, session_key) if session_key else None
    value = entry.metadata.get(METADATA_KEY) if entry is not None else None
    return dict(value) if isinstance(value, dict) else {}


def bound_root(runner, session_key: str | None) -> str:
    """The directory this chat's turns run in, or ``""`` when it follows ``terminal.cwd``."""
    stored = binding(runner, session_key)
    root = str(stored.get("root") or stored.get("dir") or "")
    return root if root and os.path.isdir(root) else ""


def reference_root(stored: dict[str, Any]) -> str:
    """The directory ``@`` references may read: the bound directory of a fork (its repositories'
    worktrees live under each repository, outside the fork root), else the root itself."""
    return str(stored.get("of") or "") if stored.get("trees") else str(stored.get("root") or "")


def bound_dirs(runner, session_key: str | None) -> list[tuple[str, str]]:
    """``[(label, checkout)]`` that ``/diff``, ``/rollback`` and ``/undo`` read: each repository
    worktree of a folder-of-repositories fork (its root is only an index), labelled by repository, else
    the bound root alone; ``[]`` when the chat follows ``terminal.cwd``."""
    stored, root = binding(runner, session_key), bound_root(runner, session_key)
    trees = [(Path(repo).name, tree) for repo, tree in stored.get("trees") or () if os.path.isdir(tree)]
    if not root:
        return []
    return trees if trees and root not in {tree for _label, tree in trees} else [("", root)]


def _fork_current(stored: dict[str, Any], directory: str) -> bool:
    """Every saved tree is still there and a folder-of-repositories fork covers every repository the
    directory holds now (an on-demand clone the agent made joins the fork on the next turn)."""
    trees, root = stored.get("trees") or (), str(stored.get("root") or "")
    return all(os.path.isdir(tree) for _repo, tree in trees) and (
        any(tree == root for _repo, tree in trees) or sorted(repo for repo, _tree in trees) == local_repos(directory))


def _resolve_root(session_key: str, stored: dict[str, Any], settings: WorkspaceSettings) -> dict[str, Any]:
    """The binding this turn runs under: forked when the settings ask for it and no current fork exists."""
    if stored.get("cleared"):
        return stored
    directory = str(stored.get("dir") or settings.dir)
    if not directory or not os.path.isdir(directory):
        return {}
    wanted = {"dir": stored.get("dir") or "", "of": directory, "mode": settings.per_thread}
    root = str(stored.get("root") or "")
    if (all(stored.get(k) == v for k, v in wanted.items()) and root and os.path.isdir(root)
            and (settings.per_thread != "worktree" or _fork_current(stored, directory))):
        return stored
    fork = fork_workspace(directory, session_key, settings) if settings.per_thread == "worktree" else None
    return {**wanted, "root": fork.root if fork else directory,
            "trees": [list(pair) for pair in fork.trees] if fork else []}


def ensure_binding(runner, session_key: str | None, user_config: Any) -> dict[str, Any]:
    """Resolve and save the chat's binding (call inside its profile scope); ``{}`` when it has none."""
    if not session_key or _routing_entry(runner, session_key) is None:
        return {}
    stored = binding(runner, session_key)
    current = _resolve_root(session_key, stored, WorkspaceSettings.from_config(user_config))
    if current != stored:
        runner.session_store.set_session_metadata(session_key, METADATA_KEY, current or None)
    return current


@contextmanager
def workspace_turn(runner, session_key: str | None, session_id: str | None, user_config: Any) -> Iterator[str]:
    """Bind the chat's workspace for one turn (inside the turn's profile scope, before the agent
    runs); yields the root, or ``""`` when the chat has none. A fork's trees stay locked for the turn."""
    current = ensure_binding(runner, session_key, user_config) if session_id else {}
    root = str(current.get("root") or "")
    if not root:
        yield ""
        return
    from agent.runtime_cwd import set_session_cwd
    from tools.terminal_tool import register_task_env_overrides, resolve_task_overrides, workspace_key_for
    set_session_cwd(root)
    overrides = resolve_task_overrides(session_id)
    if overrides.get("cwd") != root:
        register_task_env_overrides(session_id, {**overrides, "cwd": root, "cwd_source": "session",
                                                 "workspace_key": workspace_key_for(root)})
    fork = Fork(root, tuple(tuple(pair) for pair in current.get("trees") or ()))
    lock_trees(fork)
    try:
        yield root
    finally:
        unlock_trees(fork)


def forget_turn_state(runner, session_key: str, session_id: str) -> None:
    """Drop what earlier turns derived from the old binding: the in-process cwd records, the tools'
    cwd override, the saved terminal cwd and the cached agent (its prompt names the old directory)."""
    from gateway.session_policy import unbind_turn_workspace
    from gateway.session_terminal_cwd import forget_turn_cwd
    from tools.terminal_tool import clear_session_cwd
    unbind_turn_workspace(session_id)
    clear_session_cwd(session_key)
    forget_turn_cwd(runner, session_key)
    runner._evict_cached_agent(session_key)



def _target(arg: str, global_cwd: str) -> tuple[bool, dict[str, Any] | None, str]:
    """``(valid, binding, reply)`` for a ``/workspace`` argument other than ``status``."""
    from agent.i18n import t
    if arg.lower() in {"clear", "reset", "default"}:
        return True, dict(_CLEARED), t("gateway.workspace.cleared", cwd=global_cwd)
    path = Path(os.path.expanduser(arg))
    if not path.is_absolute():
        return False, None, t("gateway.workspace.relative_rejected")
    if not path.is_dir():
        return False, None, t("gateway.workspace.not_dir", path=str(path))
    resolved = str(path.resolve())
    return True, {"dir": resolved}, t("gateway.workspace.set", cwd=resolved)


async def handle_workspace_command(runner, event) -> str:
    """``/workspace [DIR|clear|status]`` (alias ``/cwd``): bind this chat to an absolute directory."""
    from agent.i18n import t
    from tools.terminal_scope import terminal_env
    entry = await runner.async_session_store.get_or_create_session(event.source)
    key, arg = entry.session_key, event.get_command_args().strip()
    global_cwd = terminal_env("TERMINAL_CWD", "") or await asyncio.to_thread(os.path.expanduser, "~")
    if arg.lower() in {"", "status", "show"}:
        root = await asyncio.to_thread(bound_root, runner, key)
        return t("gateway.workspace.status", cwd=root or global_cwd, global_cwd=global_cwd,
                 source=t("gateway.workspace.source_session" if root else "gateway.workspace.source_global"))
    valid, new, reply = await asyncio.to_thread(_target, arg, global_cwd)
    if not valid:
        return reply
    try:
        await runner.async_session_store.set_session_metadata(key, METADATA_KEY, new)
    except Exception as exc:  # health: allow BLE001 -- a failed save is reported to the user, not raised into the adapter
        logger.warning("Could not save the workspace of %s: %s", key, exc)
        return t("gateway.workspace.save_failed", error=exc)
    await asyncio.to_thread(forget_turn_state, runner, key, entry.session_id)
    return reply
