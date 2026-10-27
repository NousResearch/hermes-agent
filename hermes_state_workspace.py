"""Workspace identity for session rows: the folder a conversation belongs to, and its SQL form.

``workspace_key`` is the one grouping rule every folder-scoped surface shares (``hermes -c``,
``--resume latest``, ``sessions list``), and ``_workspace_key_clause`` is the predicate that
implements it, so a listing and a "most recent here" lookup can never disagree about which
sessions are "this folder's".
"""

from __future__ import annotations

from typing import Any, Optional

from hermes_state_common import escape_like as _escape_like


def workspace_key(row: dict[str, Any]) -> Optional[str]:
    """Workspace grouping key: git repo root, else cwd, else None (branch excluded: a checkout must not
    fragment history)."""
    return (row.get("git_repo_root") or "").strip() or (row.get("cwd") or "").strip() or None


def _cwd_prefix_clause(cwd_prefix: str) -> tuple[str, list[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    # ``_``/``%`` are LIKE wildcards but ordinary path characters: unescaped, a
    # prefix also matches sibling directories. The ``=`` arm keeps the raw prefix.
    esc = _escape_like(prefix)
    return (
        "(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\' OR s.cwd LIKE ? ESCAPE '\\')",
        [prefix, f"{esc}/%", f"{esc}\\\\%"],
    )


def _workspace_key_clause(key: str) -> tuple[str, list[str]]:
    """WHERE for ``workspace_key(row) == key``: git_repo_root equals ``key``, or (rows predating
    per-session git metadata) cwd is at/under ``key``."""
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = ? OR (COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )
