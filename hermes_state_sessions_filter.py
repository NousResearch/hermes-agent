"""``sessions s`` WHERE builders shared by the SessionSessionsMixin list/count paths so the
counts line up with listed rows. Moved out of :mod:`hermes_state_sessions` (kept its cap)."""

from __future__ import annotations

from typing import Any, Sequence

from hermes_state_common import (
    _LISTABLE_CHILD_SQL,
    _source_match_clause,
    _sql_json_extract,
    _placeholders as _session_ids_placeholders,
    escape_like as _escape_like,
)


def _delegate_from_json(col: str = "model_config") -> str:
    return _sql_json_extract(col, "$._delegate_from")


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


def _where_sql(clauses: list[str], lead: str = "") -> str:
    """``WHERE a AND b`` (with *lead* prefix) or "" when there are no clauses."""
    return f"{lead}WHERE {' AND '.join(clauses)}" if clauses else ""


def _session_filter_where(
    *, exclude_children: bool = False, source: str | None = None, sources: list[str] | None = None,
    session_key: str | None = None, exclude_sources: list[str] | None = None, cwd_prefix: str | None = None,
    min_message_count: int = 0, archived_only: bool = False, include_archived: bool = False,
    include_subagents: bool = False, local_created_sources: Sequence[str] | None = None,
) -> tuple[list[str], list[Any]]:
    """Shared ``sessions s`` WHERE builder so counts line up with listed rows. ``exclude_children``
    hides sub-agent runs and compression continuations but keeps branch/reset children
    (``_LISTABLE_CHILD_SQL``); ``include_subagents`` re-admits the sub-agent runs only
    (``sessions.show_subagents``). Clause order is part of the SQL text contract."""
    where: list[str] = []
    params: list[Any] = []
    if exclude_children and include_subagents:
        where.append(f"({_LISTABLE_CHILD_SQL} OR {_delegate_from_json('s.model_config')} IS NOT NULL)")
    elif exclude_children:
        where += [_LISTABLE_CHILD_SQL, f"{_delegate_from_json('s.model_config')} IS NULL"]
    # Show roots and user-visible branch/reset sessions, while still hiding sub-agent runs and compression
    # continuations. All four carry parent_session_id, so the shared predicate classifies the edge from
    # stable markers plus legacy-compatible parent metadata. Branch sessions are identified two ways, OR'd
    # for robustness: 1. A stable ``_branched_from`` marker in model_config, written by /branch at creation
    # time. This survives the parent being reopened and re-ended with a different end_reason (e.g.
    # tui_shutdown overwriting 'branched'), which otherwise hides the branch — see issue #20856. 2. The
    # legacy heuristic (parent ended with 'branched' before the child started), covering branch sessions
    # created before the marker existed.
    include_sources = [source] if source else list(sources or [])
    source_clause, source_params = _source_match_clause(include_sources, local_created_sources)
    for clause, values in (
        (source_clause, source_params),
        ("s.session_key = ?", [session_key] if session_key else []),
        (f"s.source NOT IN ({_session_ids_placeholders(exclude_sources or ())})", exclude_sources or []),
        (_cwd_prefix_clause(cwd_prefix) if cwd_prefix else ("", [])),
        ("s.message_count >= ?", [min_message_count] if min_message_count > 0 else []),
    ):
        if values:
            where.append(clause)
            params.extend(values)
    if archived_only:
        where.append("s.archived = 1")
    elif not include_archived:
        where.append("s.archived = 0")
    return where, params
