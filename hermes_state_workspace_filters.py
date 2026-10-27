"""Exact workspace and cwd filters for session listing."""
from typing import Any, Sequence

from hermes_state_common import (
    _LISTABLE_CHILD_SQL, _placeholders, _source_match_clause, _sql_json_extract,
    escape_like as _escape_like,
)


def session_filter_where(
    *, exclude_children: bool = False, source: str | None = None, sources: list[str] | None = None,
    session_key: str | None = None, exclude_sources: list[str] | None = None, cwd_prefix: str | None = None,
    min_message_count: int = 0, archived_only: bool = False, include_archived: bool = False,
    include_subagents: bool = False, local_created_sources: Sequence[str] | None = None,
) -> tuple[list[str], list[Any]]:
    """Build the same session filters for row counts and listings."""
    where: list[str] = []
    params: list[Any] = []
    delegate_from = _sql_json_extract("s.model_config", "$._delegate_from")
    if exclude_children and include_subagents:
        where.append(f"({_LISTABLE_CHILD_SQL} OR {delegate_from} IS NOT NULL)")
    elif exclude_children:
        where += [_LISTABLE_CHILD_SQL, f"{delegate_from} IS NULL"]
    include_sources = [source] if source else list(sources or [])
    source_clause, source_params = _source_match_clause(include_sources, local_created_sources)
    for clause, values in (
        (source_clause, source_params),
        ("s.session_key = ?", [session_key] if session_key else []),
        (f"s.source NOT IN ({_placeholders(exclude_sources or ())})", exclude_sources or []),
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


def _workspace_key_clause(key: str) -> tuple[str, list[str]]:
    """WHERE for ``workspace_key(row) == key``: git_repo_root equals ``key``, or (rows predating
    per-session git metadata) cwd is at/under ``key``."""
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = ? OR (COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )


def _cwd_prefix_clause(cwd_prefix: str) -> tuple[str, list[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    # ``_``/``%`` are LIKE wildcards but ordinary path characters: unescaped, a
    # prefix also matches sibling directories. The ``=`` arm keeps the raw prefix.
    esc = _escape_like(prefix)
    return (
        "(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\' OR s.cwd LIKE ? ESCAPE '\\')",
        [prefix, f"{esc}/%", f"{esc}\\\\%"],
    )

