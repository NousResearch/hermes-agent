"""SQL WHERE-clause builders shared by session listing, counting and maintenance."""

from hermes_state_common import escape_like as _escape_like


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
