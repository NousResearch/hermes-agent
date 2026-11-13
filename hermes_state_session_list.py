"""Session-list page projection and title-prefix filter for ``SessionDB.list_sessions_rich``."""
from typing import Any

from hermes_state_common import _COMPRESSION_CHILD_SQL, _non_continuation_child_sql, escape_like

# SQLite's LIKE folds ASCII case, so the escaped LIKE is paired with an exact substr compare.
_TITLE_PREFIX_SQL = "{a}.title LIKE ? ESCAPE '\\' AND substr({a}.title, 1, ?) = ?"


def title_prefix_clause(prefix: str, *, lineage: bool) -> tuple[str, list[Any]]:
    """``sessions s`` WHERE clause admitting rows whose title starts with *prefix* (case-sensitive).

    *lineage*: the list shows one row per compressed conversation (its root, projected onto the
    tip), but a compression rotation moves the title onto the tip and clears the ancestor's
    (``_resolve_title_conflict``), so the match is walked back to the root over the same
    continuation edge the title transfer uses. Without it every row stands for itself."""
    params = [escape_like(prefix) + "%", len(prefix), prefix]
    if not lineage:
        return f"({_TITLE_PREFIX_SQL.format(a='s')})", params
    edge = _COMPRESSION_CHILD_SQL.format(a="child") + _non_continuation_child_sql("child.", "parent.id")
    return f"""s.id IN (
        WITH RECURSIVE titled(id) AS (
            SELECT t.id FROM sessions t WHERE {_TITLE_PREFIX_SQL.format(a='t')}
            UNION
            SELECT parent.id
            FROM titled
            JOIN sessions child ON child.id = titled.id
            JOIN sessions parent ON parent.id = child.parent_session_id
            WHERE {edge}
        )
        SELECT id FROM titled)""", params


def project_compression_tips(db, sessions: list[dict[str, Any]], compact_rows: bool) -> list[dict[str, Any]]:
    """Replace each compression root's surfaced fields with its live tip's (root ``started_at`` kept
    for stable ordering), one batched query. ``_lineage_ids`` carries every chain id (a tile may
    hold a MIDDLE segment's id)."""
    chain_by_root: dict[str, list[str]] = {}  # only roots whose tip differs from themselves
    for s in sessions:
        if s.get("end_reason") == "compression":
            chain = db.get_compression_chain(s["id"])
            if chain and chain[-1] != s["id"]:
                chain_by_root[s["id"]] = chain
    tip_rows = (
        db._get_session_rich_rows_batch(
            {chain[-1] for chain in chain_by_root.values()}, compact_rows=compact_rows,
        ) if chain_by_root else {}
    )
    projected = []
    for s in sessions:
        chain = chain_by_root.get(s["id"])
        tip_row = tip_rows.get(chain[-1]) if chain else None
        if not tip_row:
            projected.append(s)
            continue
        merged = dict(s)
        for key in (
            "id", "ended_at", "end_reason", "message_count", "tool_call_count", "title", "last_active",
            "preview", "model", "system_prompt", "cwd", "git_branch", "git_repo_root",
        ):
            if key in tip_row:
                merged[key] = tip_row[key]
        if merged.get("title") is None:
            # The title is carried root->tip AFTER the publish transaction; a rotation cut off in
            # between leaves it on the ended root, and exact-title lookups (`hermes peer dm` ->
            # canonical "Bot Chat") must still see the lineage under its name (#106165).
            merged["title"] = s.get("title")
        merged["_lineage_root_id"] = s["id"]
        merged["_lineage_ids"] = chain
        # #121148: the projected row IS an automatic continuation, not a fresh conversation and
        # not a user branch. The server already classifies the edge (_COMPRESSION_CHILD_SQL) but
        # only ever used it to hide rows — surface the kind so clients can label the provenance
        # instead of rendering a sealed-and-rotated chat as a brand-new session.
        merged["continuation_kind"] = "compression"
        projected.append(merged)
    return projected
