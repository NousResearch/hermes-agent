"""Session-list page projection: compression roots surface their live tips."""


from __future__ import annotations
from typing import Any
from persistence.sqlite.conversation._store.compression import get_compression_chain
from persistence.sqlite.conversation._store.portability import _get_session_rich_rows_batch
from persistence.sqlite.conversation._store.sql_common import _id_chunks, _placeholders


def project_compression_tips(db, sessions: list[dict[str, Any]], compact_rows: bool) -> list[dict[str, Any]]:
    """Replace each compression root's surfaced fields with its live tip's (root ``started_at`` kept
    for stable ordering), one batched query. ``_lineage_ids`` carries every chain id (a tile may
    hold a MIDDLE segment's id)."""
    chain_by_root: dict[str, list[str]] = {}  # only roots whose tip differs from themselves
    for s in sessions:
        if s.get("end_reason") == "compression":
            chain = get_compression_chain(db, s["id"])
            if chain and chain[-1] != s["id"]:
                chain_by_root[s["id"]] = chain
    tip_rows = (
        _get_session_rich_rows_batch(db,
            {chain[-1] for chain in chain_by_root.values()}, compact_rows=compact_rows,
        ) if chain_by_root else {}
    )
    # An untitled tip inherits the nearest titled segment's name, not only the root's: a lineage
    # titled after its first rotation keeps that title on a MIDDLE segment, so the tip read '—'.
    untitled = sorted({i for c in chain_by_root.values() if not (tip_rows.get(c[-1]) or {}).get("title") for i in c[:-1]})
    titles = {}
    for chunk in _id_chunks(untitled):  # long chains x many rows must stay under SQLite's variable limit
        titles.update(db._read_all(
            f"SELECT id, title FROM sessions WHERE title IS NOT NULL AND id IN ({_placeholders(chunk)})", chunk))
    projected = []
    for s in sessions:
        chain = chain_by_root.get(s["id"])
        tip_row = tip_rows.get(chain[-1]) if chain else None
        if not chain or not tip_row:
            projected.append(s)
            continue
        merged = dict(s)
        for key in (
            "id", "ended_at", "end_reason", "message_count", "tool_call_count", "title", "last_active",
            "preview", "model", "system_prompt", "cwd", "git_branch", "git_repo_root", "source",
        ):
            if key in tip_row:
                merged[key] = tip_row[key]
        if merged.get("title") is None:
            # The title is carried root->tip AFTER the publish transaction; a rotation cut off in
            # between leaves it on the ended root, and exact-title lookups (`hermes peer dm` ->
            # canonical "Bot Chat") must still see the lineage under its name (#106165).
            merged["title"] = next((titles[i] for i in reversed(chain[:-1]) if titles.get(i)), s.get("title"))
        merged["_lineage_root_id"] = s["id"]
        merged["_lineage_ids"] = chain
        # #121148: the projected row IS an automatic continuation, not a fresh conversation and
        # not a user branch. The server already classifies the edge (_COMPRESSION_CHILD_SQL) but
        # only ever used it to hide rows — surface the kind so clients can label the provenance
        # instead of rendering a sealed-and-rotated chat as a brand-new session.
        merged["continuation_kind"] = "compression"
        projected.append(merged)
    return projected
