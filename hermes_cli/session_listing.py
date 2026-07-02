"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

from typing import Any


def parse_session_listing_args(raw_args: str) -> tuple[bool, bool, str]:
    """Parse `/sessions`-style args into listing flags plus a resume target.

    Returns ``(include_all_sources, include_unnamed, target)``. ``list``/``ls``
    and ``browse`` are display aliases; ``all``/``--all`` widens source scope;
    ``full``/``--full`` keeps unnamed sessions in the listing. Anything else is
    treated as a target so `/sessions <id-or-title>` can delegate to `/resume`.
    """
    import shlex

    parts = shlex.split(raw_args or "")
    include_all = False
    include_unnamed = False
    target_parts: list[str] = []
    for part in parts:
        lower = part.strip().lower()
        if lower in {"list", "ls", "browse"}:
            continue
        if lower in {"all", "--all"}:
            include_all = True
            continue
        if lower in {"full", "--full"}:
            include_unnamed = True
            continue
        target_parts.append(part)
    return include_all, include_unnamed, " ".join(target_parts).strip()


def query_session_listing(
    session_db: Any,
    *,
    source: str | None,
    current_session_id: str | None = None,
    include_all_sources: bool = False,
    include_unnamed: bool = False,
    limit: int = 10,
    exclude_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return session rows for interactive listing surfaces.

    This is the shared selection policy behind CLI/gateway session browsing:
    source-scoped by default, optionally global, hide unnamed sessions unless
    the caller asks for a full listing, and never include the current session.
    """
    query_source = None if include_all_sources else source
    fetch_limit = max(limit * 4, limit)
    rows = session_db.list_sessions_rich(
        source=query_source,
        exclude_sources=exclude_sources,
        exclude_empty_untitled=True,
        limit=fetch_limit,
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        if current_session_id and row.get("id") == current_session_id:
            continue
        if not include_unnamed and not row.get("title"):
            continue
        result.append(row)
        if len(result) >= limit:
            break
    return result


def format_gateway_session_listing(
    rows: list[dict[str, Any]],
    *,
    include_source: bool = False,
    title: str = "Sessions",
) -> str:
    """Render a compact Markdown-ish session list for gateway messengers."""
    if not rows:
        return (
            "No sessions found.\n"
            "Use `/title My Session` to name this chat, or `/sessions full` "
            "to include unnamed sessions."
        )

    lines = [f"📋 **{title}**", ""]
    for idx, row in enumerate(rows, start=1):
        session_id = str(row.get("id") or "")
        title_text = str(row.get("title") or "—")
        preview = str(row.get("preview") or "")[:40]
        source = str(row.get("source") or "")
        source_part = f" `{source}`" if include_source and source else ""
        preview_part = f" — _{preview}_" if preview else ""
        lines.append(f"{idx}. **{title_text}**{source_part} — `{session_id}`{preview_part}")
    lines.append("")
    lines.append("Resume: `/resume <session id>` or `/resume <number>` from `/resume`.")
    lines.append("More: `/sessions all`, `/sessions full`, `/sessions all full`.")
    return "\n".join(lines)
