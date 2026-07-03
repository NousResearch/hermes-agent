"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionListingRequest:
    include_all_sources: bool = False
    include_unnamed: bool = False
    target: str = ""
    search_requested: bool = False
    search_query: str = ""


def parse_session_listing_request(raw_args: str) -> SessionListingRequest:
    """Parse `/sessions` args into a typed listing/search/resume request."""
    import shlex

    parts = shlex.split(raw_args or "")
    include_all = False
    include_unnamed = False
    target_parts: list[str] = []
    search_parts: list[str] = []
    search_requested = False

    for part in parts:
        if search_requested:
            search_parts.append(part)
            continue

        lower = part.strip().lower()
        if lower in {"list", "ls", "browse"} and not target_parts:
            continue
        if lower in {"all", "--all"} and not target_parts:
            include_all = True
            continue
        if lower in {"full", "--full"} and not target_parts:
            include_unnamed = True
            continue
        if lower in {"search", "find"} and not target_parts:
            search_requested = True
            continue
        target_parts.append(part)

    return SessionListingRequest(
        include_all_sources=include_all,
        include_unnamed=include_unnamed,
        target=" ".join(target_parts).strip(),
        search_requested=search_requested,
        search_query=" ".join(search_parts).strip(),
    )


def parse_session_listing_args(raw_args: str) -> tuple[bool, bool, str]:
    """Parse `/sessions`-style args into listing flags plus a resume target.

    Returns ``(include_all_sources, include_unnamed, target)``. ``list``/``ls``
    and ``browse`` are display aliases; ``all``/``--all`` widens source scope;
    ``full``/``--full`` keeps unnamed sessions in the listing. Anything else is
    treated as a target so `/sessions <id-or-title>` can delegate to `/resume`.
    """
    parsed = parse_session_listing_request(raw_args)
    return parsed.include_all_sources, parsed.include_unnamed, parsed.target


def _matches_search(row: dict[str, Any], query: str) -> bool:
    needle = (query or "").strip().lower()
    if not needle:
        return True
    compact_needle = re.sub(r"[\W_]+", "", needle)
    haystacks = (
        str(row.get("title") or ""),
        str(row.get("id") or ""),
        str(row.get("_lineage_root_id") or ""),
    )
    return any(
        needle in value.lower()
        or (bool(compact_needle) and compact_needle in re.sub(r"[\W_]+", "", value.lower()))
        for value in haystacks
    )


def query_session_listing(
    session_db: Any,
    *,
    source: str | None,
    user_id: str | None = None,
    user_ids: list[str] | None = None,
    chat_id: str | None = None,
    chat_id_allow_empty: bool = False,
    chat_types: list[str] | None = None,
    thread_id: str | None = None,
    thread_id_allow_empty: bool = False,
    current_session_id: str | None = None,
    include_all_sources: bool = False,
    include_unnamed: bool = False,
    search_query: str | None = None,
    search_message_content: bool = False,
    limit: int = 10,
    offset: int = 0,
    exclude_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return session rows for interactive listing surfaces.

    This is the shared selection policy behind CLI/gateway session browsing:
    source-scoped by default, optionally global, hide unnamed sessions unless
    the caller asks for a full listing, and never include the current session.
    """
    query_source = None if include_all_sources else source
    fetch_limit = max(limit * 4, limit)
    search = (search_query or "").strip()
    list_kwargs = {
        "source": query_source,
        "user_id": user_id,
        "user_ids": user_ids,
        "chat_id": chat_id,
        "chat_id_allow_empty": chat_id_allow_empty,
        "chat_types": chat_types,
        "thread_id": thread_id,
        "thread_id_allow_empty": thread_id_allow_empty,
        "search_message_content": search_message_content,
        "exclude_sources": exclude_sources,
        "limit": fetch_limit,
        "offset": offset,
    }
    if search:
        list_kwargs.update(order_by_last_active=True, search_query=search)
    rows = session_db.list_sessions_rich(**list_kwargs)
    result: list[dict[str, Any]] = []
    for row in rows:
        if current_session_id and row.get("id") == current_session_id:
            continue
        if search and not search_message_content and not _matches_search(row, search):
            continue
        if not include_unnamed and not row.get("title") and not search:
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
    lines.append("Resume: `/resume <session id>` or `/sessions <session id or title>`.")
    lines.append("Numbered picks: `/resume <number>` from the latest session list.")
    lines.append("More: `/sessions all`, `/sessions full`, `/sessions search <query>`.")
    return "\n".join(lines)
