"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

import shlex
from typing import Any

_LIST_WORDS = {"list", "ls", "browse"}
_SEARCH_WORDS = {"search", "find"}

# Adaptive widening for query_session_listing(): the SQL fetch starts at 4×limit
# and grows (never OFFSET pagination — each step re-fetches from row 0 so a live
# insert can't skip or double a row) until enough *displayable* rows survive the
# Python-side visibility filter or the DB returns fewer rows than asked (nothing
# more to find). 16× is the ceiling. There is no session-count safety limit
# anywhere in the repo to cap this against, and none is needed: interactive
# callers pass limit 10–50, so the widest fetch is a few hundred rows from one
# indexed query.
_WIDEN_FACTORS = (4, 8, 16)


def parse_session_listing_args(raw_args: str) -> tuple[bool, bool, str, str | None]:
    """Parse `/sessions`-style args into ``(include_all_sources, include_unnamed, target, search_query)``.

    ``all`` widens source scope, ``full`` keeps unnamed sessions, ``search``/``find`` makes the rest
    a query (``None`` = not requested, ``""`` = requested with no terms). Flags are honored only
    before the first positional word so titles containing "all" aren't misparsed; anything else is
    a target so `/sessions <id-or-title>` can delegate to `/resume`.
    """
    parts = shlex.split(raw_args or "")
    flags = {"all": False, "full": False}
    target_parts: list[str] = []
    for i, part in enumerate(parts):
        lower = part.strip().lower()
        if not target_parts:
            if lower in _LIST_WORDS:
                continue
            if lower in {"all", "--all", "full", "--full"}:
                flags[lower.lstrip("-")] = True
                continue
            if lower in _SEARCH_WORDS:
                return flags["all"], flags["full"], "", " ".join(parts[i + 1:]).strip()
        target_parts.append(part)
    return flags["all"], flags["full"], " ".join(target_parts).strip(), None


def query_session_listing(
    session_db: Any,
    *,
    source: str | None,
    session_key: str | None = None,
    current_session_id: str | None = None,
    include_current_session: bool = False,
    include_all_sources: bool = False,
    include_unnamed: bool = False,
    search_query: str | None = None,
    limit: int = 10,
    exclude_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return session rows for interactive listing surfaces (shared CLI/gateway policy).

    Source-scoped unless global is requested; unnamed hidden unless a full listing is asked for;
    current session hidden unless requested (then marked ``is_current_session``); ``session_key``
    restricts gateway callers to one lane, in SQL. With ``search_query`` rows are filtered by
    title/id in SQL, ordered by recent activity, and unnamed sessions stay visible since an id
    match may be the only handle.

    The Python-side visibility filter (unnamed / current session) runs *after* the DB fetch, so a
    single fixed window could hide an older eligible session behind newer hidden ones. The fetch
    therefore widens adaptively (``_WIDEN_FACTORS``, always from row 0) until ``limit`` displayable
    rows survive the filter or the DB is exhausted. ``limit <= 0`` returns ``[]`` without querying.
    The returned list never exceeds ``limit``.
    """
    if limit <= 0:
        return []
    search = (search_query or "").strip()

    def _displayable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            is_current = bool(current_session_id and row.get("id") == current_session_id)
            if (is_current and not include_current_session) or (
                not include_unnamed and not row.get("title") and not search and not is_current
            ):
                continue
            out.append({**row, "is_current_session": True} if is_current else row)
            if len(out) >= limit:
                break
        return out

    result: list[dict[str, Any]] = []
    for factor in _WIDEN_FACTORS:
        fetch = limit * factor
        rows = session_db.list_sessions_rich(
            source=None if include_all_sources else source,
            session_key=session_key,
            exclude_sources=exclude_sources,
            limit=fetch,
            search_query=search or None,
            order_by_last_active=bool(search),
        )
        result = _displayable(rows)
        if len(result) >= limit or len(rows) < fetch:
            break
    return result


def format_gateway_session_listing(
    rows: list[dict[str, Any]],
    *,
    include_source: bool = False,
    title: str = "Sessions",
    notice: str | None = None,
) -> str:
    """Render a compact Markdown-ish session list for gateway messengers.

    ``notice`` adds an explanatory line above the footer — e.g. when a requested scope widening
    (``all``) was declined, so the caller isn't left guessing why sessions are missing.
    """
    if not rows:
        return "\n".join([
            "No sessions found.\n"
            "Use `/title My Session` to name this chat, or `/sessions full` "
            "to include unnamed sessions.",
            *([notice] if notice else []),
        ])
    lines = [f"📋 **{title}**", ""]
    for idx, row in enumerate(rows, start=1):
        current_part = " (current)" if row.get("is_current_session") else ""
        preview = str(row.get("preview") or "")[:40]
        source = str(row.get("source") or "")
        source_part = f" `{source}`" if include_source and source else ""
        preview_part = f" — _{preview}_" if preview else ""
        lines.append(
            f"{idx}. **{row.get('title') or '—'}**{current_part}{source_part}"
            f" — `{row.get('id') or ''}`{preview_part}"
        )
    return "\n".join([
        *lines, "", *([notice] if notice else []),
        "Resume: `/resume <session id>` or `/resume <number>` from `/resume`.",
        "More: `/sessions all`, `/sessions full`, `/sessions search <query>`.",
    ])
