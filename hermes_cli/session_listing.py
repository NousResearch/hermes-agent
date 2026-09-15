"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

import shlex
from typing import Any

_LIST_WORDS = {"list", "ls", "browse"}
_SEARCH_WORDS = {"search", "find"}

# Internal/automation session producers that must never surface in interactive
# resume/list surfaces: cron jobs, tool sub-runs, kanban workers, and subagent
# delegates are machine driven, not conversations a person picks up. Human
# conversation surfaces (cli, tui, webui, ACP adapter sessions, webhook
# sessions, custom HERMES_SESSION_SOURCE values, and every gateway platform)
# stay visible. This is a deny-list, not a human whitelist, so a newly added
# gateway platform appears automatically with no edit here.
AUTOMATION_SOURCES = frozenset({"cron", "tool", "kanban", "subagent"})


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


def _effective_listing_scope(
    *,
    source: str | None,
    session_key: str | None,
    include_all_sources: bool,
    exclude_sources: list[str] | None,
) -> tuple[str | None, list[str] | None]:
    """Resolve source/deny-list policy without weakening gateway lane authority.

    The classic local CLI historically passes ``source="cli"`` even though it
    reads the same user-owned state DB as TUI/Desktop/WebUI. Treat that label as
    provenance rather than an allow-list: local CLI discovery is cross-source
    and deny-lists only reviewed automation producers.

    Gateway callers carry ``session_key`` (or explicitly request all sources)
    and keep their existing source/origin authority. This helper only widens the
    local ``cli``/no-lane shape.
    """
    local_cli = source == "cli" and session_key is None
    if not (include_all_sources or local_cli):
        return source, exclude_sources

    effective_exclude = set(exclude_sources or ())
    if local_cli:
        effective_exclude.update(AUTOMATION_SOURCES)
    return None, sorted(effective_exclude) if effective_exclude else None


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

    Gateway callers remain source/lane scoped unless a global listing is
    requested. The local classic CLI is cross-source by default because its
    ``source="cli"`` value is provenance, not an authorization boundary;
    reviewed automation sources are denied instead. Unnamed rows are hidden
    unless a full listing is asked for; current session is hidden unless
    requested (then marked ``is_current_session``). With ``search_query`` rows
    are filtered by title/id in SQL, ordered by recent activity, and unnamed
    sessions stay visible since an id match may be the only handle.
    """
    search = (search_query or "").strip()
    effective_source, effective_exclude = _effective_listing_scope(
        source=source,
        session_key=session_key,
        include_all_sources=include_all_sources,
        exclude_sources=exclude_sources,
    )
    rows = session_db.list_sessions_rich(
        source=effective_source,
        session_key=session_key,
        exclude_sources=effective_exclude,
        limit=max(limit * 4, limit),
        search_query=search or None,
        order_by_last_active=bool(search),
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        is_current = bool(current_session_id and row.get("id") == current_session_id)
        if (is_current and not include_current_session) or (
            not include_unnamed and not row.get("title") and not search and not is_current
        ):
            continue
        result.append({**row, "is_current_session": True} if is_current else row)
        if len(result) >= limit:
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
