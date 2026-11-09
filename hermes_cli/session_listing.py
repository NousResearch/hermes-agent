"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from utils import is_truthy_value

_LIST_WORDS = {"list", "ls", "browse"}
_SEARCH_WORDS = {"search", "find"}
SUBAGENT_SOURCE = "subagent"


def show_subagent_sessions(hermes_home: str | Path) -> bool:
    """``sessions.show_subagents`` from *hermes_home*'s own config.yaml. One ``serve`` lists many
    profiles' stores, so the store's home decides, never the process ``HERMES_HOME``. False when
    the config cannot be read (the listing keeps its default shape)."""
    from hermes_cli.config import load_config_readonly
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(hermes_home))
    try:
        sessions_cfg = load_config_readonly().get("sessions") or {}
    except Exception:
        return False
    finally:
        reset_hermes_home_override(token)
    return isinstance(sessions_cfg, dict) and is_truthy_value(sessions_cfg.get("show_subagents"))


def subagent_listing_scope(
    hermes_home: str | Path, *, source: str | None = None, sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
) -> tuple[bool, list[str] | None]:
    """``(include_subagents, exclude_sources)`` for one human-facing session list (#97202).

    With ``sessions.show_subagents`` on, delegate runs join a list that is not scoped to named
    sources and either has no exclusions or excludes the ``subagent`` source itself (the desktop
    recents shape); that exclusion is dropped so the runs actually land. A slice that excludes
    other sources without ``subagent`` (the per-platform messaging lists) keeps its shape.
    """
    excluded = list(exclude_sources or [])
    if source or sources or (excluded and SUBAGENT_SOURCE not in excluded):
        return False, exclude_sources
    if not show_subagent_sessions(hermes_home):
        return False, exclude_sources
    return True, [s for s in excluded if s != SUBAGENT_SOURCE] or None


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
    restricts gateway callers to one lane before the DB limit applies. With ``search_query`` rows
    are filtered by title/id in SQL, ordered by recent activity, and unnamed sessions stay visible
    since an id match may be the only handle.
    """
    search = (search_query or "").strip()
    rows = session_db.list_sessions_rich(
        source=None if include_all_sources else source,
        session_key=session_key,
        exclude_sources=exclude_sources,
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
