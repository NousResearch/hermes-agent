"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

import json
import re
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


PING_PROBE_PREVIEW = "ping, reply with one word"
# Stable provenance values for non-user work. These are preferred over prompt
# heuristics; the latter remain only for legacy rows created before provenance
# was recorded.
AUTOMATED_SESSION_SOURCES = frozenset({
    "automation", "automated", "jev", "probe", "scheduled",
    "tool", "kanban", "cron",
})


def is_automated_session(session: dict[str, Any]) -> bool:
    """Return whether a row is explicitly marked as automation/non-chat work."""
    source = str(session.get("source") or "").strip().casefold()
    if source in AUTOMATED_SESSION_SOURCES:
        return True

    for raw in (session.get("origin_json"), session.get("model_config")):
        if not raw:
            continue
        try:
            metadata = raw if isinstance(raw, dict) else json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        origin = (
            metadata.get("_session_origin")
            or metadata.get("session_origin")
            or metadata.get("origin")
            or metadata.get("kind")
        )
        if isinstance(origin, dict):
            kind = origin.get("kind") or origin.get("type")
        else:
            kind = origin
        if str(kind or "").strip().casefold() in {"automation", "automated", "scheduled", "probe", "jev", "tool"}:
            return True
        if metadata.get("_automation") is True or metadata.get("automation") is True:
            return True
    return False


# Short, exact replies used by connectivity/startup probes. Keep this list
# deliberately narrow so a real conversation mentioning "hello" survives.
TRIVIAL_PROBE_TOKENS = frozenset({"hi", "hello", "hey", "ping", "pong", "ok"})
TOOL_INVENTORY_MAX_MESSAGES = 5


def is_tool_inventory_probe(text: str) -> bool:
    """Return whether text is a short tool/capability inventory request."""
    return bool(
        re.search(r"\blist\b.*\btools?\b", text)
        or re.search(r"\bmcp_[a-z0-9_-]+\b", text)
        or "browser automation tools" in text
    )


def is_ping_probe_session(session: dict[str, Any]) -> bool:
    """Return whether a session row is automated probe/debris, not a real chat.

    Probe text can appear in either the title or preview because automatic
    title generation sometimes copies a one-shot prompt verbatim. Titled
    sessions with ordinary names are kept, even when their preview mentions
    one of these phrases.
    """
    title = str(session.get("title") or "").strip().casefold()
    preview = str(session.get("preview") or "").strip().casefold()

    def looks_like_probe(text: str) -> bool:
        if not text:
            return False
        if text == PING_PROBE_PREVIEW:
            return True
        if re.fullmatch(r"(?:ping,\s*)?reply\s+with\s+one\s+word(?:\s+only)?", text):
            return True
        if text in TRIVIAL_PROBE_TOKENS:
            return True
        if is_tool_inventory_probe(text):
            return True
        # Connectivity/model checks commonly ask for a fixed short token. Keep
        # this anchored so a real conversation mentioning "reply with" survives.
        if re.fullmatch(
            r"(?:ping,\s*)?reply\s+with\s+(?:exactly|just)\s*:?\s*[a-z0-9_-]+(?:\s+only)?",
            text,
        ):
            return True
        return any(
            marker in text
            for marker in (
                "say hello",
                "hello, reply",
                "what model are you",
                "starting deepseek session",
                "[note: model was just switched",
                "回复一句话告诉我",
                "请只回复",
                "只回复四个字",
                "reply exactly:",
                "shall we talk in chinese or english",
            )
        )

    if title and looks_like_probe(title):
        # An exact greeting/token can be a legitimate chat title after a
        # longer conversation; only treat the short form as probe debris.
        count = session.get("message_count")
        if title in TRIVIAL_PROBE_TOKENS:
            return count is None or count <= 2
        if is_tool_inventory_probe(title):
            return count is None or count <= TOOL_INVENTORY_MAX_MESSAGES
        return True
    if not preview:
        return not title
    if looks_like_probe(preview):
        if not title:
            return True
        # Auto-titling can paraphrase a probe prompt into a real-looking title
        # ("Ping connectivity test"). A probe session is still just one
        # exchange — keep titled sessions that continued past it.
        count = session.get("message_count")
        max_probe_messages = (
            TOOL_INVENTORY_MAX_MESSAGES if is_tool_inventory_probe(preview) else 2
        )
        return count is None or count <= max_probe_messages
    return False


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
        # Probe/automation debris stays hidden in browse modes; a search by id
        # may be the only handle for such a row, so searches keep them.
        if not search and is_automated_session(row):
            continue
        if not search and is_ping_probe_session(row):
            continue
        if (row.get("source") or "").strip().lower() == "cron" and not search:
            # Scheduled watchdog/monitor runs flood the recent list; keep them
            # reachable via search.
            continue
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
