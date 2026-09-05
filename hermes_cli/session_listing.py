"""Shared session-listing helpers for CLI and gateway slash surfaces."""

from __future__ import annotations

import shlex
from typing import Any

_LIST_WORDS = {"list", "ls", "browse"}
_SEARCH_WORDS = {"search", "find"}


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


def parse_project_listing_args(raw_args: str) -> tuple[bool, str] | None:
    """Detect and parse the ``/sessions project [slug]`` form.

    Returns ``(is_project_mode, slug)`` where ``slug`` is empty to mean "group
    every project". Returns ``None`` when this isn't the project form, so the
    caller falls through to :func:`parse_session_listing_args` unchanged.

    Kept separate from the main parser because project mode changes the SHAPE
    of the output (grouped by project, not one flat list), so it can't just be
    another boolean flag on the existing path.
    """
    try:
        parts = shlex.split(raw_args or "")
    except ValueError:
        return None
    if not parts:
        return None
    head = parts[0].strip().lower()
    if head not in {"project", "projects", "proj"}:
        return None
    slug = " ".join(parts[1:]).strip()
    return True, slug


def query_sessions_by_project(
    session_db: Any,
    *,
    slug: str = "",
    limit_per_project: int = 10,
    scan_limit: int = 500,
) -> list[dict[str, Any]]:
    """Group sessions by the Hermes project owning their working directory.

    Returns ``[{"project": <Project>, "sessions": [row, ...]}, ...]`` ordered
    by project creation. A session belongs to a project when its ``cwd`` (or
    ``git_repo_root``) is the project's folder or nested under it — the same
    longest-prefix rule ``projects_db.project_for_path`` uses everywhere else,
    so this view can't disagree with kanban worktrees or desktop grouping.

    Sessions with no cwd (every gateway session predating cwd binding) simply
    match no project and are omitted rather than guessed at.
    """
    from hermes_cli import projects_db as pdb

    rows = session_db.list_sessions_rich(
        source=None, exclude_sources=["tool"], limit=scan_limit
    )
    out: list[dict[str, Any]] = []
    with pdb.connect_closing() as conn:
        projects = pdb.list_projects(conn)
        wanted = (slug or "").strip().lower()
        if wanted:
            projects = [
                p
                for p in projects
                if p.slug.lower() == wanted or p.name.lower() == wanted
            ]
        # Resolve each session ONCE, then bucket — avoids re-walking the folder
        # table per project per session.
        buckets: dict[str, list[dict[str, Any]]] = {p.id: [] for p in projects}
        for row in rows:
            cwd = row.get("cwd") or row.get("git_repo_root") or ""
            if not cwd:
                continue
            proj = pdb.project_for_path(conn, str(cwd))
            if proj is None or proj.id not in buckets:
                continue
            if len(buckets[proj.id]) < limit_per_project:
                buckets[proj.id].append(row)
        for p in projects:
            out.append({"project": p, "sessions": buckets.get(p.id, [])})
    return out


def format_project_session_listing(
    groups: list[dict[str, Any]],
    *,
    channel_name_fn: Any = None,
) -> str:
    """Render sessions grouped by project for gateway messengers."""
    if not groups:
        return (
            "No projects found.\n"
            "Create one with `hermes project create <name>`."
        )

    lines: list[str] = ["📁 **Sessions by project**", ""]
    total = 0
    for group in groups:
        proj = group["project"]
        sessions = group["sessions"]
        total += len(sessions)
        header = f"**{proj.name}**"
        if channel_name_fn is not None:
            try:
                header += f" · #{channel_name_fn(proj.slug)}"
            except Exception:
                pass
        lines.append(f"{header} — {len(sessions)} session(s)")
        if not sessions:
            lines.append("  _none yet_")
        for row in sessions:
            title = str(row.get("title") or "").strip()
            sid = str(row.get("id") or "")
            src = str(row.get("source") or "")
            label = title or (str(row.get("preview") or "")[:40] or "—")
            lines.append(f"  • {label} `{src}` — `{sid}`")
        lines.append("")

    if total == 0:
        lines.append(
            "_No sessions are linked to a project yet. A session links when it "
            "runs in a project folder — start one in that project's channel._"
        )
    else:
        lines.append("Resume any of them with `/resume <session id>`.")
    return "\n".join(lines)


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
