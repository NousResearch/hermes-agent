"""Read-only fleet session inventory; no profile runtimes or DB repairs."""
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter()

# A compact DB row is not a public summary: it still includes provider URLs,
# billing/error/origin blobs. Keep the inventory boundary explicitly narrow.
_SESSION_FIELDS = ("id", "resolved_id", "title", "preview", "model", "source", "started_at",
                   "last_active", "message_count", "unread")


def session_summary(row, profile):
    result = {key: row[key] for key in _SESSION_FIELDS if row.get(key) is not None}
    result["profile"] = profile
    result["provider"] = str(row.get("provider") or row.get("billing_provider") or "")
    return result


def stored_summaries(db, rows, profile):
    if not rows:
        return []
    # Read only the provider scalar, never hydrate/send model_config (keys,
    # base URLs) just to label an unbilled session. IDs may be compression tips.
    ids = [row.get("resolved_id") or row["id"] for row in rows]
    marks = ",".join("?" for _ in ids)
    providers = {item["id"]: item["provider"] for item in db._read_all(
        f"SELECT id, CASE WHEN json_valid(model_config) THEN json_extract(model_config, '$.provider') END AS provider "
        f"FROM sessions WHERE id IN ({marks})", ids)}
    result = []
    for row, session_id in zip(rows, ids):
        summary = session_summary(row, profile)
        provider = providers.get(session_id)
        if not summary["provider"] and isinstance(provider, str):
            summary["provider"] = provider
        result.append(summary)
    return result


@router.get("/api/profiles/agent-overview")
def get_agent_overview(limit: int = Query(100, ge=1, le=100), offset: int = Query(0, ge=0), live_only: bool = False):
    from hermes_cli.web_routers.profiles import _profile_targets
    from hermes_cli.profiles import read_profile_meta
    from hermes_state import SessionDB

    rows, canonical, errors, profile_rows = [], [], [], []
    targets = _profile_targets("GET /api/profiles/agent-overview", lightweight=True, errors=errors)
    total, remaining_offset = 0, offset
    # Page across profiles in deterministic order. Unlike the sidebar's capped
    # recency over-fetch, the DB offset advances even beyond 500 rows/profile.
    for name, home in sorted(targets, key=lambda item: item[0]):
        meta = read_profile_meta(Path(home)) if not live_only else {}
        profile_rows.append({"name": name, **{key: meta[key] for key in ("display_name", "description") if meta.get(key)}})
        if live_only:
            continue
        path = Path(home) / "state.db"
        if not path.exists():
            continue
        try:
            # Deliberately bypass the dashboard's heal-on-read helper. A broken
            # or old store is coverage failure, never permission to repair it.
            with SessionDB(path, read_only=True) as db:
                count = db.session_count(exclude_children=True, include_hidden=False)
                total += count
                if remaining_offset >= count:
                    remaining_offset -= count
                elif len(rows) < limit:
                    page = db.list_sessions_rich(limit=limit - len(rows), offset=remaining_offset,
                                                order_by_last_active=True, compact_rows=True)
                    rows.extend(stored_summaries(db, page, name))
                    remaining_offset = 0
                # The exact title registry is independent of recency/visibility.
                # Hidden rows never reach the rich page above, so project the live
                # tip's activity/preview/read watermark through the same read-only
                # rich projection the page uses; never a stored session-id pointer.
                bot = db.get_session_by_title("Bot Chat")
                if bot and not bot.get("archived"):
                    tip = db.get_compression_tip(bot["id"]) or bot["id"]
                    rich = db.get_session_rich_row(tip, compact_rows=True) or bot
                    canonical.extend(stored_summaries(db, [{**rich, "unread": db.session_unread(rich), "id": bot["id"],
                                                         "resolved_id": tip, "title": "Bot Chat"}], name))
        except Exception:
            # Raw SQLite/provider errors may include paths or credentials.
            errors.append({"profile": name, "error": "session-store-unavailable"})
    live, coverage = live_snapshot(targets)
    return {"sessions": rows, "canonical": canonical, "profiles": profile_rows,
            "live": live, "live_coverage": coverage,
            "total": total, "offset": offset, "limit": limit, "errors": errors}


def live_snapshot(targets):
    import sys
    server = sys.modules.get("tui_gateway.server")
    if server is None:
        return [], "unavailable"
    with server._sessions_lock:
        snapshot = list(server._sessions.items())
    homes = {Path(home).resolve(): name for name, home in targets}
    rows = []
    for sid, session in snapshot:
        if session.get("_finalized"):
            continue
        # None means THIS PROCESS's captured launch profile, not default.
        home = Path(session.get("profile_home") or server._hermes_home).resolve()
        profile = homes.get(home)
        if profile is None:
            continue
        override = session.get("model_override") or {}
        agent = session.get("agent")
        inflight = server._inflight_snapshot(session)
        queued = server._queued_prompt_snapshot(session)
        activity = (queued or {}).get("user") or (inflight or {}).get("assistant") or (inflight or {}).get("user")
        if not activity:
            activity = next((server._content_display_text(message.get("content", message.get("text", "")))
                             for message in reversed(session.get("history") or [])
                             if message.get("role") in ("user", "assistant") and message.get("content")), "")
        preview = " ".join(str(activity).split())[:160]
        # _session_live_item also reads its title through a writable registry
        # handle. Reuse its status/identity policies, never its writer path.
        rows.append({"id": server._session_lookup_key(session, fallback=sid), "runtime_id": sid,
                     "profile": profile, "status": server._session_live_status(sid, session),
                     "title": str(session.get("pending_title") or ""), "preview": preview,
                     "last_active": session.get("last_active") or session.get("created_at") or 0,
                     "started_at": session.get("created_at") or 0,
                     "message_count": len(session.get("history") or []),
                     "provider": str(getattr(agent, "provider", "") or override.get("provider") or ""),
                     "model": str(getattr(agent, "model", "") or override.get("model") or "")})
    return rows, "process"
