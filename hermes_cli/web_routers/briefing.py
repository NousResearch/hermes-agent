"""Daily briefing data for Jarvis: ``GET /api/briefing``.

When the user comes back ("wake up, tatuś wrócił"), the desktop asks for one bundle of real
state and hands it to the agent, which tells it out loud: what happened in the world
yesterday, what moved in AI, and how the workspace looks — yesterday's and today's sessions,
scheduled jobs (failing ones first) and the active model.

Nothing here talks to a model; it only gathers. World headlines come from
``dashboard.briefing_feeds`` (defaults: a few international and Polish news feeds), AI
headlines from the dashboard's own news aggregator, and both go through the same parser
(DTD-refusing, size-capped) and a short in-process cache. A failing feed, an unreadable
state.db or cron store is reported as absent, never as an error for the whole briefing.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter

from hermes_cli.web_deps import late
from hermes_cli.web_routers import news
from hermes_cli.web_routers._common import log as _log

router = APIRouter()

load_config = late("load_config", "hermes_cli.config")
_open_session_db_for_profile = late("_open_session_db_for_profile", "hermes_cli.web_server_sessions")
_call_cron_for_profile = late("_call_cron_for_profile", "hermes_cli.web_server_cron")

DEFAULT_BRIEFING_FEEDS: List[Dict[str, str]] = [
    {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "The Guardian · World", "url": "https://www.theguardian.com/world/rss"},
    {"name": "NPR World", "url": "https://feeds.npr.org/1004/rss.xml"},
    {"name": "TVN24", "url": "https://tvn24.pl/najnowsze.xml"},
    {"name": "Polsat News", "url": "https://www.polsatnews.pl/rss/wszystkie.xml"},
]

_MAX_WORLD = 14
_MAX_AI = 6
_MAX_TITLES = 8
_MAX_JOBS = 6
_MAX_TASKS = 6
_CACHE_TTL_S = 15 * 60

_cache: Dict[Tuple[str, ...], Tuple[float, Dict[str, Any]]] = {}


def resolve_briefing_feeds(cfg: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """``dashboard.briefing_feeds`` normalized like ``news_feeds``; defaults when unset."""
    dash = (cfg or {}).get("dashboard") if isinstance(cfg, dict) else None
    raw = dash.get("briefing_feeds") if isinstance(dash, dict) else None
    if not isinstance(raw, list) or not raw:
        return [dict(f) for f in DEFAULT_BRIEFING_FEEDS]
    resolved = news.resolve_news_feeds({"dashboard": {"news_feeds": raw}})
    # resolve_news_feeds falls back to the AI defaults when nothing valid is left.
    return resolved if resolved != news.DEFAULT_NEWS_FEEDS else [dict(f) for f in DEFAULT_BRIEFING_FEEDS]


def briefing_window(now: datetime) -> Tuple[float, float]:
    """``(yesterday 00:00, today 00:00)`` in local time, as timestamps."""
    local = now.astimezone()
    today = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return (today - timedelta(days=1)).timestamp(), today.timestamp()


def pick_headlines(items: List[Dict[str, Any]], since: float, limit: int) -> List[Dict[str, Any]]:
    """Newest-first headlines published since ``since``; undated items only fill a short list."""
    dated = [i for i in items if i.get("published") and i["published"] >= since]
    undated = [i for i in items if not i.get("published")]
    chosen = (dated + undated)[:limit]
    return [{"title": i["title"], "source": i["source"], "summary": i.get("summary", ""),
             "published": i.get("published"), "link": i["link"]} for i in chosen]


async def _cached_news(feeds: List[Dict[str, str]]) -> Dict[str, Any]:
    key = tuple(f["url"] for f in feeds)
    cached = _cache.get(key)
    if cached and time.time() - cached[0] < _CACHE_TTL_S:
        return cached[1]
    data = await news.collect_news(feeds)
    _cache[key] = (time.time(), data)
    return data


def summarize_sessions(rows: List[Dict[str, Any]], since: float, today: float) -> Dict[str, Any]:
    def _started(row: Dict[str, Any]) -> float:
        value = row.get("started_at") or 0
        return float(value) if isinstance(value, (int, float)) else 0.0

    yesterday = [r for r in rows if since <= _started(r) < today]
    today_rows = [r for r in rows if _started(r) >= today]
    recent = sorted(yesterday + today_rows, key=_started, reverse=True)
    titles = [str(r.get("title") or "").strip() for r in recent]
    return {
        "yesterday": len(yesterday),
        "today": len(today_rows),
        "titles": [t for t in titles if t][:_MAX_TITLES],
    }


def summarize_jobs(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    active = [j for j in jobs if j.get("enabled", True)]
    failing = [str(j.get("name") or j.get("id")) for j in active if j.get("last_status") == "error"]
    upcoming = sorted((j for j in active if j.get("next_run_at")), key=lambda j: str(j["next_run_at"]))
    return {
        "active": len(active),
        "failing": failing[:_MAX_JOBS],
        "next": [{"name": str(j.get("name") or j.get("id")), "at": j["next_run_at"]} for j in upcoming[:_MAX_JOBS]],
    }


_OPEN_TASK_STATUSES = ("blocked", "running", "review", "ready", "todo")


def summarize_tasks(tasks: List[Any]) -> Dict[str, Any]:
    """Open kanban tasks, most urgent state first (blocked, then running...), then by priority."""
    open_tasks = [t for t in tasks if t.status in _OPEN_TASK_STATUSES]
    open_tasks.sort(key=lambda t: (_OPEN_TASK_STATUSES.index(t.status), -int(t.priority or 0)))
    return {
        "open": len(open_tasks),
        "blocked": sum(1 for t in open_tasks if t.status == "blocked"),
        "items": [{"title": str(t.title), "status": t.status} for t in open_tasks[:_MAX_TASKS]],
    }


def _open_tasks() -> Optional[Dict[str, Any]]:
    try:
        from hermes_cli import kanban_db
        from hermes_cli.kanban_db_connect import connect_closing
        with connect_closing() as conn:
            return summarize_tasks(kanban_db.list_tasks(conn))
    except Exception:
        _log.debug("briefing: kanban tasks unavailable", exc_info=True)
        return None


def _workspace(profile: Optional[str], since: float, today: float) -> Dict[str, Any]:
    sessions: Optional[Dict[str, Any]] = None
    jobs: Optional[Dict[str, Any]] = None
    try:
        db = _open_session_db_for_profile(profile, read_only=True)
        try:
            rows = db.list_sessions_rich(limit=200, offset=0, compact_rows=True, exclude_sources=["cron"])
        finally:
            db.close()
        sessions = summarize_sessions(rows, since, today)
    except Exception:
        _log.debug("briefing: sessions unavailable", exc_info=True)
    try:
        jobs = summarize_jobs(_call_cron_for_profile(profile or None, "list_jobs", True))
    except Exception:
        _log.debug("briefing: cron jobs unavailable", exc_info=True)
    return {"sessions": sessions, "jobs": jobs, "tasks": _open_tasks()}


async def _google_snapshot(profile: Optional[str]) -> Optional[Dict[str, Any]]:
    """Today's calendar and unread mail when Google is connected; absent (never an error) otherwise."""
    try:
        from agent import google_connect
        from hermes_cli.web_routers._common import scoped_to_thread
        return await asyncio.wait_for(scoped_to_thread(profile, google_connect.today_snapshot), timeout=45)
    except Exception:
        _log.debug("briefing: Google snapshot unavailable", exc_info=True)
        return None


@router.get("/api/briefing")
async def get_briefing(profile: Optional[str] = None):
    try:
        cfg = await asyncio.to_thread(load_config)
    except Exception:
        cfg = {}
    since, today = briefing_window(datetime.now())
    world_data, ai_data, workspace, google = await asyncio.gather(
        _cached_news(resolve_briefing_feeds(cfg)),
        _cached_news(news.resolve_news_feeds(cfg)),
        asyncio.to_thread(_workspace, profile, since, today),
        _google_snapshot(profile),
    )
    model = cfg.get("model") if isinstance(cfg, dict) else None
    # `model` is either a bare id or a {default, provider, ...} mapping.
    model_id = model.get("default") if isinstance(model, dict) else model
    provider = model.get("provider") if isinstance(model, dict) else None
    return {
        "generated_at": time.time(),
        "window": {"since": since, "until": today},
        "world": pick_headlines(world_data["items"], since, _MAX_WORLD),
        "ai": pick_headlines(ai_data["items"], since, _MAX_AI),
        "feeds_failed": [f["name"] for f in world_data["feeds"] + ai_data["feeds"] if not f.get("ok")],
        "google": google,
        "workspace": {
            **workspace,
            "model": model_id or None,
            "provider": provider or None,
        },
    }
