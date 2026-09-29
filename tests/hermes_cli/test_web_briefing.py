"""Daily briefing bundle (``GET /api/briefing``): real sessions from state.db, headlines
limited to yesterday onward, and a failing source never sinks the briefing."""
from datetime import datetime

import pytest

from hermes_cli.web_routers import briefing


@pytest.fixture
def client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_briefing_reports_yesterdays_sessions_and_only_recent_headlines(client, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    since, today = briefing.briefing_window(datetime.now())
    db = SessionDB(db_path=get_hermes_home() / "state.db")
    try:
        db.create_session("s-yesterday", source="desktop")
        db.set_session_title("s-yesterday", "Raport sprzedaży")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (since + 3600, "s-yesterday"))
        db.create_session("s-old", source="desktop")
        db.set_session_title("s-old", "Zeszły tydzień")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (since - 5 * 86400, "s-old"))
        db._conn.commit()
    finally:
        db.close()

    monkeypatch.setattr(briefing, "_cache", {})
    monkeypatch.setattr(briefing, "load_config", lambda: {"model": {"default": "deepseek/deepseek-v4.1-flash",
                                                                    "provider": "openrouter"}})

    async def fake_collect(feeds):
        return {
            "items": [
                {"title": "Fresh", "source": "W", "summary": "", "published": since + 60, "link": "https://w/1"},
                {"title": "Stale", "source": "W", "summary": "", "published": since - 86400, "link": "https://w/2"},
            ],
            "feeds": [{"name": "W", "ok": True}, {"name": "Down", "ok": False}],
        }

    monkeypatch.setattr(briefing.news, "collect_news", fake_collect)

    resp = client.get("/api/briefing")
    assert resp.status_code == 200
    body = resp.json()
    assert [h["title"] for h in body["world"]] == ["Fresh"]
    assert "Down" in body["feeds_failed"]
    sessions = body["workspace"]["sessions"]
    assert sessions["yesterday"] >= 1
    assert "Raport sprzedaży" in sessions["titles"]
    assert "Zeszły tydzień" not in sessions["titles"]
    assert body["workspace"]["model"] == "deepseek/deepseek-v4.1-flash"


def test_jobs_summary_puts_failures_and_next_runs_forward():
    summary = briefing.summarize_jobs([
        {"name": "Poranny raport", "enabled": True, "last_status": "error", "next_run_at": "2026-09-26T07:00"},
        {"name": "Backup", "enabled": True, "last_status": "ok", "next_run_at": "2026-09-25T23:00"},
        {"name": "Wyłączony", "enabled": False, "last_status": "error"},
    ])
    assert summary["active"] == 2
    assert summary["failing"] == ["Poranny raport"]
    assert [j["name"] for j in summary["next"]] == ["Backup", "Poranny raport"]


def test_custom_feeds_replace_defaults_and_bad_ones_fall_back():
    custom = briefing.resolve_briefing_feeds({"dashboard": {"briefing_feeds": ["https://example.pl/rss"]}})
    assert [f["url"] for f in custom] == ["https://example.pl/rss"]
    assert briefing.resolve_briefing_feeds({"dashboard": {"briefing_feeds": ["file:///etc/passwd"]}}) == \
        briefing.DEFAULT_BRIEFING_FEEDS


def test_task_summary_lists_blocked_first_and_ignores_finished_work():
    from types import SimpleNamespace as T
    tasks = [T(title="done", status="done", priority=9), T(title="ship", status="ready", priority=1),
             T(title="stuck", status="blocked", priority=0), T(title="urgent", status="ready", priority=5)]

    got = briefing.summarize_tasks(tasks)

    assert got["open"] == 3 and got["blocked"] == 1
    assert [i["title"] for i in got["items"]] == ["stuck", "urgent", "ship"]
