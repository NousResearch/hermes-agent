"""X ingestion: registry, timeline/mentions/following scrape, freshness.

Single browser session per run. Scrapes X.com through the saved logged-in
cookies (no X API cost) and returns tweet dicts with real creation times
recovered from the tweet ID snowflake — the extractor in
engagement_x_poster leaves ``created_at`` empty, which is what made
freshness impossible before this module.

Registry: ``data/x_registry.json`` (Sahil-editable, no deploy to change).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "data" / "x_registry.json"

_DEFAULTS = {
    "extra_accounts": [],
    "freshness_hours": {"quote": 6, "standalone": 6, "reply": 6},
    "max_candidates_per_run": 40,
    "min_tweet_chars": 40,
}

# Snowflake: creation ms = (id >> 22) + 1288834974657
_SNOWFLAKE_EPOCH = 1288834974657

_REL_TIME_RE = re.compile(
    r"^\s*(?:(\d+)\s*(h|hr|hr\.?|d|d\.?|w|w\.?|mo|mo\.?|m|min|s))?", re.I
)


def load_registry() -> dict:
    """Read the Sahil-editable registry, falling back to defaults per key."""
    reg = json.loads(_DEFAULTS_JSON)
    try:
        user = json.loads(REGISTRY_PATH.read_text())
        if isinstance(user, dict):
            for key in _DEFAULTS:
                if key in user:
                    reg[key] = user[key]
            fh = reg.get("freshness_hours") or {}
            for k, v in (fh or {}).items():
                if isinstance(v, (int, float)) and v > 0:
                    reg["freshness_hours"][k] = v
    except FileNotFoundError:
        pass
    except Exception as exc:  # bad JSON must not kill the scout
        print(f"[x-ingest] registry unreadable, using defaults: {exc}")
    return reg


_DEFAULTS_JSON = json.dumps(_DEFAULTS)


def tweet_age(tweet_id: str) -> float | None:
    """Age of a tweet in hours, from its snowflake ID.

    Returns None when the id is unparseable or implausible (future-dated or
    older than 3 years — those are usually mis-scraped permalink fragments).
    """
    try:
        ts = (int(tweet_id) >> 22) + _SNOWFLAKE_EPOCH
    except (ValueError, TypeError):
        return None
    now_ms = time.time() * 1000
    age_ms = now_ms - ts
    if age_ms < -3600_000 or age_ms > 3 * 365 * 24 * 3600_000:
        return None
    return age_ms / 3600_000


def apply_freshness(rows: list[dict], hours: float) -> list[dict]:
    """Keep rows whose snowflake age is within ``hours``.

    Rows with unparseable ids are kept (never drop on ambiguity) but
    flagged ``age_unknown`` so the scout can deprioritise them.
    """
    out = []
    for row in rows:
        age = tweet_age(row.get("id", ""))
        row["age_hours"] = round(age, 2) if age is not None else None
        if age is None:
            row["age_unknown"] = True
            out.append(row)
        elif age <= hours:
            row["age_unknown"] = False
            out.append(row)
    return out


def dedupe_by_id(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for row in rows:
        tid = str(row.get("id") or "")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(row)
    return out


def _parse_rel_time(text: str) -> str | None:
    """Best-effort ISO time from X's relative time labels. None if absent."""
    m = re.match(r"^\s*(\d+)\s*(h|d|w)\b", (text or "").strip(), re.I)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower().rstrip(".")
    hours = n * {"h": 1, "d": 24, "w": 168}[unit]
    return datetime.now(timezone.utc).timestamp() - hours * 3600


def ingest(
    *,
    include_home: bool = True,
    include_mentions: bool = True,
    include_following: bool = False,
    following_limit: int = 40,
    registry: dict | None = None,
) -> list[dict]:
    """Scrape home timeline + mentions (and optionally the following graph).

    Returns tweet dicts shaped like engagement_x_poster rows plus:
      source      — 'home' | 'mention' | 'following'
      created_at  — ISO from snowflake (best effort)
      age_hours   — float or None
      in_thread   — True for mentions (reply-eligible, low-hanging fruit)
    One browser session for the whole run. Fail-closed: a dead session
    yields what was scraped so far, never a crash.
    """
    reg = registry or load_registry()
    window = float((reg.get("freshness_hours") or {}).get("quote", 6))
    from engagement_x_poster import _make_browser, _ensure_logged_in

    pw, browser, context, page = _make_browser()
    rows: list[dict] = []
    try:
        if not _ensure_logged_in(page):
            return []
        if include_home:
            rows += _scrape_feed(page, source="home", limit=following_limit,
                                 window_hours=window)
        if include_mentions:
            # Best-effort: some accounts see "page doesn't exist" on the
            # mentions tab (X free-tier quirk). Home already surfaces
            # mentions inline, so a dead tab is acceptable.
            rows += _scrape_feed(page, "https://x.com/i/mentions",
                                 source="mention", limit=30,
                                 window_hours=window)
        extra_accounts = [a for a in (reg.get("extra_accounts") or []) if a]
        if include_following:
            rows += _scrape_following_graph(page, browser, limit=following_limit)
        for acct in extra_accounts:
            rows += _scrape_account(page, acct, limit=15)
    except Exception as exc:
        print(f"[x-ingest] session error: {exc}")
    finally:
        try:
            browser.close()
            pw.stop()
        except Exception:
            pass

    for row in rows:
        age = row.get("age_hours")
        if age is None:
            age = tweet_age(row.get("id", ""))
            row["age_hours"] = age
        if age is not None:
            row["created_at"] = datetime.fromtimestamp(
                time.time() - age * 3600, tz=timezone.utc).isoformat()
    return dedupe_by_id(rows)


def _scrape_feed(page, url: str = "https://x.com/home", *,
                 source: str = "home", limit: int = 60,
                 window_hours: float = 6.0) -> list[dict]:
    """Scroll the feed, stopping early once we've filled the freshness window.

    The home feed is a relevance mix, not strictly chronological, so we scroll
    until we've collected ``limit`` rows OR enough in-window rows to be
    confident the window is represented. Stopping on window-saturation avoids
    burning the session scrolling deep into old relevance posts.
    """
    rows: list[dict] = []
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
    except Exception as exc:
        print(f"[x-ingest] {source} load failed: {exc}")
        return rows
    seen: set[str] = set()
    in_window = 0
    for _scroll in range(10):  # up to 10 scroll steps
        articles = page.locator('article[data-testid="tweet"]')
        n = articles.count()
        for i in range(n):
            art = articles.nth(i)
            try:
                row = _extract_article(art)
                if not row or row["id"] in seen:
                    continue
                seen.add(row["id"])
                row["source"] = source
                rows.append(row)
                age = tweet_age(row["id"])
                if age is not None and age <= window_hours:
                    in_window += 1
                if len(rows) >= limit:
                    return rows
            except Exception:
                continue
        # Stop if we already have a healthy in-window set — no point scrolling
        # deeper into older relevance posts.
        if in_window >= 12:
            break
        before = n
        try:
            page.evaluate("window.scrollBy(0, 1500)")
            time.sleep(1.2)
            if page.locator('article[data-testid="tweet"]').count() == before:
                break
        except Exception:
            break
    return rows


def _scrape_account(page, account: str, limit: int = 15) -> list[dict]:
    from engagement_x_poster import _extract_tweets_from_page
    rows = _extract_tweets_from_page(page, account, limit)
    for r in rows:
        r["source"] = "following"
    return rows


def _scrape_following_graph(page, browser, limit: int) -> list[dict]:
    """Full /following scrape — phase 2, off by default until a clean week."""
    rows: list[dict] = []
    try:
        page.goto("https://x.com/i/flow/your_following",
                  wait_until="domcontentloaded", timeout=30000)
        # The following grid is heavy; this phase-2 path is best-effort.
        time.sleep(3)
        handles = []
        for a in page.locator('a[href^="/"]').all()[:300]:
            href = a.get_attribute("href") or ""
            m = re.match(r"^/([A-Za-z0-9_]{2,20})$", href)
            if m and m.group(1).lower() not in (
                "home", "settings", "i", "hashtag", "search", "explore", "notifications"):
                handles.append(m.group(1))
        for acct in handles:
            if not handles or handles.index(acct) % 5 == 0:
                print(f"[x-ingest] following: @{acct}")
            for r in _scrape_account(page, acct, limit=10):
                r["following_author"] = acct
                rows.append(r)
            if len(rows) >= limit * 4:
                break
    except Exception as exc:
        print(f"[x-ingest] following scrape failed: {exc}")
    for r in rows:
        r.setdefault("source", "following")
    return rows


def _extract_article(article) -> dict | None:
    """Extract a tweet dict from a home/mentions article node."""
    link = article.locator('a[href*="/status/"]').first
    if link.count() == 0:
        return None
    href = link.get_attribute("href") or ""
    m = re.search(r"/status/(\d+)", href)
    if not m:
        return None
    tweet_id = m.group(1)
    author_handle = ""
    m2 = re.search(r"/([A-Za-z0-9_]{1,20})/status/", href)
    if m2:
        author_handle = m2.group(1)
    text_el = article.locator('div[data-testid="tweetText"]')
    text = text_el.first.inner_text() if text_el.count() else ""
    if not text.strip():
        return None
    time_el = article.locator("time")
    rel = ""
    if time_el.count():
        rel = time_el.first.get_attribute("datetime") or time_el.first.inner_text()
    return {
        "id": tweet_id,
        "text": text.strip(),
        "author": author_handle or "unknown",
        "author_name": author_handle or "unknown",
        "url": f"https://x.com/i/web/status/{tweet_id}",
        "created_at": rel,
        "source": "home",
        "age_hours": None,
    }
