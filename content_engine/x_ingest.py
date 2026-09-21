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
import math
from urllib.parse import urlsplit
import re
import time
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "data" / "x_registry.json"

_DEFAULTS = {
    "extra_accounts": [],
    "account": "Sahil_Saghir",
    "following_max_pages": 20,
    "following_max_accounts": 5000,
    "following_cache_hours": 24,
    "following_cache_path": None,
    "freshness_hours": {"quote": 6, "standalone": 6, "reply": 6},
    "context_max_sources": 8,
    "context_article_limit": 20,
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


def snowflake_created_at(tweet_id: str) -> datetime | None:
    """Validate the literal ID before deriving its authoritative UTC timestamp."""
    if not isinstance(tweet_id, str) or not re.fullmatch(r"[1-9][0-9]{14,19}", tweet_id):
        return None
    value = int(tweet_id)
    if value >= 2**63:
        return None
    return datetime.fromtimestamp(((value >> 22) + _SNOWFLAKE_EPOCH) / 1000, timezone.utc)


def tweet_age(tweet_id: str) -> float | None:
    created = snowflake_created_at(tweet_id)
    if created is None:
        return None
    age = (time.time() - created.timestamp()) / 3600
    return age if 0 <= age <= 3 * 365 * 24 else None


def _utc_time(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def normalize_source(row: dict, *, origin: str | None = None, now=None) -> dict | None:
    """Require attributable identity; never repair malformed supplied identifiers."""
    if not isinstance(row, dict):
        return None
    tid = row.get("id")
    created = snowflake_created_at(tid)
    if created is None:
        return None
    stamp = time.time() if now is None else (now.timestamp() if isinstance(now, datetime) else float(now))
    supplied = row.get("created_at")
    if supplied not in (None, ""):
        parsed = _utc_time(supplied)
        if parsed is None or parsed.timestamp() > stamp or abs(parsed.timestamp() - created.timestamp()) > 1:
            return None
    origin = origin or row.get("origin") or row.get("source")
    if not isinstance(origin, str) or origin not in {"for_you", "following", "mention", "extra", "search", "own"}:
        return None
    url = row.get("url")
    if not isinstance(url, str) or not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    match = re.fullmatch(r"/(?:[A-Za-z0-9_]{1,15}|i/web)/status/([0-9]+)", parts.path)
    if (parts.scheme != "https" or parts.netloc not in {"x.com", "twitter.com", "www.x.com", "www.twitter.com"}
            or not match or match.group(1) != tid or parts.query or parts.fragment):
        return None
    age = (stamp - created.timestamp()) / 3600
    return {**row, "id": tid, "url": url, "created_at": created.isoformat(),
            "origin": origin, "source": origin, "age_hours": age, "age_unknown": False}


def source_freshness_issues(row: dict, *, now=None, hours: float = 6) -> list[str]:
    try:
        window = min(float(hours), 6.0)
    except (TypeError, ValueError):
        return ["invalid freshness window"]
    if not math.isfinite(window) or window <= 0:
        return ["invalid freshness window"]
    source = normalize_source(row, now=now)
    if source is None:
        return ["missing or invalid source identity, UTC timestamp, URL or origin"]
    age = source["age_hours"]
    if not math.isfinite(age) or age < 0:
        return ["future or unknown source timestamp"]
    return ["source older than freshness window"] if age > window else []


def apply_freshness(rows: list[dict], hours: float = 6) -> list[dict]:
    """Six-hour hard ceiling; ambiguity, future dates and missing origin reject."""
    now = time.time()
    return [normalize_source(row, now=now) for row in rows
            if not source_freshness_issues(row, now=now, hours=hours)]


def dedupe_by_id(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in rows:
        tid = str(row.get("id") or "")
        if not tid:
            continue
        origin = row.get("origin") or row.get("source")
        observed = row.get("origins")
        origins = list(dict.fromkeys(o for o in observed if isinstance(o, str))) if isinstance(observed, list) else []
        if origin and origin not in origins:
            origins.append(origin)
        if tid not in seen:
            seen[tid] = {**row, "origins": origins}
            continue
        existing = seen[tid]
        for observed_origin in origins:
            if observed_origin not in existing["origins"]:
                existing["origins"].append(observed_origin)
        if origin == "mention":
            # It was independently observed on the mentions route, not inferred
            # from For You. Keep that eligibility when feeds overlap.
            existing["source"] = existing["origin"] = "mention"
    return list(seen.values())


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
    diagnostics: dict | None = None,
) -> list[dict]:
    """Scrape home timeline + mentions (and optionally the following graph).

    Returns tweet dicts shaped like engagement_x_poster rows plus:
      source/origin — 'for_you' | 'mention' | 'following' | 'extra'
      created_at    — authoritative snowflake timestamp, UTC ISO
      age_hours     — revalidated 0..6 (no unknown/future/stale rows)
    diagnostics reports session, bounded following registry/feed coverage and
    post filtering counts. Mentions are not proof of full thread retrieval.
    One browser session for the whole run. Fail-closed: a dead session
    yields what was scraped so far, never a crash.
    """
    reg = registry if registry is not None else load_registry()
    diagnostics = diagnostics if diagnostics is not None else {}
    window = 6.0
    from engagement_x_poster import _ensure_logged_in
    from x_thread_context import make_readonly_browser

    try:
        pw, browser, context, page = make_readonly_browser()
    except Exception as exc:
        diagnostics["session"] = {"status": "unavailable", "reason": type(exc).__name__}
        return []
    rows: list[dict] = []
    try:
        if not _ensure_logged_in(page):
            diagnostics["session"] = {"status": "unavailable", "reason": "not_logged_in"}
            return []
        diagnostics["session"] = {"status": "authenticated"}
        if include_home:
            rows += _scrape_feed(page, source="home", limit=following_limit,
                                 window_hours=window)
        if include_mentions:
            # Separate notifications route; never claim home is mentions.
            rows += _scrape_feed(page, "https://x.com/notifications/mentions",
                                 source="mention", limit=30,
                                 window_hours=window)
        extra_accounts = [a for a in (reg.get("extra_accounts") or []) if a]
        if include_following:
            rows += _scrape_following_graph(page, browser, limit=following_limit,
                                           registry=reg, diagnostics=diagnostics)
        for acct in extra_accounts:
            rows += _scrape_account(page, acct, limit=15)
        diagnostics["observed_posts"] = len(rows)
        fresh = apply_freshness(rows, window)
        diagnostics["fresh_posts"] = len(fresh)
        diagnostics["rejected_posts"] = len(rows) - len(fresh)
        from x_thread_context import enrich_sources
        rows = enrich_sources(page, dedupe_by_id(fresh),
                              max_sources=reg.get("context_max_sources", 8),
                              article_limit=reg.get("context_article_limit", 20))
        diagnostics["conversation"] = [row["context_status"] for row in rows]
    except Exception as exc:
        diagnostics["session"] = {"status": "partial", "reason": type(exc).__name__}
        print(f"[x-ingest] session error: {exc}")
    finally:
        for close in (browser.close, pw.stop):
            try:
                close()
            except Exception as exc:
                diagnostics.setdefault("cleanup_errors", []).append(type(exc).__name__)

    fresh = apply_freshness(rows, window)
    diagnostics.setdefault("observed_posts", len(rows))
    diagnostics.setdefault("fresh_posts", len(fresh))
    diagnostics.setdefault("rejected_posts", len(rows) - len(fresh))
    for row in fresh:
        row.setdefault("thread_context", [])
        row.setdefault("context_status", {"state": "unknown", "complete": False,
                                          "source_id": row["id"], "context_ids": [],
                                          "reason": "session_partial",
                                          "missing": ["conversation context not fetched"]})
    return dedupe_by_id(fresh)


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
        if source in {"home", "for_you"}:
            tab = page.get_by_role("tab", name="For you", exact=True)
            if tab.get_attribute("aria-selected") != "true":
                tab.click(timeout=10000)
            if tab.get_attribute("aria-selected") != "true":
                return []
            source = "for_you"
        page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
    except Exception as exc:
        print(f"[x-ingest] {source} load failed: {exc}")
        return rows
    seen: set[str] = set()
    in_window = 0
    stagnant = 0
    for _scroll in range(10):  # bounded even on virtualised feeds
        before_ids = len(seen)
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
                row["origin"] = source
                rows.append(row)
                age = tweet_age(row["id"])
                if age is not None and 0 <= age <= min(window_hours, 6):
                    in_window += 1
                if len(rows) >= limit:
                    return rows
            except Exception:
                continue
        # Stop if we already have a healthy in-window set — no point scrolling
        # deeper into older relevance posts.
        if in_window >= 12:
            break
        stagnant = stagnant + 1 if len(seen) == before_ids else 0
        if stagnant >= 2:
            break
        try:
            page.evaluate("window.scrollBy(0, 1500)")
            time.sleep(1.2)
        except Exception:
            break
    return rows


def _scrape_account(page, account: str, limit: int = 15, *, source: str = "extra") -> list[dict]:
    if not isinstance(account, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,15}", account):
        return []
    # Use the same timestamp-permalink extractor as every other lane. The
    # legacy engagement extractor can attach a quoted post ID to outer text.
    rows = _scrape_feed(page, f"https://x.com/search?q=from%3A{account}&src=typed_query&f=live",
                        source=source, limit=limit)
    return [row for row in rows if str(row.get("author", "")).casefold() == account.casefold()]


def load_following_registry(fetch_page, *, cache_path, account: str,
                            max_pages: int = 20, max_accounts: int = 5000,
                            ttl_hours: float = 24, now=None) -> dict:
    """Bounded cursor walk with account-bound cache and explicit coverage.

    The page adapter returns handles, next_cursor and complete (an authoritative
    end, not 'no new DOM nodes'). A stalled/repeated cursor is always partial.
    """
    import os
    import tempfile
    if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", account):
        raise ValueError("invalid account handle")
    max_pages = max(1, min(int(max_pages), 100))
    max_accounts = max(1, min(int(max_accounts), 10000))
    ttl_hours = max(0, min(float(ttl_hours), 24))
    stamp = time.time() if now is None else (now.timestamp() if isinstance(now, datetime) else float(now))
    cache_path = Path(cache_path)
    try:
        cached = json.loads(cache_path.read_text())
        fetched = _utc_time(cached.get("fetched_at"))
        handles = cached.get("handles")
        if (cached.get("schema_version") == 1 and cached.get("account") == account
                and fetched is not None and 0 <= stamp - fetched.timestamp() < ttl_hours * 3600
                and isinstance(handles, list) and len(handles) <= max_accounts
                and all(isinstance(h, str) and re.fullmatch(r"[A-Za-z0-9_]{1,15}", h) for h in handles)
                and isinstance(cached.get("complete"), bool)):
            return {**cached, "cached": True}
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    handles, seen, cursors = [], set(), set()
    cursor, complete, reason, pages = None, False, "page_limit", 0
    for _ in range(max_pages):
        try:
            result = fetch_page(cursor)
            pages += 1
            batch = result["handles"]
            if not isinstance(batch, list):
                raise ValueError("invalid following page")
            for handle in batch:
                if not isinstance(handle, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle):
                    raise ValueError("invalid following handle")
                if handle.casefold() in seen:
                    continue
                if len(handles) >= max_accounts:
                    reason = "account_limit"
                    break
                seen.add(handle.casefold())
                handles.append(handle)
            if reason == "account_limit":
                break
            next_cursor = result.get("next_cursor")
            if result.get("complete") is True and next_cursor is None:
                complete, reason = True, ""
                break
            if next_cursor is None or not isinstance(next_cursor, str) or next_cursor in cursors:
                reason = result.get("partial_reason") or "cursor_stalled"
                break
            cursors.add(next_cursor)
            cursor = next_cursor
        except Exception as exc:
            reason = f"page_error:{type(exc).__name__}"
            break
    result = {"schema_version": 1, "account": account, "handles": handles, "complete": complete,
              "partial_reason": reason, "pages": pages, "cached": False,
              "fetched_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=cache_path.parent, prefix=cache_path.name + ".")
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(result, file)
        os.replace(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return result


def _following_page_reader(page, account: str):
    """Virtualised browser pagination; exact profile count can prove completion."""
    seen = set()
    expected = None
    page_number = 0
    def fetch(cursor):
        nonlocal expected, page_number
        if cursor is None:
            page.goto(f"https://x.com/{account}", wait_until="domcontentloaded", timeout=30000)
            link = page.locator(f'a[href="/{account}/following"]').first
            try:
                text = link.inner_text(timeout=10000)
                count = re.fullmatch(r"\s*([0-9][0-9,]*)\s+Following\s*", text)
                if count:
                    expected = int(count.group(1).replace(",", ""))
            except Exception:
                expected = None
            page.goto(f"https://x.com/{account}/following", wait_until="domcontentloaded", timeout=30000)
            if expected != 0:
                page.wait_for_selector('[data-testid="UserCell"]', timeout=15000)
        else:
            page.evaluate("window.scrollBy(0, 1500)")
            page.wait_for_timeout(1200)
        hrefs = page.locator('[data-testid="UserCell"] a[href]').evaluate_all(
            "nodes => nodes.map(a => a.getAttribute('href'))")
        handles = []
        for href in hrefs:
            match = re.fullmatch(r"/([A-Za-z0-9_]{1,15})", href or "")
            if match:
                handles.append(match.group(1))
        before = len(seen)
        seen.update(h.casefold() for h in handles)
        page_number += 1
        complete = expected is not None and len(seen) == expected
        stalled = len(seen) == before
        return {"handles": handles, "complete": complete,
                "next_cursor": None if complete or stalled else str(page_number),
                "partial_reason": "terminal_not_verified" if stalled and not complete else ""}
    return fetch


def _scrape_following_graph(page, browser, limit: int, *, registry=None, diagnostics=None) -> list[dict]:
    from hermes_constants import get_hermes_home
    reg = registry or {}
    account = reg.get("account", "Sahil_Saghir")
    coverage = load_following_registry(
        _following_page_reader(page, account), account=account,
        cache_path=reg.get("following_cache_path") or get_hermes_home() / "data/x_following_registry.json",
        max_pages=reg.get("following_max_pages", 20),
        max_accounts=reg.get("following_max_accounts", 5000),
        ttl_hours=reg.get("following_cache_hours", 24))
    rows, scanned = [], []
    budget = max(0, min(int(limit), 100))
    for account in coverage["handles"][:budget]:
        try:
            batch = _scrape_account(page, account, limit=10, source="following")
            rows.extend({**r, "following_author": account} for r in batch)
            scanned.append(account)
        except Exception:
            continue
    if diagnostics is not None:
        diagnostics["following"] = {**coverage, "accounts_scanned": len(scanned),
                                    "registry_size": len(coverage["handles"]),
                                    "feed_complete": False,
                                    "feed_partial_reason": "bounded_recent_posts_per_account"}
    return rows


def _extract_article(article) -> dict | None:
    """Extract a tweet dict from a home/mentions article node."""
    social = article.locator('[data-testid="socialContext"]')
    if social.count() and re.search(r"reposted|retweeted", social.first.inner_text(), re.I):
        return None
    # Only the timestamp permalink identifies the outer post, not a quoted
    # card or a media/status link that happened to appear first.
    outer = ':not([role="link"] *):not([data-testid="quoteTweet"] *)'
    link = article.locator('a[href*="/status/"]:has(time)' + outer).first
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
    text_el = article.locator('div[data-testid="tweetText"]' + outer)
    text = text_el.first.inner_text() if text_el.count() else ""
    if not text.strip():
        return None
    time_el = link.locator("time")
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
