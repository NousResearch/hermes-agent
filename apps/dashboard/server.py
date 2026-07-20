#!/usr/bin/env python3
"""Hermes Hub — all-in-one dashboard server.

A dependency-free (stdlib only) local server that:
  * serves the dashboard frontend from ./public
  * proxies + normalizes news (RSS/Atom), weather (Open-Meteo) and market data
    behind a small JSON API, so the browser never fights CORS
  * caches upstream responses in memory with per-endpoint TTLs
  * falls back to bundled sample data whenever the network is unavailable,
    so the dashboard is fully usable offline (responses are marked
    "source": "sample" and the UI surfaces that)

Usage:
    python3 server.py [--port 8787] [--host 127.0.0.1] [--offline]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import email.utils
import hmac
import ipaddress
import socket
import sqlite3
import gzip
import html
import io
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import assistant as assistant_module
import indicators
from automations import Automations
from evolve import Reflection
from ics import parse_ics
from telemetry import Telemetry

APP_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = APP_DIR / "public"
SAMPLE_PATH = APP_DIR / "sample_data.json"

USER_AGENT = "HermesHub/1.0 (+local dashboard; personal use)"
FETCH_TIMEOUT = 12  # seconds

# ---------------------------------------------------------------------------
# Curated news sources.  Every topic maps to a list of RSS/Atom feeds; results
# are merged, deduped and sorted newest-first.
# ---------------------------------------------------------------------------
NEWS_SOURCES: dict[str, list[dict[str, str]]] = {
    "world": [
        {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
        {"name": "NPR World", "url": "https://feeds.npr.org/1004/rss.xml"},
        {"name": "The Guardian World", "url": "https://www.theguardian.com/world/rss"},
    ],
    "tech": [
        {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index"},
        {"name": "Hacker News", "url": "https://hnrss.org/frontpage"},
    ],
    "business": [
        {"name": "CNBC", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"},
        {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
    ],
    "science": [
        {"name": "ScienceDaily", "url": "https://www.sciencedaily.com/rss/all.xml"},
        {"name": "NASA", "url": "https://www.nasa.gov/feed/"},
    ],
    "sports": [
        {"name": "BBC Sport", "url": "https://feeds.bbci.co.uk/sport/rss.xml"},
        {"name": "ESPN", "url": "https://www.espn.com/espn/rss/news"},
    ],
    "entertainment": [
        {"name": "Variety", "url": "https://variety.com/feed/"},
        {"name": "BBC Entertainment", "url": "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"},
    ],
    "medicine": [
        {"name": "ScienceDaily Health", "url": "https://www.sciencedaily.com/rss/health_medicine.xml"},
        {"name": "MedlinePlus", "url": "https://medlineplus.gov/feeds/news_en.xml"},
        {"name": "STAT", "url": "https://www.statnews.com/feed/"},
        {"name": "Medscape", "url": "https://www.medscape.com/cx/rssfeeds/2700.xml"},
        {"name": "WHO News", "url": "https://www.who.int/rss-feeds/news-english.xml"},
    ],
    "gaming": [
        {"name": "IGN", "url": "https://feeds.ign.com/ign/games-all"},
        {"name": "Polygon", "url": "https://www.polygon.com/rss/index.xml"},
        {"name": "Eurogamer", "url": "https://www.eurogamer.net/feed"},
        {"name": "PC Gamer", "url": "https://www.pcgamer.com/rss/"},
        {"name": "Rock Paper Shotgun", "url": "https://www.rockpapershotgun.com/feed"},
        {"name": "GameSpot", "url": "https://www.gamespot.com/feeds/news/"},
    ],
    "southafrica": [
        {"name": "News24", "url": "https://feeds.24.com/articles/news24/TopStories/rss"},
        {"name": "Daily Maverick", "url": "https://www.dailymaverick.co.za/dmrss/"},
        {"name": "TimesLIVE", "url": "https://www.timeslive.co.za/rss/"},
        {"name": "Mail & Guardian", "url": "https://mg.co.za/feed/"},
        {"name": "IOL", "url": "https://www.iol.co.za/cmlink/1.640"},
    ],
    "africa": [
        {"name": "BBC Africa", "url": "https://feeds.bbci.co.uk/news/world/africa/rss.xml"},
        {"name": "AllAfrica", "url": "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf"},
        {"name": "Al Jazeera Africa", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    ],
    "ai": [
        {"name": "MIT Tech Review", "url": "https://www.technologyreview.com/feed/"},
        {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
        {"name": "The Verge AI", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"},
        {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab"},
    ],
    "finance": [
        {"name": "Reuters Markets", "url": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"},
        {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
        {"name": "Moneyweb (SA)", "url": "https://www.moneyweb.co.za/feed/"},
        {"name": "Investing.com", "url": "https://www.investing.com/rss/news.rss"},
    ],
}
# "top" aggregates the first feed of every topic.
NEWS_SOURCES["top"] = [sources[0] for sources in NEWS_SOURCES.values()]


class FeedConfig:
    """User-editable news sources/topics, persisted in data/feeds.json.

    "top" is virtual: it aggregates the first source of every other topic.
    """

    MAX_TOPICS = 16
    MAX_SOURCES_PER_TOPIC = 12

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._topics: dict[str, list[dict[str, str]]] = {
            k: [dict(s) for s in v] for k, v in NEWS_SOURCES.items() if k != "top"
        }
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded:
                    self._topics = {
                        str(k): [{"name": str(s["name"]), "url": str(s["url"])} for s in v]
                        for k, v in loaded.items() if k != "top" and isinstance(v, list)
                    }
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                pass  # fall back to defaults; next save rewrites the file

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._topics, ensure_ascii=False, indent=1), encoding="utf-8")

    def topics(self) -> list[str]:
        with self._lock:
            return ["top", *self._topics.keys()]

    def sources_for(self, topic: str) -> list[dict[str, str]] | None:
        with self._lock:
            if topic == "top":
                return [v[0] for v in self._topics.values() if v]
            sources = self._topics.get(topic)
            return [dict(s) for s in sources] if sources is not None else None

    def snapshot(self) -> dict:
        with self._lock:
            return {"topics": ["top", *self._topics.keys()],
                    "sources": {k: [dict(s) for s in v] for k, v in self._topics.items()}}

    def restore(self, sources: dict) -> None:
        """Adopt a backup snapshot's topic→sources map wholesale."""
        with self._lock:
            self._topics = {
                str(k)[:40]: [
                    {"name": str(s["name"]), "url": str(s["url"])}
                    for s in v if isinstance(s, dict) and s.get("url")
                ][: self.MAX_SOURCES_PER_TOPIC]
                for k, v in list(sources.items())[: self.MAX_TOPICS]
                if k != "top" and isinstance(v, list)
            }
            self._save()

    def add_topic(self, name: str) -> None:
        key = re.sub(r"[^a-z0-9-]", "", name.strip().lower().replace(" ", "-"))[:24]
        if not key or key == "top":
            raise ApiError(400, "topic name must contain letters/numbers")
        with self._lock:
            if key in self._topics:
                raise ApiError(400, f"topic {key!r} already exists")
            if len(self._topics) >= self.MAX_TOPICS:
                raise ApiError(400, f"topic limit ({self.MAX_TOPICS}) reached")
            self._topics[key] = []
            self._save()

    GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    def add_search(self, name: str, query: str) -> None:
        """Follow an arbitrary news search as a topic (Google News RSS)."""
        query = " ".join((query or "").split())[:120]
        if not query:
            raise ApiError(400, "search needs a query")
        name = (name.strip() or query)[:40]
        self.add_topic(name)  # validates + slugifies; raises on dupes
        key = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-"))[:24]
        url = self.GOOGLE_NEWS.format(q=urllib.parse.quote(query))
        self.add_source(key, f"Google News: {query[:40]}", url)

    def remove_topic(self, name: str) -> None:
        with self._lock:
            if name not in self._topics:
                raise ApiError(404, f"no topic {name!r}")
            if len(self._topics) <= 1:
                raise ApiError(400, "cannot remove the last topic")
            del self._topics[name]
            self._save()

    def add_source(self, topic: str, name: str, url: str) -> None:
        name = name.strip()[:60]
        url = url.strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ApiError(400, "source url must be http(s)")
        if not name:
            raise ApiError(400, "source needs a name")
        with self._lock:
            if topic not in self._topics:
                raise ApiError(404, f"no topic {topic!r}")
            sources = self._topics[topic]
            if len(sources) >= self.MAX_SOURCES_PER_TOPIC:
                raise ApiError(400, f"source limit ({self.MAX_SOURCES_PER_TOPIC}) reached")
            if any(s["url"] == url for s in sources):
                raise ApiError(400, "that feed is already in this topic")
            sources.append({"name": name, "url": url})
            self._save()

    def remove_source(self, topic: str, url: str) -> None:
        with self._lock:
            if topic not in self._topics:
                raise ApiError(404, f"no topic {topic!r}")
            before = len(self._topics[topic])
            self._topics[topic] = [s for s in self._topics[topic] if s["url"] != url]
            if len(self._topics[topic]) == before:
                raise ApiError(404, "no such source in this topic")
            self._save()

    def reset(self) -> None:
        with self._lock:
            self._topics = {
                k: [dict(s) for s in v] for k, v in NEWS_SOURCES.items() if k != "top"
            }
            self._save()

NEWS_TTL = 5 * 60
WEATHER_TTL = 10 * 60
MARKETS_TTL = 3 * 60
GEOCODE_TTL = 24 * 60 * 60
WORLDSTATE_TTL = 10 * 60
READER_TTL = 30 * 60
ICS_TTL = 15 * 60
BACKUP_KEEP = 20  # newest server-side snapshots retained

# ---------------------------------------------------------------------------
# "State of the world" heuristic model.
#
# Each domain starts at a calm baseline and moves as keywords are matched in
# current headlines: tension terms subtract, easing terms add.  The result is
# a 0-100 stability index with the matched headlines kept as evidence.  This
# is a transparent headline heuristic, NOT an authoritative assessment — the
# UI labels it as such.
# ---------------------------------------------------------------------------
WORLD_DOMAINS: dict[str, dict] = {
    "geopolitics": {
        "name": "Geopolitics",
        "topics": ["world", "top"],
        "baseline": 62,
        "tension": ["war", "invasion", "strike*", "missile*", "attack*", "conflict*",
                    "sanction*", "coup", "escalat*", "troops", "hostilit*", "nuclear threat"],
        "easing": ["ceasefire", "peace", "truce", "accord", "agreement", "de-escalat*",
                   "talks resume", "treaty", "diplomatic breakthrough"],
    },
    "economy": {
        "name": "Economy",
        "topics": ["business", "top"],
        "baseline": 65,
        "tension": ["recession", "inflation surge", "default*", "layoff*", "crash*",
                    "bankrupt*", "crisis", "tariff*", "downturn", "sell-off", "selloff"],
        "easing": ["rate cut", "growth", "record high", "optimism", "rally",
                   "expansion", "hiring", "investment", "steady", "no change to rates"],
    },
    "technology": {
        "name": "Technology & Cyber",
        "topics": ["tech", "top"],
        "baseline": 70,
        "tension": ["breach*", "hack*", "ransomware", "outage*", "zero-day", "exploit*",
                    "vulnerabilit*", "data leak", "cyberattack*", "malware"],
        "easing": ["open-source", "breakthrough", "launch*", "milestone", "ship*",
                   "release*", "patch*", "advance*"],
    },
    "climate": {
        "name": "Climate & Environment",
        "topics": ["science", "world"],
        "baseline": 55,
        "tension": ["hurricane*", "wildfire*", "flood*", "drought*", "heatwave*", "storm*",
                    "record temperature*", "emergenc*", "extreme weather", "eruption*"],
        "easing": ["emission*", "renewable*", "climate pledge", "conservation",
                   "restoration", "clean energy", "solar", "wind power"],
    },
    "health": {
        "name": "Public Health",
        "topics": ["science", "world"],
        "baseline": 72,
        "tension": ["outbreak*", "epidemic", "pandemic", "virus spread", "infection surge",
                    "health emergency", "contamination"],
        "easing": ["vaccine*", "cure*", "therap*", "trial success", "breakthrough",
                   "restore*", "approved treatment", "eradicat*"],
    },
    "markets": {
        "name": "Markets",
        "topics": ["business"],
        "baseline": 66,
        "tension": ["plunge*", "tumble*", "bear market", "volatilit*", "panic", "slump*"],
        "easing": ["rall*", "gain*", "record high*", "steady", "rebound*", "surge*"],
    },
}


def score_level(score: float) -> str:
    if score >= 75:
        return "stable"
    if score >= 60:
        return "watch"
    if score >= 40:
        return "elevated"
    return "critical"

DEFAULT_CRYPTO_IDS = ["bitcoin", "ethereum", "solana", "dogecoin"]

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".ics": "text/calendar; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}


class TTLCache:
    """Tiny thread-safe TTL cache."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires, value = entry
            if time.monotonic() > expires:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: object, ttl: float) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


CACHE = TTLCache()


class CalendarConfig:
    """Read-only ICS calendar subscriptions, persisted in data/calendars.json."""

    MAX_CALENDARS = 8

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._calendars: list[dict[str, str]] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self._calendars = [
                        {"name": str(c["name"]), "url": str(c["url"])}
                        for c in loaded if isinstance(c, dict) and c.get("url")
                    ]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._calendars, ensure_ascii=False, indent=1), encoding="utf-8")

    def list(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(c) for c in self._calendars]

    def add(self, name: str, url: str) -> None:
        name = name.strip()[:60]
        url = url.strip().replace("webcal://", "https://", 1)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ApiError(400, "calendar url must be http(s) or webcal")
        if not name:
            raise ApiError(400, "calendar needs a name")
        with self._lock:
            if len(self._calendars) >= self.MAX_CALENDARS:
                raise ApiError(400, f"calendar limit ({self.MAX_CALENDARS}) reached")
            if any(c["url"] == url for c in self._calendars):
                raise ApiError(400, "that calendar is already subscribed")
            self._calendars.append({"name": name, "url": url})
            self._save()

    def remove(self, url: str) -> None:
        with self._lock:
            before = len(self._calendars)
            self._calendars = [c for c in self._calendars if c["url"] != url]
            if len(self._calendars) == before:
                raise ApiError(404, "no such calendar")
            self._save()

    def restore(self, calendars: list) -> None:
        """Adopt a backup snapshot's subscription list wholesale."""
        with self._lock:
            self._calendars = [
                {"name": str(c["name"])[:60], "url": str(c["url"])}
                for c in calendars if isinstance(c, dict) and c.get("url") and c.get("name")
            ][: self.MAX_CALENDARS]
            self._save()


class StateStore:
    """SQLite-backed dashboard state, shared across devices.

    Single-row versioned blob with optimistic concurrency: writers send the
    revision they based their edit on; a mismatch returns a conflict so the
    client can adopt the newer state instead of clobbering it.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS state ("
            " id INTEGER PRIMARY KEY CHECK (id = 1),"
            " rev INTEGER NOT NULL,"
            " updated TEXT NOT NULL,"
            " payload TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self) -> dict:
        with self._lock:
            row = self._conn.execute("SELECT rev, updated, payload FROM state WHERE id = 1").fetchone()
        if row is None:
            return {"rev": 0, "updated": None, "state": None}
        return {"rev": row[0], "updated": row[1], "state": json.loads(row[2])}

    def rev(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT rev FROM state WHERE id = 1").fetchone()
        return row[0] if row else 0

    def put(self, state: dict, base_rev: int | None) -> tuple[bool, int]:
        """Returns (ok, current_rev). ok=False means a conflict."""
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(state, ensure_ascii=False)
        with self._lock:
            row = self._conn.execute("SELECT rev FROM state WHERE id = 1").fetchone()
            current = row[0] if row else 0
            if base_rev is not None and base_rev != current:
                return False, current
            new_rev = current + 1
            self._conn.execute(
                "INSERT INTO state (id, rev, updated, payload) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET rev = ?, updated = ?, payload = ?",
                (new_rev, now, payload, new_rev, now, payload),
            )
            self._conn.commit()
            return True, new_rev


# ---------------------------------------------------------------------------
# SSRF guard for the article reader.  The reader opens arbitrary URLs that
# arrive from untrusted feed items, so it must not reach loopback, link-local,
# private or otherwise non-global addresses — otherwise a crafted article link
# (or a redirect from one) could pull cloud-metadata (169.254.169.254) or poke
# internal services.  We resolve the host and reject if ANY resolved address is
# non-global, and re-check every redirect hop (urllib follows redirects, so an
# initial-URL check alone is bypassable).  News/ICS subscriptions are
# deliberately user-curated and may legitimately point at a LAN host, so they
# are not guarded here — only the size cap below applies to every fetch.
# ---------------------------------------------------------------------------
MAX_FETCH_BYTES = 8 * 1024 * 1024  # cap responses so a huge upstream can't OOM us


def host_is_blocked(host: str) -> bool:
    if not host:
        return True
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        return True
    # Literal IP: check directly (skip DNS).
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        pass
    # Hostname: resolve and reject if any address is non-global.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False  # unresolvable — let the fetch fail naturally upstream
    for info in infos:
        try:
            if not ipaddress.ip_address(info[4][0]).is_global:
                return True
        except ValueError:
            return True
    return False


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target against the SSRF guard."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme not in ("http", "https") or host_is_blocked(parsed.hostname or ""):
            raise urllib.error.URLError("blocked redirect to private/invalid host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_GUARDED_OPENER = urllib.request.build_opener(_GuardedRedirectHandler())


def fetch_url(url: str, timeout: float = FETCH_TIMEOUT, guard: bool = False) -> bytes:
    if guard:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or host_is_blocked(parsed.hostname or ""):
            raise ApiError(400, "refusing to fetch private/internal addresses")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Encoding": "gzip",
        },
    )
    opener = _GUARDED_OPENER.open if guard else urllib.request.urlopen
    with opener(req, timeout=timeout) as resp:
        raw = resp.read(MAX_FETCH_BYTES + 1)
        if len(raw) > MAX_FETCH_BYTES:
            raise ApiError(502, "upstream response too large")
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        return raw


# ---------------------------------------------------------------------------
# RSS / Atom parsing
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def strip_html(text: str, limit: int = 260) -> str:
    text = TAG_RE.sub(" ", text or "")
    text = html.unescape(text)
    text = WS_RE.sub(" ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return text


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _first_child_text(elem: ET.Element, *names: str) -> str:
    wanted = set(names)
    for child in elem:
        if _localname(child.tag) in wanted and child.text:
            return child.text.strip()
    return ""


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:  # RFC 822 (RSS)
        dt = email.utils.parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        pass
    try:  # ISO 8601 (Atom)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_feed(xml_bytes: bytes, source_name: str) -> list[dict]:
    """Parse RSS 2.0 or Atom into normalized item dicts."""
    root = ET.fromstring(xml_bytes)
    root_name = _localname(root.tag)
    items: list[dict] = []

    if root_name == "rss" or root_name == "rdf":
        nodes = [el for el in root.iter() if _localname(el.tag) == "item"]
        for node in nodes:
            title = _first_child_text(node, "title")
            link = _first_child_text(node, "link", "guid")
            desc = _first_child_text(node, "description", "encoded", "summary")
            published = parse_date(_first_child_text(node, "pubdate", "date"))
            if title and link:
                items.append(_news_item(title, link, desc, published, source_name))
    elif root_name == "feed":  # Atom
        for node in root:
            if _localname(node.tag) != "entry":
                continue
            title = _first_child_text(node, "title")
            link = ""
            for child in node:
                if _localname(child.tag) == "link":
                    rel = child.get("rel", "alternate")
                    if rel == "alternate" or not link:
                        link = child.get("href", "")
            desc = _first_child_text(node, "summary", "content")
            published = parse_date(
                _first_child_text(node, "published", "updated")
            )
            if title and link:
                items.append(_news_item(title, link, desc, published, source_name))
    else:
        raise ValueError(f"unrecognized feed root <{root_name}>")
    return items


def _news_item(title, link, desc, published: datetime | None, source: str) -> dict:
    return {
        "title": strip_html(title, 200),
        "url": link.strip(),
        "summary": strip_html(desc),
        "source": source,
        "published": published.astimezone(timezone.utc).isoformat() if published else None,
    }


def merge_items(all_items: list[dict], limit: int) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for item in all_items:
        key = item["url"] or item["title"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=lambda i: i["published"] or "", reverse=True)
    return unique[:limit]


# ---------------------------------------------------------------------------
# Sample data (offline fallback)
# ---------------------------------------------------------------------------
def load_samples() -> dict:
    with SAMPLE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


SAMPLES = load_samples()


def sample_news(topic: str, limit: int) -> dict:
    now = datetime.now(timezone.utc)
    raw = SAMPLES["news"].get(topic) or SAMPLES["news"]["top"]
    items = []
    for entry in raw[:limit]:
        item = dict(entry)
        # age_minutes keeps sample items looking current no matter when served
        item["published"] = (now - timedelta(minutes=item.pop("age_minutes", 60))).isoformat()
        items.append(item)
    return {"topic": topic, "source": "sample", "items": items}


def sample_weather(name: str | None = None) -> dict:
    data = json.loads(json.dumps(SAMPLES["weather"]))  # deep copy
    data["source"] = "sample"
    if name:
        data["location"]["name"] = name
    now = datetime.now(timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0)
    for i, hour in enumerate(data["hourly"]):
        hour["time"] = (start + timedelta(hours=i)).isoformat()
    today = now.date()
    for i, day in enumerate(data["daily"]):
        day["date"] = (today + timedelta(days=i)).isoformat()
    return data


def sample_markets(ids: list[str] | None = None) -> dict:
    data = json.loads(json.dumps(SAMPLES["markets"]))
    data["source"] = "sample"
    for a in data["assets"]:
        a.setdefault("id", a["name"].lower())
    if ids:
        wanted = {i.lower() for i in ids}
        subset = [a for a in data["assets"]
                  if a["name"].lower() in wanted or a["symbol"].lower() in wanted]
        if subset:
            data["assets"] = subset
    return data


# ---------------------------------------------------------------------------
# World state: keyword-score current headlines per domain
# ---------------------------------------------------------------------------
def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    # Whole-word match by default so "war" never fires on "awards" or
    # "warning". A trailing "*" marks a stem: "escalat*" catches "escalation".
    hits = []
    for kw in keywords:
        if kw.endswith("*"):
            pattern = rf"\b{re.escape(kw[:-1])}"
        else:
            pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, text):
            hits.append(kw.rstrip("*"))
    return hits


def compute_worldstate(news_by_topic: dict[str, dict]) -> dict:
    domains = []
    sources = set()
    for key, spec in WORLD_DOMAINS.items():
        items: list[dict] = []
        seen: set[str] = set()
        for topic in spec["topics"]:
            payload = news_by_topic.get(topic)
            if not payload:
                continue
            sources.add(payload["source"])
            for item in payload["items"]:
                if item["url"] not in seen:
                    seen.add(item["url"])
                    items.append(item)

        score = float(spec["baseline"])
        signals = []
        for item in items:
            text = f"{item['title']} {item.get('summary', '')}".lower()
            hit_tension = _keyword_hits(text, spec["tension"])
            hit_easing = _keyword_hits(text, spec["easing"])
            delta = -4.0 * len(hit_tension) + 3.0 * len(hit_easing)
            if delta:
                signals.append(
                    {
                        "headline": item["title"],
                        "url": item["url"],
                        "source": item["source"],
                        "delta": delta,
                        "keywords": hit_tension + hit_easing,
                    }
                )
                score += delta
        score = max(5.0, min(95.0, score))
        signals.sort(key=lambda s: abs(s["delta"]), reverse=True)

        rising = sum(1 for s in signals if s["delta"] > 0)
        falling = sum(1 for s in signals if s["delta"] < 0)
        if not signals:
            explanation = (
                f"No significant signals in the current feed window; index holds "
                f"at its baseline of {spec['baseline']}."
            )
        else:
            explanation = (
                f"{len(signals)} signal(s) in current headlines — "
                f"{falling} raising tension, {rising} easing it. "
                f"Index moved from baseline {spec['baseline']} to {round(score)}."
            )
        domains.append(
            {
                "id": key,
                "name": spec["name"],
                "score": round(score),
                "level": score_level(score),
                "explanation": explanation,
                "signals": signals[:4],
            }
        )
    overall = round(sum(d["score"] for d in domains) / len(domains))
    return {
        "source": "sample" if sources == {"sample"} else "live",
        "generated": datetime.now(timezone.utc).isoformat(),
        "overall": {"score": overall, "level": score_level(overall)},
        "domains": domains,
        "method": (
            "Heuristic stability index (0-100) derived from keyword analysis of "
            "current headlines against a calm baseline. Informational only — "
            "not an official assessment."
        ),
    }


# ---------------------------------------------------------------------------
# Reader view: fetch an article and extract readable text (stdlib only)
# ---------------------------------------------------------------------------
class _ArticleExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "aside", "header", "form", "noscript", "svg", "button"}
    TEXTY = {"p", "h1", "h2", "h3", "li", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self.title = ""
        self.image = ""
        self.author = ""
        self.published = ""
        self._stack: list[str] = []
        self._buffer: list[str] = []
        self._current: str | None = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            prop = (a.get("property") or a.get("name") or "").lower()
            content = a.get("content") or ""
            if prop == "og:image" and not self.image:
                self.image = content
            elif prop in ("author", "article:author") and not self.author:
                self.author = content
            elif prop in ("article:published_time", "og:article:published_time") and not self.published:
                self.published = content
            return
        if tag in self.SKIP:
            self._stack.append(tag)
        elif tag == "title" and not self.title:
            self._in_title = True
        elif not self._stack and tag in self.TEXTY and self._current is None:
            self._current = tag
            self._buffer = []

    def handle_endtag(self, tag):
        if self._stack and tag == self._stack[-1]:
            self._stack.pop()
        elif tag == "title":
            self._in_title = False
        elif self._current == tag:
            text = WS_RE.sub(" ", "".join(self._buffer)).strip()
            if len(text) > 2:
                self.blocks.append((tag, text))
            self._current = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._current and not self._stack:
            self._buffer.append(data)


def live_reader(url: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ApiError(400, "only http(s) URLs can be opened")
    if host_is_blocked(parsed.hostname or ""):
        raise ApiError(400, "refusing to fetch private/internal addresses")

    raw = fetch_url(url, guard=True)
    extractor = _ArticleExtractor()
    try:
        extractor.feed(raw.decode("utf-8", errors="replace"))
    except Exception:
        pass

    blocks = extractor.blocks
    # Drop boilerplate-ish tiny paragraphs, keep at most ~120 blocks.
    blocks = [(tag, text) for tag, text in blocks if len(text) > 30 or tag.startswith("h")][:120]
    if not blocks:
        raise RuntimeError("no readable text found")
    total = 0
    words = 0
    trimmed = []
    for tag, text in blocks:
        trimmed.append({"tag": tag, "text": text})
        total += len(text)
        words += text.count(" ") + 1
        if total > 20000:
            break
    image = extractor.image.strip()
    if image and not host_is_blocked(urllib.parse.urlparse(image).hostname or ""):
        pass  # keep only http(s) images on public hosts
    elif image and not image.startswith("http"):
        image = ""  # skip data:/relative
    return {
        "source": "live",
        "url": url,
        "title": WS_RE.sub(" ", extractor.title).strip() or url,
        "image": image,
        "author": WS_RE.sub(" ", extractor.author).strip()[:80],
        "published": extractor.published.strip()[:32],
        "readingMinutes": max(1, round(words / 200)),
        "blocks": trimmed,
    }


# ---------------------------------------------------------------------------
# Sports scoreboards (ESPN public JSON — no key; unofficial, so parse defensively)
# ---------------------------------------------------------------------------
SCORES_TTL = 60  # short: live games move fast
SPORT_LEAGUES = {
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "epl": ("soccer", "eng.1"),
    "mls": ("soccer", "usa.1"),
    "ncaaf": ("football", "college-football"),
    "wnba": ("basketball", "wnba"),
    "urc": ("rugby", "270557"),        # United Rugby Championship (SA franchises)
    "rugbyc": ("rugby", "242041"),     # The Rugby Championship (Springboks)
    "cricket": ("cricket", "8048"),    # International cricket fixtures
    # Soccer — European top flights, Champions League and the SA PSL.
    "laliga": ("soccer", "esp.1"),
    "seriea": ("soccer", "ita.1"),
    "bundesliga": ("soccer", "ger.1"),
    "ligue1": ("soccer", "fra.1"),
    "ucl": ("soccer", "uefa.champions"),
    "psl": ("soccer", "rsa.1"),        # DStv Premiership (South Africa)
    # Combat & racket — two-competitor events (athletes, not teams).
    "mma": ("mma", "ufc"),
    "atp": ("tennis", "atp"),
    "wta": ("tennis", "wta"),
}
# Leagues whose competitors are individual athletes (no home/away flags).
_INDIVIDUAL_LEAGUES = {"mma", "atp", "wta"}


def _norm_competitor(c: dict) -> dict:
    team = c.get("team") or {}
    athlete = c.get("athlete") or {}
    name = (team.get("displayName") or team.get("name")
            or athlete.get("displayName") or athlete.get("shortName") or "")
    abbr = (team.get("abbreviation") or team.get("shortDisplayName")
            or athlete.get("shortName") or name or "?")
    return {
        "abbr": abbr,
        "name": name,
        "score": c.get("score"),
        "home": c.get("homeAway") == "home",
        "winner": c.get("winner"),
    }


def live_scores(league: str) -> dict:
    sport, lg = SPORT_LEAGUES[league]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard"
    raw = json.loads(fetch_url(url))
    games = []
    for event in raw.get("events", []):
        try:
            comp = (event.get("competitions") or [{}])[0]
            status = (event.get("status") or {}).get("type") or {}
            competitors = [_norm_competitor(c) for c in comp.get("competitors", [])]
            home = next((c for c in competitors if c["home"]), None)
            away = next((c for c in competitors if not c["home"]), None)
            # Individual sports (MMA/tennis) carry no home/away flags — order them.
            if (not home or not away) and len(competitors) >= 2:
                away, home = competitors[0], competitors[1]
            if not home or not away:
                continue
            games.append({
                "id": event.get("id"),
                "state": status.get("state", "pre"),   # pre | in | post
                "status": status.get("shortDetail") or status.get("detail") or "",
                "clock": (event.get("status") or {}).get("displayClock"),
                "period": (event.get("status") or {}).get("period"),
                "start": event.get("date"),
                "home": home,
                "away": away,
            })
        except Exception:
            continue  # one malformed event must not sink the board
    return {"source": "live", "league": league, "games": games}


STANDINGS_TTL = 30 * 60
# Preferred stat columns per sport family (first few present are shown).
_STAND_STATS = ["wins", "losses", "ties", "winPercent", "points",
                "pointDifferential", "gamesBehind"]
_STAND_LABEL = {"wins": "W", "losses": "L", "ties": "T", "winPercent": "PCT",
                "points": "PTS", "pointDifferential": "DIFF", "gamesBehind": "GB"}


def live_standings(league: str) -> dict:
    sport, lg = SPORT_LEAGUES[league]
    url = f"https://site.api.espn.com/apis/v2/sports/{sport}/{lg}/standings"
    raw = json.loads(fetch_url(url))
    children = raw.get("children") or ([raw] if raw.get("standings") else [])
    cols: list[str] = []
    groups = []
    for child in children:
        entries = ((child.get("standings") or {}).get("entries")) or []
        teams = []
        for e in entries:
            team = e.get("team") or {}
            stats = {}
            for s in e.get("stats", []):
                name = s.get("name")
                if name in _STAND_STATS:
                    stats[name] = s.get("displayValue")
                    if name not in cols:
                        cols.append(name)
            teams.append({"abbr": team.get("abbreviation") or "?",
                          "name": team.get("displayName") or team.get("shortDisplayName") or "",
                          "stats": stats})
        if teams:
            groups.append({"name": child.get("name") or child.get("abbreviation") or "", "teams": teams})
    if not groups:
        raise RuntimeError("no standings")
    ordered = [c for c in _STAND_STATS if c in cols][:4]
    return {"source": "live", "league": league,
            "columns": [{"key": c, "label": _STAND_LABEL.get(c, c.upper())} for c in ordered],
            "groups": groups}


def sample_standings(league: str) -> dict:
    if league in ("epl", "mls"):
        cols = [("points", "PTS"), ("wins", "W"), ("losses", "L")]
        rows = [("ARS", "Arsenal", ["48", "15", "3"]), ("MCI", "Man City", ["45", "14", "4"]),
                ("LIV", "Liverpool", ["43", "13", "4"]), ("CHE", "Chelsea", ["38", "11", "6"])]
    else:
        cols = [("wins", "W"), ("losses", "L"), ("winPercent", "PCT")]
        rows = [("BOS", "Celtics", ["48", "12", ".800"]), ("DEN", "Nuggets", ["42", "18", ".700"]),
                ("MIL", "Bucks", ["40", "20", ".667"]), ("LAL", "Lakers", ["33", "27", ".550"])]
    teams = [{"abbr": a, "name": n, "stats": dict(zip([c[0] for c in cols], vals))}
             for a, n, vals in rows]
    return {"source": "sample", "league": league,
            "columns": [{"key": k, "label": lb} for k, lb in cols],
            "groups": [{"name": "Standings", "teams": teams}]}


def sample_scores(league: str) -> dict:
    now = datetime.now(timezone.utc)
    demo = {
        "nfl": [("KC", "Chiefs", "24", "BUF", "Bills", "20", "in", "Q4 · 2:11"),
                ("DAL", "Cowboys", "0", "PHI", "Eagles", "0", "pre", "8:20 PM ET")],
        "nba": [("BOS", "Celtics", "112", "LAL", "Lakers", "108", "post", "Final"),
                ("GSW", "Warriors", "51", "DEN", "Nuggets", "48", "in", "Q3 · 5:40")],
        "epl": [("ARS", "Arsenal", "2", "MCI", "Man City", "1", "in", "72'"),
                ("LIV", "Liverpool", "0", "CHE", "Chelsea", "0", "pre", "Sat 12:30")],
        "urc": [("BUL", "Bulls", "27", "STO", "Stormers", "24", "post", "FT"),
                ("SHA", "Sharks", "0", "LEI", "Leinster", "0", "pre", "Sat 17:00"),
                ("LIO", "Lions", "15", "MUN", "Munster", "18", "in", "58'")],
        "rugbyc": [("RSA", "South Africa", "35", "NZL", "New Zealand", "20", "post", "FT"),
                   ("AUS", "Australia", "0", "ARG", "Argentina", "0", "pre", "Sat 15:00")],
        "cricket": [("RSA", "South Africa", "287/6", "IND", "India", "—", "in", "Day 2 · Innings 1"),
                    ("SA20", "Sunrisers EC", "0", "MICT", "MI Cape Town", "0", "pre", "18:00")],
        "laliga": [("RMA", "Real Madrid", "2", "BAR", "Barcelona", "2", "in", "68'"),
                   ("ATM", "Atlético", "0", "SEV", "Sevilla", "0", "pre", "Sun 21:00")],
        "seriea": [("INT", "Inter", "1", "JUV", "Juventus", "0", "post", "FT"),
                   ("MIL", "AC Milan", "0", "NAP", "Napoli", "0", "pre", "Sat 20:45")],
        "bundesliga": [("BAY", "Bayern", "3", "BVB", "Dortmund", "1", "in", "77'"),
                       ("RBL", "RB Leipzig", "0", "B04", "Leverkusen", "0", "pre", "Sat 18:30")],
        "ligue1": [("PSG", "Paris SG", "2", "OM", "Marseille", "1", "post", "FT"),
                   ("MON", "Monaco", "0", "LYO", "Lyon", "0", "pre", "Sun 20:45")],
        "ucl": [("MCI", "Man City", "2", "RMA", "Real Madrid", "2", "in", "72'"),
                ("ARS", "Arsenal", "0", "BAY", "Bayern", "0", "pre", "Wed 22:00")],
        "psl": [("SUN", "Mamelodi Sundowns", "2", "KAI", "Kaizer Chiefs", "0", "post", "FT"),
                ("ORL", "Orlando Pirates", "0", "STE", "Stellenbosch", "0", "pre", "Sat 15:30")],
        "mma": [("MAK", "Makhachev", None, "TSA", "Tsarukyan", None, "pre", "Main Event · Sat"),
                ("ADE", "Adesanya", None, "DDP", "du Plessis", None, "pre", "Title · Sat")],
        "atp": [("ALC", "Alcaraz", "2", "SIN", "Sinner", "1", "in", "Set 4"),
                ("DJO", "Djokovic", None, "ZVE", "Zverev", None, "pre", "QF · Tomorrow")],
        "wta": [("SWI", "Świątek", "2", "SAB", "Sabalenka", "0", "post", "Final"),
                ("GAU", "Gauff", None, "RYB", "Rybakina", None, "pre", "SF · Tomorrow")],
    }
    rows = demo.get(league, demo["nba"])
    games = []
    for i, (ha, hn, hs, aa, an, as_, state, status) in enumerate(rows):
        games.append({
            "id": f"{league}-{i}", "state": state, "status": status,
            "clock": None, "period": None,
            "start": (now + timedelta(hours=i + 1)).isoformat(),
            "home": {"abbr": ha, "name": hn, "score": hs if state != "pre" else None, "home": True,
                     "winner": state == "post" and hs > as_},
            "away": {"abbr": aa, "name": an, "score": as_ if state != "pre" else None, "home": False,
                     "winner": state == "post" and as_ > hs},
        })
    return {"source": "sample", "league": league, "games": games}


# ---------------------------------------------------------------------------
# Motorsport — ESPN racing scoreboards (F1, MotoGP, NASCAR, IndyCar). Each
# event is a race weekend; we surface name, circuit, status and top finishers.
# ---------------------------------------------------------------------------
RACING_TTL = 5 * 60
RACING_SERIES = {
    "f1": ("racing", "f1", "Formula 1"),
    "motogp": ("racing", "motogp", "MotoGP"),
    "nascar": ("racing", "nascar-premier", "NASCAR Cup"),
    "indycar": ("racing", "irl", "IndyCar"),
}


def live_racing(series: str) -> dict:
    sport, lg, label = RACING_SERIES[series]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard"
    raw = json.loads(fetch_url(url))
    races = []
    for event in raw.get("events", []):
        try:
            comp = (event.get("competitions") or [{}])[0]
            status = (event.get("status") or {}).get("type") or {}
            competitors = comp.get("competitors", [])
            ordered = sorted(competitors, key=lambda c: c.get("order") or 999)
            top = []
            for c in ordered[:3]:
                ath = c.get("athlete") or {}
                top.append(ath.get("displayName") or ath.get("shortName")
                           or (c.get("team") or {}).get("displayName") or "—")
            circuit = ((comp.get("venue") or {}).get("fullName")
                       or (event.get("circuit") or {}).get("fullName") or "")
            races.append({
                "id": event.get("id"),
                "name": event.get("shortName") or event.get("name") or label,
                "circuit": circuit,
                "state": status.get("state", "pre"),
                "status": status.get("shortDetail") or status.get("detail") or "",
                "start": event.get("date"),
                "winner": top[0] if (status.get("state") == "post" and top) else None,
                "top": top,
            })
        except Exception:
            continue
    return {"source": "live", "series": series, "label": label, "races": races}


def sample_racing(series: str) -> dict:
    now = datetime.now(timezone.utc)
    label = RACING_SERIES.get(series, ("", "", "Motorsport"))[2]
    demo = {
        "f1": [("Kyalami GP", "Kyalami Circuit, Johannesburg", "post",
                ["M. Verstappen", "L. Norris", "C. Leclerc"]),
               ("Monaco GP", "Circuit de Monaco", "pre", [])],
        "motogp": [("Grand Prix of South Africa", "Phakisa Freeway", "post",
                    ["F. Bagnaia", "J. Martín", "M. Márquez"]),
                   ("Qatar GP", "Lusail Circuit", "pre", [])],
        "nascar": [("Cup Series 400", "Daytona International", "post",
                    ["K. Larson", "D. Hamlin", "C. Bell"]),
                   ("Next Race", "Talladega Superspeedway", "pre", [])],
        "indycar": [("Grand Prix", "Streets of Long Beach", "post",
                     ["A. Palou", "S. Dixon", "W. Power"]),
                    ("Next Race", "Indianapolis Motor Speedway", "pre", [])],
    }
    rows = demo.get(series, demo["f1"])
    races = []
    for i, (name, circuit, state, top) in enumerate(rows):
        races.append({
            "id": f"{series}-{i}", "name": name, "circuit": circuit,
            "state": state,
            "status": "Finished" if state == "post" else "Upcoming",
            "start": (now + timedelta(days=i * 7)).isoformat(),
            "winner": top[0] if (state == "post" and top) else None,
            "top": top,
        })
    return {"source": "sample", "series": series, "label": label, "races": races}


# ---------------------------------------------------------------------------
# Team schedule ("My Teams" — recent + upcoming fixtures) and team news
# ---------------------------------------------------------------------------
TEAM_SCHEDULE_TTL = 30 * 60
TEAM_NEWS_TTL = 30 * 60


def live_team_schedule(league: str, team: str) -> dict:
    sport, lg = SPORT_LEAGUES[league]
    url = (f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}"
           f"/teams/{urllib.parse.quote(team)}/schedule")
    raw = json.loads(fetch_url(url))
    games = []
    for event in raw.get("events", []):
        try:
            comp = (event.get("competitions") or [{}])[0]
            status = (event.get("status") or {}).get("type") or {}
            competitors = [_norm_competitor(c) for c in comp.get("competitors", [])]
            home = next((c for c in competitors if c["home"]), None)
            away = next((c for c in competitors if not c["home"]), None)
            if not home or not away:
                continue
            games.append({
                "id": event.get("id"),
                "state": status.get("state", "pre"),
                "status": status.get("shortDetail") or status.get("detail") or "",
                "start": event.get("date"),
                "home": home, "away": away,
            })
        except Exception:
            continue
    games.sort(key=lambda g: g.get("start") or "")
    return {"source": "live", "league": league, "team": team.upper(), "games": games}


def sample_team_schedule(league: str, team: str) -> dict:
    now = datetime.now(timezone.utc)
    opp = {"epl": ("TOT", "Spurs"), "mls": ("LAFC", "LAFC")}.get(league, ("LAL", "Lakers"))
    games = []
    for i, offset in enumerate((-6, -2, 3, 8)):
        played = offset < 0
        start = now + timedelta(days=offset)
        home = i % 2 == 0
        us = {"abbr": team.upper(), "name": team.upper(),
              "score": str(90 + i) if played else None, "home": home,
              "winner": played and i % 2 == 0}
        them = {"abbr": opp[0], "name": opp[1],
                "score": str(85 + i) if played else None, "home": not home,
                "winner": played and i % 2 != 0}
        games.append({
            "id": f"{league}-{team}-{i}",
            "state": "post" if played else "pre",
            "status": "Final" if played else start.strftime("%a %d %b, %H:%M"),
            "start": start.isoformat(),
            "home": us if home else them,
            "away": them if home else us,
        })
    return {"source": "sample", "league": league, "team": team.upper(), "games": games}


def live_team_news(team: str) -> dict:
    url = ("https://news.google.com/rss/search?q="
           f"{urllib.parse.quote(team + ' ' + 'team')}&hl=en-US&gl=US&ceid=US:en")
    items = merge_items(parse_feed(fetch_url(url), "Google News"), 8)
    if not items:
        raise RuntimeError("no team news")
    return {"source": "live", "team": team, "items": items}


def sample_team_news(team: str) -> dict:
    now = datetime.now(timezone.utc)
    items = [
        {"title": f"{team} secure hard-fought win in weekend clash",
         "url": "https://example.com/team-1", "source": "Sample Sports",
         "summary": "", "published": (now - timedelta(hours=3)).isoformat()},
        {"title": f"Injury update: {team} coach optimistic ahead of next fixture",
         "url": "https://example.com/team-2", "source": "Sample Sports",
         "summary": "", "published": (now - timedelta(hours=9)).isoformat()},
    ]
    return {"source": "sample", "team": team, "items": items}


# ---------------------------------------------------------------------------
# AI Radar — AI / Claude news via Google News RSS (no key)
# ---------------------------------------------------------------------------
AI_NEWS_TTL = 30 * 60
AI_NEWS_QUERIES = {
    "claude": "Anthropic OR \"Claude AI\"",
    "llm": "large language model OR LLM AI",
    "agents": "AI agents OR agentic AI OR \"AI coding\"",
    "oss": "open source AI model",
}


def live_ai_news(topic: str) -> dict:
    query = AI_NEWS_QUERIES.get(topic, AI_NEWS_QUERIES["claude"])
    url = ("https://news.google.com/rss/search?q="
           f"{urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en")
    items = merge_items(parse_feed(fetch_url(url), "Google News"), 15)
    if not items:
        raise RuntimeError("no ai news")
    return {"source": "live", "topic": topic, "items": items}


def sample_ai_news(topic: str) -> dict:
    now = datetime.now(timezone.utc)
    demo = {
        "claude": ["Anthropic ships new Claude capabilities for developers",
                   "Claude Code adds workflow features teams are adopting",
                   "Enterprises expand Claude deployments across workflows"],
        "llm": ["New open-weight LLM posts strong reasoning benchmarks",
                "Study probes long-context retrieval in large models",
                "Inference costs fall as model efficiency improves"],
        "agents": ["Agentic coding tools reshape developer workflows",
                   "Best practices emerge for reliable tool-using agents",
                   "MCP connectors proliferate across the ecosystem"],
        "oss": ["Popular open-source AI project crosses a star milestone",
                "Community releases fine-tuned models for local use",
                "New dataset released under a permissive licence"],
    }.get(topic, [])
    return {"source": "sample", "topic": topic, "items": [
        {"title": t, "url": "https://news.google.com/", "source": "Sample Wire",
         "summary": "", "published": (now - timedelta(hours=2 * i + 1)).isoformat()}
        for i, t in enumerate(demo)]}


# ---------------------------------------------------------------------------
# Stocks / indices / FX — Stooq CSV (no key)
# ---------------------------------------------------------------------------
import csv as _csv

STOCKS_TTL = 90
STOCKS_HIST_TTL = 30 * 60
STOCK_NAMES = {
    "^spx": "S&P 500", "^ndq": "Nasdaq 100", "^dji": "Dow Jones",
    "aapl.us": "Apple", "msft.us": "Microsoft", "nvda.us": "NVIDIA",
    "tsla.us": "Tesla", "amzn.us": "Amazon", "googl.us": "Alphabet",
    "eurusd": "EUR/USD", "gbpusd": "GBP/USD", "usdjpy": "USD/JPY",
}


def _num(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def live_stocks(symbols: list[str]) -> dict:
    joined = ",".join(symbols)
    raw = fetch_url(f"https://stooq.com/q/l/?s={joined}&f=sd2t2ohlcv&h&e=csv").decode("utf-8", "replace")
    rows = list(_csv.DictReader(io.StringIO(raw)))
    assets = []
    for r in rows:
        o = _num(r.get("Open"))
        c = _num(r.get("Close"))
        if c is None:
            continue  # Stooq returns N/D off-hours / for bad symbols
        sym = (r.get("Symbol") or "").lower()
        change = (c - o) if (o is not None) else None
        assets.append({
            "symbol": sym.upper().replace(".US", ""),
            "id": sym,
            "name": STOCK_NAMES.get(sym, sym.upper()),
            "price": c,
            "change": change,
            "changePct": (change / o * 100) if (o and change is not None) else None,
        })
    if not assets:
        raise RuntimeError("no stock rows")
    return {"source": "live", "assets": assets}


def live_stock_history(symbol: str) -> dict:
    raw = fetch_url(f"https://stooq.com/q/d/l/?s={symbol}&i=d").decode("utf-8", "replace")
    rows = list(_csv.DictReader(io.StringIO(raw)))[-90:]
    candles = []
    for r in rows:
        o, hi, lo, c = _num(r.get("Open")), _num(r.get("High")), _num(r.get("Low")), _num(r.get("Close"))
        if None in (o, hi, lo, c):
            continue
        candles.append({"t": r.get("Date"), "o": o, "h": hi, "l": lo, "c": c})
    if len(candles) < 2:
        raise RuntimeError("no history")
    closes = [c["c"] for c in candles]
    return {"source": "live", "symbol": symbol, "candles": candles,
            "overlays": {"sma20": indicators.sma(closes, 20), "sma50": indicators.sma(closes, 50)},
            "signals": indicators.read_signals(closes)}


# Commodities, metals & rates — Stooq symbols → grouped quotes (no key).
COMMODITIES_TTL = 5 * 60
COMMODITY_SYMBOLS = {
    "xauusd": ("Gold", "Metals", "$/oz"),
    "xagusd": ("Silver", "Metals", "$/oz"),
    "hg.f": ("Copper", "Metals", "$/lb"),
    "xptusd": ("Platinum", "Metals", "$/oz"),
    "cl.f": ("WTI Crude", "Energy", "$/bbl"),
    "cb.f": ("Brent Crude", "Energy", "$/bbl"),
    "ng.f": ("Natural Gas", "Energy", "$/MMBtu"),
    "10usy.b": ("US 10Y Yield", "Rates", "%"),
    "2usy.b": ("US 2Y Yield", "Rates", "%"),
}
_COMMODITY_SAMPLE = {
    "xauusd": 2380.5, "xagusd": 29.8, "hg.f": 4.42, "xptusd": 995.0,
    "cl.f": 78.9, "cb.f": 83.2, "ng.f": 2.31, "10usy.b": 4.28, "2usy.b": 4.72,
}


def _commodity_assets(price_of, source):
    out = []
    for sym, (name, group, unit) in COMMODITY_SYMBOLS.items():
        price, change, pct = price_of(sym)
        if price is None:
            continue
        out.append({"symbol": name, "id": sym, "group": group, "unit": unit,
                    "price": price, "change": change, "changePct": pct})
    if not out:
        raise RuntimeError("no commodity rows")
    return {"source": source, "assets": out}


def live_commodities() -> dict:
    joined = ",".join(COMMODITY_SYMBOLS)
    raw = fetch_url(f"https://stooq.com/q/l/?s={joined}&f=sd2t2ohlcv&h&e=csv").decode("utf-8", "replace")
    quotes = {}
    for r in _csv.DictReader(io.StringIO(raw)):
        o, c = _num(r.get("Open")), _num(r.get("Close"))
        sym = (r.get("Symbol") or "").lower()
        if c is None:
            continue
        change = (c - o) if o is not None else None
        quotes[sym] = (c, change, (change / o * 100) if (o and change is not None) else None)
    return _commodity_assets(lambda s: quotes.get(s, (None, None, None)), "live")


def sample_commodities() -> dict:
    import random
    def price_of(sym):
        base = _COMMODITY_SAMPLE[sym]
        rng = random.Random(sym)
        pct = rng.uniform(-1.8, 1.8)
        return round(base, 2), round(base * pct / 100, 2), round(pct, 2)
    return _commodity_assets(price_of, "sample")


def sample_stocks(symbols: list[str]) -> dict:
    import random
    base = {"^spx": 5600, "^ndq": 19800, "^dji": 41200, "aapl.us": 211.9,
            "msft.us": 448.3, "nvda.us": 128.4, "eurusd": 1.087, "gbpusd": 1.271}
    assets = []
    for sym in symbols:
        rng = random.Random(sym)
        price = base.get(sym, 100.0)
        chg = rng.uniform(-2.5, 2.5)
        assets.append({
            "symbol": sym.upper().replace(".US", ""), "id": sym,
            "name": STOCK_NAMES.get(sym, sym.upper()),
            "price": round(price, 4 if price < 10 else 2),
            "change": round(price * chg / 100, 2), "changePct": round(chg, 2),
        })
    return {"source": "sample", "assets": assets}


def sample_stock_history(symbol: str) -> dict:
    base = {"^spx": 5600, "^ndq": 19800, "aapl.us": 211.9, "eurusd": 1.087}.get(symbol, 100.0)
    candles = _synth_candles(symbol, 90, base)
    for c in candles:  # stocks use date strings, not ms timestamps
        c["t"] = None
    closes = [c["c"] for c in candles]
    return {"source": "sample", "symbol": symbol, "candles": candles,
            "overlays": {"sma20": indicators.sma(closes, 20), "sma50": indicators.sma(closes, 50)},
            "signals": indicators.read_signals(closes)}


# ---------------------------------------------------------------------------
# Gaming data — Epic free games + Steam specials (both no-key)
# ---------------------------------------------------------------------------
GAMING_TTL = 30 * 60


def _epic_image(elem: dict) -> str:
    for img in elem.get("keyImages", []) or []:
        if img.get("type") in ("OfferImageWide", "DieselStoreFrontWide", "Thumbnail"):
            return img.get("url", "")
    imgs = elem.get("keyImages") or []
    return imgs[0].get("url", "") if imgs else ""


def live_free_games() -> dict:
    url = ("https://store-site-backend-static.ak.epicgames.com/freeGamesPromotions"
           "?locale=en-US&country=US&allowCountries=US")
    raw = json.loads(fetch_url(url))
    elements = (((raw.get("data") or {}).get("Catalog") or {}).get("searchStore") or {}).get("elements", [])
    current, upcoming = [], []
    for elem in elements:
        try:
            promos = elem.get("promotions") or {}
            slug = elem.get("productSlug") or elem.get("urlSlug") or ""
            base = {"title": elem.get("title"), "image": _epic_image(elem),
                    "url": f"https://store.epicgames.com/en-US/p/{slug}" if slug else "https://store.epicgames.com/en-US/free-games"}
            now_offers = promos.get("promotionalOffers") or []
            up_offers = promos.get("upcomingPromotionalOffers") or []
            if now_offers:
                offer = (now_offers[0].get("promotionalOffers") or [{}])[0]
                current.append({**base, "end": offer.get("endDate")})
            elif up_offers:
                offer = (up_offers[0].get("promotionalOffers") or [{}])[0]
                upcoming.append({**base, "start": offer.get("startDate")})
        except Exception:
            continue
    return {"source": "live", "current": current, "upcoming": upcoming}


def live_steam_deals() -> dict:
    raw = json.loads(fetch_url("https://store.steampowered.com/api/featuredcategories/?cc=us&l=en"))
    specials = ((raw.get("specials") or {}).get("items")) or []
    deals = []
    for it in specials[:12]:
        try:
            if not it.get("discounted"):
                continue
            deals.append({
                "id": it.get("id"), "name": it.get("name"),
                "image": it.get("header_image") or it.get("large_capsule_image"),
                "discount": it.get("discount_percent"),
                "price": round((it.get("final_price") or 0) / 100, 2),
                "url": f"https://store.steampowered.com/app/{it.get('id')}",
            })
        except Exception:
            continue
    return {"source": "live", "deals": deals}


def sample_free_games() -> dict:
    now = datetime.now(timezone.utc)
    return {"source": "sample",
            "current": [
                {"title": "Neon Drifter", "image": "", "url": "https://store.epicgames.com/en-US/free-games",
                 "end": (now + timedelta(days=4)).isoformat()},
                {"title": "Ember & Ash", "image": "", "url": "https://store.epicgames.com/en-US/free-games",
                 "end": (now + timedelta(days=4)).isoformat()}],
            "upcoming": [
                {"title": "Starforge Tactics", "image": "", "url": "https://store.epicgames.com/en-US/free-games",
                 "start": (now + timedelta(days=4)).isoformat()}]}


def sample_steam_deals() -> dict:
    demo = [("Deep Rock Galactic", 67, 9.89), ("Hades II", 20, 23.99),
            ("Baldur's Gate 3", 20, 47.99), ("Vampire Survivors", 40, 2.99),
            ("Stardew Valley", 25, 10.49), ("Hollow Knight", 50, 7.49)]
    return {"source": "sample",
            "deals": [{"id": 1000 + i, "name": n, "image": "", "discount": d, "price": p,
                       "url": f"https://store.steampowered.com/app/{1000 + i}"}
                      for i, (n, d, p) in enumerate(demo)]}


# ---------------------------------------------------------------------------
# Medicine — PubMed (NCBI E-utilities) + ClinicalTrials.gov (both no-key)
# ---------------------------------------------------------------------------
PUBMED_TTL = 30 * 60
TRIALS_TTL = 30 * 60


def live_pubmed(query: str) -> dict:
    esearch = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
               f"?db=pubmed&retmode=json&retmax=15&sort=date&term={urllib.parse.quote(query)}")
    ids = json.loads(fetch_url(esearch)).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"source": "live", "query": query, "articles": []}
    return {"source": "live", "query": query, "articles": live_pubmed_ids(ids)}


def pubmed_grounding(query: str) -> dict:
    """Recent PubMed articles + their abstracts, for grounding a consult.

    Returns {articles: [...], text: "<abstracts>"} — the article list drives the
    UI's Sources panel; the abstract text is injected into the model context.
    """
    esearch = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
               f"?db=pubmed&retmode=json&retmax=5&sort=relevance&term={urllib.parse.quote(query)}")
    ids = json.loads(fetch_url(esearch)).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"articles": [], "text": ""}
    summary = live_pubmed_ids(ids)
    efetch = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
              f"?db=pubmed&rettype=abstract&retmode=text&id={','.join(ids)}")
    try:
        text = fetch_url(efetch).decode("utf-8", "replace")[:6000]
    except Exception:
        text = ""
    return {"articles": summary, "text": text}


def live_pubmed_ids(ids: list[str]) -> list[dict]:
    esummary = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&retmode=json&id={','.join(ids)}")
    result = json.loads(fetch_url(esummary)).get("result", {})
    articles = []
    for uid in result.get("uids", ids):
        doc = result.get(uid, {})
        authors = [a.get("name", "") for a in doc.get("authors", [])][:3]
        articles.append({
            "pmid": uid,
            "title": strip_html(doc.get("title", ""), 220),
            "journal": doc.get("source", ""),
            "date": doc.get("pubdate", ""),
            "authors": ", ".join(authors) + (" et al." if len(doc.get("authors", [])) > 3 else ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
        })
    return articles


def sample_pubmed(query: str) -> dict:
    demo = [
        ("Outcomes of early ART initiation in TB-HIV co-infection: a multicentre cohort",
         "Lancet HIV", "2026 Jul", "Naidoo K, Abdool Karim S"),
        ("Point-of-care ultrasound in rural emergency care: a pragmatic trial",
         "S Afr Med J", "2026 Jun", "van der Merwe J, Dlamini T"),
        ("Novel GLP-1 agonists and cardiovascular outcomes: an updated meta-analysis",
         "N Engl J Med", "2026 Jun", "Smith R, Patel A"),
    ]
    return {"source": "sample", "query": query, "articles": [
        {"pmid": f"400000{i}", "title": t, "journal": j, "date": d, "authors": a + " et al.",
         "url": "https://pubmed.ncbi.nlm.nih.gov/"}
        for i, (t, j, d, a) in enumerate(demo)]}


def live_trials(query: str) -> dict:
    url = ("https://clinicaltrials.gov/api/v2/studies"
           f"?query.term={urllib.parse.quote(query)}"
           "&sort=LastUpdatePostDate:desc&pageSize=15&format=json")
    raw = json.loads(fetch_url(url))
    trials = []
    for study in raw.get("studies", []):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        conds = proto.get("conditionsModule", {}).get("conditions", [])
        nct = ident.get("nctId", "")
        trials.append({
            "nct": nct,
            "title": strip_html(ident.get("briefTitle", ""), 200),
            "status": status.get("overallStatus", ""),
            "conditions": ", ".join(conds[:3]),
            "updated": status.get("lastUpdatePostDateStruct", {}).get("date", ""),
            "url": f"https://clinicaltrials.gov/study/{nct}" if nct else "https://clinicaltrials.gov/",
        })
    return {"source": "live", "query": query, "trials": trials}


def sample_trials(query: str) -> dict:
    demo = [
        ("Short-course regimen for drug-resistant TB in high-burden settings", "RECRUITING",
         "Tuberculosis, Drug-Resistant", "2026-07-10"),
        ("Community health worker-led hypertension control in primary care", "ACTIVE_NOT_RECRUITING",
         "Hypertension", "2026-07-02"),
        ("Long-acting injectable PrEP implementation study", "RECRUITING",
         "HIV Prevention", "2026-06-28"),
    ]
    return {"source": "sample", "query": query, "trials": [
        {"nct": f"NCT0{9000000 + i}", "title": t, "status": s, "conditions": c, "updated": u,
         "url": "https://clinicaltrials.gov/"}
        for i, (t, s, c, u) in enumerate(demo)]}


# ---------------------------------------------------------------------------
# Drug reference — openFDA drug labelling (no key). Official label text only;
# reference, not clinical advice.
# ---------------------------------------------------------------------------
DRUG_TTL = 12 * 60 * 60
# Label fields we surface, in clinical reading order (each is an array of text).
_DRUG_SECTIONS = [
    ("boxed_warning", "Boxed warning"),
    ("indications_and_usage", "Indications"),
    ("dosage_and_administration", "Dosage"),
    ("contraindications", "Contraindications"),
    ("warnings_and_cautions", "Warnings"),
    ("warnings", "Warnings"),
    ("drug_interactions", "Interactions"),
    ("adverse_reactions", "Adverse reactions"),
    ("mechanism_of_action", "Mechanism"),
    ("pregnancy", "Pregnancy"),
]


def _drug_record(result: dict) -> dict:
    openfda = result.get("openfda") or {}

    def first(key):
        v = openfda.get(key)
        return v[0] if isinstance(v, list) and v else ""

    sections = []
    seen_labels = set()
    for field, label in _DRUG_SECTIONS:
        if label in seen_labels:
            continue
        val = result.get(field)
        if isinstance(val, list) and val:
            text = strip_html(" ".join(val), 1500)
            if text:
                sections.append({"label": label, "text": text})
                seen_labels.add(label)
    return {
        "brand": first("brand_name"),
        "generic": first("generic_name"),
        "manufacturer": first("manufacturer_name"),
        "route": first("route"),
        "sections": sections,
    }


def live_drug(query: str) -> dict:
    url = ("https://api.fda.gov/drug/label.json"
           f"?search={urllib.parse.quote(query)}&limit=1")
    raw = json.loads(fetch_url(url, timeout=8))
    results = raw.get("results") or []
    if not results:
        return {"source": "live", "query": query, "drug": None}
    return {"source": "live", "query": query, "drug": _drug_record(results[0])}


def sample_drug(query: str) -> dict:
    return {
        "source": "sample", "query": query,
        "drug": {
            "brand": "Glucophage", "generic": "Metformin hydrochloride",
            "manufacturer": "Sample Pharma", "route": "ORAL",
            "sections": [
                {"label": "Indications", "text": "Adjunct to diet and exercise to improve glycemic control in adults and children with type 2 diabetes mellitus. (Sample label text — run online for live openFDA data.)"},
                {"label": "Dosage", "text": "Start 500 mg orally twice daily or 850 mg once daily with meals; titrate gradually. Maximum 2550 mg/day in divided doses."},
                {"label": "Contraindications", "text": "Severe renal impairment (eGFR below 30 mL/min/1.73m²); acute or chronic metabolic acidosis, including diabetic ketoacidosis; known hypersensitivity."},
                {"label": "Warnings", "text": "Lactic acidosis is a rare but serious metabolic complication. Risk factors include renal impairment, sepsis, dehydration, excess alcohol intake and hepatic impairment."},
                {"label": "Adverse reactions", "text": "Diarrhoea, nausea, vomiting, flatulence, abdominal discomfort and metallic taste are common, especially at initiation."},
            ],
        },
    }


# ---------------------------------------------------------------------------
# AI Lab data — trending GitHub repos + arXiv papers (free, no key)
# ---------------------------------------------------------------------------
REPOS_TTL = 60 * 60
PAPERS_TTL = 3 * 60 * 60


def live_repos(window: str) -> dict:
    # Repositories created in a recent window, ranked by stars — a good proxy
    # for "trending". GitHub search API needs no key at low volume.
    days = {"day": 1, "week": 7, "month": 30}.get(window, 7)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    q = urllib.parse.urlencode({"q": f"created:>{since}", "sort": "stars", "order": "desc", "per_page": 20})
    raw = json.loads(fetch_url(f"https://api.github.com/search/repositories?{q}", timeout=9))
    repos = []
    for r in raw.get("items", [])[:20]:
        repos.append({
            "name": r.get("full_name", ""),
            "desc": strip_html(r.get("description") or "", 160),
            "stars": r.get("stargazers_count", 0),
            "language": r.get("language") or "",
            "url": r.get("html_url", ""),
            "topics": (r.get("topics") or [])[:4],
        })
    if not repos:
        raise RuntimeError("no repos")
    return {"source": "live", "window": window, "repos": repos}


def sample_repos(window: str) -> dict:
    demo = [
        ("anthropics/claude-code", "Agentic coding tool in your terminal", 42000, "TypeScript", ["ai", "agents", "cli"]),
        ("modelcontextprotocol/servers", "Reference MCP servers", 18500, "Python", ["mcp", "tools"]),
        ("anthropics/anthropic-cookbook", "Recipes for building with Claude", 12300, "Jupyter Notebook", ["claude", "examples"]),
        ("anthropics/skills", "Example Agent Skills", 5400, "Python", ["skills", "agents"]),
        ("openai/openai-cookbook", "Examples and guides for building with LLMs", 61000, "MDX", ["llm"]),
    ]
    return {"source": "sample", "window": window, "repos": [
        {"name": n, "desc": d, "stars": s, "language": lang, "url": f"https://github.com/{n}", "topics": t}
        for n, d, s, lang, t in demo]}


def live_papers(category: str) -> dict:
    cat = category if category in ("cs.AI", "cs.CL", "cs.LG") else "cs.AI"
    url = ("http://export.arxiv.org/api/query?"
           f"search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&max_results=15")
    raw = fetch_url(url, timeout=9).decode("utf-8", "replace")
    papers = []
    for entry in re.findall(r"<entry>(.*?)</entry>", raw, re.S)[:15]:
        def field(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", entry, re.S)
            return strip_html(m.group(1), 2000).strip() if m else ""
        link = re.search(r'<link[^>]*rel="alternate"[^>]*href="([^"]+)"', entry) \
            or re.search(r"<id>(.*?)</id>", entry)
        authors = re.findall(r"<name>(.*?)</name>", entry)
        papers.append({
            "title": field("title"),
            "summary": field("summary")[:400],
            "authors": ", ".join(a.strip() for a in authors[:4]) + (" et al." if len(authors) > 4 else ""),
            "published": field("published")[:10],
            "url": (link.group(1).strip() if link else "https://arxiv.org/"),
        })
    if not papers:
        raise RuntimeError("no papers")
    return {"source": "live", "category": cat, "papers": papers}


def sample_papers(category: str) -> dict:
    now = datetime.now(timezone.utc)
    demo = [
        ("Efficient Tool Use in Long-Horizon Agents", "We study how agents plan and call tools over extended tasks…", "A. Researcher, B. Scientist et al."),
        ("Retrieval-Augmented Reasoning for Clinical QA", "A retrieval approach improving factuality on medical questions…", "C. Author, D. Coauthor"),
        ("Scaling Laws for Instruction-Tuned Models", "New empirical scaling relationships under instruction tuning…", "E. Lab, F. Group et al."),
    ]
    return {"source": "sample", "category": category or "cs.AI", "papers": [
        {"title": t, "summary": s, "authors": a, "published": (now - timedelta(days=i)).date().isoformat(),
         "url": "https://arxiv.org/"} for i, (t, s, a) in enumerate(demo)]}


# ---------------------------------------------------------------------------
# Podcasts — parse an RSS feed's audio enclosures into playable episodes
# ---------------------------------------------------------------------------
PODCAST_TTL = 30 * 60


def live_podcast(url: str) -> dict:
    root = ET.fromstring(fetch_url(url))
    channel = next((el for el in root.iter() if _localname(el.tag) == "channel"), root)
    show = _first_child_text(channel, "title") or "Podcast"
    episodes = []
    for node in root.iter():
        if _localname(node.tag) != "item":
            continue
        audio = ""
        duration = ""
        for child in node:
            ln = _localname(child.tag)
            if ln == "enclosure" and (child.get("type", "").startswith("audio")
                                      or child.get("url", "").endswith((".mp3", ".m4a", ".ogg"))):
                audio = child.get("url", "")
            elif ln == "duration" and child.text:
                duration = child.text.strip()
        if not audio:
            continue
        episodes.append({
            "title": strip_html(_first_child_text(node, "title"), 140),
            "audio": audio,
            "published": (parse_date(_first_child_text(node, "pubdate", "date")) or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
            "duration": duration[:12],
        })
        if len(episodes) >= 30:
            break
    if not episodes:
        raise RuntimeError("no audio episodes found")
    return {"source": "live", "show": show, "episodes": episodes}


def sample_podcast(url: str) -> dict:
    now = datetime.now(timezone.utc)
    demo = [("The state of open-source AI in 2026", "48:12", 1),
            ("Why SQLite is eating the database world", "39:40", 3),
            ("Designing for calm: interfaces that respect attention", "55:03", 6)]
    return {"source": "sample", "show": "Sample Cast",
            "episodes": [{"title": t, "audio": "", "duration": d,
                          "published": (now - timedelta(days=days)).isoformat()}
                         for t, d, days in demo]}


# ---------------------------------------------------------------------------
# Intel / utility feeds — earthquakes (USGS) and FX rates (Frankfurter/ECB)
# ---------------------------------------------------------------------------
QUAKES_TTL = 10 * 60
FX_TTL = 60 * 60
FX_DEFAULT = ["EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]
CONVERT_FIATS = ["EUR", "GBP", "JPY", "ZAR", "AUD", "CAD", "CHF", "CNY", "INR"]


def live_quakes() -> dict:
    raw = json.loads(fetch_url(
        "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"))
    quakes = []
    for f in raw.get("features", [])[:20]:
        p = f.get("properties") or {}
        if p.get("mag") is None:
            continue
        quakes.append({
            "mag": round(p["mag"], 1), "place": p.get("place") or "—",
            "time": p.get("time"), "url": p.get("url"),
            "tsunami": bool(p.get("tsunami")),
        })
    quakes.sort(key=lambda q: q["time"] or 0, reverse=True)
    return {"source": "live", "quakes": quakes}


def sample_quakes() -> dict:
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    demo = [(5.8, "120km SW of Tokyo, Japan", 0), (4.2, "20km E of Ridgecrest, CA", 1),
            (6.1, "Off the coast of Chile", 2), (3.7, "10km N of Reykjavík, Iceland", 3),
            (4.9, "Aegean Sea", 5)]
    return {"source": "sample", "quakes": [
        {"mag": m, "place": pl, "time": now - h * 3600000,
         "url": "https://earthquake.usgs.gov/", "tsunami": m >= 6}
        for m, pl, h in demo]}


# ---------------------------------------------------------------------------
# Space weather — NOAA SWPC (no key): planetary K-index + aurora outlook
# ---------------------------------------------------------------------------
SPACE_TTL = 30 * 60
# Kp → geomagnetic storm scale (NOAA G-scale) with a plain-language read.
_KP_BANDS = [
    (4, "Quiet", "up"), (4.99, "Unsettled", "neutral"), (5.99, "G1 minor storm", "warn"),
    (6.99, "G2 moderate storm", "warn"), (7.99, "G3 strong storm", "down"),
    (8.99, "G4 severe storm", "down"), (99, "G5 extreme storm", "down"),
]


def kp_band(kp) -> dict:
    if kp is None:
        return {"label": "—", "tone": "neutral"}
    for upper, label, tone in _KP_BANDS:
        if kp <= upper:
            return {"label": label, "tone": tone}
    return {"label": "G5 extreme storm", "tone": "down"}


def live_spaceweather() -> dict:
    raw = json.loads(fetch_url(
        "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"))
    # First row is the header: [time_tag, Kp, a_running, station_count].
    rows = raw[1:] if raw and raw[0] and raw[0][0] == "time_tag" else raw
    series = []
    for r in rows[-24:]:
        try:
            series.append({"t": r[0], "kp": float(r[1])})
        except (ValueError, IndexError, TypeError):
            continue
    if not series:
        raise RuntimeError("no k-index data")
    latest = series[-1]["kp"]
    peak = max(s["kp"] for s in series)
    aurora = ("Aurora likely at high latitudes" if peak >= 5
              else "Aurora possible at very high latitudes" if peak >= 4
              else "No significant aurora expected")
    return {"source": "live", "kp": latest, "band": kp_band(latest),
            "peak24h": peak, "aurora": aurora, "series": series}


def sample_spaceweather() -> dict:
    now = datetime.now(timezone.utc)
    series = []
    pattern = [2, 2.33, 3, 3.67, 4.33, 5, 4.67, 3.67]
    for i in range(8):
        t = (now - timedelta(hours=(8 - i) * 3)).strftime("%Y-%m-%d %H:%M:%S")
        series.append({"t": t, "kp": pattern[i]})
    latest = series[-1]["kp"]
    peak = max(s["kp"] for s in series)
    return {"source": "sample", "kp": latest, "band": kp_band(latest),
            "peak24h": peak, "aurora": "Aurora likely at high latitudes",
            "series": series}


# ---------------------------------------------------------------------------
# Flights overhead — OpenSky Network (no key, anonymous; rate-limited)
# ---------------------------------------------------------------------------
FLIGHTS_TTL = 60  # positions move fast; keep it short but gentle on the API
_FLIGHTS_BOX = 0.9  # ± degrees around the point (~100 km lat)


def _flight_track_dir(track) -> str:
    if track is None:
        return ""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int((track % 360) / 45 + 0.5) % 8]


def live_flights(lat: float, lon: float, name: str | None) -> dict:
    q = urllib.parse.urlencode({
        "lamin": f"{lat - _FLIGHTS_BOX:.4f}", "lamax": f"{lat + _FLIGHTS_BOX:.4f}",
        "lomin": f"{lon - _FLIGHTS_BOX:.4f}", "lomax": f"{lon + _FLIGHTS_BOX:.4f}"})
    raw = json.loads(fetch_url(f"https://opensky-network.org/api/states/all?{q}", timeout=9))
    flights = []
    for s in (raw.get("states") or [])[:30]:
        try:
            callsign = (s[1] or "").strip()
            flights.append({
                "icao": s[0], "callsign": callsign or "—",
                "country": s[2] or "", "lon": s[5], "lat": s[6],
                "altitude": s[7], "onGround": bool(s[8]),
                "velocity": s[9], "heading": s[10], "dir": _flight_track_dir(s[10]),
                "verticalRate": s[11],
            })
        except (IndexError, TypeError):
            continue
    airborne = [f for f in flights if not f["onGround"] and f["altitude"] is not None]
    airborne.sort(key=lambda f: f["altitude"])
    return {"source": "live", "location": {"name": name or f"{lat:.2f}, {lon:.2f}"},
            "count": len(airborne), "flights": airborne[:20]}


def sample_flights(name: str | None = None) -> dict:
    demo = [
        ("UAL245", "United States", 10668, 251, 78, "NE", 0),
        ("BAW112", "United Kingdom", 11582, 244, 265, "W", 0),
        ("DAL480", "United States", 3352, 180, 190, "S", 12.4),
        ("AFR22", "France", 9144, 236, 95, "E", -6.1),
        ("ACA759", "Canada", 11887, 249, 340, "NW", 0),
    ]
    flights = [{"icao": f"a{i:05x}", "callsign": cs, "country": co, "lon": -74 + i * 0.1,
                "lat": 40.7 + i * 0.1, "altitude": alt, "onGround": False,
                "velocity": vel, "heading": hd, "dir": d, "verticalRate": vr}
               for i, (cs, co, alt, vel, hd, d, vr) in enumerate(demo)]
    return {"source": "sample", "location": {"name": name or "New York"},
            "count": len(flights), "flights": flights}


# ---------------------------------------------------------------------------
# Weather alerts — US National Weather Service (api.weather.gov, no key)
# ---------------------------------------------------------------------------
ALERTS_TTL = 10 * 60
# NWS severity → tone for the UI (Extreme/Severe → down, Moderate → warn…).
_ALERT_TONE = {"Extreme": "down", "Severe": "down", "Moderate": "warn",
               "Minor": "neutral", "Unknown": "neutral"}


def live_alerts(lat: float, lon: float, name: str | None) -> dict:
    url = f"https://api.weather.gov/alerts/active?point={lat:.4f},{lon:.4f}"
    raw = json.loads(fetch_url(url, timeout=8))
    alerts = []
    for f in raw.get("features", [])[:15]:
        p = f.get("properties") or {}
        sev = p.get("severity") or "Unknown"
        alerts.append({
            "event": p.get("event") or "Alert",
            "severity": sev,
            "tone": _ALERT_TONE.get(sev, "neutral"),
            "urgency": p.get("urgency") or "",
            "headline": p.get("headline") or "",
            "area": strip_html(p.get("areaDesc") or "", 120),
            "effective": p.get("effective"),
            "expires": p.get("expires"),
            "sender": p.get("senderName") or "NWS",
        })
    # Most severe first (Extreme → Minor), then by soonest expiry.
    rank = {"Extreme": 0, "Severe": 1, "Moderate": 2, "Minor": 3, "Unknown": 4}
    alerts.sort(key=lambda a: (rank.get(a["severity"], 4), a["expires"] or ""))
    return {"source": "live", "location": {"name": name or f"{lat:.2f}, {lon:.2f}"},
            "alerts": alerts}


def sample_alerts(name: str | None = None) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "source": "sample",
        "location": {"name": name or "New York"},
        "alerts": [
            {"event": "Heat Advisory", "severity": "Moderate", "tone": "warn",
             "urgency": "Expected", "headline": "Heat Advisory until 8 PM",
             "area": "New York (Manhattan); Bronx", "sender": "NWS New York",
             "effective": now.isoformat(),
             "expires": (now + timedelta(hours=6)).isoformat()},
            {"event": "Severe Thunderstorm Watch", "severity": "Severe", "tone": "down",
             "urgency": "Expected", "headline": "Severe Thunderstorm Watch this evening",
             "area": "Southern New York; Northeast New Jersey", "sender": "NWS",
             "effective": now.isoformat(),
             "expires": (now + timedelta(hours=4)).isoformat()},
        ],
    }


def live_fx(base: str, symbols: list[str]) -> dict:
    q = urllib.parse.urlencode({"from": base, "to": ",".join(symbols)})
    raw = json.loads(fetch_url(f"https://api.frankfurter.app/latest?{q}"))
    return {"source": "live", "base": raw.get("base", base),
            "date": raw.get("date"), "rates": raw.get("rates", {})}


def sample_fx(base: str, symbols: list[str]) -> dict:
    table = {"EUR": 0.92, "GBP": 0.79, "JPY": 157.2, "CAD": 1.36,
             "AUD": 1.51, "CHF": 0.89, "USD": 1.0, "CNY": 7.25, "INR": 83.4}
    b = table.get(base, 1.0)
    rates = {s: round(table.get(s, 1.0) / b, 4) for s in symbols if s != base}
    return {"source": "sample", "base": base,
            "date": datetime.now(timezone.utc).date().isoformat(), "rates": rates}


# ---------------------------------------------------------------------------
# Socials hub — read-only, no-account feeds (Hacker News, Lobsters, Reddit)
# ---------------------------------------------------------------------------
SOCIAL_TTL = 5 * 60


def _social_item(title, url, author, source, score, comments, meta=""):
    return {"title": strip_html(title or "", 160), "url": (url or "").strip(),
            "author": author or "", "source": source, "score": score,
            "comments": comments, "meta": meta}


def live_social_hn() -> dict:
    ids = json.loads(fetch_url("https://hacker-news.firebaseio.com/v0/topstories.json"))[:20]

    def one(item_id):
        return json.loads(fetch_url(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"))

    items = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for story in pool.map(one, ids):
            if not story or story.get("type") != "story":
                continue
            items.append(_social_item(
                story.get("title"),
                story.get("url") or f"https://news.ycombinator.com/item?id={story.get('id')}",
                story.get("by"), "Hacker News", story.get("score"), story.get("descendants", 0),
                "news.ycombinator.com"))
    if not items:
        raise RuntimeError("no HN stories")
    return {"source": "live", "network": "hn", "items": items}


def live_social_lobsters() -> dict:
    raw = json.loads(fetch_url("https://lobste.rs/hottest.json"))
    items = []
    for s in raw[:25]:
        items.append(_social_item(
            s.get("title"), s.get("url") or s.get("comments_url"),
            (s.get("submitter_user") or {}).get("username") if isinstance(s.get("submitter_user"), dict) else s.get("submitter_user"),
            "Lobsters", s.get("score"), s.get("comment_count", 0),
            " ".join(s.get("tags", [])[:3])))
    if not items:
        raise RuntimeError("no lobsters posts")
    return {"source": "live", "network": "lobsters", "items": items}


def live_social_reddit(sub: str) -> dict:
    raw = json.loads(fetch_url(f"https://www.reddit.com/r/{sub}/hot.json?limit=25&raw_json=1"))
    items = []
    for child in raw.get("data", {}).get("children", []):
        d = child.get("data", {})
        if d.get("stickied"):
            continue
        permalink = "https://www.reddit.com" + d.get("permalink", "")
        items.append(_social_item(
            d.get("title"), d.get("url_overridden_by_dest") or permalink,
            d.get("author"), "Reddit", d.get("score"), d.get("num_comments", 0),
            f"r/{d.get('subreddit', sub)}"))
    if not items:
        raise RuntimeError("no reddit posts")
    return {"source": "live", "network": "reddit", "items": items}


def sample_social(network: str, sub: str = "") -> dict:
    base = {
        "hn": [("Show HN: I built a dependency-free dashboard in a weekend", "ycombinator", 412, 137),
               ("The hidden cost of microservices nobody talks about", "dhh_fan", 288, 201),
               ("Ask HN: What are you self-hosting in 2026?", "homelabber", 176, 342)],
        "lobsters": [("A pure-Python implementation of P-256 for fun", "cryptonerd", 64, 28),
                     ("Why I moved my side project back to SQLite", "pragmatic", 51, 40),
                     ("Understanding the event loop, from scratch", "async_a", 47, 19)],
        "reddit": [("Finally finished my custom mechanical keyboard build", "kbd_lover", 5400, 213),
                   ("TIL a fascinating fact about deep-sea creatures", "ocean_facts", 12800, 640),
                   ("My homelab rack after two years of tinkering", "rackmount", 3100, 158)],
    }
    src = {"hn": "Hacker News", "lobsters": "Lobsters", "reddit": "Reddit"}[network]
    meta = {"hn": "news.ycombinator.com", "lobsters": "rust web", "reddit": f"r/{sub or 'popular'}"}[network]
    items = [_social_item(t, f"https://example.com/sample/{network}-{i}", a, src, s, c, meta)
             for i, (t, a, s, c) in enumerate(base[network])]
    return {"source": "sample", "network": network, "items": items}


# ---------------------------------------------------------------------------
# Live upstream calls (normalized to the same shapes as the samples)
# ---------------------------------------------------------------------------
def live_news(topic: str, limit: int, sources: list[dict]) -> dict:
    collected: list[dict] = []

    def fetch_one(source: dict) -> list[dict]:
        return parse_feed(fetch_url(source["url"]), source["name"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = [pool.submit(fetch_one, source) for source in sources]
        for future in concurrent.futures.as_completed(futures):
            try:
                collected.extend(future.result())
            except Exception:
                pass  # one dead feed must not sink the topic
    if not collected:
        raise RuntimeError(f"all {len(sources)} feeds failed for topic {topic!r}")
    return {"topic": topic, "source": "live", "items": merge_items(collected, limit)}


def live_weather(lat: float, lon: float, name: str | None) -> dict:
    query = urllib.parse.urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,sunrise,sunset",
            "forecast_days": "7",
            "timezone": "auto",
        }
    )
    raw = json.loads(fetch_url(f"https://api.open-meteo.com/v1/forecast?{query}"))
    current = raw["current"]
    hourly = raw["hourly"]
    daily = raw["daily"]

    # Trim hourly to the next 24 hours starting from "now" in the location's tz.
    now_local = datetime.fromisoformat(current["time"])
    start = 0
    for i, t in enumerate(hourly["time"]):
        if datetime.fromisoformat(t) >= now_local.replace(minute=0):
            start = i
            break
    hours = []
    for i in range(start, min(start + 24, len(hourly["time"]))):
        hours.append(
            {
                "time": hourly["time"][i],
                "temp": hourly["temperature_2m"][i],
                "precipProb": hourly["precipitation_probability"][i],
                "code": hourly["weather_code"][i],
            }
        )
    days = []
    for i in range(len(daily["time"])):
        days.append(
            {
                "date": daily["time"][i],
                "min": daily["temperature_2m_min"][i],
                "max": daily["temperature_2m_max"][i],
                "code": daily["weather_code"][i],
                "precipProb": daily["precipitation_probability_max"][i],
            }
        )
    aqi = None
    try:  # air quality is a separate no-key API; missing data must not sink weather
        aq_query = urllib.parse.urlencode(
            {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "current": "us_aqi"})
        aq = json.loads(fetch_url(
            f"https://air-quality-api.open-meteo.com/v1/air-quality?{aq_query}", timeout=6))
        aqi = round(aq["current"]["us_aqi"])
    except Exception:
        pass

    sun = None
    if daily.get("sunrise") and daily.get("sunset"):
        sun = {"sunrise": daily["sunrise"][0][-5:], "sunset": daily["sunset"][0][-5:]}

    return {
        "source": "live",
        "location": {"name": name or f"{lat:.2f}, {lon:.2f}", "lat": lat, "lon": lon},
        "units": {"temp": "°C", "wind": "km/h"},
        "current": {
            "temp": current["temperature_2m"],
            "feels": current["apparent_temperature"],
            "humidity": current["relative_humidity_2m"],
            "wind": current["wind_speed_10m"],
            "code": current["weather_code"],
            "aqi": aqi,
        },
        "sun": sun,
        "hourly": hours,
        "daily": days,
    }


def live_geocode(query: str) -> dict:
    q = urllib.parse.urlencode({"name": query, "count": 5, "language": "en", "format": "json"})
    raw = json.loads(fetch_url(f"https://geocoding-api.open-meteo.com/v1/search?{q}"))
    results = [
        {
            "name": r["name"],
            "region": r.get("admin1", ""),
            "country": r.get("country", ""),
            "lat": r["latitude"],
            "lon": r["longitude"],
        }
        for r in raw.get("results", [])
    ]
    return {"source": "live", "results": results}


# ---------------------------------------------------------------------------
# Air quality + pollen (Open-Meteo Air Quality API — no key)
# ---------------------------------------------------------------------------
AIR_TTL = 30 * 60
# US AQI category bands (breakpoint upper bound → label/tone).
_AQI_BANDS = [
    (50, "Good", "up"), (100, "Moderate", "neutral"),
    (150, "Unhealthy (sensitive)", "warn"), (200, "Unhealthy", "down"),
    (300, "Very unhealthy", "down"), (10_000, "Hazardous", "down"),
]
_POLLEN_KEYS = [
    ("alder_pollen", "Alder"), ("birch_pollen", "Birch"), ("grass_pollen", "Grass"),
    ("mugwort_pollen", "Mugwort"), ("olive_pollen", "Olive"), ("ragweed_pollen", "Ragweed"),
]


def aqi_band(aqi) -> dict:
    if aqi is None:
        return {"label": "—", "tone": "neutral"}
    for upper, label, tone in _AQI_BANDS:
        if aqi <= upper:
            return {"label": label, "tone": tone}
    return {"label": "Hazardous", "tone": "down"}


def _pollen_level(grains) -> str:
    if grains is None:
        return "—"
    # grains/m³ → rough low/moderate/high/very-high bands (grass-calibrated).
    return ("Very high" if grains >= 50 else "High" if grains >= 20
            else "Moderate" if grains >= 5 else "Low")


def live_air(lat: float, lon: float, name: str | None) -> dict:
    fields = ("us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,"
              "alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen")
    q = urllib.parse.urlencode({"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "current": fields})
    raw = json.loads(fetch_url(f"https://air-quality-api.open-meteo.com/v1/air-quality?{q}", timeout=8))
    cur = raw.get("current") or {}
    aqi = cur.get("us_aqi")
    aqi = round(aqi) if aqi is not None else None
    pollutants = [
        {"key": "pm2_5", "label": "PM2.5", "value": cur.get("pm2_5"), "unit": "µg/m³"},
        {"key": "pm10", "label": "PM10", "value": cur.get("pm10"), "unit": "µg/m³"},
        {"key": "ozone", "label": "O₃", "value": cur.get("ozone"), "unit": "µg/m³"},
        {"key": "no2", "label": "NO₂", "value": cur.get("nitrogen_dioxide"), "unit": "µg/m³"},
        {"key": "so2", "label": "SO₂", "value": cur.get("sulphur_dioxide"), "unit": "µg/m³"},
        {"key": "co", "label": "CO", "value": cur.get("carbon_monoxide"), "unit": "µg/m³"},
    ]
    pollen = [{"key": k, "label": lb, "value": cur.get(k), "level": _pollen_level(cur.get(k))}
              for k, lb in _POLLEN_KEYS if cur.get(k) is not None]
    return {
        "source": "live",
        "location": {"name": name or f"{lat:.2f}, {lon:.2f}", "lat": lat, "lon": lon},
        "aqi": aqi, "band": aqi_band(aqi),
        "pollutants": [p for p in pollutants if p["value"] is not None],
        "pollen": pollen,
    }


def sample_air(name: str | None = None) -> dict:
    aqi = 42
    return {
        "source": "sample",
        "location": {"name": name or "New York", "lat": 40.71, "lon": -74.01},
        "aqi": aqi, "band": aqi_band(aqi),
        "pollutants": [
            {"key": "pm2_5", "label": "PM2.5", "value": 9.8, "unit": "µg/m³"},
            {"key": "pm10", "label": "PM10", "value": 17.2, "unit": "µg/m³"},
            {"key": "ozone", "label": "O₃", "value": 61.0, "unit": "µg/m³"},
            {"key": "no2", "label": "NO₂", "value": 12.4, "unit": "µg/m³"},
            {"key": "so2", "label": "SO₂", "value": 3.1, "unit": "µg/m³"},
            {"key": "co", "label": "CO", "value": 120.0, "unit": "µg/m³"},
        ],
        "pollen": [
            {"key": "grass_pollen", "label": "Grass", "value": 7.0, "level": "Moderate"},
            {"key": "birch_pollen", "label": "Birch", "value": 1.5, "level": "Low"},
            {"key": "ragweed_pollen", "label": "Ragweed", "value": 22.0, "level": "High"},
        ],
    }


# ---------------------------------------------------------------------------
# Marine conditions — Open-Meteo Marine API (no key): waves, swell, sea temp.
# ---------------------------------------------------------------------------
MARINE_TTL = 30 * 60
_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _compass(deg) -> str | None:
    if deg is None:
        return None
    return _COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def _sea_state(h) -> str:
    if h is None:
        return "—"
    return ("Calm" if h < 0.5 else "Smooth" if h < 1.25 else "Moderate"
            if h < 2.5 else "Rough" if h < 4 else "Very rough" if h < 6 else "High")


def live_marine(lat: float, lon: float, name: str | None) -> dict:
    fields = ("wave_height,wave_direction,wave_period,sea_surface_temperature,"
              "swell_wave_height,swell_wave_period,wind_wave_height")
    q = urllib.parse.urlencode({"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
                                "current": fields, "daily": "wave_height_max",
                                "timezone": "auto"})
    raw = json.loads(fetch_url(f"https://marine-api.open-meteo.com/v1/marine?{q}", timeout=8))
    cur = raw.get("current") or {}
    wh = cur.get("wave_height")
    daily = raw.get("daily") or {}
    peak = (daily.get("wave_height_max") or [None])[0]
    return {
        "source": "live",
        "location": {"name": name or f"{lat:.2f}, {lon:.2f}", "lat": lat, "lon": lon},
        "waveHeight": wh, "wavePeriod": cur.get("wave_period"),
        "waveDir": cur.get("wave_direction"), "waveDirText": _compass(cur.get("wave_direction")),
        "swellHeight": cur.get("swell_wave_height"), "swellPeriod": cur.get("swell_wave_period"),
        "windWaveHeight": cur.get("wind_wave_height"),
        "seaTemp": cur.get("sea_surface_temperature"),
        "seaState": _sea_state(wh), "waveMax": peak,
    }


def sample_marine(name: str | None = None) -> dict:
    return {
        "source": "sample",
        "location": {"name": name or "Cape Town", "lat": -33.92, "lon": 18.42},
        "waveHeight": 1.6, "wavePeriod": 11.2, "waveDir": 205, "waveDirText": "SSW",
        "swellHeight": 1.3, "swellPeriod": 12.4, "windWaveHeight": 0.7,
        "seaTemp": 15.8, "seaState": _sea_state(1.6), "waveMax": 2.1,
    }


def live_markets(ids: list[str] | None = None) -> dict:
    joined = ",".join(ids or DEFAULT_CRYPTO_IDS)
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency=usd&ids={joined}&sparkline=true&price_change_percentage=24h"
    )
    raw = json.loads(fetch_url(url))
    assets = []
    for coin in raw:
        spark = coin.get("sparkline_in_7d", {}).get("price", [])
        # thin the 7d hourly sparkline to ~40 points
        step = max(1, len(spark) // 40)
        assets.append(
            {
                "id": coin.get("id", coin["name"].lower()),
                "symbol": coin["symbol"].upper(),
                "name": coin["name"],
                "price": coin["current_price"],
                "change24h": coin.get("price_change_percentage_24h") or 0.0,
                "spark": [round(p, 6) for p in spark[::step]],
            }
        )
    if not assets:
        raise RuntimeError("empty market response")
    return {"source": "live", "assets": assets}


# ---------------------------------------------------------------------------
# Crypto: per-coin detail + OHLC chart with technical indicators
# ---------------------------------------------------------------------------
COIN_TTL = 90
COIN_CHART_TTL = 5 * 60
_CHART_DAYS = {"1": 1, "7": 7, "30": 30, "90": 90, "365": 365}


def live_coin_detail(coin_id: str) -> dict:
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
           "?localization=false&tickers=false&market_data=true"
           "&community_data=false&developer_data=false")
    raw = json.loads(fetch_url(url))
    m = raw.get("market_data") or {}

    def usd(block):
        return (block or {}).get("usd")

    return {
        "source": "live",
        "id": raw.get("id", coin_id),
        "symbol": (raw.get("symbol") or "").upper(),
        "name": raw.get("name", coin_id),
        "rank": raw.get("market_cap_rank"),
        "price": usd(m.get("current_price")),
        "marketCap": usd(m.get("market_cap")),
        "volume": usd(m.get("total_volume")),
        "supply": m.get("circulating_supply"),
        "totalSupply": m.get("total_supply"),
        "maxSupply": m.get("max_supply"),
        "ath": usd(m.get("ath")),
        "athChange": usd(m.get("ath_change_percentage")),
        "atl": usd(m.get("atl")),
        "atlChange": usd(m.get("atl_change_percentage")),
        "changes": {
            "1h": (m.get("price_change_percentage_1h_in_currency") or {}).get("usd"),
            "24h": m.get("price_change_percentage_24h"),
            "7d": m.get("price_change_percentage_7d"),
            "30d": m.get("price_change_percentage_30d"),
            "1y": m.get("price_change_percentage_1y"),
        },
    }


def _chart_payload(coin_id: str, days: int, candles: list[dict], source: str) -> dict:
    closes = [c["c"] for c in candles]
    sma20 = indicators.sma(closes, 20)
    sma50 = indicators.sma(closes, 50)
    boll = indicators.bollinger(closes, 20, 2.0)
    return {
        "source": source,
        "id": coin_id,
        "days": days,
        "candles": candles,
        "overlays": {
            "sma20": sma20, "sma50": sma50,
            "bollUpper": boll["upper"], "bollLower": boll["lower"],
        },
        "signals": indicators.read_signals(closes),
    }


def live_coin_chart(coin_id: str, days: int) -> dict:
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
           f"?vs_currency=usd&days={days}")
    raw = json.loads(fetch_url(url))
    candles = [
        {"t": int(row[0]), "o": row[1], "h": row[2], "l": row[3], "c": row[4]}
        for row in raw if isinstance(row, list) and len(row) >= 5
    ]
    if len(candles) < 2:
        raise RuntimeError("empty ohlc response")
    return _chart_payload(coin_id, days, candles, "live")


def _synth_candles(seed: str, days: int, base: float) -> list[dict]:
    """Deterministic pseudo-random candles so offline/e2e have a real chart."""
    import random
    rng = random.Random(f"{seed}:{days}")
    n = 60
    price = base
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    step = max(1, days) * 86400 * 1000 // n
    out = []
    for i in range(n):
        drift = rng.uniform(-0.03, 0.03)
        o = price
        c = max(0.01, price * (1 + drift))
        hi = max(o, c) * (1 + rng.uniform(0, 0.015))
        lo = min(o, c) * (1 - rng.uniform(0, 0.015))
        out.append({"t": now - (n - i) * step, "o": round(o, 4), "h": round(hi, 4),
                    "l": round(lo, 4), "c": round(c, 4)})
        price = c
    return out


def sample_coin_detail(coin_id: str) -> dict:
    base = {"bitcoin": 112840, "ethereum": 4188, "solana": 216.4}.get(coin_id, 100.0)
    return {
        "source": "sample", "id": coin_id, "symbol": coin_id[:4].upper(),
        "name": coin_id.title(), "rank": 1, "price": base,
        "marketCap": base * 19_000_000, "volume": base * 400_000,
        "supply": 19_000_000, "totalSupply": 21_000_000, "maxSupply": 21_000_000,
        "ath": base * 1.4, "athChange": -28.5, "atl": base * 0.02, "atlChange": 4200.0,
        "changes": {"1h": 0.2, "24h": 2.41, "7d": 5.1, "30d": -3.2, "1y": 64.0},
    }


def sample_coin_chart(coin_id: str, days: int) -> dict:
    base = {"bitcoin": 112840, "ethereum": 4188, "solana": 216.4}.get(coin_id, 100.0)
    return _chart_payload(coin_id, days, _synth_candles(coin_id, days, base), "sample")


CRYPTO_GLOBAL_TTL = 5 * 60
CRYPTO_TRENDING_TTL = 10 * 60


def _fear_greed() -> dict | None:
    try:
        raw = json.loads(fetch_url("https://api.alternative.me/fng/?limit=1", timeout=6))
        entry = raw["data"][0]
        return {"value": int(entry["value"]), "label": entry["value_classification"]}
    except Exception:
        return None


def live_crypto_global() -> dict:
    g = json.loads(fetch_url("https://api.coingecko.com/api/v3/global"))["data"]
    return {
        "source": "live",
        "marketCap": g["total_market_cap"]["usd"],
        "volume": g["total_volume"]["usd"],
        "btcDominance": g["market_cap_percentage"]["btc"],
        "ethDominance": g["market_cap_percentage"].get("eth"),
        "change24h": g.get("market_cap_change_percentage_24h_usd"),
        "coins": g.get("active_cryptocurrencies"),
        "fearGreed": _fear_greed(),
    }


def sample_crypto_global() -> dict:
    return {
        "source": "sample", "marketCap": 3.62e12, "volume": 1.41e11,
        "btcDominance": 54.2, "ethDominance": 12.8, "change24h": 1.83,
        "coins": 13500, "fearGreed": {"value": 62, "label": "Greed"},
    }


def live_crypto_trending() -> dict:
    raw = json.loads(fetch_url("https://api.coingecko.com/api/v3/search/trending"))
    coins = []
    for entry in raw.get("coins", [])[:10]:
        item = entry.get("item", {})
        coins.append({
            "id": item.get("id"), "symbol": (item.get("symbol") or "").upper(),
            "name": item.get("name"), "rank": item.get("market_cap_rank"),
        })
    return {"source": "live", "coins": coins}


def sample_crypto_trending() -> dict:
    names = [("pepe", "PEPE", "Pepe", 45), ("arbitrum", "ARB", "Arbitrum", 52),
             ("sui", "SUI", "Sui", 28), ("render-token", "RNDR", "Render", 31),
             ("celestia", "TIA", "Celestia", 60)]
    return {"source": "sample",
            "coins": [{"id": i, "symbol": s, "name": n, "rank": r} for i, s, n, r in names]}


# ---------------------------------------------------------------------------
# Data-source registry (§0.3). Each entry declares its cache TTL and a
# live/sample pair; Api.fetch_source(name, *args) wraps _cached uniformly so a
# new upstream is a one-liner here plus a live_*/sample_* function. Positional
# args are forwarded to both functions and, unless an explicit key is given,
# folded into the cache key. The convention: no source ships without a sample.
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "quakes": {"ttl": QUAKES_TTL, "live": live_quakes, "sample": sample_quakes},
    "spaceweather": {"ttl": SPACE_TTL, "live": live_spaceweather, "sample": sample_spaceweather},
    "crypto:global": {"ttl": CRYPTO_GLOBAL_TTL,
                      "live": live_crypto_global, "sample": sample_crypto_global},
    "crypto:trending": {"ttl": CRYPTO_TRENDING_TTL,
                        "live": live_crypto_trending, "sample": sample_crypto_trending},
    "gaming:free": {"ttl": GAMING_TTL, "live": live_free_games, "sample": sample_free_games},
    "gaming:deals": {"ttl": GAMING_TTL, "live": live_steam_deals, "sample": sample_steam_deals},
    "standings": {"ttl": STANDINGS_TTL, "live": live_standings, "sample": sample_standings},
    "scores": {"ttl": SCORES_TTL, "live": live_scores, "sample": sample_scores},
    "racing": {"ttl": RACING_TTL, "live": live_racing, "sample": sample_racing},
    "teamsched": {"ttl": TEAM_SCHEDULE_TTL,
                  "live": live_team_schedule, "sample": sample_team_schedule},
    "teamnews": {"ttl": TEAM_NEWS_TTL, "live": live_team_news, "sample": sample_team_news},
    "trials": {"ttl": TRIALS_TTL, "live": live_trials, "sample": sample_trials},
    "drug": {"ttl": DRUG_TTL, "live": live_drug, "sample": sample_drug},
    "repos": {"ttl": REPOS_TTL, "live": live_repos, "sample": sample_repos},
    "papers": {"ttl": PAPERS_TTL, "live": live_papers, "sample": sample_papers},
    "ainews": {"ttl": AI_NEWS_TTL, "live": live_ai_news, "sample": sample_ai_news},
    "commodities": {"ttl": COMMODITIES_TTL,
                    "live": live_commodities, "sample": sample_commodities},
}


# ---------------------------------------------------------------------------
# Memory recall — lexical TF-IDF ranking (zero-dependency vector-recall stand-in)
# ---------------------------------------------------------------------------
_RECALL_STOP = frozenset(
    "the a an and or but of to in on at for with my me i you your is are was were "
    "be been being this that these those it its as by from".split())


def _recall_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) >= 3 and t not in _RECALL_STOP]


def rank_facts(facts: list[str], query: str, limit: int = 12) -> list[str]:
    """Rank `facts` by lexical relevance to `query`. TF-IDF over fact terms:
    a term is worth more when it is rare across the corpus. Ties and empty
    queries fall back to recency (facts are stored oldest→newest)."""
    n = len(facts)
    if n == 0:
        return []
    q = set(_recall_tokens(query))
    if not q:
        return facts[-limit:][::-1]  # newest first
    # document frequency of each query term across the facts
    tokenized = [_recall_tokens(f) for f in facts]
    df = {term: 0 for term in q}
    for toks in tokenized:
        present = set(toks)
        for term in q:
            if term in present:
                df[term] += 1
    scored = []
    for idx, toks in enumerate(tokenized):
        counts = {}
        for t in toks:
            if t in q:
                counts[t] = counts.get(t, 0) + 1
        score = 0.0
        for term, tf in counts.items():
            idf = math.log(1 + n / (1 + df[term]))
            score += (1 + math.log(tf)) * idf
        if score > 0:
            scored.append((score, idx, facts[idx]))
    if not scored:
        return facts[-limit:][::-1]
    # highest score first; newer facts (larger idx) win ties
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [f for _, _, f in scored[:limit]]


# ---------------------------------------------------------------------------
# API dispatch: try cache → live → sample
# ---------------------------------------------------------------------------
class Api:
    def __init__(
        self,
        offline: bool = False,
        state_store: StateStore | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.offline = offline
        self.data_dir = data_dir or APP_DIR / "data"
        self.assistant = assistant_module.Assistant()
        self.assistant.services = self
        self.state_store = state_store
        self.automations = Automations(self.data_dir / "automations.json", self)
        self.feeds = FeedConfig(self.data_dir / "feeds.json")
        self.calendars = CalendarConfig(self.data_dir / "calendars.json")
        self.telemetry = Telemetry(self.data_dir / "telemetry.jsonl")
        self.evolve = Reflection(self.data_dir / "proposals.json", self)
        self._ics_epoch = 0
        self._memory_lock = threading.Lock()
        self._routing_lock = threading.Lock()
        self.assistant.router.set_overrides(self._routing_load())

    # -- agent memory (a plain markdown file the agent reads/writes) --------
    @property
    def memory_path(self) -> Path:
        return self.data_dir / "memory.md"

    def memory_read(self) -> str:
        with self._memory_lock:
            try:
                return self.memory_path.read_text(encoding="utf-8")
            except OSError:
                return ""

    def memory_recall(self, query: str, limit: int = 12) -> list[str]:
        """Return the stored facts most relevant to `query`, newest-first on
        ties. Lexical TF-IDF ranking — the zero-dependency stand-in for vector
        recall. Falls back to the most recent facts when the query is empty."""
        text = self.memory_read()
        facts = [ln[2:].strip() for ln in text.splitlines() if ln.startswith("- ")]
        return rank_facts(facts, query, limit)

    def memory_append(self, fact: str) -> None:
        fact = " ".join(fact.split())[:500]
        if not fact:
            raise ApiError(400, "empty fact")
        with self._memory_lock:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            existing = ""
            try:
                existing = self.memory_path.read_text(encoding="utf-8")
            except OSError:
                pass
            stamp = datetime.now().strftime("%Y-%m-%d")
            with self.memory_path.open("a", encoding="utf-8") as f:
                if not existing:
                    f.write("# Hermes Hub — agent memory\n")
                f.write(f"- ({stamp}) {fact}\n")
            # keep the file bounded: newest 200 facts
            lines = self.memory_path.read_text(encoding="utf-8").splitlines()
            facts = [ln for ln in lines if ln.startswith("- ")]
            if len(facts) > 200:
                kept = [lines[0]] + facts[-200:]
                self.memory_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def memory_overwrite(self, text: str) -> None:
        with self._memory_lock:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_path.write_text(text, encoding="utf-8")

    # -- learned operating guidelines (self-evolution addenda) --------------
    @property
    def agent_notes_path(self) -> Path:
        return self.data_dir / "agent_notes.md"

    def agent_notes_read(self) -> str:
        try:
            return self.agent_notes_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def agent_notes_append(self, text: str) -> None:
        text = " ".join((text or "").split())[:400]
        if not text:
            return
        with self._memory_lock:
            self.agent_notes_path.parent.mkdir(parents=True, exist_ok=True)
            existing = self.agent_notes_read()
            with self.agent_notes_path.open("a", encoding="utf-8") as f:
                if not existing:
                    f.write("# Learned operating guidelines\n")
                f.write(f"- {text}\n")

    def _cached(self, key: str, ttl: float, live_fn, sample_fn):
        cached = CACHE.get(key)
        if cached is not None:
            return cached
        if not self.offline:
            try:
                result = live_fn()
                CACHE.set(key, result, ttl)
                return result
            except Exception:
                pass
        result = sample_fn()
        # Cache samples briefly too, so an offline session isn't re-reading disk.
        CACHE.set(key, result, min(ttl, 60))
        return result

    def fetch_source(self, name: str, *args, key: str | None = None):
        """Registry-driven fetch: look up a declared source and run it through
        the standard cache → live → sample path (§0.3)."""
        spec = SOURCES[name]
        if key is None:
            key = name if not args else f"{name}:{':'.join(str(a).lower() for a in args)}"
        return self._cached(key, spec["ttl"],
                            lambda: spec["live"](*args),
                            lambda: spec["sample"](*args))

    def news(self, params: dict) -> dict:
        limit = max(1, min(int(params.get("limit", ["30"])[0]), 60))
        if params.get("all", ["0"])[0] not in ("0", "", "false"):
            return self._news_all(limit)
        topic = params.get("topic", ["top"])[0].lower()
        sources = self.feeds.sources_for(topic)
        if sources is None:
            raise ApiError(400, f"unknown topic {topic!r}; valid: {self.feeds.topics()}")
        return self._cached(
            f"news:{topic}:{limit}",
            NEWS_TTL,
            lambda: live_news(topic, limit, sources),
            lambda: sample_news(topic, limit),
        )

    def _news_all(self, limit: int) -> dict:
        # Cross-topic aggregation: merge every configured topic (bar the "top"
        # roll-up) into one stream, tagging each item with its origin topic.
        topics = [t for t in self.feeds.topics() if t != "top"]
        collected: list[dict] = []
        source = "sample"
        for topic in topics:
            try:
                data = self.news({"topic": [topic], "limit": ["24"]})
            except Exception:
                continue
            if data.get("source") == "live":
                source = "live"
            for item in data.get("items", []):
                collected.append({**item, "topic": topic})
        return {"topic": "all", "source": source, "items": merge_items(collected, limit)}

    # -- ICS calendar subscriptions ------------------------------------------
    def calendars_list(self, params: dict) -> dict:
        return {"calendars": self.calendars.list()}

    def calendars_op(self, body: dict) -> dict:
        op = body.get("op")
        if op == "add":
            self.calendars.add(str(body.get("name", "")), str(body.get("url", "")))
        elif op == "remove":
            self.calendars.remove(str(body.get("url", "")))
        else:
            raise ApiError(400, "op must be add or remove")
        self._ics_epoch += 1  # invalidates every cached merged-events window
        return {"calendars": self.calendars.list()}

    def ics_events(self, params: dict) -> dict:
        try:
            days = max(1, min(int(params.get("days", ["60"])[0]), 365))
        except ValueError:
            raise ApiError(400, "days must be an integer") from None
        cache_key = f"ics-events:{self._ics_epoch}:{days}"
        cached = CACHE.get(cache_key)
        if cached is not None:
            return cached
        window_start = datetime.now().date() - timedelta(days=1)
        window_end = window_start + timedelta(days=days + 1)
        events: list[dict] = []
        failures: list[str] = []
        for cal in self.calendars.list():
            try:
                raw = fetch_url(cal["url"]).decode("utf-8", errors="replace")
                events.extend(parse_ics(raw, cal["name"], window_start, window_end))
            except Exception:
                failures.append(cal["name"])
        events.sort(key=lambda e: (e["date"], e.get("time") or ""))
        result = {"events": events[:500], "failures": failures}
        CACHE.set(cache_key, result, ICS_TTL)
        return result

    def feeds_config(self, params: dict) -> dict:
        return self.feeds.snapshot()

    def feeds_op(self, body: dict) -> dict:
        op = body.get("op")
        if op == "add_topic":
            self.feeds.add_topic(str(body.get("name", "")))
        elif op == "add_search":
            self.feeds.add_search(str(body.get("name", "")), str(body.get("query", "")))
        elif op == "remove_topic":
            self.feeds.remove_topic(str(body.get("name", "")))
        elif op == "add_source":
            self.feeds.add_source(str(body.get("topic", "")), str(body.get("name", "")),
                                  str(body.get("url", "")))
        elif op == "remove_source":
            self.feeds.remove_source(str(body.get("topic", "")), str(body.get("url", "")))
        elif op == "reset":
            self.feeds.reset()
        else:
            raise ApiError(400, "op must be add_topic, add_search, remove_topic, add_source, remove_source or reset")
        CACHE.clear()  # cached merged topics may now be stale
        return self.feeds.snapshot()

    @staticmethod
    def _latlon(params: dict) -> tuple[float, float]:
        """Parse and validate lat/lon. Rejects non-numeric, non-finite
        (inf/nan) and out-of-range values before they reach an upstream URL."""
        try:
            lat = float(params.get("lat", ["40.7128"])[0])
            lon = float(params.get("lon", ["-74.0060"])[0])
        except (ValueError, TypeError):
            raise ApiError(400, "lat/lon must be numbers") from None
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ApiError(400, "lat/lon must be finite")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ApiError(400, "lat/lon out of range")
        return lat, lon

    def weather(self, params: dict) -> dict:
        lat, lon = self._latlon(params)
        name = params.get("name", [None])[0]
        return self._cached(
            f"weather:{lat:.3f}:{lon:.3f}",
            WEATHER_TTL,
            lambda: live_weather(lat, lon, name),
            lambda: sample_weather(name),
        )

    def air(self, params: dict) -> dict:
        lat, lon = self._latlon(params)
        name = params.get("name", [None])[0]
        return self._cached(
            f"air:{lat:.3f}:{lon:.3f}",
            AIR_TTL,
            lambda: live_air(lat, lon, name),
            lambda: sample_air(name),
        )

    def marine(self, params: dict) -> dict:
        lat, lon = self._latlon(params)
        name = params.get("name", [None])[0]
        return self._cached(
            f"marine:{lat:.3f}:{lon:.3f}",
            MARINE_TTL,
            lambda: live_marine(lat, lon, name),
            lambda: sample_marine(name),
        )

    def flights(self, params: dict) -> dict:
        lat, lon = self._latlon(params)
        name = params.get("name", [None])[0]
        return self._cached(
            f"flights:{lat:.2f}:{lon:.2f}",
            FLIGHTS_TTL,
            lambda: live_flights(lat, lon, name),
            lambda: sample_flights(name),
        )

    def alerts(self, params: dict) -> dict:
        lat, lon = self._latlon(params)
        name = params.get("name", [None])[0]
        return self._cached(
            f"alerts:{lat:.3f}:{lon:.3f}",
            ALERTS_TTL,
            lambda: live_alerts(lat, lon, name),
            lambda: sample_alerts(name),
        )

    def geocode(self, params: dict) -> dict:
        query = params.get("q", [""])[0].strip()
        if not query:
            raise ApiError(400, "missing q parameter")
        return self._cached(
            f"geocode:{query.lower()}",
            GEOCODE_TTL,
            lambda: live_geocode(query),
            lambda: {"source": "sample", "results": SAMPLES["geocode"]},
        )

    def markets(self, params: dict) -> dict:
        raw = params.get("ids", [""])[0]
        ids = [i for i in (re.sub(r"[^a-z0-9-]", "", part.lower())
                           for part in raw.split(",")) if i][:15] or None
        key = "markets:" + ",".join(ids or DEFAULT_CRYPTO_IDS)
        return self._cached(
            key, MARKETS_TTL,
            lambda: live_markets(ids),
            lambda: sample_markets(ids),
        )

    def crypto_coin(self, params: dict) -> dict:
        coin_id = re.sub(r"[^a-z0-9-]", "", params.get("id", ["bitcoin"])[0].lower())[:40]
        if not coin_id:
            raise ApiError(400, "missing coin id")
        return self._cached(
            f"coin:{coin_id}", COIN_TTL,
            lambda: live_coin_detail(coin_id),
            lambda: sample_coin_detail(coin_id),
        )

    def crypto_chart(self, params: dict) -> dict:
        coin_id = re.sub(r"[^a-z0-9-]", "", params.get("id", ["bitcoin"])[0].lower())[:40]
        if not coin_id:
            raise ApiError(400, "missing coin id")
        days = _CHART_DAYS.get(params.get("days", ["30"])[0], 30)
        return self._cached(
            f"coinchart:{coin_id}:{days}", COIN_CHART_TTL,
            lambda: live_coin_chart(coin_id, days),
            lambda: sample_coin_chart(coin_id, days),
        )

    def stocks(self, params: dict) -> dict:
        raw = params.get("symbols", [""])[0]
        syms = [s for s in (re.sub(r"[^a-z0-9.^-]", "", p.lower()) for p in raw.split(",")) if s][:15]
        syms = syms or ["^spx", "^ndq", "^dji", "aapl.us", "msft.us", "eurusd"]
        key = "stocks:" + ",".join(syms)
        return self._cached(key, STOCKS_TTL,
                            lambda: live_stocks(syms), lambda: sample_stocks(syms))

    def stocks_history(self, params: dict) -> dict:
        sym = re.sub(r"[^a-z0-9.^-]", "", params.get("symbol", ["^spx"])[0].lower())[:20]
        if not sym:
            raise ApiError(400, "missing symbol")
        return self._cached(f"stockhist:{sym}", STOCKS_HIST_TTL,
                            lambda: live_stock_history(sym), lambda: sample_stock_history(sym))

    def gaming_free(self, params: dict) -> dict:
        return self.fetch_source("gaming:free")

    def gaming_deals(self, params: dict) -> dict:
        return self.fetch_source("gaming:deals")

    def pubmed(self, params: dict) -> dict:
        query = " ".join(params.get("q", [""])[0].split())[:200] or "clinical medicine"
        return self._cached(f"pubmed:{query.lower()}", PUBMED_TTL,
                            lambda: live_pubmed(query), lambda: sample_pubmed(query))

    def pubmed_grounding_cached(self, query: str) -> dict:
        """Recent PubMed articles + abstracts for grounding SA MedBot. Offline-safe:
        returns sample article titles with no abstract text when offline."""
        query = " ".join((query or "").split())[:200]
        if not query:
            return {"articles": [], "text": ""}
        sample = lambda: {"articles": sample_pubmed(query)["articles"], "text": ""}
        if self.offline:
            return sample()
        return self._cached(f"pmground:{query.lower()}", PUBMED_TTL,
                            lambda: pubmed_grounding(query), sample)

    def pubmed_grounding(self, query: str) -> dict:
        """Grounding context for the MedBot (live only; empty offline)."""
        if self.offline:
            return {"articles": [], "text": ""}
        q = " ".join((query or "").split())[:200]
        if not q:
            return {"articles": [], "text": ""}
        cached = CACHE.get(f"pubground:{q.lower()}")
        if cached is not None:
            return cached
        try:
            result = pubmed_grounding(q)
        except Exception:
            result = {"articles": [], "text": ""}
        CACHE.set(f"pubground:{q.lower()}", result, PUBMED_TTL)
        return result

    def trials(self, params: dict) -> dict:
        query = " ".join(params.get("q", [""])[0].split())[:200] or "South Africa"
        return self.fetch_source("trials", query)

    def drug(self, params: dict) -> dict:
        query = re.sub(r"[^A-Za-z0-9 .\-]", "", params.get("q", [""])[0]).strip()[:60]
        if not query:
            raise ApiError(400, "missing drug name")
        return self.fetch_source("drug", query.lower())

    def repos(self, params: dict) -> dict:
        window = params.get("window", ["week"])[0].lower()
        if window not in ("day", "week", "month"):
            window = "week"
        return self.fetch_source("repos", window)

    def papers(self, params: dict) -> dict:
        cat = params.get("cat", ["cs.AI"])[0]
        if cat not in ("cs.AI", "cs.CL", "cs.LG"):
            cat = "cs.AI"
        return self.fetch_source("papers", cat)

    def ai_news(self, params: dict) -> dict:
        topic = params.get("topic", ["claude"])[0].lower()
        if topic not in AI_NEWS_QUERIES:
            topic = "claude"
        return self.fetch_source("ainews", topic)

    def commodities(self, params: dict) -> dict:
        return self.fetch_source("commodities")

    def podcast(self, params: dict) -> dict:
        url = params.get("url", [""])[0].strip()
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ApiError(400, "podcast url must be http(s)")
        return self._cached(f"podcast:{url}", PODCAST_TTL,
                            lambda: live_podcast(url), lambda: sample_podcast(url))

    def quakes(self, params: dict) -> dict:
        return self.fetch_source("quakes")

    def spaceweather(self, params: dict) -> dict:
        return self.fetch_source("spaceweather")

    def fx(self, params: dict) -> dict:
        base = re.sub(r"[^A-Za-z]", "", params.get("base", ["USD"])[0].upper())[:3] or "USD"
        raw = params.get("symbols", [""])[0]
        syms = [s for s in (re.sub(r"[^A-Za-z]", "", p.upper())[:3] for p in raw.split(",")) if s][:12]
        syms = syms or FX_DEFAULT
        return self._cached(f"fx:{base}:{','.join(syms)}", FX_TTL,
                            lambda: live_fx(base, syms), lambda: sample_fx(base, syms))

    def convert(self, params: dict) -> dict:
        # A rate table for the coin↔fiat↔coin converter. No new upstream:
        # reuses cached crypto USD prices + USD-based fiat rates.
        assets = self.markets({}).get("assets", [])
        coins = {a["symbol"]: {"name": a["name"], "usd": a["price"]}
                 for a in assets if a.get("price")}
        fx = self.fx({"base": ["USD"], "symbols": [",".join(CONVERT_FIATS)]})
        fiat = {"USD": 1.0}
        for cur, rate in fx.get("rates", {}).items():
            if rate:
                fiat[cur] = rate
        return {"coins": coins, "fiat": fiat,
                "asOf": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def standings(self, params: dict) -> dict:
        league = params.get("league", ["nba"])[0].lower()
        if league not in SPORT_LEAGUES:
            raise ApiError(400, f"unknown league {league!r}")
        return self.fetch_source("standings", league)

    def team_schedule(self, params: dict) -> dict:
        league = params.get("league", ["nba"])[0].lower()
        if league not in SPORT_LEAGUES:
            raise ApiError(400, f"unknown league {league!r}")
        team = re.sub(r"[^A-Za-z0-9 .-]", "", params.get("team", [""])[0]).strip()[:32]
        if not team:
            raise ApiError(400, "missing team")
        return self.fetch_source("teamsched", league, team)

    def team_news(self, params: dict) -> dict:
        team = re.sub(r"[^A-Za-z0-9 .&-]", "", params.get("team", [""])[0]).strip()[:48]
        if not team:
            raise ApiError(400, "missing team")
        return self.fetch_source("teamnews", team)

    def social(self, params: dict) -> dict:
        network = params.get("network", ["hn"])[0].lower()
        if network == "hn":
            return self._cached("social:hn", SOCIAL_TTL, live_social_hn,
                                lambda: sample_social("hn"))
        if network == "lobsters":
            return self._cached("social:lobsters", SOCIAL_TTL, live_social_lobsters,
                                lambda: sample_social("lobsters"))
        if network == "reddit":
            sub = re.sub(r"[^A-Za-z0-9_]", "", params.get("sub", ["popular"])[0])[:40] or "popular"
            return self._cached(f"social:reddit:{sub.lower()}", SOCIAL_TTL,
                                lambda: live_social_reddit(sub),
                                lambda: sample_social("reddit", sub))
        raise ApiError(400, "network must be hn, lobsters or reddit")

    def scores(self, params: dict) -> dict:
        league = params.get("league", ["nba"])[0].lower()
        if league not in SPORT_LEAGUES:
            raise ApiError(400, f"unknown league {league!r}; valid: {list(SPORT_LEAGUES)}")
        return self.fetch_source("scores", league)

    def racing(self, params: dict) -> dict:
        series = params.get("series", ["f1"])[0].lower()
        if series not in RACING_SERIES:
            raise ApiError(400, f"unknown series {series!r}; valid: {list(RACING_SERIES)}")
        return self.fetch_source("racing", series)

    def crypto_global(self, params: dict) -> dict:
        return self.fetch_source("crypto:global")

    def crypto_trending(self, params: dict) -> dict:
        return self.fetch_source("crypto:trending")

    def worldstate(self, params: dict) -> dict:
        cached = CACHE.get("worldstate")
        if cached is not None:
            return cached
        needed = sorted({t for spec in WORLD_DOMAINS.values() for t in spec["topics"]})
        news_by_topic = {}
        for topic in needed:
            try:  # a user may have deleted a default topic from their feeds
                news_by_topic[topic] = self.news({"topic": [topic], "limit": ["40"]})
            except ApiError:
                continue
        result = compute_worldstate(news_by_topic)
        CACHE.set("worldstate", result, WORLDSTATE_TTL)
        return result

    def reader(self, params: dict) -> dict:
        url = params.get("url", [""])[0].strip()
        if not url:
            raise ApiError(400, "missing url parameter")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ApiError(400, "only http(s) URLs can be opened")
        if host_is_blocked(parsed.hostname or ""):
            raise ApiError(400, "refusing to fetch private/internal addresses")
        return self._cached(
            f"reader:{url}",
            READER_TTL,
            lambda: live_reader(url),
            lambda: {
                "source": "sample",
                "url": url,
                "title": "",
                "blocks": [],
                "note": "Article fetch unavailable (offline). Showing the feed summary instead.",
            },
        )

    def health(self, params: dict) -> dict:
        return {
            "ok": True,
            "offline": self.offline,
            "sync": self.state_store is not None,
            "time": datetime.now(timezone.utc).isoformat(),
        }

    # -- cross-device state sync ------------------------------------------
    def state_get(self, params: dict) -> dict:
        if self.state_store is None:
            raise ApiError(503, "sync is not enabled on this server")
        return self.state_store.get()

    def state_rev(self, params: dict) -> dict:
        if self.state_store is None:
            raise ApiError(503, "sync is not enabled on this server")
        return {"rev": self.state_store.rev()}

    def state_put(self, body: dict) -> dict:
        if self.state_store is None:
            raise ApiError(503, "sync is not enabled on this server")
        state = body.get("state")
        if not isinstance(state, dict):
            raise ApiError(400, "state must be a JSON object")
        base_rev = body.get("baseRev")
        if base_rev is not None and not isinstance(base_rev, int):
            raise ApiError(400, "baseRev must be an integer")
        ok, rev = self.state_store.put(state, base_rev)
        if os.environ.get("HERMES_HUB_SYNC_LOG"):  # debug aid for tests
            marker = os.environ["HERMES_HUB_SYNC_LOG"]
            tasks = json.dumps(state.get("tasks", {}))
            print(f"[sync] PUT base={base_rev} -> ok={ok} rev={rev} "
                  f"tasksHaveMarker={marker in tasks}", flush=True)
        if not ok:
            raise ApiError(409, f"state changed elsewhere (server rev {rev})")
        return {"rev": rev}

    def assistant_status(self, params: dict) -> dict:
        return self.assistant.status()

    # -- routing overrides (Phase 1 UI) -------------------------------------
    @property
    def routing_path(self) -> Path:
        return self.data_dir / "routing.json"

    def _routing_load(self) -> dict:
        try:
            loaded = json.loads(self.routing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {t: str(loaded[t]) for t in ("fast", "core", "deep")
                if isinstance(loaded, dict) and loaded.get(t)}

    def routing_get(self, params: dict) -> dict:
        return self.assistant.router.snapshot()

    def routing_set(self, body: dict) -> dict:
        overrides = self._routing_load()
        updates = body.get("overrides") if isinstance(body.get("overrides"), dict) else body
        for tier in ("fast", "core", "deep"):
            if tier not in updates:
                continue
            model = updates[tier]
            if model in (None, ""):
                overrides.pop(tier, None)  # clear → back to env/default
                continue
            if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", model):
                raise ApiError(400, f"invalid model id for {tier}")
            overrides[tier] = model
        with self._routing_lock:
            self.routing_path.parent.mkdir(parents=True, exist_ok=True)
            self.routing_path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        self.assistant.router.set_overrides(overrides)
        return self.assistant.router.snapshot()

    # -- automations & notifications ----------------------------------------
    def automations_list(self, params: dict) -> dict:
        return {"rules": self.automations.list_rules()}

    def automations_op(self, body: dict) -> dict:
        op = body.get("op")
        if op == "create":
            try:
                return {"rule": self.automations.create_rule(body.get("rule") or {})}
            except ValueError as exc:
                raise ApiError(400, str(exc)) from None
        if op == "delete":
            if not self.automations.delete_rule(int(body.get("id", 0))):
                raise ApiError(404, "no such rule")
            return {"ok": True}
        if op == "toggle":
            rule = self.automations.toggle_rule(int(body.get("id", 0)))
            if rule is None:
                raise ApiError(404, "no such rule")
            return {"rule": rule}
        if op == "tick":  # evaluate immediately (used by tests and the UI)
            return {"fired": self.automations.tick()}
        raise ApiError(400, "op must be create, delete, toggle or tick")

    def notifications(self, params: dict) -> dict:
        try:
            after = int(params.get("after", ["0"])[0])
        except ValueError:
            raise ApiError(400, "after must be an integer") from None
        return self.automations.notifications_after(after)

    # -- self-evolution (Phase 6) -------------------------------------------
    def evolve_list(self, params: dict) -> dict:
        return {"proposals": self.evolve.list_proposals(), "pending": self.evolve.pending_count()}

    def evolve_history(self, params: dict) -> dict:
        try:
            limit = max(1, min(int(params.get("limit", ["30"])[0]), 100))
        except ValueError:
            raise ApiError(400, "limit must be an integer") from None
        return {"history": self.evolve.history(limit)}

    def evolve_reflect(self, body: dict) -> dict:
        created = self.evolve.reflect()
        return {"created": created, "pending": self.evolve.pending_count()}

    def evolve_proposal(self, body: dict) -> dict:
        op = body.get("op")
        try:
            pid = int(body.get("id", 0))
        except (TypeError, ValueError):
            raise ApiError(400, "id must be an integer") from None
        try:
            if op == "apply":
                return {"proposal": self.evolve.apply(pid)}
            if op == "dismiss":
                return {"proposal": self.evolve.dismiss(pid)}
            if op == "rollback":
                return {"proposal": self.evolve.rollback(pid)}
        except KeyError:
            raise ApiError(404, "no such proposal") from None
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None
        raise ApiError(400, "op must be apply, dismiss or rollback")

    # -- kill switch (Phase 4) ----------------------------------------------
    def killswitch_get(self, params: dict) -> dict:
        return {"frozen": self.automations.is_frozen()}

    def killswitch_set(self, body: dict) -> dict:
        if "frozen" not in body or not isinstance(body["frozen"], bool):
            raise ApiError(400, "body needs a boolean 'frozen'")
        frozen = self.automations.set_frozen(body["frozen"])
        self.telemetry.record({"kind": "killswitch", "frozen": frozen})
        return {"frozen": frozen}

    # -- agent telemetry (Phase 3) ------------------------------------------
    def telemetry_get(self, params: dict) -> dict:
        return {"events": self.telemetry.recent(50), "summary": self.telemetry.summary()}

    def telemetry_post(self, body: dict) -> dict:
        # The client only reports tool outcomes; route events are server-written.
        name = str(body.get("name", "")).strip()[:60]
        if not name:
            raise ApiError(400, "telemetry needs a tool name")
        approved = body.get("approved")
        self.telemetry.record({
            "kind": "tool",
            "name": name,
            "tier": str(body.get("tier", "auto"))[:12],
            "ok": bool(body.get("ok")),
            "approved": approved if isinstance(approved, bool) else None,
        })
        return {"ok": True}

    # -- server-side backups -------------------------------------------------
    @property
    def backups_dir(self) -> Path:
        return self.data_dir / "backups"

    def backup_now(self, body: dict) -> dict:
        synced = self.state_store.get() if self.state_store else {"rev": 0, "state": None}
        snap = {
            "kind": "hermes-hub-backup",
            "created": datetime.now(timezone.utc).isoformat(),
            "rev": synced["rev"],
            "state": synced["state"],
            "feeds": self.feeds.snapshot()["sources"],
            "calendars": self.calendars.list(),
            "automations": self.automations.list_rules(),
            "memory": self.memory_read(),
            "agent_notes": self.agent_notes_read(),
        }
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        name = f"hub-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
        path = self.backups_dir / name
        path.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        files = sorted(self.backups_dir.glob("hub-*.json"))
        for old in files[:-BACKUP_KEEP]:
            old.unlink()
        return {"name": name, "size": path.stat().st_size,
                "count": min(len(files), BACKUP_KEEP)}

    def backups_list(self, params: dict) -> dict:
        if not self.backups_dir.is_dir():
            return {"backups": []}
        out = []
        for f in sorted(self.backups_dir.glob("hub-*.json"), reverse=True):
            stat = f.stat()
            out.append({
                "name": f.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            })
        return {"backups": out}

    def backup_get(self, params: dict) -> dict:
        """Return a named server-side backup's full snapshot (for off-box download)."""
        name = params.get("name", [""])[0]
        if not re.fullmatch(r"hub-[0-9-]+\.json", name):
            raise ApiError(400, "bad backup name")
        path = self.backups_dir / name
        if not path.is_file():
            raise ApiError(404, "no such backup")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ApiError(500, "backup file is unreadable") from None

    def backup_import(self, body: dict) -> dict:
        """Adopt an uploaded snapshot into the server-side backup set.

        Lets a backup that was downloaded off-box (before a container reset) be
        brought back in, then restored via /api/backup/restore. Validated the
        same way as restore; the snapshot is not applied until restore is called.
        """
        snap = body.get("snapshot")
        if not isinstance(snap, dict) or snap.get("kind") != "hermes-hub-backup":
            raise ApiError(400, "not a hermes hub backup")
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        name = f"hub-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
        path = self.backups_dir / name
        path.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
        files = sorted(self.backups_dir.glob("hub-*.json"))
        for old in files[:-BACKUP_KEEP]:
            old.unlink()
        return {"name": name, "count": min(len(files), BACKUP_KEEP)}

    def backup_restore(self, body: dict) -> dict:
        name = str(body.get("name", ""))
        if not re.fullmatch(r"hub-[0-9-]+\.json", name):
            raise ApiError(400, "bad backup name")
        path = self.backups_dir / name
        if not path.is_file():
            raise ApiError(404, "no such backup")
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ApiError(500, "backup file is unreadable") from None
        if not isinstance(snap, dict) or snap.get("kind") != "hermes-hub-backup":
            raise ApiError(400, "not a hermes hub backup")
        restored: list[str] = []
        if isinstance(snap.get("state"), dict) and self.state_store is not None:
            self.state_store.put(snap["state"], None)  # rev advances; clients adopt
            restored.append("state")
        if isinstance(snap.get("feeds"), dict):
            self.feeds.restore(snap["feeds"])
            restored.append("feeds")
        if isinstance(snap.get("calendars"), list):
            self.calendars.restore(snap["calendars"])
            restored.append("calendars")
        if isinstance(snap.get("automations"), list):
            self.automations.replace_rules(snap["automations"])
            restored.append("automations")
        if isinstance(snap.get("memory"), str) and snap["memory"].strip():
            with self._memory_lock:
                self.memory_path.parent.mkdir(parents=True, exist_ok=True)
                self.memory_path.write_text(snap["memory"], encoding="utf-8")
            restored.append("memory")
        # agent_notes is restored even when empty so a rolled-back learned
        # guideline reverts to the pre-apply (possibly empty) state.
        if isinstance(snap.get("agent_notes"), str):
            with self._memory_lock:
                self.agent_notes_path.parent.mkdir(parents=True, exist_ok=True)
                self.agent_notes_path.write_text(snap["agent_notes"], encoding="utf-8")
            restored.append("agent_notes")
        CACHE.clear()
        self._ics_epoch += 1
        return {"restored": restored,
                "rev": self.state_store.rev() if self.state_store else 0}

    def assistant_medchat_stream(self, body: dict, handler) -> None:
        """SSE consult with the SA-medical persona."""
        self._sse(self.assistant.med_chat_stream(body), handler)

    def assistant_chat_stream(self, body: dict, handler) -> None:
        """SSE: delta events with live text, then one done event (chat shape)."""
        self._sse(self.assistant.chat_stream(body), handler)

    def _sse(self, generator, handler) -> None:
        try:  # pull the first event before committing to a 200 SSE response
            first = next(generator)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None
        except StopIteration:
            raise ApiError(500, "empty stream") from None

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        # SSE has no Content-Length; close the socket after the final event so
        # clients that wait for EOF (curl, urllib) terminate cleanly.
        handler.send_header("Connection", "close")
        handler.close_connection = True
        handler.end_headers()

        def emit(event: str, payload: dict) -> None:
            frame = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            handler.wfile.write(frame.encode("utf-8"))
            handler.wfile.flush()

        try:
            emit(*first)
            for event, payload in generator:
                emit(event, payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-stream
        except Exception as exc:
            try:
                emit("error", {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

    # -- agent tool proxy (tools that read server data / server state) -------
    def assistant_tool(self, body: dict) -> dict:
        name = body.get("name", "")
        tool_input = body.get("input") or {}
        try:
            result = self.assistant.run_server_tool(name, tool_input)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None
        return {"result": result}

    # POST endpoints (body is parsed JSON)
    def assistant_chat(self, body: dict) -> dict:
        try:
            return self.assistant.chat(body)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None

    def assistant_summarize(self, body: dict) -> dict:
        try:
            return self.assistant.summarize(body)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None

    def assistant_briefing(self, body: dict) -> dict:
        return self.assistant.briefing(body)


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
class HubHandler(BaseHTTPRequestHandler):
    api: Api  # set by make_server
    token: str | None = None  # set by make_server; None disables auth
    protocol_version = "HTTP/1.1"

    ROUTES = {
        "/api/news": "news",
        "/api/weather": "weather",
        "/api/air": "air",
        "/api/marine": "marine",
        "/api/spaceweather": "spaceweather",
        "/api/alerts": "alerts",
        "/api/flights": "flights",
        "/api/geocode": "geocode",
        "/api/markets": "markets",
        "/api/crypto/coin": "crypto_coin",
        "/api/crypto/chart": "crypto_chart",
        "/api/crypto/global": "crypto_global",
        "/api/crypto/trending": "crypto_trending",
        "/api/scores": "scores",
        "/api/racing": "racing",
        "/api/standings": "standings",
        "/api/team-schedule": "team_schedule",
        "/api/team-news": "team_news",
        "/api/quakes": "quakes",
        "/api/fx": "fx",
        "/api/convert": "convert",
        "/api/podcast": "podcast",
        "/api/pubmed": "pubmed",
        "/api/trials": "trials",
        "/api/drug": "drug",
        "/api/repos": "repos",
        "/api/papers": "papers",
        "/api/commodities": "commodities",
        "/api/ai-news": "ai_news",
        "/api/social": "social",
        "/api/gaming/free": "gaming_free",
        "/api/gaming/deals": "gaming_deals",
        "/api/stocks": "stocks",
        "/api/stocks/history": "stocks_history",
        "/api/worldstate": "worldstate",
        "/api/reader": "reader",
        "/api/health": "health",
        "/api/state": "state_get",
        "/api/state/rev": "state_rev",
        "/api/assistant/status": "assistant_status",
        "/api/assistant/routing": "routing_get",
        "/api/automations": "automations_list",
        "/api/notifications": "notifications",
        "/api/feeds": "feeds_config",
        "/api/calendars": "calendars_list",
        "/api/events": "ics_events",
        "/api/backups": "backups_list",
        "/api/backup/get": "backup_get",
        "/api/assistant/telemetry": "telemetry_get",
        "/api/killswitch": "killswitch_get",
        "/api/evolve": "evolve_list",
        "/api/evolve/history": "evolve_history",
    }

    POST_ROUTES = {
        "/api/state": "state_put",
        "/api/assistant/chat": "assistant_chat",
        "/api/assistant/summarize": "assistant_summarize",
        "/api/assistant/briefing": "assistant_briefing",
        "/api/assistant/tool": "assistant_tool",
        "/api/automations": "automations_op",
        "/api/feeds": "feeds_op",
        "/api/calendars": "calendars_op",
        "/api/backup": "backup_now",
        "/api/backup/restore": "backup_restore",
        "/api/backup/import": "backup_import",
        "/api/assistant/telemetry": "telemetry_post",
        "/api/killswitch": "killswitch_set",
        "/api/assistant/routing": "routing_set",
        "/api/evolve/reflect": "evolve_reflect",
        "/api/evolve/proposal": "evolve_proposal",
    }

    # POST endpoints that write their own (streaming) response.
    STREAM_ROUTES = {
        "/api/assistant/chat-stream": "assistant_chat_stream",
        "/api/assistant/medchat-stream": "assistant_medchat_stream",
    }

    # /api/health stays open so the lock screen can probe reachability.
    OPEN_PATHS = {"/api/health"}

    MAX_BODY = 512 * 1024

    def _authorized(self, path: str) -> bool:
        if self.token is None or path in self.OPEN_PATHS:
            return True
        header = self.headers.get("Authorization", "")
        supplied = header[7:] if header.startswith("Bearer ") else ""
        return hmac.compare_digest(supplied, self.token)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in self.ROUTES:
            if not self._authorized(path):
                self._send_json(401, {"error": "access code required"})
                return
            self._handle_api(self.ROUTES[path], urllib.parse.parse_qs(parsed.query))
        else:
            # The static shell contains no personal data; the API is the boundary.
            self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in self.POST_ROUTES and parsed.path not in self.STREAM_ROUTES:
            self._send_json(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > self.MAX_BODY:
                self._send_json(413, {"error": "request body too large"})
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return
        # sendBeacon (used for unload-time sync flushes) cannot set headers,
        # so the access code may ride in the body instead.
        body_token = body.pop("token", None)
        if not self._authorized(parsed.path) and not (
            self.token is not None
            and isinstance(body_token, str)
            and hmac.compare_digest(body_token, self.token)
        ):
            self._send_json(401, {"error": "access code required"})
            return
        if parsed.path in self.STREAM_ROUTES:
            try:
                getattr(self.api, self.STREAM_ROUTES[parsed.path])(body, self)
            except ApiError as exc:
                self._send_json(exc.status, {"error": exc.message})
            return
        self._handle_api(self.POST_ROUTES[parsed.path], body)

    def _handle_api(self, method: str, arg) -> None:
        try:
            payload = getattr(self.api, method)(arg)
            self._send_json(200, payload)
        except ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(500, {"error": f"internal error: {exc}"})

    def _serve_static(self, path: str) -> None:
        if path in ("/", ""):
            path = "/index.html"
        # Resolve inside PUBLIC_DIR only (no traversal). is_relative_to avoids the
        # sibling-prefix bug where "/…/public-secret" would pass a startswith check.
        candidate = (PUBLIC_DIR / path.lstrip("/")).resolve()
        if not candidate.is_relative_to(PUBLIC_DIR.resolve()) or not candidate.is_file():
            self._send_bytes(404, b"Not found", "text/plain; charset=utf-8")
            return
        mime = MIME_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
        self._send_bytes(200, candidate.read_bytes(), mime)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send_bytes(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send_bytes(self, status: int, body: bytes, mime: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionError, BrokenPipeError):
            # Browser closed the connection mid-response (navigated away,
            # refreshed, cancelled a prefetch). Nothing to send to — ignore.
            self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep the terminal quiet; errors surface as JSON


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Like ThreadingHTTPServer, but a client that hangs up mid-request does
    not spew a socket traceback to the console — those aborts are routine and
    harmless (a refresh or a cancelled prefetch), not a server fault."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def make_server(
    host: str,
    port: int,
    offline: bool,
    token: str | None = None,
    data_dir: Path | None = None,
    run_automations: bool = False,
) -> ThreadingHTTPServer:
    data_dir = data_dir or APP_DIR / "data"
    # Honour the documented HERMES_HUB_API_KEY by mapping it to the variable the
    # anthropic SDK actually reads, so either name enables the live agent.
    if os.environ.get("HERMES_HUB_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["HERMES_HUB_API_KEY"]
    store = StateStore(data_dir / "hub.db")
    api = Api(offline=offline, state_store=store, data_dir=data_dir)
    if run_automations:
        api.automations.start()
    handler = type("BoundHandler", (HubHandler,), {"api": api, "token": token})
    server = QuietThreadingHTTPServer((host, port), handler)
    server.api = api  # so callers (and tests) can reach the engine
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Hub dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never call upstream APIs; serve bundled sample data",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HERMES_HUB_TOKEN") or None,
        help="require this access code on all API calls (env: HERMES_HUB_TOKEN). "
        "Set it whenever the server is reachable beyond localhost.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="where to keep the sync database (default: ./data next to server.py)",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.token:
        print("WARNING: binding beyond localhost without --token / HERMES_HUB_TOKEN —")
        print("         anyone on the network can read and write your dashboard data.")

    server = make_server(
        args.host, args.port, args.offline, args.token, args.data_dir,
        run_automations=True,
    )
    mode = "offline (sample data)" if args.offline else "live (sample fallback)"
    lock = "locked (access code required)" if args.token else "open (localhost)"
    print(f"Hermes Hub → http://{args.host}:{args.port}  [{mode}] [{lock}] [sync on] [automations on]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
