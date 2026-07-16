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
        self._stack: list[str] = []
        self._buffer: list[str] = []
        self._current: str | None = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
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
    trimmed = []
    for tag, text in blocks:
        trimmed.append({"tag": tag, "text": text})
        total += len(text)
        if total > 20000:
            break
    return {
        "source": "live",
        "url": url,
        "title": WS_RE.sub(" ", extractor.title).strip() or url,
        "blocks": trimmed,
    }


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

    def news(self, params: dict) -> dict:
        topic = params.get("topic", ["top"])[0].lower()
        sources = self.feeds.sources_for(topic)
        if sources is None:
            raise ApiError(400, f"unknown topic {topic!r}; valid: {self.feeds.topics()}")
        limit = max(1, min(int(params.get("limit", ["30"])[0]), 60))
        return self._cached(
            f"news:{topic}:{limit}",
            NEWS_TTL,
            lambda: live_news(topic, limit, sources),
            lambda: sample_news(topic, limit),
        )

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
            raise ApiError(400, "op must be add_topic, remove_topic, add_source, remove_source or reset")
        CACHE.clear()  # cached merged topics may now be stale
        return self.feeds.snapshot()

    def weather(self, params: dict) -> dict:
        try:
            lat = float(params.get("lat", ["40.7128"])[0])
            lon = float(params.get("lon", ["-74.0060"])[0])
        except ValueError:
            raise ApiError(400, "lat/lon must be numbers") from None
        name = params.get("name", [None])[0]
        return self._cached(
            f"weather:{lat:.3f}:{lon:.3f}",
            WEATHER_TTL,
            lambda: live_weather(lat, lon, name),
            lambda: sample_weather(name),
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

    def assistant_chat_stream(self, body: dict, handler) -> None:
        """SSE: delta events with live text, then one done event (chat shape)."""
        generator = self.assistant.chat_stream(body)
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
        "/api/geocode": "geocode",
        "/api/markets": "markets",
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
        "/api/assistant/telemetry": "telemetry_post",
        "/api/killswitch": "killswitch_set",
        "/api/assistant/routing": "routing_set",
        "/api/evolve/reflect": "evolve_reflect",
        "/api/evolve/proposal": "evolve_proposal",
    }

    # POST endpoints that write their own (streaming) response.
    STREAM_ROUTES = {
        "/api/assistant/chat-stream": "assistant_chat_stream",
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
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep the terminal quiet; errors surface as JSON


def make_server(
    host: str,
    port: int,
    offline: bool,
    token: str | None = None,
    data_dir: Path | None = None,
    run_automations: bool = False,
) -> ThreadingHTTPServer:
    data_dir = data_dir or APP_DIR / "data"
    store = StateStore(data_dir / "hub.db")
    api = Api(offline=offline, state_store=store, data_dir=data_dir)
    if run_automations:
        api.automations.start()
    handler = type("BoundHandler", (HubHandler,), {"api": api, "token": token})
    server = ThreadingHTTPServer((host, port), handler)
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
