"""Last.fm tools for Hermes — 5 tools covering discovery, artist/track/tag
metadata, and charts.

Tools:
  lastfm_discover  — multi-seed similarity discovery (artists and/or tracks)
  lastfm_artist    — artist info, top tracks/albums/tags, similar artists
  lastfm_track     — track info, similar tracks, track search
  lastfm_tag       — genre/mood exploration (top artists, tracks, albums by tag)
  lastfm_charts    — trending charts (global or by country)

Auth: set LASTFM_API_KEY in your environment.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from plugins.lastfm.client import (
    LastFmAPIError,
    LastFmAuthError,
    LastFmClient,
    LastFmError,
)
from tools.registry import tool_error, tool_result

logger = logging.getLogger(__name__)


# ── auth guard ────────────────────────────────────────────────────────────────

_ATTRIBUTION = "Powered by Last.fm (https://www.last.fm)"


def _check_lastfm_available() -> bool:
    return bool(os.getenv("LASTFM_API_KEY", "").strip())


def _result(data: Dict[str, Any]) -> str:
    """Wrap tool_result, injecting required Last.fm attribution (ToS §4)."""
    data["_source"] = _ATTRIBUTION
    return tool_result(data)


def _client() -> LastFmClient:
    return LastFmClient()


def _lastfm_error(exc: Exception) -> str:
    if isinstance(exc, LastFmAuthError):
        return tool_error(
            str(exc),
            hint="Set LASTFM_API_KEY in your environment. "
                 "Free key at https://www.last.fm/api/account/create",
        )
    if isinstance(exc, LastFmAPIError):
        return tool_error(f"Last.fm API error {exc.code}: {exc}")
    return tool_error(f"Last.fm request failed: {type(exc).__name__}: {exc}")


def _coerce_limit(raw: Any, *, default: int = 20, min_: int = 1, max_: int = 100) -> int:
    try:
        return max(min_, min(max_, int(raw)))
    except Exception:
        return default


# ── discover: multi-seed similarity ─────────────────────────────────────────


def _composite_score(
    matches_per_seed: List[float], n_seeds: int, mode: str
) -> float:
    """Score an artist against multiple seeds.

    avg   — sum(matches) / total_seeds  (cross-seed artists naturally win)
    max   — best single-seed match      (pure similarity)
    boost — avg of matched seeds × (1 + 0.3 × extra_seeds)
    """
    if not matches_per_seed:
        return 0.0
    n_matched = len(matches_per_seed)
    best = max(matches_per_seed)
    avg_matched = sum(matches_per_seed) / n_matched

    if mode == "max":
        return round(best * 100, 1)
    if mode == "avg":
        return round((sum(matches_per_seed) / n_seeds) * 100, 1)
    if mode == "boost":
        return round(min(avg_matched * (1.0 + 0.3 * (n_matched - 1)), 1.0) * 100, 1)
    return round(avg_matched * 100, 1)


def _handle_lastfm_discover(args: Dict[str, Any], **_kw: Any) -> str:
    """Multi-seed music discovery: similar artists or tracks from seed lists."""
    artist_seeds: List[str] = _as_list(args.get("artists"))
    track_seeds: List[str] = _as_list(args.get("tracks"))
    count = _coerce_limit(args.get("count"), default=10, max_=50)
    scoring = str(args.get("scoring") or "avg").strip().lower()
    if scoring not in ("avg", "max", "boost"):
        scoring = "avg"
    include_tags = _coerce_limit(args.get("tags", 4), default=4, max_=10)

    if not artist_seeds and not track_seeds:
        return tool_error(
            "Provide at least one artist (artists=[...]) or track seed "
            "(tracks=['Artist:Track', ...])"
        )

    try:
        lf = _client()
    except Exception as exc:
        return _lastfm_error(exc)

    results = []
    seeds_lower = {s.lower() for s in artist_seeds}

    # ── artist seeds ─────────────────────────────────────────────────────────
    if artist_seeds:
        n_seeds = len(artist_seeds)
        pool: Dict[str, Dict] = defaultdict(lambda: {"matches": [], "from_seeds": []})

        for seed in artist_seeds:
            try:
                data = lf.artist_get_similar(seed, limit=count * 4)
                for a in data.get("similarartists", {}).get("artist", []):
                    name = a.get("name", "")
                    if name.lower() in seeds_lower:
                        continue
                    match = float(a.get("match", 0))
                    pool[name]["matches"].append(match)
                    pool[name]["from_seeds"].append(seed)
            except LastFmAPIError as exc:
                logger.debug("getSimilar failed for %r: %s", seed, exc)

        scored = sorted(
            [
                (name, _composite_score(info["matches"], n_seeds, scoring),
                 max(info["matches"]), info["from_seeds"])
                for name, info in pool.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for artist, score, best_raw, from_seeds in scored[:count]:
            entry: Dict[str, Any] = {
                "artist": artist,
                "score": score,
                "match_pct": round(best_raw * 100, 1),
                "seeds_matched": len(from_seeds),
                "total_seeds": n_seeds,
                "matched_seeds": from_seeds,
                "seed_type": "artist",
            }
            # Enrich with top track + tags
            try:
                tt = lf.artist_get_top_tracks(artist, limit=1)
                tracks = tt.get("toptracks", {}).get("track", [])
                if tracks:
                    t = tracks[0]
                    entry["top_track"] = t.get("name", "")
                    entry["top_track_plays"] = int(t.get("playcount") or 0) or None
            except Exception:
                pass
            if include_tags:
                try:
                    tg = lf.artist_get_top_tags(artist)
                    entry["tags"] = [
                        t["name"]
                        for t in tg.get("toptags", {}).get("tag", [])
                    ][:include_tags]
                except Exception:
                    entry["tags"] = []
            try:
                info = lf.artist_get_info(artist)
                a_info = info.get("artist", {})
                entry["url"] = a_info.get("url", "")
                stats = a_info.get("stats", {})
                entry["listeners"] = int(stats.get("listeners") or 0) or None
            except Exception:
                pass
            results.append(entry)

    # ── track seeds ───────────────────────────────────────────────────────────
    if track_seeds:
        n_seeds = len(track_seeds)
        tpool: Dict[str, Dict] = defaultdict(
            lambda: {"matches": [], "from_seeds": [], "playcount": None}
        )

        for raw in track_seeds:
            seed_artist, seed_track = _parse_track_seed(raw)
            if not seed_artist:
                logger.warning("Cannot parse track seed %r — use 'Artist:Track'", raw)
                continue
            try:
                data = lf.track_get_similar(seed_artist, seed_track, limit=count * 3)
                for t in data.get("similartracks", {}).get("track", []):
                    a_name = (t.get("artist") or {}).get("name", "") if isinstance(t.get("artist"), dict) else str(t.get("artist", ""))
                    t_name = t.get("name", "")
                    key = f"{a_name.lower()}::{t_name.lower()}"
                    match = float(t.get("match", 0))
                    tpool[key]["matches"].append(match)
                    tpool[key]["from_seeds"].append(raw)
                    tpool[key].setdefault("artist", a_name)
                    tpool[key].setdefault("track", t_name)
                    pc = int(t.get("playcount") or 0)
                    if pc:
                        tpool[key]["playcount"] = pc
            except LastFmAPIError as exc:
                logger.debug("getSimilar track failed for %r: %s", raw, exc)

        scored_t = sorted(
            [
                (key, _composite_score(info["matches"], n_seeds, scoring),
                 max(info["matches"]), info)
                for key, info in tpool.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        for _, score, best_raw, info in scored_t[:count]:
            artist = info["artist"]
            entry = {
                "artist": artist,
                "track": info["track"],
                "score": score,
                "match_pct": round(best_raw * 100, 1),
                "seeds_matched": len(info["from_seeds"]),
                "total_seeds": n_seeds,
                "matched_seeds": info["from_seeds"],
                "seed_type": "track",
                "playcount": info.get("playcount"),
            }
            if include_tags:
                try:
                    tg = lf.artist_get_top_tags(artist)
                    entry["tags"] = [
                        t["name"]
                        for t in tg.get("toptags", {}).get("tag", [])
                    ][:include_tags]
                except Exception:
                    entry["tags"] = []
            results.append(entry)

    # Deduplicate by (artist, track), keep highest score
    seen: Dict[str, Any] = {}
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        key = (r["artist"].lower(), r.get("track", r.get("top_track", "")).lower())
        if key not in seen:
            seen[key] = r

    final = list(seen.values())[:count]
    return _result({
        "count": len(final),
        "scoring": scoring,
        "artist_seeds": artist_seeds,
        "track_seeds": track_seeds,
        "recommendations": final,
    })


# ── artist ────────────────────────────────────────────────────────────────────

def _handle_lastfm_artist(args: Dict[str, Any], **_kw: Any) -> str:
    action = str(args.get("action") or "info").strip().lower()
    artist = str(args.get("artist") or "").strip()
    if not artist:
        return tool_error("artist is required")

    try:
        lf = _client()

        if action == "info":
            data = lf.artist_get_info(artist)
            a = data.get("artist", {})
            stats = a.get("stats", {})
            bio = a.get("bio", {})
            tags = [t["name"] for t in a.get("tags", {}).get("tag", [])]
            return _result({
                "name": a.get("name", artist),
                "url": a.get("url", ""),
                "listeners": int(stats.get("listeners") or 0) or None,
                "playcount": int(stats.get("playcount") or 0) or None,
                "tags": tags,
                "summary": _clean_bio(bio.get("summary", "")),
                "similar": [
                    s.get("name") for s in a.get("similar", {}).get("artist", [])
                ],
            })

        if action == "similar":
            limit = _coerce_limit(args.get("limit"), default=20)
            data = lf.artist_get_similar(artist, limit=limit)
            similar = data.get("similarartists", {}).get("artist", [])
            return _result({
                "artist": artist,
                "similar": [
                    {"name": a.get("name", ""), "match": round(float(a.get("match", 0)) * 100, 1)}
                    for a in similar
                ],
            })

        if action == "top_tracks":
            limit = _coerce_limit(args.get("limit"), default=10)
            data = lf.artist_get_top_tracks(artist, limit=limit)
            tracks = data.get("toptracks", {}).get("track", [])
            return _result({
                "artist": artist,
                "top_tracks": [
                    {
                        "rank": int(t.get("@attr", {}).get("rank") or i + 1),
                        "name": t.get("name", ""),
                        "playcount": int(t.get("playcount") or 0) or None,
                        "listeners": int(t.get("listeners") or 0) or None,
                        "url": t.get("url", ""),
                    }
                    for i, t in enumerate(tracks)
                ],
            })

        if action == "top_albums":
            limit = _coerce_limit(args.get("limit"), default=10)
            data = lf.artist_get_top_albums(artist, limit=limit)
            albums = data.get("topalbums", {}).get("album", [])
            return _result({
                "artist": artist,
                "top_albums": [
                    {
                        "rank": int(a.get("@attr", {}).get("rank") or i + 1),
                        "name": a.get("name", ""),
                        "playcount": int(a.get("playcount") or 0) or None,
                        "url": a.get("url", ""),
                    }
                    for i, a in enumerate(albums)
                ],
            })

        if action == "top_tags":
            data = lf.artist_get_top_tags(artist)
            tags = data.get("toptags", {}).get("tag", [])
            return _result({
                "artist": artist,
                "tags": [
                    {"name": t.get("name", ""), "count": int(t.get("count") or 0)}
                    for t in tags
                ],
            })

        if action == "search":
            limit = _coerce_limit(args.get("limit"), default=10)
            data = lf.artist_search(artist, limit=limit)
            matches = (
                data.get("results", {})
                    .get("artistmatches", {})
                    .get("artist", [])
            )
            return _result({
                "query": artist,
                "results": [
                    {
                        "name": a.get("name", ""),
                        "listeners": int(a.get("listeners") or 0) or None,
                        "url": a.get("url", ""),
                    }
                    for a in matches
                ],
            })

        return tool_error(
            f"Unknown action '{action}'. "
            "Valid: info, similar, top_tracks, top_albums, top_tags, search"
        )

    except Exception as exc:
        return _lastfm_error(exc)


# ── track ─────────────────────────────────────────────────────────────────────

def _handle_lastfm_track(args: Dict[str, Any], **_kw: Any) -> str:
    action = str(args.get("action") or "info").strip().lower()
    artist = str(args.get("artist") or "").strip()
    track = str(args.get("track") or "").strip()

    try:
        lf = _client()

        if action == "search":
            query = track or artist
            if not query:
                return tool_error("track (or artist) is required for search")
            limit = _coerce_limit(args.get("limit"), default=10)
            data = lf.track_search(query, artist=artist or None, limit=limit)
            matches = (
                data.get("results", {})
                    .get("trackmatches", {})
                    .get("track", [])
            )
            return _result({
                "query": query,
                "results": [
                    {
                        "artist": t.get("artist", ""),
                        "name": t.get("name", ""),
                        "listeners": int(t.get("listeners") or 0) or None,
                        "url": t.get("url", ""),
                    }
                    for t in matches
                ],
            })

        if not artist or not track:
            return tool_error("Both artist and track are required for this action")

        if action == "info":
            data = lf.track_get_info(artist, track)
            t = data.get("track", {})
            album = t.get("album", {})
            tags = [tg["name"] for tg in t.get("toptags", {}).get("tag", [])]
            wiki = t.get("wiki", {})
            return _result({
                "artist": t.get("artist", {}).get("name", artist) if isinstance(t.get("artist"), dict) else artist,
                "name": t.get("name", track),
                "duration_ms": int(t.get("duration") or 0) or None,
                "listeners": int(t.get("listeners") or 0) or None,
                "playcount": int(t.get("playcount") or 0) or None,
                "url": t.get("url", ""),
                "album": album.get("title", "") if album else "",
                "album_url": album.get("url", "") if album else "",
                "tags": tags,
                "summary": _clean_bio(wiki.get("summary", "")),
            })

        if action == "similar":
            limit = _coerce_limit(args.get("limit"), default=20)
            data = lf.track_get_similar(artist, track, limit=limit)
            similar = data.get("similartracks", {}).get("track", [])
            return _result({
                "artist": artist,
                "track": track,
                "similar": [
                    {
                        "artist": (t.get("artist") or {}).get("name", "") if isinstance(t.get("artist"), dict) else str(t.get("artist", "")),
                        "name": t.get("name", ""),
                        "match": round(float(t.get("match", 0)) * 100, 1),
                        "playcount": int(t.get("playcount") or 0) or None,
                        "url": t.get("url", ""),
                    }
                    for t in similar
                ],
            })

        return tool_error(
            f"Unknown action '{action}'. Valid: info, similar, search"
        )

    except Exception as exc:
        return _lastfm_error(exc)


# ── tag / genre ───────────────────────────────────────────────────────────────

def _handle_lastfm_tag(args: Dict[str, Any], **_kw: Any) -> str:
    action = str(args.get("action") or "top_artists").strip().lower()
    tag = str(args.get("tag") or "").strip()
    if not tag:
        return tool_error("tag is required (e.g. 'ambient', 'jazz', 'post-rock')")

    try:
        lf = _client()
        limit = _coerce_limit(args.get("limit"), default=20)

        if action == "info":
            data = lf.tag_get_info(tag)
            t = data.get("tag", {})
            wiki = t.get("wiki", {})
            return _result({
                "name": t.get("name", tag),
                "reach": int(t.get("reach") or 0) or None,
                "total": int(t.get("total") or 0) or None,
                "summary": _clean_bio(wiki.get("summary", "")),
            })

        if action == "top_artists":
            data = lf.tag_get_top_artists(tag, limit=limit)
            artists = data.get("topartists", {}).get("artist", [])
            return _result({
                "tag": tag,
                "top_artists": [
                    {
                        "rank": int(a.get("@attr", {}).get("rank") or i + 1),
                        "name": a.get("name", ""),
                        "url": a.get("url", ""),
                    }
                    for i, a in enumerate(artists)
                ],
            })

        if action == "top_tracks":
            data = lf.tag_get_top_tracks(tag, limit=limit)
            tracks = data.get("tracks", {}).get("track", [])
            return _result({
                "tag": tag,
                "top_tracks": [
                    {
                        "rank": int(t.get("@attr", {}).get("rank") or i + 1),
                        "artist": (t.get("artist") or {}).get("name", "") if isinstance(t.get("artist"), dict) else str(t.get("artist", "")),
                        "name": t.get("name", ""),
                        "url": t.get("url", ""),
                    }
                    for i, t in enumerate(tracks)
                ],
            })

        if action == "top_albums":
            data = lf.tag_get_top_albums(tag, limit=limit)
            albums = data.get("albums", {}).get("album", [])
            return _result({
                "tag": tag,
                "top_albums": [
                    {
                        "rank": int(a.get("@attr", {}).get("rank") or i + 1),
                        "artist": (a.get("artist") or {}).get("name", "") if isinstance(a.get("artist"), dict) else str(a.get("artist", "")),
                        "name": a.get("name", ""),
                        "url": a.get("url", ""),
                    }
                    for i, a in enumerate(albums)
                ],
            })

        if action == "similar":
            data = lf.tag_get_similar(tag)
            similar = data.get("similartags", {}).get("tag", [])
            return _result({
                "tag": tag,
                "similar_tags": [t.get("name", "") for t in similar],
            })

        return tool_error(
            f"Unknown action '{action}'. "
            "Valid: info, top_artists, top_tracks, top_albums, similar"
        )

    except Exception as exc:
        return _lastfm_error(exc)


# ── charts ────────────────────────────────────────────────────────────────────

def _handle_lastfm_charts(args: Dict[str, Any], **_kw: Any) -> str:
    action = str(args.get("action") or "top_tracks").strip().lower()
    country = str(args.get("country") or "").strip()
    limit = _coerce_limit(args.get("limit"), default=20)

    try:
        lf = _client()

        if action == "top_artists":
            if country:
                data = lf.geo_get_top_artists(country, limit=limit)
                artists = data.get("topartists", {}).get("artist", [])
            else:
                data = lf.chart_get_top_artists(limit=limit)
                artists = data.get("artists", {}).get("artist", [])
            return _result({
                "scope": country or "global",
                "top_artists": [
                    {
                        "rank": int((a.get("@attr") or {}).get("rank") or i + 1),
                        "name": a.get("name", ""),
                        "listeners": int(a.get("listeners") or 0) or None,
                        "url": a.get("url", ""),
                    }
                    for i, a in enumerate(artists)
                ],
            })

        if action == "top_tracks":
            if country:
                data = lf.geo_get_top_tracks(country, limit=limit)
                tracks = data.get("tracks", {}).get("track", [])
            else:
                data = lf.chart_get_top_tracks(limit=limit)
                tracks = data.get("tracks", {}).get("track", [])
            return _result({
                "scope": country or "global",
                "top_tracks": [
                    {
                        "rank": int((t.get("@attr") or {}).get("rank") or i + 1),
                        "artist": (t.get("artist") or {}).get("name", "") if isinstance(t.get("artist"), dict) else str(t.get("artist", "")),
                        "name": t.get("name", ""),
                        "listeners": int(t.get("listeners") or 0) or None,
                        "url": t.get("url", ""),
                    }
                    for i, t in enumerate(tracks)
                ],
            })

        return tool_error(
            f"Unknown action '{action}'. Valid: top_artists, top_tracks"
        )

    except Exception as exc:
        return _lastfm_error(exc)


# ── helpers ───────────────────────────────────────────────────────────────────

def _as_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    return [s] if s else []


def _parse_track_seed(raw: str):
    """Return (artist, track) from 'Artist:Track' or 'Artist - Track'."""
    for sep in (":", " - ", " – "):
        if sep in raw:
            a, t = raw.split(sep, 1)
            return a.strip(), t.strip()
    return None, None


def _clean_bio(text: str) -> str:
    """Strip Last.fm's appended 'Read more on Last.fm' anchor tags."""
    import re
    return re.sub(r'\s*<a[^>]*>.*?</a>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()


# ── schemas ───────────────────────────────────────────────────────────────────

_STR = {"type": "string"}
_INT = {"type": "integer"}

LASTFM_DISCOVER_SCHEMA = {
    "name": "lastfm_discover",
    "description": (
        "Find similar music based on seed artists and/or tracks using Last.fm. "
        "Accepts multiple seeds and ranks results by composite similarity score. "
        "Returns artist recommendations enriched with top tracks, genre tags, "
        "listener counts, and per-seed match details. "
        "Requires LASTFM_API_KEY environment variable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "artists": {
                "type": "array",
                "items": _STR,
                "description": "Seed artist names, e.g. ['Boards of Canada', 'Aphex Twin']",
            },
            "tracks": {
                "type": "array",
                "items": _STR,
                "description": "Seed tracks as 'Artist:Track', e.g. ['Boards of Canada:Roygbiv']",
            },
            "count": {
                **_INT,
                "description": "Number of recommendations (default: 10, max: 50)",
            },
            "scoring": {
                "type": "string",
                "enum": ["avg", "max", "boost"],
                "description": (
                    "Multi-seed scoring: "
                    "avg = sum(matches)/total_seeds (default, penalises seed-partial artists), "
                    "max = best single-seed match, "
                    "boost = avg × (1 + 0.3 × extra_seeds) cross-seed bonus"
                ),
            },
            "tags": {
                **_INT,
                "description": "Number of genre tags to include per result (default: 4, 0 to omit)",
            },
        },
        "required": [],
    },
}

LASTFM_ARTIST_SCHEMA = {
    "name": "lastfm_artist",
    "description": (
        "Get Last.fm metadata for an artist: biography, similar artists, "
        "top tracks, top albums, genre tags, or search by name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["info", "similar", "top_tracks", "top_albums", "top_tags", "search"],
                "description": (
                    "info: biography, stats, tags; "
                    "similar: artists similar to this one; "
                    "top_tracks: most played tracks; "
                    "top_albums: most played albums; "
                    "top_tags: genre/mood tags; "
                    "search: find artists matching a name query"
                ),
            },
            "artist": {**_STR, "description": "Artist name"},
            "limit": {**_INT, "description": "Result count (default: 10 for tracks/albums, 20 for similar)"},
        },
        "required": ["action", "artist"],
    },
}

LASTFM_TRACK_SCHEMA = {
    "name": "lastfm_track",
    "description": (
        "Get Last.fm metadata for a track: info (playcount, album, tags, wiki), "
        "similar tracks, or search tracks by name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["info", "similar", "search"],
                "description": (
                    "info: track details, playcount, album, tags, wiki summary; "
                    "similar: tracks similar to this one (requires artist + track); "
                    "search: find tracks by name (artist optional)"
                ),
            },
            "artist": {**_STR, "description": "Artist name (required for info and similar)"},
            "track": {**_STR, "description": "Track name"},
            "limit": {**_INT, "description": "Result count for search/similar (default: 10–20)"},
        },
        "required": ["action"],
    },
}

LASTFM_TAG_SCHEMA = {
    "name": "lastfm_tag",
    "description": (
        "Explore music by genre, mood, or era using Last.fm tags. "
        "Get top artists, tracks, or albums for a tag, find similar tags, "
        "or read a tag's description. Examples: 'ambient', 'jazz', 'post-rock', '80s'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["info", "top_artists", "top_tracks", "top_albums", "similar"],
                "description": (
                    "info: tag description and usage stats; "
                    "top_artists: most-tagged artists for this genre; "
                    "top_tracks: most-tagged tracks; "
                    "top_albums: most-tagged albums; "
                    "similar: related genre/mood tags"
                ),
            },
            "tag": {**_STR, "description": "Tag name, e.g. 'ambient', 'jazz', 'post-rock', 'shoegaze'"},
            "limit": {**_INT, "description": "Result count (default: 20)"},
        },
        "required": ["action", "tag"],
    },
}

LASTFM_CHARTS_SCHEMA = {
    "name": "lastfm_charts",
    "description": (
        "Get trending music charts from Last.fm — top artists or top tracks, "
        "globally or filtered by country."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["top_artists", "top_tracks"],
                "description": "top_artists: trending artists; top_tracks: trending tracks",
            },
            "country": {
                **_STR,
                "description": (
                    "Optional country name for geo charts, e.g. 'United Kingdom', 'Germany'. "
                    "Omit for global charts."
                ),
            },
            "limit": {**_INT, "description": "Number of results (default: 20, max: 100)"},
        },
        "required": ["action"],
    },
}
