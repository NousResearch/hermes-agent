"""Backend for the bundled Radio desktop player.

The renderer cannot read most stream metadata or directory pages directly:
stations rarely send CORS headers, and Shoutcast's directory answers no
preflight. These routes perform those requests from the Hermes backend.

Every outbound address is checked with ``tools.url_safety.is_safe_url`` before
the request. Redirects are re-checked in an httpx response hook, and requests
use ``create_ssrf_safe_client`` so validation also happens at TCP connect time.
Only http and https targets without userinfo are accepted.
"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, HTTPException, Query
from hermes_constants import get_hermes_home
from pydantic import BaseModel
from tools.url_safety import create_ssrf_safe_client, is_safe_url

router = APIRouter()

_DIRECTORY = "https://directory.shoutcast.com"
_TUNE_IN = "http://yp.shoutcast.com/sbin/tunein-station.pls?id={station_id}"
_TUNEIN_SEARCH = "https://opml.radiotime.com/Search.ashx"
_USER_AGENT = "Mozilla/5.0 (compatible; HermesAgent/1.0; RadioBackend/1.0)"
_TIMEOUT = 12.0
_META_TIMEOUT = 6.0
_STREAM_TIMEOUT = 7.0
_PROBE_TIMEOUT = 2.5
_RESOLVE_WORKERS = 12
_PROBE_WORKERS = 24
_CACHE_TTL = 900.0
_QUERY_TTL = 300.0
_NOW_PLAYING_TTL = 12.0

_query_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_stream_cache: dict[int | str, tuple[float, str]] = {}
_now_playing_cache: dict[str, tuple[float, str]] = {}
_probe_cache: dict[str, tuple[float, dict[str, Any]]] = {}


class UnsafeRedirect(Exception):
    """A redirect target failed the same URL policy as the original address."""


def _freshest(cache: dict[Any, tuple[float, Any]], key: Any, ttl: float) -> Any:
    entry = cache.get(key)
    if not entry:
        return None
    stamp, value = entry
    if time.time() - stamp > ttl:
        cache.pop(key, None)
        return None
    return value


def _clean_text(value: Any, limit: int = 160) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(re.sub(r"<[^>]{0,64}>", " ", html.unescape(value)).split())
    return text[:limit]


def _clean_stream(value: Any) -> str | None:
    """Accept an absolute http(s) address with no userinfo and a real host."""
    candidate = str(value or "").strip()
    if not candidate.lower().startswith(("http://", "https://")):
        return None
    try:
        parsed = urllib.parse.urlsplit(candidate)
    except ValueError:
        return None
    if not parsed.hostname or parsed.username or parsed.password:
        return None
    try:
        candidate.encode("ascii")
    except UnicodeEncodeError:
        return None
    return candidate


def _literal_host_blocked(host: str) -> bool:
    """Reject literal loopback, private, link-local and IPv6 ULA hosts up front."""
    try:
        address = ipaddress.ip_address(host.strip("[]").split("%", 1)[0])
    except ValueError:
        return host.lower() in {
            "localhost",
            "metadata.google.internal",
            "metadata.goog",
        }
    return (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
        or address.is_multicast
        or address.is_reserved
    )


def _safe_target(url: Any) -> str | None:
    """Syntax check, literal-address check, then DNS-resolving URL safety check."""
    candidate = _clean_stream(url)
    if not candidate:
        return None
    host = urllib.parse.urlsplit(candidate).hostname or ""
    if _literal_host_blocked(host):
        return None
    return candidate if is_safe_url(candidate) else None


def _redirect_target(response: Any) -> str | None:
    """Read the redirect target from the response inside an httpx response hook."""
    if not getattr(response, "is_redirect", False):
        return None
    location = (getattr(response, "headers", {}) or {}).get("location")
    if not location:
        next_request = getattr(response, "next_request", None)
        return str(next_request.url) if next_request else None
    return urljoin(str(getattr(response, "url", "")), str(location))


def _guard_response(response: Any) -> None:
    target = _redirect_target(response)
    if target and not _safe_target(target):
        raise UnsafeRedirect(target)


def _client(timeout: float) -> Any:
    return create_ssrf_safe_client(
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"response": [_guard_response]},
        headers={"User-Agent": _USER_AGENT},
    )


@contextmanager
def _safe_stream(
    url: Any, *, timeout: float, headers: dict[str, str] | None = None
) -> Iterator[Any]:
    address = _safe_target(url)
    if not address:
        raise HTTPException(
            status_code=400, detail="That address is not a safe http or https target."
        )
    try:
        with _client(timeout) as client:
            with client.stream("GET", address, headers=headers) as response:
                yield response
    except UnsafeRedirect as error:
        raise HTTPException(
            status_code=400, detail="The address redirected to an unsafe target."
        ) from error
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"The address could not be reached ({type(error).__name__}).",
        ) from error


def _request(method: str, url: Any, *, timeout: float = _TIMEOUT, **kwargs: Any) -> Any:
    address = _safe_target(url)
    if not address:
        raise HTTPException(
            status_code=400, detail="That address is not a safe http or https target."
        )
    try:
        with _client(timeout) as client:
            return client.request(method, address, **kwargs)
    except UnsafeRedirect as error:
        raise HTTPException(
            status_code=400, detail="The address redirected to an unsafe target."
        ) from error
    except (httpx.HTTPError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"The address could not be reached ({type(error).__name__}).",
        ) from error


def _read_prefix(response: Any, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit
    for chunk in response.iter_bytes():
        chunks.append(chunk[:remaining])
        remaining -= len(chunks[-1])
        if remaining <= 0:
            break
    return b"".join(chunks)


def _query_directory(path: str, query: str | None) -> list[dict[str, Any]]:
    key = f"{path}\u0000{query or ''}".lower()
    cached = _freshest(_query_cache, key, _QUERY_TTL)
    if cached is not None:
        return cached
    headers = {"X-Requested-With": "XMLHttpRequest"}
    if query:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        body: bytes = urllib.parse.urlencode({"query": query}).encode()
    else:
        body = b""
    response = _request(
        "POST", f"{_DIRECTORY}{path}", timeout=_TIMEOUT, content=body, headers=headers
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Shoutcast directory returned HTTP {response.status_code}.",
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail="Shoutcast directory returned no readable station list.",
        ) from error
    rows = (
        [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, list)
        else []
    )
    _query_cache[key] = (time.time(), rows)
    return rows


def _first_stream(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip().lower().startswith("file"):
            address = _clean_stream(value.strip())
            if address:
                return address
            continue
        address = _clean_stream(line)
        if address:
            return address
    return None


def _stream_for(station_id: int) -> str | None:
    cached = _freshest(_stream_cache, station_id, _CACHE_TTL)
    if cached is not None:
        return cached or None
    address: str | None = None
    try:
        with _safe_stream(
            _TUNE_IN.format(station_id=station_id), timeout=_STREAM_TIMEOUT
        ) as response:
            text = (
                response.read().decode("utf-8", "replace")
                if hasattr(response, "read")
                else _read_prefix(response, 65_536).decode("utf-8", "replace")
            )
        address = _first_stream(text)
    except (HTTPException, UnicodeError, ValueError):
        address = None
    _stream_cache[station_id] = (time.time(), address or "")
    return address


def _station_id(row: dict[str, Any]) -> int | None:
    raw = row.get("ID")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _station(row: dict[str, Any], url: str) -> dict[str, Any]:
    identifier = _station_id(row) or 0
    genre = _clean_text(row.get("Genre"), 60)
    raw_bitrate = row.get("Bitrate")
    raw_listeners = row.get("Listeners")
    bitrate = raw_bitrate if isinstance(raw_bitrate, int) else 0
    listeners = raw_listeners if isinstance(raw_listeners, int) else 0
    parts = [genre] if genre else []
    if bitrate > 0:
        parts.append(f"{bitrate} kbps")
    return {
        "id": f"sc-{identifier}",
        "name": _clean_text(row.get("Name"), 120) or f"Shoutcast {identifier}",
        "url": url,
        "provider": "Shoutcast",
        "description": " · ".join(parts),
        "nowPlaying": _clean_text(row.get("CurrentTrack"), 120),
        "homepage": _clean_stream(_clean_text(row.get("IceUrl"), 400)) or "",
        "listeners": listeners,
        "bitrate": bitrate,
    }


_STREAM_TITLE = re.compile(r"StreamTitle='(.*?)';", re.DOTALL)


def _stream_title(block: bytes) -> str:
    if not block:
        return ""
    match = _STREAM_TITLE.search(block.decode("utf-8", "replace"))
    return _clean_text(match.group(1), 160) if match else ""


def _icy_title(address: str) -> dict[str, Any]:
    target = _safe_target(address)
    if not target:
        return {
            "ok": False,
            "title": "",
            "station": "",
            "reason": "That is not a safe http or https target.",
        }
    try:
        with _safe_stream(
            target,
            timeout=_META_TIMEOUT,
            headers={"Icy-MetaData": "1", "Accept": "*/*"},
        ) as response:
            station = _clean_text(response.headers.get("icy-name", ""), 80)
            metaint = _count(response.headers.get("icy-metaint"))
            if not metaint or metaint > 65536:
                return {
                    "ok": False,
                    "title": "",
                    "station": station,
                    "reason": "This stream does not announce track titles.",
                }
            _read_prefix(response, metaint)
            size = _read_prefix(response, 1)
            length = (size[0] if size else 0) * 16
            title = _stream_title(_read_prefix(response, length) if length else b"")
    except HTTPException as error:
        return {"ok": False, "title": "", "station": "", "reason": str(error.detail)}
    return {
        "ok": bool(title),
        "title": title,
        "station": station,
        "reason": "" if title else "The stream is between tracks.",
    }


def _tunein_rows(query: str) -> list[dict[str, Any]]:
    key = f"tunein\u0000{query}".lower()
    cached = _freshest(_query_cache, key, _QUERY_TTL)
    if cached is not None:
        return cached
    params = {
        "query": query,
        "render": "json",
        "formats": "mp3,aac",
        "types": "station",
    }
    response = _request(
        "GET", f"{_TUNEIN_SEARCH}?{urllib.parse.urlencode(params)}", timeout=_TIMEOUT
    )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=502, detail=f"TuneIn returned HTTP {response.status_code}."
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise HTTPException(
            status_code=502, detail="TuneIn returned no readable station list."
        ) from error
    head = payload.get("head") if isinstance(payload, dict) else None
    status = str(head.get("status")) if isinstance(head, dict) else ""
    if status and status != "200":
        raise HTTPException(status_code=502, detail=f"TuneIn answered {status}.")
    rows = [
        row
        for row in ((payload.get("body") if isinstance(payload, dict) else None) or [])
        if isinstance(row, dict)
        and row.get("type") == "audio"
        and _clean_stream(row.get("URL"))
    ]
    _query_cache[key] = (time.time(), rows)
    return rows


def _tunein_stream(url: str) -> str | None:
    tunein_url = _clean_stream(url)
    if not tunein_url:
        return None
    cached = _freshest(_stream_cache, tunein_url, _CACHE_TTL)
    if cached is not None:
        return cached or None
    address: str | None = None
    try:
        with _safe_stream(tunein_url, timeout=_STREAM_TIMEOUT) as response:
            content_type = (
                (response.headers.get("content-type", "")).split(";")[0].strip().lower()
            )
            final = (
                _clean_stream(str(getattr(response, "url", tunein_url))) or tunein_url
            )
            text = _read_prefix(response, 65_536).decode("utf-8", "replace")
        if "#EXT-X-" in text[:1024]:
            address = final
        elif content_type in {
            "audio/x-mpegurl",
            "audio/mpegurl",
            "audio/x-scpls",
            "application/pls+xml",
        }:
            address = _first_stream(text)
        elif (
            content_type.startswith("audio/")
            or content_type == "application/octet-stream"
        ):
            address = final
        else:
            address = _first_stream(text)
    except (HTTPException, UnicodeError, ValueError):
        address = None
    _stream_cache[tunein_url] = (time.time(), address or "")
    return address


def _tunein_station(row: dict[str, Any], url: str) -> dict[str, Any]:
    guide = _clean_text(row.get("guide_id"), 40) or "station"
    bitrate = _count(row.get("bitrate"))
    formats = _clean_text(row.get("formats"), 20)
    parts = [f"{bitrate} kbps"] if bitrate else []
    if formats:
        parts.append(formats.upper())
    return {
        "id": f"ti-{guide}",
        "name": _clean_text(row.get("text"), 120) or "TuneIn station",
        "url": url,
        "provider": "TuneIn",
        "description": " · ".join(parts),
        "nowPlaying": _clean_text(row.get("playing"), 120)
        or _clean_text(row.get("subtext"), 120),
        "homepage": "",
        "listeners": 0,
        "bitrate": bitrate,
    }


def _probe(url: str) -> dict[str, Any]:
    address = _safe_target(url)
    if not address:
        return {
            "ok": False,
            "kind": "invalid",
            "status": 0,
            "contentType": "",
            "reason": "That is not a safe http or https address.",
        }
    try:
        with _safe_stream(
            address,
            timeout=_PROBE_TIMEOUT,
            headers={"Icy-MetaData": "1", "Accept": "*/*"},
        ) as response:
            content_type = (
                (response.headers.get("content-type", "")).split(";")[0].strip().lower()
            )
            name = _clean_text(response.headers.get("icy-name", ""), 80)
            body = _read_prefix(response, 2048)
            status = response.status_code
    except HTTPException as error:
        kind = "http-error" if "HTTP" in str(error.detail) else "unreachable"
        return {
            "ok": False,
            "kind": kind,
            "status": 0,
            "contentType": "",
            "reason": str(error.detail),
        }
    lowered = address.lower()
    if "mpegurl" in content_type or lowered.endswith(".m3u8") or body[:7] == b"#EXTM3U":
        kind = "hls"
    elif content_type.startswith("audio/") or content_type in {
        "application/ogg",
        "video/ogg",
        "application/octet-stream",
    }:
        kind = "audio"
    elif content_type in {
        "audio/x-scpls",
        "audio/mpegurl",
        "application/pls+xml",
    } or lowered.endswith((".pls", ".m3u")):
        kind = "playlist"
    elif (
        content_type.startswith("text/html")
        or body[:15].lower().lstrip().startswith(b"<!doctype")
        or body[:5].lower() == b"<html"
    ):
        kind = "html"
    elif content_type.startswith("text/"):
        kind = "text"
    else:
        kind = "other"
    reasons = {
        "audio": "The address returns audio.",
        "hls": "This is an HLS playlist.",
        "playlist": "This is a playlist file, not a stream address.",
        "html": "This is a web page, not an audio stream.",
        "text": "This address returns text, not audio.",
        "other": f"This address returns {content_type or 'an unknown type'}, not audio.",
    }
    return {
        "ok": kind in {"audio", "hls"},
        "kind": kind,
        "status": status,
        "contentType": content_type,
        "name": name,
        "reason": reasons[kind],
    }


def _probe_cached(address: str) -> dict[str, Any]:
    cached = _freshest(_probe_cache, address, _CACHE_TTL)
    if cached is not None:
        return cached
    verdict = _probe(address)
    _probe_cache[address] = (time.time(), verdict)
    return verdict


def _probe_many(addresses: list[str], budget: float = 2.0) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for address in dict.fromkeys(addresses):
        cached = _freshest(_probe_cache, address, _CACHE_TTL)
        if cached is not None:
            verdicts[address] = cached
        else:
            pending.append(address)
    if not pending:
        return verdicts
    pool = ThreadPoolExecutor(max_workers=min(_PROBE_WORKERS, len(pending)))
    futures = {pool.submit(_probe_cached, address): address for address in pending}
    done, not_done = wait(futures, timeout=budget)
    for future in done:
        address = futures[future]
        try:
            verdicts[address] = future.result()
        except Exception:
            verdicts[address] = {
                "ok": False,
                "kind": "unreachable",
                "status": 0,
                "contentType": "",
                "reason": "The probe failed.",
            }
    for future in not_done:
        future.cancel()
    pool.shutdown(wait=False)
    return verdicts


def _log_path() -> Path:
    """The station log lives under the active Hermes home, resolved on every call."""
    return get_hermes_home() / "plugin-data" / "radio" / "station-log.jsonl"


def _append_log(entry: dict[str, Any]) -> None:
    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 1_000_000:
            path.replace(path.with_suffix(".jsonl.1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True)[:2000] + "\n")
    except OSError:
        return


def _page(
    candidates: list[dict[str, Any]],
    total: int,
    offset: int,
    resolve: Callable[[dict[str, Any]], str | None],
    build: Callable[[dict[str, Any], str], dict[str, Any]],
    source: str,
    sort_key: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    if not candidates:
        return {"stations": [], "skipped": 0, "count": total, "next": total}
    with ThreadPoolExecutor(max_workers=min(_RESOLVE_WORKERS, len(candidates))) as pool:
        addresses = list(pool.map(resolve, candidates))
    resolved = [
        (row, address) for row, address in zip(candidates, addresses) if address
    ]
    skipped = len(candidates) - len(resolved)
    if not resolved:
        return {
            "stations": [],
            "skipped": skipped,
            "count": total,
            "next": offset + len(candidates),
        }
    verdicts = _probe_many([address for _, address in resolved])
    stations: list[dict[str, Any]] = []
    logged = 0
    for row, address in resolved:
        station = build(row, address)
        verdict = verdicts.get(address) or {
            "ok": False,
            "kind": "slow",
            "reason": "No answer within the probe budget.",
        }
        if not verdict["ok"]:
            skipped += 1
            if logged < 10:
                logged += 1
                _append_log(
                    {
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "source": source,
                        "name": station["name"],
                        "url": address,
                        "reason": f"{verdict['kind']}: {verdict['reason']}",
                    }
                )
            continue
        stations.append(station)
    stations.sort(key=sort_key, reverse=True)
    return {
        "stations": stations,
        "skipped": skipped,
        "count": total,
        "next": offset + len(candidates),
    }


def _stations(
    rows: Iterable[dict[str, Any]], limit: int, offset: int = 0
) -> dict[str, Any]:
    usable = [row for row in rows if _station_id(row) is not None]
    return _page(
        usable[offset : offset + limit],
        len(usable),
        offset,
        lambda row: _stream_for(_station_id(row) or 0),
        _station,
        "shoutcast-resolve",
        lambda station: station.get("listeners") or 0,
    )


def _tunein_page(
    rows: list[dict[str, Any]], limit: int, offset: int = 0
) -> dict[str, Any]:
    return _page(
        rows[offset : offset + limit],
        len(rows),
        offset,
        lambda row: _tunein_stream(str(row.get("URL") or "")),
        _tunein_station,
        "tunein-resolve",
        lambda station: station.get("bitrate") or 0,
    )


@router.get("/shoutcast/search")
def shoutcast_search(
    q: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(40, ge=1, le=60),
    offset: int = 0,
) -> dict[str, Any]:
    term = " ".join(q.split())[:80]
    if not term:
        raise HTTPException(status_code=400, detail="A search term is required.")
    return {
        "source": "shoutcast",
        "query": term,
        **_stations(_query_directory("/Search/UpdateSearch", term), limit, offset),
    }


@router.get("/shoutcast/top")
def shoutcast_top(
    limit: int = Query(40, ge=1, le=60), offset: int = 0
) -> dict[str, Any]:
    return {
        "source": "shoutcast",
        "query": "",
        **_stations(_query_directory("/Home/Top", None), limit, offset),
    }


@router.get("/tunein/search")
def tunein_search(
    q: str = Query(..., min_length=1, max_length=80),
    limit: int = Query(30, ge=1, le=60),
    offset: int = 0,
) -> dict[str, Any]:
    term = " ".join(q.split())[:80]
    if not term:
        raise HTTPException(status_code=400, detail="A search term is required.")
    return {
        "source": "tunein",
        "query": term,
        **_tunein_page(_tunein_rows(term), limit, offset),
    }


@router.get("/now-playing")
def now_playing(url: str = Query(..., min_length=8, max_length=1000)) -> dict[str, Any]:
    address = _clean_stream(url)
    if not address:
        raise HTTPException(
            status_code=400, detail="That is not an http or https address."
        )
    cached = _freshest(_now_playing_cache, address, _NOW_PLAYING_TTL)
    if cached is not None:
        return {
            "ok": bool(cached),
            "title": cached,
            "station": "",
            "reason": "" if cached else "No title announced.",
        }
    result = _icy_title(address)
    _now_playing_cache[address] = (time.time(), result["title"])
    return result


@router.get("/probe")
def probe_stream(url: str = Query(..., min_length=8, max_length=600)) -> dict[str, Any]:
    return _probe(url)


class StationLogEntry(BaseModel):
    name: str = ""
    url: str = ""
    reason: str = ""
    at: str = ""


@router.post("/log")
def write_station_log(payload: StationLogEntry) -> dict[str, str]:
    entry = {
        "at": payload.at or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "player",
        "name": _clean_text(payload.name, 120),
        "url": _clean_stream(_clean_text(payload.url, 400)) or "",
        "reason": _clean_text(payload.reason, 300),
    }
    _append_log(entry)
    return {"status": "logged", "path": str(_log_path())}
