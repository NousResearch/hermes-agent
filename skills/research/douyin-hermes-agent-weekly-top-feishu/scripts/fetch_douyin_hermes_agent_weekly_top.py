#!/usr/bin/env python3
"""Fetch this week's top Douyin videos about Hermes Agent.

The script uses Douyin's web video-search endpoint and emits JSON for Hermes
cron. Douyin currently requires a logged-in browser cookie for search results,
so production runs should provide DOUYIN_COOKIE or DOUYIN_SEARCH_COOKIE.
"""

from __future__ import annotations

import argparse
import configparser
import html
import json
import os
import random
import re
import shutil
import sqlite3
import string
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DOUYIN_SEARCH_API = "https://www.douyin.com/aweme/v1/web/search/item/"
REFERER_TEMPLATE = "https://www.douyin.com/search/{keyword}?type=video"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
)
RUNTIME_SOURCE = "Douyin web video search API"
CN_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")
DEFAULT_KEYWORDS = ("HermesAgent", "Hermes Agent", "hermes-agent")
DEFAULT_SORT_TYPES = ("0", "1", "2")
DEFAULT_PUBLISH_TIME = "7"
DEFAULT_FIREFOX_COOKIE_DOMAINS = ("douyin.com",)
DEFAULT_BROWSER_PARAMS = {
    "pc_client_type": "1",
    "publish_video_strategy_type": "2",
    "pc_libra_divert": "Windows",
    "version_code": "290100",
    "version_name": "29.1.0",
    "cookie_enabled": "true",
    "screen_width": "1920",
    "screen_height": "1080",
    "browser_language": "zh-CN",
    "browser_platform": "Win32",
    "browser_name": "Edge",
    "browser_version": "130.0.0.0",
    "browser_online": "true",
    "engine_name": "Blink",
    "engine_version": "130.0.0.0",
    "os_name": "Windows",
    "os_version": "10",
    "cpu_core_num": "12",
    "device_memory": "8",
    "platform": "PC",
    "downlink": "10",
    "effective_type": "4g",
    "round_trip_time": "100",
}
WEBID_API = "https://mcs.zijieapi.com/webid?aid=6383&sdk_version=5.1.18_zip&device_platform=web"
DEFAULT_VERIFY_RETRIES = 3


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_runtime_env() -> None:
    if os.getenv("DOUYIN_SKIP_HERMES_ENV_LOAD", "").lower() in {"1", "true", "yes", "on"}:
        return

    candidates: list[Path] = []
    hermes_home = os.getenv("HERMES_HOME", "").strip()
    if hermes_home:
        candidates.append(Path(hermes_home).expanduser() / ".env")
    candidates.append(Path.home() / ".hermes" / ".env")

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(path)


load_runtime_env()


def split_env_list(value: str) -> list[str]:
    return [part.strip() for part in re.split(rf"[;,{re.escape(os.pathsep)}]", value) if part.strip()]


def explicit_cookie_header() -> str | None:
    for key in ("DOUYIN_COOKIE", "DOUYIN_SEARCH_COOKIE", "DOUYIN_WEB_COOKIE"):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def firefox_profile_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("DOUYIN_FIREFOX_ROOT", "DOUYIN_FIREFOX_ROOTS"):
        raw = os.getenv(key, "").strip()
        if raw:
            roots.extend(Path(part).expanduser() for part in split_env_list(raw))

    home = Path.home()
    roots.extend(
        [
            home / ".mozilla" / "firefox",
            home / "snap" / "firefox" / "common" / ".mozilla" / "firefox",
            home / ".var" / "app" / "org.mozilla.firefox" / ".mozilla" / "firefox",
        ]
    )

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen:
            unique_roots.append(root)
            seen.add(resolved)
    return unique_roots


def profile_matches_hint(profile_path: Path, profile_name: str, profile_ini_path: str, hint: str | None) -> bool:
    if not hint:
        return False
    needle = hint.strip().lower()
    if not needle:
        return False
    return needle in {
        profile_name.lower(),
        profile_ini_path.lower(),
        profile_path.name.lower(),
        str(profile_path).lower(),
    } or needle in profile_path.name.lower()


def firefox_profiles_from_ini(root: Path, hint: str | None = None) -> list[Path]:
    ini_path = root / "profiles.ini"
    if not ini_path.exists():
        return []

    parser = configparser.ConfigParser()
    try:
        parser.read(ini_path, encoding="utf-8")
    except configparser.Error:
        return []

    hinted: list[Path] = []
    defaulted: list[Path] = []
    others: list[Path] = []
    for section in parser.sections():
        if not section.lower().startswith("profile"):
            continue
        profile_ini_path = parser.get(section, "Path", fallback="").strip()
        if not profile_ini_path:
            continue
        is_relative = parser.get(section, "IsRelative", fallback="1").strip() != "0"
        profile_path = root / profile_ini_path if is_relative else Path(profile_ini_path).expanduser()
        if not (profile_path / "cookies.sqlite").exists():
            continue
        profile_name = parser.get(section, "Name", fallback="").strip()
        if profile_matches_hint(profile_path, profile_name, profile_ini_path, hint):
            hinted.append(profile_path)
        elif parser.get(section, "Default", fallback="0").strip() == "1":
            defaulted.append(profile_path)
        else:
            others.append(profile_path)

    return hinted or defaulted or others


def find_firefox_profile(hint: str | None = None) -> Path | None:
    raw_hint = (hint or os.getenv("DOUYIN_FIREFOX_PROFILE") or os.getenv("FIREFOX_PROFILE") or "").strip()
    if raw_hint:
        hinted_path = Path(raw_hint).expanduser()
        if hinted_path.is_file() and hinted_path.name == "cookies.sqlite":
            return hinted_path.parent
        if (hinted_path / "cookies.sqlite").exists():
            return hinted_path

    for root in firefox_profile_roots():
        if not root.exists():
            continue
        profiles = firefox_profiles_from_ini(root, raw_hint or None)
        if profiles:
            return profiles[0]

        candidates = [path for path in root.iterdir() if path.is_dir() and (path / "cookies.sqlite").exists()]
        if raw_hint:
            candidates = [path for path in candidates if profile_matches_hint(path, "", path.name, raw_hint)]
        if candidates:
            candidates.sort(key=lambda path: (path / "cookies.sqlite").stat().st_mtime, reverse=True)
            return candidates[0]
    return None


def find_firefox_cookies_db() -> Path | None:
    explicit_db = os.getenv("DOUYIN_FIREFOX_COOKIES_SQLITE", "").strip()
    if explicit_db:
        path = Path(explicit_db).expanduser()
        if path.exists():
            return path
    profile = find_firefox_profile()
    if not profile:
        return None
    db_path = profile / "cookies.sqlite"
    return db_path if db_path.exists() else None


def firefox_cookie_domains() -> list[str]:
    raw = os.getenv("DOUYIN_FIREFOX_COOKIE_DOMAINS", "").strip()
    if not raw:
        return list(DEFAULT_FIREFOX_COOKIE_DOMAINS)
    domains = [part.lstrip(".").lower() for part in split_env_list(raw)]
    return domains or list(DEFAULT_FIREFOX_COOKIE_DOMAINS)


def copy_firefox_cookie_db(source_db: Path, target_dir: Path) -> Path:
    target_db = target_dir / "cookies.sqlite"
    shutil.copy2(source_db, target_db)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(source_db) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, Path(str(target_db) + suffix))
    return target_db


def read_firefox_cookie_rows(db_path: Path, domains: list[str]) -> list[tuple[str, str]]:
    clauses: list[str] = []
    params: list[Any] = []
    for domain in domains:
        suffix = f".{domain.lstrip('.').lower()}"
        clauses.append("(lower(host) = ? OR lower(host) = ? OR lower(host) LIKE ?)")
        params.extend([domain, suffix, f"%{suffix}"])
    domain_filter = " OR ".join(clauses)
    if not domain_filter:
        return []

    query = f"""
        SELECT name, value
        FROM moz_cookies
        WHERE (expiry = 0 OR expiry > ?)
          AND ({domain_filter})
        ORDER BY host, path, name
    """

    with tempfile.TemporaryDirectory(prefix="douyin-firefox-cookies-") as tmp_dir:
        copied_db = copy_firefox_cookie_db(db_path, Path(tmp_dir))
        conn = sqlite3.connect(f"file:{copied_db}?mode=ro", uri=True)
        try:
            rows = conn.execute(query, [int(time.time()), *params]).fetchall()
        finally:
            conn.close()
    return [(str(name), str(value)) for name, value in rows if name and value]


def firefox_cookie_header() -> str | None:
    db_path = find_firefox_cookies_db()
    if not db_path:
        return None
    try:
        rows = read_firefox_cookie_rows(db_path, firefox_cookie_domains())
    except (OSError, sqlite3.Error, shutil.Error):
        return None

    parts: list[str] = []
    seen_names: set[str] = set()
    for name, value in rows:
        if name in seen_names:
            continue
        parts.append(f"{name}={value}")
        seen_names.add(name)
    return "; ".join(parts) if parts else None


def cookie_header() -> str | None:
    return explicit_cookie_header() or firefox_cookie_header()


def cookie_values(name: str) -> list[str]:
    cookie = cookie_header() or ""
    prefix = f"{name}="
    values: list[str] = []
    for part in cookie.split("; "):
        if part.startswith(prefix):
            values.append(part[len(prefix) :])
    return values


def request_headers(keyword: str | None = None) -> dict[str, str]:
    encoded_keyword = urllib.parse.quote(keyword or "Hermes Agent")
    headers = {
        "User-Agent": os.getenv("DOUYIN_USER_AGENT", USER_AGENT),
        "Referer": REFERER_TEMPLATE.format(keyword=encoded_keyword),
        "Origin": "https://www.douyin.com",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    cookie = cookie_header()
    if cookie:
        headers["Cookie"] = cookie
    return headers


def browser_search_params() -> dict[str, str]:
    params = dict(DEFAULT_BROWSER_PARAMS)
    for key in tuple(params):
        env_key = f"DOUYIN_{key.upper()}"
        value = os.getenv(env_key, "").strip()
        if value:
            params[key] = value
    return params


def false_ms_token(length: int = 126) -> str:
    alphabet = string.ascii_letters + string.digits + "-_"
    return "".join(random.choice(alphabet) for _ in range(length)) + "=="


def douyin_ms_token() -> str:
    explicit = os.getenv("DOUYIN_MSTOKEN", "").strip() or os.getenv("DOUYIN_MS_TOKEN", "").strip()
    if explicit:
        return explicit
    values = cookie_values("msToken")
    if values:
        return values[-1]
    return false_ms_token()


def douyin_webid(timeout: int = 5) -> str | None:
    explicit = os.getenv("DOUYIN_WEBID", "").strip()
    if explicit:
        return explicit

    payload = json.dumps(
        {
            "app_id": 6383,
            "referer": "https://www.douyin.com/",
            "url": "https://www.douyin.com/",
            "user_agent": request_headers().get("User-Agent", USER_AGENT),
            "user_unique_id": "",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        WEBID_API,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "User-Agent": request_headers().get("User-Agent", USER_AGENT),
            "Referer": "https://www.douyin.com/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        web_id = json.loads(body).get("web_id")
    except Exception:
        return None
    return str(web_id).strip() if web_id else None


def tracking_search_params() -> dict[str, str]:
    params = {"msToken": douyin_ms_token()}
    webid = douyin_webid()
    if webid:
        params["webid"] = webid
    return params


def verify_retry_limit() -> int:
    raw_value = os.getenv("DOUYIN_VERIFY_RETRIES", "").strip()
    if not raw_value:
        return DEFAULT_VERIFY_RETRIES
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_VERIFY_RETRIES


def fetch_text(url: str, *, keyword: str, timeout: int = 15, retries: int = 2) -> str:
    last_error: Exception | None = None
    headers = request_headers(keyword)

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                text = response.read().decode(charset, errors="replace")
                if text.strip():
                    return text
        except Exception as exc:  # noqa: BLE001 - diagnostics are returned to cron JSON
            last_error = exc
            time.sleep(0.35 * (attempt + 1))

    curl = shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "-sS",
            "--max-time",
            str(timeout),
            "-A",
            headers["User-Agent"],
            "-e",
            headers["Referer"],
            "-H",
            "Accept: application/json, text/plain, */*",
            "-H",
            "Accept-Language: zh-CN,zh;q=0.9,en;q=0.8",
            "-H",
            "Origin: https://www.douyin.com",
        ]
        cookie = headers.get("Cookie")
        if cookie:
            cmd.extend(["-H", f"Cookie: {cookie}"])
        cmd.append(url)
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout + 5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
        if result.stderr.strip():
            raise RuntimeError(f"curl failed for {url}: {result.stderr.strip()}") from last_error

    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def fetch_json(url: str, *, keyword: str, timeout: int = 15, retries: int = 2) -> dict[str, Any]:
    text = fetch_text(url, keyword=keyword, timeout=timeout, retries=retries)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        prefix = text.strip().replace("\n", " ")[:160]
        raise RuntimeError(f"non-json response: {prefix!r}") from exc

    status_code = payload.get("status_code")
    if status_code not in (None, 0):
        status_msg = payload.get("status_msg") or payload.get("message") or "unknown error"
        raise RuntimeError(f"Douyin API returned status_code={status_code}: {status_msg}")
    return payload


def build_search_url(
    keyword: str,
    *,
    sort_type: str,
    offset: int,
    count: int,
    publish_time: str,
    tracking_params: dict[str, str] | None = None,
) -> str:
    params = {
        "device_platform": "webapp",
        "aid": "6383",
        "channel": "channel_pc_web",
        "search_channel": "aweme_video_web",
        "keyword": keyword,
        "search_source": "normal_search",
        "query_correct_type": "1",
        "is_filter_search": "1",
        "offset": str(offset),
        "count": str(count),
        "sort_type": str(sort_type),
        "publish_time": str(publish_time),
    }
    params.update(browser_search_params())
    if tracking_params:
        params.update({key: value for key, value in tracking_params.items() if value})
    return f"{DOUYIN_SEARCH_API}?{urllib.parse.urlencode(params)}"


def parse_now(value: str | None = None) -> datetime:
    if not value:
        return datetime.now(CN_TZ)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CN_TZ)
    return parsed.astimezone(CN_TZ)


def week_window(now: datetime) -> tuple[datetime, datetime]:
    local_now = now.astimezone(CN_TZ)
    start = (local_now - timedelta(days=local_now.weekday())).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return start, start + timedelta(days=7)


def parse_int(value: Any) -> int:
    if value in (None, "", "-"):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text or text == "-":
        return 0
    multiplier = 1
    if text.endswith("w"):
        multiplier = 10_000
        text = text[:-1]
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10_000
        text = text[:-1]
    elif text.endswith("亿"):
        multiplier = 100_000_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_hashtags(value: Any) -> list[str]:
    if not value:
        return []
    if not isinstance(value, list):
        value = [value]
    tags: list[str] = []
    for item in value:
        if isinstance(item, dict):
            tag = clean_text(item.get("hashtag_name") or item.get("cha_name") or item.get("name"))
        else:
            tag = clean_text(item)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "aweme_list", "result"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    rows = payload.get("aweme_list") or payload.get("result")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def aweme_from_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("aweme_info", "aweme", "item"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return row


def canonical_url(aweme: dict[str, Any]) -> str:
    aweme_id = str(aweme.get("aweme_id") or aweme.get("group_id") or "").strip()
    if aweme_id:
        return f"https://www.douyin.com/video/{aweme_id}"
    share_url = str(aweme.get("share_url") or "").strip()
    if share_url.startswith("http://"):
        return "https://" + share_url[len("http://") :]
    return share_url


def statistics_value(statistics: dict[str, Any], *names: str) -> int:
    for name in names:
        value = statistics.get(name)
        parsed = parse_int(value)
        if parsed:
            return parsed
    return 0


def normalize_video(row: dict[str, Any], *, keyword: str, sort_type: str, offset: int) -> dict[str, Any]:
    aweme = aweme_from_row(row)
    statistics = aweme.get("statistics") or aweme.get("statistics_info") or {}
    if not isinstance(statistics, dict):
        statistics = {}
    author = aweme.get("author") or {}
    if not isinstance(author, dict):
        author = {}

    create_ts = parse_int(aweme.get("create_time"))
    create_dt = datetime.fromtimestamp(create_ts, CN_TZ) if create_ts else None
    hashtags = normalize_hashtags(
        aweme.get("text_extra")
        or aweme.get("cha_list")
        or aweme.get("hashtags")
        or row.get("text_extra")
    )
    title = clean_text(aweme.get("desc") or aweme.get("title") or row.get("desc") or row.get("title"))
    aweme_id = str(aweme.get("aweme_id") or aweme.get("group_id") or row.get("aweme_id") or "").strip()

    return {
        "aweme_id": aweme_id,
        "url": canonical_url(aweme),
        "title": title,
        "description": clean_text(aweme.get("desc") or row.get("desc")),
        "author": clean_text(author.get("nickname") or author.get("unique_id") or row.get("nickname")),
        "author_id": clean_text(author.get("uid") or author.get("sec_uid")),
        "play": statistics_value(statistics, "play_count", "play_cnt", "play_count_value", "view_count"),
        "like": statistics_value(statistics, "digg_count", "like_count"),
        "comment": statistics_value(statistics, "comment_count"),
        "share": statistics_value(statistics, "share_count"),
        "collect": statistics_value(statistics, "collect_count", "favorite_count"),
        "create_time": create_ts,
        "create_time_text": create_dt.isoformat() if create_dt else "",
        "hashtags": hashtags,
        "source_queries": [{"keyword": keyword, "sort_type": str(sort_type), "offset": offset}],
    }


def video_key(video: dict[str, Any]) -> str:
    return str(video.get("aweme_id") or video.get("url") or "").strip()


def merge_video(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for field in ("play", "like", "comment", "share", "collect"):
        existing[field] = max(parse_int(existing.get(field)), parse_int(incoming.get(field)))
    for field in ("title", "description", "author", "author_id", "url", "create_time_text"):
        if not existing.get(field) and incoming.get(field):
            existing[field] = incoming[field]
    existing["create_time"] = max(parse_int(existing.get("create_time")), parse_int(incoming.get("create_time")))
    existing_tags = set(existing.get("hashtags") or [])
    for tag in incoming.get("hashtags") or []:
        if tag not in existing_tags:
            existing.setdefault("hashtags", []).append(tag)
            existing_tags.add(tag)
    seen = {
        (item.get("keyword"), item.get("sort_type"), item.get("offset"))
        for item in existing.get("source_queries", [])
    }
    for item in incoming.get("source_queries", []):
        key = (item.get("keyword"), item.get("sort_type"), item.get("offset"))
        if key not in seen:
            existing.setdefault("source_queries", []).append(item)
            seen.add(key)
    return existing


def dedupe_videos(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for video in videos:
        key = video_key(video)
        if not key:
            continue
        if key in by_key:
            merge_video(by_key[key], video)
        else:
            by_key[key] = dict(video)
    return list(by_key.values())


def is_hermes_agent_video(video: dict[str, Any]) -> bool:
    title = str(video.get("title") or "").lower()
    description = str(video.get("description") or "").lower()
    tags = " ".join(str(item).lower() for item in video.get("hashtags") or [])
    title_desc = f"{title} {description}"
    title_desc_collapsed = re.sub(r"[\s_/\\-]+", "", title_desc)
    tags_collapsed = re.sub(r"[\s_/\\-]+", "", tags)
    combined_collapsed = f"{title_desc_collapsed}{tags_collapsed}"

    if "hermesagent" in combined_collapsed:
        return True
    if "hermes" in title_desc and "agent" in title_desc:
        return True
    if "hermes" in title and "agent" in tags:
        return True
    return False


def rank_weekly_videos(
    videos: list[dict[str, Any]],
    *,
    week_start: datetime,
    now: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for video in dedupe_videos(videos):
        create_ts = parse_int(video.get("create_time"))
        if not create_ts:
            continue
        create_dt = datetime.fromtimestamp(create_ts, CN_TZ)
        if not (week_start <= create_dt <= now):
            continue
        if not is_hermes_agent_video(video):
            continue
        ranked.append(video)

    ranked.sort(
        key=lambda item: (
            -parse_int(item.get("play")),
            -parse_int(item.get("like")),
            -parse_int(item.get("comment")),
            -parse_int(item.get("share")),
            -parse_int(item.get("create_time")),
            str(item.get("aweme_id") or ""),
        )
    )
    for index, item in enumerate(ranked[:limit], start=1):
        item["rank"] = index
    return ranked[:limit]


def fetch_search_page(
    *,
    keyword: str,
    sort_type: str,
    offset: int,
    count: int,
    publish_time: str,
    tracking_params: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = build_search_url(
        keyword,
        sort_type=sort_type,
        offset=offset,
        count=count,
        publish_time=publish_time,
        tracking_params=tracking_params,
    )
    return fetch_json(url, keyword=keyword)


def search_nil_type(payload: dict[str, Any]) -> str | None:
    nil_info = payload.get("search_nil_info")
    if not isinstance(nil_info, dict):
        return None
    nil_type = clean_text(nil_info.get("search_nil_type") or nil_info.get("search_nil_item"))
    if not nil_type:
        return None
    return nil_type


def search_nil_error(payload: dict[str, Any]) -> str | None:
    nil_info = payload.get("search_nil_info")
    if not isinstance(nil_info, dict):
        return None
    nil_type = search_nil_type(payload)
    if not nil_type:
        return None
    if nil_type == "verify_check":
        return "Douyin returned search_nil_type=verify_check; Firefox cookies and PC browser parameters were loaded, but Douyin still requires browser verification."
    if nil_type == "antispam_check":
        nil_item = clean_text(nil_info.get("search_nil_item"))
        detail = f" ({nil_item})" if nil_item else ""
        return f"Douyin returned search_nil_type=antispam_check{detail}; the request was rejected by anti-spam validation."
    return f"Douyin returned search_nil_type={nil_type}."


def should_retry_search_nil(payload: dict[str, Any]) -> bool:
    return search_nil_type(payload) in {"verify_check", "antispam_check"}


def fetch_search_page_with_retries(
    *,
    keyword: str,
    sort_type: str,
    offset: int,
    count: int,
    publish_time: str,
    tracking_params: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str], int]:
    retry_limit = verify_retry_limit()
    current_tracking = tracking_params
    retries = 0

    for attempt in range(retry_limit):
        payload = fetch_search_page(
            keyword=keyword,
            sort_type=sort_type,
            offset=offset,
            count=count,
            publish_time=publish_time,
            tracking_params=current_tracking,
        )
        if not should_retry_search_nil(payload) or attempt >= retry_limit - 1:
            return payload, current_tracking, retries

        retries += 1
        time.sleep(0.45 * (attempt + 1))
        current_tracking = tracking_search_params()

    return payload, current_tracking, retries


def collect_search_results(
    *,
    keywords: list[str],
    sort_types: list[str],
    pages: int,
    page_size: int,
    publish_time: str,
    stop_after_matches: int | None = None,
    week_start: datetime | None = None,
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    videos: list[dict[str, Any]] = []
    errors: list[str] = []
    raw_count = 0

    if not cookie_header():
        return videos, [
            "No Douyin login cookie found. Set DOUYIN_COOKIE/DOUYIN_SEARCH_COOKIE, or log in to Douyin with Firefox and let the crawler read the Firefox profile cookies."
        ], raw_count

    tracking_params = tracking_search_params()
    for keyword in keywords:
        for sort_type in sort_types:
            for page_index in range(pages):
                offset = page_index * page_size
                try:
                    payload, tracking_params, nil_retries = fetch_search_page_with_retries(
                        keyword=keyword,
                        sort_type=sort_type,
                        offset=offset,
                        count=page_size,
                        publish_time=publish_time,
                        tracking_params=tracking_params,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded for the cron digest
                    errors.append(f"keyword={keyword!r} sort_type={sort_type} offset={offset}: {exc}")
                    break

                rows = result_rows(payload)
                if not rows:
                    nil_error = search_nil_error(payload)
                    if nil_error:
                        if nil_retries:
                            nil_error = f"{nil_error} Retried with {nil_retries + 1} tracking parameter set(s)."
                        errors.append(f"keyword={keyword!r} sort_type={sort_type} offset={offset}: {nil_error}")
                    break
                raw_count += len(rows)
                videos.extend(
                    normalize_video(row, keyword=keyword, sort_type=sort_type, offset=offset)
                    for row in rows
                )
                if stop_after_matches and week_start and now:
                    matched = rank_weekly_videos(
                        dedupe_videos(videos),
                        week_start=week_start,
                        now=now,
                        limit=stop_after_matches,
                    )
                    if len(matched) >= stop_after_matches:
                        return videos, errors, raw_count

                has_more = payload.get("has_more")
                if has_more in (0, False) or len(rows) < page_size:
                    break
                time.sleep(0.35)

    return videos, errors, raw_count


def parse_keywords(values: list[str] | None = None) -> list[str]:
    raw_values: list[str] = []
    if values:
        raw_values.extend(values)
    env_value = os.getenv("DOUYIN_HERMES_AGENT_KEYWORDS", "").strip()
    if env_value:
        raw_values.extend(part.strip() for part in re.split(r"[;,]", env_value) if part.strip())
    if not raw_values:
        raw_values = list(DEFAULT_KEYWORDS)

    keywords: list[str] = []
    for value in raw_values:
        text = value.strip()
        if text and text not in keywords:
            keywords.append(text)
    return keywords or list(DEFAULT_KEYWORDS)


def parse_sort_types(values: list[str] | None = None) -> list[str]:
    raw_values: list[str] = []
    if values:
        raw_values.extend(values)
    env_value = os.getenv("DOUYIN_SEARCH_SORT_TYPES", "").strip()
    if env_value:
        raw_values.extend(part.strip() for part in re.split(r"[;,]", env_value) if part.strip())
    if not raw_values:
        raw_values = list(DEFAULT_SORT_TYPES)

    sort_types: list[str] = []
    for value in raw_values:
        text = value.strip()
        if text and text not in sort_types:
            sort_types.append(text)
    return sort_types or list(DEFAULT_SORT_TYPES)


def build_digest(args: argparse.Namespace) -> dict[str, Any]:
    now = parse_now(args.now)
    week_start, week_end = week_window(now)
    keywords = parse_keywords(args.keyword)
    sort_types = parse_sort_types(args.sort_type)
    videos, errors, raw_count = collect_search_results(
        keywords=keywords,
        sort_types=sort_types,
        pages=args.pages,
        page_size=args.page_size,
        publish_time=args.publish_time,
        stop_after_matches=args.limit,
        week_start=week_start,
        now=now,
    )
    deduped = dedupe_videos(videos)
    weekly_items = rank_weekly_videos(deduped, week_start=week_start, now=now, limit=args.limit)
    missing_play_count = sum(1 for item in weekly_items if parse_int(item.get("play")) == 0)

    notes = [
        "Items are filtered to videos published in the current Asia/Shanghai week and ranked by current play_count.",
        "Douyin does not always expose play_count in web search responses; items without play_count are ranked as 0.",
    ]
    if missing_play_count:
        notes.append(f"{missing_play_count} matched item(s) did not expose play_count.")

    return {
        "success": True,
        "source": RUNTIME_SOURCE,
        "generated_at": now.isoformat(),
        "timezone": "Asia/Shanghai",
        "week_start": week_start.isoformat(),
        "week_end_exclusive": week_end.isoformat(),
        "filter_end": now.isoformat(),
        "search_keywords": keywords,
        "sort_types": sort_types,
        "publish_time_filter": args.publish_time,
        "requested_limit": args.limit,
        "raw_result_count": raw_count,
        "deduped_count": len(deduped),
        "matched_count": len(weekly_items),
        "items": weekly_items,
        "errors": errors,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch this week's top Douyin videos about Hermes Agent.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum videos to return, default: 10")
    parser.add_argument("--pages", type=int, default=3, help="Search pages per keyword/sort type, default: 3")
    parser.add_argument("--page-size", type=int, default=20, help="Search page size, default: 20")
    parser.add_argument("--keyword", action="append", help="Additional/override search keyword. Can be repeated.")
    parser.add_argument(
        "--sort-type",
        action="append",
        help="Douyin search sort_type to query. Can be repeated. Default: 0, 1, and 2.",
    )
    parser.add_argument(
        "--publish-time",
        default=DEFAULT_PUBLISH_TIME,
        help="Douyin publish_time filter. Default: 7 (web-search one-week filter).",
    )
    parser.add_argument("--now", help="Debug timestamp in ISO format. Defaults to current Asia/Shanghai time.")
    args = parser.parse_args(argv)

    output = build_digest(args)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
