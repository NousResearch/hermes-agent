#!/usr/bin/env python3
"""Fetch this week's top Bilibili videos about Hermes Agent.

The script uses Bilibili's public web search API and emits JSON for Hermes cron.
It searches both click-sorted and publish-date-sorted result pages, then filters
and ranks locally so the report stays bounded to the current week.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any


BILIBILI_SEARCH_API = "https://api.bilibili.com/x/web-interface/search/type"
REFERER = "https://search.bilibili.com/"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
RUNTIME_SOURCE = "Bilibili public web search API"
CN_TZ = timezone(timedelta(hours=8), "Asia/Shanghai")
DEFAULT_KEYWORDS = ("HermesAgent", "Hermes Agent")
DEFAULT_ORDERS = ("click", "pubdate")


def cookie_header() -> str | None:
    raw_cookie = os.getenv("BILIBILI_COOKIE") or os.getenv("BILI_COOKIE")
    if raw_cookie:
        return raw_cookie.strip()
    sessdata = os.getenv("BILIBILI_SESSDATA") or os.getenv("BILI_SESSDATA")
    if sessdata:
        return f"SESSDATA={sessdata.strip()}"
    return None


def request_headers() -> dict[str, str]:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": REFERER,
        "Origin": "https://search.bilibili.com",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    cookie = cookie_header()
    if cookie:
        headers["Cookie"] = cookie
    return headers


def fetch_text(url: str, *, timeout: int = 15, retries: int = 2) -> str:
    last_error: Exception | None = None
    headers = request_headers()

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
            USER_AGENT,
            "-e",
            REFERER,
            "-H",
            "Accept: application/json, text/plain, */*",
            "-H",
            "Accept-Language: zh-CN,zh;q=0.9,en;q=0.8",
            "-H",
            "Origin: https://search.bilibili.com",
        ]
        cookie = cookie_header()
        if cookie:
            cmd.extend(["-H", f"Cookie: {cookie}"])
        cmd.append(url)
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout + 5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
        if result.stderr.strip():
            raise RuntimeError(f"curl failed for {url}: {result.stderr.strip()}") from last_error

    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def fetch_json(url: str, *, timeout: int = 15, retries: int = 2) -> dict[str, Any]:
    text = fetch_text(url, timeout=timeout, retries=retries)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        prefix = text.strip().replace("\n", " ")[:120]
        raise RuntimeError(f"non-json response: {prefix!r}") from exc
    if payload.get("code") != 0:
        raise RuntimeError(f"Bilibili API returned code={payload.get('code')}: {payload.get('message')}")
    return payload


def build_search_url(
    keyword: str,
    *,
    order: str,
    page: int,
    page_size: int,
    week_start: datetime | None = None,
    week_end: datetime | None = None,
    server_time_filter: bool = False,
) -> str:
    params = {
        "search_type": "video",
        "keyword": keyword,
        "order": order,
        "page": str(page),
        "page_size": str(page_size),
    }
    if server_time_filter and week_start and week_end:
        params["pubtime_begin_s"] = str(int(week_start.timestamp()))
        params["pubtime_end_s"] = str(int(week_end.timestamp()) - 1)
    return f"{BILIBILI_SEARCH_API}?{urllib.parse.urlencode(params)}"


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
    if text.endswith("万"):
        multiplier = 10_000
        text = text[:-1]
    elif text.endswith("亿"):
        multiplier = 100_000_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def clean_html(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_url(row: dict[str, Any]) -> str:
    bvid = str(row.get("bvid") or "").strip()
    if bvid:
        return f"https://www.bilibili.com/video/{bvid}/"
    aid = row.get("aid") or row.get("id")
    if aid:
        return f"https://www.bilibili.com/video/av{aid}"
    raw = str(row.get("arcurl") or "").strip()
    if raw.startswith("http://"):
        return "https://" + raw[len("http://") :]
    if raw.startswith("//"):
        return "https:" + raw
    return raw


def normalize_video(row: dict[str, Any], *, keyword: str, order: str, page: int) -> dict[str, Any]:
    pub_ts = parse_int(row.get("pubdate"))
    pub_dt = datetime.fromtimestamp(pub_ts, CN_TZ) if pub_ts else None
    bvid = str(row.get("bvid") or "").strip()
    aid = str(row.get("aid") or row.get("id") or "").strip()
    title = clean_html(row.get("title"))
    description = clean_html(row.get("description") or row.get("desc"))
    tags = clean_html(row.get("tag") or row.get("tags"))

    return {
        "bvid": bvid,
        "aid": aid,
        "url": canonical_url(row),
        "title": title,
        "description": description,
        "author": clean_html(row.get("author") or row.get("uname")),
        "mid": parse_int(row.get("mid") or row.get("uid")),
        "play": parse_int(row.get("play")),
        "like": parse_int(row.get("like")),
        "favorites": parse_int(row.get("favorites")),
        "comments": parse_int(row.get("review")),
        "danmaku": parse_int(row.get("video_review") or row.get("danmaku")),
        "duration": clean_html(row.get("duration")),
        "typename": clean_html(row.get("typename") or row.get("cate_name")),
        "tags": tags,
        "hit_columns": [str(item) for item in row.get("hit_columns") or []],
        "pubdate": pub_ts,
        "pubdate_text": pub_dt.isoformat() if pub_dt else "",
        "source_queries": [{"keyword": keyword, "order": order, "page": page}],
    }


def video_key(video: dict[str, Any]) -> str:
    return str(video.get("bvid") or video.get("aid") or video.get("url"))


def merge_video(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for field in ("play", "like", "favorites", "comments", "danmaku"):
        existing[field] = max(parse_int(existing.get(field)), parse_int(incoming.get(field)))
    for field in ("title", "description", "author", "duration", "typename", "tags", "url", "pubdate_text"):
        if not existing.get(field) and incoming.get(field):
            existing[field] = incoming[field]
    existing["pubdate"] = max(parse_int(existing.get("pubdate")), parse_int(incoming.get("pubdate")))
    seen = {(item["keyword"], item["order"], item["page"]) for item in existing.get("source_queries", [])}
    for item in incoming.get("source_queries", []):
        key = (item["keyword"], item["order"], item["page"])
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
    tags = str(video.get("tags") or "").lower()
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
        pub_ts = parse_int(video.get("pubdate"))
        if not pub_ts:
            continue
        pub_dt = datetime.fromtimestamp(pub_ts, CN_TZ)
        if not (week_start <= pub_dt <= now):
            continue
        if not is_hermes_agent_video(video):
            continue
        ranked.append(video)

    ranked.sort(
        key=lambda item: (
            -parse_int(item.get("play")),
            -parse_int(item.get("like")),
            -parse_int(item.get("favorites")),
            -parse_int(item.get("pubdate")),
            str(item.get("bvid") or ""),
        )
    )
    for index, item in enumerate(ranked[:limit], start=1):
        item["rank"] = index
    return ranked[:limit]


def fetch_search_page(
    *,
    keyword: str,
    order: str,
    page: int,
    page_size: int,
    week_start: datetime,
    week_end: datetime,
    server_time_filter: bool,
) -> dict[str, Any]:
    url = build_search_url(
        keyword,
        order=order,
        page=page,
        page_size=page_size,
        week_start=week_start,
        week_end=week_end,
        server_time_filter=server_time_filter,
    )
    return fetch_json(url)


def collect_search_results(
    *,
    keywords: list[str],
    orders: list[str],
    pages: int,
    page_size: int,
    week_start: datetime,
    week_end: datetime,
    server_time_filter: bool,
) -> tuple[list[dict[str, Any]], list[str], int]:
    videos: list[dict[str, Any]] = []
    errors: list[str] = []
    raw_count = 0

    for keyword in keywords:
        for order in orders:
            for page in range(1, pages + 1):
                try:
                    payload = fetch_search_page(
                        keyword=keyword,
                        order=order,
                        page=page,
                        page_size=page_size,
                        week_start=week_start,
                        week_end=week_end,
                        server_time_filter=server_time_filter,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded for the cron digest
                    errors.append(f"keyword={keyword!r} order={order} page={page}: {exc}")
                    break

                rows = ((payload.get("data") or {}).get("result") or [])
                if not rows:
                    break
                raw_count += len(rows)
                page_videos = [
                    normalize_video(row, keyword=keyword, order=order, page=page)
                    for row in rows
                    if row.get("type") in (None, "video")
                ]
                videos.extend(page_videos)

                if order == "pubdate":
                    dated = [parse_int(item.get("pubdate")) for item in page_videos if parse_int(item.get("pubdate"))]
                    if dated and max(dated) < int(week_start.timestamp()):
                        break
                if len(rows) < page_size:
                    break
                time.sleep(0.25)

    return videos, errors, raw_count


def parse_keywords(values: list[str] | None = None) -> list[str]:
    raw_values: list[str] = []
    if values:
        raw_values.extend(values)
    env_value = os.getenv("BILIBILI_HERMES_AGENT_KEYWORDS", "").strip()
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


def build_digest(args: argparse.Namespace) -> dict[str, Any]:
    now = parse_now(args.now)
    week_start, week_end = week_window(now)
    keywords = parse_keywords(args.keyword)
    orders = args.order or list(DEFAULT_ORDERS)
    videos, errors, raw_count = collect_search_results(
        keywords=keywords,
        orders=orders,
        pages=args.pages,
        page_size=args.page_size,
        week_start=week_start,
        week_end=week_end,
        server_time_filter=args.server_time_filter,
    )
    deduped = dedupe_videos(videos)
    weekly_items = rank_weekly_videos(deduped, week_start=week_start, now=now, limit=args.limit)

    return {
        "success": True,
        "source": RUNTIME_SOURCE,
        "generated_at": now.isoformat(),
        "timezone": "Asia/Shanghai",
        "week_start": week_start.isoformat(),
        "week_end_exclusive": week_end.isoformat(),
        "filter_end": now.isoformat(),
        "search_keywords": keywords,
        "orders": orders,
        "requested_limit": args.limit,
        "raw_result_count": raw_count,
        "deduped_count": len(deduped),
        "matched_count": len(weekly_items),
        "items": weekly_items,
        "errors": errors,
        "notes": [
            "Items are filtered to videos published in the current Asia/Shanghai week and ranked by play count.",
            "Bilibili play counts are point-in-time values returned by the public search API.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch this week's top Bilibili videos about Hermes Agent.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum videos to return, default: 10")
    parser.add_argument("--pages", type=int, default=3, help="Search pages per keyword/order, default: 3")
    parser.add_argument("--page-size", type=int, default=20, help="Search page size, default: 20")
    parser.add_argument("--keyword", action="append", help="Additional/override search keyword. Can be repeated.")
    parser.add_argument(
        "--order",
        action="append",
        choices=("click", "pubdate", "scores", "stow", "dm"),
        help="Bilibili search order. Can be repeated. Default: click and pubdate.",
    )
    parser.add_argument("--now", help="Debug timestamp in ISO format. Defaults to current Asia/Shanghai time.")
    parser.add_argument(
        "--server-time-filter",
        action="store_true",
        help="Also send week start/end params to Bilibili. Disabled by default because it can trigger 412.",
    )
    args = parser.parse_args(argv)

    output = build_digest(args)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(1)
