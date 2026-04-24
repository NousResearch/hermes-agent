#!/usr/bin/env python3
"""ActivityWatch summary helper for Hermes optional skill.

Uses the local ActivityWatch HTTP API to:
- inspect server info
- list buckets
- summarize app/window activity for a bounded time window
- generate a lightweight project heat hint based on window titles and project roots

Stdlib only. Output is JSON by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:5600/api/0"
DEFAULT_ROOTS = [str(Path.home() / "Projects"), "/Volumes/Work/Storage/Projects"]


class AWError(RuntimeError):
    pass


def _get_json(url: str, timeout: int = 10) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise AWError(f"Failed to fetch ActivityWatch URL {url}: {exc}") from exc


def _api(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def get_info(base_url: str) -> dict[str, Any]:
    data = _get_json(_api(base_url, "info"), timeout=5)
    if not isinstance(data, dict):
        raise AWError("Unexpected /info payload")
    return data


def get_buckets(base_url: str) -> dict[str, Any]:
    data = _get_json(_api(base_url, "buckets/"), timeout=10)
    if not isinstance(data, dict):
        raise AWError("Unexpected /buckets payload")
    return data


def _parse_time_range(hours: int | None, start: str | None, end: str | None) -> tuple[str, str]:
    now = dt.datetime.now(dt.timezone.utc)
    if start:
        start_dt = dt.datetime.fromisoformat(start)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=dt.timezone.utc)
    else:
        span = dt.timedelta(hours=hours or 24)
        start_dt = now - span
    if end:
        end_dt = dt.datetime.fromisoformat(end)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=dt.timezone.utc)
    else:
        end_dt = now
    return start_dt.isoformat(), end_dt.isoformat()


def _bucket_ids_by_type(buckets: dict[str, Any], bucket_type: str) -> list[str]:
    ids = []
    for bucket_id, meta in buckets.items():
        if isinstance(meta, dict) and meta.get("type") == bucket_type:
            ids.append(bucket_id)
    return ids


def _fetch_bucket_events(base_url: str, bucket_id: str, start: str, end: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"start": start, "end": end})
    data = _get_json(_api(base_url, f"buckets/{bucket_id}/events?{query}"), timeout=20)
    if not isinstance(data, list):
        raise AWError(f"Unexpected events payload for bucket {bucket_id}")
    return [ev for ev in data if isinstance(ev, dict)]


def summarize_windows(base_url: str, start: str, end: str) -> dict[str, Any]:
    buckets = get_buckets(base_url)
    window_buckets = _bucket_ids_by_type(buckets, "currentwindow")
    totals: Counter[tuple[str, str]] = Counter()
    bucket_counts: dict[str, int] = {}
    for bucket_id in window_buckets:
        events = _fetch_bucket_events(base_url, bucket_id, start, end)
        bucket_counts[bucket_id] = len(events)
        for ev in events:
            data = ev.get("data") or {}
            app = str(data.get("app") or "unknown")
            title = str(data.get("title") or "")[:160]
            duration = float(ev.get("duration") or 0)
            if duration <= 0:
                continue
            totals[(app, title)] += duration
    top = [
        {
            "app": app,
            "title": title,
            "seconds": round(secs, 2),
            "hours": round(secs / 3600, 3),
        }
        for (app, title), secs in totals.most_common(25)
    ]
    return {
        "start": start,
        "end": end,
        "window_bucket_count": len(window_buckets),
        "bucket_event_counts": bucket_counts,
        "top_windows": top,
    }


def _project_catalog(project_roots: list[str]) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for root in project_roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        for domain in root_path.iterdir():
            if not domain.is_dir() or domain.name.startswith('.'):
                continue
            for proj in domain.iterdir():
                if not proj.is_dir() or proj.name.startswith('.'):
                    continue
                key = proj.name.lower().replace('_', '-').replace(' ', '-')
                catalog[key] = {
                    "project": proj.name,
                    "domain": domain.name,
                    "path": str(proj),
                }
    return catalog


def _tokenize_title(title: str) -> set[str]:
    pieces = re.split(r"[^a-zA-Z0-9]+", title.lower())
    return {p for p in pieces if len(p) >= 3}


def project_heat(base_url: str, start: str, end: str, project_roots: list[str]) -> dict[str, Any]:
    summary = summarize_windows(base_url, start, end)
    catalog = _project_catalog(project_roots)
    scores: defaultdict[str, float] = defaultdict(float)
    evidence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in summary["top_windows"]:
        title = item["title"]
        tokens = _tokenize_title(title)
        for key, meta in catalog.items():
            key_tokens = _tokenize_title(key)
            if key in title.lower() or (key_tokens and key_tokens.issubset(tokens)):
                scores[key] += float(item["seconds"])
                evidence[key].append({
                    "app": item["app"],
                    "title": title,
                    "hours": item["hours"],
                })

    ranked = []
    for key, secs in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        meta = catalog[key]
        ranked.append({
            "project": meta["project"],
            "domain": meta["domain"],
            "path": meta["path"],
            "seconds": round(secs, 2),
            "hours": round(secs / 3600, 3),
            "evidence": evidence[key][:5],
        })

    return {
        "start": start,
        "end": end,
        "project_roots": project_roots,
        "matched_projects": ranked,
        "note": "Heuristic only. Window titles are noisy; use alongside repo truth, not instead of it.",
    }


def list_buckets(base_url: str) -> dict[str, Any]:
    buckets = get_buckets(base_url)
    rows = []
    for bucket_id, meta in buckets.items():
        if not isinstance(meta, dict):
            continue
        rows.append({
            "id": bucket_id,
            "type": meta.get("type"),
            "client": meta.get("client"),
            "hostname": meta.get("hostname"),
            "last_updated": meta.get("last_updated"),
        })
    rows.sort(key=lambda r: (str(r.get("type") or ""), str(r.get("id") or "")))
    return {"buckets": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ActivityWatch helper for Hermes")
    parser.add_argument("command", choices=["info", "buckets", "summary", "heat"])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours when start is not provided")
    parser.add_argument("--start", help="ISO start timestamp")
    parser.add_argument("--end", help="ISO end timestamp")
    parser.add_argument(
        "--project-root",
        action="append",
        default=[],
        help="Project root to use for heat matching (repeatable). Defaults to ~/Projects and /Volumes/Work/Storage/Projects",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    base_url = args.base_url
    pretty = args.pretty

    if args.command == "info":
        result = get_info(base_url)
    elif args.command == "buckets":
        result = list_buckets(base_url)
    else:
        start, end = _parse_time_range(args.hours, args.start, args.end)
        if args.command == "summary":
            result = summarize_windows(base_url, start, end)
        else:
            project_roots = args.project_root or DEFAULT_ROOTS
            result = project_heat(base_url, start, end, project_roots)

    if pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AWError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
