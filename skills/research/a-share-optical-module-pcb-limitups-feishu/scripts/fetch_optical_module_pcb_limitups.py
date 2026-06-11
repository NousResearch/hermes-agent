#!/usr/bin/env python3
"""Fetch current-trading-day optical module and PCB concept limit-up stocks.

The script uses Eastmoney public endpoints and emits JSON for Hermes cron.
It prefers Python stdlib fetching and falls back to curl because some
Eastmoney quote hosts intermittently close Python TLS connections.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any


USER_AGENT = "Mozilla/5.0 (compatible; HermesAgentOpticalPcbLimitups/1.0)"
REFERER = "https://quote.eastmoney.com/"
H5_REFERER = "https://emdatah5.eastmoney.com/dc/zjlx/block"
QUOTE_TOKEN = "bd1d9ddb04089700cf9c27f6f7426281"
ZT_POOL_TOKEN = "7eea3edcaed734bea9cbfc24409ed989"
RUNTIME_SOURCE = "Eastmoney public quote and H5 data APIs"

DEFAULT_CONCEPTS = (
    {"keyword": "光通信模块", "code": "BK1136", "name": "光通信模块"},
    {"keyword": "CPO概念", "code": "BK1128", "name": "CPO概念"},
    {"keyword": "PCB", "code": "BK0877", "name": "PCB"},
)

QUOTE_HOSTS = (
    "https://push2.eastmoney.com",
    "https://1.push2.eastmoney.com",
    "https://79.push2.eastmoney.com",
)


def fetch_text(url: str, *, timeout: int = 12, retries: int = 3, referer: str = REFERER) -> str:
    last_error: Exception | None = None
    headers = {"User-Agent": USER_AGENT, "Referer": referer}

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                text = response.read().decode(charset, errors="replace")
                if text.strip():
                    return text
        except Exception as exc:  # noqa: BLE001 - fallback below includes diagnostics
            last_error = exc
            time.sleep(0.25 * (attempt + 1))

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
            referer,
            url,
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout + 5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
        if result.stderr.strip():
            raise RuntimeError(f"curl failed for {url}: {result.stderr.strip()}") from last_error

    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty response")
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"\((\{.*\})\)\s*;?\s*$", stripped, re.S)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"unexpected response prefix: {stripped[:80]!r}")


def fetch_json(url: str, *, timeout: int = 12, retries: int = 3, referer: str = REFERER) -> dict[str, Any]:
    return parse_json_payload(fetch_text(url, timeout=timeout, retries=retries, referer=referer))


def scale_percent(value: Any) -> float | None:
    if value in (None, "-", ""):
        return None
    number = float(value)
    return number / 100.0 if abs(number) > 200 else number


def scale_board_price(value: Any) -> float | None:
    if value in (None, "-", ""):
        return None
    number = float(value)
    return number / 100.0 if abs(number) > 1000 else number


def scale_zt_price(value: Any) -> float | None:
    if value in (None, "-", ""):
        return None
    number = float(value)
    return number / 1000.0 if abs(number) > 1000 else number


def money_yuan_text(value: Any) -> str:
    if value in (None, "-", ""):
        return "-"
    number = float(value)
    abs_number = abs(number)
    if abs_number >= 1e8:
        return f"{number / 1e8:.2f}亿"
    if abs_number >= 1e4:
        return f"{number / 1e4:.2f}万"
    return f"{number:.0f}"


def hhmmss(value: Any) -> str | None:
    if value in (None, "-", "", 0):
        return None
    text = str(int(value)).rjust(6, "0")
    return f"{text[:2]}:{text[2:4]}:{text[4:]}"


def market_label(market: Any, code: Any = None) -> str:
    code_text = str(code or "")
    if str(market) == "1":
        return "SH"
    if code_text.startswith(("4", "8", "9")):
        return "BJ"
    return "SZ"


def parse_concept_specs(raw_values: list[str] | None = None) -> list[dict[str, str]]:
    values: list[str] = []
    if raw_values:
        values.extend(raw_values)
    env_value = os.getenv("A_SHARE_OPTICAL_MODULE_PCB_CONCEPTS", "").strip()
    if env_value:
        values.extend(part.strip() for part in env_value.split(";") if part.strip())

    if not values:
        return [dict(item) for item in DEFAULT_CONCEPTS]

    specs: list[dict[str, str]] = []
    for value in values:
        parts = [part.strip() for part in value.split(":")]
        if not parts or not parts[0]:
            continue
        spec = {"keyword": parts[0]}
        if len(parts) >= 2 and parts[1]:
            spec["code"] = parts[1]
        if len(parts) >= 3 and parts[2]:
            spec["name"] = parts[2]
        specs.append(spec)
    return specs or [dict(item) for item in DEFAULT_CONCEPTS]


def search_board_code(spec: dict[str, str]) -> dict[str, str]:
    keyword = spec["keyword"]
    query = urllib.parse.urlencode({"input": keyword, "type": "14", "token": "0", "count": "20"})
    url = f"https://searchapi.eastmoney.com/api/suggest/get?{query}"

    def payload_from_row(row: dict[str, Any]) -> dict[str, str]:
        code = str(row.get("Code") or spec.get("code") or "")
        name = str(row.get("Name") or spec.get("name") or keyword)
        return {
            "keyword": keyword,
            "code": code,
            "name": name,
            "quote_id": str(row.get("QuoteID") or f"90.{code}"),
        }

    try:
        payload = fetch_json(url, timeout=10, retries=2)
        rows = ((payload.get("QuotationCodeTable") or {}).get("Data") or [])
        board_rows = [row for row in rows if row.get("Classify") == "BK"]
        for row in board_rows:
            if str(row.get("Name") or "") == keyword:
                return payload_from_row(row)
        for row in board_rows:
            name = str(row.get("Name") or "")
            if keyword in name or name in keyword:
                return payload_from_row(row)
        if board_rows:
            return payload_from_row(board_rows[0])
    except Exception:
        pass

    fallback_code = spec.get("code", "")
    if fallback_code:
        fallback_name = spec.get("name") or keyword
        return {
            "keyword": keyword,
            "code": fallback_code,
            "name": fallback_name,
            "quote_id": f"90.{fallback_code}",
        }
    raise RuntimeError(f"failed to resolve Eastmoney concept board for {keyword!r}")


def quote_clist_url(host: str, board_code: str, page: int, page_size: int = 100) -> str:
    params = {
        "np": "1",
        "fltt": "1",
        "invt": "2",
        "fs": f"b:{board_code}",
        "fields": "f12,f13,f14,f2,f3,f20,f21,f62",
        "fid": "f3",
        "pn": str(page),
        "pz": str(page_size),
        "po": "1",
        "ut": QUOTE_TOKEN,
        "dect": "1",
    }
    return f"{host}/api/qt/clist/get?{urllib.parse.urlencode(params)}"


def h5_clist_url(board_code: str, page: int, page_size: int = 100) -> str:
    params = {
        "fields": "f12,f13,f14,f2,f3,f20,f21,f62",
        "pn": str(page),
        "pz": str(page_size),
        "fid": "f3",
        "po": "1",
        "fs": f"b:{board_code}",
        "ut": QUOTE_TOKEN,
    }
    return f"https://emdatah5.eastmoney.com/dc/ZJLX/getZDYLBData?{urllib.parse.urlencode(params)}"


def normalize_constituent(item: dict[str, Any]) -> dict[str, Any]:
    code = str(item.get("f12", ""))
    return {
        "code": code,
        "market": item.get("f13"),
        "market_label": market_label(item.get("f13"), code),
        "name": str(item.get("f14", "")),
        "last_price": scale_board_price(item.get("f2")),
        "change_pct": scale_percent(item.get("f3")),
        "total_market_value": item.get("f20"),
        "float_market_value": item.get("f21"),
        "main_net_inflow": item.get("f62"),
    }


def fetch_board_constituents_from_h5(board_code: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    page_size = 100

    while True:
        payload = fetch_json(h5_clist_url(board_code, page, page_size), timeout=12, retries=3, referer=H5_REFERER)
        data = payload.get("data") or {}
        diff = data.get("diff") or []
        if page == 1 and not diff:
            raise RuntimeError("empty H5 board constituent response")
        total = int(data.get("total") or len(rows) + len(diff))
        for item in diff:
            rows.append(normalize_constituent(item))
        if len(rows) >= total or not diff:
            break
        page += 1
        if page > 20:
            break

    return rows


def fetch_board_constituents_from_quote(board_code: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    page_size = 100

    while True:
        payload = None
        errors = []
        for host in QUOTE_HOSTS:
            url = quote_clist_url(host, board_code, page, page_size)
            try:
                candidate = fetch_json(url, timeout=12, retries=2)
                data = candidate.get("data") or {}
                diff = data.get("diff") or []
                if diff:
                    payload = candidate
                    break
                errors.append(f"{host}: empty diff")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{host}: {exc}")
        if payload is None:
            if page == 1:
                raise RuntimeError("; ".join(errors) or "failed to fetch board constituents")
            break

        data = payload.get("data") or {}
        diff = data.get("diff") or []
        total = int(data.get("total") or len(rows) + len(diff))
        for item in diff:
            rows.append(normalize_constituent(item))
        if len(rows) >= total or not diff:
            break
        page += 1
        if page > 20:
            break

    return rows


def fetch_board_constituents(board_code: str) -> list[dict[str, Any]]:
    errors = []
    for fetcher in (fetch_board_constituents_from_h5, fetch_board_constituents_from_quote):
        try:
            rows = fetcher(board_code)
            if rows:
                return rows
            errors.append(f"{fetcher.__name__}: empty rows")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{fetcher.__name__}: {exc}")
    raise RuntimeError("failed to fetch board constituents: " + "; ".join(errors))


def zt_pool_url(date_text: str, page_size: int = 500) -> str:
    params = {
        "ut": ZT_POOL_TOKEN,
        "dpt": "wz.ztzt",
        "Pageindex": "0",
        "pagesize": str(page_size),
        "sort": "fbt:asc",
        "date": date_text,
    }
    return f"https://push2ex.eastmoney.com/getTopicZTPool?{urllib.parse.urlencode(params)}"


def candidate_dates(days_back: int = 10) -> list[str]:
    tz = timezone(timedelta(hours=8))
    today = datetime.now(tz).date()
    return [(today - timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days_back + 1)]


def fetch_latest_limitup_pool() -> dict[str, Any]:
    errors = []
    for date_text in candidate_dates(10):
        try:
            payload = fetch_json(zt_pool_url(date_text), timeout=15, retries=3)
            data = payload.get("data") or {}
            pool = data.get("pool") or []
            qdate = data.get("qdate")
            if qdate and pool:
                return {
                    "requested_date": date_text,
                    "trade_date": str(qdate),
                    "total_limitup_count": data.get("tc"),
                    "pool": pool,
                }
            errors.append(f"{date_text}: empty pool")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{date_text}: {exc}")
    raise RuntimeError("failed to fetch a recent limit-up pool: " + "; ".join(errors[:5]))


def stock_sort_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
    hit_count = float(item.get("concept_hit_count") or 0)
    lbc = float(item.get("consecutive_limitups") or 0)
    fund = float(item.get("sealed_fund") or 0)
    fbt = int(item.get("first_limitup_raw") or 999999)
    return (-hit_count, -lbc, -fund, fbt)


def build_concept_index(concept_specs: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    boards: list[dict[str, Any]] = []
    stock_index: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for spec in concept_specs:
        keyword = spec["keyword"]
        try:
            board = search_board_code(spec)
            constituents = fetch_board_constituents(board["code"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{keyword}: {exc}")
            continue

        boards.append(
            {
                "keyword": keyword,
                "code": board["code"],
                "name": board["name"],
                "quote_id": board["quote_id"],
                "constituent_count": len(constituents),
            }
        )

        for row in constituents:
            code = row.get("code")
            if not code:
                continue
            entry = stock_index.setdefault(
                code,
                {
                    "code": code,
                    "name": row.get("name", ""),
                    "concepts": [],
                    "board_rows": {},
                },
            )
            concept_name = board["name"]
            if concept_name not in entry["concepts"]:
                entry["concepts"].append(concept_name)
            entry["board_rows"][concept_name] = row

    return boards, stock_index, errors


def build_report(limit: int, concept_specs: list[dict[str, str]]) -> dict[str, Any]:
    boards, stock_index, concept_errors = build_concept_index(concept_specs)
    limit_pool = fetch_latest_limitup_pool()

    matches = []
    for raw in limit_pool["pool"]:
        code = str(raw.get("c", ""))
        related = stock_index.get(code)
        if not related:
            continue

        zttj = raw.get("zttj") or {}
        concepts = list(related.get("concepts") or [])
        board_rows = related.get("board_rows") or {}
        board_change_pct_by_concept = {
            name: row.get("change_pct")
            for name, row in board_rows.items()
            if row.get("change_pct") is not None
        }
        item = {
            "code": code,
            "name": str(raw.get("n") or related.get("name") or ""),
            "market": raw.get("m"),
            "market_label": market_label(raw.get("m"), code),
            "concepts": concepts,
            "concepts_text": "、".join(concepts),
            "concept_hit_count": len(concepts),
            "industry": raw.get("hybk"),
            "price": scale_zt_price(raw.get("p")),
            "limitup_pct": scale_percent(raw.get("zdp")),
            "turnover": raw.get("amount"),
            "turnover_text": money_yuan_text(raw.get("amount")),
            "float_market_value": raw.get("ltsz"),
            "float_market_value_text": money_yuan_text(raw.get("ltsz")),
            "turnover_rate": scale_percent(raw.get("hs")),
            "consecutive_limitups": raw.get("lbc") or 0,
            "first_limitup_time": hhmmss(raw.get("fbt")),
            "last_limitup_time": hhmmss(raw.get("lbt")),
            "first_limitup_raw": raw.get("fbt"),
            "sealed_fund": raw.get("fund"),
            "sealed_fund_text": money_yuan_text(raw.get("fund")),
            "open_count": raw.get("zbc"),
            "limitup_stat": {
                "days": zttj.get("days"),
                "count": zttj.get("ct"),
            },
            "board_change_pct_by_concept": board_change_pct_by_concept,
        }
        matches.append(item)

    matches.sort(key=stock_sort_key)
    top = matches[:limit]
    fetched_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    concept_names = [board["name"] for board in boards]

    warnings = []
    if len(matches) < limit:
        warnings.append(f"Only {len(matches)} matching optical module/PCB concept limit-up stock(s) found.")
    if concept_errors:
        warnings.append("Some concept boards failed to load: " + "; ".join(concept_errors))

    return {
        "fetched_at": fetched_at,
        "source": RUNTIME_SOURCE,
        "concepts": boards,
        "concept_names": concept_names,
        "concept_errors": concept_errors,
        "trade_date": limit_pool["trade_date"],
        "requested_date": limit_pool["requested_date"],
        "schedule_note": "The pool date is the latest trading date returned by Eastmoney, suitable for the current trading day at 23:30.",
        "total_limitup_count": limit_pool["total_limitup_count"],
        "concept_constituent_count": len(stock_index),
        "matched_count": len(matches),
        "limit": limit,
        "ranking_rule": "concept_hit_count desc, consecutive_limitups desc, sealed_fund desc, first_limitup_time asc",
        "items": top,
        "warnings": warnings,
        "source_urls": {
            "board_search": "https://searchapi.eastmoney.com/api/suggest/get",
            "board_constituents_h5": "https://emdatah5.eastmoney.com/dc/ZJLX/getZDYLBData",
            "board_constituents": "https://push2.eastmoney.com/api/qt/clist/get",
            "limitup_pool": "https://push2ex.eastmoney.com/getTopicZTPool",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch A-share optical module and PCB concept limit-up stocks.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--concept",
        action="append",
        dest="concepts",
        help=(
            "Concept spec, repeatable. Format: keyword, keyword:BK0000, or "
            "keyword:BK0000:DisplayName. Defaults to 光通信模块, CPO概念, PCB."
        ),
    )
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit must be positive")

    concept_specs = parse_concept_specs(args.concepts)
    payload = build_report(args.limit, concept_specs)
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
