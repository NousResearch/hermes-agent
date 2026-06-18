#!/usr/bin/env python3
"""LAPRAS job search helper (optional API). Falls back to printing browser hint."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://lapras.com/api/v1"


def _get_api_key() -> str:
    return (os.environ.get("LAPRAS_API_KEY") or "").strip()


def search_jobs(query: str, *, limit: int = 20) -> dict:
    key = _get_api_key()
    if not key:
        return {
            "ok": False,
            "error": "LAPRAS_API_KEY not set in environment",
            "hint": "Use browser_navigate to https://lapras.com/ or Gmail search instead.",
        }
    url = f"{API_BASE}/jobs?query={urllib.parse.quote(query)}&limit={limit}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return {"ok": True, "data": json.loads(body)}
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "error": exc.read().decode("utf-8", errors="replace")[:500],
            "hint": "If 404/401, use web_search + browser_navigate for LAPRAS instead.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LAPRAS job search")
    parser.add_argument("query", nargs="?", default="AIエンジニア")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    result = search_jobs(args.query, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
