#!/usr/bin/env python3
"""Call a TwittrAPI read endpoint and print the JSON body.

    python3 scripts/twittrapi.py /user/by/username/XDevelopers
    python3 scripts/twittrapi.py /search "searchItem=from:XDevelopers lang:en" category=Latest
    python3 scripts/twittrapi.py /user/-1/followers username=XDevelopers --pages 3

Sends GET with the X-API-KEY header from TWITTRAPI_API_KEY and format=simple
(flat JSON: {data, nextCursor, previousCursor, hasNextPage}). --pages N follows
nextCursor for up to N billed calls and merges the data arrays. Prints the
balance and price headers to stderr. Exit 1 on HTTP errors (body still printed),
2 on usage errors. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://api.twittrapi.com"
TIMEOUT = 60
# Python-urllib/x.y is blocked at the edge; send a named agent instead.
USER_AGENT = "hermes-twittrapi-skill/1.0"
INFO_HEADERS = ("X-Remaining-Balance", "X-Price-Per-Call", "X-Refunded")


def build_url(path: str, params: dict[str, str]) -> str:
    query = {"format": "simple", **params}
    return f"{BASE_URL}{urllib.parse.quote(path, safe='/-_.~')}?{urllib.parse.urlencode(query)}"


def fetch(url: str, key: str) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(
        url, headers={"X-API-KEY": key, "Accept": "application/json", "User-Agent": USER_AGENT},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers or {})


def next_cursor(page: dict, sent: str | None) -> str | None:
    """Cursor for the next page, or None when the listing is exhausted."""
    cursor = page.get("nextCursor")
    if not page.get("hasNextPage") or not cursor or cursor == sent:
        return None
    return cursor


def parse_params(pairs: list[str]) -> dict[str, str]:
    params = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise ValueError(f"expected name=value, got {pair!r}")
        params[name] = value
    return params


def run(path: str, params: dict[str, str], pages: int, key: str) -> int:
    merged: list = []
    page: dict = {}
    cursor = params.get("cursor")
    for _ in range(pages):
        if cursor:
            params["cursor"] = cursor
        status, body, headers = fetch(build_url(path, params), key)
        for name in INFO_HEADERS:
            value = next((v for k, v in headers.items() if k.lower() == name.lower()), None)
            if value is not None:
                print(f"{name}: {value}", file=sys.stderr)
        if not 200 <= status < 300:
            print(body.decode("utf-8", "replace"))
            print(f"HTTP {status}", file=sys.stderr)
            return 1
        page = json.loads(body or b"{}")
        if pages == 1 or not isinstance(page.get("data"), list):
            print(json.dumps(page, ensure_ascii=False, indent=2))
            return 0
        merged.extend(page["data"])
        cursor = next_cursor(page, cursor)
        if cursor is None:
            break
    result = {
        "data": merged,
        "nextCursor": cursor,
        "hasNextPage": cursor is not None and bool(page.get("hasNextPage")),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GET a TwittrAPI read endpoint (format=simple).")
    parser.add_argument("path", help="endpoint path, e.g. /user/by/username/XDevelopers")
    parser.add_argument("params", nargs="*", help="query parameters as name=value")
    parser.add_argument("--pages", type=int, default=1, help="follow nextCursor for up to N calls")
    args = parser.parse_args(argv)

    if not args.path.startswith("/"):
        print("path must start with /", file=sys.stderr)
        return 2
    if args.pages < 1:
        print("--pages must be at least 1", file=sys.stderr)
        return 2
    key = os.environ.get("TWITTRAPI_API_KEY", "").strip()
    if not key:
        print(
            "no TwittrAPI key configured: set TWITTRAPI_API_KEY "
            "(create a key at https://app.twittrapi.com)",
            file=sys.stderr,
        )
        return 2
    try:
        params = parse_params(args.params)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    return run(args.path, params, args.pages, key)


if __name__ == "__main__":
    sys.exit(main())
