#!/usr/bin/env python3
"""Distill a HAR file into an API summary an agent can turn into a client.

Usage:
  python3 har_to_client.py <input.har> [--include-static] [--host SUBSTRING] [--max-body 600]

Filters to XHR/fetch/JSON traffic by default, groups by (method, host, path
template), and prints per-endpoint: query params, interesting request headers,
request body sample, response content-type/status, and a response body sample.
Numeric/UUID-ish path segments are collapsed to {id} so repeated calls group.
Also prints "### Replay hints": the browser User-Agent plus whether cookies or
auth/token headers were present -- send those in the derived client or you may
get a 403/401.
"""
import argparse
import json
import re
import sys
from collections import OrderedDict
from urllib.parse import urlsplit

BORING_HEADERS = {
    "accept-encoding", "accept-language", "connection", "content-length",
    "host", "origin", "referer", "sec-ch-ua", "sec-ch-ua-mobile",
    "sec-ch-ua-platform", "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site",
    "user-agent", "pragma", "cache-control", "priority", "te",
    "upgrade-insecure-requests", "cookie",
}
ID_SEG = re.compile(r"^(\d+|[0-9a-f]{8}-[0-9a-f-]{27,}|[0-9a-f]{16,})$", re.I)
STATIC_EXT = re.compile(r"\.(js|css|png|jpe?g|gif|svg|webp|ico|woff2?|ttf|mp4|map)$", re.I)


def path_template(path: str) -> str:
    segs = path.split("/")
    return "/".join("{id}" if ID_SEG.match(s) else s for s in segs)


def is_api_entry(entry: dict) -> bool:
    req = entry["request"]
    resp = entry.get("response", {})
    rtype = (entry.get("_resourceType") or "").lower()
    mime = (resp.get("content", {}).get("mimeType") or "").lower()
    if rtype in ("xhr", "fetch"):
        return True
    if "json" in mime:
        return True
    if req["method"] not in ("GET", "HEAD") and not STATIC_EXT.search(urlsplit(req["url"]).path):
        return True
    return False


def trunc(text, n: int) -> str:
    text = text if isinstance(text, str) else str(text)
    return text if len(text) <= n else text[:n] + f"... [{len(text)} chars total]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("har")
    ap.add_argument("--include-static", action="store_true")
    ap.add_argument("--host", default=None, help="only endpoints whose host contains this")
    ap.add_argument("--max-body", type=int, default=600)
    args = ap.parse_args()

    with open(args.har, encoding="utf-8") as f:
        har = json.load(f)

    groups = OrderedDict()
    for entry in har["log"]["entries"]:
        req = entry["request"]
        url = urlsplit(req["url"])
        if url.scheme not in ("http", "https"):
            continue
        if args.host and args.host not in url.netloc:
            continue
        if not args.include_static:
            if STATIC_EXT.search(url.path) or not is_api_entry(entry):
                continue
        key = (req["method"], url.netloc, path_template(url.path))
        g = groups.setdefault(key, {"count": 0, "queries": set(), "headers": {},
                                    "req_body": None, "resp": None})
        g["count"] += 1
        for q in req.get("queryString", []):
            g["queries"].add((q["name"], trunc(q["value"], 80)))
        for h in req.get("headers", []):
            name = h["name"].lower().lstrip(":")
            if name in BORING_HEADERS or name in ("method", "path", "scheme", "authority"):
                continue
            g["headers"][name] = trunc(h["value"], 120)
        post = req.get("postData", {})
        if post.get("text") and g["req_body"] is None:
            g["req_body"] = (post.get("mimeType", ""), trunc(post["text"], args.max_body))
        resp = entry.get("response", {})
        if g["resp"] is None and resp:
            content = resp.get("content", {})
            g["resp"] = (resp.get("status"), content.get("mimeType", ""),
                         trunc(content.get("text") or "", args.max_body))

    if not groups:
        print("No API-looking entries found. Re-run with --include-static to see everything.")
        return 1

    # Surface the browser identity so the replay client can match it (many
    # sites 403 a default library User-Agent).
    ua = None
    saw_cookie = saw_auth = False
    for entry in har["log"]["entries"]:
        for h in entry["request"].get("headers", []):
            n = h["name"].lower()
            if n == "user-agent" and ua is None:
                ua = h["value"]
            if n == "cookie":
                saw_cookie = True
            if n in ("authorization", "x-api-key") or "token" in n:
                saw_auth = True
    print("### Replay hints")
    if ua:
        print(f"  User-Agent (send this): {ua}")
    if saw_cookie:
        print("  Cookies present -> session may be auth-gated; capture & resend the Cookie header.")
    if saw_auth:
        print("  Authorization/token header present -> extract and resend it.")

    for (method, host, path), g in groups.items():
        print(f"\n=== {method} https://{host}{path}  (x{g['count']})")
        if g["queries"]:
            print("  query params:")
            for name, val in sorted(g["queries"]):
                print(f"    {name} = {val}")
        if g["headers"]:
            print("  request headers (non-boring):")
            for name, val in sorted(g["headers"].items()):
                print(f"    {name}: {val}")
        if g["req_body"]:
            print(f"  request body ({g['req_body'][0]}):")
            print(f"    {g['req_body'][1]}")
        if g["resp"]:
            status, mime, body = g["resp"]
            print(f"  response: {status} {mime}")
            if body:
                print(f"    {body}")
    print(f"\n{len(groups)} distinct endpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
