#!/usr/bin/env python3
"""Recover a blocked / paywalled / WAF'd page from third-party copies.

Ladder (cheapest first):
  1. Wayback Machine "available" API  -> dated snapshot   (provenance: snapshot)
  2. archive.today domain rotation    -> dated snapshot   (provenance: snapshot)
  3. Jina Reader (JINA_API_KEY only)  -> live re-render   (provenance: live)

Every candidate body is validated before being declared a win: byte floors,
redirect-stub detection (meta-refresh/JS pointing back at the original host),
and interstitial-title rejection. Fake 200s are the norm in this space.

Stdlib only. Usage:
    python3 recover_page.py URL [--json] [--out FILE] [--timeout N]

Exit codes: 0 recovered, 1 nothing worked, 2 bad invocation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

ARCHIVE_TODAY_HOSTS = ["archive.ph", "archive.md", "archive.li", "archive.is"]

# Titles that mean "this is not the page you asked for".
INTERSTITIAL_TITLES = (
    "just a moment",
    "redirecting",
    "google search",
    "attention required",
    "access denied",
    "are you a robot",
    "one more step",
)

# Below these floors a body is a stub or an error page, not content.
MIN_BODY_BYTES = {"wayback": 3072, "archive_today": 3072, "jina": 512}

REDIRECT_STUB_RE = re.compile(
    r'http-equiv=["\']?refresh|window\.location|location\.replace', re.IGNORECASE
)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _fetch(
    url: str,
    timeout: int,
    headers: dict | None = None,
    retries_on_429: int = 2,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(retries_on_429 + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries_on_429:
                time.sleep(5 * (attempt + 1))
                continue
            return exc.code, exc.read() if exc.fp else b""
        except (urllib.error.URLError, OSError, ValueError):
            return 0, b""
    return 0, b""


def _fetch_follow(url: str, timeout: int) -> tuple[int, bytes, str]:
    """Like _fetch but also returns the final URL after redirects."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.geturl()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() if exc.fp else b"", exc.geturl() or url
    except (urllib.error.URLError, OSError, ValueError):
        return 0, b"", url


def _page_title(body: bytes) -> str:
    m = TITLE_RE.search(body[:65536].decode("utf-8", "replace"))
    return re.sub(r"\s+", " ", m.group(1)).strip().lower() if m else ""


def validate(body: bytes, route: str, target_url: str) -> str | None:
    """Return a rejection reason, or None if the body looks like real content."""
    floor = MIN_BODY_BYTES.get(route, 3072)
    if len(body) < floor:
        return f"body_too_small:{len(body)}<{floor}"
    title = _page_title(body)
    for marker in INTERSTITIAL_TITLES:
        if marker in title:
            return f"interstitial_title:{marker!r}"
    # Redirect stub: small-ish page whose only job is bouncing back to the
    # original (blocked) host — the classic AMP-cache failure mode.
    if len(body) < 8192 and REDIRECT_STUB_RE.search(body.decode("utf-8", "replace")):
        target_host = urllib.parse.urlsplit(target_url).hostname or ""
        if target_host and target_host.encode() in body:
            return "redirect_stub_to_origin"
    return None


def try_wayback(url: str, timeout: int) -> dict | None:
    snap_url = None
    snap_ts = None
    discovery = "https://archive.org/wayback/available?url=" + urllib.parse.quote(url, safe="")
    status, raw = _fetch(discovery, timeout)
    if status == 200:
        try:
            closest = json.loads(raw).get("archived_snapshots", {}).get("closest", {})
        except (json.JSONDecodeError, AttributeError):
            closest = {}
        if closest.get("available") and closest.get("url"):
            snap_url = closest["url"].replace(
                "http://web.archive.org", "https://web.archive.org"
            )
            snap_ts = closest.get("timestamp")
    if snap_url is None:
        # Discovery API is rate-limited far more aggressively than snapshot
        # serving. Fall back to the redirect form: /web/2/<url> bounces to
        # the newest snapshot if one exists (404 page otherwise).
        snap_url = "https://web.archive.org/web/2/" + url
    status, body, final_url = _fetch_follow(snap_url, timeout)
    if status != 200 or validate(body, "wayback", url):
        return None
    if snap_ts is None:
        m = re.search(r"/web/(\d{14})", final_url)
        snap_ts = m.group(1) if m else None
    return {
        "route": "wayback",
        "provenance": "snapshot",
        "snapshot_timestamp": snap_ts,
        "source_url": final_url,
        "body": body,
    }


def try_archive_today(url: str, timeout: int) -> dict | None:
    for host in ARCHIVE_TODAY_HOSTS:
        fetch_url = f"https://{host}/newest/{url}"
        status, body = _fetch(fetch_url, timeout)
        if status != 200:
            continue
        if validate(body, "archive_today", url):
            continue  # 429 bodies and interstitials land here
        return {
            "route": f"archive_today:{host}",
            "provenance": "snapshot",
            "snapshot_timestamp": None,  # archive.today embeds the date in-page
            "source_url": fetch_url,
            "body": body,
        }
    return None


def try_jina(url: str, timeout: int) -> dict | None:
    key = os.environ.get("JINA_API_KEY")
    if not key:
        return None
    status, body = _fetch(
        "https://r.jina.ai/" + url, timeout, headers={"Authorization": f"Bearer {key}"}
    )
    if status != 200 or validate(body, "jina", url):
        return None
    return {
        "route": "jina_reader",
        "provenance": "live",
        "snapshot_timestamp": None,
        "source_url": "https://r.jina.ai/" + url,
        "body": body,
    }


ROUTES = (try_wayback, try_archive_today, try_jina)


def recover(url: str, timeout: int = 25) -> dict | None:
    for route_fn in ROUTES:
        result = route_fn(url, timeout)
        if result:
            return result
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recover a blocked / paywalled / WAF'd page from third-party copies."
    )
    ap.add_argument("url")
    ap.add_argument("--json", action="store_true", help="print metadata as JSON")
    ap.add_argument("--out", help="write recovered body to this file")
    ap.add_argument("--timeout", type=int, default=25)
    args = ap.parse_args()

    if not args.url.startswith(("http://", "https://")):
        print("error: URL must start with http:// or https://", file=sys.stderr)
        return 2

    result = recover(args.url, args.timeout)
    if not result:
        msg = {"recovered": False, "url": args.url,
               "hint": "No archive copy found. Try the API-first pivot or the browser tool."}
        print(json.dumps(msg, indent=2) if args.json else msg["hint"], file=sys.stderr)
        return 1

    body = result.pop("body")
    result.update({"recovered": True, "url": args.url, "body_bytes": len(body)})
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(body)
        result["saved_to"] = args.out

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k}: {v}")
        if not args.out:
            print("\n--- body (first 2000 chars) ---")
            print(body[:2000].decode("utf-8", "replace"))
    if result["provenance"] == "snapshot":
        print(
            "\nNOTE: this is an ARCHIVED SNAPSHOT, not the live page. "
            "Cite it with its timestamp.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
