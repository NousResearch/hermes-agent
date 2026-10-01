#!/usr/bin/env python3
"""Read Reddit without a browser: listings, search, threads with comments, user pages.

Two backends, chosen automatically:

* **OAuth API** (preferred when ``REDDIT_CLIENT_ID`` + ``REDDIT_CLIENT_SECRET`` are set):
  app-only ``client_credentials`` grant for a free "script" app registered at
  https://www.reddit.com/prefs/apps. No username, password or cookie is ever used and
  the script never acts as a user. ~100 requests/minute, full JSON including scores
  and nested comments.
* **Anonymous Atom feeds** (``.rss`` endpoints): the only unauthenticated path Reddit
  still serves to server IPs (``.json`` and old.reddit return 403 / an empty shell).
  Roughly ONE request per minute per IP; the script sleeps until the window resets
  when it hits a 429 and retries once.

    python3 reddit.py sub LocalLLaMA [--sort hot|new|top] [--limit N]
    python3 reddit.py search "hermes agent" [--sub LocalLLaMA] [--sort new] [--limit N]
    python3 reddit.py thread https://www.reddit.com/r/x/comments/abc123/... [--limit N]
    python3 reddit.py user spez [--limit N]
    python3 reddit.py doctor            # which backend is active, and why

Add ``--json`` to any read command for machine-readable output. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

USER_AGENT = "hermes-agent/1.0 (reddit-reading skill; +https://github.com/NousResearch/hermes-agent)"
TIMEOUT = 25
ATOM = {"a": "http://www.w3.org/2005/Atom"}
WWW = "https://www.reddit.com"
OAUTH = "https://oauth.reddit.com"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_THREAD_RE = re.compile(r"reddit\.com/r/([^/]+)/comments/([a-z0-9]+)", re.I)


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    # Reddit wraps entry bodies in a <table> with a "submitted by /u/x [link] [comments]" footer.
    text = _TAG_RE.sub(" ", html.unescape(text))
    text = re.sub(r"submitted by\s+/u/\S+|\[link\]|\[comments\]", " ", text)
    return _WS_RE.sub(" ", html.unescape(text)).strip()


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _get(url: str, headers: dict | None = None, retry_on_429: bool = True) -> tuple[bytes, dict]:
    hdrs = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and retry_on_429:
            wait = _reset_seconds(exc.headers)
            print(f"reddit: 429 rate-limited, sleeping {wait}s until the window resets", file=sys.stderr)
            time.sleep(wait)
            return _get(url, headers, retry_on_429=False)
        raise


def _reset_seconds(headers) -> int:
    for key in ("x-ratelimit-reset", "retry-after"):
        val = headers.get(key) if headers else None
        if val:
            try:
                return max(1, min(int(float(val)) + 1, 120))
            except ValueError:
                pass
    return 61


# ── OAuth backend ────────────────────────────────────────────────────────────

def oauth_credentials() -> tuple[str, str] | None:
    cid, secret = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
    return (cid, secret) if cid and secret else None


def oauth_token(cid: str, secret: str) -> str:
    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    req = urllib.request.Request(
        f"{WWW}/api/v1/access_token", data=body,
        headers={"Authorization": f"Basic {auth}", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())["access_token"]


def _api(path: str, token: str, **params):
    params.setdefault("raw_json", 1)
    url = f"{OAUTH}{path}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
    data, _ = _get(url, {"Authorization": f"Bearer {token}"})
    return json.loads(data)


def _post_from_api(child: dict) -> dict:
    d = child["data"]
    return {
        "title": d.get("title"),
        "author": d.get("author"),
        "subreddit": d.get("subreddit"),
        "score": d.get("score"),
        "num_comments": d.get("num_comments"),
        "created_utc": d.get("created_utc"),
        "url": f"{WWW}{d['permalink']}" if d.get("permalink") else d.get("url"),
        "external_url": None if d.get("is_self") else d.get("url"),
        "body": (d.get("selftext") or "")[:4000],
    }


def _flatten_comments(children: list, depth: int = 0, out: list | None = None) -> list:
    out = out if out is not None else []
    for c in children:
        if c.get("kind") != "t1":
            continue
        d = c["data"]
        out.append({
            "author": d.get("author"), "score": d.get("score"), "depth": depth,
            "created_utc": d.get("created_utc"), "body": (d.get("body") or "")[:4000],
            "url": f"{WWW}{d['permalink']}" if d.get("permalink") else None,
        })
        replies = d.get("replies")
        if isinstance(replies, dict):
            _flatten_comments(replies["data"]["children"], depth + 1, out)
    return out


def api_listing(token: str, path: str, limit: int, **params) -> list[dict]:
    data = _api(path, token, limit=limit, **params)
    return [_post_from_api(c) for c in data["data"]["children"] if c.get("kind") == "t3"]


def api_thread(token: str, sub: str, post_id: str, limit: int) -> dict:
    data = _api(f"/r/{sub}/comments/{post_id}", token, limit=limit, depth=10, sort="top")
    post = _post_from_api(data[0]["data"]["children"][0])
    post["comments"] = _flatten_comments(data[1]["data"]["children"])[:limit]
    return post


# ── Anonymous Atom backend ───────────────────────────────────────────────────

def _entries(url: str) -> list[dict]:
    data, _ = _get(url)
    root = ET.fromstring(data)
    out = []
    for e in root.findall("a:entry", ATOM):
        link = e.find("a:link", ATOM)
        out.append({
            "title": strip_html(e.findtext("a:title", default="", namespaces=ATOM)),
            "author": (e.findtext("a:author/a:name", default="", namespaces=ATOM) or "").replace("/u/", "") or None,
            "created": e.findtext("a:updated", default="", namespaces=ATOM) or None,
            "url": link.get("href") if link is not None else None,
            "body": strip_html(e.findtext("a:content", default="", namespaces=ATOM))[:4000],
        })
    return out


def atom_listing(path: str, limit: int, **params) -> list[dict]:
    params["limit"] = limit
    return _entries(f"{WWW}{path}.rss?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}")


def atom_thread(sub: str, post_id: str, limit: int) -> dict:
    entries = _entries(f"{WWW}/r/{sub}/comments/{post_id}/.rss?limit={limit}")
    if not entries:
        raise SystemExit("thread feed returned no entries")
    post, comments = entries[0], entries[1:]
    post["comments"] = [{"author": c["author"], "created": c["created"], "body": c["body"], "url": c["url"]} for c in comments]
    post["note"] = ("anonymous feed: scores and nesting unavailable; register a free Reddit script app and set "
                    "REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET (no user login) for full data")
    return post


# ── Commands ─────────────────────────────────────────────────────────────────

def parse_thread_url(url: str) -> tuple[str, str]:
    m = _THREAD_RE.search(url)
    if not m:
        raise SystemExit(f"not a Reddit thread URL: {url}")
    return m.group(1), m.group(2)


def cmd_sub(a, token):
    path = f"/r/{a.name}/{a.sort}"
    if token:
        return api_listing(token, path, a.limit, t=a.time if a.sort == "top" else None)
    return atom_listing(path, a.limit, t=a.time if a.sort == "top" else None)


def cmd_search(a, token):
    path = f"/r/{a.sub}/search" if a.sub else "/search"
    params = {"q": a.query, "sort": a.sort, "restrict_sr": 1 if a.sub else None, "t": a.time}
    return api_listing(token, path, a.limit, **params) if token else atom_listing(path, a.limit, **params)


def cmd_thread(a, token):
    sub, post_id = parse_thread_url(a.url)
    return api_thread(token, sub, post_id, a.limit) if token else atom_thread(sub, post_id, a.limit)


def cmd_user(a, token):
    path = f"/user/{a.name}"
    if token:
        data = _api(f"{path}/overview", token, limit=a.limit)
        out = []
        for c in data["data"]["children"]:
            out.append(_post_from_api(c) if c["kind"] == "t3" else _flatten_comments([c])[0])
        return out
    return atom_listing(path, a.limit)


def cmd_doctor(a, token):
    report = {"oauth_credentials": bool(oauth_credentials()), "user_agent": USER_AGENT}
    if token:
        try:
            _api("/r/announcements/hot", token, limit=1)
            report["active_backend"] = "oauth"
        except (urllib.error.URLError, OSError, KeyError) as exc:
            report["active_backend"] = "oauth (broken)"
            report["oauth_error"] = str(exc)
    else:
        report["active_backend"] = "anonymous-atom"
    try:
        data, headers = _get(f"{WWW}/r/announcements/.rss?limit=1", retry_on_429=False)
        report["anonymous_feed"] = "ok" if b"<feed" in data[:200] else "unexpected body"
        report["anonymous_ratelimit"] = {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}
    except urllib.error.HTTPError as exc:
        report["anonymous_feed"] = f"HTTP {exc.code}"
    report["notes"] = [
        "anonymous .rss needs no account, login, cookie or key; ~1 request/minute per IP",
        "www.reddit.com .json, api.reddit.com and old.reddit are 403 / an empty shell for server IPs",
        "for more than a few calls per task register a free 'script' app at reddit.com/prefs/apps and set "
        "REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET in .env (app-only credentials; Hermes never logs in as the user)",
    ]
    return report


COMMANDS = {"sub": cmd_sub, "search": cmd_search, "thread": cmd_thread, "user": cmd_user, "doctor": cmd_doctor}


def render(cmd: str, result) -> str:
    if cmd == "doctor":
        return "\n".join(f"{k}: {v}" for k, v in result.items())
    if cmd == "thread":
        p = result
        lines = [f"# {p.get('title')}  — u/{p.get('author')}  score={p.get('score', '?')}  {p.get('url')}", p.get("body", "")[:1500], ""]
        for c in p["comments"]:
            indent = "  " * c.get("depth", 0)
            lines.append(f"{indent}- u/{c.get('author')} (score {c.get('score', '?')}): {c.get('body', '')[:600]}")
        if p.get("note"):
            lines.append(f"\n[{p['note']}]")
        return "\n".join(lines)
    lines = []
    for p in result:
        score = f" ↑{p['score']}" if p.get("score") is not None else ""
        nc = f" 💬{p['num_comments']}" if p.get("num_comments") is not None else ""
        lines.append(f"- {p.get('title') or p.get('body', '')[:80]}{score}{nc}  — u/{p.get('author')}\n  {p.get('url')}")
        if p.get("body") and p.get("title"):
            lines.append(f"  {p['body'][:300]}")
    return "\n".join(lines) or "(no results)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sub"); s.add_argument("name"); s.add_argument("--sort", default="hot", choices=["hot", "new", "top", "rising"]); s.add_argument("--time", default="week", choices=["hour", "day", "week", "month", "year", "all"]); s.add_argument("--limit", type=int, default=15)
    q = sub.add_parser("search"); q.add_argument("query"); q.add_argument("--sub"); q.add_argument("--sort", default="relevance", choices=["relevance", "new", "top", "comments"]); q.add_argument("--time", default="all", choices=["hour", "day", "week", "month", "year", "all"]); q.add_argument("--limit", type=int, default=15)
    t = sub.add_parser("thread"); t.add_argument("url"); t.add_argument("--limit", type=int, default=40)
    u = sub.add_parser("user"); u.add_argument("name"); u.add_argument("--limit", type=int, default=15)
    sub.add_parser("doctor")
    args = ap.parse_args(argv)

    creds = oauth_credentials()
    token = None
    if creds:
        try:
            token = oauth_token(*creds)
        except (urllib.error.URLError, OSError, KeyError) as exc:
            print(f"reddit: OAuth token failed ({exc}); falling back to anonymous feeds", file=sys.stderr)
    try:
        result = COMMANDS[args.cmd](args, token)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} for {exc.url}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, ET.ParseError, json.JSONDecodeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else render(args.cmd, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
