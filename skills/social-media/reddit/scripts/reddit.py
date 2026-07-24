#!/usr/bin/env python3
"""Reddit CLI helper — read posts, comments, and search subreddits.

Usage:
    python3 reddit.py hot <subreddit> [--limit 10]
    python3 reddit.py new <subreddit> [--limit 10]
    python3 reddit.py top <subreddit> [--limit 10] [--time day|week|month|year|all]
    python3 reddit.py search <query> [--subreddit <sub>] [--limit 10] [--sort relevance|new|hot|top]
    python3 reddit.py post <subreddit> <post_id>
    python3 reddit.py user <username> [--limit 10]
    python3 reddit.py about <subreddit>
    python3 reddit.py multisub <sub1,sub2,sub3> [--limit 10] [--sort hot|new|top]
    python3 reddit.py --json ...          (JSON output for any command)
"""

import json
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone

BASE = "https://www.reddit.com"
UA = "hermes-agent/1.0 (open-source AI assistant)"
TIMEOUT = 15

_json_mode = False


def _get(url: str) -> dict:
    """GET request to Reddit public JSON endpoint."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("Rate limited by Reddit. Wait a moment and try again.", file=sys.stderr)
        elif e.code == 403:
            print("Access denied. The subreddit may be private or quarantined.", file=sys.stderr)
        elif e.code == 404:
            print("Not found. Check the subreddit name or post ID.", file=sys.stderr)
        else:
            print(f"HTTP {e.code}: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def _time_ago(utc_ts: float) -> str:
    """Convert UTC timestamp to human-readable relative time."""
    now = datetime.now(tz=timezone.utc)
    dt = datetime.fromtimestamp(utc_ts, tz=timezone.utc)
    diff = now - dt
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"


def _truncate(text: str, length: int = 300) -> str:
    """Truncate text to given length."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) <= length:
        return text
    return text[:length] + "..."


def _print_post(p: dict, index: int = 0):
    """Print a post summary."""
    prefix = f"{index}. " if index else ""
    flair = f" [{p.get('link_flair_text', '')}]" if p.get("link_flair_text") else ""
    nsfw = " [NSFW]" if p.get("over_18") else ""
    stickied = " [PINNED]" if p.get("stickied") else ""

    print(f"{prefix}{p['title']}{flair}{nsfw}{stickied}")
    print(f"  r/{p['subreddit']} | u/{p['author']} | {_time_ago(p['created_utc'])}")
    print(f"  Score: {p['score']} | Comments: {p['num_comments']} | Upvote: {int(p.get('upvote_ratio', 0) * 100)}%")

    if p.get("selftext"):
        print(f"  {_truncate(p['selftext'], 200)}")
    if p.get("url") and not p.get("is_self"):
        print(f"  Link: {p['url']}")

    print(f"  https://reddit.com{p['permalink']}")
    print()


def _print_comment(c: dict, depth: int = 0):
    """Print a comment with indentation for nesting."""
    indent = "  " * (depth + 1)
    score = c.get("score", 0)
    author = c.get("author", "[deleted]")
    body = _truncate(c.get("body", ""), 400)
    time = _time_ago(c.get("created_utc", 0))

    print(f"{indent}u/{author} ({score} pts) | {time}")
    print(f"{indent}  {body}")
    print()

    replies = c.get("replies")
    if isinstance(replies, dict):
        children = replies.get("data", {}).get("children", [])
        for child in children:
            if child.get("kind") == "t1":
                _print_comment(child["data"], depth + 1)


def cmd_listing(subreddit: str, sort: str = "hot", limit: int = 10, time: str = "day"):
    """Fetch a subreddit listing (hot, new, top, rising)."""
    params = f"?limit={limit}&raw_json=1"
    if sort == "top":
        params += f"&t={time}"

    data = _get(f"{BASE}/r/{urllib.parse.quote(subreddit)}/{sort}.json{params}")
    posts = data.get("data", {}).get("children", [])

    if not posts:
        print(f"No posts found in r/{subreddit}/{sort}")
        return

    if _json_mode:
        result = []
        for c in posts:
            p = c["data"]
            result.append({
                "title": p["title"],
                "author": p["author"],
                "subreddit": p["subreddit"],
                "score": p["score"],
                "num_comments": p["num_comments"],
                "upvote_ratio": p.get("upvote_ratio"),
                "created_utc": p["created_utc"],
                "url": p.get("url"),
                "permalink": f"https://reddit.com{p['permalink']}",
                "selftext": _truncate(p.get("selftext", ""), 500),
                "flair": p.get("link_flair_text"),
                "is_nsfw": p.get("over_18", False),
            })
        print(json.dumps(result, indent=2))
        return

    print(f"r/{subreddit} — {sort.upper()}" + (f" (t={time})" if sort == "top" else "") + f" — {len(posts)} posts\n")
    for i, c in enumerate(posts, 1):
        _print_post(c["data"], index=i)


def cmd_search(query: str, subreddit: str = None, limit: int = 10, sort: str = "relevance"):
    """Search Reddit for posts."""
    params = {
        "q": query,
        "limit": str(limit),
        "sort": sort,
        "raw_json": "1",
    }
    if subreddit:
        params["restrict_sr"] = "on"
        url = f"{BASE}/r/{urllib.parse.quote(subreddit)}/search.json?{urllib.parse.urlencode(params)}"
    else:
        url = f"{BASE}/search.json?{urllib.parse.urlencode(params)}"

    data = _get(url)
    posts = data.get("data", {}).get("children", [])

    if not posts:
        scope = f"r/{subreddit}" if subreddit else "all of Reddit"
        print(f"No results for \"{query}\" in {scope}")
        return

    if _json_mode:
        result = []
        for c in posts:
            p = c["data"]
            result.append({
                "title": p["title"],
                "author": p["author"],
                "subreddit": p["subreddit"],
                "score": p["score"],
                "num_comments": p["num_comments"],
                "permalink": f"https://reddit.com{p['permalink']}",
                "selftext": _truncate(p.get("selftext", ""), 500),
            })
        print(json.dumps(result, indent=2))
        return

    scope = f"r/{subreddit}" if subreddit else "Reddit"
    print(f"Search results for \"{query}\" in {scope} (sort={sort}) — {len(posts)} posts\n")
    for i, c in enumerate(posts, 1):
        _print_post(c["data"], index=i)


def cmd_post(subreddit: str, post_id: str):
    """Fetch a post and its comments."""
    data = _get(f"{BASE}/r/{urllib.parse.quote(subreddit)}/comments/{post_id}.json?limit=20&depth=3&raw_json=1")

    if not isinstance(data, list) or len(data) < 2:
        print("Could not load post.", file=sys.stderr)
        sys.exit(1)

    post = data[0]["data"]["children"][0]["data"]
    comments = data[1]["data"]["children"]

    if _json_mode:
        result = {
            "title": post["title"],
            "author": post["author"],
            "subreddit": post["subreddit"],
            "score": post["score"],
            "num_comments": post["num_comments"],
            "upvote_ratio": post.get("upvote_ratio"),
            "created_utc": post["created_utc"],
            "selftext": post.get("selftext", ""),
            "url": post.get("url"),
            "permalink": f"https://reddit.com{post['permalink']}",
            "comments": [],
        }
        for c in comments:
            if c.get("kind") == "t1":
                cd = c["data"]
                result["comments"].append({
                    "author": cd["author"],
                    "score": cd["score"],
                    "body": cd.get("body", ""),
                    "created_utc": cd["created_utc"],
                })
        print(json.dumps(result, indent=2))
        return

    print(f"{'=' * 60}")
    print(f"{post['title']}")
    print(f"r/{post['subreddit']} | u/{post['author']} | {_time_ago(post['created_utc'])}")
    print(f"Score: {post['score']} | Comments: {post['num_comments']} | Upvote: {int(post.get('upvote_ratio', 0) * 100)}%")
    if post.get("selftext"):
        print(f"\n{post['selftext'][:2000]}")
    if post.get("url") and not post.get("is_self"):
        print(f"\nLink: {post['url']}")
    print(f"{'=' * 60}\n")

    print(f"Top comments ({len([c for c in comments if c.get('kind') == 't1'])}):\n")
    for c in comments:
        if c.get("kind") == "t1":
            _print_comment(c["data"], depth=0)


def cmd_user(username: str, limit: int = 10):
    """Fetch recent posts and comments by a user."""
    data = _get(f"{BASE}/user/{urllib.parse.quote(username)}/overview.json?limit={limit}&raw_json=1")
    items = data.get("data", {}).get("children", [])

    if not items:
        print(f"No activity found for u/{username}")
        return

    if _json_mode:
        result = []
        for item in items:
            d = item["data"]
            entry = {
                "type": "comment" if item["kind"] == "t1" else "post",
                "subreddit": d["subreddit"],
                "author": d["author"],
                "score": d["score"],
                "created_utc": d["created_utc"],
                "permalink": f"https://reddit.com{d['permalink']}",
            }
            if item["kind"] == "t1":
                entry["body"] = _truncate(d.get("body", ""), 500)
                entry["link_title"] = d.get("link_title", "")
            else:
                entry["title"] = d.get("title", "")
                entry["selftext"] = _truncate(d.get("selftext", ""), 500)
            result.append(entry)
        print(json.dumps(result, indent=2))
        return

    print(f"u/{username} — recent activity ({len(items)} items)\n")
    for item in items:
        d = item["data"]
        if item["kind"] == "t1":
            print(f"  [COMMENT] r/{d['subreddit']} | {d.get('link_title', '')[:80]}")
            print(f"  Score: {d['score']} | {_time_ago(d['created_utc'])}")
            print(f"  {_truncate(d.get('body', ''), 200)}")
            print()
        elif item["kind"] == "t3":
            _print_post(d)


def cmd_about(subreddit: str):
    """Fetch subreddit information."""
    data = _get(f"{BASE}/r/{urllib.parse.quote(subreddit)}/about.json?raw_json=1")
    sub = data.get("data", {})

    if not sub:
        print(f"Could not find r/{subreddit}")
        return

    if _json_mode:
        result = {
            "name": sub.get("display_name"),
            "title": sub.get("title"),
            "description": sub.get("public_description", ""),
            "subscribers": sub.get("subscribers", 0),
            "active_users": sub.get("accounts_active", 0),
            "created_utc": sub.get("created_utc"),
            "nsfw": sub.get("over18", False),
            "type": sub.get("subreddit_type"),
            "url": f"https://reddit.com/r/{sub.get('display_name', subreddit)}",
        }
        print(json.dumps(result, indent=2))
        return

    subscribers = sub.get("subscribers", 0)
    active = sub.get("accounts_active", 0)
    created = datetime.fromtimestamp(sub.get("created_utc", 0), tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"r/{sub.get('display_name', subreddit)}")
    print(f"  {sub.get('title', '')}")
    print(f"  Subscribers: {subscribers:,} | Active: {active:,} | Created: {created}")
    print(f"  Type: {sub.get('subreddit_type', 'public')}" + (" | NSFW" if sub.get("over18") else ""))
    desc = sub.get("public_description", "")
    if desc:
        print(f"\n  {_truncate(desc, 500)}")
    print()


def cmd_multisub(subreddits: str, limit: int = 10, sort: str = "hot"):
    """Fetch posts from multiple subreddits combined."""
    subs = "+".join(s.strip() for s in subreddits.split(","))
    data = _get(f"{BASE}/r/{urllib.parse.quote(subs)}/{sort}.json?limit={limit}&raw_json=1")
    posts = data.get("data", {}).get("children", [])

    if not posts:
        print(f"No posts found in r/{subs}/{sort}")
        return

    if _json_mode:
        result = []
        for c in posts:
            p = c["data"]
            result.append({
                "title": p["title"],
                "author": p["author"],
                "subreddit": p["subreddit"],
                "score": p["score"],
                "num_comments": p["num_comments"],
                "permalink": f"https://reddit.com{p['permalink']}",
                "selftext": _truncate(p.get("selftext", ""), 500),
            })
        print(json.dumps(result, indent=2))
        return

    readable = ", ".join(f"r/{s.strip()}" for s in subreddits.split(","))
    print(f"{readable} — {sort.upper()} — {len(posts)} posts\n")
    for i, c in enumerate(posts, 1):
        _print_post(c["data"], index=i)


def _parse_flag(args: list, flag: str, default=None):
    """Extract a flag value from args list."""
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return default


def main():
    global _json_mode
    args = sys.argv[1:]

    if "--json" in args:
        _json_mode = True
        args.remove("--json")

    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return

    cmd = args[0]
    limit = int(_parse_flag(args, "--limit", "10"))

    if cmd in ("hot", "new", "top", "rising") and len(args) >= 2:
        time_filter = _parse_flag(args, "--time", "day")
        cmd_listing(args[1], sort=cmd, limit=limit, time=time_filter)
    elif cmd == "search" and len(args) >= 2:
        query_parts = []
        i = 1
        while i < len(args):
            if args[i].startswith("--"):
                break
            query_parts.append(args[i])
            i += 1
        query = " ".join(query_parts)
        subreddit = _parse_flag(args, "--subreddit")
        sort = _parse_flag(args, "--sort", "relevance")
        cmd_search(query, subreddit=subreddit, limit=limit, sort=sort)
    elif cmd == "post" and len(args) >= 3:
        cmd_post(args[1], args[2])
    elif cmd == "user" and len(args) >= 2:
        cmd_user(args[1], limit=limit)
    elif cmd == "about" and len(args) >= 2:
        cmd_about(args[1])
    elif cmd == "multisub" and len(args) >= 2:
        sort = _parse_flag(args, "--sort", "hot")
        cmd_multisub(args[1], limit=limit, sort=sort)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
