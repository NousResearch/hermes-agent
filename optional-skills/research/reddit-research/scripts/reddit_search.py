#!/usr/bin/env python3
"""
Reddit Research CLI — search and analyze Reddit discussions via Reddit's JSON API.

Usage:
    python3 reddit_search.py search "Nvidia" --subreddit wallstreetbets --limit 10
    python3 reddit_search.py hot "AI" --hours 24
    python3 reddit_search.py subreddit "cryptocurrency" --since 2025 --limit 20
    python3 reddit_search.py comments abc123 --limit 10
    python3 reddit_search.py author "DeepFuckingValue" --limit 5

No API key required. Uses Reddit's public JSON API.
"""

import json
import sys
import time
import urllib.parse
import urllib.request

REDDIT_BASE = "https://www.reddit.com"
USER_AGENT = "python:research-agent:v1.0 (by /u/research)"


def api_request(url: str) -> dict:
    """Make a Reddit API request with error handling."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"Reddit API returned HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"Reddit API request failed: {e.reason}"}
    except json.JSONDecodeError:
        return {"error": "Reddit API returned invalid JSON"}


def search_submissions(params: dict, limit: int = 10) -> list:
    # Build Reddit search URL
    subreddit = params.pop("subreddit", None)
    if subreddit:
        url = f"{REDDIT_BASE}/r/{subreddit}/search.json"
    else:
        url = f"{REDDIT_BASE}/search.json"

    params["limit"] = limit
    params["raw_json"] = 1
    q = params.pop("q", "")
    params["q"] = q
    if "sort" not in params:
        params["sort"] = "top"
    if "restrict_sr" not in params and subreddit:
        params["restrict_sr"] = 1

    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    data = api_request(full_url)

    results = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        results.append({
            "id": post.get("id"),
            "title": post.get("title", ""),
            "selftext": (post.get("selftext") or "")[:500],
            "subreddit": post.get("subreddit"),
            "author": post.get("author"),
            "score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "created_utc": post.get("created_utc"),
            "url": post.get("url"),
            "permalink": f"https://reddit.com{post.get('permalink', '')}",
            "domain": post.get("domain"),
        })
    return results


def search_comments(params: dict, limit: int = 10) -> list:
    # Use Reddit's comment search
    subreddit = params.pop("subreddit", None)
    if subreddit:
        url = f"{REDDIT_BASE}/r/{subreddit}/comments.json"
    else:
        url = f"{REDDIT_BASE}/comments.json"

    params["limit"] = limit
    params["raw_json"] = 1
    if "q" in params:
        url = f"{REDDIT_BASE}/search.json"
        params["restrict_sr"] = 1 if subreddit else 0

    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    data = api_request(full_url)

    results = []
    for child in data.get("data", {}).get("children", []):
        comment = child.get("data", {})
        results.append({
            "id": comment.get("id"),
            "body": (comment.get("body") or "")[:500],
            "subreddit": comment.get("subreddit"),
            "author": comment.get("author"),
            "score": comment.get("score", 0),
            "created_utc": comment.get("created_utc"),
            "permalink": f"https://reddit.com{comment.get('permalink', '')}",
            "parent_id": comment.get("parent_id"),
            "link_id": comment.get("link_id"),
        })
    return results


def cmd_search(args):
    params = {"q": args.query, "sort": "score", "sort_type": "score", "order": "desc"}
    if args.subreddit:
        params["subreddit"] = args.subreddit
    if args.since:
        params["after"] = str(int(time.time()) - args.since * 86400)
    if args.author:
        params["author"] = args.author
    results = search_submissions(params, args.limit)
    output = {"query": args.query, "total_found": len(results), "results": results}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_hot(args):
    after = int(time.time()) - (args.hours * 3600)
    params = {
        "q": args.query,
        "after": str(after),
        "sort": "score",
        "sort_type": "score",
        "order": "desc",
    }
    if args.subreddit:
        params["subreddit"] = args.subreddit
    results = search_submissions(params, args.limit)
    output = {"query": args.query, "hours": args.hours, "results": results}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_subreddit(args):
    params = {
        "subreddit": args.name,
        "sort": "score",
        "sort_type": "score",
        "order": "desc",
    }
    if args.since:
        params["after"] = str(int(time.time()) - args.since * 86400)
    results = search_submissions(params, args.limit)
    output = {"subreddit": args.name, "results": results}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_comments(args):
    results = fetch_post_comments(args.post_id, args.limit)
    output = {"post_id": args.post_id, "results": results}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_author(args):
    params = {
        "author": args.name,
        "sort": "score",
        "sort_type": "score",
        "order": "desc",
    }
    results = search_submissions(params, args.limit)
    output = {"author": args.name, "results": results}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    command = sys.argv[1]

    if command == "search":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("query")
        p.add_argument("--subreddit")
        p.add_argument("--since", type=int, help="Days to look back")
        p.add_argument("--author")
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_search(args)
    elif command == "hot":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("query")
        p.add_argument("--hours", type=int, default=24)
        p.add_argument("--subreddit")
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_hot(args)
    elif command == "subreddit":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("name")
        p.add_argument("--since", type=int, help="Days to look back")
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_subreddit(args)
    elif command == "comments":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("query")
        p.add_argument("--subreddit")
        p.add_argument("--since", type=int)
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_comments(args)
    elif command == "author":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("name")
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_author(args)
    elif command in ("--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
