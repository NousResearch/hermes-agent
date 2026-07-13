#!/usr/bin/env python3
"""Watch GitHub activity — issues, pulls, releases, or commits — with dedup.

Usage (via cron with --no-agent):

    hermes cron create hermes-issues \\
      --schedule "*/5 * * * *" --no-agent \\
      --script "$HERMES_HOME/skills/devops/watchers/scripts/watch_github.py" \\
      --script-args "--name hermes-issues --repo NousResearch/hermes-agent --scope issues"

Set GITHUB_TOKEN (or GH_TOKEN) in ~/.hermes/.env to avoid the 60 req/hr
anonymous rate limit.

Scopes: issues | pulls | releases | commits.  Or pass --search QUERY to
use the /search/issues endpoint instead of /repos/:owner/:repo/:scope.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _watermark import Watermark, format_items_as_markdown  # type: ignore


VALID_SCOPES = ("issues", "pulls", "releases", "commits")


def _flatten_commit(item):
    """Commit objects nest title/author/date under 'commit' — flatten for rendering."""
    commit = item.get("commit") or {}
    msg = (commit.get("message") or "").strip().splitlines()
    title = msg[0] if msg else ""
    body = "\n".join(msg[1:]).strip() if len(msg) > 1 else ""
    author = (item.get("author") or {}).get("login") or (commit.get("author") or {}).get("name", "")
    date = (commit.get("author") or {}).get("date", "")
    return {
        "id": item.get("sha", ""),
        "title": f"{title}  ({author})" if author else title,
        "url": item.get("html_url"),
        "body": body,
        "created_at": date,
    }


def _flatten_issue_or_release(item):
    return {
        "id": str(item.get("id", "")),
        "title": item.get("title") or item.get("name") or "",
        "url": item.get("html_url") or item.get("url"),
        "body": (item.get("body") or "").strip(),
        "state": item.get("state"),
        "author": (item.get("user") or {}).get("login")
        or (item.get("author") or {}).get("login"),
        "created_at": item.get("created_at"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Watch GitHub issues / pulls / releases / commits.")
    p.add_argument("--name", required=True, help="Watcher name (used for state file)")
    p.add_argument("--repo", default="",
                   help="owner/name of the repo (one of --repo or --search is required)")
    p.add_argument("--scope", default="issues", choices=VALID_SCOPES,
                   help="What to poll (default: issues)")
    p.add_argument("--search", default="",
                   help="GitHub issues search query (alternative to --repo/--scope)")
    p.add_argument("--per-page", type=int, default=30,
                   help="Results per page (default: 30, max: 100)")
    p.add_argument("--max", type=int, default=20,
                   help="Max new items to emit per tick (default: 20)")
    p.add_argument("--with-body", action="store_true",
                   help="Include issue/commit body as a snippet under each item")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="HTTP timeout in seconds (default: 30)")
    args = p.parse_args()

    if not args.repo and not args.search:
        print("watch_github: one of --repo or --search is required", file=sys.stderr)
        return 2
    if args.repo and not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", args.repo):
        print(f"watch_github: --repo must be owner/name (got {args.repo!r})", file=sys.stderr)
        return 2

    # URL + flattening strategy.
    if args.search:
        url = (
            "https://api.github.com/search/issues"
            f"?q={urllib.parse.quote(args.search)}&per_page={args.per_page}"
        )
        flatten = _flatten_issue_or_release
        items_path = "items"
    elif args.scope == "commits":
        url = f"https://api.github.com/repos/{args.repo}/commits?per_page={args.per_page}"
        flatten = _flatten_commit
        items_path = ""
    else:
        url = (
            f"https://api.github.com/repos/{args.repo}/{args.scope}"
            f"?per_page={args.per_page}&state=all"
        )
        flatten = _flatten_issue_or_release
        items_path = ""

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Hermes-Watcher/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url)
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        print(f"watch_github: HTTP {e.code} from {url}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"watch_github: network error: {e}", file=sys.stderr)
        return 2

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"watch_github: response is not valid JSON: {e}", file=sys.stderr)
        return 2

    # Drill into items_path if needed (search endpoint returns {"items":[...]}).
    if items_path:
        data = data.get(items_path) if isinstance(data, dict) else None
    if not isinstance(data, list):
        print(f"watch_github: expected a list of items; got {type(data).__name__}",
              file=sys.stderr)
        return 2

    items = [flatten(i) for i in data if isinstance(i, dict)]
    # Drop any items that flattened without an ID (defensive).
    items = [i for i in items if i.get("id")]

    wm = Watermark.load(args.name)
    new_items = wm.filter_new(items, id_key="id")
    wm.save()

    if args.max > 0:
        new_items = new_items[: args.max]

    body_key = "body" if args.with_body else None
    output = format_items_as_markdown(new_items, body_key=body_key)
    if output:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
