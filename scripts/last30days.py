#!/usr/bin/env python3
"""last30days — 搜索社区最近30天讨论，聚合为 briefing

用法：
    python3 last30days.py --query "agent browser playwright" [--output briefing.md] [--max 10]
    python3 last30days.py --query "claude code" --output /tmp/last30days.md
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

SOCKS_PROXY = "socks5h://127.0.0.1:1080"
CUTOFF_TS = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp())

# ============================================================
# Source fetchers
# ============================================================

def _socks_opener():
    import socks
    from sockshandler import SocksiPyHandler
    return urllib.request.build_opener(SocksiPyHandler(socks.SOCKS5, "127.0.0.1", 1080))


def fetch_hn(query: str, max_results: int = 10) -> list[dict]:
    """Hacker News via Algolia API (free, no key). Needs VPN."""
    params = urllib.parse.urlencode({
        "query": query,
        "tags": "story",
        "hitsPerPage": max_results,
        "numericFilters": f"created_at_i>{CUTOFF_TS}",
    })
    url = f"https://hn.algolia.com/api/v1/search?{params}"
    try:
        opener = _socks_opener()
        with opener.open(url, timeout=15) as resp:
            data = json.loads(resp.read())
        results = []
        for hit in data.get("hits", [])[:max_results]:
            results.append({
                "source": "HN",
                "title": hit.get("title", ""),
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                "points": hit.get("points", 0),
                "comments": hit.get("num_comments", 0),
                "date": hit.get("created_at", ""),
            })
        return results
    except Exception as e:
        return [{"source": "HN", "error": str(e)}]


def fetch_reddit(query: str, max_results: int = 10) -> list[dict]:
    """Reddit search via RSS endpoint (bypasses API 403 on VPN IPs). Needs VPN."""
    import xml.etree.ElementTree as ET
    params = urllib.parse.urlencode({"q": query, "sort": "new", "restrict_sr": "off", "limit": max_results})
    url = f"https://www.reddit.com/search.rss?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "last30days/1.0"})
    try:
        opener = _socks_opener()
        with opener.open(req, timeout=15) as resp:
            tree = ET.parse(resp)
    except Exception as e:
        return [{"source": "Reddit", "error": str(e)}]
    results = []
    for entry in tree.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title_el = entry.find("{http://www.w3.org/2005/Atom}title")
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        title = (title_el.text or "").strip() if title_el is not None else ""
        href = link_el.attrib.get("href", "") if link_el is not None else ""
        results.append({"source": "Reddit", "title": title, "url": href})
        if len(results) >= max_results:
            break
    if not results:
        results.append({"source": "Reddit", "error": "no results via RSS"})
    return results


def fetch_v2ex(query: str, max_results: int = 10) -> list[dict]:
    """V2EX via public API (bypasses Cloudflare). Needs VPN (DNS).
    
    V2EX API endpoints are NOT behind Cloudflare — only the web UI is.
    api/topics/latest.json returns ~38 most recent topics.
    We fetch them all and filter locally by keyword.
    """
    url = "https://www.v2ex.com/api/topics/latest.json"
    req = urllib.request.Request(url, headers={"User-Agent": "last30days/1.0"})
    try:
        opener = _socks_opener()
        with opener.open(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return [{"source": "V2EX", "error": str(e)}]

    # filter by keyword (case-insensitive title match)
    keywords = query.lower().split()
    results = []
    for topic in data:
        title = topic.get("title", "")
        title_lower = title.lower()
        if any(kw in title_lower for kw in keywords):
            results.append({
                "source": "V2EX",
                "title": title,
                "url": topic.get("url", f"https://www.v2ex.com/t/{topic.get('id')}"),
                "node": topic.get("node", {}).get("title", ""),
                "replies": topic.get("replies", 0),
            })
            if len(results) >= max_results:
                break

    if not results:
        results.append({"source": "V2EX", "error": f"no topic matched '{query}' in latest {len(data)} posts"})
    return results


def fetch_github_discussions(query: str, max_results: int = 10) -> list[dict]:
    """GitHub Discussions search (no VPN but needs token for rate)."""
    token = _github_token()
    params = urllib.parse.urlencode({
        "q": f"{query} type:discussion created:>={_cutoff_date_str()}",
        "sort": "updated",
        "order": "desc",
        "per_page": max_results,
    })
    url = f"https://api.github.com/search/issues?{params}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "last30days/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append({
                "source": "GitHub",
                "title": item.get("title", ""),
                "url": item.get("html_url", ""),
                "repo": "/".join(item.get("repository_url", "").split("/")[-2:]),
                "date": item.get("updated_at", ""),
            })
        return results
    except Exception as e:
        return [{"source": "GitHub", "error": str(e)}]


# ============================================================
# Helpers
# ============================================================

def _github_token() -> str:
    """Try to read GitHub token from creds vault."""
    try:
        result = subprocess.run(
            ["python3", str(Path.home() / ".hermes/scripts/creds.py"), "get", "github", "token"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            creds = json.loads(result.stdout)
            return creds.get("password", "") or creds.get("token", "")
    except Exception:
        pass
    return ""


def _cutoff_date_str() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")


def _connect_vpn():
    subprocess.run(["bash", str(Path.home() / ".hermes/scripts/vpn-connect.sh"), "sg"],
                   capture_output=True, timeout=20)
    time.sleep(2)


def _disconnect_vpn():
    subprocess.run(["bash", str(Path.home() / ".hermes/scripts/vpn-disconnect.sh")],
                   capture_output=True, timeout=10)


def _format_briefing(query: str, all_results: list[dict]) -> str:
    """Format aggregated results as markdown."""
    lines = [
        f"# last30days briefing: {query}",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"时间窗口：{_cutoff_date_str()} ~ {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    # Group by source
    by_source: dict[str, list[dict]] = {}
    for r in all_results:
        src = r.pop("source", "?")
        by_source.setdefault(src, []).append(r)

    for source in ["HN", "Reddit", "V2EX", "GitHub"]:
        items = by_source.get(source, [])
        lines.append(f"## {source} ({len(items)} 条)")
        lines.append("")
        if items and "error" in items[0]:
            lines.append(f"❌ {items[0]['error']}")
            lines.append("")
            continue
        for item in items:
            title = item.get("title", "(无标题)")
            url = item.get("url", "")
            # build metadata line
            meta_parts = []
            if source == "HN":
                if item.get("points"):
                    meta_parts.append(f"{item['points']} 分")
                if item.get("comments"):
                    meta_parts.append(f"{item['comments']} 评论")
            elif source == "Reddit":
                if item.get("subreddit"):
                    meta_parts.append(item["subreddit"])
                if item.get("score"):
                    meta_parts.append(f"{item['score']} 票")
                if item.get("comments"):
                    meta_parts.append(f"{item['comments']} 评论")
            elif source == "GitHub" and item.get("repo"):
                meta_parts.append(item["repo"])
            meta = " / ".join(meta_parts)
            lines.append(f"- [{title}]({url})" + (f" — {meta}" if meta else ""))
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="last30days — 社区讨论聚合")
    parser.add_argument("--query", "-q", required=True, help="搜索关键词")
    parser.add_argument("--output", "-o", help="输出文件路径（可选，默认打印到 stdout）")
    parser.add_argument("--max", type=int, default=10, help="每源最大条目数（默认 10）")
    parser.add_argument("--sources", default="hn,reddit,v2ex,github",
                        help="逗号分隔的源列表（默认 hn,reddit,v2ex,github）")
    args = parser.parse_args()
    sources = [s.strip() for s in args.sources.split(",")]

    all_results: list[dict] = []
    need_vpn = bool({"hn", "reddit", "v2ex"} & set(sources))

    if need_vpn:
        print("🔗 连接 VPN ...", file=sys.stderr)
        _connect_vpn()

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            if "hn" in sources:
                futures["HN"] = pool.submit(fetch_hn, args.query, args.max)
            if "reddit" in sources:
                futures["Reddit"] = pool.submit(fetch_reddit, args.query, args.max)
            if "v2ex" in sources:
                futures["V2EX"] = pool.submit(fetch_v2ex, args.query, args.max)
            if "github" in sources:
                futures["GitHub"] = pool.submit(fetch_github_discussions, args.query, args.max)

            for name, future in futures.items():
                try:
                    result = future.result(timeout=30)
                    print(f"  ✅ {name}: {len(result) if result and 'error' not in result[0] else 'FAIL'} 条", file=sys.stderr)
                    all_results.extend(result)
                except Exception as e:
                    print(f"  ❌ {name}: {e}", file=sys.stderr)
                    all_results.append({"source": name, "error": str(e)})

    finally:
        if need_vpn:
            _disconnect_vpn()
            print("🔓 已断开 VPN", file=sys.stderr)

    output = _format_briefing(args.query, all_results)

    if args.output:
        Path(args.output).write_text(output)
        print(f"📄 已写入 {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
