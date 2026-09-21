#!/usr/bin/env python3
"""
Dezzy Design Research pre-processor — fetches from design-focused sources,
filters against seen-items cache, outputs ONLY new items for LLM synthesis.

Sources (early signal first):
1. GitHub: new repos tagged design-system, component-library, ui-kit, tailwind, react-native
2. npm: new RN/Expo UI component packages
3. Product Hunt RSS: design/dev tool launches
4. Design blog RSS feeds (Smashing, CSS-Tricks, UX Collective, Expo blog)
5. X design community (early-signal accounts)

Focus: UI/UX quality, reliability, trending/hot, new tools, patterns, NOT stack introspection.
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)
import json, os, sys, re, subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

RESEARCH_DIR = Path.home() / ".hermes" / "research"
SEEN_CACHE = RESEARCH_DIR / "seen-items.json"
OUTPUT_FILE = Path("/tmp/dezzy-design-candidates.json")
TODAY = datetime.now().strftime("%Y-%m-%d")

def load_seen():
    if SEEN_CACHE.exists():
        data = json.loads(SEEN_CACHE.read_text())
        return data.get("items", {})
    return {}

def save_seen(seen):
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    pruned = {k: v for k, v in seen.items() if v.get("first_seen", "") > cutoff}
    SEEN_CACHE.write_text(json.dumps({"items": pruned, "last_cleanup": datetime.now().isoformat()}, indent=2))

def fetch_url(url, timeout=15):
    try:
        req = Request(url, headers={"User-Agent": "KenseiBot/1.0"})
        return urlopen(req, timeout=timeout).read().decode("utf-8", errors="replace")
    except:
        return None

def run_cmd(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except:
        return None

def is_new(source, item_id, seen):
    return f"{source}:{item_id}" not in seen

def mark_seen(source, item_id, seen, title=""):
    seen[f"{source}:{item_id}"] = {"first_seen": datetime.now().isoformat(), "title": title[:100]}

def parse_rss(text, source):
    items = []
    entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
    if not entries:
        entries = re.findall(r'<item>(.*?)</item>', text, re.DOTALL)
    for entry in entries:
        title_m = re.search(r'<title[^>]*>(.*?)</title>', entry, re.DOTALL)
        link_m = re.search(r'<link[^>]*href="([^"]*)"', entry) or re.search(r'<link>(.*?)</link>', entry, re.DOTALL)
        title = title_m.group(1).strip() if title_m else ""
        link = link_m.group(1).strip() if link_m else ""
        if title:
            items.append({"title": title, "url": link, "source": source})
    return items

def main():
    seen = load_seen()
    candidates = []

    # 1. GitHub new repos — design-focused topics
    date_3d = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    design_topics = [
        ("design-system", "Design System"),
        ("component-library", "Component Library"),
        ("ui-kit", "UI Kit"),
        ("tailwind", "Tailwind/CSS"),
        ("react-native", "React Native UI"),
        ("figma-plugin", "Figma Plugin"),
        ("design-tokens", "Design Tokens"),
        ("animation", "Animation/Motion"),
    ]
    for topic, label in design_topics:
        result = run_cmd(f'gh api "/search/repositories?q=created:>{date_3d}+stars:>5+topic:{topic}&sort=stars&order=desc&per_page=3" --jq \'.items[] | "{{\\"id\\":\\"\\(.id)\\",\\"name\\":\\"\\(.full_name)\\",\\"stars\\":\\(.stargazers_count),\\"desc\\":\\"\\(.description // \\"\\")\\",\\"url\\":\\"\\(.html_url)\\"}},"\' 2>/dev/null')
        if result and result.strip():
            result = "[" + result.strip().rstrip(",") + "]"
            try:
                repos = json.loads(result)
                for repo in repos:
                    if is_new("design-gh", repo["id"], seen):
                        candidates.append({
                            "source": f"GitHub {label}",
                            "title": f"{repo['name']} — {repo['desc'][:100]}",
                            "url": repo["url"],
                            "stars": repo["stars"],
                            "freshness": "today"
                        })
                        mark_seen("design-gh", repo["id"], seen, repo["name"])
            except:
                pass

    # 2. npm new RN/UI component packages
    npm_queries = [
        "keywords:react-native+keywords:ui+keywords:component",
        "keywords:expo+keywords:ui",
        "keywords:tailwind+keywords:react-native",
        "keywords:design-system+keywords:react",
    ]
    for query in npm_queries:
        raw = fetch_url(f"https://registry.npmjs.org/-/v1/search?text={query}&size=5")
        if raw:
            try:
                data = json.loads(raw)
                for obj in data.get("objects", []):
                    pkg = obj["package"]
                    pkg_key = pkg["name"]
                    if is_new("npm-design", pkg_key, seen):
                        candidates.append({
                            "source": "npm Package",
                            "title": f"{pkg['name']} v{pkg['version']}",
                            "url": pkg.get("links", {}).get("npm", f"https://www.npmjs.com/package/{pkg['name']}"),
                            "description": pkg.get("description", "")[:120],
                            "freshness": "today"
                        })
                        mark_seen("npm-design", pkg_key, seen, pkg["name"])
            except:
                pass

    # 3. Product Hunt RSS (filter for design/dev entries)
    raw = fetch_url("https://www.producthunt.com/feed")
    if raw:
        items = parse_rss(raw, "Product Hunt")
        for item in items[:20]:
            item_key = item["url"] or item["title"]
            if is_new("design-ph", item_key, seen):
                candidates.append({
                    "source": "Product Hunt",
                    "title": item["title"],
                    "url": item["url"],
                    "freshness": "today"
                })
                mark_seen("design-ph", item_key, seen, item["title"])

    # 4. Design blog RSS feeds
    blog_feeds = {
        "Smashing Magazine": "https://www.smashingmagazine.com/feed/",
        "CSS-Tricks": "https://css-tricks.com/feed/",
        "Expo Blog": "https://expo.dev/blog/rss.xml",
    }
    for name, feed_url in blog_feeds.items():
        raw = fetch_url(feed_url)
        if raw:
            items = parse_rss(raw, name)
            for item in items[:3]:
                item_key = item["url"] or item["title"]
                if is_new(f"design-blog-{name}", item_key, seen):
                    candidates.append({
                        "source": f"{name} Blog",
                        "title": item["title"],
                        "url": item["url"],
                        "freshness": "today"
                    })
                    mark_seen(f"design-blog-{name}", item_key, seen, item["title"])

    # 5. GitHub releases for design-focused repos
    design_repos = [
        "shadcn-ui/ui", "tailwindlabs/tailwindcss", "expo/expo",
        "software-mansion/react-native-reanimated", "software-mansion/react-native-gesture-handler",
        "gluestack/gluestack-ui", "radix-ui/primitives", "microsoft/fluentui",
        "adobe/react-spectrum", "shopify/polaris", "carbon-design-system/carbon-design-system",
        "founded-labs/react-native-reusables",
    ]
    for repo in design_repos:
        result = run_cmd(f"gh release list -R {repo} -L 1 --json tagName,publishedAt 2>/dev/null")
        if result:
            try:
                releases = json.loads(result)
                if releases:
                    rel = releases[0]
                    pub_date = rel.get("publishedAt", "")
                    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                    if pub_date and pub_date > cutoff:
                        rel_id = f"{repo}:{rel['tagName']}"
                        if is_new("design-release", rel_id, seen):
                            candidates.append({
                                "source": "Design Release",
                                "title": f"{repo} — {rel['tagName']}",
                                "url": f"https://github.com/{repo}/releases",
                                "freshness": "today" if pub_date > (datetime.now() - timedelta(days=1)).isoformat() else "recent"
                            })
                            mark_seen("design-release", rel_id, seen, rel["tagName"])
            except:
                pass

    # Write output
    output = {
        "generated_at": datetime.now().isoformat(),
        "date": TODAY,
        "total_candidates": len(candidates),
        "candidates": candidates
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    save_seen(seen)

    print(f"Dezzy pre-processor: {len(candidates)} new design candidates from {len(set(c['source'] for c in candidates))} sources")
    print(f"Seen cache: {len(seen)} items tracked")
    print(f"Output: {OUTPUT_FILE}")

    if candidates:
        print("\n=== DESIGN CANDIDATES ===")
        for c in candidates:
            print(f"[{c['source']}] {c['title'][:80]}")

    return 0

if __name__ == "__main__":
    sys.exit(main())