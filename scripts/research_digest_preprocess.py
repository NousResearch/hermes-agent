#!/usr/bin/env python3
"""
Research Digest pre-processor — fetches from 15+ programmatic sources,
filters against seen-items cache, outputs ONLY new items as JSON for LLM synthesis.

Sources (early signal first):
1. HN Show HN (new tool debuts BEFORE front page)
2. HN newest AI stories
3. GitHub API: new repos created in last 48h with growing stars (AI/agent/LLM topics)
4. GitHub releases for tracked repos
5. HuggingFace newest models (not just trending)
6. HuggingFace daily papers
7. npm registry: new AI/tool packages
8. Product Hunt RSS: new product launches
9. Official blog RSS feeds (Anthropic, OpenAI, Google AI, Ollama)
10. llm-stats.com changelog (model releases/pricing changes)
11. arXiv RSS (raw new AI submissions)

Output: JSON array of new items to /tmp/research-digest-candidates.json
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)
import json, os, sys, re, subprocess, time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

RESEARCH_DIR = Path.home() / ".hermes" / "research"
SEEN_CACHE = RESEARCH_DIR / "seen-items.json"
OUTPUT_FILE = Path("/tmp/research-digest-candidates.json")
TODAY = datetime.now().strftime("%Y-%m-%d")
CUTOFF = (datetime.now() - timedelta(hours=48)).isoformat()

def load_seen():
    if SEEN_CACHE.exists():
        data = json.loads(SEEN_CACHE.read_text())
        return data.get("items", {})
    return {}

def save_seen(seen):
    # Prune items older than 30 days
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    pruned = {k: v for k, v in seen.items() if v.get("first_seen", "") > cutoff}
    SEEN_CACHE.write_text(json.dumps({"items": pruned, "last_cleanup": datetime.now().isoformat()}, indent=2))

def fetch_url(url, headers=None, timeout=15):
    try:
        req = Request(url, headers=headers or {"User-Agent": "KenseiBot/1.0"})
        return urlopen(req, timeout=timeout).read().decode("utf-8", errors="replace")
    except Exception as e:
        return None

def fetch_json(url, headers=None, timeout=15):
    raw = fetch_url(url, headers, timeout)
    if raw:
        try:
            return json.loads(raw)
        except:
            return None
    return None

def run_cmd(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except:
        return None

def make_key(source, item_id):
    return f"{source}:{item_id}"

def is_new(source, item_id, seen):
    return make_key(source, item_id) not in seen

def mark_seen(source, item_id, seen, title=""):
    key = make_key(source, item_id)
    seen[key] = {"first_seen": datetime.now().isoformat(), "title": title[:100]}

def parse_rss(text, source):
    """Extract titles and links from RSS/Atom feed."""
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

    # 1. HN Show HN (new tool debuts)
    data = fetch_json("https://hn.algolia.com/api/v1/search_by_date?tags=story,show_hn&query=AI&hitsPerPage=20")
    if data:
        for hit in data.get("hits", []):
            item_id = hit.get("objectID", "")
            title = hit.get("title", "")
            if is_new("hn-showhn", item_id, seen) and hit.get("points", 0) >= 5:
                candidates.append({
                    "source": "HN Show HN",
                    "title": title,
                    "url": f"https://news.ycombinator.com/item?id={item_id}",
                    "points": hit.get("points", 0),
                    "age_hours": (datetime.now() - datetime.fromtimestamp(hit.get("created_at_i", 0))).total_seconds() / 3600,
                    "freshness": "today" if (datetime.now() - datetime.fromtimestamp(hit.get("created_at_i", 0))).total_seconds() < 86400 else "yesterday"
                })
                mark_seen("hn-showhn", item_id, seen, title)

    # 2. HN newest AI stories (broader)
    data = fetch_json("https://hn.algolia.com/api/v1/search_by_date?tags=story&query=AI+LLM+agent&hitsPerPage=15")
    if data:
        for hit in data.get("hits", []):
            item_id = hit.get("objectID", "")
            title = hit.get("title", "")
            if is_new("hn-ai", item_id, seen) and hit.get("points", 0) >= 15:
                candidates.append({
                    "source": "HN AI",
                    "title": title,
                    "url": f"https://news.ycombinator.com/item?id={item_id}",
                    "points": hit.get("points", 0),
                    "freshness": "today" if (datetime.now() - datetime.fromtimestamp(hit.get("created_at_i", 0))).total_seconds() < 86400 else "yesterday"
                })
                mark_seen("hn-ai", item_id, seen, title)

    # 3. GitHub new repos (created in last 48h, stars > 10, AI topics)
    date_2d = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    for topic in ["ai-agent", "llm", "mcp", "coding-agent"]:
        result = run_cmd(f'gh api "/search/repositories?q=created:>{date_2d}+stars:>10+topic:{topic}&sort=stars&order=desc&per_page=5" --jq \'.items[] | "{{\\"id\\":\\"\\(.id)\\",\\"name\\":\\"\\(.full_name)\\",\\"stars\\":\\(.stargazers_count),\\"desc\\":\\"\\(.description // \\"\\")\\",\\"url\\":\\"\\(.html_url)\\"}},"\' 2>/dev/null')
        if result and result.strip():
            # Parse the JSON-like output
            result = "[" + result.strip().rstrip(",") + "]"
            try:
                repos = json.loads(result)
                for repo in repos:
                    if is_new("github-new", repo["id"], seen):
                        candidates.append({
                            "source": "GitHub New Repo",
                            "title": f"{repo['name']} — {repo['desc'][:80]}",
                            "url": repo["url"],
                            "stars": repo["stars"],
                            "topic": topic,
                            "freshness": "today"
                        })
                        mark_seen("github-new", repo["id"], seen, repo["name"])
            except:
                pass

    # 4. GitHub releases for tracked repos
    tracked_repos = [
        "ollama/ollama", "vllm-project/vllm", "ggml-org/llama.cpp",
        "anthics/claude-code", "openai/codex", "anthropics/anthropic-sdk-python",
        "expo/expo", "vercel/next.js", "tailwindlabs/tailwindcss",
        "shadcn-ui/ui", "huggingface/transformers"
    ]
    for repo in tracked_repos:
        result = run_cmd(f"gh release list -R {repo} -L 1 --json tagName,publishedAt 2>/dev/null")
        if result:
            try:
                releases = json.loads(result)
                if releases:
                    rel = releases[0]
                    pub_date = rel.get("publishedAt", "")
                    if pub_date and pub_date > CUTOFF:
                        rel_id = f"{repo}:{rel['tagName']}"
                        if is_new("gh-release", rel_id, seen):
                            candidates.append({
                                "source": "GitHub Release",
                                "title": f"{repo} — {rel['tagName']}",
                                "url": f"https://github.com/{repo}/releases/tag/{rel['tagName']}",
                                "freshness": "today" if pub_date > (datetime.now() - timedelta(days=1)).isoformat() else "yesterday"
                            })
                            mark_seen("gh-release", rel_id, seen, rel["tagName"])
            except:
                pass

    # 5. HuggingFace newest models (filter out personal/random uploads)
    data = fetch_json("https://huggingface.co/api/models?sort=lastModified&direction=-1&limit=50")
    if data and isinstance(data, list):
        for model in data[:50]:
            model_id = model.get("id", model.get("modelId", ""))
            if not "/" in model_id or not is_new("hf-model", model_id, seen):
                continue
            # Skip personal/random uploads — look for org-backed or trending models
            org = model_id.split("/")[0].lower()
            skip_orgs = {"user", "test", "demo", "tmp", "dev"}
            # Only include if from a known org OR has high downloads
            downloads = model.get("downloads", 0)
            known_orgs = {"meta-llama", "mistralai", "qwen", "deepseek-ai", "google", "microsoft",
                         "anthropic", "nvidia", "openai", "stabilityai", "blackforestlabs",
                         "NousResearch", "teknium", "allenai", "bigscience", "facebook", "microsoft",
                         "huggingface", "Qwen", "microsoft", "aisingapore", "internlm", "01-ai"}
            if org in known_orgs or downloads > 100:
                candidates.append({
                    "source": "HuggingFace Model",
                    "title": model_id,
                    "url": f"https://huggingface.co/{model_id}",
                    "downloads": downloads,
                    "freshness": "today"
                })
                mark_seen("hf-model", model_id, seen, model_id)

    # 6. HuggingFace daily papers
    data = fetch_json("https://huggingface.co/api/daily_papers?limit=10")
    if data and isinstance(data, list):
        for paper in data[:10]:
            paper_id = paper.get("paper", {}).get("id", "")
            title = paper.get("paper", {}).get("title", "")
            if is_new("hf-paper", paper_id, seen) and paper_id:
                candidates.append({
                    "source": "HuggingFace Paper",
                    "title": title[:120],
                    "url": f"https://huggingface.co/papers/{paper_id}",
                    "freshness": "today"
                })
                mark_seen("hf-paper", paper_id, seen, title)

    # 7. Product Hunt RSS
    raw = fetch_url("https://www.producthunt.com/feed")
    if raw:
        items = parse_rss(raw, "Product Hunt")
        for item in items[:15]:
            item_key = item["url"] or item["title"]
            if is_new("producthunt", item_key, seen):
                candidates.append({
                    "source": "Product Hunt",
                    "title": item["title"],
                    "url": item["url"],
                    "freshness": "today"
                })
                mark_seen("producthunt", item_key, seen, item["title"])

    # 8. Official blog RSS feeds
    blog_feeds = {
        "Anthropic": "https://www.anthropic.com/news/rss.xml",
        "Ollama": "https://ollama.com/blog/rss.xml",
    }
    for name, feed_url in blog_feeds.items():
        raw = fetch_url(feed_url)
        if raw:
            items = parse_rss(raw, name)
            for item in items[:3]:
                item_key = item["url"] or item["title"]
                if is_new(f"blog-{name}", item_key, seen):
                    candidates.append({
                        "source": f"{name} Blog",
                        "title": item["title"],
                        "url": item["url"],
                        "freshness": "today"
                    })
                    mark_seen(f"blog-{name}", item_key, seen, item["title"])

    # Write candidates for LLM synthesis
    output = {
        "generated_at": datetime.now().isoformat(),
        "date": TODAY,
        "total_candidates": len(candidates),
        "candidates": candidates
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    
    # Save seen cache
    save_seen(seen)
    
    print(f"Pre-processor complete: {len(candidates)} new candidates from {len(set(c['source'] for c in candidates))} sources")
    print(f"Seen cache: {len(seen)} items tracked")
    print(f"Output: {OUTPUT_FILE}")
    
    # Also print a compact summary for the LLM context injection
    if candidates:
        print("\n=== CANDIDATES FOR LLM SYNTHESIS ===")
        for c in candidates:
            print(f"[{c['source']}] {c['title'][:80]} → {c.get('url', '')[:60]}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())