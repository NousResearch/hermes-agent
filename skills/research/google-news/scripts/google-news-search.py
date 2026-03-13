#!/usr/bin/env python3
"""
Google News — Full access via Scrapling CLI.
Supports: homepage, category pages, and search queries.
Returns structured JSON with title, source, URL for each article.

Usage:
  python3 google-news-search.py                          # Homepage
  python3 google-news-search.py technology               # Category
  python3 google-news-search.py search "AI agents"       # Search
  python3 google-news-search.py categories               # List all categories
  python3 google-news-search.py technology GB en 10      # Category + region + limit

Categories: homepage, us, world, business, technology, entertainment,
            sports, science, health, local
"""

import subprocess
import tempfile
import re
import json
import sys
import os

# Script directory for loading config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'google-news-categories.json')

# Load categories from config, fallback to hardcoded
def load_categories():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        cats = {}
        for key, val in config['categories'].items():
            cats[key] = val.get('id')
        return cats, config['categories']
    except:
        # Fallback
        return {
            "homepage": None,
            "us": "CAAqIggKIhxDQkFTRHdvSkwyMHZNRGxqTjNjd0VnSmxiaWdBUAE",
            "world": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB",
            "business": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
            "technology": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
            "entertainment": "CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB",
            "sports": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB",
            "science": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB",
            "health": "CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ",
            "local": "CAAqHAgKIhZDQklTQ2pvSWJHOWpZV3hmZGpJb0FBUAE",
        }, None

CATEGORIES, CATEGORY_META = load_categories()

# Aliases for convenience
ALIASES = {
    "home": "homepage",
    "top": "homepage",
    "top_stories": "homepage",
    "tech": "technology",
    "biz": "business",
    "ent": "entertainment",
    "sci": "science",
}


def build_url(category, region="US", language="en", query=None):
    """Build Google News URL for category or search."""
    hl = f"{language}-{region}"
    ceid = f"{region}:{language}"
    
    if query:
        return f"https://news.google.com/search?q={query.replace(' ', '+')}&hl={hl}&gl={region}&ceid={ceid}"
    
    cat_id = CATEGORIES.get(category)
    if cat_id is None:
        return f"https://news.google.com/home?hl={hl}&gl={region}&ceid={ceid}"
    else:
        return f"https://news.google.com/topics/{cat_id}?hl={hl}&gl={region}&ceid={ceid}"


def fetch_and_parse(url, max_articles=20):
    """Fetch Google News page and parse articles."""
    html_file = tempfile.mktemp(suffix='.html')
    
    cmd = [
        "scrapling", "extract", "fetch", url, html_file,
        "--headless", "--disable-resources",
        "--timeout", "20000", "--wait", "3000"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {"error": f"Fetch failed: {result.stderr}", "articles": []}
    
    from scrapling.parser import Selector
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    
    sel = Selector(html)
    
    # Parse from aria-labels
    aria_links = sel.css('a[aria-label]')
    articles = []
    seen = set()
    
    for a in aria_links:
        label = a.attrib.get('aria-label', '')
        href = a.attrib.get('href', '')
        
        if len(label) < 20 or not ('./read/' in href or './articles/' in href):
            continue
        if label in seen:
            continue
        seen.add(label)
        
        if href.startswith('./read/'):
            article_url = f"https://news.google.com/read/{href[7:]}"
        elif href.startswith('./articles/'):
            article_url = f"https://news.google.com/articles/{href[12:]}"
        else:
            article_url = href
        
        label = label.strip().rstrip(' -')
        parts = label.split(' - ', 1)
        
        if len(parts) == 2:
            title = parts[0].strip()
            source = parts[1].strip().split(' - ')[0].strip()
        else:
            title = label
            source = ''
        
        if len(title) >= 15:
            articles.append({
                'title': title,
                'source': source,
                'url': article_url
            })
    
    # Cleanup
    try:
        os.unlink(html_file)
    except:
        pass
    
    if max_articles:
        articles = articles[:max_articles]
    
    return articles


def list_categories():
    """List all available categories with descriptions."""
    if CATEGORY_META:
        result = []
        for key, meta in sorted(CATEGORY_META.items()):
            result.append({
                'id': key,
                'label': meta.get('label', key),
                'type': meta.get('type', 'unknown'),
                'description': meta.get('description', ''),
                'has_url': meta.get('url') is not None
            })
        return result
    else:
        return [{'id': k, 'label': k.title()} for k in CATEGORIES.keys()]


def main():
    args = sys.argv[1:]
    
    category = "homepage"
    region = "US"
    language = "en"
    max_results = 20
    query = None
    
    if not args:
        pass
    elif args[0] == "search":
        query = args[1] if len(args) > 1 else "AI agents"
        region = args[2] if len(args) > 2 else "US"
        language = args[3] if len(args) > 3 else "en"
        max_results = int(args[4]) if len(args) > 4 else 20
    elif args[0] == "categories":
        cats = list_categories()
        print(json.dumps({"total": len(cats), "categories": cats}, indent=2))
        return
    else:
        cat = args[0].lower()
        category = ALIASES.get(cat, cat)
        
        if category not in CATEGORIES:
            print(json.dumps({
                "error": f"Unknown category: {category}",
                "available": list(CATEGORIES.keys()),
                "hint": "Use 'categories' command to list all"
            }))
            return
        
        region = args[1] if len(args) > 1 else "US"
        language = args[2] if len(args) > 2 else "en"
        max_results = int(args[3]) if len(args) > 3 else 20
    
    url = build_url(category, region, language, query)
    articles = fetch_and_parse(url, max_results)
    
    result = {
        "category": category if not query else "search",
        "query": query,
        "region": region,
        "language": language,
        "total": len(articles),
        "articles": articles
    }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
