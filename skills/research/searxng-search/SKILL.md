---
name: searxng-search
description: "Search the web using SearXNG metasearch engine. Supports local instance installation, JSON API queries with filtering by time/language/category."
version: 1.0.0
author: Hermes Agent Community
license: MIT
metadata:
  hermes:
    tags: [research, web-search, privacy, metasearch]
    related_skills: [arxiv]
    triggers:
      - "searxng search"
      - "search with searxng"
      - "metasearch"
      - "web search"
      - "search engine"
requirements:
  - python3
  - internet connection (for public instances) OR local SearXNG instance
  - docker (optional, for quick local setup)
---

# SearXNG Search Skill

Universal web search via SearXNG - a privacy-respecting metasearch engine that aggregates results from multiple search engines (Google, DuckDuckGo, Brave, Startpage, etc.) without tracking users.

## What This Skill Does

- **Web Search**: Query multiple search engines simultaneously through SearXNG
- **Privacy**: No user tracking, no search history profiling  
- **Filtering**: Filter by time (day/month/year), language, category
- **Multiple Formats**: JSON, CSV, RSS output
- **Local Instance**: Can run completely offline with local SearXNG server

## When to Use

- Real-time web search is needed
- Privacy-preserving search required
- Need results from multiple search engines aggregated
- User wants unfiltered/different search results than single-engine search
- Research tasks requiring diverse sources

## Quick Reference

| Task | Command |
|------|---------|
| Basic search | `python3 scripts/searxng_search.py "query" -i http://localhost:8080` |
| With filters | `python3 scripts/searxng_search.py "ml tutorial" -i URL -t month -l en` |
| News only | `python3 scripts/searxng_search.py "tech news" -i URL -c news -t day` |
| Demo mode | `python3 scripts/searxng_search.py --demo` |
| Raw JSON | `python3 scripts/searxng_search.py "query" -i URL --raw` |

## Installation

### Quick Start with Docker (Recommended)

```bash
docker run --rm -d \
  -p 8080:8080 \
  -e "BASE_URL=http://localhost:8080" \
  -v "$(pwd)/searxng-settings:/etc/searxng" \
  searxng/searxng
```

Wait 10 seconds, then use `http://localhost:8080`

### From Source (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-babel python3-venv git \
    build-essential libxslt-dev zlib1g-dev libffi-dev libssl-dev

# Clone and setup
mkdir -p /usr/local/searxng
git clone "https://github.com/searxng/searxng" /usr/local/searxng/searxng-src
cd /usr/local/searxng
python3 -m venv searx-pyenv

# Install
./searx-pyenv/bin/pip install -U pip setuptools wheel pyyaml msgspec typing_extensions
cd searxng-src
../searx-pyenv/bin/pip install --use-pep517 --no-build-isolation -e .

# Create config
sudo mkdir -p /etc/searxng
sudo tee /etc/searxng/settings.yml > /dev/null << 'EOF'
use_default_settings: true
general:
  debug: false
  instance_name: "SearXNG-Local"
search:
  safe_search: 0
  autocomplete: 'duckduckgo'
  formats:
    - html
    - json
server:
  secret_key: "random-secret-key-here"
  limiter: false
  image_proxy: true
  bind_address: "0.0.0.0"
  port: 8888
EOF

# Start
export SEARXNG_SETTINGS_PATH="/etc/searxng/settings.yml"
./searx-pyenv/bin/python searx/webapp.py
```

## Usage

### CLI

```bash
python3 scripts/searxng_search.py "query" -i http://localhost:8080
python3 scripts/searxng_search.py "machine learning" -i http://127.0.0.1:8888 -n 10
python3 scripts/searxng_search.py "python tutorial" -i http://localhost:8080 -l ru
python3 scripts/searxng_search.py "tech news" -i http://localhost:8080 -t month -c news
```

### Python API

```python
import sys
sys.path.insert(0, '/root/.hermes/skills/research/searxng-search/scripts')
from searxng_search import searxng_search, format_results

result = searxng_search(
    query="python tutorial",
    instance_url="http://localhost:8080"
)
print(format_results(result))
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | str | required | Search query string |
| instance_url | str | None | SearXNG instance URL (uses public if not set) |
| categories | str | "general" | general, images, videos, news, map, music, it, science, files, social_media |
| language | str | auto | Language code: en, ru, de, fr, etc. |
| time_range | str | None | day, month, year |
| safesearch | int | 0 | 0=Off, 1=Moderate, 2=Strict |
| pageno | int | 1 | Page number |
| format | str | "json" | json, csv, rss |
| fallback | bool | True | Try multiple public instances if one fails |

## Examples by Use Case

### Research & Documentation
```python
result = searxng_search(
    query="tensorflow 2.0 documentation",
    instance_url="http://localhost:8080"
)
```

### Current News
```python
result = searxng_search(
    query="AI news today",
    instance_url="http://localhost:8080",
    categories="news",
    time_range="day"
)
```

### Academic Papers
```python
result = searxng_search(
    query="transformer architecture paper site:arxiv.org",
    instance_url="http://localhost:8080"
)
```

### Multi-Language
```python
# Russian
result = searxng_search(
    query="питон обучение",
    instance_url="http://localhost:8080",
    language="ru"
)
```

## Pitfalls

| Issue | Solution |
|-------|----------|
| "Connection refused" | SearXNG not running. Start with Docker or manual install |
| "403 Forbidden" | JSON API disabled. Use local instance |
| "429 Too Many Requests" | Rate limit on public instance. Add delay or use local |
| Empty results | Check `formats: [json]` in settings.yml |

## Verification

Test the skill works:
```bash
python3 scripts/searxng_search.py --demo
python3 scripts/searxng_search.py "test" -i http://localhost:8080 --raw
```

## Links

- SearXNG Project: https://github.com/searxng/searxng
- Docker Image: https://hub.docker.com/r/searxng/searxng
- Public Instances: https://searx.space/
- API Docs: https://docs.searxng.org/dev/search_api.html
