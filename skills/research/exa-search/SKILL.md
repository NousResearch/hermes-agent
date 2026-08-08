---
name: exa-search
description: "Use when you need semantic/neural web search that finds documents by meaning not just keywords, with highlighted query-relevant excerpts and date/domain filtering. Requires exa-py and EXA_API_KEY. Free tier: 1000 searches/mo."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [web, search, semantic, neural, highlights, research]
    related_skills: [firecrawl, tavily]
---

# Exa Search

Website: https://exa.ai | Docs: https://docs.exa.ai

## When to Use

* Searching for documents/pages by meaning rather than keyword matching
* Research tasks needing highlighted excerpts showing matched content in context
* Date-filtered news search (e.g. "after March 2026")
* Finding similar pages to a known URL (semantic "more like this")
* Structured output via `outputSchema` for data extraction pipelines

**Trigger examples:**
- "Find articles about AI model releases this month"
- "Search for news on X with highlights"
- "Find pages similar to this URL"
- "Deep research on [topic] with citations"

Exa's neural search understands intent and context, outperforming keyword search for nuanced queries. Its `highlights` feature is especially useful for citation-based work — each result comes back with query-relevant excerpts ready to reference.

## Setup

```bash
pip install exa-py
```

Add to `~/.hermes/.env`:
```
EXA_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

Free tier: 1000 searches/month. Dashboard: https://dashboard.exa.ai

## Core Usage

### Basic Search with Highlights

```python
import os
from exa_py import Exa

exa = Exa(api_key=os.environ.get('EXA_API_KEY'))

results = exa.search(
    'AI model releases March 2026',
    num_results=10,
    highlights=True
)
for r in results.results:
    print(r.url, r.title)
    if r.highlights:
        print(f"  Highlight: {r.highlights[0][:200]}")
```

### News Search with Date Filter

```python
results = exa.search(
    'OpenAI announcements',
    type='auto',
    category='news',
    start_published_date='2026-03-01',   # use start_published_date (not start_date!)
    num_results=10,
    contents={'highlights': True}
)
```

### Domain Filtering

```python
results = exa.search(
    'Kimi K2.6 release',
    domains=['huggingface.co', 'techcrunch.com', 'venturebeat.com'],
    num_results=10,
    highlights=True
)
```

### Find Similar Pages

```python
results = exa.find_similar('https://example.com/article', num_results=5)
for r in results.results:
    print(r.url, r.title, r.score)
```

### Structured Output (outputSchema)

```python
results = exa.search(
    'AI company funding rounds',
    type='auto',
    num_results=10,
    outputSchema={
        'type': 'object',
        'properties': {
            'companies': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'amount': {'type': 'string'}
                    }
                }
            }
        }
    },
    contents={'highlights': True}
)
# Structured result is in: results.results[0].text (or parsed)
```

## Search Type Reference

| Type | Latency | Best For |
|------|---------|----------|
| `auto` | ~1s | Most queries — balanced (default) |
| `fast` | ~450ms | Latency-sensitive queries |
| `instant` | ~250ms | Chat/autocomplete |
| `deep-lite` | ~4s | Cheap synthesis |
| `deep` | ~4-15s | Research, thorough results |
| `deep-reasoning` | ~12-40s | Complex multi-step reasoning |

## Common Pitfalls

1. **Wrong date filter parameter name.** Use `start_published_date` — the parameter `start_date` does NOT exist and will silently be ignored.

2. **Using `domains` vs `includeDomains`.** Exa uses `domains` (list) in the Python SDK, not `includeDomains` (that's the REST API field). In Python: `exa.search(query, domains=[...])`.

3. **`highlights` not showing up.** Highlights must be requested — pass `highlights=True` or `contents={'highlights': True}` in the search call.

4. **Free tier limits.** 1000 searches/mo resets monthly. Set up caching if running recurring research tasks.

5. **Slow response on `deep` types.** Deep search runs multiple query variations. Use `auto` for speed unless you specifically need synthesis quality.

## Verification Checklist

- [ ] `pip install exa-py` completes without error
- [ ] `EXA_API_KEY` set in `~/.hermes/.env`
- [ ] `python -c "from exa_py import Exa; print('OK')"` runs cleanly
- [ ] `exa.search('test', num_results=1, highlights=True)` returns results with highlights