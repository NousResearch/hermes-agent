---
name: olostep
description: Web search, scraping, and AI-powered answers via the Olostep cloud API. Search the web, fetch page content as Markdown, and get AI answers with citations using the Python SDK or REST API.
version: 1.0.0
author: Olostep
license: MIT
metadata:
  hermes:
    tags: [Web Search, Scraping, Research, AI Answers, Cloud API]
    related_skills: [duckduckgo-search, domain-intel, scrapling]
    homepage: https://www.olostep.com
prerequisites:
  commands: [python]
  env: [OLOSTEP_API_KEY]
---

# Olostep

[Olostep](https://www.olostep.com) is a cloud API for web search, URL scraping, and AI-powered research answers. No local browser required — all requests are cloud-based and return clean, structured data.

## When to Use

- Search the web and get a list of relevant links with titles and descriptions
- Scrape a URL and get clean Markdown content without running a browser locally
- Get an AI-generated answer to a research question with web sources
- When scrapling is overkill (no browser needed, no anti-bot bypass required)
- When you need cloud-based scraping that works without local browser install
- Quick research tasks that need both web search and content extraction

## Installation

```bash
pip install olostep
```

Get your API key at [https://www.olostep.com/dashboard](https://www.olostep.com/dashboard), then set it:

```bash
export OLOSTEP_API_KEY=your_key_here
```

## Quick Reference

| Task | Method | Returns |
|------|--------|---------|
| Web search | `requests.post /v1/searches` | `links[]` with title, url, description |
| Scrape URL | `client.scrapes.create()` | `markdown_content` / `html_content` |
| AI answer | `client.answers.create()` | `answer` (str) |
| Async | `AsyncOlostep` | All above, async |

## Python: Web Search

Use the REST API directly (the Python SDK does not have a searches namespace):

```python
import requests
import os

api_key = os.getenv("OLOSTEP_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://api.olostep.com/v1/searches",
    headers=headers,
    json={"query": "best Python web scraping libraries"},
    timeout=30,
)
data = response.json()

for link in data["result"]["links"]:
    print(f"{link['title']}")
    print(f"  {link['url']}")
    print(f"  {link['description']}\n")
```

## Python: Scrape a URL

### Markdown Format

```python
from olostep import Olostep
import os

client = Olostep(api_key=os.getenv("OLOSTEP_API_KEY"))

result = client.scrapes.create(
    url="https://example.com",
    formats=["markdown"]
)

if result.markdown_content:
    print(result.markdown_content)
else:
    print("Could not convert to Markdown")
```

### HTML Format

```python
from olostep import Olostep
import os

client = Olostep(api_key=os.getenv("OLOSTEP_API_KEY"))

result = client.scrapes.create(
    url="https://example.com",
    formats=["html"]
)

if result.html_content:
    print(result.html_content)
else:
    print("Could not fetch HTML")
```

### Both Formats

```python
result = client.scrapes.create(
    url="https://example.com",
    formats=["markdown", "html"]
)

# Use whichever format worked best
content = result.markdown_content or result.html_content
```

## Python: AI Answers

Get an AI-powered answer to a research question:

```python
from olostep import Olostep
import os

client = Olostep(api_key=os.getenv("OLOSTEP_API_KEY"))

result = client.answers.create(
    task="What is the capital of France and what are its major attractions?"
)

print(result.answer)
```

## Python: Async Usage

Use `AsyncOlostep` for concurrent requests:

```python
import asyncio
import os
from olostep import AsyncOlostep

async def main():
    async with AsyncOlostep(api_key=os.getenv("OLOSTEP_API_KEY")) as client:
        # Scrape a URL
        scrape_result = await client.scrapes.create(
            url="https://example.com",
            formats=["markdown"]
        )
        
        # Get an AI answer
        answer_result = await client.answers.create(
            task="Summarize the content from example.com"
        )
        
        print(scrape_result.markdown_content)
        print(answer_result.answer)

asyncio.run(main())
```

## Error Handling

```python
from olostep import Olostep, Olostep_BaseError
import os

client = Olostep(api_key=os.getenv("OLOSTEP_API_KEY"))

try:
    result = client.scrapes.create(
        url="https://example.com",
        formats=["markdown"]
    )
except Olostep_BaseError as e:
    print(f"Olostep error: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

## Pitfalls

- **`OLOSTEP_API_KEY` must be set** — all calls fail silently or raise `Olostep_BaseError` if the API key is missing or invalid
- **The Python SDK does NOT have a `searches` namespace** — use `requests.post()` to call `/v1/searches` directly, do not try `client.searches`
- **`formats=` must be a list, not a string** — `formats=["markdown"]` is correct, `formats="markdown"` will not work
- **`result.markdown_content` may be `None`** — the page might not be convertible to Markdown; always check before using
- **`answers.create()` takes `task=` not `query=`** — the parameter name is `task`, not `query` or `question`
- **Cloud-based means latency** — expect 2-10 seconds per request depending on page complexity; not suitable for real-time applications with strict latency budgets
