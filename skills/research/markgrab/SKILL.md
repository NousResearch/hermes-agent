---
name: markgrab
description: Universal web content extraction — convert URLs to clean, LLM-ready markdown. Supports HTML (with noise removal and JS-rendering fallback), YouTube transcripts, PDF, and DOCX. Use after web search to get full page content.
version: 0.1.1
author: QuartzUnit
license: MIT
metadata:
  hermes:
    tags: [web-extraction, markdown, content, html, youtube, pdf, docx, rag]
    related_skills: [duckduckgo-search, arxiv, blogwatcher]
prerequisites:
  commands: [markgrab]
---

# MarkGrab — Web Content Extraction

Extract full content from any URL and convert to clean markdown. Complements search skills (duckduckgo-search, arxiv) by fetching the full page content that search results only summarize.

## Setup

```bash
pip install markgrab
```

Optional extras:
```bash
pip install "markgrab[browser]"    # Playwright for JS-rendered pages
pip install "markgrab[youtube]"    # YouTube transcripts
pip install "markgrab[pdf]"        # PDF extraction
pip install "markgrab[docx]"       # DOCX extraction
pip install "markgrab[all]"        # everything
```

## CLI Usage

```bash
# HTML page → markdown (noise removed: nav, sidebar, ads)
markgrab https://example.com/article

# YouTube transcript
markgrab https://youtube.com/watch?v=VIDEO_ID

# PDF extraction
markgrab https://arxiv.org/pdf/1706.03762

# Output formats
markgrab https://example.com --format text     # plain text
markgrab https://example.com --format json     # structured JSON

# Force browser rendering for JS-heavy pages
markgrab https://example.com --browser

# Limit output length
markgrab https://example.com --max-chars 30000
```

## Python API

```python
import asyncio
from markgrab import extract

async def main():
    result = await extract("https://example.com/article")
    print(result.markdown)     # clean markdown
    print(result.title)        # page title
    print(result.word_count)   # word count
    print(result.language)     # detected language

asyncio.run(main())
```

## Workflow: Search then Extract

1. **Search** with duckduckgo-search to find relevant URLs
2. **Extract** full content with markgrab

```bash
# After finding URLs via ddgs
markgrab https://found-url.com/article --max-chars 30000
```

Or in Python:
```python
from ddgs import DDGS
from markgrab import extract
import asyncio

with DDGS() as ddgs:
    results = list(ddgs.text("fastapi deployment", max_results=3))

async def get_content():
    for r in results:
        content = await extract(r["href"], max_chars=30_000)
        print(f"# {content.title}\n{content.markdown[:500]}\n")

asyncio.run(get_content())
```

## Supported Content Types

| Type | Detection | Features |
|------|-----------|----------|
| **HTML** | Default | Content density filtering, noise removal, Playwright auto-fallback |
| **YouTube** | `youtube.com`, `youtu.be` | Transcript extraction with timestamps, multi-language |
| **PDF** | `.pdf` extension or Content-Type | Text extraction with page structure |
| **DOCX** | `.docx` extension | Paragraph and heading extraction |

## Limitations

- **JS-heavy pages**: Require Playwright (`pip install "markgrab[browser]"`). Without it, JS-rendered content may be thin.
- **Rate limiting**: Respects server responses. No built-in throttling — add delays for bulk extraction.
- **No auth**: Cannot extract content behind login walls.
- **Max output**: Defaults to 50,000 characters. Use `--max-chars` to adjust.

## Environment

No environment variables required. No API keys needed — runs entirely locally.
