---
name: exa
description: Optional vendor skill for Exa — AI-native web search, contents retrieval, and find-similar. Prefer JSON output and non-interactive flows.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, Web, Search, Contents, CLI]
    related_skills: [duckduckgo-search, parallel-cli]
---

# Exa

Use `exa` when the user explicitly wants Exa, or when a workflow would benefit from Exa's search, contents retrieval, or find-similar capabilities.

This is an optional third-party workflow, not a Hermes core capability.

Important expectations:
- Exa is a paid service with a free tier, not a fully free local tool.
- It overlaps with Hermes native `web_search` / `web_extract`, so do not prefer it by default for ordinary lookups.
- Prefer this skill when the user mentions Exa specifically or needs capabilities like Exa's find-similar, deep search, or answer workflows.

## When to use it

Prefer this skill when:
- The user explicitly mentions Exa
- The task needs semantic or neural search rather than keyword matching
- You need find-similar functionality to discover pages related to a given URL
- You need Exa's answer endpoint for direct question answering with citations

Prefer Hermes native `web_search` / `web_extract` for quick one-off lookups when Exa is not specifically requested.

## Installation

```bash
pip install exa-py
```

## Authentication

API key environment variable:

```bash
export EXA_API_KEY="your-api-key"
```

Get your API key at: https://dashboard.exa.ai/api-keys

## Core rule set

1. Always use the Python SDK (`exa_py`) for programmatic access.
2. Cite only URLs returned by Exa output.
3. Save large outputs to a temp file when follow-up questions are likely.
4. Prefer Hermes native tools unless the user wants Exa specifically or needs Exa-only workflows.

## Quick reference

```python
from exa_py import Exa

exa = Exa(api_key="your-api-key")

# Search
results = exa.search("query", num_results=10, contents={"text": True})

# Find similar pages
results = exa.find_similar("https://example.com", contents={"text": True})

# Get contents from URLs
results = exa.get_contents(["https://example.com"], text=True)

# Answer a question
response = exa.answer("What is the capital of France?")
```

## Search

Use for current web lookups with structured results.

```python
# Basic search with text contents
results = exa.search("latest AI research papers", contents={"text": True})

# Search with filters
results = exa.search(
    "climate tech startups",
    num_results=20,
    start_published_date="2025-01-01",
    include_domains=["techcrunch.com", "wired.com"],
    contents={"text": True, "highlights": True},
)

# Deep search for complex queries
results = exa.search(
    "What are the latest battery breakthroughs?",
    type="deep",
    contents={"text": True},
)
```

Useful parameters:
- `num_results` for controlling result count
- `include_domains` / `exclude_domains` for domain filtering
- `start_published_date` / `end_published_date` for date filtering
- `type="deep"` for more thorough search on complex queries
- `contents={"text": True, "highlights": True}` for inline content retrieval

## Contents retrieval

Use to pull clean text content from URLs.

```python
results = exa.get_contents(
    ["https://example.com"],
    text=True,
)
```

Options:
- `text=True` or `text={"max_characters": 5000}` for text content
- `highlights=True` for key excerpts
- `livecrawl="always"` to force fresh crawling

## Find similar

Use to discover pages similar to a given URL.

```python
results = exa.find_similar(
    "https://paulgraham.com/greatwork.html",
    num_results=10,
    contents={"text": True},
)
```

This is useful when the user wants to find related content, competitor pages, or similar articles.

## Answer

Use for direct question answering with citations.

```python
response = exa.answer("What is the latest funding round for Anthropic?")
```

## Error handling

If you hit auth errors:
1. Confirm `EXA_API_KEY` is set correctly
2. Verify the key at https://dashboard.exa.ai/api-keys
3. Check that the `exa-py` package is installed and up to date

## Pitfalls

- Do not cite sources not present in the Exa output.
- Prefer foreground execution for short tasks.
- For large result sets, save output to a temp file instead of stuffing everything into context.
- Do not silently choose Exa when Hermes native tools are already sufficient.
- Remember this is a vendor workflow that requires an API key and paid usage beyond the free tier.
