---
name: tavily
description: "Use when you need web search with built-in citations, direct Q&A answers, or deep extraction from multiple URLs. Requires tavily-python and TAVILY_API_KEY. Free tier: 1000 searches/mo."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [web, search, Q&A, citations, RAG, research]
    related_skills: [firecrawl, exa-search]
---

# Tavily

Website: https://tavily.com | Docs: https://docs.tavily.com

## When to Use

* General web search that needs source citations automatically
* Asking a question and wanting a direct answer with attribution
* Research across multiple URLs simultaneously
* RAG pipelines where content needs to be pre-structured for LLM consumption

**Trigger examples:**
- "Search for recent AI model releases"
- "Answer: what is the latest news on X?"
- "Extract content from these URLs"
- "What are the top results for ..."

Tavily is specifically designed for AI workflows. Output is clean, pre-structured, and citation-aware. Compared to raw curl or even Firecrawl, Tavily adds relevance scoring and semantic understanding.

## Setup

```bash
pip install tavily-python
```

Add to `~/.hermes/.env`:
```
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

Free tier: 1000 searches/month. Pro: unlimited.

## Core Usage

### Basic Search

```python
from tavily import TavilyClient
tc = TavilyClient()

results = tc.search('your query', max_results=10)
for r in results['results']:
    print(r['url'], r['title'])
    print(r['content'][:200])
    print(f"  Score: {r['score']}")
    print()
```

### Q&A (direct answers with source citations)

```python
answer = tc.qna_search('your question', max_results=5)
print(answer['answer'])
for src in answer.get('sources', []):
    print(f"  Source: {src['url']}")
```

### Extract (deep content from multiple URLs)

```python
content = tc.extract(['https://url1.com', 'https://url2.com'])
for item in content['results']:
    print(item['url'])
    print(item['raw_content'][:300])
    print()
```

## Method Reference

| Method | Best For | Output |
|--------|----------|--------|
| `search(query, max_results)` | Broad discovery | List of ranked results with url, title, content, score |
| `qna_search(question, max_results)` | Direct answers | Answer string + source list |
| `extract(urls)` | Content extraction | Full raw_content per URL |

## Common Pitfalls

1. **No API key.** Tavily requires a key even on free tier. Sign up at tavily.com. Free tier is generous at 1000 searches/mo.

2. **`qna_search` returning empty answers.** This happens when Tavily cannot find a confident answer. Fall back to `search()` and summarize from results.

3. **`max_results` too high.** Each result adds latency and token cost. For quick lookups use `max_results=3`. For research use `max_results=10`.

4. **Rate limits on free tier.** At 1000/mo, be strategic. Cache results if the same query might run again.

## Verification Checklist

- [ ] `pip install tavily-python` completes without error
- [ ] `TAVILY_API_KEY` set in `~/.hermes/.env`
- [ ] `python -c "from tavily import TavilyClient; print('OK')"` runs cleanly
- [ ] `tc.search('test', max_results=1)` returns results with url and content