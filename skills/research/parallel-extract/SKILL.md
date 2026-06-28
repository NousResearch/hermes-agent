---
name: parallel-extract
description: "Use when bulk-fetching many public URLs into markdown for a RAG or agent KB via Parallel Extract API (JS pages, PDFs, >5 URLs, objective-focused excerpts or full_content)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [parallel, extract, rag, web, documentation, research]
    related_skills: [llm-wiki, arxiv]
---

# Parallel Extract API

## Overview

Parallel **Extract** (`POST /v1/extract`) turns public URLs into LLM-ready markdown (JavaScript sites and PDFs). Use it when you have a **known URL list** and need **dense excerpts** aligned to an objective, or **full page markdown** for ingestion. Pair with Parallel **Search** when you must discover URLs first.

Hermes `web_extract` is fine for a handful of pages; Parallel Extract scales to **20 URLs per request**, shared `session_id` across batches, and fetch policies (freshness, timeout, cache).

## When to Use

- User provides 6+ documentation or GitHub README URLs for a knowledge base
- User asks for Parallel `client.extract`, `PARALLEL_API_KEY`, or `parallel-web`
- Need `full_content` markdown plus optional objective-driven `excerpts`
- LangSmith / MCP wiring for Parallel Task (separate product — do not confuse with Extract)

Don't use for:

- Unknown URLs → Parallel Search or `web_search` first
- Single page, no JS/PDF pain → Hermes `web_extract` (cheaper, already in-agent)
- Pasting Python/`requests` code into the API `objective` field (invalid — see pitfalls)

## Prerequisites

```bash
pip install "parallel-web>=1.0.1"   # import: from parallel import Parallel
export PARALLEL_API_KEY="..."       # platform.parallel.ai — never commit keys
```

Docs index: https://docs.parallel.ai/llms.txt · Quickstart: https://docs.parallel.ai/extract/quickstart

**Note:** Installed `hermes-agent` may pin `parallel-web<1`. For v1 Extract features, use a venv or accept a resolver warning when upgrading `parallel-web`.

## Minimal call (SDK)

```python
from parallel import Parallel

client = Parallel()  # PARALLEL_API_KEY from env
extract = client.extract(
    urls=["https://docs.parallel.ai/extract/quickstart"],
    objective="Extract setup steps, response fields, and Python example.",
    search_queries=["Extract API", "full_content", "excerpts"],
)
for r in extract.results:
    print(r.title, r.url, len(r.full_content or ""), sum(len(e) for e in r.excerpts))
for e in extract.errors:
    print("FAIL", e.url, e.error_type)
```

## RAG-oriented settings

| Goal | `advanced_settings` |
|------|---------------------|
| Full pages for chunking | `"full_content": True` |
| Token-efficient snippets only | `"full_content": False` (default) |
| Fresh docs | `fetch_policy.max_age_seconds` (e.g. 172800 = 48h), `disable_cache_fallback: True` |
| Slow sites | `fetch_policy.timeout_seconds` up to 120 |

`objective` = natural-language extraction goal (what to keep from each page).  
`search_queries` = 2–5 keyword angles that match the corpus (not single generic words).

Optional: `client_model="claude-opus-4-7"` for provider-side tuning.

## Batching (>20 URLs)

Hard limit: **20 URLs per request**. Chunk URLs; pass returned `session_id` into the next batch.

```python
def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

session_id = None
for batch in chunk(urls, 20):
    resp = client.extract(urls=batch, objective=obj, search_queries=queries, session_id=session_id, ...)
    session_id = resp.session_id
```

**Done when:** every input URL appears in `results[].url` or `errors[].url` with no silent drops.

## Response shape (persist for RAG)

Per `results[]` item:

| Field | Use |
|-------|-----|
| `url`, `title`, `publish_date` | Metadata |
| `excerpts[]` | Objective-focused markdown chunks |
| `full_content` | Full page markdown if enabled |

Top-level: `extract_id`, `session_id`, `errors[]`, `usage[]` (e.g. `sku_extract_excerpts`).

Serialize JSON yourself; prefer `full_content` for embedding, `excerpts` for quick lookup.

## Hermes workflow

1. Collect URL list from user or prior Search; save as `urls.txt` (one URL per line, `#` comments OK).
2. From repo root or any cwd:

```bash
export PARALLEL_API_KEY="..."
python3 skills/research/parallel-extract/scripts/extract_docs.py urls.txt -o extracted_docs.json
```

3. Or import `extract_urls(urls, out_path=...)` from that script in a one-off notebook/script.
4. Verify: `len(results) + len(errors) == len(urls)`; spot-check largest `full_content` lengths.
5. Optional: split each `full_content` to `docs/<slug>.md` for llm-wiki / vector store.

## Common Pitfalls

1. **`objective` contains Python or JSON** — API expects prose goals only; embedded `requests.post` snippets break focus and may confuse the extractor.
2. **>20 URLs in one call** — last URLs fail or are rejected; always batch.
3. **`full_content: False` while user asked for complete docs** — enable `full_content: True` for RAG.
4. **Hardcoding API keys in source** — use `PARALLEL_API_KEY`; rotate if leaked in chat.
5. **Ignoring `errors[]`** — failed fetches are not in `results`; re-run failed URLs or lower freshness constraints.
6. **OAuth LangSmith “parallel-task-mcp” screen** — that shares your key for **Task** MCP, not required for a one-off Extract script.

## Verification Checklist

- [ ] `parallel-web` import works (`python3 -c "from parallel import Parallel"`)
- [ ] `objective` is plain-language, matches user's KB goals
- [ ] URL count handled in batches of ≤20 with `session_id` threaded
- [ ] Output JSON includes `publish_date`, `usage`, and all `errors`
- [ ] User told output path (e.g. `extracted_docs.json`) and result/error counts

## One-Shot Recipes

**Smoke test (1 URL):**

```bash
PARALLEL_API_KEY="$PARALLEL_API_KEY" python3 -c "
from parallel import Parallel
r = Parallel().extract(urls=['https://modelcontextprotocol.io/docs/getting-started/intro'],
  objective='MCP intro: transports, messages, capabilities.')
print(len(r.results), r.errors, r.usage)
"
```

**Full corpus:**

```bash
python3 skills/research/parallel-extract/scripts/extract_docs.py urls.txt -o extracted_docs.json
```