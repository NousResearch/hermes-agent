---
name: wiki-search
description: Search markdown wikis semantically with local Ollama.
version: 1.1.0
author: PINKIIILQWQ (@PINKIIILQWQ) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [wiki, search, semantic-search, markdown, ollama]
    related_skills: [llm-wiki, qmd]
---

# Wiki Search

Search Markdown wikis by meaning with local Ollama embeddings. The skill keeps
each index isolated by canonical wiki root, embedding model, and cache schema,
and falls back to keyword search when Ollama is unavailable.

## Commands

```bash
python3 scripts/wiki-search --wiki ~/wiki index
python3 scripts/wiki-search --wiki ~/wiki reindex
python3 scripts/wiki-search --wiki ~/wiki --hybrid "photography composition"
python3 scripts/wiki-search --wiki ~/wiki --status
python3 scripts/wiki-search --wiki ~/wiki --clean
```

Use `--model <name>` to select an embedding model and `--json` for stable,
machine-readable output. JSON search responses have `schema_version`, `command`,
`wiki`, `model`, `mode`, `fallback`, and `results` fields.

```python
result = json.loads(subprocess.run(
    ["python3", "scripts/wiki-search", "--json", "--wiki", wiki, "question"],
    check=True,
    capture_output=True,
    text=True,
).stdout)
for item in result["results"]:
    print(item["file"], item["heading"])
```

## Prerequisites

Ollama is optional. For semantic results, install it and pull an embedding model:

```bash
ollama pull all-minilm
```

Without Ollama, `wiki-search` returns keyword matches instead of failing.

## Index Behavior

Indexes live beneath `~/.cache/wiki-search/v1/` in a namespace derived from the
resolved wiki root and exact model name. Changing a model creates a separate
index. Deleted Markdown files are pruned at the next index run. A corrupted or
incompatible index is rebuilt rather than mixing vector dimensions.
