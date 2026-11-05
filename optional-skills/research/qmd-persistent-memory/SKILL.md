---
name: qmd-persistent-memory
description: Give persistent agent memory local semantic search via QMD.
version: 0.1.0
author: Joerg Peetz (JPeetz), Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Memory, RAG, Semantic-Search, QMD, MCP, Local-AI]
    related_skills: [qmd, obsidian, hermes-agent]
---

# QMD Persistent Memory

Give an agent's persistent memory stores (an Obsidian vault, a notes wiki, a
MeMex-style knowledge base) **local, private, semantic search** using QMD —
embeddings for recall, with a fast vector path and an optional hybrid+rerank
path. Everything runs on-device; no cloud, no API cost, memory never leaves
the machine.

This is the wrapper pattern: keep the memory files where they are, index them
into a named QMD index (multi-store isolation), and expose semantic search
through a single shared MCP server so every connected agent (and your main
agent) gets the same tools. Helpful for questions like "when did we decide
X?" where keyword grep misses the doc because it uses different words.

Does **not** replace the `qmd` skill (CLI + native `qmd mcp` daemon). Prefer
this skill when you want to fold semantic search into a server you already
expose, or you must keep several memory roots in one searchable index.

## When to Use

- User asks "when did we decide / why did we choose / what did we learn about X"
- You want conceptual search over memory, not exact-keyword grep
- You already run a memory MCP server (`hermes-hq`, custom) and want to add a
  semantic-search tool to it
- You need to search across two or more memory roots (vault + wiki) at once
- A fleet of agents connects to one MCP server and should all get memory search
- Keywords: "search my notes", "memory search", "find in my vault", "RAG"
- Trigger proactively when ripgrep-style search returns nothing but you suspect
  a related note exists under different wording

## Prerequisites

1. **Node.js ≥ 22** — `node --version`.
2. **Python 3.10+** and the `mcp` package for the server wrapper:
   `pip install "mcp[cli]"`, or reuse an existing FastMCP server.
3. **QMD** installed and healthy:
   ```bash
   npm install -g --allow-scripts=node-llama-cpp @tobilu/qmd
   qmd doctor        # verifies sqlite-vec, model cache, CPU/GPU mode
   qmd pull          # downloads ~2GB of local GGUF models (one-time)
   ```
   The `--allow-scripts` flag is required so the native `node-llama-cpp`
   binary (the GGUF/embedding engine) actually compiles.
4. **OS note:** on macOS use Homebrew SQLite so QMD can load its extension:
   `brew install sqlite`.
5. Create a **named QMD index** per memory deployment so stores stay isolated:
   ```bash
   qmd --index mem vault/memory-store collection add ~/notes
   qmd --index mem vault/memory-store collection add ~/knowledge-base
   qmd --index mem vault/memory-store embed
   ```

## How to Run

Quick verification through the `terminal` tool (no MCP needed), pointing at
your named index:

```python
terminal(command="qmd --index mem vault/memory-store vsearch 'why did we defer it' --json -n 5", timeout=60)
```

## Quick Reference

| Surface | Command / Tool | Speed (CPU, first run) |
|---------|----------------|------------------------|
| Fast semantic | `qmd vsearch "q"` | ~5s (embedding model only) |
| Hybrid + rerank | `qmd query "q"` | ~5 min cold (loads 1.2GB expansion + 610MB reranker) |
| Keyword | `qmd search "q"` | ~0.2s (BM25, no models) |
| Index health | `qmd status` | instant |
| MCP wrapper | `scripts/qmd_memory_search.py` | proxy to the above |

## Procedure

### 1. Pick the index name and add all memory roots

A single named index can hold many collections. This is what gives you
multi-store search (the "all" scope) instead of one grep per root:

```bash
qmd --index <name> collection add ~/vault
qmd --index <name> collection add ~/wiki
qmd --index <name> status          # confirm doc count before embedding
```

### 2. Embed (one-time, CPU-bound on a Mac without a GPU)

`qmd embed` chunk-splits each file and generates vectors. On a 12-core Intel
Mac a few hundred docs take several minutes. It is resumable — re-run after
the process is interrupted to continue. Check progress with `qmd status`.

```bash
qmd --index <name> embed     # re-run after adding docs (incremental)
```

### 3. Expose through a shared MCP server

Copy `scripts/qmd_memory_search.py` into your existing FastMCP server (it is
self-contained), or run it standalone as its own MCP server. It registers:

- `semantic_search_memory(query, n, base, mode)` — ranked results
- `qmd_status()` — index health

The script shells out to QMD with the named index, strips the `qmd://` scheme
from result paths, and handles the banner-before-JSON output. Point `QMD_INDEX`
at the name you chose, and `QMD_BASES` at your collection→`base` labels so the
`base="all" | "vault" | "wiki"` scope works.

### 4. Restart the server and verify

```bash
# after wiring the tool in:
# restart your MCP server process, then from a client:
#   tools/list   → semantic_search_memory + qmd_status present
#   tools/call semantic_search_memory {"query":"why did we defer it","mode":"fast"}
```

## Select a search mode (important for latency)

QMD runs fully on CPU here (no GPU). The rerank/expansion models are large, so
the hybrid path is slow to start. Default to **`mode="fast"` (vsearch)** which
loads only the 300MB embedding model (~5s). Reserve `mode="hybrid"` (query) for
hard lookups where you accept a multi-minute cold start.

## Pitfalls

- **`npm install -g @tobilu/qmd` without `--allow-scripts`** silently skips the
  native `node-llama-cpp` build → `qmd doctor` shows a broken/Metal warning and
  embeddings fail. Reinstall with `--allow-scripts=node-llama-cpp,...`.
- **Search returns nothing but you *know* the note exists** — you used `query`
  (hybrid) and it timed out cold, or you hit the banner-parse path. Use
  `vsearch` (fast) and confirm with `qmd status` that vectors are non-zero.
- **`qmd embed` left "N pending"** — the process was interrupted. It is
  resumable; re-run until pending = 0. Do not assume a crashed run failed.
- **QMD prints a banner** ("Expanding query… Searching…") to stdout *before*
  the JSON. Parsers must cut to the first `[` or json.load fails.
- **Result paths come back as `qmd://...?...index=...`** — strip the scheme and
  query-string so the path is actionable.
- **macOS default SQLite can't load the vec extension** — `brew install sqlite`
  and ensure it's on PATH ahead of the system copy, or `qmd doctor` reports no
  sqlite-vec.

## Verification

- `qmd --index <name> status` → **Vectors: N, Pending: 0**
- A `vsearch` returns the top-N ranked results with actionable paths
- `tools/list` on the MCP server shows `semantic_search_memory` + `qmd_status`
- A `tools/call semantic_search_memory` returns correctly-ranked, `qmd://`-free
  snippets from the right memory root