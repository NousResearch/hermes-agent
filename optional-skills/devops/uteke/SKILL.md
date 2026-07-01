---
name: uteke
description: "Offline semantic memory — local, zero-config, fast recall"
version: 0.6.1
author: codecoradev
license: Apache-2.0
platforms:
  - linux
  - macos
  - windows
metadata:
  hermes:
    tags:
      - Uteke
      - Memory
      - Semantic-Search
      - Offline
      - Knowledge-Graph
      - Local-First
    homepage: https://github.com/codecoradev/uteke
    related_skills:
      - hermes-agent
prerequisites:
  commands:
    - uteke
---

# Uteke — Offline Semantic Memory for Hermes

[Uteke](https://github.com/codecoradev/uteke) is a local-first semantic memory engine. Single Rust binary, zero config, ~30ms recall. No API keys, no cloud, no Python runtime.

## When to Use

- Setting up offline semantic memory for Hermes agents
- Multi-agent namespace isolation (each agent gets its own memory silo)
- Knowledge graph — nodes, edges, shortest path queries
- Collaborative rooms — shared context across agents/namespaces
- Importing knowledge bases from Markdown, JSONL, or text files
- Auto-consolidation of near-duplicate memories
- Document management — wiki/knowledge base with hierarchy

## Quick Reference

### Core Operations

| Command | Description |
|---------|-------------|
| `uteke remember "content" --tags tag1,tag2` | Store a memory |
| `uteke recall "query" --limit 5` | Semantic search (vector) |
| `uteke search "keywords"` | Keyword search (FTS5) |
| `uteke list --tag mytag --limit 20` | List memories by tag |
| `uteke get <id>` | Get single memory by UUID |
| `uteke forget <id>` | Delete a memory |
| `uteke stats` | Store statistics |
| `uteke doctor` | Health check (DB, index, model) |

### Lifecycle & Maintenance

| Command | Description |
|---------|-------------|
| `uteke verify` | DB + index consistency check |
| `uteke repair` | Rebuild index from SQLite |
| `uteke consolidate --threshold 0.85` | Merge near-duplicates |
| `uteke dream` | Full maintenance pipeline (lint → dedup → orphans → verify) |
| `uteke aging status` | View memory hot/warm/cold distribution |
| `uteke aging cleanup` | Remove cold memories past TTL |
| `uteke pin <id>` | Prevent memory from decaying |
| `uteke unpin <id>` | Allow memory to decay again |
| `uteke prune --ttl 30d` | Delete memories older than N days |

### Import / Export

| Command | Description |
|---------|-------------|
| `uteke export output.jsonl --namespace myns` | Export all memories to JSONL |
| `uteke import input.jsonl --namespace myns` | Import JSONL (re-embeds) |
| `uteke import doc.md --format markdown` | Import Markdown file |
| `uteke import --batch-dir ./notes/` | Batch import directory |
| `uteke import doc.md --extract` | LLM-extract atomic facts from document |

### Knowledge Graph

| Command | Description |
|---------|-------------|
| `uteke graph nodes` | List all graph nodes |
| `uteke graph edges` | List all edges |
| `uteke graph neighbors "entity"` | Find connected nodes (BFS) |
| `uteke graph path "A" "B"` | Shortest path between entities |
| `uteke graph query --relation "uses"` | Query edges by type |
| `uteke edges <id>` | List auto-wired edges for a memory |

### Rooms (Multi-Agent Collaboration)

| Command | Description |
|---------|-------------|
| `uteke room create --id sprint --title "Sprint"` | Create a room |
| `uteke room list` | List all rooms |
| `uteke room recall --room sprint "query"` | Recall from room |
| `uteke room summary --room sprint` | Topic clustering summary |
| `uteke room stats --room sprint` | Room analytics |
| `uteke room delete --room sprint` | Delete room (memories preserved) |

### Documents (Wiki / Knowledge Base)

| Command | Description |
|---------|-------------|
| `uteke doc create --slug getting-started --content "..."` | Create document |
| `uteke doc get --slug getting-started` | Get document |
| `uteke doc list` | List documents |
| `uteke doc search "query"` | Search documents |
| `uteke doc move --slug old --new-parent new-parent` | Move in hierarchy |

## Setup

### 1. Install Binary

```bash
# Install script (recommended)
curl -fsSL https://raw.githubusercontent.com/codecoradev/uteke/main/install.sh | sh

# Or download manually from GitHub Releases
# https://github.com/codecoradev/uteke/releases/latest

# Available assets:
#   uteke-aarch64-apple-darwin-vX.Y.Z.tar.gz   (macOS ARM)
#   uteke-aarch64-unknown-linux-gnu-vX.Y.Z.tar.gz  (Linux ARM)
#   uteke-x86_64-unknown-linux-gnu-vX.Y.Z.tar.gz    (Linux x86_64)
#   uteke-x86_64-pc-windows-msvc-vX.Y.Z.zip        (Windows)
```

### 2. Verify Installation

```bash
uteke --version          # should print version
uteke doctor             # health check: DB, index, model, consistency
```

### 3. Hermes Integration

Three integration modes. Use one or combine:

**Mode A — uteke-tool plugin (manual actions, requires daemon):**

```bash
uteke init --agent hermes                          # generates plugin
uteke-serve --port 8767                            # start warm server
# Plugin auto-loads in new Hermes sessions
```

**Mode C — pre_llm_call shell hook (automatic recall, no daemon):**

Register in `~/.hermes/profiles/<profile>/config.yaml`:

```yaml
hooks:
  pre_llm_call:
    - command: "uteke hook recall --limit 5"
      timeout: 20
hooks_auto_accept: true
```

**MCP Server (alternative for MCP-compatible agents):**

```bash
# Stdio transport
hermes mcp add uteke --command uteke-mcp

# HTTP transport (requires uteke-serve)
hermes mcp add uteke --url http://127.0.0.1:8767/mcp
```

### 4. Warm Server (Optional)

For sub-50ms recall, run the persistent server daemon:

```bash
uteke-serve --port 8767 --auth-token <secret>
```

| Metric | CLI (cold) | uteke-serve (warm) |
|--------|-----------|-------------------|
| Recall | ~2.9s (first call, model load) | **~30ms** |
| Remember | ~1.2s | **~100ms** |
| Search | ~3.0s | **~26ms** |
| RAM | 0 (on-demand) | ~208MB (model in RAM) |

## CLI Reference

### `uteke remember`

```
uteke remember "content" [OPTIONS]
  --tags tag1,tag2        Categorization tags
  --namespace myns        Namespace isolation (default: "default")
  --type fact|procedure|preference|decision|context|note|insight|reference|event
  --entity "ProjectName"  Structured metadata entity
  --detect-contradiction  Auto-deprecate conflicting memories
  --category "infra"      Classification category
  --meta key:value,...    Arbitrary metadata pairs
  --room sprint           Link to collaborative room
  --source "url|file"     Provenance tracking
```

### `uteke recall`

```
uteke recall "query" [OPTIONS]
  --limit N               Max results (default: 5)
  --namespace myns
  --tags tag1,tag2        Filter by tags
  --entity "name"         Filter by entity
  --min 0.5              Minimum similarity threshold (0.0-1.0)
  --strategy vector|fts5|hybrid|graph   Search strategy
  --salience              Boost by importance (weight: 0.15)
  --recency               Boost by freshness (weight: 0.15)
  --related               Follow relationship edges
  --depth 2               Edge traversal depth (with --related)
  --context               Format output for AI prompt injection
  --at 2026-06-01T12:00:00Z  Point-in-time recall
  --where key=value       Filter by JSON metadata field
```

### `uteke import`

```
uteke import [INPUT] [OPTIONS]
  --format auto|jsonl|markdown|text   Input format (auto-detect)
  --namespace myns
  --tags tag1,tag2        Apply tags to all imported memories
  --extract               LLM-extract atomic facts from document
  --batch-dir ./notes/    Import all files in directory
  --recursive             Include subdirectories
  --dry-run               Preview without importing
  --max-size 1048576      Max file size in bytes (default: 1MB)
```

### `uteke graph`

```
uteke graph <COMMAND>
  nodes          List all graph nodes
  edges          List all edges
  neighbors "X"  Find connected nodes (BFS)
  path "A" "B"   Shortest path between two nodes
  query --relation "uses"  Query edges by relation type
  stats          Graph statistics
```

## Usage Patterns

### Basic Remember + Recall

```bash
# Store
uteke remember "User prefers Rust over Go for CLI tools" --tags preference,lang --type preference

# Recall
uteke recall "language preferences for CLI tools" --limit 3 --json

# JSON output format:
# [{"memory": {"id": "uuid", "content": "...", "tags": [...], "namespace": "..."}, "score": 0.72}]
# Results nested under "memory" key — parse accordingly.
```

### Namespace Isolation for Multi-Agent

```bash
# Each agent uses its own namespace — strict silo, no cross-ns search
uteke remember "CTO approved migration plan" --namespace cto --tags decision
uteke recall "migration plan" --namespace cto

# List all namespaces
uteke namespace list
uteke namespace stats --namespace cto
```

### Import Knowledge Base

```bash
# From JSONL (each line: {"content":"...", "tags":[], "metadata":{}})
uteke import knowledge.jsonl --namespace project --tags imported

# From Markdown directory
uteke import --batch-dir ./docs/ --recursive --namespace docs

# With LLM extraction (distills documents into atomic facts)
uteke import architecture.md --extract --namespace knowledge
```

### Knowledge Graph

```bash
uteke remember "PostgreSQL is used for user data" --entity PostgreSQL
uteke remember "Redis caches session tokens" --entity Redis
uteke remember "PostgreSQL connects to Redis for cache invalidation" --entity PostgreSQL

uteke graph path "PostgreSQL" "Redis"    # shortest path
uteke graph neighbors "PostgreSQL"        # all connected entities
uteke graph query --relation "uses"       # all "uses" relationships
```

### Room-Based Collaboration

```bash
uteke room create --id sprint-42 --title "Sprint 42 Planning"
uteke remember "API deploy scheduled Friday" --namespace agent1 --room sprint-42 --author agent1
uteke room recall --room sprint-42 "deploy timeline"
uteke room summary --room sprint-42
```

## Configuration

### `uteke.toml` (optional, `~/.uteke/uteke.toml`)

```toml
[recall]
default_strategy = "vector"    # vector | fts5 | hybrid | graph
min_score = 0.45
min_score_strict = 0.6
salience_weight = 0.15
recency_weight = 0.15

[aging]
hot_days = 7
warm_days = 30
cold_days = 90

[server]
port = 8767
host = "127.0.0.1"
# cors_origins = ["http://localhost:3000"]

[extraction]
model = "your-chat-model"
base_url = "https://your-endpoint/v1"
api_key_env = "OPENAI_API_KEY"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UTEKE_HOME` | `~/.uteke` | Data directory (DB + model + index) |
| `UTEKE_AUTH_TOKEN` | — | Bearer token for uteke-serve API auth |
| `UTEKE_READ_ONLY_TOKEN` | — | Read-only token (GET endpoints only) |
| `UTEKE_NAMESPACE` | `default` | Default namespace for CLI commands |
| `UTEKE_RECALL_LIMIT` | `5` | Default recall limit |
| `UTEKE_RECALL_MIN_SCORE` | `0.45` | Default minimum similarity score |
| `UTEKE_EXTRACT` | `false` | Enable LLM fact extraction on import |
| `UTEKE_EXTRACT_MODEL` | — | Chat model for extraction |
| `UTEKE_EXTRACT_API_KEY` | — | API key for extraction |
| `UTEKE_EXTRACT_BASE_URL` | — | OpenAI-compatible base URL |

## Pitfalls

- **Namespace = strict silo.** There is NO cross-namespace search. `recall --namespace cto` returns ONLY memories stored in `cto`. Use `--namespace` explicitly — omitting it defaults to `default`, not `all`.
- **Cold start ~3s.** First CLI call loads the ONNX model (EmbeddingGemma Q4, 768d). Subsequent calls are fast. Use `uteke-serve` for persistent warm model in RAM.
- **EmbeddingGemma max ~8K chars (~2048 tokens).** Content longer than this is silently truncated for embedding. The full text is still stored in SQLite and returned in results — only the vector is truncated.
- **`recall --json` output is nested.** Results are under `"memory"` key: `[{"memory": {"content": "...", ...}, "score": 0.72}]`. Parse accordingly — do not treat as flat `{content, score}`.
- **No entity field in DB schema.** Uteke stores entity as metadata (`--entity` flag). Use `--entity` consistently for graph operations to work. Without it, `graph nodes` and `neighbors` return nothing.
- **`uteke-serve` is a separate binary.** Install from the same release tarball. Both `uteke` and `uteke-serve` are bundled together.
- **Import always re-embeds.** Even if you export and re-import the same JSONL, vectors are regenerated. This is by design — portable format carries content only, not vectors.
