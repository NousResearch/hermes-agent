# Maindex Memory Provider

Structured memory graph with semantic + relational recall via the Maindex Expert REST API. Multi-tier retrieval: exact match, relaxed OR fallback with synonym expansion, fuzzy trigram, and semantic/hybrid search.

## Requirements

- `pip install httpx`
- API key from [maindex.io/dashboard](https://maindex.io/dashboard)

## Setup

```bash
hermes memory setup    # select "maindex"
```

Or manually:

```bash
hermes config set memory.provider maindex
echo "MAINDEX_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MAINDEX_API_KEY` | API key (one of api_key or token required) |
| `MAINDEX_TOKEN` | OAuth bearer token (alternative to API key) |
| `MAINDEX_COLLECTION` | Default collection slug (optional) |

### Config File

Config file: `$HERMES_HOME/maindex.json`

| Key | Default | Description |
|-----|---------|-------------|
| `collection` | `""` | Default collection slug for scoping memories |

## Tools

| Tool | Description |
|------|-------------|
| `maindex_search` | Full-text, fuzzy, semantic, and hybrid search. Filter by tags, kind, collection. |
| `maindex_keep` | Store a new memory with headline, body, tags, kind, and collection. |
| `maindex_recall` | Retrieve a specific memory by ID (UUID or short ID like `mem-1a`). |
| `maindex_update` | Revise an existing memory with full revision history. Modes: body_append, body_replace, headline_replace, headline_and_body_replace, revision_only. Also supports changing kind, canon_status, confidence, verification_status. |
| `maindex_forget` | Soft-delete a memory (restorable). |

## Lifecycle Hooks

| Hook | Behavior |
|------|----------|
| `system_prompt_block` | Injects provider info into system prompt |
| `prefetch` / `queue_prefetch` | Semantic search before each turn (top 5 results) |
| `sync_turn` | Stores each conversation turn as a note (non-blocking, background thread) |
| `on_memory_write` | Mirrors MEMORY.md/USER.md writes as facts |
| `on_pre_compress` | Saves context snapshot before compression discards messages |
| `shutdown` | Clean thread + client teardown |

## Circuit Breaker

5 consecutive failures triggers a 120-second cooldown before retrying API calls.

## Links

- [Website](https://maindex.io)
- [Expert API Docs](https://expert.maindex.io/docs)
- [Dashboard](https://maindex.io/dashboard)
- [Help](https://maindex.io/help)
