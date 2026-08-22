# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search and hybrid multi-signal retrieval via the Mem0 Platform v3 API.

## Requirements

- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

## Setup

```bash
hermes memory setup    # select "mem0"
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

Behavioral settings live in `$HERMES_HOME/mem0.json` (set them via `hermes memory setup`). Only the secret `MEM0_API_KEY` belongs in `~/.hermes/.env`.

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `platform` | `platform` (Mem0 Cloud) or `oss` (self-managed, in-process) |
| `host` | — | Self-hosted Mem0 server URL (the Docker dashboard). When set, connects over HTTP with `X-API-Key`. Don't combine with `mode: oss` |
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `false` | Rerank search results for relevance (platform mode only) |
| `shared_pool.enabled` | `false` | Enable the agent-scoped shared "company knowledge" pool (`mem0_search_shared` / `mem0_add_shared`) |
| `shared_pool.authorized_submitters` | `[]` | Operator identifiers allowed to WRITE to the shared pool. **Empty (the default) means ANY operator may contribute — only set it and list specific operator IDs if you actually want to restrict writes.** Matching is case-insensitive. IDs are whatever each gateway resolves for the operator (Telegram/Discord id, CLI username, email-style id, …), so use the exact id your gateway reports |

The plugin has three connection modes:

- **Platform** — Mem0's hosted cloud (`api.mem0.ai`). Set `MEM0_API_KEY`. (default)
- **Self-hosted dashboard** — a Mem0 server you run yourself via Docker. Set `host`. See below.
- **OSS** — run Mem0 in-process with your own LLM + vector store. Set `mode: oss`. See below.

## Self-Hosted Dashboard (Server) Mode

Connect the plugin to a standalone Mem0 server you run yourself — the Docker-shipped Mem0 dashboard/server with its own REST API. Unlike OSS mode (which runs `mem0ai` in-process with your own vector store), here the plugin just talks HTTP to your server.

1. Run the Mem0 server (FastAPI + pgvector) from its Docker image and note its URL and `ADMIN_API_KEY`.
2. Point the plugin at it — via the setup wizard:
   ```bash
   hermes memory setup    # select "mem0" → "Self-hosted server"
   # Or non-interactive:
   hermes memory setup mem0 --mode selfhosted --host http://localhost:8888 --api-key your-admin-api-key
   ```
   or via env vars:
   ```bash
   echo "MEM0_HOST=http://localhost:8888" >> ~/.hermes/.env
   echo "MEM0_API_KEY=your-admin-api-key" >> ~/.hermes/.env
   ```
   or in `$HERMES_HOME/mem0.json`:
   ```json
   {
     "host": "http://localhost:8888",
     "api_key": "your-admin-api-key"
   }
   ```
3. Start a fresh Hermes session and call `mem0_search` — it connects to your server.

The plugin authenticates with `X-API-Key` and uses the server's `/search` and `/memories` routes. `api_key` is optional — omit it only for servers running with `AUTH_DISABLED`.

> Setting `host` routes to the self-hosted server automatically. Don't set `mode: oss` — OSS takes precedence and ignores `host`.

## OSS (Self-Hosted) Mode

Run Mem0 locally with your own LLM, embedder, and vector store. This is the in-process SDK mode. To instead connect to a Mem0 server you run via Docker, see [Self-Hosted Dashboard (Server) Mode](#self-hosted-dashboard-server-mode) above.

### Interactive Setup

```bash
hermes memory setup
# Select "mem0" → "Open Source (self-hosted)"
# Follow prompts for LLM, embedder, and vector store
```

### Agent-Driven Setup (Flags)

```bash
hermes memory setup mem0 --mode oss \
  --oss-llm openai --oss-llm-key sk-... \
  --oss-vector qdrant
```

### Supported Providers

| Component | Providers |
|-----------|-----------|
| LLM | openai, ollama |
| Embedder | openai, ollama |
| Vector Store | qdrant (local/server), pgvector |

### Flags Reference

| Flag | Description |
|------|-------------|
| `--mode` | `platform` or `oss` |
| `--oss-llm` | LLM provider (default: openai) |
| `--oss-llm-key` | LLM API key |
| `--oss-embedder` | Embedder provider (default: openai) |
| `--oss-vector` | Vector store (default: qdrant) |
| `--oss-vector-path` | Qdrant local path |
| `--user-id` | User identifier |

## Switching Modes

### Platform to OSS

```bash
hermes memory setup mem0 --mode oss --oss-llm-key sk-...
```

Or edit `$HERMES_HOME/mem0.json` directly:
```json
{
  "mode": "oss",
  "oss": {
    "llm": {"provider": "openai", "config": {"model": "gpt-5-mini"}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "vector_store": {"provider": "qdrant", "config": {"path": "~/.hermes/mem0_qdrant"}}
  }
}
```

### OSS to Platform

```bash
hermes memory setup mem0 --mode platform --api-key sk-...
```

### Dry Run (preview without writing)

```bash
hermes memory setup mem0 --mode oss --oss-llm-key sk-... --dry-run
```

## Tools

| Tool | Description |
|------|-------------|
| `mem0_search` | Semantic search by meaning (per-user scope) |
| `mem0_add` | Store a fact verbatim (no LLM extraction) (per-user scope) |
| `mem0_update` | Update a memory's text by ID |
| `mem0_delete` | Delete a memory by ID |
| `mem0_search_shared` | Search the agent-scoped shared "company knowledge" pool (enabled via `shared_pool.enabled`) |
| `mem0_add_shared` | Store a company-wide fact in the shared pool (refused for operators not in `shared_pool.authorized_submitters`) |

## Shared Company Knowledge Pool

By default, every mem0 memory is scoped to a single operator (`user_id`): each operator's conversations, preferences, and notes stay isolated under their own identity. This is correct for a user's private memory.

For a **team agent** — one Hermes agent that talks to several operators (e.g. an "employee" agent serving multiple people) — you often want a shared pool that every operator can read, alongside each operator's private memory. Enable it in `$HERMES_HOME/mem0.json`:

```json
{
  "shared_pool": {
    "enabled": true,
    "authorized_submitters": ["telegram:123456789", "cli:kyle", "operator-b@example.com"]
  }
}
```

> Operator identifiers (`user_id`) are whatever each gateway resolves for the operator — a Telegram/Discord numeric id, a CLI username, an email-style id, etc. Use the exact id form your gateway reports; matching is case-insensitive but otherwise exact.

When enabled, two extra tools are registered:

- **`mem0_search_shared`** – reads the agent-scoped shared pool. It returns only records that are **positively tagged as shared** — `mem0_add_shared` writes each company fact with `metadata.scope="shared"` and **no** `user_id` — and then further excludes anything that carries a per-user scope. This two-belt check (positive shared marker, and no user identity) keeps the company view an intersection, so per-user private memories (`mem0_search` / `mem0_add`) are individually scoped and stay out of the company view, even if the backend changes how it reports user identity.
- **`mem0_add_shared`** – writes a company-wide fact into the shared pool (scoped to `agent_id`, with no `user_id`, tagged `metadata.scope="shared"`). Whether a given operator may call it is enforced in code:

  > **Security note:** an empty `authorized_submitters` (the default) means **any** operator may write to the shared pool. If you need to restrict who can contribute company facts, list the specific operator IDs — do not rely on the default.

  - If `shared_pool.authorized_submitters` is **empty** (default), any operator may contribute.
  - If it is **non-empty**, only operators whose `user_id` appears in the list may write (matched case-insensitively and gateway-agnostically — the id is whatever each gateway resolved for the operator, e.g. a Telegram/Discord numeric id, a CLI username, or an email-style id, so an entry that differs only in casing is not silently refused). Everyone else gets a refusal from `mem0_add_shared` and can still read via `mem0_search_shared`.

This enables the common pattern: **everyone reads company knowledge, only specified team members write it.** Private per-user memory (`mem0_search` / `mem0_add`) is unaffected.

## Troubleshooting

### "Mem0 temporarily unavailable"

Circuit breaker tripped after 5 consecutive failures. Resets after 2 minutes.

- **Platform mode**: Check API key and internet connectivity.
- **OSS mode**: Check that your vector store (qdrant/pgvector) is running.

### OSS: Qdrant connection refused

```bash
# If using local Qdrant, check the storage path is writable:
ls -la ~/.hermes/mem0_qdrant

# If using Qdrant server, check it's reachable:
curl http://localhost:6333/healthz
```

### OSS: PGVector connection refused

```bash
# Verify PostgreSQL is running and accepting connections:
pg_isready -h localhost -p 5432
```

### OSS: Ollama not reachable

```bash
# Check Ollama is running:
curl http://localhost:11434/api/tags
```

### Memories not appearing

- `mem0_add` stores verbatim (no extraction). Use `sync_turn` for LLM extraction.
- Search uses semantic matching — try broader queries.
- Check `user_id` matches between sessions (`$HERMES_HOME/mem0.json`).
