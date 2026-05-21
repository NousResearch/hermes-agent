# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

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

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |
| `mode` | `platform` | Use `local` for local Mem0 with local embedder/LLM |
| `local_path` | `$HERMES_HOME/local-mem0-shadow` | Local history DB and embedded Qdrant path |
| `local_qdrant_url` | empty | Optional Qdrant server URL, e.g. `http://localhost:6333`; when set, disables embedded Qdrant path storage |
| `local_qdrant_host` / `local_qdrant_port` | empty | Alternative host/port for Qdrant server |
| `local_qdrant_api_key` | empty | Optional Qdrant API key |

## Local Qdrant Server

For gateway + tool concurrency, prefer Qdrant server over embedded local path storage:

```json
{
  "mode": "local",
  "local_qdrant_url": "http://localhost:6333"
}
```

Embedded Qdrant uses a file lock and only one process can open it at a time.

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
