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

Mem0 reads its config from three sources, in order of increasing precedence:

1. **Environment variables** — `MEM0_API_KEY`, `MEM0_USER_ID`, `MEM0_AGENT_ID`.
2. **`$HERMES_HOME/mem0.json`** — a JSON sidecar for the keys below.
3. **`config.yaml > memory.mem0.*`** — the same key/value pairs in
   the main Hermes config. Useful for per-profile setups so the
   profile's `config.yaml` is the single source of truth.

```yaml
# config.yaml
memory:
  provider: mem0
  mem0:
    user_id: jereme
    agent_id: stack
    rerank: true
```

| Key | Default | Description |
|-----|---------|-------------|
| `user_id` | `hermes-user` | User identifier on Mem0. Gateway-supplied `user_id` (e.g. a Telegram chat id) wins over config. |
| `agent_id` | `hermes` | Agent identifier. If left unset everywhere, defaults to the active Hermes profile name (so multi-profile users get per-profile write attribution automatically). |
| `rerank` | `true` | Enable reranking for recall |

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
