# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

Supports both [Mem0 Cloud](https://app.mem0.ai) and self-hosted instances.

## Requirements

- `pip install mem0ai`
- Mem0 Cloud API key **or** a self-hosted Mem0 server

## Setup

### Cloud

```bash
hermes memory setup    # select "mem0"
```

Or manually:

```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

### Self-Hosted

```bash
hermes config set memory.provider mem0
echo "MEM0_HOST=http://your-mem0-server:24220" >> ~/.hermes/.env
echo "MEM0_API_KEY=your-api-key" >> ~/.hermes/.env   # if auth is enabled
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | — | API key (required for cloud; optional for self-hosted without auth) |
| `host` | `https://api.mem0.ai` | Self-hosted Mem0 URL. When set, overrides the cloud endpoint. |
| `user_id` | `hermes-user` | User identifier |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
