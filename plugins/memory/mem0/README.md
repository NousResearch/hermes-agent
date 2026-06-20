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
| `auto_sync_turns` | `true` | Write every turn to Mem0. Set `false` to skip turn-by-turn writes entirely — only explicit `mem0_conclude` calls will store memories. |
| `auto_extract_facts` | `true` | When `auto_sync_turns` is `true`, controls whether Mem0's server-side LLM extracts facts from each turn (`true`, legacy) or stores the messages verbatim (`false`). No effect when `auto_sync_turns` is `false`. |

### Reducing noise

For long-running gateway or CLI sessions, the default behaviour can grow
your Mem0 store with low-value entries (extracted "User …" facts or
verbatim chat lines). Two recommended overrides:

- **Quiet store, server-side extraction off:** `auto_sync_turns: true`,
  `auto_extract_facts: false`. Every turn is written verbatim — searchable
  but not LLM-summarized.
- **Manual-only:** `auto_sync_turns: false`. Mem0 only stores what you
  explicitly call `mem0_conclude` on.

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |
